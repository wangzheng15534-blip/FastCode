"""Typed file inventory records for indexing plans."""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from git import InvalidGitRepositoryError, NoSuchPathError, Repo

from fastcode.utils.filesystem import compute_file_hash, normalize_path
from fastcode.utils.paths import get_language_from_extension
from .ignore import should_ignore_path

_PACKAGE_SCOPE_MARKERS = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "tsconfig.json",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "composer.json",
    "Project.toml",
)

_TOOL_ELIGIBLE_LANGUAGES = frozenset(
    {
        "c",
        "cpp",
        "csharp",
        "fortran",
        "go",
        "java",
        "javascript",
        "julia",
        "kotlin",
        "python",
        "ruby",
        "rust",
        "scala",
        "typescript",
        "zig",
    }
)


@dataclass(frozen=True)
class FileFingerprint:
    """Canonical per-file planner record for one index run."""

    path: str
    relative_path: str
    size: int
    mtime: float
    extension: str
    language: str
    package_root: str
    supported_tool_eligible: bool
    content_hash: str | None = None
    git_blob_oid: str | None = None
    fingerprint_source: str | None = None

    @property
    def identity(self) -> str | None:
        """Return the preferred file content identity for planning."""
        return self.git_blob_oid or self.content_hash

    def to_mapping(self) -> dict[str, Any]:
        """Return the legacy dict adapter consumed by existing call sites."""
        return {
            "path": self.path,
            "relative_path": self.relative_path,
            "size": self.size,
            "mtime": self.mtime,
            "extension": self.extension,
            "language": self.language,
            "package_root": self.package_root,
            "supported_tool_eligible": self.supported_tool_eligible,
            "content_hash": self.content_hash,
            "blob_oid": self.git_blob_oid,
            "git_blob_oid": self.git_blob_oid,
            "content_identity": self.identity,
            "fingerprint_source": self.fingerprint_source,
        }


@dataclass(frozen=True)
class FileInventory:
    """Repository file inventory produced once for an index run."""

    repo_root: str
    files: tuple[FileFingerprint, ...]

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_size_bytes(self) -> int:
        return sum(file.size for file in self.files)

    def to_file_info_list(self) -> list[dict[str, Any]]:
        return [file.to_mapping() for file in self.files]

    def metrics(self) -> dict[str, Any]:
        git_blob_count = sum(1 for file in self.files if file.git_blob_oid)
        content_hash_count = sum(
            1 for file in self.files if file.content_hash and not file.git_blob_oid
        )
        return {
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "git_blob_oid_count": git_blob_count,
            "content_hash_count": content_hash_count,
            "fingerprinted_file_count": git_blob_count + content_hash_count,
            "supported_tool_eligible_count": sum(
                1 for file in self.files if file.supported_tool_eligible
            ),
        }


def _git_worktree_root(repo_path: str) -> tuple[Repo, str] | None:
    try:
        repo = Repo(repo_path, search_parent_directories=True)
    except (InvalidGitRepositoryError, NoSuchPathError):
        return None
    worktree = repo.working_tree_dir
    if not worktree:
        return None
    return repo, os.path.abspath(worktree)


def _git_relative_path(abs_path: str, worktree_root: str) -> str | None:
    try:
        rel_path = os.path.relpath(abs_path, worktree_root)
    except ValueError:
        return None
    if rel_path.startswith(".."):
        return None
    return normalize_path(rel_path)


def _dirty_git_paths(repo: Repo) -> set[str]:
    dirty: set[str] = set()
    try:
        for diff in repo.index.diff(None):
            if diff.a_path:
                dirty.add(normalize_path(diff.a_path))
            if diff.b_path:
                dirty.add(normalize_path(diff.b_path))
    except Exception:
        return set()
    with suppress(Exception):
        dirty.update(normalize_path(path) for path in repo.untracked_files)
    return dirty


def _index_blob_oids(repo: Repo) -> dict[str, str]:
    blob_oids: dict[str, str] = {}
    try:
        entries = repo.index.entries
    except Exception:
        return blob_oids
    for (path, stage), entry in entries.items():
        if stage != 0:
            continue
        hexsha = getattr(entry, "hexsha", None)
        if hexsha:
            blob_oids[normalize_path(str(path))] = str(hexsha)
    return blob_oids


def _package_root_for_path(repo_root: str, relative_path: str) -> str:
    current = Path(relative_path).parent
    while True:
        candidate = "." if str(current) in {"", "."} else normalize_path(str(current))
        absolute = Path(repo_root) if candidate == "." else Path(repo_root) / candidate
        if any((absolute / marker).exists() for marker in _PACKAGE_SCOPE_MARKERS):
            return candidate
        if candidate == ".":
            return "."
        current = current.parent


def build_file_inventory(
    *,
    repo_root: str,
    supported_extensions: list[str],
    ignore_patterns: list[str],
    max_file_size_mb: int | float,
    include_fingerprints: bool,
    logger: logging.Logger | None = None,
) -> FileInventory:
    """Scan a repository once into typed file-fingerprint records."""
    log = logger or logging.getLogger(__name__)
    repo_root = os.path.abspath(repo_root)
    max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
    git_context = _git_worktree_root(repo_root) if include_fingerprints else None
    repo, worktree_root = git_context if git_context else (None, None)
    dirty_paths: set[str] = _dirty_git_paths(repo) if repo is not None else set()
    git_blob_oids = _index_blob_oids(repo) if repo is not None else {}
    files: list[FileFingerprint] = []

    for root, dirs, filenames in os.walk(repo_root):
        dirs[:] = [
            d
            for d in dirs
            if not should_ignore_path(os.path.join(root, d), ignore_patterns)
        ]

        for filename in filenames:
            file_path = os.path.join(root, filename)
            relative_path = normalize_path(os.path.relpath(file_path, repo_root))
            if should_ignore_path(relative_path, ignore_patterns):
                continue
            extension = Path(file_path).suffix
            if extension not in supported_extensions:
                continue
            try:
                stat = os.stat(file_path)
            except OSError as exc:
                log.warning("Error accessing file %s: %s", relative_path, exc)
                continue
            if stat.st_size > max_file_size_bytes:
                log.warning(
                    "Skipping large file: %s (%.2f MB)",
                    relative_path,
                    stat.st_size / 1024 / 1024,
                )
                continue

            content_hash: str | None = None
            git_blob_oid: str | None = None
            fingerprint_source: str | None = None
            if include_fingerprints:
                git_relative = (
                    _git_relative_path(file_path, worktree_root)
                    if worktree_root is not None
                    else None
                )
                if (
                    git_relative
                    and git_relative not in dirty_paths
                    and git_relative in git_blob_oids
                ):
                    git_blob_oid = git_blob_oids[git_relative]
                    fingerprint_source = "git_blob_oid"
                else:
                    content_hash = compute_file_hash(file_path) or None
                    fingerprint_source = "content_hash" if content_hash else None

            language = get_language_from_extension(extension)
            files.append(
                FileFingerprint(
                    path=normalize_path(file_path),
                    relative_path=relative_path,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    extension=extension,
                    language=language,
                    package_root=_package_root_for_path(repo_root, relative_path),
                    supported_tool_eligible=language in _TOOL_ELIGIBLE_LANGUAGES,
                    content_hash=content_hash,
                    git_blob_oid=git_blob_oid,
                    fingerprint_source=fingerprint_source,
                )
            )

    return FileInventory(repo_root=normalize_path(repo_root), files=tuple(files))
