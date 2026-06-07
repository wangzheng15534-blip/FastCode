"""Small stdlib-only filesystem and path helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


def compute_file_hash(file_path: str) -> str:
    """Compute an MD5 hash for a local file, returning empty string on failure."""
    hash_md5 = hashlib.md5(usedforsecurity=False)
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except OSError:
        return ""


def compute_file_sha256(file_path: str | Path) -> str | None:
    """Compute a SHA-256 hash for a local file, returning None on failure."""
    hash_sha256 = hashlib.sha256()
    try:
        with Path(file_path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hash_sha256.update(chunk)
    except OSError:
        return None
    return hash_sha256.hexdigest()


def ensure_dir(directory: str) -> None:
    """Ensure a directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def file_content_identity(file_path: str | Path) -> dict[str, Any] | None:
    """Return stable size and content hash identity for a readable local file."""
    path = Path(file_path)
    try:
        stat = path.stat()
    except OSError:
        return None
    digest = compute_file_sha256(path)
    if digest is None:
        return None
    return {"size": int(stat.st_size), "content_hash": digest}


def file_stat_identity(file_path: str | Path) -> dict[str, Any] | None:
    """Return size and mtime identity for a local path without hashing content."""
    try:
        stat = Path(file_path).stat()
    except OSError:
        return None
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def get_file_extension(file_path: str) -> str:
    """Return the suffix for a file path."""
    return Path(file_path).suffix


def get_repo_name_from_url(url: str) -> str:
    """Extract a repository name from a URL or URL-like path."""
    url = url.removesuffix(".git")
    parts = url.rstrip("/").split("/")
    return parts[-1] if parts else "unknown_repo"


def is_supported_file(file_path: str, supported_extensions: list[str]) -> bool:
    """Return whether *file_path* has a supported extension."""
    return get_file_extension(file_path) in supported_extensions


def is_text_file(file_path: str) -> bool:
    """Return whether a file can be read as UTF-8 text."""
    try:
        with open(file_path, encoding="utf-8") as f:
            f.read(512)
        return True
    except (OSError, UnicodeDecodeError):
        return False


def normalize_path(path: str) -> str:
    """Normalize a path to a forward-slash representation."""
    return os.path.normpath(path).replace("\\", "/")


def resolve_absolute_root(root: str | None) -> str:
    """Resolve an optional root path to an absolute filesystem path."""
    return os.path.abspath(root or os.getcwd())
