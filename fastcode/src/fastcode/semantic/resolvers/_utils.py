"""Shared utilities for semantic resolver implementations."""

from __future__ import annotations

__all__ = ["_hash_id", "_normalize_path", "validate_helper_paths"]

import hashlib
import posixpath
from pathlib import Path


def _hash_id(prefix: str, payload: str) -> str:
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()
    return f"{prefix}:{digest}"


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return posixpath.normpath(normalized)


def validate_helper_paths(
    paths: list[str], repo_root: str
) -> tuple[list[str], list[str]]:
    """Validate helper file paths are regular files within repo root.

    Returns (safe_paths, rejected_paths). Rejects symlinks, missing files,
    and files outside the repo root.
    """
    repo = Path(repo_root).resolve()
    safe: list[str] = []
    rejected: list[str] = []
    for p in paths:
        original = Path(p)
        resolved = original.resolve()
        if original.is_symlink():
            rejected.append(p)
            continue
        if not resolved.is_file():
            rejected.append(p)
            continue
        try:
            resolved.relative_to(repo)
        except ValueError:
            rejected.append(p)
            continue
        safe.append(p)
    return safe, rejected
