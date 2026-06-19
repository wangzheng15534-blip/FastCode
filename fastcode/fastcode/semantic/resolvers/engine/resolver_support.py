"""Support helpers for semantic resolver implementations."""

from __future__ import annotations

__all__ = ["_hash_id", "_normalize_path"]

import hashlib
import posixpath


def _hash_id(prefix: str, payload: str) -> str:
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()
    return f"{prefix}:{digest}"


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    normalized = normalized.removeprefix("./")
    return posixpath.normpath(normalized)
