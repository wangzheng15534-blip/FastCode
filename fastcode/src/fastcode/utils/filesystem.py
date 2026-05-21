"""Small stdlib-only filesystem and path helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


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


def ensure_dir(directory: str) -> None:
    """Ensure a directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_extension(file_path: str) -> str:
    """Return the suffix for a file path."""
    return Path(file_path).suffix


def get_repo_name_from_url(url: str) -> str:
    """Extract a repository name from a URL or URL-like path."""
    if url.endswith(".git"):
        url = url[:-4]
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
