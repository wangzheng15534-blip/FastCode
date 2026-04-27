# fastcode/effects/fs.py
"""Thin wrappers for file system I/O."""

from __future__ import annotations

import os


def read_file(path: str) -> str:
    """Read file contents."""
    with open(path) as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    """Write file contents."""
    with open(path, "w") as f:
        f.write(content)


def file_exists(path: str) -> bool:
    """Check if file exists."""
    return os.path.exists(path)
