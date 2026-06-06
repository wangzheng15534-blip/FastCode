"""Ignore-pattern matching for indexing shell scans."""

from __future__ import annotations

from typing import Any

from pathspec import PathSpec  # type: ignore[import-untyped]
from pathspec.patterns.gitwildmatch import (
    GitWildMatchPattern,  # type: ignore[import-untyped]
)


def should_ignore_path(path: str, ignore_patterns: list[str]) -> bool:
    """Return whether a relative path matches repository ignore patterns."""
    from_lines: Any = PathSpec.from_lines
    spec: Any = from_lines(GitWildMatchPattern, ignore_patterns)
    return bool(spec.match_file(path))
