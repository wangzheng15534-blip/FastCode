"""Domain-independent path and extension utilities."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".rs": "rust",
    ".zig": "zig",
    ".f": "fortran",
    ".for": "fortran",
    ".f77": "fortran",
    ".f90": "fortran",
    ".f95": "fortran",
    ".f03": "fortran",
    ".f08": "fortran",
    ".jl": "julia",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
}

_CPP_HEADER_PATTERNS = (
    re.compile(r"\bnamespace\b"),
    re.compile(r"\btemplate\s*<"),
    re.compile(r"\bclass\s+[A-Za-z_]\w*"),
    re.compile(r"\bconstexpr\b"),
    re.compile(r"\btypename\b"),
    re.compile(r"\bstd::"),
    re.compile(r"\bvirtual\b"),
    re.compile(r"\bfriend\b"),
    re.compile(r"\boperator\s*[^;{]+\("),
    re.compile(r"^\s*(public|private|protected)\s*:", re.MULTILINE),
)


def _normalize_posix_path(value: str) -> str:
    return value.replace("\\", "/")


def _contains_cpp_header_markers(content: str | None) -> bool:
    if not content:
        return False
    return any(pattern.search(content) for pattern in _CPP_HEADER_PATTERNS)


def _infer_header_language_from_candidates(
    file_path: str,
    sibling_paths: Iterable[str],
) -> str | None:
    header_path = PurePosixPath(_normalize_posix_path(file_path))
    counts = {"c": 0, "cpp": 0}

    for sibling in sibling_paths:
        sibling_path = PurePosixPath(_normalize_posix_path(sibling))
        if sibling_path == header_path or sibling_path.parent != header_path.parent:
            continue
        language = get_language_from_extension(sibling_path.suffix)
        if language not in counts:
            continue
        if sibling_path.stem == header_path.stem:
            return language
        counts[language] += 1

    if counts["cpp"] > counts["c"] and counts["cpp"] > 0:
        return "cpp"
    if counts["c"] > counts["cpp"] and counts["c"] > 0:
        return "c"
    return None


def _infer_header_language_from_filesystem(file_path: str) -> str | None:
    try:
        parent = Path(file_path).parent
        if not parent.exists():
            return None
        sibling_paths = [str(child) for child in parent.iterdir() if child.is_file()]
    except OSError:
        return None
    return _infer_header_language_from_candidates(file_path, sibling_paths)


def projection_scope_key(
    scope_kind: str,
    snapshot_id: str,
    query: str | None,
    target_id: str | None,
    filters: dict[str, Any] | None,
) -> str:
    """Compute a deterministic hash key for a projection scope."""
    base = {
        "scope_kind": scope_kind,
        "snapshot_id": snapshot_id,
        "query": query or "",
        "target_id": target_id or "",
        "filters": filters or {},
    }
    payload = json.dumps(base, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


def get_language_from_extension(ext: str) -> str:
    """Get programming language from extension."""
    return _LANGUAGE_MAP.get(ext.lower(), "unknown")


def infer_language_from_file_context(
    file_path: str,
    content: str | None = None,
    *,
    sibling_paths: Iterable[str] | None = None,
) -> str:
    """Infer language from path plus lightweight header context.

    `.h` stays baseline `c` at the extension level, but this helper can
    upgrade it to `cpp` when the content or nearby source layout makes that
    intent explicit.
    """
    extension = Path(file_path).suffix.lower()
    language = get_language_from_extension(extension)
    if extension != ".h":
        return language
    if _contains_cpp_header_markers(content):
        return "cpp"
    sibling_language = (
        _infer_header_language_from_candidates(file_path, sibling_paths)
        if sibling_paths is not None
        else _infer_header_language_from_filesystem(file_path)
    )
    return sibling_language or language
