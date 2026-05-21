"""
Multi-language SCIP indexer profiles and language detection helpers.

This module maps languages to SCIP toolchain profiles and detects candidate
languages from repository file extensions.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SCIPIndexerProfile:
    language: str
    binary_name: str
    extra_args: tuple[str, ...]
    experimental: bool = False


SUPPORTED_LANGUAGES: dict[str, tuple[str, list[str]]] = {
    "java": ("scip-java", ["index", "--output"]),
    "kotlin": ("scip-java", ["index", "--output"]),
    "scala": ("scip-java", ["index", "--output"]),
    "go": ("scip-go", ["index", "--output"]),
    "python": ("scip-python", ["index", "--output"]),
    "ruby": ("scip-ruby", ["index", "--output"]),
    "typescript": ("scip-typescript", ["index", "--output"]),
    "javascript": ("scip-typescript", ["index", "--output"]),
    "c": ("scip-clang", ["index", "--output"]),
    "cpp": ("scip-clang", ["index", "--output"]),
    "csharp": ("scip-dotnet", ["index", "--output"]),
    "rust": ("rust-analyzer", ["scip", "--output"]),
    "zig": ("zls", ["scip", "--output"]),
    "fortran": ("fortls", ["--scip-output"]),
    "julia": (
        "julia",
        [
            "--project=@.",
            "-e",
            "using SymbolServer; SymbolServer.scip_index()",
            "--output",
        ],
    ),
}


def get_indexer_command(
    language: str,
    output_path: str,
) -> list[str] | None:
    """Build the indexer command for a language. Returns None if unsupported."""
    profile = get_scip_indexer_profile(language)
    if profile is None:
        return None
    return [profile.binary_name, *profile.extra_args, output_path]


def get_scip_indexer_profile(language: str) -> SCIPIndexerProfile | None:
    entry = SUPPORTED_LANGUAGES.get(language)
    if not entry:
        return None
    binary_name, extra_args = entry
    return SCIPIndexerProfile(
        language=language,
        binary_name=binary_name,
        extra_args=tuple(extra_args),
        experimental=language in _EXPERIMENTAL_SCIP_LANGUAGES,
    )


# Languages whose SCIP tooling is experimental / unstable.
_EXPERIMENTAL_SCIP_LANGUAGES = frozenset({"zig", "fortran", "julia"})


_EXTENSION_MAP: dict[str, str] = {
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".go": "go",
    ".py": "python",
    ".rb": "ruby",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
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
}

_SKIP_DIRS = frozenset({".git", ".hg", "node_modules", "__pycache__", ".venv", "venv"})


def detect_scip_languages(repo_path: str) -> list[str]:
    seen: set[str] = set()
    for _, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            _, ext = os.path.splitext(fname)
            lang = _EXTENSION_MAP.get(ext)
            if lang:
                seen.add(lang)
    return sorted(seen)


def detect_scip_languages_from_file_infos(
    file_infos: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Detect SCIP languages from a precomputed repository file inventory."""
    seen: set[str] = set()
    for file_info in file_infos:
        language = file_info.get("language")
        if isinstance(language, str) and language in SUPPORTED_LANGUAGES:
            seen.add(language)
            continue
        extension = file_info.get("extension")
        if not isinstance(extension, str) or not extension:
            path = file_info.get("relative_path") or file_info.get("path")
            _, extension = os.path.splitext(str(path or ""))
        lang = _EXTENSION_MAP.get(extension)
        if lang:
            seen.add(lang)
    return sorted(seen)


def detect_scip_languages_in_paths(
    repo_path: str,
    relative_paths: list[str],
) -> list[str]:
    seen: set[str] = set()
    repo_root = os.path.abspath(repo_path)
    scan_roots: set[str] = set()
    for raw_path in relative_paths:
        rel_path = os.path.normpath(str(raw_path or ""))
        if not rel_path:
            continue
        candidate_path = (
            os.path.abspath(rel_path)
            if os.path.isabs(rel_path)
            else os.path.abspath(os.path.join(repo_root, rel_path))
        )
        if not (
            candidate_path == repo_root
            or candidate_path.startswith(f"{repo_root}{os.sep}")
        ):
            continue

        _, ext = os.path.splitext(rel_path)
        lang = _EXTENSION_MAP.get(ext)
        if lang:
            seen.add(lang)
            continue
        if os.path.isdir(candidate_path):
            scan_roots.add(candidate_path)

    for scan_root in sorted(scan_roots):
        for _, dirs, files in os.walk(scan_root):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for fname in files:
                _, ext = os.path.splitext(fname)
                lang = _EXTENSION_MAP.get(ext)
                if lang:
                    seen.add(lang)
    return sorted(seen)
