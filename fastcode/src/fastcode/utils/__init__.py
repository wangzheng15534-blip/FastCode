"""fastcode.utils - domain-independent utility exports.

Keep this package import light. Pure modules import ``fastcode.utils.json`` and
``fastcode.utils.paths`` frequently, so package initialization must not import
optional tokenizer/config/pathspec dependencies from ``_compat``.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastcode.utils.hashing import deterministic_event_id, projection_params_hash
from fastcode.utils.json import (
    extract_json_from_response,
    remove_json_comments,
    robust_json_parse,
    safe_jsonable,
    sanitize_json_string,
)
from fastcode.utils.paths import (
    get_language_from_extension,
    infer_language_from_file_context,
    projection_scope_key,
)

if TYPE_CHECKING:
    from fastcode.utils._compat import (
        calculate_code_complexity,
        clean_docstring,
        config_to_legacy_dict,
        count_tokens,
        extract_code_snippet,
        format_code_block,
        load_config,
        load_runtime_config,
        merge_dicts,
        prepare_runtime_config_mapping,
        resolve_config_paths,
        safe_get,
        setup_logging,
        should_ignore_path,
        truncate_to_tokens,
    )

_COMPAT_EXPORTS = {
    "calculate_code_complexity",
    "clean_docstring",
    "config_to_legacy_dict",
    "count_tokens",
    "extract_code_snippet",
    "format_code_block",
    "load_config",
    "load_runtime_config",
    "merge_dicts",
    "prepare_runtime_config_mapping",
    "resolve_config_paths",
    "safe_get",
    "setup_logging",
    "should_ignore_path",
    "truncate_to_tokens",
}


def utc_now() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


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


def __getattr__(name: str) -> Any:
    if name in _COMPAT_EXPORTS:
        value = getattr(import_module("fastcode.utils._compat"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "calculate_code_complexity",
    "clean_docstring",
    "compute_file_hash",
    "config_to_legacy_dict",
    "count_tokens",
    "deterministic_event_id",
    "ensure_dir",
    "extract_code_snippet",
    "extract_json_from_response",
    "format_code_block",
    "get_file_extension",
    "get_language_from_extension",
    "get_repo_name_from_url",
    "infer_language_from_file_context",
    "is_supported_file",
    "is_text_file",
    "load_config",
    "load_runtime_config",
    "merge_dicts",
    "normalize_path",
    "prepare_runtime_config_mapping",
    "projection_params_hash",
    "projection_scope_key",
    "remove_json_comments",
    "resolve_config_paths",
    "robust_json_parse",
    "safe_get",
    "safe_jsonable",
    "sanitize_json_string",
    "setup_logging",
    "should_ignore_path",
    "truncate_to_tokens",
    "utc_now",
]
