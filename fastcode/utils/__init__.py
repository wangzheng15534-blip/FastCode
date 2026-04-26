"""fastcode.utils — domain-independent utilities (copy-paste test)."""

# Re-export all public names from the original utils module so that
# existing `from fastcode.utils import X` statements continue to work.
from fastcode.utils._compat import (
    calculate_code_complexity,
    clean_docstring,
    compute_file_hash,
    count_tokens,
    ensure_dir,
    extract_code_snippet,
    format_code_block,
    get_file_extension,
    get_repo_name_from_url,
    is_supported_file,
    is_text_file,
    load_config,
    merge_dicts,
    normalize_path,
    resolve_config_paths,
    safe_get,
    setup_logging,
    should_ignore_path,
    truncate_to_tokens,
    utc_now,
)
from fastcode.utils.hashing import deterministic_event_id, projection_params_hash
from fastcode.utils.json import (
    extract_json_from_response,
    remove_json_comments,
    robust_json_parse,
    safe_jsonable,
    sanitize_json_string,
)
from fastcode.utils.paths import get_language_from_extension, projection_scope_key

__all__ = [
    # From _compat (legacy re-exports)
    "calculate_code_complexity",
    "clean_docstring",
    "compute_file_hash",
    "count_tokens",
    # From utils sub-modules (new canonical locations)
    "deterministic_event_id",
    "ensure_dir",
    "extract_code_snippet",
    "extract_json_from_response",
    "format_code_block",
    "get_file_extension",
    "get_language_from_extension",
    "get_repo_name_from_url",
    "is_supported_file",
    "is_text_file",
    "load_config",
    "merge_dicts",
    "normalize_path",
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
