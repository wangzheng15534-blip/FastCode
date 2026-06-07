"""Verify entry frames access use_flow operations through facade attributes only.

FCIS rule: entry_frame → use_flow (NOT entry_frame → assembly_root → use_flow).

Entry frames (api/, mcp/, client/) must call facades (fc.indexing.*, fc.query.*,
fc.store.*, fc.context.*, fc.cache.*, fc.publishing.*, fc.projection.*).
They must NOT call use_flow dispatch methods directly on FastCode.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "fastcode"

ENTRY_FRAME_FILES = [
    PACKAGE_ROOT / "api" / "routes.py",
    PACKAGE_ROOT / "api" / "web.py",
    PACKAGE_ROOT / "mcp" / "server.py",
    PACKAGE_ROOT / "client" / "cli.py",
]

FACADE_ATTRS = {
    "indexing",
    "query",
    "store",
    "context",
    "cache",
    "publishing",
    "projection",
}

# Methods that are legitimate assembly_root concerns and may be called directly.
ASSEMBLY_ROOT_PUBLIC_API = {
    "build_diagnostic_bundle",
    "cleanup",
    "shutdown",
    "remove_repository",
    "ensure_loaded",
    "ensure_repos_ready",
    "apply_env_ignore_patterns",
    "apply_repository_runtime_overrides",
}

# Use_flow methods that MUST be accessed through facade attributes.
USE_FLOW_METHODS = {
    # IndexingFacade
    "load_repository",
    "upload_repository_zip",
    "index_repository",
    "load_and_index",
    "upload_and_index",
    "load_multiple_repositories",
    "reindex_repository",
    "run_index_pipeline",
    "incremental_reindex",
    # QueryFacade
    "query",
    "query_snapshot",
    "query_stream",
    "search_symbols",
    "get_file_structure",
    "walk_call_chain",
    # StoreFacade
    "get_status_info",
    "get_repository_summary",
    "list_snapshots",
    "get_snapshot",
    "list_repository_branches",
    "get_symbol_info",
    "is_repo_indexed",
    "repo_name_from_source",
    # ContextFacade
    "list_sessions",
    "get_session",
    "get_session_context",
    "get_context",
    "build_context",
    "create_session",
    "delete_session",
    "get_session_multi_turn",
    # CacheFacade
    "clear_cache",
    "get_cache_stats",
    "refresh_index_cache",
    "load_cached_repos",
    "invalidate_scan_cache",
    "cache_query",
    "list_cache_entries",
    "clear_query_cache",
    "get_cache_info",
    # PublishingFacade
    "get_index_run",
    "publish_index_run",
    "retry_pending_publishes",
    "retry_index_run_recovery",
    "process_semantic_repair_frontier",
    "process_redo_tasks",
    # ProjectionFacade
    "build_projection",
    "get_projection",
    "list_projections",
    "delete_projection",
}


def _extract_fc_attribute_calls(
    source: str, fc_var_patterns: list[str]
) -> list[tuple[str, int, str]]:
    """Find all `fc.method()` calls where method is a use_flow dispatch.

    Skips `fc.facade.method()` (two-level attribute access is facade usage).

    Returns list of (filename_stem, line_number, method_name).
    """
    calls: list[tuple[str, int, str]] = []
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue

        # Match fc.method() or fastcode.method() or fastcode_instance.method()
        if isinstance(func.value, ast.Name) and func.value.id in fc_var_patterns:
            method = func.attr
            if method in USE_FLOW_METHODS:
                calls.append(("", node.lineno, method))

    return calls


def _check_raw_pattern(
    source: str, fc_var_patterns: list[str]
) -> list[tuple[int, str]]:
    """Regex fallback for patterns AST might miss (e.g., function references)."""
    violations: list[tuple[int, str]] = []
    for pattern in fc_var_patterns:
        for method in USE_FLOW_METHODS:
            # Match fc.method but NOT fc.method.xxx (which is facade attribute access)
            regex = rf"\b{pattern}\.{method}\b(?!\.)"
            for match in re.finditer(regex, source):
                line = source[: match.start()].count("\n") + 1
                violations.append((line, f"{pattern}.{method}"))
    return violations


def test_entry_frames_use_facades():
    """Entry frames must call use_flow methods through facade attributes."""
    fc_patterns = ["fastcode", "fastcode_instance", "fc"]
    violations: list[str] = []

    for path in ENTRY_FRAME_FILES:
        if not path.exists():
            continue
        source = path.read_text()
        stem = path.stem

        # AST-based check
        ast_calls = _extract_fc_attribute_calls(source, fc_patterns)
        for _, lineno, method in ast_calls:
            violations.append(
                f"{stem}:{lineno}: direct call fc.{method}() instead of fc.<facade>.{method}()"
            )

        # Regex-based check for function references (e.g., passed as callbacks)
        raw = _check_raw_pattern(source, fc_patterns)
        for lineno, call in raw:
            violations.append(
                f"{stem}:{lineno}: direct reference {call} instead of fc.<facade>.{call}"
            )

    assert not violations, "FCIS violation: entry frames bypass facades\n" + "\n".join(
        violations
    )


def test_facade_attrs_are_documented():
    """All facade attributes used in entry frames should be in FACADE_ATTRS."""
    found_attrs: set[str] = set()
    for path in ENTRY_FRAME_FILES:
        if not path.exists():
            continue
        source = path.read_text()
        for pattern in ["fastcode", "fastcode_instance", "fc"]:
            for attr in FACADE_ATTRS:
                if f"{pattern}.{attr}." in source:
                    found_attrs.add(attr)

    missing = found_attrs - FACADE_ATTRS
    assert not missing, f"Facade attrs used but not in FACADE_ATTRS: {missing}"
