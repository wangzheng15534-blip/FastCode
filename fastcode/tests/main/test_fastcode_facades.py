"""Tests for FastCode facade methods that hide internal state from entry frames."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


def _minimal_fastcode() -> Any:
    """Create a FastCode.__new__ shell with just enough state for facade tests."""
    fc = object.__new__(pytest.importorskip("fastcode.main.fastcode").FastCode)
    fc.repo_loaded = True
    fc.repo_indexed = True
    fc.repo_info = {"name": "test-repo", "file_count": 10}
    fc.multi_repo_mode = False
    fc.config = {
        "retrieval": {
            "retrieval_backend": "pg_hybrid",
            "graph_expansion_backend": "ir",
        }
    }
    fc.vector_store = MagicMock()
    fc.vector_store.scan_available_indexes.return_value = [
        {"name": "test-repo", "element_count": 100}
    ]
    fc.loaded_repositories = {"test-repo": {"file_count": 10}}
    fc.snapshot_store = SimpleNamespace(db_runtime=SimpleNamespace(backend="sqlite"))
    return fc


class TestGetStatusInfo:
    def test_returns_all_status_fields(self):
        fc = _minimal_fastcode()
        info = fc.get_status_info()
        assert info["repo_loaded"] is True
        assert info["repo_indexed"] is True
        assert info["storage_backend"] == "sqlite"
        assert info["retrieval_backend"] == "pg_hybrid"
        assert info["graph_expansion_backend"] == "ir"
        assert len(info["available_repositories"]) == 1
        assert len(info["loaded_repositories"]) >= 0

    def test_full_scan_bypasses_cache(self):
        fc = _minimal_fastcode()
        fc.get_status_info(full_scan=True)
        fc.vector_store.scan_available_indexes.assert_called_with(use_cache=False)

    def test_default_uses_cache(self):
        fc = _minimal_fastcode()
        fc.get_status_info()
        fc.vector_store.scan_available_indexes.assert_called_with(use_cache=True)


class TestSearchSymbols:
    def test_ranks_exact_over_prefix_over_contains(self):
        fc = _minimal_fastcode()
        fc.vector_store.metadata = [
            {"name": "load_config", "type": "function"},
            {"name": "load_config_all", "type": "function"},
            {"name": "reload_config", "type": "function"},
            {"name": "get_config_loader", "type": "function"},
            {"name": "Repo", "type": "class"},
            {"type": "repository_overview"},
        ]
        results = fc.search_symbols("load_config")
        names = [r["name"] for r in results]
        # exact match comes before prefix match
        assert names.index("load_config") < names.index("load_config_all")
        # contains match comes after prefix
        assert "reload_config" in names
        # non-matching items excluded
        assert "get_config_loader" not in names
        assert "Repo" not in names

    def test_filters_by_type(self):
        fc = _minimal_fastcode()
        fc.vector_store.metadata = [
            {"name": "FastCode", "type": "class"},
            {"name": "fastcode_query", "type": "function"},
        ]
        results = fc.search_symbols("fastcode", symbol_type="class")
        assert len(results) == 1
        assert results[0]["name"] == "FastCode"

    def test_limits_to_20(self):
        fc = _minimal_fastcode()
        fc.vector_store.metadata = [
            {"name": f"item_{i}", "type": "function"} for i in range(30)
        ]
        results = fc.search_symbols("item")
        assert len(results) == 20


class TestGetFileStructure:
    def test_returns_none_for_unknown_file(self):
        fc = _minimal_fastcode()
        fc.vector_store.metadata = []
        assert fc.get_file_structure("nonexistent.py") is None

    def test_returns_file_classes_and_functions(self):
        fc = _minimal_fastcode()
        fc.vector_store.metadata = [
            {
                "name": "main.py",
                "type": "file",
                "relative_path": "src/main.py",
                "language": "python",
                "metadata": {"total_lines": 100, "code_lines": 80},
            },
            {
                "name": "App",
                "type": "class",
                "relative_path": "src/main.py",
                "signature": "class App",
                "start_line": 10,
                "end_line": 50,
                "metadata": {"methods": ["run", "shutdown"]},
            },
            {
                "name": "helper",
                "type": "function",
                "relative_path": "src/main.py",
                "signature": "def helper()",
                "start_line": 52,
                "end_line": 60,
                "metadata": {},
            },
            {"type": "repository_overview"},
        ]
        result = fc.get_file_structure("main.py")
        assert result is not None
        assert len(result["classes"]) == 1
        assert len(result["functions"]) == 1
        assert result["file"]["language"] == "python"


class TestWalkCallChain:
    def test_returns_none_for_unknown_symbol(self):
        fc = _minimal_fastcode()
        fc.graph_builder = MagicMock()
        fc.graph_builder.element_by_name = {}
        fc.graph_builder.element_by_id = {}
        assert fc.walk_call_chain("nonexistent_fn") is None

    def test_returns_target_with_callers_and_callees(self):
        fc = _minimal_fastcode()
        target = SimpleNamespace(
            id="fn1",
            name="main",
            type="function",
            relative_path="main.py",
            start_line=10,
        )
        caller = SimpleNamespace(
            id="fn0",
            name="entry",
            type="function",
            relative_path="entry.py",
            start_line=5,
        )
        callee = SimpleNamespace(
            id="fn2",
            name="helper",
            type="function",
            relative_path="util.py",
            start_line=20,
        )
        fc.graph_builder = MagicMock()
        fc.graph_builder.element_by_name = {"main": target}
        fc.graph_builder.element_by_id = {
            "fn0": caller,
            "fn1": target,
            "fn2": callee,
        }
        fc.graph_builder.get_callers.return_value = ["fn0"]
        fc.graph_builder.get_callees.return_value = ["fn2"]

        result = fc.walk_call_chain("main", direction="both", max_hops=2)
        assert result is not None
        assert result["name"] == "main"
        assert len(result["callers"]) == 1
        assert result["callers"][0]["name"] == "entry"
        assert len(result["callees"]) == 1
        assert result["callees"][0]["name"] == "helper"


class TestReindexRepository:
    def test_rejects_nonexistent_local_path(self):
        fc = _minimal_fastcode()
        fc.apply_env_ignore_patterns = MagicMock()
        fc.load_repository = MagicMock()
        fc.index_repository = MagicMock()
        fc.logger = MagicMock()
        result = fc.reindex_repository("/no/such/path/repo")
        assert "Error" in result
        assert "does not exist" in result
