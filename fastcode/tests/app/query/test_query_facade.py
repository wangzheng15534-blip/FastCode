"""Tests for QueryFacade -- extracted query methods from FastCode."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from fastcode.app.query.facade import QueryFacade
from fastcode.runtime_support.runtime_state import RuntimeState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FacadeHarness:
    """Test harness that holds the facade and its mock dependencies."""

    def __init__(self) -> None:
        self.query_handler = MagicMock()
        self.vector_store = MagicMock()
        self.graph_builder = MagicMock()
        self.snapshot_store = MagicMock()
        self.ir_graph_builder = MagicMock()
        self.snapshot_symbol_index = MagicMock()
        self.pipeline = MagicMock()
        self.state = RuntimeState()
        self.facade = QueryFacade(
            query_handler=self.query_handler,
            vector_store=self.vector_store,
            graph_builder=self.graph_builder,
            snapshot_store=self.snapshot_store,
            ir_graph_builder=self.ir_graph_builder,
            snapshot_symbol_index=self.snapshot_symbol_index,
            pipeline=self.pipeline,
            state=self.state,
        )


# ---------------------------------------------------------------------------
# Pure delegation tests: query, query_snapshot
# ---------------------------------------------------------------------------


class TestQuery:
    def test_delegates_to_query_handler(self) -> None:
        h = _FacadeHarness()
        h.query_handler.query.return_value = {"answer": "yes"}
        result = h.facade.query(
            question="what is X?",
            filters=None,
            repo_filter=["repo"],
            session_id="s1",
            enable_multi_turn=True,
        )
        assert result == {"answer": "yes"}
        h.query_handler.query.assert_called_once_with(
            question="what is X?",
            filters=None,
            repo_filter=["repo"],
            session_id="s1",
            enable_multi_turn=True,
            use_agency_mode=None,
            prompt_builder=None,
        )

    def test_passes_all_kwargs(self) -> None:
        h = _FacadeHarness()
        h.query_handler.query.return_value = {}
        pb = lambda q, s, f, c: q  # noqa: E731
        h.facade.query(
            question="q",
            filters={"k": "v"},
            repo_filter=None,
            session_id="s2",
            enable_multi_turn=False,
            use_agency_mode=True,
            prompt_builder=pb,
        )
        h.query_handler.query.assert_called_once_with(
            question="q",
            filters={"k": "v"},
            repo_filter=None,
            session_id="s2",
            enable_multi_turn=False,
            use_agency_mode=True,
            prompt_builder=pb,
        )

    def test_acquires_read_lock(self) -> None:
        h = _FacadeHarness()
        lock_held = False

        def _side_effect(**kwargs: Any) -> dict[str, Any]:
            nonlocal lock_held
            # During call, the read lock should be held
            lock_held = h.state._lock._readers > 0
            return {}

        h.query_handler.query.side_effect = _side_effect
        h.facade.query(question="q")
        assert lock_held


class TestQuerySnapshot:
    def test_delegates_to_query_handler(self) -> None:
        h = _FacadeHarness()
        h.query_handler.query_snapshot.return_value = {"answer": "snap"}
        result = h.facade.query_snapshot(
            question="q",
            repo_name="repo",
            ref_name="main",
            snapshot_id="snap:repo:abc",
        )
        assert result == {"answer": "snap"}
        h.query_handler.query_snapshot.assert_called_once_with(
            question="q",
            repo_name="repo",
            ref_name="main",
            snapshot_id="snap:repo:abc",
            filters=None,
            session_id=None,
            enable_multi_turn=None,
        )

    def test_acquires_read_lock(self) -> None:
        h = _FacadeHarness()
        lock_held = False

        def _side_effect(**kwargs: Any) -> dict[str, Any]:
            nonlocal lock_held
            lock_held = h.state._lock._readers > 0
            return {}

        h.query_handler.query_snapshot.side_effect = _side_effect
        h.facade.query_snapshot(question="q")
        assert lock_held


# ---------------------------------------------------------------------------
# search_symbols
# ---------------------------------------------------------------------------


class TestSearchSymbols:
    def test_ranks_exact_over_prefix_over_contains(self) -> None:
        h = _FacadeHarness()
        h.vector_store.metadata = [
            {"name": "load_config", "type": "function"},
            {"name": "load_config_all", "type": "function"},
            {"name": "reload_config", "type": "function"},
            {"name": "get_config_loader", "type": "function"},
            {"name": "Repo", "type": "class"},
            {"type": "repository_overview"},
        ]
        results = h.facade.search_symbols("load_config")
        names = [r["name"] for r in results]
        assert names.index("load_config") < names.index("load_config_all")
        assert "reload_config" in names
        assert "get_config_loader" not in names
        assert "Repo" not in names

    def test_filters_by_type(self) -> None:
        h = _FacadeHarness()
        h.vector_store.metadata = [
            {"name": "FastCode", "type": "class"},
            {"name": "fastcode_query", "type": "function"},
        ]
        results = h.facade.search_symbols("fastcode", symbol_type="class")
        assert len(results) == 1
        assert results[0]["name"] == "FastCode"

    def test_limits_to_20(self) -> None:
        h = _FacadeHarness()
        h.vector_store.metadata = [
            {"name": f"item_{i}", "type": "function"} for i in range(30)
        ]
        results = h.facade.search_symbols("item")
        assert len(results) == 20

    def test_case_insensitive(self) -> None:
        h = _FacadeHarness()
        h.vector_store.metadata = [
            {"name": "MyFunction", "type": "function"},
        ]
        results = h.facade.search_symbols("myfunction")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# get_file_structure
# ---------------------------------------------------------------------------


class TestGetFileStructure:
    def test_returns_none_for_unknown_file(self) -> None:
        h = _FacadeHarness()
        h.vector_store.metadata = []
        assert h.facade.get_file_structure("nonexistent.py") is None

    def test_returns_file_classes_and_functions(self) -> None:
        h = _FacadeHarness()
        h.vector_store.metadata = [
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
        result = h.facade.get_file_structure("main.py")
        assert result is not None
        assert len(result["classes"]) == 1
        assert len(result["functions"]) == 1
        assert result["file"]["language"] == "python"


# ---------------------------------------------------------------------------
# walk_call_chain
# ---------------------------------------------------------------------------


class TestWalkCallChain:
    def test_returns_none_for_unknown_symbol(self) -> None:
        h = _FacadeHarness()
        h.graph_builder.element_by_name = {}
        h.graph_builder.element_by_id = {}
        assert h.facade.walk_call_chain("nonexistent_fn") is None

    def test_returns_target_with_callers_and_callees(self) -> None:
        h = _FacadeHarness()
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
        h.graph_builder.element_by_name = {"main": target}
        h.graph_builder.element_by_id = {
            "fn0": caller,
            "fn1": target,
            "fn2": callee,
        }
        h.graph_builder.get_callers.return_value = ["fn0"]
        h.graph_builder.get_callees.return_value = ["fn2"]

        result = h.facade.walk_call_chain("main", direction="both", max_hops=2)
        assert result is not None
        assert result["name"] == "main"
        assert len(result["callers"]) == 1
        assert result["callers"][0]["name"] == "entry"
        assert len(result["callees"]) == 1
        assert result["callees"][0]["name"] == "helper"

    def test_max_hops_capped_at_5(self) -> None:
        h = _FacadeHarness()
        target = SimpleNamespace(
            id="fn1", name="fn", type="function",
            relative_path="a.py", start_line=1,
        )
        h.graph_builder.element_by_name = {"fn": target}
        h.graph_builder.element_by_id = {"fn1": target}
        h.graph_builder.get_callers.return_value = []
        h.graph_builder.get_callees.return_value = []

        h.facade.walk_call_chain("fn", max_hops=100)
        # The method caps at 5 -- just verify it doesn't crash


# ---------------------------------------------------------------------------
# query_stream
# ---------------------------------------------------------------------------


class TestQueryStream:
    def test_delegates_without_snapshot(self) -> None:
        h = _FacadeHarness()
        chunks = [("hello", None), (" world", {"status": "complete"})]
        h.query_handler.query_stream.return_value = iter(chunks)

        results = list(h.facade.query_stream(question="q"))
        assert len(results) == 2
        assert results[0] == ("hello", None)
        h.query_handler.query_stream.assert_called_once_with(
            question="q",
            filters=None,
            repo_filter=None,
            session_id=None,
            enable_multi_turn=None,
            use_agency_mode=None,
            prompt_builder=None,
        )

    def test_with_snapshot_loads_artifacts(self) -> None:
        h = _FacadeHarness()
        snapshot_record = MagicMock()
        snapshot_record.artifact_key = "key-1"
        h.snapshot_store.get_snapshot_record.return_value = snapshot_record

        loaded_artifacts = MagicMock()
        loaded_artifacts.artifact_key = "key-1"
        loaded_artifacts.retriever = MagicMock()
        h.pipeline.load_snapshot_artifacts_handle.return_value = loaded_artifacts

        h.query_handler.query_stream.return_value = iter([("result", None)])

        results = list(
            h.facade.query_stream(
                question="q",
                filters={"snapshot_id": "snap:repo:abc"},
            )
        )
        assert len(results) == 1
        h.snapshot_store.get_snapshot_record.assert_called_once_with("snap:repo:abc")
        h.pipeline.load_snapshot_artifacts_handle.assert_called_once_with(
            "key-1", snapshot_id="snap:repo:abc"
        )
        h.query_handler._ensure_snapshot_symbol_index.assert_called_once_with(
            "snap:repo:abc"
        )


# ---------------------------------------------------------------------------
# _escalate_query_semantics
# ---------------------------------------------------------------------------


class TestEscalateQuerySemantics:
    def test_skips_when_snapshot_not_found(self) -> None:
        h = _FacadeHarness()
        h.snapshot_store.load_snapshot.return_value = None
        result = h.facade._escalate_query_semantics(
            snapshot_id="snap:x",
            retrieved=[],
            processed_query=None,
            budget="changed_files",
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "snapshot_not_found"

    def test_skips_when_no_target_paths(self) -> None:
        h = _FacadeHarness()
        h.snapshot_store.load_snapshot.return_value = MagicMock()
        result = h.facade._escalate_query_semantics(
            snapshot_id="snap:x",
            retrieved=[{"element": {"name": "x"}}],
            processed_query=MagicMock(filters={}),
            budget="changed_files",
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "no_target_paths"

    def test_applies_semantic_resolvers(self) -> None:
        h = _FacadeHarness()
        snapshot = MagicMock()
        snapshot.metadata = {}
        h.snapshot_store.load_snapshot.return_value = snapshot

        elem = SimpleNamespace(id="e1", name="foo", type="function")
        h.graph_builder.element_by_id = {"e1": elem}
        h.graph_builder.element_by_name = {}

        upgraded = MagicMock()
        upgraded.metadata = {"semantic_resolver_runs": ["r1"]}
        h.pipeline._apply_semantic_resolvers.return_value = upgraded

        mock_ir_graph = MagicMock()
        h.ir_graph_builder.build_graphs.return_value = mock_ir_graph

        retriever = MagicMock()

        result = h.facade._escalate_query_semantics(
            snapshot_id="snap:x",
            retrieved=[
                {"element": {"relative_path": "src/main.py"}},
            ],
            processed_query=MagicMock(filters={}),
            budget="changed_files",
            retriever=retriever,
            graph_builder=h.graph_builder,
        )
        assert result["status"] == "applied"
        assert result["rerun_retrieval"] is True
        h.pipeline._apply_semantic_resolvers.assert_called_once()
        h.snapshot_symbol_index.register_snapshot.assert_called_once_with(upgraded)
        h.ir_graph_builder.build_graphs.assert_called_once_with(upgraded)


# ---------------------------------------------------------------------------
# _apply_semantic_resolvers (thin wrapper)
# ---------------------------------------------------------------------------


class TestApplySemanticResolvers:
    def test_delegates_to_pipeline(self) -> None:
        h = _FacadeHarness()
        expected = MagicMock()
        h.pipeline._apply_semantic_resolvers.return_value = expected
        snap = MagicMock()
        result = h.facade._apply_semantic_resolvers(
            snapshot=snap,
            elements=[],
            graph_context=None,
            target_paths=set(),
            warnings=[],
            budget="changed_files",
        )
        assert result is expected
        h.pipeline._apply_semantic_resolvers.assert_called_once_with(
            snapshot=snap,
            elements=[],
            graph_context=None,
            target_paths=set(),
            warnings=[],
            budget="changed_files",
        )
