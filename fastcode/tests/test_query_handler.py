from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from fastcode.indexer import CodeElement
from fastcode.main import FastCode
from fastcode.query_handler import QueryPipeline
from fastcode.query_processor import ProcessedQuery
from fastcode.semantic_ir import IRSnapshot


def _processed_query(
    *,
    question: str,
    intent: str = "debug",
    filters: dict[str, object] | None = None,
) -> ProcessedQuery:
    return ProcessedQuery(
        original=question,
        expanded=question,
        keywords=[],
        intent=intent,
        subqueries=[],
        filters=dict(filters or {}),
        rewritten_query=None,
        pseudocode_hints=None,
        search_strategy=None,
    )


def _element(path: str, *, element_id: str = "file:a") -> CodeElement:
    return CodeElement(
        id=element_id,
        type="file",
        name=path,
        file_path=path,
        relative_path=path,
        language="python",
        start_line=1,
        end_line=1,
        code="pass",
        signature=None,
        docstring=None,
        summary=None,
        metadata={},
    )


def _query_pipeline(
    *,
    semantic_escalation_cb: Callable[..., dict[str, Any] | None] | None = None,
    retrieve_side_effect: Any = None,
) -> QueryPipeline:
    retriever = MagicMock()
    retriever.enable_agency_mode = False
    retriever.iterative_agent = None
    retriever.retrieve.side_effect = retrieve_side_effect or [
        [{"element": {"relative_path": "a.py"}}]
    ]
    answer_generator = MagicMock()
    answer_generator.generate.return_value = {"answer": "ok", "sources": []}
    query_processor = MagicMock()
    cache_manager = MagicMock()
    cache_manager.get_recent_summaries.return_value = []
    cache_manager.get_dialogue_history.return_value = []
    return QueryPipeline(
        config={"generation": {"enable_multi_turn": False}},
        logger=MagicMock(),
        retriever=retriever,
        query_processor=query_processor,
        answer_generator=answer_generator,
        cache_manager=cache_manager,
        manifest_store=MagicMock(),
        snapshot_store=MagicMock(),
        snapshot_symbol_index=MagicMock(),
        is_repo_indexed=lambda: True,
        load_artifacts_by_key=lambda artifact_key: True,
        semantic_escalation_cb=semantic_escalation_cb,
    )


def test_query_pipeline_reruns_retrieval_after_semantic_escalation() -> None:
    processed_query = _processed_query(
        question="How does auth reach the token store?",
        filters={"snapshot_id": "snap:1"},
    )
    pipeline = _query_pipeline(
        semantic_escalation_cb=lambda **kwargs: {
            "status": "applied",
            "budget": "path-critical",
            "rerun_retrieval": True,
        },
        retrieve_side_effect=[
            [{"element": {"relative_path": "a.py"}, "total_score": 1.0}],
            [{"element": {"relative_path": "a.py"}, "total_score": 2.0}],
        ],
    )
    pipeline.query_processor.process.return_value = processed_query

    result = pipeline.query(
        "How does auth reach the token store?",
        filters={"snapshot_id": "snap:1"},
    )

    assert pipeline.retriever.retrieve.call_count == 2
    assert result["semantic_escalation"]["budget"] == "path-critical"


def test_query_pipeline_skips_semantic_escalation_without_snapshot_scope() -> None:
    callback = MagicMock(return_value={"rerun_retrieval": True})
    processed_query = _processed_query(question="What is auth?", intent="what")
    pipeline = _query_pipeline(semantic_escalation_cb=callback)
    pipeline.query_processor.process.return_value = processed_query

    result = pipeline.query("What is auth?")

    callback.assert_not_called()
    assert pipeline.retriever.retrieve.call_count == 1
    assert "semantic_escalation" not in result


def test_fastcode_query_semantic_escalation_updates_ir_graphs() -> None:
    snapshot = IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:1",
        metadata={"semantic_resolver_runs": [{"language": "python"}]},
    )
    upgraded_snapshot = IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:1",
        metadata={"semantic_resolver_runs": [{"language": "python"}]},
    )
    element = _element("src/a.py")
    fc = FastCode.__new__(FastCode)
    fc.snapshot_store = SimpleNamespace(load_snapshot=lambda snapshot_id: snapshot)
    fc.graph_builder = SimpleNamespace(element_by_id={element.id: element})
    fc.ir_graph_builder = SimpleNamespace(build_graphs=lambda snap: "ir-graphs")
    fc.retriever = SimpleNamespace(set_ir_graphs=MagicMock())
    fc.snapshot_symbol_index = SimpleNamespace(register_snapshot=MagicMock())
    fc._apply_semantic_resolvers = MagicMock(return_value=upgraded_snapshot)

    result = fc._escalate_query_semantics(
        snapshot_id="snap:1",
        retrieved=[{"element": {"relative_path": "src/a.py"}}],
        processed_query=_processed_query(
            question="How does auth reach the token store?",
            filters={"file_path": "src/a.py"},
        ),
        budget="path-critical",
    )

    assert result["status"] == "applied"
    assert result["rerun_retrieval"] is True
    assert result["target_paths"] == ["src/a.py"]
    fc.retriever.set_ir_graphs.assert_called_once_with(
        "ir-graphs", snapshot_id="snap:1"
    )
    fc.snapshot_symbol_index.register_snapshot.assert_called_once_with(
        upgraded_snapshot
    )


# ---------------------------------------------------------------------------
# Regression gap #2: query-time semantic escalation changing IR graph
# expansion behavior end to end
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_semantic_escalation_changes_retrieval_results_end_to_end() -> None:
    """After escalation, the answer generator must receive expanded results
    from the second retrieval (not the first).
    """
    processed_query = _processed_query(
        question="How does auth reach the token store?",
        filters={"snapshot_id": "snap:repo:abc123"},
    )
    first_retrieval = [
        {"element": {"relative_path": "src/auth.py"}, "total_score": 1.0},
    ]
    second_retrieval = [
        {"element": {"relative_path": "src/auth.py"}, "total_score": 1.5},
        {"element": {"relative_path": "src/token_store.py"}, "total_score": 1.2},
        {"element": {"relative_path": "src/middleware.py"}, "total_score": 0.9},
    ]
    pipeline = _query_pipeline(
        semantic_escalation_cb=lambda **kwargs: {
            "status": "applied",
            "budget": "path-critical",
            "rerun_retrieval": True,
        },
        retrieve_side_effect=[first_retrieval, second_retrieval],
    )
    pipeline.query_processor.process.return_value = processed_query

    result = pipeline.query(
        "How does auth reach the token store?",
        filters={"snapshot_id": "snap:repo:abc123"},
    )

    assert pipeline.retriever.retrieve.call_count == 2
    assert result["semantic_escalation"]["budget"] == "path-critical"
    assert result["semantic_escalation"]["status"] == "applied"
    # The answer generator must receive the SECOND retrieval (expanded).
    answer_call_args = pipeline.answer_generator.generate.call_args
    generated_retrieved = answer_call_args[0][1]
    assert len(generated_retrieved) == 3
    paths = [r["element"]["relative_path"] for r in generated_retrieved]
    assert "src/token_store.py" in paths
    assert "src/middleware.py" in paths


@pytest.mark.regression
def test_local_budget_triggers_escalation_for_find_intent() -> None:
    """Queries with find intent and snapshot scope should escalate with
    budget='local' and still trigger a rerun.
    """
    processed_query = _processed_query(
        question="Where is the config loaded?",
        intent="find",
        filters={"snapshot_id": "snap:1"},
    )
    pipeline = _query_pipeline(
        semantic_escalation_cb=lambda **kwargs: {
            "status": "applied",
            "budget": "local",
            "rerun_retrieval": True,
        },
        retrieve_side_effect=[
            [{"element": {"relative_path": "a.py"}, "total_score": 1.0}],
            [{"element": {"relative_path": "a.py"}, "total_score": 1.5}],
        ],
    )
    pipeline.query_processor.process.return_value = processed_query

    result = pipeline.query(
        "Where is the config loaded?",
        filters={"snapshot_id": "snap:1"},
    )

    assert pipeline.retriever.retrieve.call_count == 2
    assert result["semantic_escalation"]["budget"] == "local"


@pytest.mark.regression
def test_escalate_query_semantics_returns_skipped_when_snapshot_not_found() -> None:
    """_escalate_query_semantics must return early when the snapshot cannot
    be loaded.
    """
    fc = FastCode.__new__(FastCode)
    fc.snapshot_store = SimpleNamespace(load_snapshot=lambda snapshot_id: None)

    result = fc._escalate_query_semantics(
        snapshot_id="snap:missing",
        retrieved=[{"element": {"relative_path": "a.py"}}],
        processed_query=_processed_query(question="test", filters={}),
        budget="path-critical",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "snapshot_not_found"
    assert result["rerun_retrieval"] is False


@pytest.mark.regression
def test_escalate_query_semantics_returns_skipped_when_no_target_paths() -> None:
    """_escalate_query_semantics must return early when no target paths can
    be extracted from retrieved elements or filters.
    """
    snapshot = IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:1",
        metadata={},
    )
    fc = FastCode.__new__(FastCode)
    fc.snapshot_store = SimpleNamespace(load_snapshot=lambda snapshot_id: snapshot)

    result = fc._escalate_query_semantics(
        snapshot_id="snap:1",
        retrieved=[],
        processed_query=_processed_query(question="test", filters={}),
        budget="path-critical",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "no_target_paths"
    assert result["rerun_retrieval"] is False


@pytest.mark.regression
def test_escalate_query_semantics_returns_degraded_when_warnings_present() -> None:
    """_escalate_query_semantics must return status='degraded' when resolvers
    produce warnings.
    """
    snapshot = IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:1",
        metadata={"semantic_resolver_runs": [{"language": "python"}]},
    )
    upgraded_snapshot = IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:1",
        metadata={"semantic_resolver_runs": [{"language": "python"}]},
    )
    element = _element("src/a.py")
    fc = FastCode.__new__(FastCode)
    fc.snapshot_store = SimpleNamespace(load_snapshot=lambda sid: snapshot)
    fc.graph_builder = SimpleNamespace(element_by_id={element.id: element})
    fc.ir_graph_builder = SimpleNamespace(build_graphs=lambda snap: "ir-graphs")
    fc.retriever = SimpleNamespace(set_ir_graphs=MagicMock())
    fc.snapshot_symbol_index = SimpleNamespace(register_snapshot=MagicMock())

    def apply_with_warnings(
        *,
        snapshot: object,
        elements: object,
        legacy_graph_builder: object,
        target_paths: object,
        warnings: list[str],
        budget: str,
    ) -> IRSnapshot:
        warnings.append("resolver produced a partial result")
        return upgraded_snapshot

    fc._apply_semantic_resolvers = apply_with_warnings

    result = fc._escalate_query_semantics(
        snapshot_id="snap:1",
        retrieved=[{"element": {"relative_path": "src/a.py"}}],
        processed_query=_processed_query(question="test", filters={}),
        budget="path-critical",
    )

    assert result["status"] == "degraded"
    assert result["warnings"] == ["resolver produced a partial result"]
    assert result["rerun_retrieval"] is True


@pytest.mark.regression
def test_escalation_does_not_rerun_when_callback_returns_none() -> None:
    """If the escalation callback returns None, no second retrieval should
    occur and no escalation metadata should appear in the result.
    """
    processed_query = _processed_query(
        question="How does auth reach the token store?",
        filters={"snapshot_id": "snap:1"},
        intent="debug",
    )
    pipeline = _query_pipeline(
        semantic_escalation_cb=lambda **kwargs: None,
        retrieve_side_effect=[
            [{"element": {"relative_path": "a.py"}, "total_score": 1.0}],
        ],
    )
    pipeline.query_processor.process.return_value = processed_query

    result = pipeline.query(
        "How does auth reach the token store?",
        filters={"snapshot_id": "snap:1"},
    )

    assert pipeline.retriever.retrieve.call_count == 1
    assert "semantic_escalation" not in result
