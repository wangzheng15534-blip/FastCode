from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

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
