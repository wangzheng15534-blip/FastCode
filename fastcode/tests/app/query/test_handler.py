from __future__ import annotations

import json
from collections.abc import Callable, Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import networkx as nx
import pytest

from fastcode.app.query.context_payloads import distillation_payload
from fastcode.app.query.orchestration.handler import QueryPipeline
from fastcode.app.query.orchestration.processor import ProcessedQuery
from fastcode.app.query.selection.retriever import HybridRetriever
from fastcode.app.store.cache.contracts import ContextDistillationRecord
from fastcode.ir.element import CodeElement
from fastcode.ir.graph import IRGraphs
from fastcode.ir.types import IRSnapshot
from fastcode.main.fastcode import FastCode
from fastcode.retrieval.context.agent_context import (
    AcceptanceContract,
    AcceptedFact,
    ActivationRecord,
    ContextBundle,
    DistillationRecord,
    EvidenceRef,
    Hypothesis,
    RejectedHypothesisEntry,
    RiskState,
    ToolObservation,
    TurnIntent,
    TurnJournal,
    TurnPlan,
    WorkingMemoryArtifact,
    WorkingSet,
)
from fastcode.semantic.symbol_index import SnapshotSymbolIndex

AGENT_CONTEXT_RECORD_CLASSES = (
    EvidenceRef,
    ToolObservation,
    Hypothesis,
    RejectedHypothesisEntry,
    AcceptedFact,
    RiskState,
    AcceptanceContract,
    TurnIntent,
    TurnPlan,
    WorkingSet,
    WorkingMemoryArtifact,
    TurnJournal,
    DistillationRecord,
    ActivationRecord,
    ContextBundle,
)


def _assert_agent_context_has_no_compat_serializers() -> None:
    for cls in AGENT_CONTEXT_RECORD_CLASSES:
        assert not hasattr(cls, "to_dict")
        assert not hasattr(cls, "from_dict")


@contextmanager
def _null_context() -> Generator[None, None, None]:
    yield


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
    if retrieve_side_effect is None:
        retriever.retrieve.return_value = [{"element": {"relative_path": "a.py"}}]
    else:
        retriever.retrieve.side_effect = retrieve_side_effect
    answer_generator = MagicMock()
    answer_generator.generate.return_value = {"answer": "ok", "sources": []}
    query_processor = MagicMock()
    cache_manager = MagicMock()
    cache_manager.get_recent_summaries.return_value = []
    cache_manager.get_dialogue_history.return_value = []
    cache_manager.get_session_index_record.return_value = None
    cache_manager._get_session_index.return_value = None
    cache_manager.get_latest_working_memory_record.return_value = None
    cache_manager.find_reusable_context_distillation_record.return_value = None
    cache_manager.save_working_memory_record.return_value = True
    cache_manager.save_turn_journal_record.return_value = True
    cache_manager.save_context_bundle_record.return_value = True
    cache_manager.save_context_distillation_record.return_value = True
    cache_manager.save_context_activation_record.return_value = True
    cache_manager.save_dialogue_turn.return_value = True
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


def test_agency_mode_uses_detected_intent_for_local_escalation() -> None:
    processed_intents: list[str] = []

    pipeline = _query_pipeline(
        semantic_escalation_cb=lambda **kwargs: {
            "status": "applied",
            "budget": kwargs["budget"],
            "rerun_retrieval": True,
        },
        retrieve_side_effect=[
            [{"element": {"relative_path": "src/config.py"}, "total_score": 1.0}],
            [{"element": {"relative_path": "src/config.py"}, "total_score": 1.5}],
        ],
    )
    pipeline.retriever.enable_agency_mode = True
    pipeline.retriever.iterative_agent = object()
    pipeline.query_processor._detect_intent.side_effect = lambda question: "where"
    retrievals = iter(
        [
            [{"element": {"relative_path": "src/config.py"}, "total_score": 1.0}],
            [{"element": {"relative_path": "src/config.py"}, "total_score": 1.5}],
        ]
    )

    def retrieve_with_intent_capture(
        processed_query: ProcessedQuery, *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        processed_intents.append(processed_query.intent)
        return next(retrievals)

    pipeline.retriever.retrieve.side_effect = retrieve_with_intent_capture

    result = pipeline.query(
        "Where is the config loaded?",
        filters={"snapshot_id": "snap:1"},
    )

    assert processed_intents == ["where", "where"]
    assert pipeline.retriever.retrieve.call_count == 2
    assert result["semantic_escalation"]["budget"] == "local"


def test_streaming_agency_mode_uses_detected_intent() -> None:
    processed_intents: list[str] = []
    pipeline = _query_pipeline()
    pipeline.retriever.enable_agency_mode = True
    pipeline.retriever.iterative_agent = object()
    pipeline.query_processor._detect_intent.side_effect = lambda question: "where"
    pipeline.answer_generator.generate_stream.return_value = [
        ("ok", None),
        (None, {"sources": []}),
    ]

    def retrieve_with_intent_capture(
        processed_query: ProcessedQuery, *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        processed_intents.append(processed_query.intent)
        return [{"element": {"relative_path": "src/config.py"}, "total_score": 1.0}]

    pipeline.retriever.retrieve.side_effect = retrieve_with_intent_capture

    chunks = list(
        pipeline.query_stream(
            "Where is the config loaded?",
            filters={"snapshot_id": "snap:1"},
        )
    )

    assert processed_intents == ["where"]
    assert chunks[-1][1]["status"] == "complete"


def test_query_uses_explicit_processed_query_payload() -> None:
    processed_query = _processed_query(
        question="Where is auth handled?",
        intent="where",
        filters={"snapshot_id": "snap:1"},
    )
    pipeline = _query_pipeline()
    pipeline.query_processor.process.return_value = processed_query

    with pytest.MonkeyPatch.context() as monkeypatch:

        def _boom_to_dict(self: ProcessedQuery) -> dict[str, Any]:
            msg = "query pipeline must not call ProcessedQuery.to_dict()"
            raise AssertionError(msg)

        monkeypatch.setattr(ProcessedQuery, "to_dict", _boom_to_dict)
        result = pipeline.query(
            "Where is auth handled?",
            filters={"snapshot_id": "snap:1"},
        )

    assert result["answer"] == "ok"
    query_info = pipeline.answer_generator.generate.call_args.kwargs["query_info"]
    assert query_info == {
        "original": "Where is auth handled?",
        "expanded": "Where is auth handled?",
        "keywords": [],
        "intent": "where",
        "subqueries": [],
        "filters": {"snapshot_id": "snap:1"},
        "rewritten_query": None,
        "pseudocode_hints": None,
        "search_strategy": None,
    }


def test_query_stream_uses_explicit_processed_query_payload() -> None:
    processed_query = _processed_query(
        question="Where is auth handled?",
        intent="where",
        filters={"snapshot_id": "snap:1"},
    )
    pipeline = _query_pipeline()
    pipeline.query_processor.process.return_value = processed_query
    pipeline.answer_generator.generate_stream.return_value = [
        (None, {"sources": []}),
        ("ok", None),
        (None, {"complete": True}),
    ]

    with pytest.MonkeyPatch.context() as monkeypatch:

        def _boom_to_dict(self: ProcessedQuery) -> dict[str, Any]:
            msg = "query stream must not call ProcessedQuery.to_dict()"
            raise AssertionError(msg)

        monkeypatch.setattr(ProcessedQuery, "to_dict", _boom_to_dict)
        chunks = list(
            pipeline.query_stream(
                "Where is auth handled?",
                filters={"snapshot_id": "snap:1"},
            )
        )

    assert chunks[-1][1]["status"] == "complete"
    query_info = pipeline.answer_generator.generate_stream.call_args.kwargs[
        "query_info"
    ]
    assert query_info["intent"] == "where"
    assert query_info["filters"] == {"snapshot_id": "snap:1"}


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


def test_query_snapshot_uses_loaded_artifact_handle() -> None:
    processed_query = _processed_query(
        question="Where is auth?",
        filters={"snapshot_id": "snap:1"},
    )
    pipeline = _query_pipeline()
    pipeline.load_artifacts_by_key = MagicMock(
        side_effect=AssertionError("legacy artifact loader should not run")
    )
    pipeline.snapshot_store.get_snapshot_record.return_value = SimpleNamespace(
        artifact_key="art_snap_1"
    )
    pipeline.snapshot_symbol_index.has_snapshot.return_value = True
    handle_retriever = MagicMock()
    handle_retriever.enable_agency_mode = False
    handle_retriever.iterative_agent = None
    handle_retriever.retrieve.return_value = [
        {"element": {"relative_path": "src/auth.py"}, "total_score": 1.0}
    ]
    pipeline.load_snapshot_artifacts = MagicMock(
        return_value=SimpleNamespace(
            artifact_key="art_snap_1",
            retriever=handle_retriever,
            graph_builder=MagicMock(),
        )
    )
    pipeline.query_processor.process.return_value = processed_query

    result = pipeline.query_snapshot("Where is auth?", snapshot_id="snap:1")

    pipeline.load_snapshot_artifacts.assert_called_once_with(
        "art_snap_1",
        snapshot_id="snap:1",
    )
    handle_retriever.retrieve.assert_called_once()
    assert result["artifact_key"] == "art_snap_1"
    assert result["snapshot_id"] == "snap:1"


def test_query_snapshot_registers_compact_symbol_index_without_full_snapshot_load() -> (
    None
):
    processed_query = _processed_query(
        question="Where is auth?",
        filters={"snapshot_id": "snap:1"},
    )
    pipeline = _query_pipeline()
    pipeline.snapshot_store.get_snapshot_record.return_value = SimpleNamespace(
        artifact_key="art_snap_1"
    )
    pipeline.snapshot_store.load_snapshot_symbol_index_payload.return_value = {
        "schema_version": "snapshot_symbol_index.v1",
        "snapshot_id": "snap:1",
        "symbols": [
            {
                "canonical": "sym:auth",
                "aliases": ["scip:auth"],
                "names": ["AuthService"],
                "path": "src/auth.py",
            }
        ],
    }
    pipeline.snapshot_store.load_snapshot.side_effect = AssertionError(
        "query_snapshot should not full-load IRSnapshot when symbol sidecar exists"
    )
    pipeline.snapshot_symbol_index = SnapshotSymbolIndex()
    handle_retriever = MagicMock()
    handle_retriever.enable_agency_mode = False
    handle_retriever.iterative_agent = None
    handle_retriever.retrieve.return_value = [
        {"element": {"relative_path": "src/auth.py"}, "total_score": 1.0}
    ]
    pipeline.load_snapshot_artifacts = MagicMock(
        return_value=SimpleNamespace(
            artifact_key="art_snap_1",
            retriever=handle_retriever,
            graph_builder=MagicMock(),
        )
    )
    pipeline.query_processor.process.return_value = processed_query

    result = pipeline.query_snapshot("Where is auth?", snapshot_id="snap:1")

    assert result["snapshot_id"] == "snap:1"
    assert (
        pipeline.snapshot_symbol_index.resolve_symbol("snap:1", name="AuthService")
        == "sym:auth"
    )
    pipeline.snapshot_store.load_snapshot.assert_not_called()


def test_query_pipeline_skips_semantic_escalation_without_snapshot_scope() -> None:
    callback = MagicMock(return_value={"rerun_retrieval": True})
    processed_query = _processed_query(question="What is auth?", intent="what")
    pipeline = _query_pipeline(semantic_escalation_cb=callback)
    pipeline.query_processor.process.return_value = processed_query

    result = pipeline.query("What is auth?")

    callback.assert_not_called()
    assert pipeline.retriever.retrieve.call_count == 1
    assert "semantic_escalation" not in result


def test_query_pipeline_compiles_context_and_persists_typed_records() -> None:
    _assert_agent_context_has_no_compat_serializers()
    processed_query = _processed_query(
        question="Where is auth handled?",
        filters={"snapshot_id": "snap:1", "artifact_key": "art:1"},
    )
    pipeline = _query_pipeline()
    pipeline.query_processor.process.return_value = processed_query
    pipeline.get_session_prefix = MagicMock(
        return_value={
            "snapshot_id": "snap:1",
            "projection_id": "proj:1",
            "l0": {"summary": "Repository overview"},
            "l1": {"summary": "Package map"},
        }
    )
    pipeline.cache_manager.get_latest_working_memory_record.return_value = (
        SimpleNamespace(full_fcx="<fcx:stable>\nprior\n</fcx:stable>")
    )
    pipeline.retriever.retrieve.return_value = [
        {
            "element": {
                "relative_path": "src/auth.py",
                "repo_name": "repo",
                "type": "file",
                "name": "auth.py",
                "start_line": 10,
                "end_line": 80,
            },
            "total_score": 1.4,
        }
    ]
    pipeline.answer_generator.generate.return_value = {
        "answer": "Auth is handled in src/auth.py",
        "sources": [
            {
                "file": "src/auth.py",
                "repo": "repo",
                "type": "file",
                "name": "auth.py",
                "start_line": 10,
                "end_line": 80,
            }
        ],
    }

    result = pipeline.query(
        "Where is auth handled?",
        filters={"snapshot_id": "snap:1", "artifact_key": "art:1"},
        session_id="sess-1",
    )

    retrieve_kwargs = pipeline.retriever.retrieve.call_args.kwargs
    generate_kwargs = pipeline.answer_generator.generate.call_args.kwargs

    assert retrieve_kwargs["compiled_context"] == "<fcx:stable>\nprior\n</fcx:stable>"
    assert "<fcx:stable>" in generate_kwargs["compiled_context"]
    assert "<fcx:turn>" in generate_kwargs["compiled_context"]
    assert pipeline.cache_manager.save_working_memory_record.call_count == 1
    assert pipeline.cache_manager.save_turn_journal_record.call_count == 1
    assert pipeline.cache_manager.save_context_bundle_record.call_count == 1
    assert pipeline.cache_manager.save_context_distillation_record.call_count == 1
    assert pipeline.cache_manager.save_context_activation_record.call_count == 1
    bundle_record = pipeline.cache_manager.save_context_bundle_record.call_args.args[0]
    distillation_record = (
        pipeline.cache_manager.save_context_distillation_record.call_args.args[0]
    )
    activation_record = (
        pipeline.cache_manager.save_context_activation_record.call_args.args[0]
    )
    bundle_payload = json.loads(bundle_record.payload_json)
    distillation_payload = json.loads(distillation_record.payload_json)
    activation_payload = json.loads(activation_record.payload_json)
    assert bundle_record.invalidation_key.startswith("ctxinv_")
    assert bundle_payload["distillation"]["source_refs"][0]["path"] == "src/auth.py"
    assert distillation_record.source_ref_ids == ("e1",)
    assert distillation_payload["source_refs"][0]["lines"] == "10-80"
    assert activation_payload["active_ref_ids"] == ["e1"]
    assert activation_record.bundle_id == bundle_record.bundle_id
    assert pipeline.cache_manager.save_dialogue_turn.call_args.kwargs["summary"]
    assert result["turn_number"] == 1


def test_query_pipeline_reuses_context_distillation_when_sources_match() -> None:
    _assert_agent_context_has_no_compat_serializers()
    processed_query = _processed_query(
        question="Where is auth handled?",
        filters={"snapshot_id": "snap:1", "artifact_key": "art:1"},
    )
    pipeline = _query_pipeline()
    pipeline.query_processor.process.return_value = processed_query
    pipeline.retriever.retrieve.return_value = [
        {
            "element": {
                "relative_path": "src/auth.py",
                "repo_name": "repo",
                "type": "file",
                "name": "auth.py",
                "start_line": 10,
                "end_line": 80,
            },
            "total_score": 1.4,
        }
    ]
    prior_distillation = DistillationRecord(
        distillation_id="dist_previous",
        session_id="sess-1",
        turn_number=1,
        snapshot_id="snap:1",
        compiler_fingerprint="fcx-v1",
        summary="Prior auth distillation",
        source_refs=(
            EvidenceRef(
                ref_id="e1",
                kind="range",
                repo_name="repo",
                snapshot_id="snap:1",
                path="src/auth.py",
                lines="10-80",
                label="auth.py",
                source="retrieval",
                fresh="ok",
            ),
        ),
        accepted_facts=(),
        reused_from_distillation_id=None,
        invalidation_key="placeholder",
        created_at=100.0,
    )

    def _find_reusable(
        session_id: str,
        *,
        invalidation_key: str,
        compiler_fingerprint: str,
    ) -> ContextDistillationRecord:
        reused = DistillationRecord(
            distillation_id=prior_distillation.distillation_id,
            session_id=prior_distillation.session_id,
            turn_number=prior_distillation.turn_number,
            snapshot_id=prior_distillation.snapshot_id,
            compiler_fingerprint=compiler_fingerprint,
            summary=prior_distillation.summary,
            source_refs=prior_distillation.source_refs,
            accepted_facts=prior_distillation.accepted_facts,
            reused_from_distillation_id=None,
            invalidation_key=invalidation_key,
            created_at=prior_distillation.created_at,
        )
        return ContextDistillationRecord(
            distillation_id=reused.distillation_id,
            session_id=session_id,
            turn_number=1,
            snapshot_id=reused.snapshot_id,
            compiler_fingerprint=reused.compiler_fingerprint,
            summary=reused.summary,
            payload_json=json.dumps(
                distillation_payload(reused), separators=(",", ":"), sort_keys=True
            ),
            invalidation_key=invalidation_key,
            source_ref_ids=("e1",),
            reused_from_distillation_id=None,
            created_at=reused.created_at,
        )

    pipeline.cache_manager.find_reusable_context_distillation_record.side_effect = (
        _find_reusable
    )
    pipeline.answer_generator.generate.return_value = {
        "answer": "Auth is handled in src/auth.py",
        "sources": [
            {
                "file": "src/auth.py",
                "repo": "repo",
                "type": "file",
                "name": "auth.py",
                "start_line": 10,
                "end_line": 80,
            }
        ],
    }

    pipeline.query(
        "Where is auth handled?",
        filters={"snapshot_id": "snap:1", "artifact_key": "art:1"},
        session_id="sess-1",
    )

    saved = pipeline.cache_manager.save_context_distillation_record.call_args.args[0]
    payload = json.loads(saved.payload_json)
    assert saved.distillation_id == "dist_previous"
    assert saved.reused_from_distillation_id == "dist_previous"
    assert saved.source_ref_ids == ("e1",)
    assert payload["source_refs"][0]["path"] == "src/auth.py"


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
    fc.query_handler = SimpleNamespace(retriever=fc.retriever)
    fc.pipeline = SimpleNamespace(
        _apply_semantic_resolvers=MagicMock(return_value=upgraded_snapshot)
    )
    from fastcode.app.query.facade import QueryFacade

    fc.query = QueryFacade(
        query_handler=fc.query_handler,
        vector_store=SimpleNamespace(),
        graph_builder=fc.graph_builder,
        snapshot_store=fc.snapshot_store,
        ir_graph_builder=fc.ir_graph_builder,
        snapshot_symbol_index=fc.snapshot_symbol_index,
        pipeline=fc.pipeline,
        state=SimpleNamespace(read_lock=_null_context),
    )

    result = fc.query._escalate_query_semantics(
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
def test_semantic_escalation_enables_real_ir_graph_expansion() -> None:
    auth = _element("src/auth.py", element_id="file:auth")
    token_store = _element("src/token_store.py", element_id="file:token_store")
    auth.metadata["ir_symbol_id"] = "unit:auth"
    token_store.metadata["ir_symbol_id"] = "unit:token_store"

    real_retriever = HybridRetriever.__new__(HybridRetriever)
    real_retriever.graph_expansion_backend = "ir"
    real_retriever.ir_graphs = None
    real_retriever.ir_snapshot_id = None
    real_retriever.allow_graph_builder_fallback = False
    real_retriever.graph_weight = 1.0
    real_retriever.logger = MagicMock()
    real_retriever.graph_builder = SimpleNamespace(
        element_by_id={auth.id: auth, token_store.id: token_store},
        get_related_elements=lambda *args, **kwargs: set(),
    )

    def retrieve_from_real_graph(
        processed_query: ProcessedQuery, *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        seed_results = [
            {
                "element": auth.to_dict(),
                "semantic_score": 1.0,
                "keyword_score": 0.0,
                "graph_score": 0.0,
                "total_score": 1.0,
            }
        ]
        return real_retriever._expand_with_graph(seed_results, max_hops=1)

    def install_call_graph(**kwargs: Any) -> dict[str, Any]:
        call_graph: nx.DiGraph[str] = nx.DiGraph()
        call_graph.add_edge("unit:auth", "unit:token_store")
        empty: nx.DiGraph[str] = nx.DiGraph()
        real_retriever.set_ir_graphs(
            IRGraphs(
                dependency_graph=empty.copy(),
                call_graph=call_graph,
                inheritance_graph=empty.copy(),
                reference_graph=empty.copy(),
                containment_graph=empty.copy(),
            ),
            snapshot_id=kwargs["snapshot_id"],
        )
        return {"status": "applied", "budget": "path-critical", "rerun_retrieval": True}

    pipeline = _query_pipeline(semantic_escalation_cb=install_call_graph)
    pipeline.retriever.retrieve.side_effect = retrieve_from_real_graph
    pipeline.query_processor.process.return_value = _processed_query(
        question="How does auth reach the token store?",
        filters={"snapshot_id": "snap:1"},
    )

    result = pipeline.query(
        "How does auth reach the token store?",
        filters={"snapshot_id": "snap:1"},
    )

    generated_retrieved = pipeline.answer_generator.generate.call_args[0][1]
    paths = [row["element"]["relative_path"] for row in generated_retrieved]
    assert result["semantic_escalation"]["status"] == "applied"
    assert paths == ["src/auth.py", "src/token_store.py"]


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
    fc.query_handler = SimpleNamespace(retriever=SimpleNamespace())
    fc.pipeline = SimpleNamespace()
    from fastcode.app.query.facade import QueryFacade

    fc.query = QueryFacade(
        query_handler=fc.query_handler,
        vector_store=SimpleNamespace(),
        graph_builder=SimpleNamespace(element_by_id={}),
        snapshot_store=fc.snapshot_store,
        ir_graph_builder=SimpleNamespace(),
        snapshot_symbol_index=SimpleNamespace(),
        pipeline=fc.pipeline,
        state=SimpleNamespace(read_lock=_null_context),
    )

    result = fc.query._escalate_query_semantics(
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
    fc.query_handler = SimpleNamespace(retriever=SimpleNamespace())
    fc.pipeline = SimpleNamespace()
    from fastcode.app.query.facade import QueryFacade

    fc.query = QueryFacade(
        query_handler=fc.query_handler,
        vector_store=SimpleNamespace(),
        graph_builder=SimpleNamespace(element_by_id={}),
        snapshot_store=fc.snapshot_store,
        ir_graph_builder=SimpleNamespace(),
        snapshot_symbol_index=SimpleNamespace(),
        pipeline=fc.pipeline,
        state=SimpleNamespace(read_lock=_null_context),
    )

    result = fc.query._escalate_query_semantics(
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
    fc.query_handler = SimpleNamespace(retriever=fc.retriever)

    def apply_with_warnings(
        *,
        snapshot: object,
        elements: object,
        graph_context: object,
        target_paths: object,
        warnings: list[str],
        budget: str,
    ) -> IRSnapshot:
        warnings.append("resolver produced a partial result")
        return upgraded_snapshot

    fc.pipeline = SimpleNamespace(_apply_semantic_resolvers=apply_with_warnings)
    from fastcode.app.query.facade import QueryFacade

    fc.query = QueryFacade(
        query_handler=fc.query_handler,
        vector_store=SimpleNamespace(),
        graph_builder=fc.graph_builder,
        snapshot_store=fc.snapshot_store,
        ir_graph_builder=fc.ir_graph_builder,
        snapshot_symbol_index=fc.snapshot_symbol_index,
        pipeline=fc.pipeline,
        state=SimpleNamespace(read_lock=_null_context),
    )

    result = fc.query._escalate_query_semantics(
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


# ---------------------------------------------------------------------------
# Regression: concurrent query_snapshot must serialize artifact loading
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_concurrent_query_snapshot_sequential_artifact_loading() -> None:
    """Two concurrent _load_artifacts_by_key calls on the same IndexPipeline
    must be serialized.  Without a lock, concurrent loads can overwrite
    shared state on vector_store, retriever, and graph_builder.
    """
    import threading
    import time

    from fastcode.app.indexing.pipeline.service import IndexPipeline

    concurrent_count = 0
    max_concurrent = 0
    count_lock = threading.Lock()
    load_calls: list[str] = []
    start_barrier = threading.Barrier(2, timeout=5)

    slow_vector_store = MagicMock()

    def _slow_load(artifact_key: str) -> bool:
        nonlocal concurrent_count, max_concurrent
        with count_lock:
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            load_calls.append(artifact_key)
        # Hold the critical section long enough to detect overlap.
        time.sleep(0.05)
        with count_lock:
            concurrent_count -= 1
        return True

    slow_vector_store.load = _slow_load

    pipeline = IndexPipeline.__new__(IndexPipeline)
    pipeline._artifact_lock = threading.RLock()
    pipeline.vector_store = slow_vector_store
    pipeline.retriever = MagicMock()
    pipeline.graph_builder = MagicMock()
    pipeline.graph_artifact_store = SimpleNamespace(
        load=lambda _builder, _artifact_key: True
    )
    pipeline.snapshot_store = MagicMock()
    pipeline.snapshot_store.find_by_artifact_key.return_value = None
    pipeline._set_repo_indexed = MagicMock()
    pipeline._set_repo_loaded = MagicMock()

    errors: list[Exception] = []

    def _load(key: str) -> None:
        # Ensure both threads are ready to call _load_artifacts_by_key
        # at roughly the same time.
        try:
            start_barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            return
        try:
            pipeline._load_artifacts_by_key(key)
        except Exception as exc:
            errors.append(exc)

    t_a = threading.Thread(target=_load, args=("key_alpha",))
    t_b = threading.Thread(target=_load, args=("key_beta",))
    t_a.start()
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    assert not errors, f"Threads raised: {errors}"
    assert max_concurrent <= 1, (
        f"Artifact loading was not serialized (max concurrent: {max_concurrent})"
    )
    assert set(load_calls) == {"key_alpha", "key_beta"}
