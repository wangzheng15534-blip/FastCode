from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from fastcode.indexing import pipeline as pipeline_module
from fastcode.ir.element import CodeElement
from fastcode.ir.types import IRSnapshot
from fastcode.main.fastcode import FastCode, _ReadWriteStateLock
from fastcode.query.processor import ProcessedQuery


def _processed_query(
    *,
    question: str = "How does auth reach the token store?",
    filters: dict[str, object] | None = None,
) -> ProcessedQuery:
    return ProcessedQuery(
        original=question,
        expanded=question,
        keywords=[],
        intent="debug",
        subqueries=[],
        filters=dict(filters or {}),
        rewritten_query=None,
        pseudocode_hints=None,
        search_strategy=None,
    )


def _element(path: str) -> CodeElement:
    return CodeElement(
        id="file:a",
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


def test_service_state_lock_emits_structured_acquire_release_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    lock = _ReadWriteStateLock()

    with caplog.at_level(logging.DEBUG, logger="fastcode.main.fastcode.state_lock"):
        with lock.write_lock():
            pass
        with lock.read_lock():
            pass

    events = [
        (record.fc_event, record.lock_mode)
        for record in caplog.records
        if hasattr(record, "fc_event")
    ]
    assert ("service_lock_acquire", "write") in events
    assert ("service_lock_acquired", "write") in events
    assert ("service_lock_released", "write") in events
    assert ("service_lock_acquire", "read") in events
    assert ("service_lock_acquired", "read") in events
    assert ("service_lock_released", "read") in events


def test_snapshot_artifact_handle_emits_structured_swap_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeVectorStore:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config
            self.metadata: list[dict[str, Any]] = []

        def load(self, artifact_key: str) -> bool:
            return artifact_key == "art:cache"

    class FakeGraphBuilder:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

        def load(self, artifact_key: str) -> bool:
            return artifact_key == "art:cache"

    class FakeRetriever:
        def __init__(
            self,
            config: dict[str, Any],
            vector_store: Any,
            embedder: Any,
            graph_builder: Any,
            repo_root: str | None = None,
        ) -> None:
            self.config = config
            self.vector_store = vector_store
            self.graph_builder = graph_builder
            self.repo_root = repo_root

        def set_pg_retrieval_store(self, store: Any) -> None:
            self.store = store

        def load_bm25(self, artifact_key: str) -> bool:
            return artifact_key == "art:cache"

        def set_ir_graph_loader(
            self,
            graph_loader: Any,
            *,
            snapshot_id: str | None,
        ) -> None:
            self.ir_graph_loader = graph_loader
            self.snapshot_id = snapshot_id

        def build_repo_overview_bm25(self) -> None:
            return None

    monkeypatch.setattr(pipeline_module, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(pipeline_module, "CodeGraphBuilder", FakeGraphBuilder)
    monkeypatch.setattr(pipeline_module, "HybridRetriever", FakeRetriever)

    pipeline = pipeline_module.IndexPipeline.__new__(pipeline_module.IndexPipeline)
    pipeline.config = {"query": {"snapshot_handle_cache_size": 4}}
    pipeline.embedder = object()
    pipeline.loader = SimpleNamespace(repo_path="/tmp/repo")
    pipeline.snapshot_store = SimpleNamespace(load_ir_graphs=lambda _snapshot_id: None)
    pipeline.pg_retrieval_store = None
    pipeline._artifact_lock = threading.RLock()
    pipeline._artifact_handle_cache = OrderedDict()

    logger = logging.getLogger("fastcode.test.artifacts")
    pipeline.logger = logger
    with caplog.at_level(logging.INFO, logger="fastcode.test.artifacts"):
        handle = pipeline.load_snapshot_artifacts_handle(
            "art:cache",
            snapshot_id="snap:1",
        )

    assert handle is not None
    record = next(
        record
        for record in caplog.records
        if getattr(record, "fc_event", None) == "artifact_handle_swap"
    )
    assert record.artifact_key == "art:cache"
    assert record.snapshot_id == "snap:1"
    assert record.artifact_handle_kind == "snapshot"
    assert record.artifact_loaded is True


def test_query_semantic_escalation_emits_structured_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
    fc.logger = logging.getLogger("fastcode.test.semantic_escalation")
    fc.snapshot_store = SimpleNamespace(load_snapshot=lambda _snapshot_id: snapshot)
    fc.graph_builder = SimpleNamespace(element_by_id={element.id: element})
    fc.ir_graph_builder = SimpleNamespace(build_graphs=lambda _snapshot: "ir-graphs")
    fc.retriever = SimpleNamespace(set_ir_graphs=MagicMock())
    fc.snapshot_symbol_index = SimpleNamespace(register_snapshot=MagicMock())
    fc._apply_semantic_resolvers = MagicMock(return_value=upgraded_snapshot)

    with caplog.at_level(
        logging.INFO,
        logger="fastcode.test.semantic_escalation",
    ):
        result = fc._escalate_query_semantics(
            snapshot_id="snap:1",
            retrieved=[{"element": {"relative_path": "src/a.py"}}],
            processed_query=_processed_query(filters={"file_path": "src/a.py"}),
            budget="path-critical",
        )

    assert result["status"] == "applied"
    record = next(
        record
        for record in caplog.records
        if getattr(record, "fc_event", None) == "semantic_escalation"
    )
    assert record.snapshot_id == "snap:1"
    assert record.semantic_budget == "path-critical"
    assert record.semantic_status == "applied"
    assert record.target_path_count == 1
    assert record.resolver_runs == 1
    assert record.warning_count == 0
