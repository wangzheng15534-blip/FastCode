"""Tests for main module."""

from __future__ import annotations

import pickle
from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from fastcode.ir.element import CodeElement
from fastcode.main import FastCode
from fastcode.schemas.config import config_from_mapping

# ---------------------------------------------------------------------------
# Helpers (basic / doc pipeline tests)
# ---------------------------------------------------------------------------


def _make_fastcode(
    *,
    doc_ingester_enabled: bool = True,
    storage_backend: str = "sqlite",
    graph_enabled: bool = False,
    sync_result: bool = True,
) -> Any:
    fc = FastCode.__new__(FastCode)
    fc.doc_ingester = SimpleNamespace(enabled=doc_ingester_enabled)
    fc.snapshot_store = SimpleNamespace(
        db_runtime=SimpleNamespace(backend=storage_backend)
    )
    fc.graph_runtime = SimpleNamespace(
        enabled=graph_enabled, sync_docs=lambda **_: sync_result
    )
    return fc


# --- Doc pipeline tests ---


def test_should_ingest_docs_requires_active_sink():
    fc = _make_fastcode(
        doc_ingester_enabled=True, storage_backend="sqlite", graph_enabled=False
    )
    assert fc._should_ingest_docs() is False

    fc = _make_fastcode(
        doc_ingester_enabled=True, storage_backend="postgres", graph_enabled=False
    )
    assert fc._should_ingest_docs() is True

    fc = _make_fastcode(
        doc_ingester_enabled=True, storage_backend="sqlite", graph_enabled=True
    )
    assert fc._should_ingest_docs() is True


def test_should_ingest_docs_requires_feature_flag():
    fc = _make_fastcode(
        doc_ingester_enabled=False, storage_backend="postgres", graph_enabled=True
    )
    assert fc._should_ingest_docs() is False


def test_sync_doc_overlay_records_false_return_as_warning():
    fc = _make_fastcode(graph_enabled=True, sync_result=False)
    warnings = []

    fc._sync_doc_overlay(chunks=[{"chunk_id": "c1"}], mentions=[], warnings=warnings)

    assert warnings == ["ladybug_doc_sync_failed"]


def test_sync_doc_overlay_records_exceptions_as_warning():
    fc = FastCode.__new__(FastCode)
    fc.graph_runtime = SimpleNamespace(
        enabled=True,
        sync_docs=lambda **_: (_ for _ in ()).throw(RuntimeError("db offline")),
    )
    warnings = []

    fc._sync_doc_overlay(chunks=[{"chunk_id": "c1"}], mentions=[], warnings=warnings)

    assert warnings == ["ladybug_doc_sync_failed: db offline"]


def test_apply_repository_runtime_overrides_refreshes_loader_and_runtime_config():
    fc = FastCode.__new__(FastCode)
    fc.runtime_config = config_from_mapping(
        {
            "repository": {
                "ignore_patterns": ["base"],
                "exclude_site_packages": False,
            }
        }
    )
    fc.config = fc.runtime_config.to_dict()
    fc.eval_config = fc.config.get("evaluation", {})
    fc.eval_mode = False
    fc.in_memory_index = False
    fc.loader = SimpleNamespace(
        repo_config=fc.config["repository"],
        ignore_patterns=fc.config["repository"]["ignore_patterns"],
    )

    fc.apply_repository_runtime_overrides(
        ignore_patterns=("base", ".venv"),
        exclude_site_packages=True,
    )

    assert is_dataclass(fc.runtime_config)
    assert fc.runtime_config.repository.ignore_patterns == ("base", ".venv")
    assert fc.runtime_config.repository.exclude_site_packages is True
    assert fc.loader.repo_config["ignore_patterns"] == ("base", ".venv")
    assert fc.loader.repo_config["exclude_site_packages"] is True
    assert fc.loader.ignore_patterns == ("base", ".venv")


def test_process_semantic_repair_frontier_replays_pipeline_with_payload():
    fc = FastCode.__new__(FastCode)
    fc.pipeline = SimpleNamespace(
        run_semantic_repair_frontier=lambda **kwargs: {
            "status": "repaired",
            "kwargs": kwargs,
            "repair_frontier": {
                "scope_kind": kwargs["scope_kind"],
                "scope_roots": kwargs["scope_roots"],
                "changed_paths": kwargs["changed_paths"],
                "target_paths": kwargs["changed_paths"],
            },
        }
    )

    result = fc.process_semantic_repair_frontier(
        {
            "snapshot_id": "snap:1",
            "repo_name": "repo",
            "changed_paths": ["a.py"],
            "reason": "api_or_edge_surface_changed",
            "scope_kind": "package",
            "scope_roots": ["pkg"],
        }
    )

    assert result["status"] == "repaired"
    assert result["kwargs"]["snapshot_id"] == "snap:1"
    assert result["repair_frontier"]["snapshot_id"] == "snap:1"
    assert result["repair_frontier"]["changed_paths"] == ["a.py"]
    assert result["repair_frontier"]["scope_kind"] == "package"
    assert result["repair_frontier"]["scope_roots"] == ["pkg"]


def test_process_semantic_repair_frontier_marks_existing_projections_dirty():
    fc = FastCode.__new__(FastCode)
    fc.pipeline = SimpleNamespace(
        run_semantic_repair_frontier=lambda **kwargs: {
            "status": "repaired",
            "warnings": [],
            "repair_frontier": {
                "scope_kind": kwargs["scope_kind"],
                "scope_roots": kwargs["scope_roots"],
                "changed_paths": kwargs["changed_paths"],
                "target_paths": ["pkg/a.py", "pkg/b.py"],
            },
        }
    )
    marked: list[dict[str, Any]] = []
    fc.projection_store = SimpleNamespace(
        enabled=True,
        list_builds_for_snapshot=lambda _snapshot_id: [
            {"scope_kind": "snapshot", "scope_key": "scope:snapshot"},
            {"scope_kind": "query", "scope_key": "scope:query"},
        ],
        mark_dirty=lambda **kwargs: marked.append(kwargs),
    )

    result = fc.process_semantic_repair_frontier(
        {
            "snapshot_id": "snap:1",
            "repo_name": "repo",
            "changed_paths": ["pkg/a.py"],
            "scope_kind": "package",
            "scope_roots": ["pkg"],
            "change_kinds": ["api_surface_hash"],
        }
    )

    assert result["projection_dirty"]["marked"] == 2
    assert result["projection_dirty"]["reason"] == "api"
    assert {entry["scope_key"] for entry in marked} == {
        "scope:snapshot",
        "scope:query",
    }
    assert marked[0]["dirty_paths"] == ["pkg/a.py", "pkg/b.py"]


def test_process_semantic_repair_frontier_skips_unrelated_projection_scopes():
    fc = FastCode.__new__(FastCode)
    fc.pipeline = SimpleNamespace(
        run_semantic_repair_frontier=lambda **kwargs: {
            "status": "repaired",
            "warnings": [],
            "repair_frontier": {
                "scope_kind": kwargs["scope_kind"],
                "scope_roots": kwargs["scope_roots"],
                "changed_paths": kwargs["changed_paths"],
                "target_paths": ["pkg/a.py"],
            },
        }
    )
    marked: list[dict[str, Any]] = []
    fc.projection_store = SimpleNamespace(
        enabled=True,
        list_builds_for_snapshot=lambda _snapshot_id: [
            {
                "scope_kind": "snapshot",
                "scope_key": "scope:snapshot",
                "coverage_paths": ["pkg/a.py"],
            },
            {
                "scope_kind": "query",
                "scope_key": "scope:query",
                "coverage_paths": ["other/c.py"],
            },
        ],
        mark_dirty=lambda **kwargs: marked.append(kwargs),
        mark_all_dirty=lambda *args, **kwargs: None,
    )

    result = fc.process_semantic_repair_frontier(
        {
            "snapshot_id": "snap:1",
            "repo_name": "repo",
            "changed_paths": ["pkg/a.py"],
            "scope_kind": "package",
            "scope_roots": ["pkg"],
            "change_kinds": ["embedding_text_hash"],
        }
    )

    assert result["projection_dirty"]["marked"] == 1
    assert {entry["scope_key"] for entry in marked} == {"scope:snapshot"}


def test_process_semantic_repair_frontier_widens_topology_dirty_scopes():
    fc = FastCode.__new__(FastCode)
    fc.pipeline = SimpleNamespace(
        run_semantic_repair_frontier=lambda **kwargs: {
            "status": "repaired",
            "warnings": [],
            "repair_frontier": {
                "scope_kind": kwargs["scope_kind"],
                "scope_roots": kwargs["scope_roots"],
                "changed_paths": kwargs["changed_paths"],
                "target_paths": ["pkg/a.py", "pkg/b.py", "pkg/c.py"],
            },
        }
    )
    all_dirty: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    fc.projection_store = SimpleNamespace(
        enabled=True,
        list_builds_for_snapshot=lambda _snapshot_id: [
            {
                "scope_kind": "snapshot",
                "scope_key": "scope:snapshot",
                "coverage_paths": ["pkg/a.py"],
            }
        ],
        mark_dirty=lambda **kwargs: None,
        mark_all_dirty=lambda *args, **kwargs: all_dirty.append((args, kwargs)),
    )

    result = fc.process_semantic_repair_frontier(
        {
            "snapshot_id": "snap:1",
            "repo_name": "repo",
            "changed_paths": ["pkg/a.py"],
            "scope_kind": "package",
            "scope_roots": ["pkg"],
            "change_kinds": ["api_surface_hash", "edge_surface_hash"],
        }
    )

    assert result["projection_dirty"]["reason"] == "graph_topology"
    assert result["projection_dirty"]["widened"] is True
    assert all_dirty


def test_load_multi_repo_cache_uses_explicit_code_element_deserializer(
    tmp_path: Path,
) -> None:
    fc = FastCode.__new__(FastCode)
    fc.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
    )
    fc.embedder = SimpleNamespace(embedding_dim=3)
    fc.loaded_repositories = {}
    fc.vector_store = SimpleNamespace(
        persist_dir=str(tmp_path),
        initialize=lambda _dimension: None,
        merge_from_index=lambda _repo_name: True,
        get_count=lambda: 1,
    )
    fc.retriever = SimpleNamespace(
        persist_dir=str(tmp_path),
        full_bm25_elements=[],
        full_bm25_corpus=[],
        full_bm25=None,
        index_for_bm25=lambda _elements: None,
        build_repo_overview_bm25=lambda: None,
    )
    fc.graph_builder = SimpleNamespace(
        load=lambda _repo_name: True,
        merge_from_file=lambda _repo_name: True,
    )
    fc._reconstruct_elements_from_metadata = lambda: []

    (tmp_path / "repo.faiss").touch()
    with open(tmp_path / "repo_metadata.pkl", "wb") as handle:
        pickle.dump({"metadata": []}, handle)

    payload = {
        "id": "file:service",
        "type": "file",
        "name": "service.py",
        "file_path": "/repo/service.py",
        "relative_path": "service.py",
        "language": "python",
        "start_line": 1,
        "end_line": 2,
        "code": "pass\n",
        "signature": None,
        "docstring": None,
        "summary": None,
        "metadata": {"stable_unit_id": "unit:file:service"},
        "repo_name": "repo",
        "repo_url": None,
    }
    with open(tmp_path / "repo_bm25.pkl", "wb") as handle:
        pickle.dump({"bm25_corpus": [["service"]], "bm25_elements": [payload]}, handle)

    calls: list[dict[str, Any]] = []

    def _deserialize(element_payload: dict[str, Any]) -> CodeElement:
        calls.append(element_payload)
        return CodeElement(
            id=element_payload["id"],
            type=element_payload["type"],
            name=element_payload["name"],
            file_path=element_payload["file_path"],
            relative_path=element_payload["relative_path"],
            language=element_payload["language"],
            start_line=element_payload["start_line"],
            end_line=element_payload["end_line"],
            code=element_payload["code"],
            signature=element_payload["signature"],
            docstring=element_payload["docstring"],
            summary=element_payload["summary"],
            metadata=element_payload["metadata"],
            repo_name=element_payload["repo_name"],
            repo_url=element_payload["repo_url"],
        )

    with patch(
        "fastcode.main.fastcode.deserialize_code_element",
        side_effect=_deserialize,
    ) as mock_deserialize:
        assert fc._load_multi_repo_cache(["repo"]) is True

    assert mock_deserialize.call_count == 1
    assert calls == [payload]
    assert fc.retriever.full_bm25_elements[0].id == "file:service"
    assert fc.retriever.full_bm25_corpus == [["service"]]


def _run_refresh_index_cache(fc: Any) -> None:
    from fastcode.api.web import _refresh_index_cache_sync

    _refresh_index_cache_sync(fc)


@pytest.mark.parametrize(
    ("mutation_name", "run_mutation"),
    [
        ("load", lambda fc: fc.load_repository("/tmp/repo", False)),
        ("index", lambda fc: fc.index_repository(force=True)),
        ("delete", lambda fc: fc.remove_repository("repo", delete_source=False)),
        ("refresh", _run_refresh_index_cache),
        ("cleanup", lambda fc: fc.cleanup()),
    ],
)
def test_service_state_lock_serializes_query_with_mutations(
    mutation_name: str,
    run_mutation: Any,
) -> None:
    """Query serving must not overlap mutable singleton-state operations."""
    import threading
    import time

    fc = FastCode.__new__(FastCode)
    fc._service_state_lock = threading.RLock()
    fc._redo_worker = None
    fc.graph_runtime = None

    concurrent_count = 0
    max_concurrent = 0
    calls: list[str] = []
    count_lock = threading.Lock()
    start_barrier = threading.Barrier(2, timeout=5)
    errors: list[Exception] = []

    def _enter_critical(name: str) -> None:
        nonlocal concurrent_count, max_concurrent
        with count_lock:
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            calls.append(name)
        time.sleep(0.05)
        with count_lock:
            concurrent_count -= 1

    fc.query_handler = SimpleNamespace(
        query=lambda **_kwargs: (
            _enter_critical("query") or {"answer": "ok", "sources": []}
        )
    )
    fc._load_repository_unlocked = lambda *_args, **_kwargs: _enter_critical("load")
    fc._index_repository_unlocked = lambda **_kwargs: _enter_critical("index")
    fc._remove_repository_unlocked = lambda *_args, **_kwargs: (
        _enter_critical("delete") or {"deleted_files": [], "freed_mb": 0.0}
    )
    fc.vector_store = SimpleNamespace(
        invalidate_scan_cache=lambda: _enter_critical("refresh"),
        scan_available_indexes=lambda use_cache=True: [],
    )
    fc.loader = SimpleNamespace(cleanup=lambda: _enter_critical("cleanup"))
    fc.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)

    def _run_query() -> None:
        try:
            start_barrier.wait(timeout=5)
            fc.query("Where is auth?")
        except Exception as exc:
            errors.append(exc)

    def _run_mutation() -> None:
        try:
            start_barrier.wait(timeout=5)
            run_mutation(fc)
        except Exception as exc:
            errors.append(exc)

    query_thread = threading.Thread(target=_run_query)
    mutation_thread = threading.Thread(target=_run_mutation)
    query_thread.start()
    mutation_thread.start()
    query_thread.join(timeout=10)
    mutation_thread.join(timeout=10)

    assert not query_thread.is_alive()
    assert not mutation_thread.is_alive()
    assert not errors, f"Threads raised: {errors}"
    assert max_concurrent <= 1, f"query overlapped with {mutation_name}; calls={calls}"
    assert "query" in calls
    assert mutation_name in calls


def test_snapshot_artifact_handle_loader_caches_by_artifact_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    from collections import OrderedDict

    import fastcode.indexing.pipeline as pipeline_module

    vector_loads: list[str] = []
    bm25_loads: list[str] = []

    class FakeVectorStore:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config
            self.metadata = [
                {
                    "id": "file:auth",
                    "type": "file",
                    "name": "auth.py",
                    "file_path": "auth.py",
                    "relative_path": "auth.py",
                    "language": "python",
                    "start_line": 1,
                    "end_line": 1,
                    "code": "pass",
                    "signature": None,
                    "docstring": None,
                    "summary": None,
                    "metadata": {},
                    "repo_name": "repo",
                    "repo_url": None,
                }
            ]

        def load(self, artifact_key: str) -> bool:
            vector_loads.append(artifact_key)
            return True

    class FakeGraphBuilder:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

        def load(self, artifact_key: str) -> bool:
            return True

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
            bm25_loads.append(artifact_key)
            return True

        def set_ir_graphs(self, ir_graphs: Any, snapshot_id: str | None = None) -> None:
            self.ir_graphs = ir_graphs
            self.snapshot_id = snapshot_id

        def build_repo_overview_bm25(self) -> None:
            return None

    monkeypatch.setattr(pipeline_module, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(pipeline_module, "CodeGraphBuilder", FakeGraphBuilder)
    monkeypatch.setattr(pipeline_module, "HybridRetriever", FakeRetriever)

    pipeline = pipeline_module.IndexPipeline.__new__(pipeline_module.IndexPipeline)
    pipeline.config = {"query": {"snapshot_handle_cache_size": 2}}
    pipeline.embedder = object()
    pipeline.loader = SimpleNamespace(repo_path="/tmp/repo")
    pipeline.snapshot_store = SimpleNamespace(
        load_ir_graphs=lambda snapshot_id: "ir-graphs",
        find_by_artifact_key=lambda artifact_key: {"snapshot_id": "snap:1"},
    )
    pipeline.pg_retrieval_store = None
    pipeline._artifact_lock = threading.RLock()
    pipeline._artifact_handle_cache = OrderedDict()

    first = pipeline.load_snapshot_artifacts_handle(
        "snap_cache",
        snapshot_id="snap:1",
    )
    second = pipeline.load_snapshot_artifacts_handle(
        "snap_cache",
        snapshot_id="snap:1",
    )

    assert first is second
    assert vector_loads == ["snap_cache"]
    assert bm25_loads == ["snap_cache"]
