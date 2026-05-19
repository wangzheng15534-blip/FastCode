"""Tests for main module."""

from __future__ import annotations

import json
from collections.abc import Generator
from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from fastcode.ir.element import CodeElement
from fastcode.ir.graph import IRGraphs, IRGraphView
from fastcode.ir.types import IRCodeUnit, IRSnapshot
from fastcode.main import FastCode
from fastcode.retrieval.core.agent_context import (
    AcceptedFact,
    EvidenceRef,
    Hypothesis,
    RiskState,
    TurnIntent,
    WorkingMemoryArtifact,
)
from fastcode.retrieval.core.context_compiler import (
    build_context_bundle,
    build_tool_observation,
    build_turn_journal,
    build_turn_plan,
    compile_working_memory,
)
from fastcode.schemas.config import config_from_mapping
from fastcode.semantic.symbol_index import SnapshotSymbolIndex
from fastcode.store.records import (
    ContextActivationRecord,
    ContextBundleRecord,
    TurnJournalRecord,
    WorkingMemoryRecord,
)

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


def _make_working_memory_record(
    *,
    session_id: str = "sess-1",
    turn_number: int = 2,
) -> tuple[WorkingMemoryRecord, dict[str, Any]]:
    from fastcode.retrieval.core.agent_context import build_acceptance_contract

    intent = TurnIntent(
        session_id=session_id,
        turn_number=turn_number,
        question="Where is auth handled?",
        kind="debug",
        requested_outcome="answer",
        snapshot_id="snap:1",
        artifact_key="art:1",
        repo_filter=("repo",),
    )
    contract = build_acceptance_contract(requested_outcome="answer")
    risk_state = RiskState(
        evidence_gap=0,
        conflict_level=0,
        freshness_risk=0,
        requirement_ambiguity=0,
        execution_risk=0,
        verifier_status="clean",
        action_bias="answer",
    )
    plan = build_turn_plan(risk_state=risk_state, contract=contract)
    evidence_refs = (
        EvidenceRef(
            ref_id="e1",
            kind="range",
            repo_name="repo",
            snapshot_id="snap:1",
            path="src/auth.py",
            lines="10-20",
            label="auth.py",
            score=0.9,
            source="retrieval",
            fresh="ok",
        ),
    )
    observations = (
        build_tool_observation(
            observation_id="o1",
            tool="retrieve",
            ok=True,
            parameters={"mode": "standard"},
            ref_ids=("e1",),
            summary="Retrieved auth evidence.",
            round_number=0,
        ),
    )
    accepted_facts = (
        AcceptedFact(
            fact_id="f1",
            statement="Auth behavior is grounded in src/auth.py.",
            ref_ids=("e1",),
            scope="turn",
        ),
    )
    hypotheses = (
        Hypothesis(
            hypothesis_id="h1",
            statement="Auth logic lives in src/auth.py.",
            confidence=0.91,
            support_ref_ids=("e1",),
            conflict_ref_ids=(),
            state="favored",
        ),
    )
    artifact = compile_working_memory(
        intent=intent,
        contract=contract,
        risk_state=risk_state,
        plan=plan,
        evidence_refs=evidence_refs,
        observations=observations,
        accepted_facts=accepted_facts,
        hypotheses=hypotheses,
        rejected_hypotheses=(),
        unresolved_questions=("Verify auth call chain",),
        session_prefix=None,
        created_at=1234.5,
    )
    record = WorkingMemoryRecord(
        session_id=artifact.session_id,
        turn_number=artifact.turn_number,
        snapshot_id=artifact.snapshot_id,
        artifact_key=artifact.artifact_key,
        compiler_fingerprint=artifact.compiler_fingerprint,
        payload_json=json.dumps(
            artifact.to_dict(), separators=(",", ":"), sort_keys=True
        ),
        stable_fcx=artifact.stable_fcx,
        turn_fcx=artifact.turn_fcx,
        obs_fcx=artifact.obs_fcx,
        full_fcx=artifact.full_fcx,
        created_at=artifact.created_at,
    )
    return record, artifact.to_dict()


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


def test_process_semantic_repair_frontier_uses_coverage_nodes_when_paths_absent():
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
    snapshot = IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:1",
        units=[
            IRCodeUnit(
                unit_id="unit:a",
                kind="function",
                path="pkg/a.py",
                language="python",
                display_name="a",
            ),
            IRCodeUnit(
                unit_id="unit:c",
                kind="function",
                path="other/c.py",
                language="python",
                display_name="c",
            ),
        ],
    )
    marked: list[dict[str, Any]] = []
    fc.snapshot_store = SimpleNamespace(load_snapshot=lambda _snapshot_id: snapshot)
    fc.projection_store = SimpleNamespace(
        enabled=True,
        list_builds_for_snapshot=lambda _snapshot_id: [
            {
                "scope_kind": "snapshot",
                "scope_key": "scope:snapshot",
                "coverage_paths": [],
                "coverage_nodes": ["unit:a"],
            },
            {
                "scope_kind": "query",
                "scope_key": "scope:query",
                "coverage_paths": [],
                "coverage_nodes": ["unit:c"],
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


def test_resolve_snapshot_symbol_uses_compact_payload_without_full_snapshot_load():
    fc = FastCode.__new__(FastCode)
    fc.snapshot_symbol_index = SnapshotSymbolIndex()
    fc.snapshot_store = SimpleNamespace(
        load_snapshot_symbol_index_payload=lambda snapshot_id: {
            "schema_version": "snapshot_symbol_index.v1",
            "snapshot_id": snapshot_id,
            "symbols": [
                {
                    "canonical": "sym:auth",
                    "aliases": ["scip:auth"],
                    "names": ["AuthService"],
                    "path": "src/auth.py",
                }
            ],
        },
        load_snapshot=lambda _snapshot_id: (_ for _ in ()).throw(
            AssertionError("resolve_snapshot_symbol should not full-load IRSnapshot")
        ),
    )

    assert fc.resolve_snapshot_symbol("snap:1", name="AuthService") == "sym:auth"


def test_find_symbol_uses_compact_symbol_record_without_full_snapshot_load():
    fc = FastCode.__new__(FastCode)
    fc.snapshot_symbol_index = SnapshotSymbolIndex()
    fc.snapshot_store = SimpleNamespace(
        load_snapshot_symbol_index_payload=lambda snapshot_id: {
            "schema_version": "snapshot_symbol_index.v1",
            "snapshot_id": snapshot_id,
            "symbols": [
                {
                    "canonical": "sym:auth",
                    "aliases": ["scip:auth"],
                    "names": ["AuthService"],
                    "path": "src/auth.py",
                }
            ],
        },
        load_snapshot_symbol_record=lambda _snapshot_id, symbol_id: {
            "symbol_id": symbol_id,
            "display_name": "AuthService",
            "path": "src/auth.py",
        },
        load_snapshot=lambda _snapshot_id: (_ for _ in ()).throw(
            AssertionError("find_symbol should not full-load IRSnapshot")
        ),
    )

    assert fc.find_symbol("snap:1", name="AuthService") == {
        "symbol_id": "sym:auth",
        "display_name": "AuthService",
        "path": "src/auth.py",
    }


def test_graph_helpers_use_compact_bounded_traversal_without_networkx():
    fc = FastCode.__new__(FastCode)
    graphs = IRGraphs(
        dependency_graph=IRGraphView(
            edges=[
                ("doc:a", "doc:b", {}),
                ("doc:b", "doc:c", {}),
            ]
        ),
        call_graph=IRGraphView(
            edges=[
                ("sym:a", "sym:b", {}),
                ("sym:b", "sym:c", {}),
                ("sym:caller", "sym:a", {}),
            ]
        ),
        inheritance_graph=IRGraphView(),
        reference_graph=IRGraphView(),
        containment_graph=IRGraphView(),
    )
    fc.snapshot_store = SimpleNamespace(load_ir_graphs=lambda _snapshot_id: graphs)

    with patch(
        "networkx.single_source_shortest_path_length",
        side_effect=AssertionError("main graph helpers should use compact traversal"),
    ):
        assert fc.get_graph_callees("snap:1", "sym:a", max_hops=2) == [
            {"symbol_id": "sym:b", "distance": 1},
            {"symbol_id": "sym:c", "distance": 2},
        ]
        assert fc.get_graph_callers("snap:1", "sym:a", max_hops=1) == [
            {"symbol_id": "sym:caller", "distance": 1},
        ]
        assert fc.get_graph_dependencies("snap:1", "doc:a", max_hops=1) == [
            {"doc_id": "doc:b", "distance": 1},
        ]


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
        scan_available_indexes=lambda _use_cache=False: [{"name": "repo"}],
    )
    fc.retriever = SimpleNamespace(
        persist_dir=str(tmp_path),
        full_bm25_elements=[],
        full_bm25_corpus=[],
        full_bm25=None,
        index_for_bm25=lambda _elements: None,
        build_repo_overview_bm25=lambda: None,
        load_bm25_payload=lambda _repo_name: {
            "bm25_corpus": [["service"]],
            "bm25_elements": [payload],
        },
    )
    fc.graph_builder = SimpleNamespace(
        load=lambda _repo_name: True,
        merge_from_file=lambda _repo_name: True,
    )
    fc._reconstruct_elements_from_metadata = lambda: []

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


def test_remove_repository_removes_sharded_artifacts(tmp_path: Path) -> None:
    fc = FastCode.__new__(FastCode)
    fc.logger = SimpleNamespace(info=lambda *a, **kw: None)
    fc.config = {"repo_root": str(tmp_path / "repos")}
    fc.loader = SimpleNamespace(safe_repo_root=str(tmp_path / "repos"))
    invalidations: list[bool] = []
    fc.vector_store = SimpleNamespace(
        persist_dir=str(tmp_path),
        vector_artifact_paths=lambda _repo_name: [
            str(tmp_path / "repo_vector_manifest.json"),
            str(tmp_path / "repo_vector_shards"),
        ],
        metadata_artifact_paths=lambda _repo_name: [
            str(tmp_path / "repo_metadata_manifest.json"),
            str(tmp_path / "repo_metadata_shards"),
        ],
        delete_repo_overview=lambda _repo_name: False,
        invalidate_scan_cache=lambda: invalidations.append(True),
    )
    fc.retriever = SimpleNamespace(
        persist_dir=str(tmp_path),
        bm25_artifact_paths=lambda _repo_name: [
            str(tmp_path / "repo_bm25_manifest.json"),
            str(tmp_path / "repo_bm25_shards"),
        ],
    )
    fc.graph_builder = SimpleNamespace(
        graph_artifact_paths=lambda _repo_name: [
            str(tmp_path / "repo_graph_manifest.json"),
            str(tmp_path / "repo_graph_shards"),
        ]
    )

    (tmp_path / "repo_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "repo_vector_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "repo_graph_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "repo_metadata_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "repo_bm25_manifest.json").write_text("{}", encoding="utf-8")

    vector_shards = tmp_path / "repo_vector_shards"
    vector_shards.mkdir()
    (vector_shards / "a.npz").write_bytes(b"vector")

    graph_shards = tmp_path / "repo_graph_shards"
    graph_shards.mkdir()
    (graph_shards / "a.pkl").write_bytes(b"graph")

    metadata_shards = tmp_path / "repo_metadata_shards"
    metadata_shards.mkdir()
    (metadata_shards / "a.pkl").write_bytes(b"meta")

    bm25_shards = tmp_path / "repo_bm25_shards"
    bm25_shards.mkdir()
    (bm25_shards / "a.pkl").write_bytes(b"bm25")

    result = fc._remove_repository_unlocked("repo", delete_source=False)

    assert result["freed_bytes"] > 0
    assert invalidations == [True]
    assert set(result["deleted_files"]) == {
        "repo_manifest.json",
        "repo_vector_manifest.json",
        "repo_vector_shards",
        "repo_graph_manifest.json",
        "repo_graph_shards",
        "repo_metadata_manifest.json",
        "repo_metadata_shards",
        "repo_bm25_manifest.json",
        "repo_bm25_shards",
    }
    assert not (tmp_path / "repo_manifest.json").exists()
    assert not (tmp_path / "repo_vector_manifest.json").exists()
    assert not (tmp_path / "repo_vector_shards").exists()
    assert not (tmp_path / "repo_graph_manifest.json").exists()
    assert not (tmp_path / "repo_graph_shards").exists()
    assert not (tmp_path / "repo_metadata_manifest.json").exists()
    assert not (tmp_path / "repo_metadata_shards").exists()
    assert not (tmp_path / "repo_bm25_manifest.json").exists()
    assert not (tmp_path / "repo_bm25_shards").exists()


def _run_refresh_index_cache(fc: Any) -> None:
    from fastcode.api.web import _refresh_index_cache_sync

    _refresh_index_cache_sync(fc)


def test_index_repository_uses_snapshot_pipeline_by_default() -> None:
    fc = FastCode.__new__(FastCode)
    calls: list[dict[str, Any]] = []

    def _run_index_pipeline(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "succeeded", "snapshot_id": "snap:repo:1"}

    fc.config = {"indexing": {}}
    fc.eval_config = {}
    fc.repo_loaded = True
    fc.loader = SimpleNamespace(repo_path="/repo")
    fc.loaded_repositories = {}
    fc.graph_runtime = None
    fc.pipeline = SimpleNamespace(run_index_pipeline=_run_index_pipeline)

    result = fc._index_repository_unlocked(force=True)

    assert result == {"status": "succeeded", "snapshot_id": "snap:repo:1"}
    assert calls == [
        {
            "source": "/repo",
            "is_url": False,
            "force": True,
            "publish": True,
            "enable_scip": True,
            "load_repository_cb": None,
            "get_loaded_repositories": calls[0]["get_loaded_repositories"],
            "graph_runtime": None,
        }
    ]
    assert calls[0]["get_loaded_repositories"]() is fc.loaded_repositories


def test_index_repository_legacy_direct_path_requires_explicit_flag() -> None:
    fc = FastCode.__new__(FastCode)
    fc.config = {"indexing": {"allow_legacy_direct_index": True}}
    fc._index_repository_legacy_unlocked = lambda force=False: {
        "legacy": True,
        "force": force,
    }

    assert fc._index_repository_unlocked(force=True) == {
        "legacy": True,
        "force": True,
    }


def test_load_multiple_repositories_uses_snapshot_pipeline_by_default() -> None:
    fc = FastCode.__new__(FastCode)
    loads: list[tuple[str, bool | None, bool]] = []
    pipeline_calls: list[dict[str, Any]] = []

    def _run_index_pipeline(**kwargs: Any) -> dict[str, Any]:
        pipeline_calls.append(kwargs)
        load_cb = kwargs["load_repository_cb"]
        assert load_cb is not None
        load_cb(kwargs["source"], is_url=kwargs["is_url"])
        return {"status": "succeeded", "repo_name": "repo"}

    fc.config = {"indexing": {}}
    fc.logger = SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
    )
    fc.multi_repo_mode = False
    fc.loaded_repositories = {}
    fc.graph_runtime = None
    fc.pipeline = SimpleNamespace(run_index_pipeline=_run_index_pipeline)
    fc._load_repository_unlocked = lambda source, is_url=None, is_zip=False: (
        loads.append((source, is_url, is_zip))
    )

    result = fc._load_multiple_repositories_unlocked(
        [{"source": "/repo", "is_url": False}]
    )

    assert result["status"] == "succeeded"
    assert result["repositories"] == ["repo"]
    assert loads == [("/repo", False, False)]
    assert pipeline_calls[0]["source"] == "/repo"
    assert pipeline_calls[0]["is_url"] is False
    assert pipeline_calls[0]["publish"] is True
    assert pipeline_calls[0]["enable_scip"] is True


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


def test_service_state_lock_allows_concurrent_queries() -> None:
    """Independent queries should share the read side of the service lock."""
    import threading
    import time

    fc = FastCode.__new__(FastCode)

    concurrent_count = 0
    max_concurrent = 0
    count_lock = threading.Lock()
    start_barrier = threading.Barrier(2, timeout=5)
    errors: list[Exception] = []

    def _enter_query() -> dict[str, Any]:
        nonlocal concurrent_count, max_concurrent
        with count_lock:
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
        time.sleep(0.05)
        with count_lock:
            concurrent_count -= 1
        return {"answer": "ok", "sources": []}

    fc.query_handler = SimpleNamespace(query=lambda **_kwargs: _enter_query())

    def _run_query() -> None:
        try:
            start_barrier.wait(timeout=5)
            fc.query("Where is auth?")
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=_run_query)
    second = threading.Thread(target=_run_query)
    first.start()
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors, f"Threads raised: {errors}"
    assert max_concurrent == 2


def test_snapshot_query_stream_releases_service_lock_after_handle_capture() -> None:
    """Snapshot streams use immutable handles and do not fence later mutations."""
    import threading
    import time

    fc = FastCode.__new__(FastCode)

    concurrent_count = 0
    max_concurrent = 0
    calls: list[str] = []
    count_lock = threading.Lock()
    stream_started = threading.Event()
    mutation_started = threading.Event()
    errors: list[Exception] = []
    handle_retriever = SimpleNamespace()

    def _enter_critical(name: str) -> None:
        nonlocal concurrent_count, max_concurrent
        with count_lock:
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            calls.append(name)
        if name == "stream":
            stream_started.set()
            mutation_started.wait(timeout=2)
            time.sleep(0.05)
        else:
            mutation_started.set()
            time.sleep(0.05)
        with count_lock:
            concurrent_count -= 1

    def _query_stream(
        **kwargs: Any,
    ) -> Generator[tuple[str | None, dict[str, Any] | None], None, None]:
        assert kwargs["filters"] == {
            "snapshot_id": "snap:1",
            "artifact_key": "art_snap_1",
        }
        assert kwargs["retriever"] is handle_retriever
        _enter_critical("stream")
        yield None, {"status": "complete", "sources": []}

    fc.snapshot_store = SimpleNamespace(
        get_snapshot_record=lambda snapshot_id: SimpleNamespace(
            artifact_key="art_snap_1"
        )
    )
    fc.pipeline = SimpleNamespace(
        load_snapshot_artifacts_handle=lambda artifact_key, **_kwargs: SimpleNamespace(
            artifact_key=artifact_key,
            retriever=handle_retriever,
        )
    )
    fc.query_handler = SimpleNamespace(
        _ensure_snapshot_symbol_index=lambda snapshot_id: None,
        query_stream=_query_stream,
    )
    fc._index_repository_unlocked = lambda **_kwargs: _enter_critical("index")

    def _run_stream() -> None:
        try:
            list(fc.query_stream("Where is auth?", filters={"snapshot_id": "snap:1"}))
        except Exception as exc:
            errors.append(exc)

    def _run_mutation() -> None:
        try:
            assert stream_started.wait(timeout=5)
            fc.index_repository(force=True)
        except Exception as exc:
            errors.append(exc)

    stream_thread = threading.Thread(target=_run_stream)
    mutation_thread = threading.Thread(target=_run_mutation)
    stream_thread.start()
    mutation_thread.start()
    stream_thread.join(timeout=10)
    mutation_thread.join(timeout=10)

    assert not stream_thread.is_alive()
    assert not mutation_thread.is_alive()
    assert not errors, f"Threads raised: {errors}"
    assert calls == ["stream", "index"]
    assert max_concurrent == 2


def test_turn_context_facade_uses_typed_working_memory_payloads() -> None:
    record, artifact = _make_working_memory_record()
    fc = FastCode.__new__(FastCode)
    fc.cache_manager = SimpleNamespace(
        get_latest_working_memory_record=lambda session_id: record,
        get_working_memory_record=lambda session_id, turn_number: record,
    )

    latest = fc.get_turn_context("sess-1")
    structured = fc.get_turn_context("sess-1", 2, "json")
    expanded = fc.expand_context_ref("sess-1", 2, "e1")

    assert latest["format"] == "fcx"
    assert latest["full_fcx"] == record.full_fcx
    assert structured["artifact"]["turn_number"] == 2
    assert structured["artifact"]["accepted_facts"][0]["fact_id"] == "f1"
    assert expanded == {
        "session_id": "sess-1",
        "turn_number": 2,
        "depth": "L2",
        "ref_id": "e1",
        "kind": "range",
        "repo_name": "repo",
        "snapshot_id": "snap:1",
        "path": "src/auth.py",
        "symbol_id": None,
        "lines": "10-20",
        "label": "auth.py",
        "score": 0.9,
        "source": "retrieval",
        "fresh": "ok",
    }
    assert artifact["evidence_refs"][0]["ref_id"] == "e1"


def test_handoff_facade_persists_and_restores_typed_handoff_artifact() -> None:
    record, _artifact = _make_working_memory_record()
    saved_records: list[Any] = []
    fc = FastCode.__new__(FastCode)
    fc.cache_manager = SimpleNamespace(
        get_latest_working_memory_record=lambda session_id: record,
        get_working_memory_record=lambda session_id, turn_number: record,
        save_handoff_artifact_record=lambda handoff_record: (
            saved_records.append(handoff_record) or True
        ),
        get_handoff_artifact_record=lambda artifact_id: saved_records[0],
    )

    handoff = fc.create_handoff("sess-1", 2, "delegate")
    restored = fc.get_handoff_artifact(handoff["artifact_id"])

    assert handoff["artifact_id"].startswith("hf_")
    assert handoff["mode"] == "delegate"
    assert handoff["accepted_facts"][0]["fact_id"] == "f1"
    assert handoff["unresolved_questions"] == ["Verify auth call chain"]
    assert saved_records
    assert saved_records[0].artifact_id == handoff["artifact_id"]
    assert restored == handoff


def test_context_bundle_facade_reads_renders_expands_and_activates() -> None:
    working_memory_record, artifact_payload = _make_working_memory_record()
    working_memory = WorkingMemoryArtifact.from_dict(artifact_payload)
    journal = build_turn_journal(
        intent=TurnIntent.from_dict(
            {
                "session_id": working_memory.session_id,
                "turn_number": working_memory.turn_number,
                "question": "Where is auth handled?",
                "kind": "debug",
                "requested_outcome": "answer",
                "snapshot_id": working_memory.snapshot_id,
                "artifact_key": working_memory.artifact_key,
                "repo_filter": ("repo",),
            }
        ),
        plan=build_turn_plan(
            risk_state=working_memory.risk_state,
            contract=working_memory.acceptance_contract,
        ),
        observations=(),
        evidence_refs=working_memory.evidence_refs,
        risk_state=working_memory.risk_state,
        acceptance_contract=working_memory.acceptance_contract,
        hypotheses=working_memory.hypotheses,
        rejected_hypotheses=working_memory.rejected_hypotheses,
        accepted_facts=working_memory.accepted_facts,
        working_set=working_memory.working_set,
        answer_summary="Auth behavior is grounded in src/auth.py.",
        created_at=working_memory.created_at,
    )
    bundle = build_context_bundle(
        working_memory=working_memory,
        turn_journal=journal,
    )
    bundle_record = ContextBundleRecord(
        bundle_id=bundle.bundle_id,
        session_id=bundle.session_id,
        turn_number=bundle.turn_number,
        snapshot_id=bundle.snapshot_id,
        artifact_key=bundle.artifact_key,
        compiler_fingerprint=bundle.compiler_fingerprint,
        payload_json=json.dumps(
            bundle.to_dict(), separators=(",", ":"), sort_keys=True
        ),
        invalidation_key=bundle.distillation.invalidation_key,
        created_at=bundle.created_at,
    )
    journal_record = TurnJournalRecord(
        session_id=journal.session_id,
        turn_number=journal.turn_number,
        snapshot_id=journal.snapshot_id,
        artifact_key=journal.artifact_key,
        compiler_fingerprint=journal.compiler_fingerprint,
        payload_json=json.dumps(
            journal.to_dict(), separators=(",", ":"), sort_keys=True
        ),
        created_at=journal.created_at,
    )
    saved_activations: list[ContextActivationRecord] = []

    fc = FastCode.__new__(FastCode)
    fc.cache_manager = SimpleNamespace(
        get_latest_context_bundle_record=lambda session_id: bundle_record,
        get_context_bundle_record=lambda session_id, turn_number: bundle_record,
        get_context_bundle_record_by_id=lambda bundle_id: bundle_record,
        get_latest_working_memory_record=lambda session_id: working_memory_record,
        get_working_memory_record=lambda session_id, turn_number: working_memory_record,
        get_turn_journal_record=lambda session_id, turn_number: journal_record,
        save_context_activation_record=lambda record: (
            saved_activations.append(record) or True
        ),
    )

    structured = fc.get_context_bundle("sess-1", 2, "json")
    rendered = fc.get_context_bundle_by_id(bundle.bundle_id, "rendered", 24)
    expanded = fc.expand_context_bundle_ref("e1", bundle_id=bundle.bundle_id)
    activation = fc.create_context_activation(
        bundle_id=bundle.bundle_id,
        active_ref_ids=("e1",),
        active_fact_ids=("f1",),
        active_hypothesis_ids=("h1",),
        reason="focused_answer",
    )

    assert structured["bundle_id"] == bundle.bundle_id
    assert structured["bundle"]["distillation"]["source_refs"][0]["path"] == (
        "src/auth.py"
    )
    assert rendered["rendered"]["text"].startswith(f"bundle {bundle.bundle_id}")
    assert expanded["path"] == "src/auth.py"
    assert expanded["lines"] == "10-20"
    assert activation["active_ref_ids"] == ["e1"]
    assert activation["active_fact_ids"] == ["f1"]
    assert activation["active_hypothesis_ids"] == ["h1"]
    assert saved_activations
    assert saved_activations[0].reason == "focused_answer"


def _snapshot_artifact_handle_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cache_size: int = 2,
) -> tuple[Any, dict[str, list[str]]]:
    import threading
    from collections import OrderedDict

    import fastcode.indexing.pipeline as pipeline_module

    loads: dict[str, list[str]] = {
        "vector": [],
        "bm25": [],
        "graph": [],
        "ir_graph": [],
    }

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
            loads["vector"].append(artifact_key)
            return True

    class FakeGraphBuilder:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

        def load(self, artifact_key: str) -> bool:
            loads["graph"].append(artifact_key)
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
            loads["bm25"].append(artifact_key)
            return True

        def set_ir_graphs(self, ir_graphs: Any, snapshot_id: str | None = None) -> None:
            self.ir_graphs = ir_graphs
            self.snapshot_id = snapshot_id

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
    pipeline.config = {"query": {"snapshot_handle_cache_size": cache_size}}
    pipeline.embedder = object()
    pipeline.loader = SimpleNamespace(repo_path="/tmp/repo")

    def _load_ir_graphs(snapshot_id: str) -> str:
        loads["ir_graph"].append(snapshot_id)
        return f"ir-graphs:{snapshot_id}"

    pipeline.snapshot_store = SimpleNamespace(
        load_ir_graphs=_load_ir_graphs,
        find_by_artifact_key=lambda artifact_key: {
            "snapshot_id": artifact_key.replace("art:", "snap:")
        },
    )
    pipeline.pg_retrieval_store = None
    pipeline._artifact_lock = threading.RLock()
    pipeline._artifact_handle_cache = OrderedDict()
    return pipeline, loads


def test_snapshot_artifact_handle_loader_caches_by_artifact_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, loads = _snapshot_artifact_handle_pipeline(monkeypatch)

    first = pipeline.load_snapshot_artifacts_handle(
        "art:cache",
        snapshot_id="snap:1",
    )
    second = pipeline.load_snapshot_artifacts_handle(
        "art:cache",
        snapshot_id="snap:1",
    )

    assert first is second
    assert loads["vector"] == ["art:cache"]
    assert loads["bm25"] == ["art:cache"]
    assert loads["graph"] == ["art:cache"]
    assert loads["ir_graph"] == []
    assert first.retriever.snapshot_id == "snap:1"
    assert first.retriever.ir_graph_loader("snap:1") == "ir-graphs:snap:1"
    assert loads["ir_graph"] == ["snap:1"]


def test_snapshot_artifact_handle_cache_is_bounded_lru(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, loads = _snapshot_artifact_handle_pipeline(monkeypatch, cache_size=2)

    first = pipeline.load_snapshot_artifacts_handle("art:one", snapshot_id="snap:one")
    second = pipeline.load_snapshot_artifacts_handle("art:two", snapshot_id="snap:two")
    refreshed_first = pipeline.load_snapshot_artifacts_handle(
        "art:one",
        snapshot_id="snap:one",
    )
    third = pipeline.load_snapshot_artifacts_handle("art:three", snapshot_id="snap:3")
    reloaded_second = pipeline.load_snapshot_artifacts_handle(
        "art:two",
        snapshot_id="snap:two",
    )

    assert first is refreshed_first
    assert second is not reloaded_second
    assert third is not None
    assert loads["vector"] == ["art:one", "art:two", "art:three", "art:two"]
    assert list(pipeline._artifact_handle_cache) == ["art:three", "art:two"]


def test_snapshot_artifact_handle_cache_isolates_distinct_snapshot_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, loads = _snapshot_artifact_handle_pipeline(monkeypatch)

    first = pipeline.load_snapshot_artifacts_handle("art:one", snapshot_id="snap:one")
    second = pipeline.load_snapshot_artifacts_handle("art:two", snapshot_id="snap:two")

    assert first is not second
    assert first.snapshot_id == "snap:one"
    assert second.snapshot_id == "snap:two"
    assert first.retriever is not second.retriever
    assert first.retriever.snapshot_id == "snap:one"
    assert second.retriever.snapshot_id == "snap:two"
    assert first.retriever.ir_graph_loader("snap:one") == "ir-graphs:snap:one"
    assert second.retriever.ir_graph_loader("snap:two") == "ir-graphs:snap:two"
    assert loads["vector"] == ["art:one", "art:two"]
    assert loads["ir_graph"] == ["snap:one", "snap:two"]
