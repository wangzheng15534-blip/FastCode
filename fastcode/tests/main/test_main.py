"""Tests for main module."""

from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from fastcode.app.query.context_payloads import (
    context_bundle_payload,
    turn_journal_payload,
    working_memory_from_payload,
    working_memory_payload,
)
from fastcode.app.query.selection.retriever import HybridRetriever
from fastcode.app.store.artifacts.graph import GraphArtifactStore
from fastcode.app.store.cache.contracts import (
    ContextActivationRecord,
    ContextBundleRecord,
    TurnJournalRecord,
    WorkingMemoryRecord,
)
from fastcode.app.store.runs.index_run_contracts import IndexRunRecord
from fastcode.app.store.snapshots.manifest_contracts import ManifestRecord
from fastcode.app.store.snapshots.snapshot_contracts import (
    SCIPArtifactRecord,
    SnapshotRecord,
    SnapshotRefRecord,
)
from fastcode.app.store.vectors.vector import VectorStore
from fastcode.graph.build import CodeGraphBuilder
from fastcode.ir.element import CodeElement
from fastcode.ir.graph import IRGraphs, IRGraphView
from fastcode.ir.types import (
    IRCodeUnit,
    IRRelation,
    IRSnapshot,
    IRSymbol,
    IRUnitSupport,
)
from fastcode.main.config import config_from_mapping, config_to_runtime_mapping
from fastcode.main.fastcode import FastCode
from fastcode.main.runtime_state import RuntimeState
from fastcode.retrieval.context.agent_context import (
    AcceptedFact,
    EvidenceRef,
    Hypothesis,
    RiskState,
    TurnIntent,
)
from fastcode.retrieval.context.context_compiler import (
    build_context_bundle,
    build_tool_observation,
    build_turn_journal,
    build_turn_plan,
    compile_working_memory,
)
from fastcode.semantic.symbol_index import SnapshotSymbolIndex

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
    fc.state = RuntimeState()
    fc.doc_ingester = SimpleNamespace(enabled=doc_ingester_enabled)
    fc.snapshot_store = SimpleNamespace(
        db_runtime=SimpleNamespace(backend=storage_backend)
    )
    fc.graph_runtime = SimpleNamespace(
        enabled=graph_enabled, sync_docs=lambda **_: sync_result
    )
    return fc


def test_api_facade_refs_and_manifests_use_explicit_record_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    ref_record = SnapshotRefRecord(
        ref_id=1,
        repo_name="repo",
        branch="main",
        commit_id="abc123",
        tree_id="tree123",
        snapshot_id="snap:repo:abc123",
        created_at="2026-05-05T00:00:00+00:00",
    )
    manifest_record = ManifestRecord(
        manifest_id="manifest_1",
        repo_name="repo",
        ref_name="main",
        snapshot_id="snap:repo:abc123",
        index_run_id="run_1",
        published_at="2026-05-05T00:00:01+00:00",
        previous_manifest_id=None,
        status="published",
    )

    def _boom_to_dict(_: object) -> dict[str, Any]:
        raise AssertionError("FastCode API facade must not call record.to_dict()")

    def _boom_store_dict(*_: object, **__: object) -> dict[str, Any]:
        raise AssertionError("FastCode API facade must prefer typed record APIs")

    monkeypatch.setattr(SnapshotRefRecord, "to_dict", _boom_to_dict)
    monkeypatch.setattr(ManifestRecord, "to_dict", _boom_to_dict)

    fc.snapshot_store = SimpleNamespace(
        list_repo_ref_records=lambda _repo_name: [ref_record]
    )
    fc.manifest_store = SimpleNamespace(
        get_branch_manifest_record=lambda _repo_name, _ref_name: manifest_record,
        get_snapshot_manifest_record=lambda _snapshot_id: manifest_record,
        get_branch_manifest=_boom_store_dict,
        get_snapshot_manifest=_boom_store_dict,
    )

    refs = fc.list_repo_refs("repo")
    branch_manifest = fc.get_branch_manifest("repo", "main")
    snapshot_manifest = fc.get_snapshot_manifest("snap:repo:abc123")

    assert refs == [
        {
            "ref_id": 1,
            "repo_name": "repo",
            "branch": "main",
            "commit_id": "abc123",
            "tree_id": "tree123",
            "snapshot_id": "snap:repo:abc123",
            "created_at": "2026-05-05T00:00:00+00:00",
        }
    ]
    assert branch_manifest is not None
    assert branch_manifest["manifest_id"] == "manifest_1"
    assert branch_manifest["snapshot_id"] == "snap:repo:abc123"
    assert snapshot_manifest == branch_manifest


def test_api_facade_scip_artifacts_use_explicit_record_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    primary_record = SCIPArtifactRecord(
        artifact_id="snap:repo:abc123:scip:0",
        snapshot_id="snap:repo:abc123",
        sequence_no=0,
        role="primary",
        indexer_name="scip-python",
        indexer_version="1.0",
        artifact_path="/tmp/python.scip",
        checksum="abc123",
        created_at="2026-05-05T00:00:00+00:00",
        metadata_json='{"language":"python"}',
    )
    secondary_record = SCIPArtifactRecord(
        artifact_id="snap:repo:abc123:scip:1",
        snapshot_id="snap:repo:abc123",
        sequence_no=1,
        role="secondary",
        indexer_name="scip-go",
        indexer_version=None,
        artifact_path="/tmp/go.scip",
        checksum="def456",
        created_at="2026-05-05T00:00:00+00:00",
        metadata_json=None,
    )

    def _boom_to_dict(_: object) -> dict[str, Any]:
        raise AssertionError("FastCode API facade must not call record.to_dict()")

    def _boom_store_dict(*_: object, **__: object) -> dict[str, Any]:
        raise AssertionError("FastCode API facade must prefer typed record APIs")

    monkeypatch.setattr(SCIPArtifactRecord, "to_dict", _boom_to_dict)

    fc.snapshot_store = SimpleNamespace(
        get_scip_artifact_ref_record=lambda _snapshot_id: primary_record,
        list_scip_artifact_ref_records=lambda _snapshot_id: [
            primary_record,
            secondary_record,
        ],
        get_scip_artifact_ref=_boom_store_dict,
        list_scip_artifact_refs=_boom_store_dict,
    )

    artifact = fc.get_scip_artifact_ref("snap:repo:abc123")
    artifacts = fc.list_scip_artifact_refs("snap:repo:abc123")

    assert artifact is not None
    assert artifact["artifact_id"] == "snap:repo:abc123:scip:0"
    assert artifact["metadata"] == {"language": "python"}
    assert [item["artifact_path"] for item in artifacts] == [
        "/tmp/python.scip",
        "/tmp/go.scip",
    ]
    assert artifacts[1]["metadata"] == {}


def test_code_status_pack_uses_snapshot_record_and_explicit_ir_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    snapshot_id = "snap:repo:code-status"
    snapshot = IRSnapshot(
        repo_name="repo",
        snapshot_id=snapshot_id,
        branch="main",
        commit_id="abc123",
        tree_id="tree123",
        units=[
            IRCodeUnit(
                unit_id="file:pkg/a.py",
                kind="file",
                path="pkg/a.py",
                language="python",
                display_name="pkg/a.py",
                source_set={"fc_structure"},
                metadata={"content_hash": "hash-a", "blob_oid": "blob-a"},
            ),
            IRCodeUnit(
                unit_id="sym:pkg.a.work",
                kind="function",
                path="pkg/a.py",
                language="python",
                display_name="work",
                qualified_name="pkg.a.work",
                signature="def work() -> int",
                start_line=1,
                end_line=2,
                source_set={"fc_structure", "scip"},
            ),
        ],
        supports=[
            IRUnitSupport(
                support_id="support:work",
                unit_id="sym:pkg.a.work",
                source="scip",
                support_kind="occurrence",
                role="definition",
                path="pkg/a.py",
                start_line=1,
                end_line=1,
            )
        ],
        relations=[
            IRRelation(
                relation_id="rel:contain",
                src_unit_id="file:pkg/a.py",
                dst_unit_id="sym:pkg.a.work",
                relation_type="contain",
                resolution_state="structural",
                support_ids=["support:work"],
            )
        ],
    )
    snapshot_record = SnapshotRecord(
        snapshot_id=snapshot_id,
        repo_name="repo",
        branch="main",
        commit_id="abc123",
        tree_id="tree123",
        artifact_key="artifact-code-status",
        ir_path="/tmp/ir.json",
        ir_graphs_path=None,
        created_at="2026-05-21T00:00:00+00:00",
        metadata_json=None,
    )
    manifest = ManifestRecord(
        manifest_id="manifest-code-status",
        repo_name="repo",
        ref_name="main",
        snapshot_id=snapshot_id,
        index_run_id="run-code-status",
        published_at="2026-05-21T00:00:01+00:00",
        previous_manifest_id=None,
        status="published",
    )

    def _boom_to_dict(_: object) -> dict[str, Any]:
        raise AssertionError("code status export must not call to_dict()")

    monkeypatch.setattr(IRSnapshot, "to_dict", _boom_to_dict)
    monkeypatch.setattr(IRCodeUnit, "to_dict", _boom_to_dict)
    monkeypatch.setattr(IRUnitSupport, "to_dict", _boom_to_dict)
    monkeypatch.setattr(IRRelation, "to_dict", _boom_to_dict)
    monkeypatch.setattr(ManifestRecord, "to_dict", _boom_to_dict)

    fc.snapshot_store = SimpleNamespace(
        get_snapshot_record=lambda requested: (
            snapshot_record if requested == snapshot_id else None
        ),
        load_snapshot=lambda requested: snapshot if requested == snapshot_id else None,
        load_ir_graphs=lambda _requested: None,
    )
    fc.manifest_store = SimpleNamespace(
        get_snapshot_manifest_record=lambda requested: (
            manifest if requested == snapshot_id else None
        )
    )

    pack = fc.get_code_status_pack(snapshot_id, include_graph_facts=False)

    assert pack["schema_version"] == "code_status_pack.v0"
    assert pack["snapshot"]["artifact_key"] == "artifact-code-status"
    assert pack["manifest"]["manifest_id"] == "manifest-code-status"
    assert pack["source_files"][0]["path"] == "pkg/a.py"
    assert pack["relation_facts"][0]["support_span_ids"]


def test_diagnostic_bundle_reports_support_safe_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    fc.config = {
        "storage": {
            "backend": "postgres",
            "postgres_dsn": "postgresql://user:secret@db/fastcode",
            "pool_min": 1,
            "pool_max": 4,
        },
        "repository": {
            "ignore_patterns": ["node_modules", ".git"],
            "supported_extensions": [".py", ".ts"],
            "exclude_site_packages": True,
            "max_file_size_mb": 7,
        },
        "embedding": {
            "provider": "ollama",
            "model": "all-minilm",
            "ollama_url": "http://127.0.0.1:11434",
        },
        "retrieval": {
            "retrieval_backend": "pg_hybrid",
            "graph_expansion_backend": "ir",
        },
        "generation": {
            "provider": "openai",
            "model": "gpt-test",
            "api_key": "should-not-leak",
        },
        "cache": {"enabled": True, "backend": "disk"},
        "terminus": {"endpoint": "https://terminus.example", "api_key": "secret"},
        "projection": {"postgres_dsn": "postgresql://projection"},
        "repo_root": "/repo",
        "vector_store": {"persist_directory": "/tmp/vector?api_key=vector-secret"},
    }
    fc.state = RuntimeState()
    fc.state.repo_loaded = True
    fc.state.repo_indexed = True
    fc.state.multi_repo_mode = False
    fc.state.repo_info = {
        "name": "repo",
        "url": "https://oauth2:repo-secret@example.test/org/repo.git",
        "file_count": 3,
        "total_size_mb": 0.01,
    }
    fc.state.loaded_repositories = {"repo": fc.state.repo_info}
    latest_run = IndexRunRecord(
        run_id="run_1",
        repo_name="repo",
        snapshot_id="snap:repo:1",
        branch="main",
        commit_id="abc123",
        idempotency_key="idem",
        status="succeeded",
        error_message="publish failed Authorization: Bearer run-secret",
        warnings_json='["terminus_not_configured", "api_key=warning-secret"]',
        created_at="2026-05-20T00:00:00+00:00",
        started_at="2026-05-20T00:00:01+00:00",
        completed_at="2026-05-20T00:00:02+00:00",
    )
    snapshot_record = SnapshotRecord(
        snapshot_id="snap:repo:1",
        repo_name="repo",
        branch="main",
        commit_id="abc123",
        tree_id="tree123",
        artifact_key="snap_repo_1",
        ir_path="/tmp/ir.json",
        ir_graphs_path="/tmp/graphs",
        created_at="2026-05-20T00:00:00+00:00",
        metadata_json=json.dumps(
            {
                "warnings": ["terminus_not_configured"],
                "pipeline_layers": [
                    {
                        "name": "plain_ast_embedding",
                        "status": "succeeded",
                        "api_key": "layer-secret",
                    }
                ],
                "pipeline_metrics": {
                    "cache_update": {
                        "parse_cache": {
                            "hit_count": 2,
                            "detail": "token=metric-secret",
                        }
                    }
                },
            }
        ),
    )
    fc.index_run_store = SimpleNamespace(get_latest_run_record=lambda: latest_run)
    fc.snapshot_store = SimpleNamespace(
        db_runtime=SimpleNamespace(
            backend="postgres",
            sqlite_path=None,
            postgres_dsn="postgresql://user:secret@db/fastcode",
            pool_min=1,
            pool_max=4,
            pool=object(),
        ),
        get_snapshot_record=lambda _snapshot_id: snapshot_record,
    )
    fc.vector_store = SimpleNamespace(persist_dir="/var/lib/fastcode/vector")
    fc.cache_manager = SimpleNamespace()
    fc.projection_store = SimpleNamespace(enabled=True)
    fc.loader = SimpleNamespace(repo_path="/repo")

    monkeypatch.setattr(
        FastCode,
        "_dependency_available",
        staticmethod(lambda import_name: import_name in {"numpy", "git"}),
    )
    monkeypatch.setattr(
        "fastcode.main.fastcode.shutil.which",
        lambda executable: f"/usr/bin/{executable}" if executable == "git" else None,
    )

    bundle = fc.build_diagnostic_bundle()

    assert bundle["schema_version"] == "fastcode.diagnostic_bundle.v1"
    assert bundle["runtime"]["repo_loaded"] is True
    assert bundle["runtime"]["loaded_repository_count"] == 1
    assert bundle["config_summary"]["storage"] == {
        "backend": "postgres",
        "postgres_dsn_configured": True,
        "pool_min": 1,
        "pool_max": 4,
    }
    serialized_bundle = json.dumps(bundle)
    assert "postgresql://user:secret" not in serialized_bundle
    assert "should-not-leak" not in serialized_bundle
    assert "repo-secret" not in serialized_bundle
    assert "run-secret" not in serialized_bundle
    assert "warning-secret" not in serialized_bundle
    assert "layer-secret" not in serialized_bundle
    assert "metric-secret" not in serialized_bundle
    assert "vector-secret" not in serialized_bundle
    assert bundle["runtime"]["repo_info"]["url"] == (
        "https://[redacted]@example.test/org/repo.git"
    )
    assert bundle["storage"]["backend"] == "postgres"
    assert bundle["storage"]["postgres_dsn_configured"] is True
    assert bundle["storage"]["pool_configured"] is True
    python_deps = {
        item["name"]: item["available"] for item in bundle["dependencies"]["python"]
    }
    assert python_deps["numpy"] is True
    assert python_deps["openai"] is False
    tool_deps = {
        item["name"]: item["available"]
        for item in bundle["dependencies"]["external_tools"]
    }
    assert tool_deps["git"] is True
    assert bundle["latest_index_run"]["run_id"] == "run_1"
    assert bundle["latest_index_run"]["error_message"] == (
        "publish failed Authorization: Bearer [redacted]"
    )
    assert bundle["latest_index_run"]["warnings"] == [
        "terminus_not_configured",
        "api_key=[redacted]",
    ]
    assert bundle["latest_index_run"]["snapshot"]["artifact_key"] == "snap_repo_1"
    assert (
        bundle["latest_index_run"]["snapshot"]["pipeline_layers"][0]["api_key"]
        == "[redacted]"
    )
    assert bundle["latest_index_run"]["snapshot"]["pipeline_metrics"][
        "cache_update"
    ] == {"parse_cache": {"hit_count": 2, "detail": "token=[redacted]"}}


def _make_working_memory_record(
    *,
    session_id: str = "sess-1",
    turn_number: int = 2,
) -> tuple[WorkingMemoryRecord, dict[str, Any]]:
    from fastcode.retrieval.context.agent_context import build_acceptance_contract

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
            working_memory_payload(artifact), separators=(",", ":"), sort_keys=True
        ),
        stable_fcx=artifact.stable_fcx,
        turn_fcx=artifact.turn_fcx,
        obs_fcx=artifact.obs_fcx,
        full_fcx=artifact.full_fcx,
        created_at=artifact.created_at,
    )
    return record, working_memory_payload(artifact)


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
    fc.state = RuntimeState()
    fc.graph_runtime = SimpleNamespace(
        enabled=True,
        sync_docs=lambda **_: (_ for _ in ()).throw(RuntimeError("db offline")),
    )
    warnings = []

    fc._sync_doc_overlay(chunks=[{"chunk_id": "c1"}], mentions=[], warnings=warnings)

    assert warnings == ["ladybug_doc_sync_failed: db offline"]


def test_apply_repository_runtime_overrides_refreshes_loader_and_runtime_config():
    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    fc.runtime_config = config_from_mapping(
        {
            "repository": {
                "ignore_patterns": ["base"],
                "exclude_site_packages": False,
            }
        }
    )
    fc.config = config_to_runtime_mapping(fc.runtime_config)
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
    fc.state = RuntimeState()
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
    fc.state = RuntimeState()
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
    fc.state = RuntimeState()
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
    fc.state = RuntimeState()
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
    fc.state = RuntimeState()
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
    fc.state = RuntimeState()
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
            "metadata": {"tags": {"core"}},
        },
        load_snapshot=lambda _snapshot_id: (_ for _ in ()).throw(
            AssertionError("find_symbol should not full-load IRSnapshot")
        ),
    )

    assert fc.find_symbol("snap:1", name="AuthService") == {
        "symbol_id": "sym:auth",
        "display_name": "AuthService",
        "path": "src/auth.py",
        "metadata": {"tags": ["core"]},
    }


def test_find_symbol_fallback_uses_explicit_symbol_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    fc.snapshot_symbol_index = SnapshotSymbolIndex()
    symbol = IRSymbol(
        symbol_id="sym:auth",
        external_symbol_id="scip:auth",
        path="src/auth.py",
        display_name="AuthService",
        kind="class",
        language="python",
        qualified_name="auth.AuthService",
        signature="class AuthService",
        start_line=1,
        start_col=0,
        end_line=10,
        end_col=0,
        source_priority=100,
        source_set={"scip", "fc_structure"},
        metadata={"rank": 1, "tags": {"core"}},
    )
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
        load_snapshot_symbol_record=lambda _snapshot_id, _symbol_id: None,
        load_snapshot=lambda _snapshot_id: IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:1",
            symbols=[symbol],
        ),
    )

    def _boom(_: IRSymbol) -> dict[str, Any]:
        raise AssertionError("find_symbol fallback must not call IRSymbol.to_dict()")

    monkeypatch.setattr(IRSymbol, "to_dict", _boom)

    assert fc.find_symbol("snap:1", name="AuthService") == {
        "symbol_id": "sym:auth",
        "external_symbol_id": "scip:auth",
        "path": "src/auth.py",
        "display_name": "AuthService",
        "kind": "class",
        "language": "python",
        "qualified_name": "auth.AuthService",
        "signature": "class AuthService",
        "start_line": 1,
        "start_col": 0,
        "end_line": 10,
        "end_col": 0,
        "source_priority": 100,
        "source_set": ["fc_structure", "scip"],
        "metadata": {"rank": 1, "tags": ["core"]},
    }


def test_graph_helpers_use_compact_bounded_traversal_without_networkx():
    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
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
    fc.state = RuntimeState()
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


def test_load_multi_repo_cache_delegates_legacy_bm25_without_rebuild(
    tmp_path: Path,
) -> None:
    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    fc.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
    )
    fc.embedder = SimpleNamespace(embedding_dim=3)
    fc.state.loaded_repositories = {}
    fc.vector_store = SimpleNamespace(
        persist_dir=str(tmp_path),
        initialize=lambda _dimension: None,
        merge_from_index=lambda _repo_name: True,
        get_count=lambda: 1,
        scan_available_indexes=lambda _use_cache=False: [{"name": "repo"}],
    )
    legacy_calls: list[tuple[list[str], bool]] = []

    def _load_legacy(names: Sequence[str], *, filtered: bool) -> bool:
        legacy_calls.append((list(names), filtered))
        fc.retriever.full_bm25_elements = [
            CodeElement(
                id="file:service",
                type="file",
                name="service.py",
                file_path="/repo/service.py",
                relative_path="service.py",
                language="python",
                start_line=1,
                end_line=2,
                code="pass\n",
                signature=None,
                docstring=None,
                summary=None,
                metadata={"stable_unit_id": "unit:file:service"},
                repo_name="repo",
                repo_url=None,
            )
        ]
        fc.retriever.full_bm25_corpus = []
        fc.retriever.full_bm25 = None
        return True

    fc.retriever = SimpleNamespace(
        persist_dir=str(tmp_path),
        full_bm25_elements=[],
        full_bm25_corpus=[],
        full_bm25=None,
        index_for_bm25=lambda _elements: None,
        build_repo_overview_bm25=lambda: None,
        load_bm25_sources=lambda _names, *, filtered: False,
        load_bm25_legacy_sources=_load_legacy,
    )
    fc.graph_builder = SimpleNamespace()
    fc.graph_artifact_store = SimpleNamespace(
        load=lambda _builder, _repo_name: True,
        merge=lambda _builder, _repo_name: True,
    )
    fc._reconstruct_elements_from_metadata = lambda: []

    assert fc._load_multi_repo_cache(["repo"]) is True

    assert legacy_calls == [(["repo"], False)]
    assert fc.retriever.full_bm25_elements[0].id == "file:service"
    assert fc.retriever.full_bm25_corpus == []
    assert fc.retriever.full_bm25 is None


def test_load_multi_repo_cache_uses_shard_native_bm25_when_available(
    tmp_path: Path,
) -> None:
    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    fc.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
    )
    fc.embedder = SimpleNamespace(embedding_dim=3)
    fc.state.loaded_repositories = {}
    fc.vector_store = SimpleNamespace(
        persist_dir=str(tmp_path),
        initialize=lambda _dimension: None,
        merge_from_index=lambda _repo_name: True,
        get_count=lambda: 2,
        scan_available_indexes=lambda _use_cache=False: [{"name": "repo"}],
    )
    load_calls: list[tuple[list[str], bool]] = []
    fc.retriever = SimpleNamespace(
        persist_dir=str(tmp_path),
        full_bm25_elements=[],
        full_bm25_corpus=[],
        full_bm25=None,
        build_repo_overview_bm25=lambda: None,
        load_bm25_sources=lambda names, *, filtered: (
            load_calls.append((list(names), filtered)) or True
        ),
    )
    fc.graph_builder = SimpleNamespace()
    fc.graph_artifact_store = SimpleNamespace(
        load=lambda _builder, _repo_name: True,
        merge=lambda _builder, _repo_name: True,
    )
    fc._reconstruct_elements_from_metadata = lambda: (_ for _ in ()).throw(
        AssertionError("shard-native multi-repo load should not reconstruct metadata")
    )

    with patch(
        "fastcode.app.query.selection.retriever.BM25Okapi",
        side_effect=AssertionError("shard-native multi-repo load should not rebuild"),
    ):
        assert fc._load_multi_repo_cache(["repo"]) is True

    assert load_calls == [(["repo"], False)]


def test_load_multi_repo_cache_real_shard_artifacts_use_retriever_runtime(
    tmp_path: Path,
) -> None:
    def _make_real_retriever() -> HybridRetriever:
        return HybridRetriever(
            {"vector_store": {"persist_directory": str(tmp_path)}},
            vector_store=SimpleNamespace(load_repo_overviews=lambda **_: {}),
            embedder=SimpleNamespace(embedding_dim=3),
            graph_builder=SimpleNamespace(),
        )

    source = _make_real_retriever()
    source.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )
    source.full_bm25_corpus = [["shared", "alpha"], ["only", "repo_a"], ["other"]]
    source.full_bm25_elements = [
        CodeElement(
            id="file:a",
            type="file",
            name="a.py",
            file_path="/repo_a/a.py",
            relative_path="a.py",
            language="python",
            start_line=1,
            end_line=2,
            code="pass\n",
            signature=None,
            docstring=None,
            summary=None,
            metadata={},
            repo_name="repo_a",
            repo_url=None,
        ),
        CodeElement(
            id="file:a2",
            type="file",
            name="a2.py",
            file_path="/repo_a/a2.py",
            relative_path="a2.py",
            language="python",
            start_line=1,
            end_line=2,
            code="pass\n",
            signature=None,
            docstring=None,
            summary=None,
            metadata={},
            repo_name="repo_a",
            repo_url=None,
        ),
        CodeElement(
            id="file:a3",
            type="file",
            name="a3.py",
            file_path="/repo_a/a3.py",
            relative_path="a3.py",
            language="python",
            start_line=1,
            end_line=2,
            code="pass\n",
            signature=None,
            docstring=None,
            summary=None,
            metadata={},
            repo_name="repo_a",
            repo_url=None,
        ),
    ]
    assert source.save_bm25("repo_a") is True

    source.full_bm25_corpus = [["shared", "beta"], ["only", "repo_b"], ["other"]]
    source.full_bm25_elements = [
        CodeElement(
            id="file:b",
            type="file",
            name="b.py",
            file_path="/repo_b/b.py",
            relative_path="b.py",
            language="python",
            start_line=1,
            end_line=2,
            code="pass\n",
            signature=None,
            docstring=None,
            summary=None,
            metadata={},
            repo_name="repo_b",
            repo_url=None,
        ),
        CodeElement(
            id="file:b2",
            type="file",
            name="b2.py",
            file_path="/repo_b/b2.py",
            relative_path="b2.py",
            language="python",
            start_line=1,
            end_line=2,
            code="pass\n",
            signature=None,
            docstring=None,
            summary=None,
            metadata={},
            repo_name="repo_b",
            repo_url=None,
        ),
        CodeElement(
            id="file:b3",
            type="file",
            name="b3.py",
            file_path="/repo_b/b3.py",
            relative_path="b3.py",
            language="python",
            start_line=1,
            end_line=2,
            code="pass\n",
            signature=None,
            docstring=None,
            summary=None,
            metadata={},
            repo_name="repo_b",
            repo_url=None,
        ),
    ]
    assert source.save_bm25("repo_b") is True

    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    fc.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
    )
    fc.embedder = SimpleNamespace(embedding_dim=3)
    fc.state.loaded_repositories = {}
    fc.vector_store = SimpleNamespace(
        persist_dir=str(tmp_path),
        initialize=lambda _dimension: None,
        merge_from_index=lambda _repo_name: True,
        get_count=lambda: 2,
        scan_available_indexes=lambda _use_cache=False: [
            {"name": "repo_a"},
            {"name": "repo_b"},
        ],
    )
    fc.retriever = _make_real_retriever()
    fc.retriever.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )
    fc.graph_builder = SimpleNamespace()
    fc.graph_artifact_store = SimpleNamespace(
        load=lambda _builder, _repo_name: True,
        merge=lambda _builder, _repo_name: True,
    )
    fc._reconstruct_elements_from_metadata = lambda: (_ for _ in ()).throw(
        AssertionError("real shard-native multi-repo load should not reconstruct")
    )

    with patch(
        "fastcode.app.query.selection.retriever.BM25Okapi",
        side_effect=AssertionError("main multi-repo load should not rebuild BM25"),
    ):
        assert fc._load_multi_repo_cache(["repo_a", "repo_b"]) is True
        results = fc.retriever._keyword_search(
            "shared beta", top_k=3, repo_filter=["repo_b"]
        )

    assert fc.retriever.full_bm25 is None
    assert fc.retriever.full_bm25_corpus == []
    assert fc.retriever._full_bm25_shard_runtime is not None
    assert [row["id"] for row, _score in results] == ["file:b"]


def test_load_multi_repo_cache_real_shard_artifacts_respects_requested_subset(
    tmp_path: Path,
) -> None:
    def _make_real_retriever() -> HybridRetriever:
        return HybridRetriever(
            {"vector_store": {"persist_directory": str(tmp_path)}},
            vector_store=SimpleNamespace(load_repo_overviews=lambda **_: {}),
            embedder=SimpleNamespace(embedding_dim=3),
            graph_builder=SimpleNamespace(),
        )

    def _element(repo_name: str, rel_path: str, elem_id: str) -> CodeElement:
        return CodeElement(
            id=elem_id,
            type="file",
            name=rel_path,
            file_path=f"/{repo_name}/{rel_path}",
            relative_path=rel_path,
            language="python",
            start_line=1,
            end_line=2,
            code="pass\n",
            signature=None,
            docstring=None,
            summary=None,
            metadata={},
            repo_name=repo_name,
            repo_url=None,
        )

    writer = _make_real_retriever()
    writer.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )
    writer.full_bm25_corpus = [["shared", "alpha"], ["only", "repo_a"], ["other"]]
    writer.full_bm25_elements = [
        _element("repo_a", "a.py", "file:a"),
        _element("repo_a", "a2.py", "file:a2"),
        _element("repo_a", "a3.py", "file:a3"),
    ]
    assert writer.save_bm25("repo_a") is True

    writer.full_bm25_corpus = [["shared", "beta"], ["only", "repo_b"], ["other"]]
    writer.full_bm25_elements = [
        _element("repo_b", "b.py", "file:b"),
        _element("repo_b", "b2.py", "file:b2"),
        _element("repo_b", "b3.py", "file:b3"),
    ]
    assert writer.save_bm25("repo_b") is True

    merged: list[str] = []
    graph_loads: list[str] = []
    graph_merges: list[str] = []

    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    fc.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
    )
    fc.embedder = SimpleNamespace(embedding_dim=3)
    fc.state.loaded_repositories = {}
    fc.vector_store = SimpleNamespace(
        persist_dir=str(tmp_path),
        initialize=lambda _dimension: None,
        merge_from_index=lambda repo_name: merged.append(repo_name) or True,
        get_count=lambda: len(merged),
        scan_available_indexes=lambda _use_cache=False: [
            {"name": "repo_a"},
            {"name": "repo_b"},
        ],
    )
    fc.retriever = _make_real_retriever()
    fc.retriever.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )
    fc.graph_builder = SimpleNamespace()
    fc.graph_artifact_store = SimpleNamespace(
        load=lambda _builder, repo_name: graph_loads.append(repo_name) or True,
        merge=lambda _builder, repo_name: graph_merges.append(repo_name) or True,
    )
    fc._reconstruct_elements_from_metadata = lambda: (_ for _ in ()).throw(
        AssertionError("subset shard-native load should not reconstruct")
    )

    with patch(
        "fastcode.app.query.selection.retriever.BM25Okapi",
        side_effect=AssertionError("subset shard-native load should not rebuild"),
    ):
        assert fc._load_multi_repo_cache(["repo_b"]) is True
        results = fc.retriever._keyword_search("shared beta", top_k=3)

    assert merged == ["repo_b"]
    assert graph_loads == ["repo_b"]
    assert graph_merges == []
    assert list(fc.state.loaded_repositories) == ["repo_b"]
    assert [row["id"] for row, _score in results] == ["file:b"]


def test_load_multi_repo_cache_real_vector_and_bm25_artifacts_merge_subset(
    tmp_path: Path,
) -> None:
    def _make_real_retriever(vector_store: Any) -> HybridRetriever:
        return HybridRetriever(
            {"vector_store": {"persist_directory": str(tmp_path)}},
            vector_store=vector_store,
            embedder=SimpleNamespace(embedding_dim=3),
            graph_builder=SimpleNamespace(),
        )

    def _element(repo_name: str, rel_path: str, elem_id: str) -> CodeElement:
        return CodeElement(
            id=elem_id,
            type="file",
            name=rel_path,
            file_path=f"/{repo_name}/{rel_path}",
            relative_path=rel_path,
            language="python",
            start_line=1,
            end_line=2,
            code="pass\n",
            signature=None,
            docstring=None,
            summary=None,
            metadata={},
            repo_name=repo_name,
            repo_url=None,
        )

    def _meta(repo_name: str, rel_path: str, elem_id: str) -> dict[str, Any]:
        return {
            "id": elem_id,
            "type": "file",
            "name": rel_path,
            "file_path": f"/{repo_name}/{rel_path}",
            "relative_path": rel_path,
            "language": "python",
            "start_line": 1,
            "end_line": 2,
            "code": "pass\n",
            "signature": None,
            "docstring": None,
            "summary": None,
            "metadata": {},
            "repo_name": repo_name,
            "repo_url": None,
        }

    writer_a = VectorStore({"vector_store": {"persist_directory": str(tmp_path)}})
    writer_a.initialize(3)
    writer_a.add_vectors(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        [_meta("repo_a", "a.py", "vec:a")],
    )
    writer_a.save("repo_a")

    writer_b = VectorStore({"vector_store": {"persist_directory": str(tmp_path)}})
    writer_b.initialize(3)
    writer_b.add_vectors(
        np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        [_meta("repo_b", "b.py", "vec:b")],
    )
    writer_b.save("repo_b")

    bm25_writer = _make_real_retriever(
        VectorStore({"vector_store": {"persist_directory": str(tmp_path)}})
    )
    bm25_writer.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )
    bm25_writer.full_bm25_corpus = [["shared", "alpha"], ["only", "repo_a"], ["other"]]
    bm25_writer.full_bm25_elements = [
        _element("repo_a", "a.py", "file:a"),
        _element("repo_a", "a2.py", "file:a2"),
        _element("repo_a", "a3.py", "file:a3"),
    ]
    assert bm25_writer.save_bm25("repo_a") is True
    bm25_writer.full_bm25_corpus = [["shared", "beta"], ["only", "repo_b"], ["other"]]
    bm25_writer.full_bm25_elements = [
        _element("repo_b", "b.py", "file:b"),
        _element("repo_b", "b2.py", "file:b2"),
        _element("repo_b", "b3.py", "file:b3"),
    ]
    assert bm25_writer.save_bm25("repo_b") is True

    target_store = VectorStore({"vector_store": {"persist_directory": str(tmp_path)}})
    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    fc.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
    )
    fc.embedder = SimpleNamespace(embedding_dim=3)
    fc.state.loaded_repositories = {}
    fc.vector_store = target_store
    fc.retriever = _make_real_retriever(target_store)
    fc.retriever.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )
    fc.graph_builder = SimpleNamespace()
    fc.graph_artifact_store = SimpleNamespace(
        load=lambda _builder, _repo_name: True,
        merge=lambda _builder, _repo_name: True,
    )
    fc._reconstruct_elements_from_metadata = lambda: (_ for _ in ()).throw(
        AssertionError("real vector+bm25 subset load should not reconstruct")
    )

    with patch(
        "fastcode.app.query.selection.retriever.BM25Okapi",
        side_effect=AssertionError("subset load should not rebuild BM25"),
    ):
        assert fc._load_multi_repo_cache(["repo_b"]) is True
        results = fc.retriever._keyword_search("shared beta", top_k=3)

    assert fc.vector_store.get_count() == 1
    assert [row["id"] for row in fc.vector_store.metadata] == ["vec:b"]
    assert list(fc.state.loaded_repositories) == ["repo_b"]
    assert [row["id"] for row, _score in results] == ["file:b"]


def test_load_multi_repo_cache_real_vector_bm25_and_graph_artifacts_merge_subset(
    tmp_path: Path,
) -> None:
    config = {"vector_store": {"persist_directory": str(tmp_path)}}

    def _make_real_retriever(vector_store: Any, graph_builder: Any) -> HybridRetriever:
        return HybridRetriever(
            config,
            vector_store=vector_store,
            embedder=SimpleNamespace(embedding_dim=3),
            graph_builder=graph_builder,
        )

    def _element(repo_name: str, rel_path: str, elem_id: str) -> CodeElement:
        return CodeElement(
            id=elem_id,
            type="function",
            name=elem_id,
            file_path=f"/{repo_name}/{rel_path}",
            relative_path=rel_path,
            language="python",
            start_line=1,
            end_line=2,
            code="pass\n",
            signature=None,
            docstring=None,
            summary=None,
            metadata={"stable_unit_id": f"stable:{elem_id}"},
            repo_name=repo_name,
            repo_url=None,
        )

    def _meta(repo_name: str, rel_path: str, elem_id: str) -> dict[str, Any]:
        return {
            "id": elem_id,
            "type": "function",
            "name": elem_id,
            "file_path": f"/{repo_name}/{rel_path}",
            "relative_path": rel_path,
            "language": "python",
            "start_line": 1,
            "end_line": 2,
            "code": "pass\n",
            "signature": None,
            "docstring": None,
            "summary": None,
            "metadata": {"stable_unit_id": f"stable:{elem_id}"},
            "repo_name": repo_name,
            "repo_url": None,
        }

    writer_a = VectorStore(config)
    writer_a.initialize(3)
    writer_a.add_vectors(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        [_meta("repo_a", "a.py", "vec:a")],
    )
    writer_a.save("repo_a")

    writer_b = VectorStore(config)
    writer_b.initialize(3)
    writer_b.add_vectors(
        np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        [_meta("repo_b", "b.py", "vec:b")],
    )
    writer_b.save("repo_b")

    bm25_writer = _make_real_retriever(
        VectorStore(config),
        CodeGraphBuilder(config),
    )
    bm25_writer.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )
    bm25_writer.full_bm25_corpus = [["shared", "alpha"], ["only", "repo_a"], ["other"]]
    bm25_writer.full_bm25_elements = [
        _element("repo_a", "a.py", "file:a"),
        _element("repo_a", "a2.py", "file:a2"),
        _element("repo_a", "a3.py", "file:a3"),
    ]
    assert bm25_writer.save_bm25("repo_a") is True
    bm25_writer.full_bm25_corpus = [["shared", "beta"], ["only", "repo_b"], ["other"]]
    bm25_writer.full_bm25_elements = [
        _element("repo_b", "b.py", "file:b"),
        _element("repo_b", "b2.py", "file:b2"),
        _element("repo_b", "b3.py", "file:b3"),
    ]
    assert bm25_writer.save_bm25("repo_b") is True

    graph_store = GraphArtifactStore(config)
    graph_a = CodeGraphBuilder(config)
    elem_a = _element("repo_a", "a.py", "graph:a")
    elem_a_dep = _element("repo_a", "a_dep.py", "graph:a_dep")
    graph_a.element_by_name = {elem_a.name: elem_a, elem_a_dep.name: elem_a_dep}
    graph_a.element_by_id = {elem_a.id: elem_a, elem_a_dep.id: elem_a_dep}
    graph_a.call_graph.add_edge(elem_a.id, elem_a_dep.id, type="calls")
    assert graph_store.save(graph_a, "repo_a") is True

    graph_b = CodeGraphBuilder(config)
    elem_b = _element("repo_b", "b.py", "graph:b")
    elem_b_dep = _element("repo_b", "b_dep.py", "graph:b_dep")
    graph_b.element_by_name = {elem_b.name: elem_b, elem_b_dep.name: elem_b_dep}
    graph_b.element_by_id = {elem_b.id: elem_b, elem_b_dep.id: elem_b_dep}
    graph_b.call_graph.add_edge(elem_b.id, elem_b_dep.id, type="calls")
    assert graph_store.save(graph_b, "repo_b") is True

    target_store = VectorStore(config)
    target_graph_builder = CodeGraphBuilder(config)
    target_graph_store = GraphArtifactStore(config)

    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    fc.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
    )
    fc.embedder = SimpleNamespace(embedding_dim=3)
    fc.state.loaded_repositories = {}
    fc.vector_store = target_store
    fc.graph_builder = target_graph_builder
    fc.graph_artifact_store = target_graph_store
    fc.retriever = _make_real_retriever(target_store, target_graph_builder)
    fc.retriever.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )
    fc._reconstruct_elements_from_metadata = lambda: (_ for _ in ()).throw(
        AssertionError("real vector+bm25+graph subset load should not reconstruct")
    )

    with patch(
        "fastcode.app.query.selection.retriever.BM25Okapi",
        side_effect=AssertionError("subset load should not rebuild BM25"),
    ):
        assert fc._load_multi_repo_cache(["repo_b"]) is True
        results = fc.retriever._keyword_search("shared beta", top_k=3)

    assert fc.vector_store.get_count() == 1
    assert [row["id"] for row in fc.vector_store.metadata] == ["vec:b"]
    assert list(fc.state.loaded_repositories) == ["repo_b"]
    assert [row["id"] for row, _score in results] == ["file:b"]
    assert set(fc.graph_builder.element_by_id) == {"graph:b", "graph:b_dep"}
    assert fc.graph_builder.get_related_elements("graph:b", max_hops=1) == {
        "graph:b",
        "graph:b_dep",
    }


def test_load_multi_repo_cache_real_vector_bm25_and_graph_artifacts_merge_all_repos(
    tmp_path: Path,
) -> None:
    config = {"vector_store": {"persist_directory": str(tmp_path)}}

    def _make_real_retriever(vector_store: Any, graph_builder: Any) -> HybridRetriever:
        return HybridRetriever(
            config,
            vector_store=vector_store,
            embedder=SimpleNamespace(embedding_dim=3),
            graph_builder=graph_builder,
        )

    def _element(repo_name: str, rel_path: str, elem_id: str) -> CodeElement:
        return CodeElement(
            id=elem_id,
            type="function",
            name=elem_id,
            file_path=f"/{repo_name}/{rel_path}",
            relative_path=rel_path,
            language="python",
            start_line=1,
            end_line=2,
            code="pass\n",
            signature=None,
            docstring=None,
            summary=None,
            metadata={"stable_unit_id": f"stable:{elem_id}"},
            repo_name=repo_name,
            repo_url=None,
        )

    def _meta(repo_name: str, rel_path: str, elem_id: str) -> dict[str, Any]:
        return {
            "id": elem_id,
            "type": "function",
            "name": elem_id,
            "file_path": f"/{repo_name}/{rel_path}",
            "relative_path": rel_path,
            "language": "python",
            "start_line": 1,
            "end_line": 2,
            "code": "pass\n",
            "signature": None,
            "docstring": None,
            "summary": None,
            "metadata": {"stable_unit_id": f"stable:{elem_id}"},
            "repo_name": repo_name,
            "repo_url": None,
        }

    writer_a = VectorStore(config)
    writer_a.initialize(3)
    writer_a.add_vectors(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        [_meta("repo_a", "a.py", "vec:a")],
    )
    writer_a.save("repo_a")

    writer_b = VectorStore(config)
    writer_b.initialize(3)
    writer_b.add_vectors(
        np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        [_meta("repo_b", "b.py", "vec:b")],
    )
    writer_b.save("repo_b")

    bm25_writer = _make_real_retriever(
        VectorStore(config),
        CodeGraphBuilder(config),
    )
    bm25_writer.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )
    bm25_writer.full_bm25_corpus = [["shared", "alpha"], ["only", "repo_a"], ["other"]]
    bm25_writer.full_bm25_elements = [
        _element("repo_a", "a.py", "file:a"),
        _element("repo_a", "a2.py", "file:a2"),
        _element("repo_a", "a3.py", "file:a3"),
    ]
    assert bm25_writer.save_bm25("repo_a") is True
    bm25_writer.full_bm25_corpus = [["shared", "beta"], ["only", "repo_b"], ["other"]]
    bm25_writer.full_bm25_elements = [
        _element("repo_b", "b.py", "file:b"),
        _element("repo_b", "b2.py", "file:b2"),
        _element("repo_b", "b3.py", "file:b3"),
    ]
    assert bm25_writer.save_bm25("repo_b") is True

    graph_store = GraphArtifactStore(config)
    graph_a = CodeGraphBuilder(config)
    elem_a = _element("repo_a", "a.py", "graph:a")
    elem_a_dep = _element("repo_a", "a_dep.py", "graph:a_dep")
    graph_a.element_by_name = {elem_a.name: elem_a, elem_a_dep.name: elem_a_dep}
    graph_a.element_by_id = {elem_a.id: elem_a, elem_a_dep.id: elem_a_dep}
    graph_a.call_graph.add_edge(elem_a.id, elem_a_dep.id, type="calls")
    assert graph_store.save(graph_a, "repo_a") is True

    graph_b = CodeGraphBuilder(config)
    elem_b = _element("repo_b", "b.py", "graph:b")
    elem_b_dep = _element("repo_b", "b_dep.py", "graph:b_dep")
    graph_b.element_by_name = {elem_b.name: elem_b, elem_b_dep.name: elem_b_dep}
    graph_b.element_by_id = {elem_b.id: elem_b, elem_b_dep.id: elem_b_dep}
    graph_b.call_graph.add_edge(elem_b.id, elem_b_dep.id, type="calls")
    assert graph_store.save(graph_b, "repo_b") is True

    target_store = VectorStore(config)
    target_graph_builder = CodeGraphBuilder(config)
    target_graph_store = GraphArtifactStore(config)

    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    fc.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
    )
    fc.embedder = SimpleNamespace(embedding_dim=3)
    fc.state.loaded_repositories = {}
    fc.vector_store = target_store
    fc.graph_builder = target_graph_builder
    fc.graph_artifact_store = target_graph_store
    fc.retriever = _make_real_retriever(target_store, target_graph_builder)
    fc.retriever.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )
    fc._reconstruct_elements_from_metadata = lambda: (_ for _ in ()).throw(
        AssertionError("real vector+bm25+graph multi-repo load should not reconstruct")
    )

    with patch(
        "fastcode.app.query.selection.retriever.BM25Okapi",
        side_effect=AssertionError("multi-repo load should not rebuild BM25"),
    ):
        assert fc._load_multi_repo_cache(["repo_a", "repo_b"]) is True
        repo_a_results = fc.retriever._keyword_search(
            "shared alpha", top_k=3, repo_filter=["repo_a"]
        )
        repo_b_results = fc.retriever._keyword_search(
            "shared beta", top_k=3, repo_filter=["repo_b"]
        )

    assert fc.vector_store.get_count() == 2
    assert {row["id"] for row in fc.vector_store.metadata} == {"vec:a", "vec:b"}
    assert list(fc.state.loaded_repositories) == ["repo_a", "repo_b"]
    assert [row["id"] for row, _score in repo_a_results] == ["file:a"]
    assert [row["id"] for row, _score in repo_b_results] == ["file:b"]
    assert set(fc.graph_builder.element_by_id) == {
        "graph:a",
        "graph:a_dep",
        "graph:b",
        "graph:b_dep",
    }
    assert fc.graph_builder.get_related_elements("graph:a", max_hops=1) == {
        "graph:a",
        "graph:a_dep",
    }
    assert fc.graph_builder.get_related_elements("graph:b", max_hops=1) == {
        "graph:b",
        "graph:b_dep",
    }


def test_remove_repository_removes_sharded_artifacts(tmp_path: Path) -> None:
    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
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
    fc.graph_artifact_store = SimpleNamespace(
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
    fc.refresh_index_cache()


def test_index_repository_uses_snapshot_pipeline_by_default() -> None:
    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    calls: list[dict[str, Any]] = []

    def _run_index_pipeline(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "succeeded", "snapshot_id": "snap:repo:1"}

    fc.config = {"indexing": {}}
    fc.eval_config = {}
    fc.state.repo_loaded = True
    fc.loader = SimpleNamespace(repo_path="/repo")
    fc.state.loaded_repositories = {}
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
    assert calls[0]["get_loaded_repositories"]() is fc.state.loaded_repositories


def test_index_repository_direct_path_requires_explicit_flag() -> None:
    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
    fc.config = {"indexing": {"allow_direct_index": True}}
    fc._index_repository_direct_unlocked = lambda force=False: {
        "direct": True,
        "force": force,
    }

    assert fc._index_repository_unlocked(force=True) == {
        "direct": True,
        "force": True,
    }


def test_load_multiple_repositories_uses_snapshot_pipeline_by_default() -> None:
    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
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
    fc.state.multi_repo_mode = False
    fc.state.loaded_repositories = {}
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
    fc.state = RuntimeState()
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
    fc.state = RuntimeState()

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
    fc.state = RuntimeState()

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
    fc.vector_store = SimpleNamespace(invalidate_scan_cache=lambda: None)

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
    fc.state = RuntimeState()
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
    fc.state = RuntimeState()
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
    working_memory = working_memory_from_payload(artifact_payload)
    journal = build_turn_journal(
        intent=TurnIntent(
            session_id=working_memory.session_id,
            turn_number=working_memory.turn_number,
            question="Where is auth handled?",
            kind="debug",
            requested_outcome="answer",
            snapshot_id=working_memory.snapshot_id,
            artifact_key=working_memory.artifact_key,
            repo_filter=("repo",),
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
            context_bundle_payload(bundle), separators=(",", ":"), sort_keys=True
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
            turn_journal_payload(journal), separators=(",", ":"), sort_keys=True
        ),
        created_at=journal.created_at,
    )
    saved_activations: list[ContextActivationRecord] = []

    fc = FastCode.__new__(FastCode)
    fc.state = RuntimeState()
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

    import fastcode.app.indexing.pipeline.service as pipeline_module

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
    pipeline.graph_artifact_store = SimpleNamespace(
        load=lambda _builder, artifact_key: loads["graph"].append(artifact_key) is None,
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


def test_snapshot_artifact_handle_loader_prefers_typed_artifact_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, loads = _snapshot_artifact_handle_pipeline(monkeypatch)
    pipeline.snapshot_store.find_by_artifact_key_record = lambda _artifact_key: (
        SimpleNamespace(snapshot_id="snap:typed")
    )
    pipeline.snapshot_store.find_by_artifact_key = lambda _artifact_key: (
        _ for _ in ()
    ).throw(AssertionError("artifact loader should prefer typed snapshot records"))

    handle = pipeline.load_snapshot_artifacts_handle("snap_typed")

    assert handle is not None
    assert handle.snapshot_id == "snap:typed"
    assert handle.retriever.snapshot_id == "snap:typed"
    assert loads["vector"] == ["snap_typed"]
    assert loads["bm25"] == ["snap_typed"]
    assert loads["graph"] == ["snap_typed"]


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
