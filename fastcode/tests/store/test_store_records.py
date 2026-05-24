"""Tests for store_records dataclass records."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from fastcode.store.cache_contracts import (
    DialogueSessionRecord,
    DialogueTurnRecord,
    QueryResultCacheRecord,
)
from fastcode.store.index_run_contracts import (
    IndexRunRecord,
    PublishTaskRecord,
)
from fastcode.store.manifest_contracts import ManifestRecord
from fastcode.store.pg_retrieval_contracts import (
    PgRetrievalElementRecord,
    PgRetrievalResultRecord,
)
from fastcode.store.projection_contracts import (
    ProjectionBuildRecord,
    ProjectionDirtyScopeRecord,
)
from fastcode.store.snapshot_contracts import (
    OutboxEventRecord,
    RedoTaskRecord,
    SCIPArtifactRecord,
    SnapshotRecord,
    SnapshotRefRecord,
)
from fastcode.store.vector_contracts import (
    RepositoryOverviewRecord,
    VectorSearchResultRecord,
)

# --- SnapshotRecord ---


def test_snapshot_record_roundtrip():
    record = SnapshotRecord(
        snapshot_id="snap:repo:abc",
        repo_name="repo",
        branch="main",
        commit_id="abc123",
        tree_id="tree1",
        artifact_key="snap_repo_abc",
        ir_path="/tmp/snap_repo_abc/ir_snapshot.json",
        ir_graphs_path="/tmp/snap_repo_abc/ir_graphs.json",
        created_at="2026-01-01T00:00:00",
        metadata_json='{"source_modes": ["scip"]}',
    )
    d = record.to_dict()
    assert d["snapshot_id"] == "snap:repo:abc"
    assert d["artifact_key"] == "snap_repo_abc"
    restored = SnapshotRecord.from_dict(d)
    assert restored == record


def test_snapshot_record_from_dict_handles_nulls():
    record = SnapshotRecord.from_dict(
        {
            "snapshot_id": "snap:1",
            "repo_name": "r",
            "artifact_key": "snap_1",
            "ir_path": "/tmp/ir.json",
            "created_at": "2026-01-01",
        }
    )
    assert record.branch is None
    assert record.commit_id is None
    assert record.tree_id is None
    assert record.ir_graphs_path is None
    assert record.metadata_json is None


def test_snapshot_record_is_frozen():
    record = SnapshotRecord(
        snapshot_id="snap:1",
        repo_name="r",
        branch=None,
        commit_id=None,
        tree_id=None,
        artifact_key="snap_1",
        ir_path="/tmp/ir.json",
        ir_graphs_path=None,
        created_at="2026-01-01",
        metadata_json=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.snapshot_id = "changed"


# --- ManifestRecord ---


def test_manifest_record_roundtrip():
    record = ManifestRecord(
        manifest_id="man:1",
        repo_name="repo",
        ref_name="main",
        snapshot_id="snap:repo:abc",
        index_run_id="run:1",
        published_at="2026-01-01T00:00:00",
        previous_manifest_id="man:0",
        status="active",
    )
    d = record.to_dict()
    assert d["manifest_id"] == "man:1"
    assert d["previous_manifest_id"] == "man:0"
    restored = ManifestRecord.from_dict(d)
    assert restored == record


def test_manifest_record_from_dict_handles_nulls():
    record = ManifestRecord.from_dict(
        {
            "manifest_id": "man:2",
            "repo_name": "repo",
            "ref_name": "dev",
            "snapshot_id": "snap:repo:abc",
            "index_run_id": "run:2",
            "published_at": "2026-01-01",
            "status": "active",
        }
    )
    assert record.previous_manifest_id is None


def test_manifest_record_is_frozen():
    record = ManifestRecord(
        manifest_id="man:1",
        repo_name="repo",
        ref_name="main",
        snapshot_id="snap:repo:abc",
        index_run_id="run:1",
        published_at="2026-01-01",
        previous_manifest_id=None,
        status="active",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.manifest_id = "changed"


# --- SnapshotRefRecord ---


def test_snapshot_ref_record_roundtrip():
    record = SnapshotRefRecord(
        ref_id=42,
        repo_name="repo",
        branch="main",
        commit_id="abc123",
        tree_id="tree1",
        snapshot_id="snap:repo:abc",
        created_at="2026-01-01T00:00:00",
    )
    d = record.to_dict()
    assert d["ref_id"] == 42
    assert d["snapshot_id"] == "snap:repo:abc"
    restored = SnapshotRefRecord.from_dict(d)
    assert restored == record


def test_snapshot_ref_record_from_dict_handles_nulls():
    record = SnapshotRefRecord.from_dict(
        {
            "repo_name": "repo",
            "snapshot_id": "snap:1",
            "created_at": "2026-01-01",
        }
    )
    assert record.ref_id is None
    assert record.branch is None
    assert record.commit_id is None
    assert record.tree_id is None


def test_snapshot_ref_record_is_frozen():
    record = SnapshotRefRecord(
        ref_id=1,
        repo_name="repo",
        branch="main",
        commit_id="abc",
        tree_id="t1",
        snapshot_id="snap:1",
        created_at="2026-01-01",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.snapshot_id = "changed"


def test_scip_artifact_record_roundtrip():
    record = SCIPArtifactRecord(
        snapshot_id="snap:repo:abc",
        indexer_name="scip-python",
        indexer_version="1.0.0",
        artifact_path="/tmp/index.scip",
        checksum="deadbeef",
        created_at="2026-01-01T00:00:00",
        artifact_id="snap:repo:abc:scip:0",
        sequence_no=0,
        role="primary",
        metadata_json='{"language":"python"}',
    )
    restored = SCIPArtifactRecord.from_dict(record.to_dict())
    assert restored == record


def test_redo_task_record_roundtrip():
    record = RedoTaskRecord(
        task_id="redo_1",
        task_type="index_run_recovery",
        payload_json='{"run_id":"run1"}',
        status="running",
        attempts=2,
        last_error="boom",
        next_attempt_at="2026-01-01T00:00:10",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:05",
    )
    restored = RedoTaskRecord.from_dict(record.to_dict())
    assert restored == record


def test_outbox_event_record_roundtrip():
    record = OutboxEventRecord(
        event_id="evt_1",
        event_type="lineage_publish",
        payload='{"snapshot_id":"snap:1"}',
        snapshot_id="snap:1",
        status="failed",
        attempts=1,
        max_attempts=5,
        created_at="2026-01-01T00:00:00",
        last_attempt_at="2026-01-01T00:00:05",
        error_message="timeout",
    )
    restored = OutboxEventRecord.from_dict(record.to_dict())
    assert restored == record


def test_projection_dirty_scope_record_roundtrip():
    record = ProjectionDirtyScopeRecord(
        snapshot_id="snap:projection",
        scope_kind="query",
        scope_key="hash123",
        dirty_paths=["pkg/a.py"],
        dirty_units=["unit:a"],
        dirty_package_roots=["pkg"],
        dirty_reason="semantic_repair",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:05",
    )
    restored = ProjectionDirtyScopeRecord.from_dict(record.to_dict())
    assert restored == record


def test_projection_build_record_roundtrip():
    record = ProjectionBuildRecord(
        projection_id="proj_1",
        snapshot_id="snap:projection",
        scope_kind="snapshot",
        scope_key="snapshot:*",
        params_hash="hash123",
        status="ready",
        warnings=["warn1"],
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:05",
        query="where config",
        target_id="unit:a",
        filters={"language": "python"},
        coverage_paths=["pkg/a.py"],
        coverage_nodes=["unit:a"],
    )
    restored = ProjectionBuildRecord.from_dict(record.to_dict())
    assert restored == record


def test_index_run_record_roundtrip():
    record = IndexRunRecord(
        run_id="run_1",
        repo_name="repo",
        snapshot_id="snap:1",
        branch="main",
        commit_id="abc123",
        idempotency_key="idem1",
        status="running",
        error_message=None,
        warnings_json='["warn1"]',
        created_at="2026-01-01T00:00:00",
        started_at="2026-01-01T00:00:05",
        completed_at=None,
    )
    restored = IndexRunRecord.from_dict(record.to_dict())
    assert restored == record


def test_publish_task_record_roundtrip():
    record = PublishTaskRecord(
        task_id="pub_1",
        run_id="run_1",
        snapshot_id="snap:1",
        manifest_id="manifest1",
        status="pending",
        attempts=1,
        last_error="boom",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:05",
    )
    restored = PublishTaskRecord.from_dict(record.to_dict())
    assert restored == record


def test_repository_overview_record_roundtrip():
    record = RepositoryOverviewRecord(
        repo_name="repo",
        content="overview",
        metadata_json='{"summary":"stored"}',
        embedding=(1.0, 2.0),
        embedding_fingerprint={"provider": "test", "model": "m1"},
    )

    restored = RepositoryOverviewRecord.from_dict(record.to_dict())

    assert restored == record


def test_query_result_cache_record_roundtrip():
    record = QueryResultCacheRecord(
        query="Where is config loaded?",
        repo_hash="repo-hash",
        result={"answer": "Use src/config.py"},
        created_at=1234.5,
    )

    restored = QueryResultCacheRecord.from_dict(record.to_dict())

    assert restored == record


def test_vector_search_result_record_roundtrip():
    record = VectorSearchResultRecord(
        metadata={
            "id": "elem:1",
            "type": "function",
            "name": "f",
            "file_path": "/repo/pkg/a.py",
            "relative_path": "pkg/a.py",
            "language": "python",
            "start_line": 1,
            "end_line": 2,
            "code": "return 1",
            "signature": None,
            "docstring": None,
            "summary": None,
            "metadata": {},
            "repo_name": "repo",
            "repo_url": None,
        },
        score=0.75,
        index=3,
    )

    restored = VectorSearchResultRecord.from_dict(record.to_dict())

    assert restored == record


def test_pg_retrieval_result_record_roundtrip():
    element = PgRetrievalElementRecord(
        id="elem:1",
        element_type="function",
        name="f",
        relative_path="pkg/a.py",
        repo_name="repo",
        start_line=10,
        end_line=12,
        embedding_fingerprint={"provider": "test", "model": "m1"},
        metadata={"embedding_text_hash": "hash"},
        present_fields=("id", "type", "name", "relative_path", "repo_name", "metadata"),
    )
    record = PgRetrievalResultRecord(element=element, score=0.75)

    restored = PgRetrievalResultRecord.from_dict(record.to_dict())

    assert restored == record


def test_pg_retrieval_result_record_avoids_nested_record_serializers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = PgRetrievalElementRecord(
        id="elem:1",
        element_type="function",
        name="f",
        relative_path="pkg/a.py",
        repo_name="repo",
        start_line=10,
        end_line=12,
        embedding_fingerprint={"provider": "test", "model": "m1"},
        metadata={"embedding_text_hash": "hash"},
        present_fields=("id", "type", "name", "relative_path", "repo_name", "metadata"),
    )
    record = PgRetrievalResultRecord(element=element, score=0.75)

    def _boom_to_dict(_: PgRetrievalElementRecord) -> dict[str, Any]:
        raise AssertionError("PgRetrievalResultRecord must not call element.to_dict()")

    def _boom_from_dict(
        cls: type[PgRetrievalElementRecord],
        data: dict[str, Any],
    ) -> PgRetrievalElementRecord:
        raise AssertionError(
            f"PgRetrievalResultRecord must not call {cls.__name__}.from_dict()"
        )

    monkeypatch.setattr(PgRetrievalElementRecord, "to_dict", _boom_to_dict)
    monkeypatch.setattr(
        PgRetrievalElementRecord,
        "from_dict",
        classmethod(_boom_from_dict),
    )

    restored = PgRetrievalResultRecord.from_dict(record.to_dict())

    assert restored == record


def test_dialogue_turn_record_roundtrip():
    record = DialogueTurnRecord(
        session_id="session-1",
        turn_number=1,
        timestamp=123.45,
        query="Where is config loaded?",
        answer="src/config.py",
        summary="Found config loader",
        retrieved_elements=[{"file": "src/config.py", "type": "file"}],
        metadata={"multi_turn": True},
    )
    restored = DialogueTurnRecord.from_dict(record.to_dict())
    assert restored == record


def test_dialogue_session_record_roundtrip():
    record = DialogueSessionRecord(
        session_id="session-1",
        created_at=100.0,
        total_turns=3,
        last_updated=105.0,
        multi_turn=True,
    )
    restored = DialogueSessionRecord.from_dict(record.to_dict())
    assert restored == record
