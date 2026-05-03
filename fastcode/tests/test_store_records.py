"""Tests for store_records dataclass records."""

from __future__ import annotations

import dataclasses

import pytest

from fastcode.store_records import ManifestRecord, SnapshotRecord, SnapshotRefRecord

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
