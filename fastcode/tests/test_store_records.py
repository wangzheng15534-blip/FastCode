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
