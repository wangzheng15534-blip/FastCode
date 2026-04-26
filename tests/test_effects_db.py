# tests/test_effects_db.py
"""Tests for DB effects — verify frozen dataclass returns."""

import sqlite3
from dataclasses import FrozenInstanceError

import pytest

from fastcode.schema.core_types import SnapshotRecord
from fastcode.effects.db import load_snapshot_record, save_snapshot_record


def _setup_test_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE snapshots (
            snapshot_id TEXT PRIMARY KEY,
            repo_name TEXT,
            branch TEXT,
            commit_id TEXT,
            tree_id TEXT
        )
    """)
    return conn


class TestLoadSnapshotRecord:
    def test_returns_dataclass_not_dict(self):
        conn = _setup_test_db()
        conn.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?)",
            ("snap:test:abc123", "myrepo", "main", "abc123", "tree1"),
        )
        result = load_snapshot_record(conn, "snap:test:abc123")
        assert isinstance(result, SnapshotRecord)
        assert result.snapshot_id == "snap:test:abc123"
        assert result.repo_name == "myrepo"

    def test_not_found_returns_none(self):
        conn = _setup_test_db()
        result = load_snapshot_record(conn, "nonexistent")
        assert result is None

    def test_result_is_frozen(self):
        conn = _setup_test_db()
        conn.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?)",
            ("snap:test:abc123", "myrepo", "main", "abc123", "tree1"),
        )
        result = load_snapshot_record(conn, "snap:test:abc123")
        with pytest.raises(FrozenInstanceError):
            result.repo_name = "changed"  # type: ignore[misc]


class TestSaveSnapshotRecord:
    def test_insert_and_load(self):
        conn = _setup_test_db()
        record = SnapshotRecord(
            snapshot_id="snap:test:new",
            repo_name="newrepo",
            branch="dev",
            commit_id="def456",
            tree_id="tree2",
        )
        save_snapshot_record(conn, record)
        loaded = load_snapshot_record(conn, "snap:test:new")
        assert loaded is not None
        assert loaded.repo_name == "newrepo"
        assert loaded.branch == "dev"
