# fastcode/effects/db.py
"""Thin wrappers for database I/O — each function does one query.

Rule 2: Database Trusts Dataclasses.
Every function maps DB rows into frozen dataclasses before returning.
No dict[str, Any] returns.
"""
from __future__ import annotations

from typing import Any

from fastcode.core.types import SnapshotRecord


def load_snapshot_record(
    conn: Any,
    snapshot_id: str,
) -> SnapshotRecord | None:
    """Load a snapshot record by ID. Returns frozen dataclass, not dict."""
    cursor = conn.execute(
        "SELECT snapshot_id, repo_name, branch, commit_id, tree_id "
        "FROM snapshots WHERE snapshot_id = ?",
        (snapshot_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return SnapshotRecord(
        snapshot_id=row[0],
        repo_name=row[1],
        branch=row[2],
        commit_id=row[3],
        tree_id=row[4],
    )


def save_snapshot_record(conn: Any, record: SnapshotRecord) -> None:
    """Insert or update a snapshot record. Accepts frozen dataclass."""
    conn.execute(
        "INSERT OR REPLACE INTO snapshots "
        "(snapshot_id, repo_name, branch, commit_id, tree_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (record.snapshot_id, record.repo_name, record.branch,
         record.commit_id, record.tree_id),
    )
