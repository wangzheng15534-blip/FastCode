"""
Snapshot metadata and artifact persistence.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .semantic_ir import IRSnapshot
from .utils import ensure_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SnapshotStore:
    def __init__(self, persist_dir: str):
        self.persist_dir = os.path.abspath(persist_dir)
        self.snapshot_root = os.path.join(self.persist_dir, "snapshots")
        ensure_dir(self.persist_dir)
        ensure_dir(self.snapshot_root)
        self.db_path = os.path.join(self.persist_dir, "lineage.db")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    repo_name TEXT NOT NULL,
                    branch TEXT,
                    commit_id TEXT,
                    tree_id TEXT,
                    artifact_key TEXT NOT NULL,
                    ir_path TEXT NOT NULL,
                    ir_graphs_path TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            conn.commit()

    def artifact_key_for_snapshot(self, snapshot_id: str) -> str:
        return f"snap_{hashlib.md5(snapshot_id.encode('utf-8')).hexdigest()[:20]}"

    def snapshot_dir(self, snapshot_id: str) -> str:
        safe = self.artifact_key_for_snapshot(snapshot_id)
        path = os.path.join(self.snapshot_root, safe)
        ensure_dir(path)
        return path

    def save_snapshot(self, snapshot: IRSnapshot, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        snap_dir = self.snapshot_dir(snapshot.snapshot_id)
        ir_path = os.path.join(snap_dir, "ir_snapshot.json")
        with open(ir_path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, ensure_ascii=False, indent=2)

        artifact_key = self.artifact_key_for_snapshot(snapshot.snapshot_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO snapshots (
                    snapshot_id, repo_name, branch, commit_id, tree_id, artifact_key,
                    ir_path, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    repo_name=excluded.repo_name,
                    branch=excluded.branch,
                    commit_id=excluded.commit_id,
                    tree_id=excluded.tree_id,
                    artifact_key=excluded.artifact_key,
                    ir_path=excluded.ir_path,
                    metadata_json=excluded.metadata_json
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.repo_name,
                    snapshot.branch,
                    snapshot.commit_id,
                    snapshot.tree_id,
                    artifact_key,
                    ir_path,
                    _utc_now(),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            conn.commit()

        return {
            "snapshot_id": snapshot.snapshot_id,
            "artifact_key": artifact_key,
            "ir_path": ir_path,
            "dir": snap_dir,
        }

    def save_ir_graphs(self, snapshot_id: str, ir_graphs: Any) -> str:
        snap_dir = self.snapshot_dir(snapshot_id)
        path = os.path.join(snap_dir, "ir_graphs.pkl")
        with open(path, "wb") as f:
            pickle.dump(ir_graphs, f)
        with self._connect() as conn:
            conn.execute(
                "UPDATE snapshots SET ir_graphs_path=? WHERE snapshot_id=?",
                (path, snapshot_id),
            )
            conn.commit()
        return path

    def load_snapshot(self, snapshot_id: str) -> Optional[IRSnapshot]:
        row = self.get_snapshot_record(snapshot_id)
        if not row:
            return None
        ir_path = row.get("ir_path")
        if not ir_path or not os.path.exists(ir_path):
            return None
        with open(ir_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return IRSnapshot.from_dict(data)

    def get_snapshot_record(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM snapshots WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchone()
        return dict(row) if row else None

    def find_by_repo_commit(self, repo_name: str, commit_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM snapshots
                WHERE repo_name=? AND commit_id=?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (repo_name, commit_id),
            ).fetchone()
        return dict(row) if row else None

