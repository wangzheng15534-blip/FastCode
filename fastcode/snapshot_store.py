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
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshot_refs (
                    ref_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_name TEXT NOT NULL,
                    branch TEXT,
                    commit_id TEXT,
                    tree_id TEXT,
                    snapshot_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(repo_name, commit_id, snapshot_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scip_artifacts (
                    snapshot_id TEXT PRIMARY KEY,
                    indexer_name TEXT NOT NULL,
                    indexer_version TEXT,
                    artifact_path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_snapshot_refs_repo_branch
                ON snapshot_refs (repo_name, branch, created_at DESC)
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
        tmp_ir_path = f"{ir_path}.tmp"
        with open(tmp_ir_path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_ir_path, ir_path)

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
            conn.execute(
                """
                INSERT INTO snapshot_refs (
                    repo_name, branch, commit_id, tree_id, snapshot_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_name, commit_id, snapshot_id) DO NOTHING
                """,
                (
                    snapshot.repo_name,
                    snapshot.branch,
                    snapshot.commit_id,
                    snapshot.tree_id,
                    snapshot.snapshot_id,
                    _utc_now(),
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

    def load_ir_graphs(self, snapshot_id: str) -> Optional[Any]:
        row = self.get_snapshot_record(snapshot_id)
        if not row:
            return None
        path = row.get("ir_graphs_path")
        if not path or not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

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

    def find_by_artifact_key(self, artifact_key: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM snapshots
                WHERE artifact_key=?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (artifact_key,),
            ).fetchone()
        return dict(row) if row else None

    def resolve_snapshot_for_ref(self, repo_name: str, branch: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM snapshot_refs
                WHERE repo_name=? AND branch=?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (repo_name, branch),
            ).fetchone()
        return dict(row) if row else None

    def save_scip_artifact_ref(
        self,
        snapshot_id: str,
        indexer_name: str,
        indexer_version: Optional[str],
        artifact_path: str,
        checksum: str,
    ) -> Dict[str, Any]:
        created_at = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scip_artifacts (
                    snapshot_id, indexer_name, indexer_version, artifact_path, checksum, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    indexer_name=excluded.indexer_name,
                    indexer_version=excluded.indexer_version,
                    artifact_path=excluded.artifact_path,
                    checksum=excluded.checksum,
                    created_at=excluded.created_at
                """,
                (snapshot_id, indexer_name, indexer_version, artifact_path, checksum, created_at),
            )
            conn.commit()
        return {
            "snapshot_id": snapshot_id,
            "indexer_name": indexer_name,
            "indexer_version": indexer_version,
            "artifact_path": artifact_path,
            "checksum": checksum,
            "created_at": created_at,
        }

    def get_scip_artifact_ref(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scip_artifacts WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchone()
        return dict(row) if row else None
