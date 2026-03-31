"""
Published manifest storage for branch/ref -> snapshot mapping.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ManifestStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
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
                CREATE TABLE IF NOT EXISTS manifests (
                    manifest_id TEXT PRIMARY KEY,
                    repo_name TEXT NOT NULL,
                    ref_name TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    index_run_id TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    previous_manifest_id TEXT,
                    status TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_manifest_repo_ref_time
                ON manifests (repo_name, ref_name, published_at DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS manifest_heads (
                    repo_name TEXT NOT NULL,
                    ref_name TEXT NOT NULL,
                    manifest_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (repo_name, ref_name),
                    FOREIGN KEY (manifest_id) REFERENCES manifests(manifest_id)
                )
                """
            )
            conn.commit()

    def publish(
        self,
        repo_name: str,
        ref_name: str,
        snapshot_id: str,
        index_run_id: str,
        status: str = "published",
    ) -> Dict[str, Any]:
        manifest_id = f"manifest_{uuid.uuid4().hex[:16]}"
        now = _utc_now()

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            previous_row = conn.execute(
                """
                SELECT m.* FROM manifest_heads h
                JOIN manifests m ON m.manifest_id = h.manifest_id
                WHERE h.repo_name=? AND h.ref_name=?
                """,
                (repo_name, ref_name),
            ).fetchone()
            previous = dict(previous_row) if previous_row else None
            previous_id = previous["manifest_id"] if previous else None
            conn.execute(
                """
                INSERT INTO manifests (
                    manifest_id, repo_name, ref_name, snapshot_id, index_run_id,
                    published_at, previous_manifest_id, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest_id,
                    repo_name,
                    ref_name,
                    snapshot_id,
                    index_run_id,
                    now,
                    previous_id,
                    status,
                ),
            )
            conn.execute(
                """
                INSERT INTO manifest_heads (repo_name, ref_name, manifest_id, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(repo_name, ref_name) DO UPDATE SET
                    manifest_id=excluded.manifest_id,
                    updated_at=excluded.updated_at
                """,
                (repo_name, ref_name, manifest_id, now),
            )
            conn.commit()

        return {
            "manifest_id": manifest_id,
            "repo_name": repo_name,
            "ref_name": ref_name,
            "snapshot_id": snapshot_id,
            "index_run_id": index_run_id,
            "published_at": now,
            "previous_manifest_id": previous_id,
            "status": status,
        }

    def get_branch_manifest(self, repo_name: str, ref_name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT m.* FROM manifest_heads h
                JOIN manifests m ON m.manifest_id = h.manifest_id
                WHERE h.repo_name=? AND h.ref_name=?
                """,
                (repo_name, ref_name),
            ).fetchone()
        return dict(row) if row else None

    def get_snapshot_manifest(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM manifests
                WHERE snapshot_id=?
                ORDER BY published_at DESC
                LIMIT 1
                """,
                (snapshot_id,),
            ).fetchone()
        return dict(row) if row else None
