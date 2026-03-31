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
            conn.commit()

    def publish(
        self,
        repo_name: str,
        ref_name: str,
        snapshot_id: str,
        index_run_id: str,
        status: str = "published",
    ) -> Dict[str, Any]:
        previous = self.get_branch_manifest(repo_name, ref_name)
        manifest_id = f"manifest_{uuid.uuid4().hex[:16]}"
        now = _utc_now()
        previous_id = previous["manifest_id"] if previous else None

        with self._connect() as conn:
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
                SELECT * FROM manifests
                WHERE repo_name=? AND ref_name=?
                ORDER BY published_at DESC
                LIMIT 1
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

