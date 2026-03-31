"""
Published manifest storage for branch/ref -> snapshot mapping.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .db_runtime import DBRuntime

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ManifestStore:
    def __init__(self, db_path_or_runtime: str | DBRuntime):
        if isinstance(db_path_or_runtime, DBRuntime):
            self.db_runtime = db_path_or_runtime
        else:
            self.db_runtime = DBRuntime(backend="sqlite", sqlite_path=db_path_or_runtime)
        self._init_db()

    def _init_db(self) -> None:
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
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
            self.db_runtime.execute(
                conn,
                """
                CREATE INDEX IF NOT EXISTS idx_manifest_repo_ref_time
                ON manifests (repo_name, ref_name, published_at DESC)
                """
            )
            self.db_runtime.execute(
                conn,
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
            self.db_runtime.execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    component TEXT NOT NULL,
                    version TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    PRIMARY KEY (component, version)
                )
                """,
            )
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO schema_migrations (component, version, applied_at)
                VALUES (?, ?, ?)
                ON CONFLICT(component, version) DO NOTHING
                """,
                ("manifest_store", "v1", _utc_now()),
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

        with self.db_runtime.connect() as conn:
            self.db_runtime.begin_write(conn)
            previous_row = self.db_runtime.execute(
                conn,
                """
                SELECT m.* FROM manifest_heads h
                JOIN manifests m ON m.manifest_id = h.manifest_id
                WHERE h.repo_name=? AND h.ref_name=?
                """,
                (repo_name, ref_name),
            ).fetchone()
            previous = self.db_runtime.row_to_dict(previous_row) if previous_row else None
            previous_id = previous["manifest_id"] if previous else None
            self.db_runtime.execute(
                conn,
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
            self.db_runtime.execute(
                conn,
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
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                """
                SELECT m.* FROM manifest_heads h
                JOIN manifests m ON m.manifest_id = h.manifest_id
                WHERE h.repo_name=? AND h.ref_name=?
                """,
                (repo_name, ref_name),
            ).fetchone()
        return self.db_runtime.row_to_dict(row)

    def get_snapshot_manifest(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM manifests
                WHERE snapshot_id=?
                ORDER BY published_at DESC
                LIMIT 1
                """,
                (snapshot_id,),
            ).fetchone()
        return self.db_runtime.row_to_dict(row)
