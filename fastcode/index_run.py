"""
Index run tracking and publish retry queue.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from .db_runtime import DBRuntime
from .utils import utc_now


class IndexRunStore:
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
                CREATE TABLE IF NOT EXISTS index_runs (
                    run_id TEXT PRIMARY KEY,
                    repo_name TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    branch TEXT,
                    commit_id TEXT,
                    idempotency_key TEXT UNIQUE,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    warnings_json TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                )
                """
            )
            self.db_runtime.execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS publish_tasks (
                    task_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    manifest_id TEXT,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(run_id, snapshot_id, manifest_id, status)
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
                ("index_run_store", "v1", utc_now()),
            )
            conn.commit()

    def create_run(
        self,
        repo_name: str,
        snapshot_id: str,
        branch: Optional[str],
        commit_id: Optional[str],
        idempotency_key: Optional[str] = None,
    ) -> str:
        if idempotency_key:
            with self.db_runtime.connect() as conn:
                existing = self.db_runtime.execute(
                    conn,
                    "SELECT run_id FROM index_runs WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    return existing["run_id"]

        run_id = f"run_{uuid.uuid4().hex[:16]}"
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO index_runs (
                    run_id, repo_name, snapshot_id, branch, commit_id, idempotency_key, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, repo_name, snapshot_id, branch, commit_id, idempotency_key, "queued", utc_now()),
            )
            conn.commit()
        return run_id

    def mark_started(self, run_id: str) -> None:
        self._set_run_fields(run_id, status="running", started_at=utc_now())

    def mark_status(self, run_id: str, status: str) -> None:
        self._set_run_fields(run_id, status=status)

    def mark_completed(self, run_id: str, status: str = "succeeded", warnings: Optional[list] = None) -> None:
        self._set_run_fields(
            run_id,
            status=status,
            completed_at=utc_now(),
            warnings_json=json.dumps(warnings or [], ensure_ascii=False),
        )

    def mark_failed(self, run_id: str, error_message: str) -> None:
        self._set_run_fields(run_id, status="failed", error_message=error_message, completed_at=utc_now())

    def enqueue_publish_retry(
        self,
        run_id: str,
        snapshot_id: str,
        manifest_id: Optional[str],
        error_message: str,
    ) -> str:
        now = utc_now()
        with self.db_runtime.connect() as conn:
            existing = self.db_runtime.execute(
                conn,
                """
                SELECT task_id FROM publish_tasks
                WHERE run_id=? AND snapshot_id=?
                  AND ((manifest_id=? ) OR (manifest_id IS NULL AND ? IS NULL))
                  AND status IN ('pending', 'running')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (run_id, snapshot_id, manifest_id, manifest_id),
            ).fetchone()
            if existing:
                self.db_runtime.execute(
                    conn,
                    """
                    UPDATE publish_tasks
                    SET last_error=?, updated_at=?
                    WHERE task_id=?
                    """,
                    (error_message, now, existing["task_id"]),
                )
                conn.commit()
                self._set_run_fields(run_id, status="publish_pending")
                return existing["task_id"]

            task_id = f"pub_{uuid.uuid4().hex[:16]}"
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO publish_tasks (
                    task_id, run_id, snapshot_id, manifest_id, status,
                    attempts, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (task_id, run_id, snapshot_id, manifest_id, "pending", 0, error_message, now, now),
            )
            conn.commit()
        self._set_run_fields(run_id, status="publish_pending")
        return task_id

    def mark_publish_task_done(self, task_id: str) -> None:
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                UPDATE publish_tasks
                SET status='completed', updated_at=?
                WHERE task_id=?
                """,
                (utc_now(), task_id),
            )
            conn.commit()

    def claim_next_publish_task(self) -> Optional[Dict[str, Any]]:
        with self.db_runtime.connect() as conn:
            if self.db_runtime.backend == "postgres":
                row = self.db_runtime.execute(
                    conn,
                    """
                    SELECT * FROM publish_tasks
                    WHERE status='pending'
                    ORDER BY updated_at ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                    """,
                ).fetchone()
            else:
                row = self.db_runtime.execute(
                    conn,
                    """
                    SELECT * FROM publish_tasks
                    WHERE status='pending'
                    ORDER BY updated_at ASC
                    LIMIT 1
                    """,
                ).fetchone()
            if not row:
                return None
            self.db_runtime.execute(
                conn,
                """
                UPDATE publish_tasks
                SET status='running', attempts=attempts+1, updated_at=?
                WHERE task_id=?
                """,
                (utc_now(), row["task_id"]),
            )
            conn.commit()
        return self.db_runtime.row_to_dict(row)

    def mark_publish_task_failed(self, task_id: str, error: str) -> None:
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                UPDATE publish_tasks
                SET status='pending', last_error=?, updated_at=?
                WHERE task_id=?
                """,
                (error, utc_now(), task_id),
            )
            conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(conn, "SELECT * FROM index_runs WHERE run_id=?", (run_id,)).fetchone()
        return self.db_runtime.row_to_dict(row)

    def _set_run_fields(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k}=?" for k in fields.keys())
        values = list(fields.values()) + [run_id]
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(conn, f"UPDATE index_runs SET {assignments} WHERE run_id=?", tuple(values))
            conn.commit()
