"""
Index run tracking and publish retry queue.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IndexRunStore:
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
            conn.execute(
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
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT run_id FROM index_runs WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    return existing["run_id"]

        run_id = f"run_{uuid.uuid4().hex[:16]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO index_runs (
                    run_id, repo_name, snapshot_id, branch, commit_id, idempotency_key, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, repo_name, snapshot_id, branch, commit_id, idempotency_key, "queued", _utc_now()),
            )
            conn.commit()
        return run_id

    def mark_started(self, run_id: str) -> None:
        self._set_run_fields(run_id, status="running", started_at=_utc_now())

    def mark_status(self, run_id: str, status: str) -> None:
        self._set_run_fields(run_id, status=status)

    def mark_completed(self, run_id: str, status: str = "succeeded", warnings: Optional[list] = None) -> None:
        self._set_run_fields(
            run_id,
            status=status,
            completed_at=_utc_now(),
            warnings_json=json.dumps(warnings or [], ensure_ascii=False),
        )

    def mark_failed(self, run_id: str, error_message: str) -> None:
        self._set_run_fields(run_id, status="failed", error_message=error_message, completed_at=_utc_now())

    def enqueue_publish_retry(
        self,
        run_id: str,
        snapshot_id: str,
        manifest_id: Optional[str],
        error_message: str,
    ) -> str:
        now = _utc_now()
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT task_id FROM publish_tasks
                WHERE run_id=? AND snapshot_id=? AND manifest_id IS ?
                  AND status IN ('pending', 'running')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (run_id, snapshot_id, manifest_id),
            ).fetchone()
            if existing:
                conn.execute(
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
            conn.execute(
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
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE publish_tasks
                SET status='completed', updated_at=?
                WHERE task_id=?
                """,
                (_utc_now(), task_id),
            )
            conn.commit()

    def claim_next_publish_task(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM publish_tasks
                WHERE status='pending'
                ORDER BY updated_at ASC
                LIMIT 1
                """
            ).fetchone()
            if not row:
                return None
            conn.execute(
                """
                UPDATE publish_tasks
                SET status='running', attempts=attempts+1, updated_at=?
                WHERE task_id=?
                """,
                (_utc_now(), row["task_id"]),
            )
            conn.commit()
        return dict(row)

    def mark_publish_task_failed(self, task_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE publish_tasks
                SET status='pending', last_error=?, updated_at=?
                WHERE task_id=?
                """,
                (error, _utc_now(), task_id),
            )
            conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM index_runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def _set_run_fields(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k}=?" for k in fields.keys())
        values = list(fields.values()) + [run_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE index_runs SET {assignments} WHERE run_id=?", values)
            conn.commit()
