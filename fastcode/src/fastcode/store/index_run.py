"""
Index run tracking and publish retry queue.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, cast

from ..db_runtime import DBRuntime
from ..utils import utc_now
from .records import IndexRunRecord, PublishTaskRecord


class IndexRunStore:
    _RUN_FIELDS = (
        "run_id",
        "repo_name",
        "snapshot_id",
        "branch",
        "commit_id",
        "idempotency_key",
        "status",
        "error_message",
        "warnings_json",
        "created_at",
        "started_at",
        "completed_at",
    )
    _PUBLISH_TASK_FIELDS = (
        "task_id",
        "run_id",
        "snapshot_id",
        "manifest_id",
        "status",
        "attempts",
        "last_error",
        "created_at",
        "updated_at",
    )

    def __init__(self, db_path_or_runtime: str | DBRuntime) -> None:
        if isinstance(db_path_or_runtime, DBRuntime):
            self.db_runtime = db_path_or_runtime
        else:
            self.db_runtime = DBRuntime(
                backend="sqlite", sqlite_path=db_path_or_runtime
            )
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
                """,
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
                """,
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
        branch: str | None,
        commit_id: str | None,
        idempotency_key: str | None = None,
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
                (
                    run_id,
                    repo_name,
                    snapshot_id,
                    branch,
                    commit_id,
                    idempotency_key,
                    "queued",
                    utc_now(),
                ),
            )
            conn.commit()
        return run_id

    def mark_started(self, run_id: str) -> None:
        self._set_run_fields(run_id, status="running", started_at=utc_now())

    def mark_status(self, run_id: str, status: str) -> None:
        self._set_run_fields(run_id, status=status)

    def mark_completed(
        self, run_id: str, status: str = "succeeded", warnings: list[str] | None = None
    ) -> None:
        self._set_run_fields(
            run_id,
            status=status,
            completed_at=utc_now(),
            warnings_json=json.dumps(warnings or [], ensure_ascii=False),
        )

    def mark_failed(self, run_id: str, error_message: str) -> None:
        self._set_run_fields(
            run_id, status="failed", error_message=error_message, completed_at=utc_now()
        )

    def enqueue_publish_retry(
        self,
        run_id: str,
        snapshot_id: str,
        manifest_id: str | None,
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
                (
                    task_id,
                    run_id,
                    snapshot_id,
                    manifest_id,
                    "pending",
                    0,
                    error_message,
                    now,
                    now,
                ),
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

    def claim_next_publish_task_record(self) -> PublishTaskRecord | None:
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
            task = self._row_to_publish_task_record(row)
            if task is None:
                return None
            now = utc_now()
            self.db_runtime.execute(
                conn,
                """
                UPDATE publish_tasks
                SET status='running', attempts=attempts+1, updated_at=?
                WHERE task_id=?
                """,
                (now, task.task_id),
            )
            conn.commit()
        return PublishTaskRecord(
            task_id=task.task_id,
            run_id=task.run_id,
            snapshot_id=task.snapshot_id,
            manifest_id=task.manifest_id,
            status="running",
            attempts=task.attempts + 1,
            last_error=task.last_error,
            created_at=task.created_at,
            updated_at=now,
        )

    def claim_next_publish_task(self) -> dict[str, Any] | None:
        task = self.claim_next_publish_task_record()
        return self._publish_task_payload(task) if task is not None else None

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

    def get_run_record(self, run_id: str) -> IndexRunRecord | None:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn, "SELECT * FROM index_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return self._row_to_run_record(row)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        record = self.get_run_record(run_id)
        return self._run_payload(record) if record is not None else None

    def get_latest_run_record(self) -> IndexRunRecord | None:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM index_runs
                ORDER BY created_at DESC, run_id DESC
                LIMIT 1
                """,
            ).fetchone()
        return self._row_to_run_record(row)

    def get_latest_run(self) -> dict[str, Any] | None:
        record = self.get_latest_run_record()
        return self._run_payload(record) if record is not None else None

    _ALLOWED_RUN_FIELDS = frozenset(
        {
            "status",
            "started_at",
            "completed_at",
            "error_message",
            "warnings_json",
        }
    )

    def _set_run_fields(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        unknown = set(fields.keys()) - self._ALLOWED_RUN_FIELDS
        if unknown:
            raise ValueError(f"Unknown run fields: {unknown}")
        assignments = ", ".join(f"{k}=?" for k in fields)
        values = [*list(fields.values()), run_id]
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                f"UPDATE index_runs SET {assignments} WHERE run_id=?",
                tuple(values),
            )
            conn.commit()

    @staticmethod
    def _row_value(row: Any, index: int, key: str) -> Any:
        if row is None:
            return None
        if isinstance(row, dict):
            return cast(dict[str, Any], row).get(key)
        try:
            return row[key]
        except (IndexError, KeyError, TypeError):
            try:
                return row[index]
            except (IndexError, KeyError, TypeError):
                return None

    @classmethod
    def _payload_from_row(
        cls, row: Any, fields: tuple[str, ...]
    ) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = {
            field_name: cls._row_value(row, index, field_name)
            for index, field_name in enumerate(fields)
        }
        first_field = fields[0]
        return payload if payload.get(first_field) is not None else None

    @classmethod
    def _row_to_run_record(cls, row: Any) -> IndexRunRecord | None:
        payload = cls._payload_from_row(row, cls._RUN_FIELDS)
        return IndexRunRecord.from_dict(payload) if payload is not None else None

    @classmethod
    def _row_to_publish_task_record(cls, row: Any) -> PublishTaskRecord | None:
        payload = cls._payload_from_row(row, cls._PUBLISH_TASK_FIELDS)
        return PublishTaskRecord.from_dict(payload) if payload is not None else None

    @staticmethod
    def _run_payload(record: IndexRunRecord) -> dict[str, Any]:
        return {
            "run_id": record.run_id,
            "repo_name": record.repo_name,
            "snapshot_id": record.snapshot_id,
            "branch": record.branch,
            "commit_id": record.commit_id,
            "idempotency_key": record.idempotency_key,
            "status": record.status,
            "error_message": record.error_message,
            "warnings_json": record.warnings_json,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "completed_at": record.completed_at,
        }

    @staticmethod
    def _publish_task_payload(record: PublishTaskRecord) -> dict[str, Any]:
        return {
            "task_id": record.task_id,
            "run_id": record.run_id,
            "snapshot_id": record.snapshot_id,
            "manifest_id": record.manifest_id,
            "status": record.status,
            "attempts": record.attempts,
            "last_error": record.last_error,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }
