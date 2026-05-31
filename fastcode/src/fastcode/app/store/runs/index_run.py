"""
Index run tracking and publish retry queue.
"""

from __future__ import annotations

import json
from typing import Any, cast

from fastcode.ports.storage import StoreDatabaseRuntime
from fastcode.utils.clock import SystemClock
from fastcode.utils.ids import PrefixedIdGenerator

from .index_run_contracts import IndexRunRecord, PublishTaskRecord


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

    def __init__(
        self,
        db_runtime: StoreDatabaseRuntime,
        *,
        clock: SystemClock | None = None,
        id_generator: PrefixedIdGenerator | None = None,
    ) -> None:
        self.db_runtime = db_runtime
        self.clock = clock or SystemClock()
        self.id_generator = id_generator or PrefixedIdGenerator()
        self._init_db()

    def _utc_now(self) -> str:
        return self.clock.utc_now()

    def _new_id(self, prefix: str) -> str:
        return self.id_generator.new_id(prefix)

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
                ("index_run_store", "v1", self._utc_now()),
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

        run_id = self._new_id("run")
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
                    self._utc_now(),
                ),
            )
            conn.commit()
        return run_id

    def mark_started(self, run_id: str) -> None:
        self._set_run_fields(run_id, status="running", started_at=self._utc_now())

    def mark_status(self, run_id: str, status: str) -> None:
        self._set_run_fields(run_id, status=status)

    def mark_completed(
        self, run_id: str, status: str = "succeeded", warnings: list[str] | None = None
    ) -> None:
        self._set_run_fields(
            run_id,
            status=status,
            completed_at=self._utc_now(),
            warnings_json=json.dumps(warnings or [], ensure_ascii=False),
        )

    def mark_failed(self, run_id: str, error_message: str) -> None:
        self._set_run_fields(
            run_id,
            status="failed",
            error_message=error_message,
            completed_at=self._utc_now(),
        )

    def enqueue_publish_retry(
        self,
        run_id: str,
        snapshot_id: str,
        manifest_id: str | None,
        error_message: str,
    ) -> str:
        now = self._utc_now()
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

            task_id = self._new_id("pub")
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
                (self._utc_now(), task_id),
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
            now = self._utc_now()
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

    def mark_publish_task_failed(self, task_id: str, error: str) -> None:
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                UPDATE publish_tasks
                SET status='pending', last_error=?, updated_at=?
                WHERE task_id=?
                """,
                (error, self._utc_now(), task_id),
            )
            conn.commit()

    def get_run_record(self, run_id: str) -> IndexRunRecord | None:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn, "SELECT * FROM index_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return self._row_to_run_record(row)

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

    @staticmethod
    def _string_value(value: Any) -> str:
        return str(value or "")

    @staticmethod
    def _optional_string_value(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _int_value(value: Any) -> int:
        if value is None or isinstance(value, bool):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _row_to_run_record(cls, row: Any) -> IndexRunRecord | None:
        raw_run_id = cls._row_value(row, 0, "run_id")
        if raw_run_id is None:
            return None
        return IndexRunRecord(
            run_id=cls._string_value(raw_run_id),
            repo_name=cls._string_value(cls._row_value(row, 1, "repo_name")),
            snapshot_id=cls._string_value(cls._row_value(row, 2, "snapshot_id")),
            branch=cls._optional_string_value(cls._row_value(row, 3, "branch")),
            commit_id=cls._optional_string_value(cls._row_value(row, 4, "commit_id")),
            idempotency_key=cls._optional_string_value(
                cls._row_value(row, 5, "idempotency_key")
            ),
            status=cls._string_value(cls._row_value(row, 6, "status")),
            error_message=cls._optional_string_value(
                cls._row_value(row, 7, "error_message")
            ),
            warnings_json=cls._optional_string_value(
                cls._row_value(row, 8, "warnings_json")
            ),
            created_at=cls._string_value(cls._row_value(row, 9, "created_at")),
            started_at=cls._optional_string_value(
                cls._row_value(row, 10, "started_at")
            ),
            completed_at=cls._optional_string_value(
                cls._row_value(row, 11, "completed_at")
            ),
        )

    @classmethod
    def _row_to_publish_task_record(cls, row: Any) -> PublishTaskRecord | None:
        raw_task_id = cls._row_value(row, 0, "task_id")
        if raw_task_id is None:
            return None
        return PublishTaskRecord(
            task_id=cls._string_value(raw_task_id),
            run_id=cls._string_value(cls._row_value(row, 1, "run_id")),
            snapshot_id=cls._string_value(cls._row_value(row, 2, "snapshot_id")),
            manifest_id=cls._optional_string_value(
                cls._row_value(row, 3, "manifest_id")
            ),
            status=cls._string_value(cls._row_value(row, 4, "status")),
            attempts=cls._int_value(cls._row_value(row, 5, "attempts")),
            last_error=cls._optional_string_value(cls._row_value(row, 6, "last_error")),
            created_at=cls._string_value(cls._row_value(row, 7, "created_at")),
            updated_at=cls._string_value(cls._row_value(row, 8, "updated_at")),
        )
