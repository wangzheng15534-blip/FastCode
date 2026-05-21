"""
Published manifest storage for branch/ref -> snapshot mapping.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from ..utils.clock import utc_now
from .infrastructure.runtime import DBRuntime
from .records import ManifestRecord


class ManifestStore:
    _MANIFEST_FIELDS = (
        "manifest_id",
        "repo_name",
        "ref_name",
        "snapshot_id",
        "index_run_id",
        "published_at",
        "previous_manifest_id",
        "status",
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
                """,
            )
            self.db_runtime.execute(
                conn,
                """
                CREATE INDEX IF NOT EXISTS idx_manifest_repo_ref_time
                ON manifests (repo_name, ref_name, published_at DESC)
                """,
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
                ("manifest_store", "v1", utc_now()),
            )
            conn.commit()

    def publish(
        self,
        repo_name: str,
        ref_name: str,
        snapshot_id: str,
        index_run_id: str,
        status: str = "published",
    ) -> dict[str, Any]:
        record = self.publish_record(
            repo_name=repo_name,
            ref_name=ref_name,
            snapshot_id=snapshot_id,
            index_run_id=index_run_id,
            status=status,
        )
        return self._manifest_payload(record)

    def publish_record(
        self,
        repo_name: str,
        ref_name: str,
        snapshot_id: str,
        index_run_id: str,
        status: str = "published",
    ) -> ManifestRecord:
        manifest_id = f"manifest_{uuid.uuid4().hex[:16]}"
        now = utc_now()

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
            previous = self._row_to_manifest_record(previous_row)
            previous_id = previous.manifest_id if previous else None
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

        return ManifestRecord(
            manifest_id=manifest_id,
            repo_name=repo_name,
            ref_name=ref_name,
            snapshot_id=snapshot_id,
            index_run_id=index_run_id,
            published_at=now,
            previous_manifest_id=previous_id,
            status=status,
        )

    def get_branch_manifest(
        self, repo_name: str, ref_name: str
    ) -> dict[str, Any] | None:
        record = self.get_branch_manifest_record(repo_name, ref_name)
        return self._manifest_payload(record) if record is not None else None

    def get_branch_manifest_record(
        self, repo_name: str, ref_name: str
    ) -> ManifestRecord | None:
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
        return self._row_to_manifest_record(row)

    def get_snapshot_manifest(self, snapshot_id: str) -> dict[str, Any] | None:
        record = self.get_snapshot_manifest_record(snapshot_id)
        return self._manifest_payload(record) if record is not None else None

    def get_snapshot_manifest_record(self, snapshot_id: str) -> ManifestRecord | None:
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
        return self._row_to_manifest_record(row)

    def _row_to_manifest_record(self, row: Any) -> ManifestRecord | None:
        manifest_id = self._row_value(row, 0, "manifest_id")
        if manifest_id is None:
            return None
        return ManifestRecord(
            manifest_id=str(manifest_id),
            repo_name=str(self._row_value(row, 1, "repo_name") or ""),
            ref_name=str(self._row_value(row, 2, "ref_name") or ""),
            snapshot_id=str(self._row_value(row, 3, "snapshot_id") or ""),
            index_run_id=str(self._row_value(row, 4, "index_run_id") or ""),
            published_at=str(self._row_value(row, 5, "published_at") or ""),
            previous_manifest_id=(
                str(previous_manifest_id)
                if (
                    previous_manifest_id := self._row_value(
                        row, 6, "previous_manifest_id"
                    )
                )
                is not None
                else None
            ),
            status=str(self._row_value(row, 7, "status") or ""),
        )

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
    def _manifest_payload(cls, record: ManifestRecord) -> dict[str, Any]:
        return {
            field_name: getattr(record, field_name)
            for field_name in cls._MANIFEST_FIELDS
        }
