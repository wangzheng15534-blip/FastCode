"""Persistent per-unit artifact metadata for graceful incremental updates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

from ..db_runtime import DBRuntime
from ..utils import utc_now
from .records import UnitArtifactRecord


class UnitArtifactStore:
    _EXTRA_COLUMNS: ClassVar[dict[str, str]] = {
        "embedding_artifact_ref": "TEXT",
        "scoped_tool_ref": "TEXT",
        "package_root": "TEXT",
        "repair_frontier_summary": "TEXT",
    }
    _UNIT_ARTIFACT_FIELDS: ClassVar[tuple[str, ...]] = (
        "snapshot_id",
        "stable_unit_id",
        "relative_path",
        "unit_type",
        "content_hash",
        "syntax_hash",
        "signature_hash",
        "edge_surface_hash",
        "embedding_text_hash",
        "api_surface_hash",
        "embedding_artifact_ref",
        "scoped_tool_ref",
        "package_root",
        "repair_frontier_summary",
        "metadata_json",
        "created_at",
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
                CREATE TABLE IF NOT EXISTS unit_artifacts (
                    snapshot_id TEXT NOT NULL,
                    stable_unit_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    unit_type TEXT NOT NULL,
                    content_hash TEXT,
                    syntax_hash TEXT,
                    signature_hash TEXT,
                    edge_surface_hash TEXT,
                    embedding_text_hash TEXT,
                    api_surface_hash TEXT,
                    embedding_artifact_ref TEXT,
                    scoped_tool_ref TEXT,
                    package_root TEXT,
                    repair_frontier_summary TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (snapshot_id, stable_unit_id)
                )
                """,
            )
            self.db_runtime.execute(
                conn,
                """
                CREATE INDEX IF NOT EXISTS idx_unit_artifacts_path
                ON unit_artifacts (snapshot_id, relative_path)
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
                ("unit_artifact_store", "v1", utc_now()),
            )
            self._ensure_extra_columns(conn)
            conn.commit()

    def _column_exists(self, conn: Any, table: str, column: str) -> bool:
        if self.db_runtime.backend == "postgres":
            row = self.db_runtime.execute(
                conn,
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_name=? AND column_name=?
                LIMIT 1
                """,
                (table, column),
            ).fetchone()
            return bool(row)

        rows = self.db_runtime.execute(conn, f"PRAGMA table_info({table})").fetchall()
        return any(self._row_value(row, 1, "name") == column for row in rows)

    def _ensure_extra_columns(self, conn: Any) -> None:
        for column, column_type in self._EXTRA_COLUMNS.items():
            if self._column_exists(conn, "unit_artifacts", column):
                continue
            self.db_runtime.execute(
                conn,
                f"ALTER TABLE unit_artifacts ADD COLUMN {column} {column_type}",
            )

    @staticmethod
    def _text_ref(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

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
    def _metadata_mapping(cls, elem: dict[str, Any]) -> dict[str, Any]:
        raw_metadata = elem.get("metadata")
        return (
            cast(dict[str, Any], raw_metadata) if isinstance(raw_metadata, dict) else {}
        )

    @classmethod
    def _json_safe_value(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): cls._json_safe_value(item)
                for key, item in cast(Mapping[Any, Any], value).items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._json_safe_value(item) for item in cast(Sequence[Any], value)]
        if isinstance(value, set):
            return [cls._json_safe_value(item) for item in cast(set[Any], value)]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return repr(value)

    @classmethod
    def _serialize_metadata_json(cls, metadata: dict[str, Any]) -> str:
        return json.dumps(
            cls._json_safe_value(metadata),
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _deserialize_metadata_json(raw_metadata: Any) -> dict[str, Any]:
        if raw_metadata is None:
            return {}
        if isinstance(raw_metadata, dict):
            return cast(dict[str, Any], raw_metadata)
        try:
            metadata = json.loads(str(raw_metadata))
        except (json.JSONDecodeError, TypeError):
            return {}
        return cast(dict[str, Any], metadata) if isinstance(metadata, dict) else {}

    @classmethod
    def _unit_record_from_element(
        cls,
        snapshot_id: str,
        elem: dict[str, Any],
    ) -> UnitArtifactRecord | None:
        metadata = cls._metadata_mapping(elem)
        stable_unit_id = str(metadata.get("stable_unit_id") or "")
        if not stable_unit_id:
            return None
        embedding_artifact_ref = cls._text_ref(
            elem.get("embedding_artifact_ref") or metadata.get("embedding_artifact_ref")
        )
        scoped_tool_ref = cls._text_ref(
            elem.get("scoped_tool_ref") or metadata.get("scoped_tool_ref")
        )
        package_root = cls._text_ref(
            elem.get("package_root") or metadata.get("package_root")
        )
        repair_frontier_summary = cls._text_ref(
            elem.get("repair_frontier_summary")
            or metadata.get("repair_frontier_summary")
        )
        return UnitArtifactRecord(
            snapshot_id=snapshot_id,
            stable_unit_id=stable_unit_id,
            relative_path=str(elem.get("relative_path") or elem.get("file_path") or ""),
            unit_type=str(elem.get("type") or ""),
            content_hash=cls._optional_text(
                metadata.get("content_hash") or elem.get("content_hash")
            ),
            syntax_hash=cls._optional_text(
                metadata.get("syntax_hash") or elem.get("syntax_hash")
            ),
            signature_hash=cls._optional_text(
                metadata.get("signature_hash") or elem.get("signature_hash")
            ),
            edge_surface_hash=cls._optional_text(
                metadata.get("edge_surface_hash") or elem.get("edge_surface_hash")
            ),
            embedding_text_hash=cls._optional_text(
                metadata.get("embedding_text_hash") or elem.get("embedding_text_hash")
            ),
            api_surface_hash=cls._optional_text(
                metadata.get("api_surface_hash") or elem.get("api_surface_hash")
            ),
            embedding_artifact_ref=embedding_artifact_ref,
            scoped_tool_ref=scoped_tool_ref,
            package_root=package_root,
            repair_frontier_summary=repair_frontier_summary,
            metadata_json=cls._serialize_metadata_json(metadata),
            created_at=utc_now(),
        )

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    def _insert_unit_record(self, conn: Any, record: UnitArtifactRecord) -> None:
        self.db_runtime.execute(
            conn,
            """
            INSERT INTO unit_artifacts (
                snapshot_id, stable_unit_id, relative_path, unit_type,
                content_hash, syntax_hash, signature_hash, edge_surface_hash,
                embedding_text_hash, api_surface_hash, embedding_artifact_ref,
                scoped_tool_ref, package_root, repair_frontier_summary,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_id, stable_unit_id) DO UPDATE SET
                relative_path=excluded.relative_path,
                unit_type=excluded.unit_type,
                content_hash=excluded.content_hash,
                syntax_hash=excluded.syntax_hash,
                signature_hash=excluded.signature_hash,
                edge_surface_hash=excluded.edge_surface_hash,
                embedding_text_hash=excluded.embedding_text_hash,
                api_surface_hash=excluded.api_surface_hash,
                embedding_artifact_ref=excluded.embedding_artifact_ref,
                scoped_tool_ref=excluded.scoped_tool_ref,
                package_root=excluded.package_root,
                repair_frontier_summary=excluded.repair_frontier_summary,
                metadata_json=excluded.metadata_json,
                created_at=excluded.created_at
            """,
            (
                record.snapshot_id,
                record.stable_unit_id,
                record.relative_path,
                record.unit_type,
                record.content_hash,
                record.syntax_hash,
                record.signature_hash,
                record.edge_surface_hash,
                record.embedding_text_hash,
                record.api_surface_hash,
                record.embedding_artifact_ref,
                record.scoped_tool_ref,
                record.package_root,
                record.repair_frontier_summary,
                record.metadata_json,
                record.created_at,
            ),
        )

    def _insert_unit(
        self,
        conn: Any,
        snapshot_id: str,
        elem: dict[str, Any],
    ) -> None:
        record = self._unit_record_from_element(snapshot_id, elem)
        if record is None:
            return
        self._insert_unit_record(conn, record)

    def replace_snapshot_units(
        self,
        snapshot_id: str,
        *,
        elements: list[dict[str, Any]],
    ) -> None:
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                "DELETE FROM unit_artifacts WHERE snapshot_id=?",
                (snapshot_id,),
            )
            for elem in elements:
                self._insert_unit(conn, snapshot_id, elem)
            conn.commit()

    def publish_snapshot_units_delta(
        self,
        snapshot_id: str,
        *,
        previous_snapshot_id: str,
        changed_paths: list[str],
        removed_paths: list[str],
        elements: list[dict[str, Any]],
    ) -> dict[str, int | str]:
        excluded_paths = sorted({str(path) for path in changed_paths + removed_paths})
        excluded_path_set = set(excluded_paths)
        previous_records = [
            record
            for record in self.list_snapshot_unit_records(previous_snapshot_id)
            if record.relative_path not in excluded_path_set
        ]
        copied = 0
        with self.db_runtime.connect() as conn:
            for record in previous_records:
                self._insert_unit_record(
                    conn,
                    UnitArtifactRecord(
                        snapshot_id=snapshot_id,
                        stable_unit_id=record.stable_unit_id,
                        relative_path=record.relative_path,
                        unit_type=record.unit_type,
                        content_hash=record.content_hash,
                        syntax_hash=record.syntax_hash,
                        signature_hash=record.signature_hash,
                        edge_surface_hash=record.edge_surface_hash,
                        embedding_text_hash=record.embedding_text_hash,
                        api_surface_hash=record.api_surface_hash,
                        embedding_artifact_ref=record.embedding_artifact_ref,
                        scoped_tool_ref=record.scoped_tool_ref,
                        package_root=record.package_root,
                        repair_frontier_summary=record.repair_frontier_summary,
                        metadata_json=record.metadata_json,
                        created_at=utc_now(),
                    ),
                )
                copied += 1
            for elem in elements:
                self._insert_unit(conn, snapshot_id, elem)
            conn.commit()
        return {
            "mode": "delta",
            "previous_snapshot_id": previous_snapshot_id,
            "copied_rows": copied,
            "changed_rows": len(elements),
            "excluded_path_count": len(excluded_paths),
        }

    def refresh_units(
        self,
        snapshot_id: str,
        *,
        stable_unit_ids: list[str],
        elements: list[dict[str, Any]],
    ) -> None:
        requested_ids = {
            str(stable_unit_id) for stable_unit_id in stable_unit_ids if stable_unit_id
        }
        if not requested_ids:
            return
        with self.db_runtime.connect() as conn:
            for stable_unit_id in sorted(requested_ids):
                self.db_runtime.execute(
                    conn,
                    """
                    DELETE FROM unit_artifacts
                    WHERE snapshot_id=? AND stable_unit_id=?
                    """,
                    (snapshot_id, stable_unit_id),
                )
            for elem in elements:
                metadata = self._metadata_mapping(elem)
                stable_unit_id = str(metadata.get("stable_unit_id") or "")
                if stable_unit_id in requested_ids:
                    self._insert_unit(conn, snapshot_id, elem)
            conn.commit()

    def list_snapshot_unit_records(self, snapshot_id: str) -> list[UnitArtifactRecord]:
        with self.db_runtime.connect() as conn:
            rows = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM unit_artifacts
                WHERE snapshot_id=?
                ORDER BY relative_path ASC, stable_unit_id ASC
                """,
                (snapshot_id,),
            ).fetchall()
        return [
            record
            for row in rows
            if (record := self._unit_record_from_row(row)) is not None
        ]

    @classmethod
    def _unit_record_from_row(cls, row: Any) -> UnitArtifactRecord | None:
        stable_unit_id = cls._row_value(row, 1, "stable_unit_id")
        if stable_unit_id is None:
            return None
        return UnitArtifactRecord(
            snapshot_id=str(cls._row_value(row, 0, "snapshot_id") or ""),
            stable_unit_id=str(stable_unit_id),
            relative_path=str(cls._row_value(row, 2, "relative_path") or ""),
            unit_type=str(cls._row_value(row, 3, "unit_type") or ""),
            content_hash=cls._optional_text(cls._row_value(row, 4, "content_hash")),
            syntax_hash=cls._optional_text(cls._row_value(row, 5, "syntax_hash")),
            signature_hash=cls._optional_text(cls._row_value(row, 6, "signature_hash")),
            edge_surface_hash=cls._optional_text(
                cls._row_value(row, 7, "edge_surface_hash")
            ),
            embedding_text_hash=cls._optional_text(
                cls._row_value(row, 8, "embedding_text_hash")
            ),
            api_surface_hash=cls._optional_text(
                cls._row_value(row, 9, "api_surface_hash")
            ),
            embedding_artifact_ref=cls._optional_text(
                cls._row_value(row, 10, "embedding_artifact_ref")
            ),
            scoped_tool_ref=cls._optional_text(
                cls._row_value(row, 11, "scoped_tool_ref")
            ),
            package_root=cls._optional_text(cls._row_value(row, 12, "package_root")),
            repair_frontier_summary=cls._optional_text(
                cls._row_value(row, 13, "repair_frontier_summary")
            ),
            metadata_json=cls._optional_text(cls._row_value(row, 14, "metadata_json")),
            created_at=str(cls._row_value(row, 15, "created_at") or ""),
        )

    @classmethod
    def _unit_payload_from_record(
        cls,
        record: UnitArtifactRecord,
    ) -> dict[str, Any]:
        return {
            "snapshot_id": record.snapshot_id,
            "stable_unit_id": record.stable_unit_id,
            "relative_path": record.relative_path,
            "unit_type": record.unit_type,
            "content_hash": record.content_hash,
            "syntax_hash": record.syntax_hash,
            "signature_hash": record.signature_hash,
            "edge_surface_hash": record.edge_surface_hash,
            "embedding_text_hash": record.embedding_text_hash,
            "api_surface_hash": record.api_surface_hash,
            "embedding_artifact_ref": record.embedding_artifact_ref,
            "scoped_tool_ref": record.scoped_tool_ref,
            "package_root": record.package_root,
            "repair_frontier_summary": record.repair_frontier_summary,
            "created_at": record.created_at,
            "metadata": cls._deserialize_metadata_json(record.metadata_json),
        }

    def list_snapshot_units(self, snapshot_id: str) -> list[dict[str, Any]]:
        return [
            self._unit_payload_from_record(record)
            for record in self.list_snapshot_unit_records(snapshot_id)
        ]
