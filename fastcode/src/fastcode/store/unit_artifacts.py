"""Persistent per-unit artifact metadata for graceful incremental updates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

from ..db_runtime import DBRuntime
from ..utils import utc_now


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

    def _insert_unit(
        self,
        conn: Any,
        snapshot_id: str,
        elem: dict[str, Any],
    ) -> None:
        metadata = self._metadata_mapping(elem)
        stable_unit_id = str(metadata.get("stable_unit_id") or "")
        if not stable_unit_id:
            return
        embedding_artifact_ref = self._text_ref(
            elem.get("embedding_artifact_ref") or metadata.get("embedding_artifact_ref")
        )
        scoped_tool_ref = self._text_ref(
            elem.get("scoped_tool_ref") or metadata.get("scoped_tool_ref")
        )
        package_root = self._text_ref(
            elem.get("package_root") or metadata.get("package_root")
        )
        repair_frontier_summary = self._text_ref(
            elem.get("repair_frontier_summary")
            or metadata.get("repair_frontier_summary")
        )
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
                snapshot_id,
                stable_unit_id,
                str(elem.get("relative_path") or elem.get("file_path") or ""),
                str(elem.get("type") or ""),
                metadata.get("content_hash") or elem.get("content_hash"),
                metadata.get("syntax_hash") or elem.get("syntax_hash"),
                metadata.get("signature_hash") or elem.get("signature_hash"),
                metadata.get("edge_surface_hash") or elem.get("edge_surface_hash"),
                metadata.get("embedding_text_hash") or elem.get("embedding_text_hash"),
                metadata.get("api_surface_hash") or elem.get("api_surface_hash"),
                embedding_artifact_ref,
                scoped_tool_ref,
                package_root,
                repair_frontier_summary,
                self._serialize_metadata_json(metadata),
                utc_now(),
            ),
        )

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

    def list_snapshot_units(self, snapshot_id: str) -> list[dict[str, Any]]:
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
            payload
            for row in rows
            if (payload := self._unit_payload_from_row(row)) is not None
        ]

    @classmethod
    def _unit_payload_from_row(cls, row: Any) -> dict[str, Any] | None:
        stable_unit_id = cls._row_value(row, 1, "stable_unit_id")
        if stable_unit_id is None:
            return None
        payload = {
            field_name: cls._row_value(row, index, field_name)
            for index, field_name in enumerate(cls._UNIT_ARTIFACT_FIELDS)
        }
        metadata_json = payload.pop("metadata_json", None)
        payload["metadata"] = cls._deserialize_metadata_json(metadata_json)
        return payload
