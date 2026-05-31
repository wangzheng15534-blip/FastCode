"""Persistent per-unit artifact metadata for graceful incremental updates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

from fastcode.ports.storage import StoreDatabaseRuntime
from fastcode.utils.clock import SystemClock
from fastcode.utils.filesystem import normalize_path

from .unit_contracts import FileIRShardRecord, UnitArtifactRecord


class UnitArtifactStore:
    FILE_IR_SHARD_SCHEMA_VERSION: ClassVar[str] = "fastcode.file_ir_shard.v1"
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
    _FILE_IR_SHARD_FIELDS: ClassVar[tuple[str, ...]] = (
        "snapshot_id",
        "relative_path",
        "schema_version",
        "payload_json",
        "unit_count",
        "support_count",
        "relation_count",
        "embedding_count",
        "content_hash",
        "created_at",
    )

    def __init__(
        self,
        db_runtime: StoreDatabaseRuntime,
        *,
        clock: SystemClock | None = None,
    ) -> None:
        self.db_runtime = db_runtime
        self.clock = clock or SystemClock()
        self._init_db()

    def _utc_now(self) -> str:
        return self.clock.utc_now()

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
                CREATE TABLE IF NOT EXISTS file_ir_shards (
                    snapshot_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    unit_count INTEGER NOT NULL,
                    support_count INTEGER NOT NULL,
                    relation_count INTEGER NOT NULL,
                    embedding_count INTEGER NOT NULL,
                    content_hash TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (snapshot_id, relative_path)
                )
                """,
            )
            self.db_runtime.execute(
                conn,
                """
                CREATE INDEX IF NOT EXISTS idx_file_ir_shards_path
                ON file_ir_shards (snapshot_id, relative_path)
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
                ("unit_artifact_store", "v1", self._utc_now()),
            )
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO schema_migrations (component, version, applied_at)
                VALUES (?, ?, ?)
                ON CONFLICT(component, version) DO NOTHING
                """,
                ("unit_artifact_store", "v2_file_ir_shards", self._utc_now()),
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
        *,
        created_at: str,
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
            relative_path=normalize_path(
                str(elem.get("relative_path") or elem.get("file_path") or "")
            ),
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
            created_at=created_at,
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
        record = self._unit_record_from_element(
            snapshot_id,
            elem,
            created_at=self._utc_now(),
        )
        if record is None:
            return
        self._insert_unit_record(conn, record)

    @classmethod
    def _mapping_list_payload(cls, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, (list, tuple)):
            return []
        payload: list[dict[str, Any]] = []
        for item in cast(Sequence[Any], value):
            if isinstance(item, Mapping):
                payload.append(
                    cast(
                        dict[str, Any],
                        cls._json_safe_value(
                            {
                                str(key): sub_item
                                for key, sub_item in cast(
                                    Mapping[Any, Any], item
                                ).items()
                            }
                        ),
                    )
                )
        return payload

    @classmethod
    def _file_ir_content_hash(
        cls,
        *,
        relative_path: str,
        explicit_hash: Any,
        units: Sequence[Mapping[str, Any]],
    ) -> str | None:
        if explicit_hash is not None:
            return str(explicit_hash)
        for unit in units:
            unit_path = normalize_path(str(unit.get("path") or ""))
            if unit_path != relative_path:
                continue
            metadata = unit.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            metadata_payload = cast(Mapping[str, Any], metadata)
            content_hash = metadata_payload.get("content_hash") or metadata_payload.get(
                "blob_oid"
            )
            if content_hash is not None:
                return str(content_hash)
        return None

    @classmethod
    def _file_ir_record_from_payload(
        cls,
        snapshot_id: str,
        shard: Mapping[str, Any],
        *,
        created_at: str,
    ) -> FileIRShardRecord | None:
        relative_path = normalize_path(
            str(shard.get("relative_path") or shard.get("path") or "")
        )
        if not relative_path:
            return None
        units = cls._mapping_list_payload(shard.get("units"))
        supports = cls._mapping_list_payload(shard.get("supports"))
        relations = cls._mapping_list_payload(shard.get("relations"))
        embeddings = cls._mapping_list_payload(shard.get("embeddings"))
        content_hash = cls._file_ir_content_hash(
            relative_path=relative_path,
            explicit_hash=shard.get("content_hash"),
            units=units,
        )
        envelope = {
            "schema_version": cls.FILE_IR_SHARD_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "relative_path": relative_path,
            "content_hash": content_hash,
            "units": units,
            "supports": supports,
            "relations": relations,
            "embeddings": embeddings,
        }
        return FileIRShardRecord(
            snapshot_id=snapshot_id,
            relative_path=relative_path,
            schema_version=cls.FILE_IR_SHARD_SCHEMA_VERSION,
            payload_json=json.dumps(
                cls._json_safe_value(envelope),
                ensure_ascii=False,
                sort_keys=True,
            ),
            unit_count=len(units),
            support_count=len(supports),
            relation_count=len(relations),
            embedding_count=len(embeddings),
            content_hash=content_hash,
            created_at=created_at,
        )

    def _insert_file_ir_shard_record(
        self,
        conn: Any,
        record: FileIRShardRecord,
    ) -> None:
        self.db_runtime.execute(
            conn,
            """
            INSERT INTO file_ir_shards (
                snapshot_id, relative_path, schema_version, payload_json,
                unit_count, support_count, relation_count, embedding_count,
                content_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_id, relative_path) DO UPDATE SET
                schema_version=excluded.schema_version,
                payload_json=excluded.payload_json,
                unit_count=excluded.unit_count,
                support_count=excluded.support_count,
                relation_count=excluded.relation_count,
                embedding_count=excluded.embedding_count,
                content_hash=excluded.content_hash,
                created_at=excluded.created_at
            """,
            (
                record.snapshot_id,
                record.relative_path,
                record.schema_version,
                record.payload_json,
                record.unit_count,
                record.support_count,
                record.relation_count,
                record.embedding_count,
                record.content_hash,
                record.created_at,
            ),
        )

    @classmethod
    def _file_ir_shard_record_from_row(cls, row: Any) -> FileIRShardRecord | None:
        relative_path = cls._row_value(row, 1, "relative_path")
        if relative_path is None:
            return None
        return FileIRShardRecord(
            snapshot_id=str(cls._row_value(row, 0, "snapshot_id") or ""),
            relative_path=normalize_path(str(relative_path)),
            schema_version=str(cls._row_value(row, 2, "schema_version") or ""),
            payload_json=str(cls._row_value(row, 3, "payload_json") or "{}"),
            unit_count=int(cls._row_value(row, 4, "unit_count") or 0),
            support_count=int(cls._row_value(row, 5, "support_count") or 0),
            relation_count=int(cls._row_value(row, 6, "relation_count") or 0),
            embedding_count=int(cls._row_value(row, 7, "embedding_count") or 0),
            content_hash=cls._optional_text(cls._row_value(row, 8, "content_hash")),
            created_at=str(cls._row_value(row, 9, "created_at") or ""),
        )

    @classmethod
    def _retarget_file_ir_shard_record(
        cls,
        record: FileIRShardRecord,
        *,
        snapshot_id: str,
        created_at: str,
    ) -> FileIRShardRecord:
        payload = cls._deserialize_metadata_json(record.payload_json)
        payload.update(
            {
                "schema_version": record.schema_version,
                "snapshot_id": snapshot_id,
                "relative_path": record.relative_path,
                "content_hash": record.content_hash,
            }
        )
        return FileIRShardRecord(
            snapshot_id=snapshot_id,
            relative_path=record.relative_path,
            schema_version=record.schema_version,
            payload_json=json.dumps(
                cls._json_safe_value(payload),
                ensure_ascii=False,
                sort_keys=True,
            ),
            unit_count=record.unit_count,
            support_count=record.support_count,
            relation_count=record.relation_count,
            embedding_count=record.embedding_count,
            content_hash=record.content_hash,
            created_at=created_at,
        )

    @classmethod
    def _file_ir_shard_payload_from_record(
        cls,
        record: FileIRShardRecord,
    ) -> dict[str, Any]:
        payload = cls._deserialize_metadata_json(record.payload_json)
        payload["snapshot_id"] = record.snapshot_id
        payload["relative_path"] = record.relative_path
        payload["schema_version"] = record.schema_version
        payload["content_hash"] = record.content_hash
        payload["created_at"] = record.created_at
        payload["counts"] = {
            "units": record.unit_count,
            "supports": record.support_count,
            "relations": record.relation_count,
            "embeddings": record.embedding_count,
        }
        return payload

    def replace_snapshot_file_ir_shards(
        self,
        snapshot_id: str,
        *,
        shards: Sequence[Mapping[str, Any]],
    ) -> dict[str, int | str]:
        written = 0
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                "DELETE FROM file_ir_shards WHERE snapshot_id=?",
                (snapshot_id,),
            )
            for shard in shards:
                record = self._file_ir_record_from_payload(
                    snapshot_id,
                    shard,
                    created_at=self._utc_now(),
                )
                if record is None:
                    continue
                self._insert_file_ir_shard_record(conn, record)
                written += 1
            conn.commit()
        return {"mode": "full", "written_shards": written}

    def publish_snapshot_file_ir_shards_delta(
        self,
        snapshot_id: str,
        *,
        previous_snapshot_id: str,
        changed_paths: Sequence[str],
        removed_paths: Sequence[str],
        shards: Sequence[Mapping[str, Any]],
        reused_shards: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, int | str]:
        changed_path_set = {normalize_path(str(path)) for path in changed_paths if path}
        removed_path_set = {normalize_path(str(path)) for path in removed_paths if path}
        excluded_path_set = changed_path_set | removed_path_set
        reused_records: list[FileIRShardRecord] = []
        for shard in reused_shards or ():
            record = self._file_ir_record_from_payload(
                snapshot_id,
                shard,
                created_at=self._utc_now(),
            )
            if record is None or record.relative_path in excluded_path_set:
                continue
            reused_records.append(record)
        reused_path_set = {record.relative_path for record in reused_records}
        previous_records = [
            record
            for record in self.list_file_ir_shard_records(previous_snapshot_id)
            if record.relative_path not in excluded_path_set
            and record.relative_path not in reused_path_set
        ]
        incoming_records: list[FileIRShardRecord] = []
        for shard in shards:
            record = self._file_ir_record_from_payload(
                snapshot_id,
                shard,
                created_at=self._utc_now(),
            )
            if record is None:
                continue
            if record.relative_path in changed_path_set:
                incoming_records.append(record)
        copied = 0
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                "DELETE FROM file_ir_shards WHERE snapshot_id=?",
                (snapshot_id,),
            )
            for record in previous_records:
                self._insert_file_ir_shard_record(
                    conn,
                    self._retarget_file_ir_shard_record(
                        record,
                        snapshot_id=snapshot_id,
                        created_at=self._utc_now(),
                    ),
                )
                copied += 1
            for record in reused_records:
                self._insert_file_ir_shard_record(conn, record)
            for record in incoming_records:
                self._insert_file_ir_shard_record(conn, record)
            conn.commit()
        summary: dict[str, int | str] = {
            "mode": "delta",
            "previous_snapshot_id": previous_snapshot_id,
            "copied_shards": copied,
            "written_shards": len(incoming_records),
            "removed_shards": len(removed_path_set),
            "excluded_path_count": len(excluded_path_set),
        }
        if reused_shards is not None:
            summary["reused_content_addressed_shards"] = len(reused_records)
        return summary

    def list_file_ir_shard_records(self, snapshot_id: str) -> list[FileIRShardRecord]:
        with self.db_runtime.connect() as conn:
            rows = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM file_ir_shards
                WHERE snapshot_id=?
                ORDER BY relative_path ASC
                """,
                (snapshot_id,),
            ).fetchall()
        return [
            record
            for row in rows
            if (record := self._file_ir_shard_record_from_row(row)) is not None
        ]

    def list_file_ir_shards(self, snapshot_id: str) -> list[dict[str, Any]]:
        return [
            self._file_ir_shard_payload_from_record(record)
            for record in self.list_file_ir_shard_records(snapshot_id)
        ]

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
        excluded_paths = sorted(
            normalize_path(str(path)) for path in changed_paths + removed_paths if path
        )
        excluded_path_set = set(excluded_paths)
        previous_records = [
            record
            for record in self.list_snapshot_unit_records(previous_snapshot_id)
            if record.relative_path not in excluded_path_set
        ]
        copied = 0
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                "DELETE FROM unit_artifacts WHERE snapshot_id=?",
                (snapshot_id,),
            )
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
                        created_at=self._utc_now(),
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
            relative_path=normalize_path(
                str(cls._row_value(row, 2, "relative_path") or "")
            ),
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
