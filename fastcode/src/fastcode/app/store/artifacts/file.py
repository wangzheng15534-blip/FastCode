"""Content-addressed per-file artifacts for incremental indexing."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

from fastcode.ports.artifacts import (
    FileArtifactRecordView,
)
from fastcode.ports.artifacts import (
    FileArtifactStore as FileArtifactStorePort,
)
from fastcode.ports.runtime import Clock
from fastcode.ports.storage import StoreDatabaseRuntime
from fastcode.utils.clock import SystemClock
from fastcode.utils.filesystem import normalize_path

from .file_contracts import FileArtifactRecord


class FileArtifactStore(FileArtifactStorePort):
    """Store reusable file artifacts keyed by repository, path, and content ID."""

    PARSED_ELEMENTS_ARTIFACT_TYPE: ClassVar[str] = "parsed_elements"
    parsed_elements_artifact_type: ClassVar[str] = PARSED_ELEMENTS_ARTIFACT_TYPE
    FILE_IR_ARTIFACT_TYPE: ClassVar[str] = "file_ir"
    EMBEDDING_REFS_ARTIFACT_TYPE: ClassVar[str] = "embedding_refs"
    SEMANTIC_FACTS_ARTIFACT_TYPE: ClassVar[str] = "semantic_facts"
    PARSED_ELEMENTS_SCHEMA_VERSION: ClassVar[str] = "fastcode.parsed_elements.v1"
    FILE_IR_SCHEMA_VERSION: ClassVar[str] = "fastcode.file_ir_shard.v1"
    EMBEDDING_REFS_SCHEMA_VERSION: ClassVar[str] = "fastcode.embedding_refs.v1"
    SEMANTIC_FACTS_SCHEMA_VERSION: ClassVar[str] = "fastcode.semantic_facts.v1"
    _FIELDS: ClassVar[tuple[str, ...]] = (
        "repo_name",
        "relative_path",
        "identity_kind",
        "identity_value",
        "artifact_type",
        "schema_version",
        "payload_json",
        "unit_count",
        "support_count",
        "relation_count",
        "embedding_count",
        "metadata_json",
        "created_at",
    )

    def __init__(
        self,
        db_runtime: StoreDatabaseRuntime,
        *,
        clock: Clock | None = None,
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
                CREATE TABLE IF NOT EXISTS file_artifacts (
                    repo_name TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    identity_kind TEXT NOT NULL,
                    identity_value TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    unit_count INTEGER NOT NULL,
                    support_count INTEGER NOT NULL,
                    relation_count INTEGER NOT NULL,
                    embedding_count INTEGER NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (
                        repo_name, relative_path, identity_kind, identity_value,
                        artifact_type
                    )
                )
                """,
            )
            self.db_runtime.execute(
                conn,
                """
                CREATE INDEX IF NOT EXISTS idx_file_artifacts_repo_path
                ON file_artifacts (repo_name, relative_path, artifact_type)
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
                ("file_artifact_store", "v1", self._utc_now()),
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
    def _mapping_list_payload(cls, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, (list, tuple)):
            return []
        payload: list[dict[str, Any]] = []
        for item in cast(Sequence[Any], value):
            if not isinstance(item, Mapping):
                continue
            payload.append(
                cast(
                    dict[str, Any],
                    cls._json_safe_value(
                        {
                            str(key): sub_item
                            for key, sub_item in cast(Mapping[Any, Any], item).items()
                        }
                    ),
                )
            )
        return payload

    @staticmethod
    def _deserialize_json_mapping(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return cast(dict[str, Any], raw)
        try:
            decoded = json.loads(str(raw))
        except (json.JSONDecodeError, TypeError):
            return {}
        return cast(dict[str, Any], decoded) if isinstance(decoded, dict) else {}

    @classmethod
    def _identity_from_file_info(
        cls,
        file_info: Mapping[str, Any] | None,
    ) -> tuple[str, str] | None:
        if file_info is None:
            return None
        blob_oid = file_info.get("git_blob_oid") or file_info.get("blob_oid")
        if blob_oid:
            return "blob_oid", str(blob_oid)
        content_hash = file_info.get("content_hash")
        if content_hash:
            return "content_hash", str(content_hash)
        return None

    @staticmethod
    def _file_info_by_path(
        file_infos: Sequence[Mapping[str, Any]] | None,
    ) -> dict[str, Mapping[str, Any]]:
        if not file_infos:
            return {}
        return {
            normalize_path(
                str(file_info.get("relative_path") or file_info.get("path") or "")
            ): file_info
            for file_info in file_infos
            if file_info.get("relative_path") or file_info.get("path")
        }

    @classmethod
    def _identity_from_file_ir_shard(
        cls,
        shard: Mapping[str, Any],
    ) -> tuple[str, str] | None:
        identity_kind = shard.get("identity_kind")
        identity_value = shard.get("identity_value")
        if identity_kind and identity_value:
            return str(identity_kind), str(identity_value)
        blob_oid = shard.get("blob_oid") or shard.get("git_blob_oid")
        if blob_oid:
            return "blob_oid", str(blob_oid)
        content_hash = shard.get("content_hash")
        if content_hash:
            return "content_hash", str(content_hash)
        for unit in cls._mapping_list_payload(shard.get("units")):
            metadata = unit.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            metadata_payload = cast(Mapping[str, Any], metadata)
            blob_oid = metadata_payload.get("blob_oid") or metadata_payload.get(
                "git_blob_oid"
            )
            if blob_oid:
                return "blob_oid", str(blob_oid)
            content_hash = metadata_payload.get("content_hash")
            if content_hash:
                return "content_hash", str(content_hash)
        return None

    @classmethod
    def _record_from_file_ir_shard(
        cls,
        *,
        repo_name: str,
        shard: Mapping[str, Any],
        file_info_by_path: Mapping[str, Mapping[str, Any]] | None = None,
        created_at: str,
    ) -> FileArtifactRecord | None:
        relative_path = normalize_path(
            str(shard.get("relative_path") or shard.get("path") or "")
        )
        file_info = (
            file_info_by_path.get(relative_path)
            if file_info_by_path is not None
            else None
        )
        identity = cls._identity_from_file_info(
            file_info
        ) or cls._identity_from_file_ir_shard(shard)
        if not repo_name or not relative_path or identity is None:
            return None
        units = cls._mapping_list_payload(shard.get("units"))
        supports = cls._mapping_list_payload(shard.get("supports"))
        relations = cls._mapping_list_payload(shard.get("relations"))
        embeddings = cls._mapping_list_payload(shard.get("embeddings"))
        identity_kind, identity_value = identity
        envelope = {
            "schema_version": cls.FILE_IR_SCHEMA_VERSION,
            "repo_name": repo_name,
            "relative_path": relative_path,
            "identity_kind": identity_kind,
            "identity_value": identity_value,
            "content_hash": shard.get("content_hash"),
            "blob_oid": shard.get("blob_oid") or shard.get("git_blob_oid"),
            "units": units,
            "supports": supports,
            "relations": relations,
            "embeddings": embeddings,
        }
        metadata = {
            "source_snapshot_id": shard.get("snapshot_id"),
            "artifact_type": cls.FILE_IR_ARTIFACT_TYPE,
        }
        return FileArtifactRecord(
            repo_name=repo_name,
            relative_path=relative_path,
            identity_kind=identity_kind,
            identity_value=identity_value,
            artifact_type=cls.FILE_IR_ARTIFACT_TYPE,
            schema_version=cls.FILE_IR_SCHEMA_VERSION,
            payload_json=json.dumps(
                cls._json_safe_value(envelope),
                ensure_ascii=False,
                sort_keys=True,
            ),
            unit_count=len(units),
            support_count=len(supports),
            relation_count=len(relations),
            embedding_count=len(embeddings),
            metadata_json=json.dumps(
                cls._json_safe_value(metadata),
                ensure_ascii=False,
                sort_keys=True,
            ),
            created_at=created_at,
        )

    @classmethod
    def _parsed_element_payload(cls, element: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            str(key): item for key, item in cast(Mapping[Any, Any], element).items()
        }
        payload.pop("embedding", None)
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            metadata_payload = {
                str(key): item
                for key, item in cast(Mapping[Any, Any], metadata).items()
                if str(key) != "embedding"
            }
            payload["metadata"] = metadata_payload
        return cast(dict[str, Any], cls._json_safe_value(payload))

    @classmethod
    def _record_from_parsed_element_group(
        cls,
        *,
        repo_name: str,
        relative_path: str,
        elements: Sequence[Mapping[str, Any]],
        file_info_by_path: Mapping[str, Mapping[str, Any]],
        created_at: str,
    ) -> FileArtifactRecord | None:
        relative_path = normalize_path(relative_path)
        identity = cls._identity_from_file_info(file_info_by_path.get(relative_path))
        if not repo_name or not relative_path or identity is None or not elements:
            return None
        identity_kind, identity_value = identity
        parsed_elements = [cls._parsed_element_payload(element) for element in elements]
        envelope = {
            "schema_version": cls.PARSED_ELEMENTS_SCHEMA_VERSION,
            "repo_name": repo_name,
            "relative_path": relative_path,
            "identity_kind": identity_kind,
            "identity_value": identity_value,
            "elements": parsed_elements,
        }
        metadata = {"artifact_type": cls.PARSED_ELEMENTS_ARTIFACT_TYPE}
        return FileArtifactRecord(
            repo_name=repo_name,
            relative_path=relative_path,
            identity_kind=identity_kind,
            identity_value=identity_value,
            artifact_type=cls.PARSED_ELEMENTS_ARTIFACT_TYPE,
            schema_version=cls.PARSED_ELEMENTS_SCHEMA_VERSION,
            payload_json=json.dumps(
                cls._json_safe_value(envelope),
                ensure_ascii=False,
                sort_keys=True,
            ),
            unit_count=len(parsed_elements),
            support_count=0,
            relation_count=0,
            embedding_count=0,
            metadata_json=json.dumps(
                cls._json_safe_value(metadata),
                ensure_ascii=False,
                sort_keys=True,
            ),
            created_at=created_at,
        )

    @classmethod
    def _record_from_embedding_ref_group(
        cls,
        *,
        repo_name: str,
        relative_path: str,
        rows: Sequence[Mapping[str, Any]],
        file_info_by_path: Mapping[str, Mapping[str, Any]],
        created_at: str,
    ) -> FileArtifactRecord | None:
        relative_path = normalize_path(relative_path)
        identity = cls._identity_from_file_info(file_info_by_path.get(relative_path))
        if not repo_name or not relative_path or identity is None:
            return None
        identity_kind, identity_value = identity
        embeddings: list[dict[str, Any]] = []
        for row in rows:
            metadata = row.get("metadata")
            metadata_payload: Mapping[str, Any]
            if isinstance(metadata, Mapping):
                metadata_payload = cast(Mapping[str, Any], metadata)
            else:
                metadata_payload = {}
            embedding_text_hash = row.get(
                "embedding_text_hash"
            ) or metadata_payload.get("embedding_text_hash")
            embedding_artifact_ref = row.get(
                "embedding_artifact_ref"
            ) or metadata_payload.get("embedding_artifact_ref")
            embedding_fingerprint = metadata_payload.get("embedding_fingerprint")
            if embedding_text_hash is None and embedding_artifact_ref is None:
                continue
            embeddings.append(
                cast(
                    dict[str, Any],
                    cls._json_safe_value(
                        {
                            "stable_unit_id": metadata_payload.get("stable_unit_id")
                            or row.get("stable_unit_id"),
                            "unit_type": row.get("type") or row.get("unit_type"),
                            "content_hash": row.get("content_hash")
                            or metadata_payload.get("content_hash"),
                            "embedding_text_hash": embedding_text_hash,
                            "embedding_artifact_ref": embedding_artifact_ref,
                            "embedding_fingerprint": embedding_fingerprint,
                        }
                    ),
                )
            )
        if not embeddings:
            return None
        envelope = {
            "schema_version": cls.EMBEDDING_REFS_SCHEMA_VERSION,
            "repo_name": repo_name,
            "relative_path": relative_path,
            "identity_kind": identity_kind,
            "identity_value": identity_value,
            "embeddings": embeddings,
        }
        metadata = {"artifact_type": cls.EMBEDDING_REFS_ARTIFACT_TYPE}
        return FileArtifactRecord(
            repo_name=repo_name,
            relative_path=relative_path,
            identity_kind=identity_kind,
            identity_value=identity_value,
            artifact_type=cls.EMBEDDING_REFS_ARTIFACT_TYPE,
            schema_version=cls.EMBEDDING_REFS_SCHEMA_VERSION,
            payload_json=json.dumps(
                cls._json_safe_value(envelope),
                ensure_ascii=False,
                sort_keys=True,
            ),
            unit_count=len(rows),
            support_count=0,
            relation_count=0,
            embedding_count=len(embeddings),
            metadata_json=json.dumps(
                cls._json_safe_value(metadata),
                ensure_ascii=False,
                sort_keys=True,
            ),
            created_at=created_at,
        )

    @classmethod
    def _record_from_semantic_fact_shard(
        cls,
        *,
        repo_name: str,
        shard: Mapping[str, Any],
        file_info_by_path: Mapping[str, Mapping[str, Any]],
        created_at: str,
    ) -> FileArtifactRecord | None:
        relative_path = normalize_path(
            str(shard.get("relative_path") or shard.get("path") or "")
        )
        file_info = file_info_by_path.get(relative_path)
        identity = cls._identity_from_file_info(
            file_info
        ) or cls._identity_from_file_ir_shard(shard)
        if not repo_name or not relative_path or identity is None:
            return None
        supports = cls._mapping_list_payload(shard.get("supports"))
        relations = cls._mapping_list_payload(shard.get("relations"))
        if not supports and not relations:
            return None
        identity_kind, identity_value = identity
        envelope = {
            "schema_version": cls.SEMANTIC_FACTS_SCHEMA_VERSION,
            "repo_name": repo_name,
            "relative_path": relative_path,
            "identity_kind": identity_kind,
            "identity_value": identity_value,
            "supports": supports,
            "relations": relations,
        }
        metadata = {"artifact_type": cls.SEMANTIC_FACTS_ARTIFACT_TYPE}
        return FileArtifactRecord(
            repo_name=repo_name,
            relative_path=relative_path,
            identity_kind=identity_kind,
            identity_value=identity_value,
            artifact_type=cls.SEMANTIC_FACTS_ARTIFACT_TYPE,
            schema_version=cls.SEMANTIC_FACTS_SCHEMA_VERSION,
            payload_json=json.dumps(
                cls._json_safe_value(envelope),
                ensure_ascii=False,
                sort_keys=True,
            ),
            unit_count=0,
            support_count=len(supports),
            relation_count=len(relations),
            embedding_count=0,
            metadata_json=json.dumps(
                cls._json_safe_value(metadata),
                ensure_ascii=False,
                sort_keys=True,
            ),
            created_at=created_at,
        )

    @classmethod
    def _record_from_row(cls, row: Any) -> FileArtifactRecord | None:
        repo_name = cls._row_value(row, 0, "repo_name")
        relative_path = cls._row_value(row, 1, "relative_path")
        identity_value = cls._row_value(row, 3, "identity_value")
        if repo_name is None or relative_path is None or identity_value is None:
            return None
        return FileArtifactRecord(
            repo_name=str(repo_name),
            relative_path=normalize_path(str(relative_path)),
            identity_kind=str(cls._row_value(row, 2, "identity_kind") or ""),
            identity_value=str(identity_value),
            artifact_type=str(cls._row_value(row, 4, "artifact_type") or ""),
            schema_version=str(cls._row_value(row, 5, "schema_version") or ""),
            payload_json=str(cls._row_value(row, 6, "payload_json") or "{}"),
            unit_count=int(cls._row_value(row, 7, "unit_count") or 0),
            support_count=int(cls._row_value(row, 8, "support_count") or 0),
            relation_count=int(cls._row_value(row, 9, "relation_count") or 0),
            embedding_count=int(cls._row_value(row, 10, "embedding_count") or 0),
            metadata_json=(
                str(metadata_json)
                if (metadata_json := cls._row_value(row, 11, "metadata_json"))
                is not None
                else None
            ),
            created_at=str(cls._row_value(row, 12, "created_at") or ""),
        )

    def _insert_record(self, conn: Any, record: FileArtifactRecord) -> None:
        self.db_runtime.execute(
            conn,
            """
            INSERT INTO file_artifacts (
                repo_name, relative_path, identity_kind, identity_value,
                artifact_type, schema_version, payload_json, unit_count,
                support_count, relation_count, embedding_count, metadata_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                repo_name, relative_path, identity_kind, identity_value, artifact_type
            ) DO UPDATE SET
                schema_version=excluded.schema_version,
                payload_json=excluded.payload_json,
                unit_count=excluded.unit_count,
                support_count=excluded.support_count,
                relation_count=excluded.relation_count,
                embedding_count=excluded.embedding_count,
                metadata_json=excluded.metadata_json,
                created_at=excluded.created_at
            """,
            (
                record.repo_name,
                record.relative_path,
                record.identity_kind,
                record.identity_value,
                record.artifact_type,
                record.schema_version,
                record.payload_json,
                record.unit_count,
                record.support_count,
                record.relation_count,
                record.embedding_count,
                record.metadata_json,
                record.created_at,
            ),
        )

    def upsert_file_ir_shards(
        self,
        *,
        repo_name: str,
        shards: Sequence[Mapping[str, Any]],
        file_infos: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[dict[str, int | str], list[FileArtifactRecord]]:
        file_info_by_path = self._file_info_by_path(file_infos)
        records = [
            record
            for shard in shards
            if (
                record := self._record_from_file_ir_shard(
                    repo_name=repo_name,
                    shard=shard,
                    file_info_by_path=file_info_by_path,
                    created_at=self._utc_now(),
                )
            )
            is not None
        ]
        self._upsert_records(records)
        return (
            {
                "mode": "content_addressed",
                "artifact_type": self.FILE_IR_ARTIFACT_TYPE,
                "written_records": len(records),
            },
            records,
        )

    def upsert_parsed_elements(
        self,
        *,
        repo_name: str,
        elements: Sequence[Mapping[str, Any]],
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str] | None = None,
    ) -> tuple[dict[str, int | str], list[FileArtifactRecord]]:
        path_filter = (
            {normalize_path(str(path)) for path in paths if path}
            if paths is not None
            else None
        )
        elements_by_path: dict[str, list[Mapping[str, Any]]] = {}
        for element in elements:
            relative_path = normalize_path(
                str(element.get("relative_path") or element.get("file_path") or "")
            )
            if not relative_path:
                continue
            if path_filter is not None and relative_path not in path_filter:
                continue
            elements_by_path.setdefault(relative_path, []).append(element)
        file_info_by_path = self._file_info_by_path(file_infos)
        records = [
            record
            for relative_path, grouped_elements in sorted(elements_by_path.items())
            if (
                record := self._record_from_parsed_element_group(
                    repo_name=repo_name,
                    relative_path=relative_path,
                    elements=grouped_elements,
                    file_info_by_path=file_info_by_path,
                    created_at=self._utc_now(),
                )
            )
            is not None
        ]
        self._upsert_records(records)
        return (
            {
                "mode": "content_addressed",
                "artifact_type": self.PARSED_ELEMENTS_ARTIFACT_TYPE,
                "candidate_files": len(elements_by_path),
                "written_records": len(records),
            },
            records,
        )

    def upsert_embedding_refs(
        self,
        *,
        repo_name: str,
        rows: Sequence[Mapping[str, Any]],
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str] | None = None,
    ) -> tuple[dict[str, int | str], list[FileArtifactRecord]]:
        path_filter = (
            {normalize_path(str(path)) for path in paths if path}
            if paths is not None
            else None
        )
        rows_by_path: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            relative_path = normalize_path(
                str(row.get("relative_path") or row.get("file_path") or "")
            )
            if not relative_path:
                continue
            if path_filter is not None and relative_path not in path_filter:
                continue
            rows_by_path.setdefault(relative_path, []).append(row)
        file_info_by_path = self._file_info_by_path(file_infos)
        records = [
            record
            for relative_path, grouped_rows in sorted(rows_by_path.items())
            if (
                record := self._record_from_embedding_ref_group(
                    repo_name=repo_name,
                    relative_path=relative_path,
                    rows=grouped_rows,
                    file_info_by_path=file_info_by_path,
                    created_at=self._utc_now(),
                )
            )
            is not None
        ]
        self._upsert_records(records)
        return (
            {
                "mode": "content_addressed",
                "artifact_type": self.EMBEDDING_REFS_ARTIFACT_TYPE,
                "candidate_files": len(rows_by_path),
                "written_records": len(records),
            },
            records,
        )

    def upsert_semantic_fact_shards(
        self,
        *,
        repo_name: str,
        shards: Sequence[Mapping[str, Any]],
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str] | None = None,
    ) -> tuple[dict[str, int | str], list[FileArtifactRecord]]:
        path_filter = (
            {normalize_path(str(path)) for path in paths if path}
            if paths is not None
            else None
        )
        publish_shards = [
            shard
            for shard in shards
            if path_filter is None
            or normalize_path(
                str(shard.get("relative_path") or shard.get("path") or "")
            )
            in path_filter
        ]
        file_info_by_path = self._file_info_by_path(file_infos)
        records = [
            record
            for shard in publish_shards
            if (
                record := self._record_from_semantic_fact_shard(
                    repo_name=repo_name,
                    shard=shard,
                    file_info_by_path=file_info_by_path,
                    created_at=self._utc_now(),
                )
            )
            is not None
        ]
        self._upsert_records(records)
        return (
            {
                "mode": "content_addressed",
                "artifact_type": self.SEMANTIC_FACTS_ARTIFACT_TYPE,
                "candidate_shards": len(publish_shards),
                "written_records": len(records),
            },
            records,
        )

    def _upsert_records(self, records: Sequence[FileArtifactRecord]) -> None:
        with self.db_runtime.connect() as conn:
            for record in records:
                self._insert_record(conn, record)
            conn.commit()

    def get_record(
        self,
        *,
        repo_name: str,
        relative_path: str,
        identity_kind: str,
        identity_value: str,
        artifact_type: str,
    ) -> FileArtifactRecord | None:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM file_artifacts
                WHERE repo_name=? AND relative_path=? AND identity_kind=?
                    AND identity_value=? AND artifact_type=?
                """,
                (
                    repo_name,
                    normalize_path(relative_path),
                    identity_kind,
                    identity_value,
                    artifact_type,
                ),
            ).fetchone()
        return self._record_from_row(row)

    def list_file_ir_records_for_file_infos(
        self,
        *,
        repo_name: str,
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str],
    ) -> list[FileArtifactRecord]:
        return self._list_records_for_file_infos(
            repo_name=repo_name,
            file_infos=file_infos,
            paths=paths,
            artifact_type=self.FILE_IR_ARTIFACT_TYPE,
        )

    def list_parsed_element_records_for_file_infos(
        self,
        *,
        repo_name: str,
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str],
    ) -> list[FileArtifactRecord]:
        return self._list_records_for_file_infos(
            repo_name=repo_name,
            file_infos=file_infos,
            paths=paths,
            artifact_type=self.PARSED_ELEMENTS_ARTIFACT_TYPE,
        )

    def list_embedding_ref_records_for_file_infos(
        self,
        *,
        repo_name: str,
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str],
    ) -> list[FileArtifactRecord]:
        return self._list_records_for_file_infos(
            repo_name=repo_name,
            file_infos=file_infos,
            paths=paths,
            artifact_type=self.EMBEDDING_REFS_ARTIFACT_TYPE,
        )

    def list_semantic_fact_records_for_file_infos(
        self,
        *,
        repo_name: str,
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str],
    ) -> list[FileArtifactRecord]:
        return self._list_records_for_file_infos(
            repo_name=repo_name,
            file_infos=file_infos,
            paths=paths,
            artifact_type=self.SEMANTIC_FACTS_ARTIFACT_TYPE,
        )

    def _list_records_for_file_infos(
        self,
        *,
        repo_name: str,
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str],
        artifact_type: str,
    ) -> list[FileArtifactRecord]:
        file_info_by_path = self._file_info_by_path(file_infos)
        records: list[FileArtifactRecord] = []
        for raw_path in sorted({normalize_path(str(path)) for path in paths if path}):
            file_info = file_info_by_path.get(raw_path)
            identity = self._identity_from_file_info(file_info)
            if identity is None:
                continue
            identity_kind, identity_value = identity
            record = self.get_record(
                repo_name=repo_name,
                relative_path=raw_path,
                identity_kind=identity_kind,
                identity_value=identity_value,
                artifact_type=artifact_type,
            )
            if record is not None:
                records.append(record)
        return records

    @classmethod
    def file_ir_payload_from_record(
        cls,
        record: FileArtifactRecordView,
        *,
        snapshot_id: str | None = None,
    ) -> dict[str, Any]:
        payload = cls._deserialize_json_mapping(record.payload_json)
        if snapshot_id is not None:
            payload["snapshot_id"] = snapshot_id
        payload["repo_name"] = record.repo_name
        payload["relative_path"] = record.relative_path
        payload["identity_kind"] = record.identity_kind
        payload["identity_value"] = record.identity_value
        payload["schema_version"] = record.schema_version
        payload["created_at"] = record.created_at
        payload["counts"] = {
            "units": record.unit_count,
            "supports": record.support_count,
            "relations": record.relation_count,
            "embeddings": record.embedding_count,
        }
        return payload

    @classmethod
    def parsed_elements_payload_from_record(
        cls,
        record: FileArtifactRecordView,
    ) -> dict[str, Any]:
        payload = cls._deserialize_json_mapping(record.payload_json)
        payload["repo_name"] = record.repo_name
        payload["relative_path"] = record.relative_path
        payload["identity_kind"] = record.identity_kind
        payload["identity_value"] = record.identity_value
        payload["schema_version"] = record.schema_version
        payload["created_at"] = record.created_at
        payload["counts"] = {
            "elements": record.unit_count,
        }
        return payload

    @classmethod
    def embedding_refs_payload_from_record(
        cls,
        record: FileArtifactRecordView,
    ) -> dict[str, Any]:
        payload = cls._deserialize_json_mapping(record.payload_json)
        payload["repo_name"] = record.repo_name
        payload["relative_path"] = record.relative_path
        payload["identity_kind"] = record.identity_kind
        payload["identity_value"] = record.identity_value
        payload["schema_version"] = record.schema_version
        payload["created_at"] = record.created_at
        payload["counts"] = {
            "units": record.unit_count,
            "embeddings": record.embedding_count,
        }
        return payload

    @classmethod
    def semantic_facts_payload_from_record(
        cls,
        record: FileArtifactRecordView,
    ) -> dict[str, Any]:
        payload = cls._deserialize_json_mapping(record.payload_json)
        payload["repo_name"] = record.repo_name
        payload["relative_path"] = record.relative_path
        payload["identity_kind"] = record.identity_kind
        payload["identity_value"] = record.identity_value
        payload["schema_version"] = record.schema_version
        payload["created_at"] = record.created_at
        payload["counts"] = {
            "supports": record.support_count,
            "relations": record.relation_count,
        }
        return payload
