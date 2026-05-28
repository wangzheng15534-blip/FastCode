"""
PostgreSQL retrieval store using pgvector + FTS, with graceful fallback.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from fastcode.ports.storage import StoreDatabaseRuntime

from .pg_retrieval_contracts import PgRetrievalElementRecord, PgRetrievalResultRecord
from .vector_math import as_float32_matrix, as_float32_vector


class PgRetrievalStore:
    _ELEMENT_JSON_FIELDS = (
        "id",
        "type",
        "name",
        "file_path",
        "relative_path",
        "language",
        "start_line",
        "end_line",
        "code",
        "signature",
        "docstring",
        "summary",
        "repo_name",
        "repo_url",
        "snapshot_id",
        "source_priority",
        "embedding_text",
        "embedding_artifact_ref",
        "embedding_fingerprint",
        "ir_symbol_id",
        "stable_unit_id",
        "content_hash",
        "syntax_hash",
        "signature_hash",
        "edge_surface_hash",
        "embedding_text_hash",
        "api_surface_hash",
    )

    def __init__(
        self, db_runtime: StoreDatabaseRuntime, config: dict[str, Any]
    ) -> None:
        self.db_runtime = db_runtime
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.enabled = self.db_runtime.backend == "postgres"
        self.backend_mode = config.get("retrieval", {}).get(
            "retrieval_backend", "pg_hybrid"
        )
        self.last_upsert_metrics: dict[str, Any] = {}
        if self.enabled:
            self._init_db()

    def _init_db(self) -> None:
        with self.db_runtime.connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception as e:
                self.logger.warning(f"pgvector extension unavailable: {e}")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_vectors (
                    snapshot_id TEXT NOT NULL,
                    element_id TEXT NOT NULL,
                    repo_name TEXT,
                    relative_path TEXT,
                    language TEXT,
                    element_type TEXT,
                    embedding vector,
                    embedding_arr DOUBLE PRECISION[],
                    metadata_json JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (snapshot_id, element_id)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embedding_snapshot_repo
                ON embedding_vectors (snapshot_id, repo_name)
                """
            )
            # HNSW optional; skip hard-fail if not supported by deployment.
            cur.execute("SAVEPOINT sp_hnsw")
            try:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_embedding_hnsw
                    ON embedding_vectors USING hnsw (embedding vector_cosine_ops)
                    """
                )
                cur.execute("RELEASE SAVEPOINT sp_hnsw")
            except Exception:
                cur.execute("ROLLBACK TO SAVEPOINT sp_hnsw")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS search_documents (
                    snapshot_id TEXT NOT NULL,
                    element_id TEXT NOT NULL,
                    repo_name TEXT,
                    relative_path TEXT,
                    language TEXT,
                    element_type TEXT,
                    text_content TEXT NOT NULL,
                    metadata_json JSONB NOT NULL,
                    search_tsv tsvector GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(text_content, ''))
                    ) STORED,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (snapshot_id, element_id)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_search_documents_tsv
                ON search_documents USING GIN(search_tsv)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_search_documents_snapshot_repo
                ON search_documents (snapshot_id, repo_name)
                """
            )
            conn.commit()

    def is_active(self) -> bool:
        return self.enabled and self.backend_mode == "pg_hybrid"

    @staticmethod
    def _vector_array(vec: Any) -> np.ndarray | None:
        return as_float32_vector(vec, copy_policy="contiguous")

    @staticmethod
    def _vector_literal_from_array(vec: np.ndarray) -> str:
        if vec.size == 0:
            raise ValueError("Cannot create vector literal from empty sequence")
        return "[" + ",".join(f"{float(v):.8f}" for v in vec) + "]"

    @classmethod
    def _vector_literal(cls, vec: Sequence[float] | np.ndarray) -> str:
        array = cls._vector_array(vec)
        if array is None:
            raise ValueError("Cannot create vector literal from empty sequence")
        return cls._vector_literal_from_array(array)

    def _vector_parameter(self, vec: np.ndarray | None) -> np.ndarray | str | None:
        if vec is None:
            return None
        array = as_float32_vector(vec, copy_policy="contiguous")
        if array is None:
            return None
        if not self.db_runtime.supports_pgvector_adapter():
            return PgRetrievalStore._vector_literal_from_array(array)
        return array

    @staticmethod
    def _row_value(row: Any, index: int, key: str) -> Any:
        if isinstance(row, dict):
            return cast(dict[str, Any], row).get(key)
        try:
            return row[index]
        except (IndexError, KeyError, TypeError):
            return None

    @staticmethod
    def _decode_metadata(raw_metadata: Any) -> dict[str, Any] | None:
        if isinstance(raw_metadata, Mapping):
            return {
                str(key): value
                for key, value in cast(Mapping[Any, Any], raw_metadata).items()
            }
        if not raw_metadata:
            return None
        try:
            parsed = json.loads(str(raw_metadata))
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parsed, Mapping):
            return None
        return {
            str(key): value for key, value in cast(Mapping[Any, Any], parsed).items()
        }

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        return str(value) if value is not None else None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _mapping_payload(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        return {str(key): item for key, item in cast(Mapping[Any, Any], value).items()}

    @classmethod
    def _element_record_from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> PgRetrievalElementRecord | None:
        source = cls._mapping_payload(payload)
        raw_metadata = source.get("metadata")
        metadata = cls._mapping_payload(raw_metadata)
        raw_type = source.get("type")
        if raw_type is None:
            raw_type = source.get("element_type")
        if raw_type is None:
            raw_type = metadata.get("type") or metadata.get("element_type")
        raw_fingerprint = source.get("embedding_fingerprint")
        if not isinstance(raw_fingerprint, Mapping):
            raw_fingerprint = metadata.get("embedding_fingerprint")
        fingerprint = (
            cls._mapping_payload(raw_fingerprint)
            if isinstance(raw_fingerprint, Mapping)
            else None
        )
        present_fields = tuple(
            field_name
            for field_name in ("metadata", "element_type", *cls._ELEMENT_JSON_FIELDS)
            if field_name in source
        )
        return PgRetrievalElementRecord(
            id=str(source.get("id") or ""),
            element_type=cls._optional_str(raw_type),
            name=cls._optional_str(source.get("name")),
            file_path=cls._optional_str(source.get("file_path")),
            relative_path=cls._optional_str(source.get("relative_path")),
            language=cls._optional_str(source.get("language")),
            start_line=cls._optional_int(source.get("start_line")),
            end_line=cls._optional_int(source.get("end_line")),
            code=cls._optional_str(source.get("code")),
            signature=cls._optional_str(source.get("signature")),
            docstring=cls._optional_str(source.get("docstring")),
            summary=cls._optional_str(source.get("summary")),
            repo_name=cls._optional_str(source.get("repo_name")),
            repo_url=cls._optional_str(source.get("repo_url")),
            snapshot_id=cls._optional_str(source.get("snapshot_id")),
            source_priority=source.get("source_priority"),
            embedding_text=cls._optional_str(source.get("embedding_text")),
            embedding_artifact_ref=cls._optional_str(
                source.get("embedding_artifact_ref")
            ),
            embedding_fingerprint=fingerprint,
            ir_symbol_id=cls._optional_str(source.get("ir_symbol_id")),
            stable_unit_id=cls._optional_str(source.get("stable_unit_id")),
            content_hash=cls._optional_str(source.get("content_hash")),
            syntax_hash=cls._optional_str(source.get("syntax_hash")),
            signature_hash=cls._optional_str(source.get("signature_hash")),
            edge_surface_hash=cls._optional_str(source.get("edge_surface_hash")),
            embedding_text_hash=cls._optional_str(source.get("embedding_text_hash")),
            api_surface_hash=cls._optional_str(source.get("api_surface_hash")),
            metadata=metadata,
            present_fields=present_fields,
        )

    @classmethod
    def _result_record_from_raw_metadata(
        cls,
        raw_metadata: Any,
        raw_score: Any,
        *,
        query_embedding_fingerprint: Mapping[str, Any] | None = None,
    ) -> PgRetrievalResultRecord | None:
        payload = cls._decode_metadata(raw_metadata)
        if payload is None:
            return None
        element = cls._element_record_from_payload(payload)
        if element is None:
            return None
        if not cls._element_record_embedding_fingerprint_matches(
            element,
            query_embedding_fingerprint,
        ):
            return None
        return PgRetrievalResultRecord(
            element=element,
            score=float(raw_score or 0.0),
        )

    @classmethod
    def _element_record_embedding_fingerprint_matches(
        cls,
        element: PgRetrievalElementRecord,
        expected: Mapping[str, Any] | None,
    ) -> bool:
        if expected is None:
            return True
        fingerprint: Mapping[str, Any] | None = element.embedding_fingerprint
        if not isinstance(fingerprint, Mapping):
            nested = element.metadata.get("embedding_fingerprint")
            if isinstance(nested, Mapping):
                fingerprint = cast(Mapping[str, Any], nested)
        if not isinstance(fingerprint, Mapping):
            return False
        for field_name, expected_value in expected.items():
            if fingerprint.get(field_name) != expected_value:
                return False
        return True

    @classmethod
    def _element_payload_from_record(
        cls,
        record: PgRetrievalElementRecord,
    ) -> dict[str, Any]:
        present_fields = set(record.present_fields)
        payload: dict[str, Any] = {}
        if "metadata" in present_fields:
            payload["metadata"] = dict(record.metadata)
        field_values: dict[str, Any] = {
            "id": record.id,
            "type": record.element_type,
            "element_type": record.element_type,
            "name": record.name,
            "file_path": record.file_path,
            "relative_path": record.relative_path,
            "language": record.language,
            "start_line": record.start_line,
            "end_line": record.end_line,
            "code": record.code,
            "signature": record.signature,
            "docstring": record.docstring,
            "summary": record.summary,
            "repo_name": record.repo_name,
            "repo_url": record.repo_url,
            "snapshot_id": record.snapshot_id,
            "source_priority": record.source_priority,
            "embedding_text": record.embedding_text,
            "embedding_artifact_ref": record.embedding_artifact_ref,
            "embedding_fingerprint": (
                dict(record.embedding_fingerprint)
                if record.embedding_fingerprint is not None
                else None
            ),
            "ir_symbol_id": record.ir_symbol_id,
            "stable_unit_id": record.stable_unit_id,
            "content_hash": record.content_hash,
            "syntax_hash": record.syntax_hash,
            "signature_hash": record.signature_hash,
            "edge_surface_hash": record.edge_surface_hash,
            "embedding_text_hash": record.embedding_text_hash,
            "api_surface_hash": record.api_surface_hash,
        }
        for field_name in ("element_type", *cls._ELEMENT_JSON_FIELDS):
            if field_name in present_fields:
                payload[field_name] = field_values[field_name]
        return payload

    @classmethod
    def _result_payloads_from_records(
        cls,
        records: Sequence[PgRetrievalResultRecord],
    ) -> list[tuple[dict[str, Any], float]]:
        return [
            (cls._element_payload_from_record(record.element), record.score)
            for record in records
        ]

    @staticmethod
    def _embedding_fingerprint_sql_filter(
        expected: Mapping[str, Any] | None,
    ) -> tuple[str | None, str | None, str | None]:
        if expected is None:
            return (None, None, None)
        fingerprint_json = json.dumps(
            {"embedding_fingerprint": dict(expected)},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return (fingerprint_json, fingerprint_json, fingerprint_json)

    @classmethod
    def _metadata_score_records(
        cls,
        rows: Any,
        *,
        query_embedding_fingerprint: Mapping[str, Any] | None = None,
    ) -> list[PgRetrievalResultRecord]:
        out: list[PgRetrievalResultRecord] = []
        for row in rows:
            raw_metadata = cls._row_value(row, 0, "metadata_json")
            raw_score = cls._row_value(row, 1, "score")
            record = cls._result_record_from_raw_metadata(
                raw_metadata,
                raw_score,
                query_embedding_fingerprint=query_embedding_fingerprint,
            )
            if record is not None:
                out.append(record)
        return out

    @classmethod
    def _row_filter_value(
        cls,
        row: Any,
        index: int,
        key: str,
        raw_metadata: Any,
        metadata_key: str,
    ) -> str | None:
        raw_value = cls._row_value(row, index, key)
        if raw_value is not None:
            return str(raw_value)
        if isinstance(raw_metadata, dict):
            metadata = cast(dict[str, Any], raw_metadata)
            fallback = metadata.get(metadata_key) or metadata.get(key)
            return str(fallback) if fallback is not None else None
        return None

    @staticmethod
    def _looks_like_numeric_vector(value: Any) -> bool:
        if isinstance(value, np.ndarray):
            return True
        if not isinstance(value, (list, tuple)):
            return False
        sequence = cast(Sequence[Any], value)
        if not sequence:
            return False
        return all(
            isinstance(item, (int, float, np.integer, np.floating))
            and not isinstance(item, bool)
            for item in sequence
        )

    @staticmethod
    def _is_embedding_like_metadata_key(key: str) -> bool:
        normalized = key.lower()
        return "embedding" in normalized or normalized in {
            "vector",
            "vectors",
            "vector_arr",
            "vector_array",
            "vector_values",
        }

    @staticmethod
    def _json_safe_payload(value: Any, *, metadata_key: str | None = None) -> Any:
        if (
            metadata_key
            and PgRetrievalStore._is_embedding_like_metadata_key(metadata_key)
            and PgRetrievalStore._looks_like_numeric_vector(value)
        ):
            raise ValueError(
                "Embedding/vector arrays must not be serialized into PG metadata JSON"
            )
        if isinstance(value, Mapping):
            payload: dict[str, Any] = {}
            for k, v in cast(Mapping[Any, Any], value).items():
                key_str = str(k)
                if key_str == "embedding":
                    continue
                payload[key_str] = PgRetrievalStore._json_safe_payload(
                    v, metadata_key=key_str
                )
            return payload
        if isinstance(value, (list, tuple)):
            return [
                PgRetrievalStore._json_safe_payload(item, metadata_key=metadata_key)
                for item in cast(Sequence[Any], value)
            ]
        if isinstance(value, set):
            return [
                PgRetrievalStore._json_safe_payload(item, metadata_key=metadata_key)
                for item in cast(set[Any], value)
            ]
        if isinstance(value, np.ndarray):
            raise ValueError(
                "NumPy arrays must not be serialized into PG metadata JSON"
            )
        if isinstance(value, np.integer):  # type: ignore[arg-type]
            return int(cast(Any, value))
        if isinstance(value, np.floating):  # type: ignore[arg-type]
            return float(cast(Any, value))
        if isinstance(value, np.bool_):  # type: ignore[arg-type]
            return bool(cast(Any, value))
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return repr(value)

    @staticmethod
    def _embedding_ref_metadata(elem: dict[str, Any]) -> dict[str, Any]:
        raw_meta = elem.get("metadata")
        meta = cast(dict[str, Any], raw_meta) if isinstance(raw_meta, dict) else {}
        out: dict[str, Any] = {}
        for key in (
            "embedding_artifact_ref",
            "embedding_fingerprint",
            "embedding_text_hash",
        ):
            value = elem.get(key)
            if value is None:
                value = meta.get(key)
            if value is not None:
                out[key] = value
        return out

    @classmethod
    def _serialize_element_payload(cls, elem: dict[str, Any]) -> dict[str, Any]:
        raw_meta = elem.get("metadata")
        metadata = (
            dict(cast(dict[str, Any], raw_meta)) if isinstance(raw_meta, dict) else {}
        )
        metadata.update(cls._embedding_ref_metadata(elem))
        payload: dict[str, Any] = {"metadata": cls._json_safe_payload(metadata)}
        for field_name in cls._ELEMENT_JSON_FIELDS:
            if field_name not in elem:
                continue
            payload[field_name] = cls._json_safe_payload(elem.get(field_name))
        return payload

    def upsert_elements(self, snapshot_id: str, elements: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        vector_rows: list[tuple[Any, ...]] = []
        search_rows: list[tuple[Any, ...]] = []
        for elem in elements:
            element_id = str(elem.get("id") or "")
            if not element_id:
                continue
            raw_meta = elem.get("metadata")
            meta = cast(dict[str, Any], raw_meta) if isinstance(raw_meta, dict) else {}
            embedding: Any = meta.get("embedding")
            if embedding is None:
                embedding = elem.get("embedding")
            repo_name: str = cast(str, elem.get("repo_name") or meta.get("repo_name"))
            relative_path = elem.get("relative_path")
            language = elem.get("language")
            element_type = elem.get("type")
            text_content = " ".join(
                [
                    str(elem.get("name") or ""),
                    str(elem.get("summary") or ""),
                    str(elem.get("signature") or ""),
                    str(elem.get("docstring") or ""),
                    str(elem.get("code") or "")[:2000],
                ]
            )

            serializable_elem = self._serialize_element_payload(elem)
            metadata_json = json.dumps(serializable_elem, ensure_ascii=False)

            embedding_array = self._vector_array(embedding)
            embedding_param = self._vector_parameter(embedding_array)

            vector_rows.append(
                (
                    snapshot_id,
                    element_id,
                    repo_name,
                    relative_path,
                    language,
                    element_type,
                    embedding_param,
                    None,
                    metadata_json,
                )
            )
            search_rows.append(
                (
                    snapshot_id,
                    element_id,
                    repo_name,
                    relative_path,
                    language,
                    element_type,
                    text_content,
                    metadata_json,
                )
            )
        if not vector_rows:
            self.last_upsert_metrics = {
                "row_count": 0,
                "batch_count": 0,
                "vector_adapter_path": "none",
            }
            return
        try:
            with self.db_runtime.connect() as conn:
                cur = conn.cursor()
                cur.executemany(
                    """
                    INSERT INTO embedding_vectors (
                        snapshot_id, element_id, repo_name, relative_path, language, element_type,
                        embedding, embedding_arr, metadata_json, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s::jsonb, NOW())
                    ON CONFLICT (snapshot_id, element_id) DO UPDATE SET
                        repo_name=EXCLUDED.repo_name,
                        relative_path=EXCLUDED.relative_path,
                        language=EXCLUDED.language,
                        element_type=EXCLUDED.element_type,
                        embedding=EXCLUDED.embedding,
                        embedding_arr=EXCLUDED.embedding_arr,
                        metadata_json=EXCLUDED.metadata_json,
                        updated_at=EXCLUDED.updated_at
                    """,
                    vector_rows,
                )
                cur.executemany(
                    """
                    INSERT INTO search_documents (
                        snapshot_id, element_id, repo_name, relative_path, language, element_type,
                        text_content, metadata_json, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
                    ON CONFLICT (snapshot_id, element_id) DO UPDATE SET
                        repo_name=EXCLUDED.repo_name,
                        relative_path=EXCLUDED.relative_path,
                        language=EXCLUDED.language,
                        element_type=EXCLUDED.element_type,
                        text_content=EXCLUDED.text_content,
                        metadata_json=EXCLUDED.metadata_json,
                        updated_at=EXCLUDED.updated_at
                    """,
                    search_rows,
                )
                conn.commit()
        except Exception:
            self.last_upsert_metrics = {
                "row_count": len(vector_rows),
                "batch_count": 2,
                "error": True,
            }
            raise
        adapter_paths: set[str] = set()
        for row in vector_rows:
            vector_param = row[6]
            if isinstance(vector_param, np.ndarray):
                adapter_paths.add("pgvector_adapter")
            elif isinstance(vector_param, str):
                adapter_paths.add("literal")
            elif vector_param is None:
                adapter_paths.add("none")
        self.last_upsert_metrics = {
            "row_count": len(vector_rows),
            "batch_count": 2,
            "vector_adapter_path": "+".join(sorted(adapter_paths)) or "none",
        }

    def publish_elements_delta(
        self,
        snapshot_id: str,
        *,
        previous_snapshot_id: str,
        changed_paths: list[str],
        removed_paths: list[str],
        elements: list[dict[str, Any]],
    ) -> dict[str, int | str]:
        if not self.enabled:
            self.last_upsert_metrics = {
                "row_count": 0,
                "batch_count": 0,
                "vector_adapter_path": "none",
            }
            return {"mode": "disabled", "copied_rows": 0, "changed_rows": 0}
        excluded_paths = sorted({str(path) for path in changed_paths + removed_paths})
        copied_vectors = 0
        copied_search = 0
        with self.db_runtime.connect() as conn:
            cur = conn.cursor()
            if excluded_paths:
                vector_sql = """
                    INSERT INTO embedding_vectors (
                        snapshot_id, element_id, repo_name, relative_path, language,
                        element_type, embedding, embedding_arr, metadata_json, updated_at
                    )
                    SELECT %s, element_id, repo_name, relative_path, language,
                        element_type, embedding, embedding_arr, metadata_json, NOW()
                    FROM embedding_vectors
                    WHERE snapshot_id=%s
                      AND NOT (relative_path = ANY(%s))
                    ON CONFLICT (snapshot_id, element_id) DO NOTHING
                """
                search_sql = """
                    INSERT INTO search_documents (
                        snapshot_id, element_id, repo_name, relative_path, language,
                        element_type, text_content, metadata_json, updated_at
                    )
                    SELECT %s, element_id, repo_name, relative_path, language,
                        element_type, text_content, metadata_json, NOW()
                    FROM search_documents
                    WHERE snapshot_id=%s
                      AND NOT (relative_path = ANY(%s))
                    ON CONFLICT (snapshot_id, element_id) DO NOTHING
                """
                params: tuple[Any, ...] = (
                    snapshot_id,
                    previous_snapshot_id,
                    excluded_paths,
                )
            else:
                vector_sql = """
                    INSERT INTO embedding_vectors (
                        snapshot_id, element_id, repo_name, relative_path, language,
                        element_type, embedding, embedding_arr, metadata_json, updated_at
                    )
                    SELECT %s, element_id, repo_name, relative_path, language,
                        element_type, embedding, embedding_arr, metadata_json, NOW()
                    FROM embedding_vectors
                    WHERE snapshot_id=%s
                    ON CONFLICT (snapshot_id, element_id) DO NOTHING
                """
                search_sql = """
                    INSERT INTO search_documents (
                        snapshot_id, element_id, repo_name, relative_path, language,
                        element_type, text_content, metadata_json, updated_at
                    )
                    SELECT %s, element_id, repo_name, relative_path, language,
                        element_type, text_content, metadata_json, NOW()
                    FROM search_documents
                    WHERE snapshot_id=%s
                    ON CONFLICT (snapshot_id, element_id) DO NOTHING
                """
                params = (snapshot_id, previous_snapshot_id)
            cur.execute(vector_sql, params)
            copied_vectors = max(0, int(cur.rowcount or 0))
            cur.execute(search_sql, params)
            copied_search = max(0, int(cur.rowcount or 0))
            conn.commit()
        self.upsert_elements(snapshot_id=snapshot_id, elements=elements)
        return {
            "mode": "delta",
            "previous_snapshot_id": previous_snapshot_id,
            "copied_vector_rows": copied_vectors,
            "copied_search_rows": copied_search,
            "changed_rows": len(elements),
            "excluded_path_count": len(excluded_paths),
        }

    def semantic_search(
        self,
        snapshot_id: str,
        query_embedding: Sequence[float],
        *,
        repo_filter: list[str] | None = None,
        element_types: list[str] | None = None,
        top_k: int = 20,
        query_embedding_fingerprint: Mapping[str, Any] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        return self._result_payloads_from_records(
            self.semantic_search_records(
                snapshot_id,
                query_embedding,
                repo_filter=repo_filter,
                element_types=element_types,
                top_k=top_k,
                query_embedding_fingerprint=query_embedding_fingerprint,
            )
        )

    def semantic_search_records(
        self,
        snapshot_id: str,
        query_embedding: Sequence[float],
        *,
        repo_filter: list[str] | None = None,
        element_types: list[str] | None = None,
        top_k: int = 20,
        query_embedding_fingerprint: Mapping[str, Any] | None = None,
    ) -> list[PgRetrievalResultRecord]:
        query_vector = self._vector_array(query_embedding)
        if not self.enabled or query_vector is None or top_k <= 0:
            return []
        vector_param = self._vector_parameter(query_vector)
        fetch_limit = top_k if query_embedding_fingerprint is None else top_k * 8
        fingerprint_params = self._embedding_fingerprint_sql_filter(
            query_embedding_fingerprint
        )

        with self.db_runtime.connect() as conn:
            cur = conn.cursor()
            try:
                if repo_filter and element_types:
                    cur.execute(
                        """
                        SELECT metadata_json, (1 - (embedding <=> %s::vector)) AS score
                        FROM embedding_vectors
                        WHERE snapshot_id=%s
                          AND repo_name = ANY(%s)
                          AND element_type = ANY(%s)
                          AND (
                            %s::jsonb IS NULL
                            OR metadata_json @> %s::jsonb
                            OR (metadata_json->'metadata') @> %s::jsonb
                          )
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            vector_param,
                            snapshot_id,
                            repo_filter,
                            element_types,
                            *fingerprint_params,
                            vector_param,
                            fetch_limit,
                        ),
                    )
                elif repo_filter:
                    cur.execute(
                        """
                        SELECT metadata_json, (1 - (embedding <=> %s::vector)) AS score
                        FROM embedding_vectors
                        WHERE snapshot_id=%s
                          AND repo_name = ANY(%s)
                          AND (
                            %s::jsonb IS NULL
                            OR metadata_json @> %s::jsonb
                            OR (metadata_json->'metadata') @> %s::jsonb
                          )
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            vector_param,
                            snapshot_id,
                            repo_filter,
                            *fingerprint_params,
                            vector_param,
                            fetch_limit,
                        ),
                    )
                elif element_types:
                    cur.execute(
                        """
                        SELECT metadata_json, (1 - (embedding <=> %s::vector)) AS score
                        FROM embedding_vectors
                        WHERE snapshot_id=%s
                          AND element_type = ANY(%s)
                          AND (
                            %s::jsonb IS NULL
                            OR metadata_json @> %s::jsonb
                            OR (metadata_json->'metadata') @> %s::jsonb
                          )
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            vector_param,
                            snapshot_id,
                            element_types,
                            *fingerprint_params,
                            vector_param,
                            fetch_limit,
                        ),
                    )
                else:
                    cur.execute(
                        """
                        SELECT metadata_json, (1 - (embedding <=> %s::vector)) AS score
                        FROM embedding_vectors
                        WHERE snapshot_id=%s
                          AND (
                            %s::jsonb IS NULL
                            OR metadata_json @> %s::jsonb
                            OR (metadata_json->'metadata') @> %s::jsonb
                          )
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            vector_param,
                            snapshot_id,
                            *fingerprint_params,
                            vector_param,
                            fetch_limit,
                        ),
                    )
                return self._metadata_score_records(
                    cur.fetchall(),
                    query_embedding_fingerprint=query_embedding_fingerprint,
                )[:top_k]
            except Exception as exc:
                self.logger.debug(
                    "pgvector query failed, falling back to client-side cosine: %s", exc
                )
                return self._semantic_search_fallback_records(
                    cur,
                    snapshot_id,
                    query_vector,
                    repo_filter=repo_filter,
                    element_types=element_types,
                    top_k=top_k,
                    query_embedding_fingerprint=query_embedding_fingerprint,
                )

    def _semantic_search_fallback(
        self,
        cur: Any,
        snapshot_id: str,
        query_vector: np.ndarray,
        *,
        repo_filter: list[str] | None = None,
        element_types: list[str] | None = None,
        top_k: int = 20,
        query_embedding_fingerprint: Mapping[str, Any] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        return self._result_payloads_from_records(
            self._semantic_search_fallback_records(
                cur,
                snapshot_id,
                query_vector,
                repo_filter=repo_filter,
                element_types=element_types,
                top_k=top_k,
                query_embedding_fingerprint=query_embedding_fingerprint,
            )
        )

    def _semantic_search_fallback_records(
        self,
        cur: Any,
        snapshot_id: str,
        query_vector: np.ndarray,
        *,
        repo_filter: list[str] | None = None,
        element_types: list[str] | None = None,
        top_k: int = 20,
        query_embedding_fingerprint: Mapping[str, Any] | None = None,
    ) -> list[PgRetrievalResultRecord]:
        # Fallback: compute cosine with embedding_arr client-side. Keep metadata
        # raw until ranking picks the rows that must cross the Python boundary.
        limit = max(top_k, 1) * 8
        if repo_filter and element_types:
            cur.execute(
                """
                SELECT metadata_json, embedding, embedding_arr, repo_name, element_type
                FROM embedding_vectors
                WHERE snapshot_id=%s
                  AND repo_name = ANY(%s)
                  AND element_type = ANY(%s)
                LIMIT %s
                """,
                (snapshot_id, repo_filter, element_types, limit),
            )
        elif repo_filter:
            cur.execute(
                """
                SELECT metadata_json, embedding, embedding_arr, repo_name, element_type
                FROM embedding_vectors
                WHERE snapshot_id=%s
                  AND repo_name = ANY(%s)
                LIMIT %s
                """,
                (snapshot_id, repo_filter, limit),
            )
        elif element_types:
            cur.execute(
                """
                SELECT metadata_json, embedding, embedding_arr, repo_name, element_type
                FROM embedding_vectors
                WHERE snapshot_id=%s
                  AND element_type = ANY(%s)
                LIMIT %s
                """,
                (snapshot_id, element_types, limit),
            )
        else:
            cur.execute(
                """
                SELECT metadata_json, embedding, embedding_arr, repo_name, element_type
                FROM embedding_vectors
                WHERE snapshot_id=%s
                LIMIT %s
                """,
                (snapshot_id, limit),
            )

        query_norm = float(np.linalg.norm(query_vector))
        if query_norm <= 0.0:
            return []
        allowed_types = set(element_types) if element_types else None
        allowed_repos = set(repo_filter) if repo_filter else None
        raw_metadata_by_index: list[Any] = []
        vectors: list[np.ndarray] = []

        for row in cur.fetchall():
            raw_metadata = self._row_value(row, 0, "metadata_json")
            if allowed_repos:
                row_repo = self._row_filter_value(
                    row, 3, "repo_name", raw_metadata, "repo_name"
                )
                if row_repo is not None and row_repo not in allowed_repos:
                    continue
            if allowed_types:
                row_type = self._row_filter_value(
                    row, 4, "element_type", raw_metadata, "type"
                )
                if row_type is not None and row_type not in allowed_types:
                    continue
            embedding = self._vector_array(self._row_value(row, 1, "embedding"))
            if embedding is None:
                embedding = self._vector_array(self._row_value(row, 2, "embedding_arr"))
            if embedding is None or embedding.size != query_vector.size:
                continue
            if query_embedding_fingerprint is not None:
                candidate = self._result_record_from_raw_metadata(
                    raw_metadata,
                    0.0,
                    query_embedding_fingerprint=query_embedding_fingerprint,
                )
                if candidate is None:
                    continue
            raw_metadata_by_index.append(raw_metadata)
            vectors.append(embedding)

        if not vectors:
            return []

        matrix = as_float32_matrix(vectors, copy_policy="contiguous")
        if matrix.size == 0:
            return []
        denominators = np.linalg.norm(matrix, axis=1) * query_norm
        scores = np.divide(
            matrix @ query_vector,
            denominators,
            out=np.zeros(len(vectors), dtype=np.float32),
            where=denominators > 0.0,
        )

        out: list[PgRetrievalResultRecord] = []
        for raw_index in np.argsort(scores)[::-1]:
            index = int(raw_index)
            result = self._result_record_from_raw_metadata(
                raw_metadata_by_index[index],
                float(scores[index]),
                query_embedding_fingerprint=query_embedding_fingerprint,
            )
            if result is None:
                continue
            element = result.element
            if allowed_types:
                element_type = element.element_type
                if element_type not in allowed_types:
                    continue
            if allowed_repos:
                repo_name = element.repo_name
                if repo_name not in allowed_repos:
                    continue
            out.append(result)
            if len(out) >= top_k:
                break
        return out

    def keyword_search(
        self,
        snapshot_id: str,
        query: str,
        *,
        repo_filter: list[str] | None = None,
        element_types: list[str] | None = None,
        top_k: int = 20,
    ) -> list[tuple[dict[str, Any], float]]:
        return self._result_payloads_from_records(
            self.keyword_search_records(
                snapshot_id,
                query,
                repo_filter=repo_filter,
                element_types=element_types,
                top_k=top_k,
            )
        )

    def keyword_search_records(
        self,
        snapshot_id: str,
        query: str,
        *,
        repo_filter: list[str] | None = None,
        element_types: list[str] | None = None,
        top_k: int = 20,
    ) -> list[PgRetrievalResultRecord]:
        if not self.enabled or not query.strip():
            return []
        try:
            return self._keyword_search_records_inner(
                snapshot_id,
                query,
                repo_filter=repo_filter,
                element_types=element_types,
                top_k=top_k,
            )
        except Exception as exc:
            self.logger.warning("keyword_search failed: %s", exc)
            return []

    def _keyword_search_inner(
        self,
        snapshot_id: str,
        query: str,
        *,
        repo_filter: list[str] | None = None,
        element_types: list[str] | None = None,
        top_k: int = 20,
    ) -> list[tuple[dict[str, Any], float]]:
        return self._result_payloads_from_records(
            self._keyword_search_records_inner(
                snapshot_id,
                query,
                repo_filter=repo_filter,
                element_types=element_types,
                top_k=top_k,
            )
        )

    def _keyword_search_records_inner(
        self,
        snapshot_id: str,
        query: str,
        *,
        repo_filter: list[str] | None = None,
        element_types: list[str] | None = None,
        top_k: int = 20,
    ) -> list[PgRetrievalResultRecord]:
        with self.db_runtime.connect() as conn:
            cur = conn.cursor()
            if repo_filter and element_types:
                cur.execute(
                    """
                    SELECT metadata_json, ts_rank(search_tsv, plainto_tsquery('english', %s)) AS score
                    FROM search_documents
                    WHERE snapshot_id=%s
                      AND repo_name = ANY(%s)
                      AND element_type = ANY(%s)
                      AND search_tsv @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, snapshot_id, repo_filter, element_types, query, top_k),
                )
            elif repo_filter:
                cur.execute(
                    """
                    SELECT metadata_json, ts_rank(search_tsv, plainto_tsquery('english', %s)) AS score
                    FROM search_documents
                    WHERE snapshot_id=%s
                      AND repo_name = ANY(%s)
                      AND search_tsv @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, snapshot_id, repo_filter, query, top_k),
                )
            elif element_types:
                cur.execute(
                    """
                    SELECT metadata_json, ts_rank(search_tsv, plainto_tsquery('english', %s)) AS score
                    FROM search_documents
                    WHERE snapshot_id=%s
                      AND element_type = ANY(%s)
                      AND search_tsv @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, snapshot_id, element_types, query, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT metadata_json, ts_rank(search_tsv, plainto_tsquery('english', %s)) AS score
                    FROM search_documents
                    WHERE snapshot_id=%s
                      AND search_tsv @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, snapshot_id, query, top_k),
                )
            return self._metadata_score_records(cur.fetchall())
