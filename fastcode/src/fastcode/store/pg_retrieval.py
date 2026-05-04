"""
PostgreSQL retrieval store using pgvector + FTS, with graceful fallback.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from ..db_runtime import DBRuntime


class PgRetrievalStore:
    def __init__(self, db_runtime: DBRuntime, config: dict[str, Any]) -> None:
        self.db_runtime = db_runtime
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.enabled = self.db_runtime.backend == "postgres"
        self.backend_mode = config.get("retrieval", {}).get("backend", "pg_hybrid")
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
        try:
            array = np.asarray(vec, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if array.size == 0:
            return None
        if not bool(np.isfinite(array).all()):
            array = np.nan_to_num(array, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        return array

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
        if isinstance(raw_metadata, dict):
            return cast(dict[str, Any], raw_metadata)
        if not raw_metadata:
            return None
        try:
            parsed = json.loads(str(raw_metadata))
        except (json.JSONDecodeError, TypeError):
            return None
        return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None

    @classmethod
    def _metadata_score_rows(cls, rows: Any) -> list[tuple[dict[str, Any], float]]:
        out: list[tuple[dict[str, Any], float]] = []
        for row in rows:
            raw_metadata = cls._row_value(row, 0, "metadata_json")
            metadata = cls._decode_metadata(raw_metadata)
            if metadata is None:
                continue
            raw_score = cls._row_value(row, 1, "score")
            out.append((metadata, float(raw_score or 0.0)))
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
    def _json_safe_payload(value: Any) -> Any:
        if isinstance(value, dict):
            payload: dict[str, Any] = {}
            mapping = cast(dict[Any, Any], value)
            for key, item in mapping.items():
                key_str = str(key)
                if key_str == "embedding":
                    continue
                payload[key_str] = PgRetrievalStore._json_safe_payload(item)
            return payload
        if isinstance(value, list):
            sequence = cast(Sequence[Any], value)
            return [PgRetrievalStore._json_safe_payload(item) for item in sequence]
        if isinstance(value, tuple):
            sequence = cast(Sequence[Any], value)
            return [PgRetrievalStore._json_safe_payload(item) for item in sequence]
        if isinstance(value, set):
            items = cast(set[Any], value)
            return [PgRetrievalStore._json_safe_payload(item) for item in items]
        if isinstance(value, np.ndarray):
            return [
                PgRetrievalStore._json_safe_payload(item)
                for item in value.astype(np.float32).tolist()
            ]
        if isinstance(value, np.integer):  # type: ignore[arg-type]
            return int(cast(Any, value))
        if isinstance(value, np.floating):  # type: ignore[arg-type]
            return float(cast(Any, value))
        if isinstance(value, np.bool_):  # type: ignore[arg-type]
            return bool(cast(Any, value))
        return value

    def upsert_elements(self, snapshot_id: str, elements: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        with self.db_runtime.connect() as conn:
            cur = conn.cursor()
            for elem in elements:
                element_id = str(elem.get("id") or "")
                if not element_id:
                    continue
                meta = cast(dict[str, Any], elem.get("metadata") or {})
                embedding: Any = meta.get("embedding")
                repo_name: str = cast(
                    str, elem.get("repo_name") or meta.get("repo_name")
                )
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

                serializable_elem = self._json_safe_payload(elem)
                metadata_json = json.dumps(serializable_elem, ensure_ascii=False)

                vector_literal: str | None = None
                embedding_arr: list[float] | None = None
                embedding_array = self._vector_array(embedding)
                if embedding_array is not None:
                    vector_literal = self._vector_literal_from_array(embedding_array)
                    embedding_arr = [float(value) for value in embedding_array]

                cur.execute(
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
                    (
                        snapshot_id,
                        element_id,
                        repo_name,
                        relative_path,
                        language,
                        element_type,
                        vector_literal,
                        embedding_arr,
                        metadata_json,
                    ),
                )
                cur.execute(
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
                    (
                        snapshot_id,
                        element_id,
                        repo_name,
                        relative_path,
                        language,
                        element_type,
                        text_content,
                        metadata_json,
                    ),
                )
            conn.commit()

    def semantic_search(
        self,
        snapshot_id: str,
        query_embedding: Sequence[float],
        *,
        repo_filter: list[str] | None = None,
        element_types: list[str] | None = None,
        top_k: int = 20,
    ) -> list[tuple[dict[str, Any], float]]:
        query_vector = self._vector_array(query_embedding)
        if not self.enabled or query_vector is None or top_k <= 0:
            return []
        vector_literal = self._vector_literal_from_array(query_vector)

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
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            vector_literal,
                            snapshot_id,
                            repo_filter,
                            element_types,
                            vector_literal,
                            top_k,
                        ),
                    )
                elif repo_filter:
                    cur.execute(
                        """
                        SELECT metadata_json, (1 - (embedding <=> %s::vector)) AS score
                        FROM embedding_vectors
                        WHERE snapshot_id=%s
                          AND repo_name = ANY(%s)
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            vector_literal,
                            snapshot_id,
                            repo_filter,
                            vector_literal,
                            top_k,
                        ),
                    )
                elif element_types:
                    cur.execute(
                        """
                        SELECT metadata_json, (1 - (embedding <=> %s::vector)) AS score
                        FROM embedding_vectors
                        WHERE snapshot_id=%s
                          AND element_type = ANY(%s)
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            vector_literal,
                            snapshot_id,
                            element_types,
                            vector_literal,
                            top_k,
                        ),
                    )
                else:
                    cur.execute(
                        """
                        SELECT metadata_json, (1 - (embedding <=> %s::vector)) AS score
                        FROM embedding_vectors
                        WHERE snapshot_id=%s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (vector_literal, snapshot_id, vector_literal, top_k),
                    )
                return self._metadata_score_rows(cur.fetchall())
            except Exception as exc:
                self.logger.debug(
                    "pgvector query failed, falling back to client-side cosine: %s", exc
                )
                return self._semantic_search_fallback(
                    cur,
                    snapshot_id,
                    query_vector,
                    repo_filter=repo_filter,
                    element_types=element_types,
                    top_k=top_k,
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
    ) -> list[tuple[dict[str, Any], float]]:
        # Fallback: compute cosine with embedding_arr client-side. Keep metadata
        # raw until ranking picks the rows that must cross the Python boundary.
        limit = max(top_k, 1) * 8
        if repo_filter and element_types:
            cur.execute(
                """
                SELECT metadata_json, embedding_arr, repo_name, element_type
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
                SELECT metadata_json, embedding_arr, repo_name, element_type
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
                SELECT metadata_json, embedding_arr, repo_name, element_type
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
                SELECT metadata_json, embedding_arr, repo_name, element_type
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
                    row, 2, "repo_name", raw_metadata, "repo_name"
                )
                if row_repo is not None and row_repo not in allowed_repos:
                    continue
            if allowed_types:
                row_type = self._row_filter_value(
                    row, 3, "element_type", raw_metadata, "type"
                )
                if row_type is not None and row_type not in allowed_types:
                    continue
            embedding = self._vector_array(self._row_value(row, 1, "embedding_arr"))
            if embedding is None or embedding.size != query_vector.size:
                continue
            raw_metadata_by_index.append(raw_metadata)
            vectors.append(embedding)

        if not vectors:
            return []

        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        denominators = np.linalg.norm(matrix, axis=1) * query_norm
        scores = np.divide(
            matrix @ query_vector,
            denominators,
            out=np.zeros(len(vectors), dtype=np.float32),
            where=denominators > 0.0,
        )

        out: list[tuple[dict[str, Any], float]] = []
        for raw_index in np.argsort(scores)[::-1]:
            index = int(raw_index)
            metadata = self._decode_metadata(raw_metadata_by_index[index])
            if metadata is None:
                continue
            if allowed_types:
                meta_type = metadata.get("type") or metadata.get("element_type")
                if meta_type not in allowed_types:
                    continue
            if allowed_repos:
                meta_repo = metadata.get("repo_name")
                if meta_repo not in allowed_repos:
                    continue
            out.append((metadata, float(scores[index])))
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
        if not self.enabled or not query.strip():
            return []
        try:
            return self._keyword_search_inner(
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
            return self._metadata_score_rows(cur.fetchall())
