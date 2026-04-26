"""
PostgreSQL retrieval store using pgvector + FTS, with graceful fallback.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .db_runtime import DBRuntime


class PgRetrievalStore:
    def __init__(self, db_runtime: DBRuntime, config: dict[str, Any]):
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
    def _vector_literal(vec: Sequence[float]) -> str:
        if not vec:
            raise ValueError("Cannot create vector literal from empty sequence")
        cleaned = []
        for v in vec:
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                f = 0.0
            cleaned.append(f)
        return "[" + ",".join(f"{v:.8f}" for v in cleaned) + "]"

    def upsert_elements(self, snapshot_id: str, elements: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        with self.db_runtime.connect() as conn:
            cur = conn.cursor()
            for elem in elements:
                element_id = str(elem.get("id") or "")
                if not element_id:
                    continue
                meta = elem.get("metadata") or {}
                embedding = meta.get("embedding")
                repo_name = elem.get("repo_name") or meta.get("repo_name")
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
                # Strip numpy arrays and other non-serializable values from metadata
                # before JSON encoding. Embeddings are stored separately in the
                # embedding/embedding_arr columns.
                def _make_json_safe(val):
                    if isinstance(val, np.ndarray):
                        return val.tolist()
                    if isinstance(val, (np.integer,)):
                        return int(val)
                    if isinstance(val, (np.floating,)):
                        return float(val)
                    if isinstance(val, np.bool_):
                        return bool(val)
                    return val

                serializable_elem = {
                    k: _make_json_safe(v)
                    for k, v in elem.items()
                }
                if isinstance(serializable_elem.get("metadata"), dict):
                    serializable_elem["metadata"] = {
                        k: _make_json_safe(v)
                        for k, v in serializable_elem["metadata"].items()
                    }
                metadata_json = json.dumps(serializable_elem, ensure_ascii=False)

                vector_literal = None
                embedding_arr = None
                if isinstance(embedding, (list, tuple)) and embedding:
                    embedding_arr = [float(x) for x in embedding]
                    vector_literal = self._vector_literal(embedding_arr)
                if isinstance(embedding, np.ndarray) and embedding.size > 0:
                    embedding_arr = [float(x) for x in embedding.tolist()]
                    vector_literal = self._vector_literal(embedding_arr)

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
        if not self.enabled or query_embedding is None:
            return []
        vector_literal = self._vector_literal(query_embedding)

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
                        (vector_literal, snapshot_id, repo_filter, element_types, vector_literal, top_k),
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
                        (vector_literal, snapshot_id, repo_filter, vector_literal, top_k),
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
                        (vector_literal, snapshot_id, element_types, vector_literal, top_k),
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
                rows = cur.fetchall()
                out = []
                for row in rows:
                    raw_meta = row.get("metadata_json") if isinstance(row, dict) else row[0]
                    raw_score = row.get("score") if isinstance(row, dict) else row[1]
                    if isinstance(raw_meta, dict):
                        meta = raw_meta
                    elif raw_meta:
                        try:
                            meta = json.loads(raw_meta)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    else:
                        continue
                    out.append((meta, float(raw_score or 0.0)))
                return out
            except Exception as exc:
                self.logger.debug("pgvector query failed, falling back to client-side cosine: %s", exc)
                # Fallback: compute cosine with embedding_arr client-side.
                if repo_filter and element_types:
                    cur.execute(
                        """
                        SELECT metadata_json, embedding_arr
                        FROM embedding_vectors
                        WHERE snapshot_id=%s
                          AND repo_name = ANY(%s)
                          AND element_type = ANY(%s)
                        LIMIT %s
                        """,
                        (snapshot_id, repo_filter, element_types, top_k * 8),
                    )
                elif repo_filter:
                    cur.execute(
                        """
                        SELECT metadata_json, embedding_arr
                        FROM embedding_vectors
                        WHERE snapshot_id=%s
                          AND repo_name = ANY(%s)
                        LIMIT %s
                        """,
                        (snapshot_id, repo_filter, top_k * 8),
                    )
                elif element_types:
                    cur.execute(
                        """
                        SELECT metadata_json, embedding_arr
                        FROM embedding_vectors
                        WHERE snapshot_id=%s
                          AND element_type = ANY(%s)
                        LIMIT %s
                        """,
                        (snapshot_id, element_types, top_k * 8),
                    )
                else:
                    cur.execute(
                        """
                        SELECT metadata_json, embedding_arr
                        FROM embedding_vectors
                        WHERE snapshot_id=%s
                        LIMIT %s
                        """,
                        (snapshot_id, top_k * 8),
                    )
                q = np.array(query_embedding, dtype=np.float32)
                q_norm = float(np.linalg.norm(q)) or 1.0
                scored = []
                allowed_types = set(element_types) if element_types else None
                allowed_repos = set(repo_filter) if repo_filter else None
                for row in cur.fetchall():
                    meta_raw, emb_arr = row
                    if not emb_arr:
                        continue
                    emb = np.array(emb_arr, dtype=np.float32)
                    denom = (float(np.linalg.norm(emb)) or 1.0) * q_norm
                    score = float(np.dot(emb, q) / denom)
                    meta = meta_raw if isinstance(meta_raw, dict) else json.loads(meta_raw)
                    if allowed_types:
                        meta_type = meta.get("type") or meta.get("element_type")
                        if meta_type not in allowed_types:
                            continue
                    if allowed_repos:
                        meta_repo = meta.get("repo_name")
                        if meta_repo not in allowed_repos:
                            continue
                    scored.append((meta, score))
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[:top_k]

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
            return self._keyword_search_inner(snapshot_id, query, repo_filter=repo_filter, element_types=element_types, top_k=top_k)
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
            out = []
            for row in cur.fetchall():
                raw_meta = row.get("metadata_json") if isinstance(row, dict) else row[0]
                raw_score = row.get("score") if isinstance(row, dict) else row[1]
                if isinstance(raw_meta, dict):
                    meta = raw_meta
                elif raw_meta:
                    try:
                        meta = json.loads(raw_meta)
                    except (json.JSONDecodeError, TypeError):
                        continue
                else:
                    continue
                out.append((meta, float(raw_score or 0.0)))
            return out
