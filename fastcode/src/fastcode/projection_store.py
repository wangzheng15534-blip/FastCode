"""
PostgreSQL-backed projection artifact store.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .projection_models import ProjectionBuildResult, ProjectionScope
from .utils import utc_now

try:
    import psycopg
except Exception:  # pragma: no cover - optional dependency
    psycopg = None

try:
    from psycopg_pool import ConnectionPool
except Exception:  # pragma: no cover - optional dependency
    ConnectionPool = None


class ProjectionStore:
    def __init__(self, config: dict[str, Any]) -> None:
        self.logger = logging.getLogger(__name__)
        proj_cfg = config.get("projection", {})
        storage_cfg = config.get("storage", {})
        self.dsn = (
            proj_cfg.get("postgres_dsn")
            or storage_cfg.get("postgres_dsn")
            or os.getenv("FASTCODE_PROJECTION_POSTGRES_DSN")
            or os.getenv("FASTCODE_POSTGRES_DSN")
            or ""
        )
        self.enabled = bool(self.dsn)
        self.pool = None
        if self.enabled and psycopg is None:
            raise RuntimeError(
                "projection store requires psycopg; install dependency first"
            )
        if self.enabled:
            pool_min = int(storage_cfg.get("pool_min", 1))
            pool_max = int(storage_cfg.get("pool_max", 8))
            if ConnectionPool is not None:
                self.pool = ConnectionPool(
                    conninfo=self.dsn,
                    min_size=pool_min,
                    max_size=pool_max,
                    kwargs={"autocommit": False},
                )
            self._init_db()

    def _connect(self) -> Any:
        if not self.enabled:
            raise RuntimeError(
                "projection store is not configured (projection.postgres_dsn missing)"
            )
        if self.pool is not None:
            return self.pool.connection()
        if psycopg is None:
            raise RuntimeError("projection store requires psycopg")
        return psycopg.connect(self.dsn, autocommit=False)

    def _init_db(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS projection_builds (
                        projection_id TEXT PRIMARY KEY,
                        snapshot_id TEXT NOT NULL,
                        scope_kind TEXT NOT NULL,
                        scope_key TEXT NOT NULL,
                        params_hash TEXT NOT NULL,
                        status TEXT NOT NULL,
                        warnings_json TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_projection_scope
                    ON projection_builds (snapshot_id, scope_kind, scope_key, params_hash, updated_at DESC)
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS projection_views (
                        projection_id TEXT NOT NULL,
                        layer TEXT NOT NULL,
                        node_json JSONB NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        PRIMARY KEY (projection_id, layer)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS projection_chunks (
                        projection_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        chunk_json JSONB NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        PRIMARY KEY (projection_id, chunk_id)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        component TEXT NOT NULL,
                        version TEXT NOT NULL,
                        applied_at TEXT NOT NULL,
                        PRIMARY KEY (component, version)
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO schema_migrations (component, version, applied_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT(component, version) DO NOTHING
                    """,
                    ("projection_store", "v1", utc_now()),
                )
            conn.commit()

    def find_cached_projection_id(
        self, scope: ProjectionScope, params_hash: str
    ) -> str | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT projection_id
                    FROM projection_builds
                    WHERE snapshot_id=%s
                      AND scope_kind=%s
                      AND scope_key=%s
                      AND params_hash=%s
                      AND status='ready'
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                (scope.snapshot_id, scope.scope_kind, scope.scope_key, params_hash),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def save(self, result: ProjectionBuildResult, params_hash: str) -> None:
        now = utc_now()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO projection_builds (
                        projection_id, snapshot_id, scope_kind, scope_key, params_hash,
                        status, warnings_json, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (projection_id) DO UPDATE SET
                        snapshot_id=EXCLUDED.snapshot_id,
                        scope_kind=EXCLUDED.scope_kind,
                        scope_key=EXCLUDED.scope_key,
                        params_hash=EXCLUDED.params_hash,
                        status=EXCLUDED.status,
                        warnings_json=EXCLUDED.warnings_json,
                        updated_at=EXCLUDED.updated_at
                    """,
                    (
                        result.projection_id,
                        result.snapshot_id,
                        result.scope_kind,
                        result.scope_key,
                        params_hash,
                        "ready",
                        json.dumps(result.warnings or [], ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                for layer, payload in [
                    ("L0", result.l0),
                    ("L1", result.l1),
                    ("L2", result.l2_index),
                ]:
                    cur.execute(
                        """
                        INSERT INTO projection_views (projection_id, layer, node_json, updated_at)
                        VALUES (%s, %s, %s::jsonb, %s)
                        ON CONFLICT (projection_id, layer) DO UPDATE SET
                            node_json=EXCLUDED.node_json,
                            updated_at=EXCLUDED.updated_at
                        """,
                        (
                            result.projection_id,
                            layer,
                            json.dumps(payload, ensure_ascii=False),
                            now,
                        ),
                    )
                for chunk in result.chunks:
                    cur.execute(
                        """
                        INSERT INTO projection_chunks (projection_id, chunk_id, chunk_json, updated_at)
                        VALUES (%s, %s, %s::jsonb, %s)
                        ON CONFLICT (projection_id, chunk_id) DO UPDATE SET
                            chunk_json=EXCLUDED.chunk_json,
                            updated_at=EXCLUDED.updated_at
                        """,
                        (
                            result.projection_id,
                            chunk["chunk_id"],
                            json.dumps(chunk, ensure_ascii=False),
                            now,
                        ),
                    )
            conn.commit()

    def get_layer(self, projection_id: str, layer: str) -> dict[str, Any] | None:
        layer = layer.upper()
        if layer not in {"L0", "L1", "L2"}:
            return None
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT node_json
                    FROM projection_views
                    WHERE projection_id=%s AND layer=%s
                    """,
                (projection_id, layer),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def get_chunk(self, projection_id: str, chunk_id: str) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT chunk_json
                    FROM projection_chunks
                    WHERE projection_id=%s AND chunk_id=%s
                    """,
                (projection_id, chunk_id),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def get_build(self, projection_id: str) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT projection_id, snapshot_id, scope_kind, scope_key, params_hash, status,
                           warnings_json, created_at, updated_at
                    FROM projection_builds
                    WHERE projection_id=%s
                    """,
                (projection_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            warnings = row[6]
            try:
                warnings_parsed = json.loads(warnings) if warnings else []
            except Exception:
                warnings_parsed = [str(warnings)]
            return {
                "projection_id": row[0],
                "snapshot_id": row[1],
                "scope_kind": row[2],
                "scope_key": row[3],
                "params_hash": row[4],
                "status": row[5],
                "warnings": warnings_parsed,
                "created_at": str(row[7]),
                "updated_at": str(row[8]),
            }
