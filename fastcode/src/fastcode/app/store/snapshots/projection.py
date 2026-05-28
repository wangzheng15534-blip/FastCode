"""
PostgreSQL-backed projection artifact store.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from fastcode.ir.projection import ProjectionBuildResult, ProjectionScope
from fastcode.ports.runtime import Clock
from fastcode.utils.clock import SystemClock, utc_now
from .projection_contracts import ProjectionBuildRecord, ProjectionDirtyScopeRecord

try:
    import psycopg
except Exception:  # pragma: no cover - optional dependency
    psycopg = None

try:
    from psycopg_pool import ConnectionPool
except Exception:  # pragma: no cover - optional dependency
    ConnectionPool = None


class ProjectionStore:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        clock: Clock | None = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.clock = clock or SystemClock()
        proj_cfg = config.get("projection", {})
        storage_cfg = config.get("storage", {})
        self.dsn = proj_cfg.get("postgres_dsn") or storage_cfg.get("postgres_dsn") or ""
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

    def _utc_now(self) -> str:
        clock = getattr(self, "clock", None)
        return clock.utc_now() if clock is not None else utc_now()

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
                        query TEXT,
                        target_id TEXT,
                        filters_json JSONB,
                        coverage_paths_json JSONB,
                        coverage_nodes_json JSONB,
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
                for column_sql in (
                    "ADD COLUMN IF NOT EXISTS query TEXT",
                    "ADD COLUMN IF NOT EXISTS target_id TEXT",
                    "ADD COLUMN IF NOT EXISTS filters_json JSONB",
                    "ADD COLUMN IF NOT EXISTS coverage_paths_json JSONB",
                    "ADD COLUMN IF NOT EXISTS coverage_nodes_json JSONB",
                ):
                    cur.execute(f"ALTER TABLE projection_builds {column_sql}")
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
                    CREATE TABLE IF NOT EXISTS projection_dirty_scopes (
                        snapshot_id TEXT NOT NULL,
                        scope_kind TEXT NOT NULL,
                        scope_key TEXT NOT NULL,
                        dirty_paths JSONB NOT NULL,
                        dirty_units JSONB,
                        dirty_package_roots JSONB,
                        dirty_reason TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        PRIMARY KEY (snapshot_id, scope_kind, scope_key)
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
                    ("projection_store", "v1", self._utc_now()),
                )
            conn.commit()

    @staticmethod
    def _json_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return [value]
            return parsed if isinstance(parsed, list) else [parsed]
        return list(value) if isinstance(value, (tuple, set)) else [value]

    @staticmethod
    def _row_value(row: Any, index: int, key: str) -> Any:
        if row is None:
            return None
        if isinstance(row, Mapping):
            return row.get(key)
        try:
            return row[key]
        except (IndexError, KeyError, TypeError):
            try:
                return row[index]
            except (IndexError, KeyError, TypeError):
                return None

    @classmethod
    def _json_mapping(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items()}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return cls._json_mapping(parsed)
        return {}

    @classmethod
    def _string_list(cls, value: Any) -> list[str]:
        return [str(item) for item in cls._json_list(value) if item is not None]

    def _row_to_dirty_scope_record(self, row: Any) -> ProjectionDirtyScopeRecord | None:
        snapshot_id = self._row_value(row, 0, "snapshot_id")
        if snapshot_id is None:
            return None
        return ProjectionDirtyScopeRecord(
            snapshot_id=str(snapshot_id),
            scope_kind=str(self._row_value(row, 1, "scope_kind") or ""),
            scope_key=str(self._row_value(row, 2, "scope_key") or ""),
            dirty_paths=self._string_list(self._row_value(row, 3, "dirty_paths")),
            dirty_units=self._string_list(self._row_value(row, 4, "dirty_units")),
            dirty_package_roots=self._string_list(
                self._row_value(row, 5, "dirty_package_roots")
            ),
            dirty_reason=str(self._row_value(row, 6, "dirty_reason") or ""),
            created_at=str(self._row_value(row, 7, "created_at") or ""),
            updated_at=str(self._row_value(row, 8, "updated_at") or ""),
        )

    @staticmethod
    def _dirty_scope_payload(record: ProjectionDirtyScopeRecord) -> dict[str, Any]:
        return {
            "snapshot_id": record.snapshot_id,
            "scope_kind": record.scope_kind,
            "scope_key": record.scope_key,
            "dirty_paths": list(record.dirty_paths),
            "dirty_units": list(record.dirty_units),
            "dirty_package_roots": list(record.dirty_package_roots),
            "dirty_reason": record.dirty_reason,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    def get_dirty_scope_record(
        self, snapshot_id: str, scope_kind: str, scope_key: str
    ) -> ProjectionDirtyScopeRecord | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT snapshot_id, scope_kind, scope_key, dirty_paths,
                           dirty_units, dirty_package_roots, dirty_reason,
                           created_at, updated_at
                    FROM projection_dirty_scopes
                    WHERE snapshot_id=%s AND scope_kind=%s AND scope_key=%s
                    """,
                (snapshot_id, scope_kind, scope_key),
            )
            return self._row_to_dirty_scope_record(cur.fetchone())

    def is_dirty(self, snapshot_id: str, scope_kind: str, scope_key: str) -> bool:
        return bool(
            self.get_dirty_scope_record(snapshot_id, scope_kind, scope_key)
            or self.get_dirty_scope_record(snapshot_id, "all", "*")
        )

    def list_dirty_scope_records(
        self, snapshot_id: str
    ) -> list[ProjectionDirtyScopeRecord]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT snapshot_id, scope_kind, scope_key, dirty_paths,
                           dirty_units, dirty_package_roots, dirty_reason,
                           created_at, updated_at
                    FROM projection_dirty_scopes
                    WHERE snapshot_id=%s
                    ORDER BY updated_at DESC
                    """,
                (snapshot_id,),
            )
            return [
                record
                for row in cur.fetchall()
                if (record := self._row_to_dirty_scope_record(row)) is not None
            ]

    def clear_dirty(self, snapshot_id: str, scope_kind: str, scope_key: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM projection_dirty_scopes
                    WHERE snapshot_id=%s AND scope_kind=%s AND scope_key=%s
                    """,
                    (snapshot_id, scope_kind, scope_key),
                )
            conn.commit()

    def mark_dirty(
        self,
        *,
        snapshot_id: str,
        scope_kind: str,
        scope_key: str,
        dirty_paths: list[str],
        dirty_reason: str,
        dirty_units: list[str] | None = None,
        dirty_package_roots: list[str] | None = None,
    ) -> None:
        now = self._utc_now()
        existing = self.get_dirty_scope_record(snapshot_id, scope_kind, scope_key)
        paths = sorted(set(dirty_paths) | set(existing.dirty_paths if existing else []))
        units = sorted(
            set(dirty_units or []) | set(existing.dirty_units if existing else [])
        )
        package_roots = sorted(
            set(dirty_package_roots or [])
            | set(existing.dirty_package_roots if existing else [])
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO projection_dirty_scopes (
                        snapshot_id, scope_kind, scope_key, dirty_paths,
                        dirty_units, dirty_package_roots, dirty_reason,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s)
                    ON CONFLICT (snapshot_id, scope_kind, scope_key) DO UPDATE SET
                        dirty_paths=EXCLUDED.dirty_paths,
                        dirty_units=EXCLUDED.dirty_units,
                        dirty_package_roots=EXCLUDED.dirty_package_roots,
                        dirty_reason=EXCLUDED.dirty_reason,
                        updated_at=EXCLUDED.updated_at
                    """,
                    (
                        snapshot_id,
                        scope_kind,
                        scope_key,
                        json.dumps(paths, ensure_ascii=False),
                        json.dumps(units, ensure_ascii=False),
                        json.dumps(package_roots, ensure_ascii=False),
                        dirty_reason,
                        now,
                        now,
                    ),
                )
            conn.commit()

    def mark_all_dirty(
        self,
        snapshot_id: str,
        dirty_reason: str,
        *,
        dirty_paths: list[str] | None = None,
        dirty_units: list[str] | None = None,
        dirty_package_roots: list[str] | None = None,
    ) -> None:
        self.mark_dirty(
            snapshot_id=snapshot_id,
            scope_kind="all",
            scope_key="*",
            dirty_paths=dirty_paths or [],
            dirty_units=dirty_units,
            dirty_package_roots=dirty_package_roots,
            dirty_reason=dirty_reason,
        )

    def _row_to_build_record(self, row: Any) -> ProjectionBuildRecord | None:
        projection_id = self._row_value(row, 0, "projection_id")
        if projection_id is None:
            return None
        return ProjectionBuildRecord(
            projection_id=str(projection_id),
            snapshot_id=str(self._row_value(row, 1, "snapshot_id") or ""),
            scope_kind=str(self._row_value(row, 2, "scope_kind") or ""),
            scope_key=str(self._row_value(row, 3, "scope_key") or ""),
            params_hash=str(self._row_value(row, 4, "params_hash") or ""),
            status=str(self._row_value(row, 5, "status") or ""),
            warnings=self._string_list(self._row_value(row, 6, "warnings_json")),
            created_at=str(self._row_value(row, 7, "created_at") or ""),
            updated_at=str(self._row_value(row, 8, "updated_at") or ""),
            query=(
                str(query)
                if (query := self._row_value(row, 9, "query")) is not None
                else None
            ),
            target_id=(
                str(target_id)
                if (target_id := self._row_value(row, 10, "target_id")) is not None
                else None
            ),
            filters=self._json_mapping(self._row_value(row, 11, "filters_json")),
            coverage_paths=self._string_list(
                self._row_value(row, 12, "coverage_paths_json")
            ),
            coverage_nodes=self._string_list(
                self._row_value(row, 13, "coverage_nodes_json")
            ),
        )

    @staticmethod
    def _build_payload(record: ProjectionBuildRecord) -> dict[str, Any]:
        return {
            "projection_id": record.projection_id,
            "snapshot_id": record.snapshot_id,
            "scope_kind": record.scope_kind,
            "scope_key": record.scope_key,
            "params_hash": record.params_hash,
            "status": record.status,
            "warnings": list(record.warnings),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "query": record.query,
            "target_id": record.target_id,
            "filters": dict(record.filters),
            "coverage_paths": list(record.coverage_paths),
            "coverage_nodes": list(record.coverage_nodes),
        }

    def list_build_records_for_snapshot(
        self, snapshot_id: str
    ) -> list[ProjectionBuildRecord]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT projection_id, snapshot_id, scope_kind, scope_key,
                           params_hash, status, warnings_json, created_at, updated_at,
                           query, target_id, filters_json, coverage_paths_json,
                           coverage_nodes_json
                    FROM projection_builds
                    WHERE snapshot_id=%s AND status='ready'
                    ORDER BY updated_at DESC
                    """,
                (snapshot_id,),
            )
            return [
                record
                for row in cur.fetchall()
                if (record := self._row_to_build_record(row)) is not None
            ]

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

    @staticmethod
    def _coverage_nodes(result: ProjectionBuildResult) -> list[str]:
        nodes: set[str] = set()
        for layer in (result.l0, result.l1, result.l2_index):
            meta = layer.get("meta") or {}
            for node_id in meta.get("covers_nodes") or []:
                if node_id:
                    nodes.add(str(node_id))
        return sorted(nodes)

    def save(
        self,
        result: ProjectionBuildResult,
        params_hash: str,
        *,
        scope: ProjectionScope,
        coverage_paths: list[str] | None = None,
    ) -> None:
        now = self._utc_now()
        coverage_nodes = self._coverage_nodes(result)
        chunk_ids = sorted(
            {
                str(chunk_id)
                for chunk in result.chunks
                if (chunk_id := chunk.get("chunk_id"))
            }
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO projection_builds (
                        projection_id, snapshot_id, scope_kind, scope_key, params_hash,
                        query, target_id, filters_json, coverage_paths_json,
                        coverage_nodes_json, status, warnings_json, created_at,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                              %s::jsonb, %s, %s, %s, %s)
                    ON CONFLICT (projection_id) DO UPDATE SET
                        snapshot_id=EXCLUDED.snapshot_id,
                        scope_kind=EXCLUDED.scope_kind,
                        scope_key=EXCLUDED.scope_key,
                        params_hash=EXCLUDED.params_hash,
                        query=EXCLUDED.query,
                        target_id=EXCLUDED.target_id,
                        filters_json=EXCLUDED.filters_json,
                        coverage_paths_json=EXCLUDED.coverage_paths_json,
                        coverage_nodes_json=EXCLUDED.coverage_nodes_json,
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
                        scope.query,
                        scope.target_id,
                        json.dumps(scope.filters or {}, ensure_ascii=False),
                        json.dumps(
                            sorted(set(coverage_paths or [])),
                            ensure_ascii=False,
                        ),
                        json.dumps(coverage_nodes, ensure_ascii=False),
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
                if chunk_ids:
                    cur.execute(
                        """
                        DELETE FROM projection_chunks
                        WHERE projection_id=%s
                          AND NOT (chunk_id = ANY(%s::text[]))
                        """,
                        (result.projection_id, chunk_ids),
                    )
                else:
                    cur.execute(
                        """
                        DELETE FROM projection_chunks
                        WHERE projection_id=%s
                        """,
                        (result.projection_id,),
                    )
                for chunk in result.chunks:
                    chunk_id = chunk.get("chunk_id")
                    if not chunk_id:
                        continue
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
                            str(chunk_id),
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

    def get_build_record(self, projection_id: str) -> ProjectionBuildRecord | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT projection_id, snapshot_id, scope_kind, scope_key, params_hash, status,
                           warnings_json, created_at, updated_at, query, target_id,
                           filters_json, coverage_paths_json, coverage_nodes_json
                    FROM projection_builds
                    WHERE projection_id=%s
                    """,
                (projection_id,),
            )
            return self._row_to_build_record(cur.fetchone())
