"""
Shared DB runtime for SQLite/PostgreSQL-backed stores.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - optional dependency
    psycopg = None
    dict_row = None

try:
    from psycopg_pool import ConnectionPool
except Exception:  # pragma: no cover - optional dependency
    ConnectionPool = None


class DBRuntime:
    def __init__(
        self,
        *,
        backend: str,
        sqlite_path: Optional[str] = None,
        postgres_dsn: Optional[str] = None,
        pool_min: int = 1,
        pool_max: int = 8,
    ):
        self.backend = (backend or "sqlite").lower()
        self.sqlite_path = sqlite_path
        self.postgres_dsn = postgres_dsn
        self.pool_min = int(pool_min)
        self.pool_max = int(pool_max)
        self.pool = None

        if self.backend == "postgres":
            if not self.postgres_dsn:
                raise RuntimeError("postgres backend selected but no postgres_dsn configured")
            if psycopg is None:
                raise RuntimeError("postgres backend requires psycopg")
            if ConnectionPool is not None:
                self.pool = ConnectionPool(
                    conninfo=self.postgres_dsn,
                    min_size=self.pool_min,
                    max_size=self.pool_max,
                    kwargs={"autocommit": False, "row_factory": dict_row},
                )
            else:
                logger.warning("psycopg_pool not available — PostgreSQL connections will not be pooled")
        else:
            if not self.sqlite_path:
                raise RuntimeError("sqlite backend requires sqlite_path")

    def close(self) -> None:
        """Close the connection pool if it exists."""
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    def __enter__(self) -> "DBRuntime":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @classmethod
    def from_storage_config(cls, *, sqlite_path: str, storage_cfg: Optional[dict]) -> "DBRuntime":
        cfg = storage_cfg or {}
        backend = (cfg.get("backend") or os.getenv("FASTCODE_STORAGE_BACKEND") or "sqlite").lower()
        dsn = cfg.get("postgres_dsn") or os.getenv("FASTCODE_POSTGRES_DSN")
        pool_min = cfg.get("pool_min", 1)
        pool_max = cfg.get("pool_max", 8)
        return cls(
            backend=backend,
            sqlite_path=sqlite_path,
            postgres_dsn=dsn,
            pool_min=pool_min,
            pool_max=pool_max,
        )

    def adapt_sql(self, sql: str) -> str:
        if self.backend != "postgres":
            return sql
        # Quote-aware replacement: only replace ? outside single-quoted strings
        parts = sql.split("'")
        for i in range(0, len(parts), 2):
            parts[i] = parts[i].replace("?", "%s")
        return "'".join(parts)

    @contextlib.contextmanager
    def connect(self) -> Iterator[Any]:
        if self.backend == "postgres":
            if self.pool is not None:
                with self.pool.connection() as conn:
                    yield conn
                return
            conn = psycopg.connect(self.postgres_dsn, autocommit=False, row_factory=dict_row)
            try:
                yield conn
            finally:
                conn.close()
            return

        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
        finally:
            conn.close()

    def execute(self, conn: Any, sql: str, params: tuple = ()):
        cur = conn.cursor()
        cur.execute(self.adapt_sql(sql), params)
        return cur

    @staticmethod
    def row_to_dict(row: Any) -> Optional[dict]:
        if not row:
            return None
        if isinstance(row, dict):
            return row
        return dict(row)

    def begin_write(self, conn: Any) -> None:
        if self.backend == "sqlite":
            conn.execute("BEGIN IMMEDIATE")
        # PostgreSQL with autocommit=False: transaction already implicit, no BEGIN needed

