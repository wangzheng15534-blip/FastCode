"""
Optional LadybugDB graph runtime overlay.

PostgreSQL remains source-of-truth. This runtime is best-effort and optional.
Uses LadybugDB's Cypher-like query language for graph storage.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _esc(val: Any) -> str:
    """Escape a value for inline Cypher property strings."""
    if val is None:
        return "NULL"
    s = str(val)
    s = s.replace("\x00", "").replace("\n", " ").replace("\r", " ")
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return f'"{s}"'


class LadybugGraphRuntime:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        graph_cfg = (config.get("graph", {}) or {}).get("ladybug", {}) or {}
        self.enabled = bool(graph_cfg.get("enabled", False))
        self.db_path = graph_cfg.get("db_path", "./data/ladybug/fastcode.lb")
        self.postgres_attach_dsn = graph_cfg.get("postgres_attach_dsn", "")
        self._conn = None
        if self.enabled:
            self._init()

    def _init(self) -> None:
        try:
            from real_ladybug import Connection, Database  # type: ignore

            db = Database(self.db_path)
            self._conn = Connection(database=db)
            self._create_schema()
            if self.postgres_attach_dsn:
                self._attach_postgres(self.postgres_attach_dsn)
            self.logger.info("Ladybug runtime initialized")
        except Exception as e:
            self.logger.warning(f"Ladybug runtime unavailable, disabling overlay: {e}")
            self.enabled = False
            self._conn = None

    def _create_schema(self) -> None:
        if not self._conn:
            return
        # LadybugDB uses node/rel tables (not SQL tables).
        # Create one at a time; IF NOT EXISTS prevents errors on re-init.
        node_tables = [
            (
                "CREATE NODE TABLE IF NOT EXISTS design_documents "
                "(chunk_id STRING, snapshot_id STRING, repo_name STRING, "
                "path STRING, title STRING, heading STRING, doc_type STRING, "
                "content STRING, PRIMARY KEY(chunk_id))"
            ),
            (
                "CREATE NODE TABLE IF NOT EXISTS mentions "
                "(mention_id STRING, chunk_id STRING, symbol_id STRING, "
                "confidence STRING, PRIMARY KEY(mention_id))"
            ),
        ]
        for s in node_tables:
            try:
                self._conn.execute(s)
            except Exception as e:
                self.logger.warning(f"Ladybug schema statement failed: {e}; sql={s}")

        # Create rel table only after both node tables exist.
        try:
            self._conn.execute(
                "CREATE REL TABLE IF NOT EXISTS has_mention "
                "(FROM design_documents TO mentions)"
            )
        except Exception as e:
            self.logger.warning(f"Ladybug rel table creation failed: {e}")

    @staticmethod
    def _sanitize_attach_dsn(dsn: str) -> str:
        raw = (dsn or "").strip()
        if not raw:
            raise ValueError("empty postgres attach dsn")
        # Reject SQL-control characters/tokens to prevent statement injection.
        forbidden_tokens = (";", "--", "/*", "*/", "\x00", "\n", "\r")
        if any(tok in raw for tok in forbidden_tokens):
            raise ValueError("unsafe postgres attach dsn")
        if "://" in raw:
            parsed = urlparse(raw)
            if parsed.scheme not in {"postgres", "postgresql"}:
                raise ValueError("unsupported postgres attach dsn scheme")
        elif not re.fullmatch(r"[A-Za-z0-9_\-.:/@?&=%+, ]+", raw):
            raise ValueError("unsupported postgres attach dsn format")
        # Escape single quotes for SQL literal context.
        return raw.replace("'", "''")

    def _attach_postgres(self, dsn: str) -> None:
        if not self._conn:
            return
        try:
            safe_dsn = self._sanitize_attach_dsn(dsn)
            self._conn.execute(f"ATTACH '{safe_dsn}' AS pg (dbtype postgres)")
        except Exception as e:
            self.logger.warning(f"Ladybug ATTACH postgres failed: {e}")

    def sync_docs(
        self,
        *,
        chunks: Iterable[dict[str, Any]],
        mentions: Iterable[dict[str, Any]],
    ) -> bool:
        """Sync doc chunks and mentions to LadybugDB as graph nodes."""
        if not self.enabled or not self._conn:
            return False
        try:
            for c in chunks:
                chunk_id = c.get("chunk_id", "")
                # Upsert: delete old node then create new one
                self._conn.execute(
                    f"MATCH (d:design_documents {{chunk_id: {_esc(chunk_id)}}}) DELETE d"
                )
                self._conn.execute(
                    "CREATE (d:design_documents {"
                    f"chunk_id: {_esc(chunk_id)}, "
                    f"snapshot_id: {_esc(c.get('snapshot_id'))}, "
                    f"repo_name: {_esc(c.get('repo_name'))}, "
                    f"path: {_esc(c.get('path'))}, "
                    f"title: {_esc(c.get('title'))}, "
                    f"heading: {_esc(c.get('heading'))}, "
                    f"doc_type: {_esc(c.get('doc_type'))}, "
                    f"content: {_esc(c.get('content'))}"
                    "})"
                )
            for m in mentions:
                # Generate synthetic mention_id from composite key
                mention_id = f"{m.get('chunk_id', '')}:{m.get('symbol_id', '')}"
                # Upsert: delete old then create new
                self._conn.execute(
                    f"MATCH (mt:mentions {{mention_id: {_esc(mention_id)}}}) DELETE mt"
                )
                self._conn.execute(
                    "CREATE (mt:mentions {"
                    f"mention_id: {_esc(mention_id)}, "
                    f"chunk_id: {_esc(m.get('chunk_id'))}, "
                    f"symbol_id: {_esc(m.get('symbol_id'))}, "
                    f"confidence: {_esc(m.get('confidence'))}"
                    "})"
                )
            return True
        except Exception as e:
            self.logger.warning(f"Ladybug sync failed: {e}")
            return False

    def query_docs(self, *, snapshot_id: str) -> list[dict[str, Any]]:
        """Query doc chunks from LadybugDB by snapshot_id."""
        if not self.enabled or not self._conn:
            return []
        try:
            result = self._conn.execute(
                f"MATCH (d:design_documents {{snapshot_id: {_esc(snapshot_id)}}}) "
                "RETURN d.chunk_id, d.heading, d.doc_type, d.content"
            )
            rows = []
            for row in result:
                rows.append({
                    "chunk_id": row[0],
                    "heading": row[1],
                    "doc_type": row[2],
                    "content": row[3],
                })
            return rows
        except Exception as e:
            self.logger.warning(f"Ladybug query failed: {e}")
            return []

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
