"""
Optional Ladybug graph runtime overlay.

PostgreSQL remains source-of-truth. This runtime is best-effort and optional.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Iterable
from urllib.parse import urlparse


class LadybugGraphRuntime:
    def __init__(self, config: Dict[str, Any]):
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
            import ladybugdb  # type: ignore

            ensure_dir = os.path.dirname(os.path.abspath(self.db_path))
            if ensure_dir:
                os.makedirs(ensure_dir, exist_ok=True)
            self._conn = ladybugdb.connect(self.db_path)
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
        stmts = [
            "CREATE TABLE IF NOT EXISTS design_documents (chunk_id TEXT PRIMARY KEY, snapshot_id TEXT, repo_name TEXT, path TEXT, title TEXT, heading TEXT, doc_type TEXT, content TEXT)",
            "CREATE TABLE IF NOT EXISTS mentions (chunk_id TEXT, symbol_id TEXT, confidence TEXT)",
        ]
        for s in stmts:
            try:
                self._conn.execute(s)
            except Exception as e:
                self.logger.warning(f"Ladybug schema statement failed: {e}; sql={s}")

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
        else:
            if not re.fullmatch(r"[A-Za-z0-9_\-.:/@?&=%+, ]+", raw):
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
        chunks: Iterable[Dict[str, Any]],
        mentions: Iterable[Dict[str, Any]],
    ) -> bool:
        if not self.enabled or not self._conn:
            return False
        try:
            for c in chunks:
                self._conn.execute(
                    "INSERT OR REPLACE INTO design_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        c.get("chunk_id"),
                        c.get("snapshot_id"),
                        c.get("repo_name"),
                        c.get("path"),
                        c.get("title"),
                        c.get("heading"),
                        c.get("doc_type"),
                        c.get("content"),
                    ],
                )
            for m in mentions:
                self._conn.execute(
                    "INSERT INTO mentions VALUES (?, ?, ?)",
                    [m.get("chunk_id"), m.get("symbol_id"), m.get("confidence")],
                )
            return True
        except Exception as e:
            self.logger.warning(f"Ladybug sync failed: {e}")
            return False

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
