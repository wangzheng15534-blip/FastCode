"""
Optional LadybugDB graph runtime overlay.

PostgreSQL remains source-of-truth. This runtime is best-effort and optional.
Uses LadybugDB's Cypher-like query language for graph storage.
"""

from __future__ import annotations

import builtins
import logging
import re
from collections.abc import Iterable
from typing import Any, cast
from urllib.parse import parse_qsl, unquote, urlparse

from fastcode.ports.storage import DocumentGraphRuntime

logger = logging.getLogger(__name__)


def _esc(val: Any) -> str:
    """Escape a value for inline Cypher property strings."""
    if val is None:
        return "NULL"
    s = str(val)
    s = s.replace("\x00", "").replace("\n", " ").replace("\r", " ")
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return f'"{s}"'


class LadybugGraphRuntime(DocumentGraphRuntime):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        graph_cfg = cast(dict[str, Any], config.get("graph") or {})
        ladybug_cfg = cast(dict[str, Any], graph_cfg.get("ladybug") or {})
        self.enabled = bool(ladybug_cfg.get("enabled", False))
        self.db_path: str = str(
            ladybug_cfg.get("db_path", "./data/ladybug/fastcode.lb")
        )
        self.postgres_attach_dsn: str = str(ladybug_cfg.get("postgres_attach_dsn", ""))
        self.postgres_attached = False
        self.postgres_attach_error: str | None = None
        if self.postgres_attach_dsn:
            self._doc_table = "fastcode_design_documents"
            self._mention_table = "fastcode_mentions"
            self._mention_rel_table = "fastcode_has_mention"
        else:
            self._doc_table = "design_documents"
            self._mention_table = "mentions"
            self._mention_rel_table = "has_mention"
        self._conn: Any = None
        if self.enabled:
            self._init()

    def _init(self) -> None:
        try:
            import real_ladybug  # type: ignore[import-untyped]

            # ruff-format: off
            _Database: Any = builtins.getattr(real_ladybug, "Database")  # noqa: B009
            _Connection: Any = builtins.getattr(real_ladybug, "Connection")  # noqa: B009
            # ruff-format: on
            db: Any = _Database(self.db_path)
            self._conn = _Connection(database=db)
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
                f"CREATE NODE TABLE IF NOT EXISTS {self._doc_table} "
                "(chunk_id STRING, snapshot_id STRING, repo_name STRING, "
                "path STRING, title STRING, heading STRING, doc_type STRING, "
                "content STRING, PRIMARY KEY(chunk_id))"
            ),
            (
                f"CREATE NODE TABLE IF NOT EXISTS {self._mention_table} "
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
                f"CREATE REL TABLE IF NOT EXISTS {self._mention_rel_table} "
                f"(FROM {self._doc_table} TO {self._mention_table})"
            )
        except Exception as e:
            self.logger.warning(f"Ladybug rel table creation failed: {e}")

    @staticmethod
    def _libpq_conninfo_value(value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_./:@%+=,-]+", value):
            return value
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"

    @classmethod
    def _postgres_attach_conninfo(cls, dsn: str) -> str:
        raw = (dsn or "").strip()
        if "://" not in raw:
            return raw
        parsed = urlparse(raw)
        if parsed.scheme not in {"postgres", "postgresql"}:
            return raw

        fields: dict[str, str] = {}
        if parsed.hostname:
            fields["host"] = unquote(parsed.hostname)
        elif parsed.path and parsed.netloc.endswith("@"):
            # libpq accepts Unix socket directories as host values. URLs like
            # postgresql://user:pass@/var/run/postgresql?dbname=db encode the
            # socket directory as the path and the database in the query.
            fields["host"] = unquote(parsed.path)
        if parsed.port is not None:
            fields["port"] = str(parsed.port)
        if parsed.username:
            fields["user"] = unquote(parsed.username)
        if parsed.password:
            fields["password"] = unquote(parsed.password)
        query = dict(parse_qsl(parsed.query))
        query_dbname = query.pop("dbname", "") or query.pop("database", "")
        if query_dbname:
            fields["dbname"] = unquote(query_dbname)
        elif parsed.path and not (
            parsed.hostname is None and parsed.netloc.endswith("@")
        ):
            fields["dbname"] = unquote(parsed.path.lstrip("/"))
        for key, value in query.items():
            if key and value:
                fields[key] = unquote(value)
        return " ".join(
            f"{key}={cls._libpq_conninfo_value(value)}"
            for key, value in fields.items()
            if value
        )

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

    def _load_postgres_extension(self) -> None:
        if not self._conn:
            return
        try:
            self._conn.execute("LOAD postgres")
            return
        except Exception as load_error:
            if "has not been installed" not in str(load_error):
                raise
        self._conn.execute("INSTALL postgres")
        self._conn.execute("LOAD postgres")

    def _attach_postgres(self, dsn: str) -> None:
        if not self._conn:
            return
        try:
            self._load_postgres_extension()
            attach_dsn = self._postgres_attach_conninfo(dsn)
            safe_dsn = self._sanitize_attach_dsn(attach_dsn)
            self._conn.execute(f"ATTACH '{safe_dsn}' AS pg (dbtype postgres)")
            self.postgres_attached = True
            self.postgres_attach_error = None
        except Exception as e:
            self.postgres_attached = False
            self.postgres_attach_error = str(e)
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
                chunk_id: str = c.get("chunk_id", "")
                # Upsert: delete old node then create new one
                self._conn.execute(
                    f"MATCH (d:{self._doc_table} "
                    f"{{chunk_id: {_esc(chunk_id)}}}) DELETE d"
                )
                self._conn.execute(
                    f"CREATE (d:{self._doc_table} {{"
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
                    f"MATCH (mt:{self._mention_table} "
                    f"{{mention_id: {_esc(mention_id)}}}) DELETE mt"
                )
                self._conn.execute(
                    f"CREATE (mt:{self._mention_table} {{"
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
                f"MATCH (d:{self._doc_table} "
                f"{{snapshot_id: {_esc(snapshot_id)}}}) "
                "RETURN d.chunk_id, d.heading, d.doc_type, d.content"
            )
            rows: list[dict[str, Any]] = []
            for row in result:
                rows.append(
                    {
                        "chunk_id": row[0],
                        "heading": row[1],
                        "doc_type": row[2],
                        "content": row[3],
                    }
                )
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
