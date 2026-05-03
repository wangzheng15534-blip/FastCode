"""Persistent per-unit artifact metadata for graceful incremental updates."""

from __future__ import annotations

import json
from typing import Any

from ..db_runtime import DBRuntime
from ..utils import utc_now


class UnitArtifactStore:
    def __init__(self, db_path_or_runtime: str | DBRuntime) -> None:
        if isinstance(db_path_or_runtime, DBRuntime):
            self.db_runtime = db_path_or_runtime
        else:
            self.db_runtime = DBRuntime(
                backend="sqlite", sqlite_path=db_path_or_runtime
            )
        self._init_db()

    def _init_db(self) -> None:
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS unit_artifacts (
                    snapshot_id TEXT NOT NULL,
                    stable_unit_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    unit_type TEXT NOT NULL,
                    content_hash TEXT,
                    syntax_hash TEXT,
                    signature_hash TEXT,
                    edge_surface_hash TEXT,
                    embedding_text_hash TEXT,
                    api_surface_hash TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (snapshot_id, stable_unit_id)
                )
                """,
            )
            self.db_runtime.execute(
                conn,
                """
                CREATE INDEX IF NOT EXISTS idx_unit_artifacts_path
                ON unit_artifacts (snapshot_id, relative_path)
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
                ("unit_artifact_store", "v1", utc_now()),
            )
            conn.commit()

    def replace_snapshot_units(
        self,
        snapshot_id: str,
        *,
        elements: list[dict[str, Any]],
    ) -> None:
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                "DELETE FROM unit_artifacts WHERE snapshot_id=?",
                (snapshot_id,),
            )
            for elem in elements:
                metadata = dict(elem.get("metadata", {}) or {})
                stable_unit_id = str(metadata.get("stable_unit_id") or "")
                if not stable_unit_id:
                    continue
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO unit_artifacts (
                        snapshot_id, stable_unit_id, relative_path, unit_type,
                        content_hash, syntax_hash, signature_hash, edge_surface_hash,
                        embedding_text_hash, api_surface_hash, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        stable_unit_id,
                        str(elem.get("relative_path") or elem.get("file_path") or ""),
                        str(elem.get("type") or ""),
                        metadata.get("content_hash"),
                        metadata.get("syntax_hash"),
                        metadata.get("signature_hash"),
                        metadata.get("edge_surface_hash"),
                        metadata.get("embedding_text_hash"),
                        metadata.get("api_surface_hash"),
                        json.dumps(metadata, ensure_ascii=False),
                        utc_now(),
                    ),
                )
            conn.commit()

    def list_snapshot_units(self, snapshot_id: str) -> list[dict[str, Any]]:
        with self.db_runtime.connect() as conn:
            rows = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM unit_artifacts
                WHERE snapshot_id=?
                ORDER BY relative_path ASC, stable_unit_id ASC
                """,
                (snapshot_id,),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            payload = self.db_runtime.row_to_dict(row) or {}
            metadata_json = payload.pop("metadata_json", None)
            if metadata_json:
                try:
                    payload["metadata"] = json.loads(metadata_json)
                except (json.JSONDecodeError, TypeError):
                    payload["metadata"] = {}
            results.append(payload)
        return results
