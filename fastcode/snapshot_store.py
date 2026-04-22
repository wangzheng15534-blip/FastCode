"""
Snapshot metadata and artifact persistence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from .db_runtime import DBRuntime
from .scip_models import SCIPArtifactRef
from .semantic_ir import IRSnapshot
from .utils import ensure_dir, utc_now


class SnapshotStore:
    def __init__(self, persist_dir: str, storage_cfg: dict[str, Any] | None = None):
        self.persist_dir = os.path.abspath(persist_dir)
        self.snapshot_root = os.path.join(self.persist_dir, "snapshots")
        ensure_dir(self.persist_dir)
        ensure_dir(self.snapshot_root)
        self.db_path = os.path.join(self.persist_dir, "lineage.db")
        self.db_runtime = DBRuntime.from_storage_config(sqlite_path=self.db_path, storage_cfg=storage_cfg)
        self._init_db()

    def _init_db(self) -> None:
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    repo_name TEXT NOT NULL,
                    branch TEXT,
                    commit_id TEXT,
                    tree_id TEXT,
                    artifact_key TEXT NOT NULL,
                    ir_path TEXT NOT NULL,
                    ir_graphs_path TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            if self.db_runtime.backend == "postgres":
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS snapshot_refs (
                        ref_id BIGSERIAL PRIMARY KEY,
                        repo_name TEXT NOT NULL,
                        branch TEXT,
                        commit_id TEXT,
                        tree_id TEXT,
                        snapshot_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(repo_name, commit_id, snapshot_id)
                    )
                    """,
                )
            else:
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS snapshot_refs (
                        ref_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        repo_name TEXT NOT NULL,
                        branch TEXT,
                        commit_id TEXT,
                        tree_id TEXT,
                        snapshot_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(repo_name, commit_id, snapshot_id)
                    )
                    """,
                )
            self.db_runtime.execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS scip_artifacts (
                    snapshot_id TEXT PRIMARY KEY,
                    indexer_name TEXT NOT NULL,
                    indexer_version TEXT,
                    artifact_path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
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
                CREATE INDEX IF NOT EXISTS idx_snapshot_refs_repo_branch
                ON snapshot_refs (repo_name, branch, created_at DESC)
                """
            )
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO schema_migrations (component, version, applied_at)
                VALUES (?, ?, ?)
                ON CONFLICT(component, version) DO NOTHING
                """,
                ("core_metadata", "v1", utc_now()),
            )
            if self.db_runtime.backend == "postgres":
                # Git backbone
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS repositories (
                        repo_id TEXT PRIMARY KEY,
                        repo_name TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS git_refs (
                        repo_id TEXT NOT NULL,
                        ref_name TEXT NOT NULL,
                        commit_id TEXT,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (repo_id, ref_name)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS git_commits (
                        repo_id TEXT NOT NULL,
                        commit_id TEXT NOT NULL,
                        tree_id TEXT,
                        parent_commit_id TEXT,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (repo_id, commit_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS git_trees (
                        repo_id TEXT NOT NULL,
                        tree_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (repo_id, tree_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS git_blobs (
                        repo_id TEXT NOT NULL,
                        blob_id TEXT NOT NULL,
                        path TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (repo_id, blob_id)
                    )
                    """,
                )
                # Relational facts
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS snapshot_documents (
                        snapshot_id TEXT NOT NULL,
                        doc_id TEXT NOT NULL,
                        path TEXT NOT NULL,
                        language TEXT,
                        metadata_json TEXT,
                        PRIMARY KEY (snapshot_id, doc_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS symbols (
                        snapshot_id TEXT NOT NULL,
                        symbol_id TEXT NOT NULL,
                        path TEXT,
                        display_name TEXT,
                        qualified_name TEXT,
                        kind TEXT,
                        language TEXT,
                        source_priority INTEGER,
                        metadata_json TEXT,
                        PRIMARY KEY (snapshot_id, symbol_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS occurrences (
                        snapshot_id TEXT NOT NULL,
                        occurrence_id TEXT NOT NULL,
                        symbol_id TEXT NOT NULL,
                        doc_id TEXT NOT NULL,
                        role TEXT,
                        start_line INTEGER,
                        start_col INTEGER,
                        end_line INTEGER,
                        end_col INTEGER,
                        source TEXT,
                        metadata_json TEXT,
                        PRIMARY KEY (snapshot_id, occurrence_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS edges (
                        snapshot_id TEXT NOT NULL,
                        edge_id TEXT NOT NULL,
                        src_id TEXT NOT NULL,
                        dst_id TEXT NOT NULL,
                        edge_type TEXT NOT NULL,
                        source TEXT,
                        confidence TEXT,
                        doc_id TEXT,
                        metadata_json TEXT,
                        PRIMARY KEY (snapshot_id, edge_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS attachments (
                        snapshot_id TEXT NOT NULL,
                        attachment_id TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        target_type TEXT NOT NULL,
                        attachment_type TEXT NOT NULL,
                        source TEXT,
                        confidence TEXT,
                        payload_json TEXT,
                        metadata_json TEXT,
                        PRIMARY KEY (snapshot_id, attachment_id)
                    )
                    """,
                )
                # Staging + hardening
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS snapshot_staging (
                        stage_id TEXT PRIMARY KEY,
                        snapshot_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        metadata_json TEXT,
                        created_at TEXT NOT NULL,
                        promoted_at TEXT
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS resource_locks (
                        lock_name TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        fencing_token BIGINT NOT NULL DEFAULT 0
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    ALTER TABLE resource_locks
                    ADD COLUMN IF NOT EXISTS fencing_token BIGINT NOT NULL DEFAULT 0
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS redo_tasks (
                        task_id TEXT PRIMARY KEY,
                        task_type TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        last_error TEXT,
                        next_attempt_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS design_documents (
                        snapshot_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        repo_name TEXT NOT NULL,
                        path TEXT NOT NULL,
                        title TEXT,
                        heading TEXT,
                        doc_type TEXT,
                        content TEXT NOT NULL,
                        metadata_json TEXT,
                        PRIMARY KEY (snapshot_id, chunk_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS design_doc_mentions (
                        snapshot_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        symbol_id TEXT NOT NULL,
                        symbol_name TEXT,
                        confidence TEXT,
                        metadata_json TEXT,
                        PRIMARY KEY (snapshot_id, chunk_id, symbol_id)
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
                    ("pg_full_spec_alignment", "v1", utc_now()),
                )
            conn.commit()

    def artifact_key_for_snapshot(self, snapshot_id: str) -> str:
        return f"snap_{hashlib.md5(snapshot_id.encode('utf-8')).hexdigest()[:20]}"

    def snapshot_dir(self, snapshot_id: str) -> str:
        safe = self.artifact_key_for_snapshot(snapshot_id)
        path = os.path.join(self.snapshot_root, safe)
        ensure_dir(path)
        return path

    def save_snapshot(self, snapshot: IRSnapshot, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        snap_dir = self.snapshot_dir(snapshot.snapshot_id)
        ir_path = os.path.join(snap_dir, "ir_snapshot.json")
        tmp_ir_path = f"{ir_path}.tmp"
        with open(tmp_ir_path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_ir_path, ir_path)

        artifact_key = self.artifact_key_for_snapshot(snapshot.snapshot_id)
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO snapshots (
                    snapshot_id, repo_name, branch, commit_id, tree_id, artifact_key,
                    ir_path, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    repo_name=excluded.repo_name,
                    branch=excluded.branch,
                    commit_id=excluded.commit_id,
                    tree_id=excluded.tree_id,
                    artifact_key=excluded.artifact_key,
                    ir_path=excluded.ir_path,
                    metadata_json=excluded.metadata_json
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.repo_name,
                    snapshot.branch,
                    snapshot.commit_id,
                    snapshot.tree_id,
                    artifact_key,
                    ir_path,
                    utc_now(),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO snapshot_refs (
                    repo_name, branch, commit_id, tree_id, snapshot_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_name, commit_id, snapshot_id) DO NOTHING
                """,
                (
                    snapshot.repo_name,
                    snapshot.branch,
                    snapshot.commit_id,
                    snapshot.tree_id,
                    snapshot.snapshot_id,
                    utc_now(),
                ),
            )
            conn.commit()

        return {
            "snapshot_id": snapshot.snapshot_id,
            "artifact_key": artifact_key,
            "ir_path": ir_path,
            "dir": snap_dir,
        }

    def update_snapshot_metadata(self, snapshot_id: str, metadata: dict[str, Any]) -> None:
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                "UPDATE snapshots SET metadata_json=? WHERE snapshot_id=?",
                (json.dumps(metadata or {}, ensure_ascii=False), snapshot_id),
            )
            conn.commit()

    def save_ir_graphs(self, snapshot_id: str, ir_graphs: Any) -> str:
        snap_dir = self.snapshot_dir(snapshot_id)
        path = os.path.join(snap_dir, "ir_graphs.pkl")
        with open(path, "wb") as f:
            pickle.dump(ir_graphs, f)
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                "UPDATE snapshots SET ir_graphs_path=? WHERE snapshot_id=?",
                (path, snapshot_id),
            )
            conn.commit()
        return path

    def load_ir_graphs(self, snapshot_id: str) -> Any | None:
        row = self.get_snapshot_record(snapshot_id)
        if not row:
            return None
        path = row.get("ir_graphs_path")
        if not path or not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_snapshot(self, snapshot_id: str) -> IRSnapshot | None:
        row = self.get_snapshot_record(snapshot_id)
        if not row:
            return None
        ir_path = row.get("ir_path")
        if not ir_path or not os.path.exists(ir_path):
            return None
        with open(ir_path, encoding="utf-8") as f:
            data = json.load(f)
        return IRSnapshot.from_dict(data)

    def get_snapshot_record(self, snapshot_id: str) -> dict[str, Any] | None:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                "SELECT * FROM snapshots WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchone()
        return self.db_runtime.row_to_dict(row)

    def find_by_repo_commit(self, repo_name: str, commit_id: str) -> dict[str, Any] | None:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM snapshots
                WHERE repo_name=? AND commit_id=?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (repo_name, commit_id),
            ).fetchone()
        return self.db_runtime.row_to_dict(row)

    def find_by_artifact_key(self, artifact_key: str) -> dict[str, Any] | None:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM snapshots
                WHERE artifact_key=?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (artifact_key,),
            ).fetchone()
        return self.db_runtime.row_to_dict(row)

    def resolve_snapshot_for_ref(self, repo_name: str, branch: str) -> dict[str, Any] | None:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM snapshot_refs
                WHERE repo_name=? AND branch=?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (repo_name, branch),
            ).fetchone()
        return self.db_runtime.row_to_dict(row)

    def save_scip_artifact_ref(
        self,
        snapshot_id: str,
        *,
        indexer_name: str = "unknown",
        indexer_version: str | None = None,
        artifact_path: str = "",
        checksum: str = "",
    ) -> dict[str, Any]:
        created_at = utc_now()
        artifact_ref = SCIPArtifactRef(
            snapshot_id=snapshot_id,
            indexer_name=indexer_name,
            indexer_version=indexer_version,
            artifact_path=artifact_path,
            checksum=checksum,
            created_at=created_at,
        )
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO scip_artifacts (
                    snapshot_id, indexer_name, indexer_version, artifact_path, checksum, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    indexer_name=excluded.indexer_name,
                    indexer_version=excluded.indexer_version,
                    artifact_path=excluded.artifact_path,
                    checksum=excluded.checksum,
                    created_at=excluded.created_at
                """,
                (
                    artifact_ref.snapshot_id,
                    artifact_ref.indexer_name,
                    artifact_ref.indexer_version,
                    artifact_ref.artifact_path,
                    artifact_ref.checksum,
                    created_at,
                ),
            )
            conn.commit()
        return artifact_ref.to_dict()

    def get_scip_artifact_ref(self, snapshot_id: str) -> dict[str, Any] | None:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                "SELECT * FROM scip_artifacts WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchone()
        return self.db_runtime.row_to_dict(row)

    def import_git_backbone(self, snapshot: IRSnapshot, git_meta: dict[str, Any] | None = None) -> None:
        if self.db_runtime.backend != "postgres":
            return
        git_meta = git_meta or {}
        repo_id = snapshot.repo_name
        now = utc_now()
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO repositories (repo_id, repo_name, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(repo_id) DO UPDATE SET repo_name=excluded.repo_name
                """,
                (repo_id, snapshot.repo_name, now),
            )
            if snapshot.commit_id:
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO git_commits (repo_id, commit_id, tree_id, parent_commit_id, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(repo_id, commit_id) DO UPDATE SET
                        tree_id=excluded.tree_id,
                        parent_commit_id=excluded.parent_commit_id
                    """,
                    (repo_id, snapshot.commit_id, snapshot.tree_id, git_meta.get("parent_commit_id"), now),
                )
            if snapshot.tree_id:
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO git_trees (repo_id, tree_id, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(repo_id, tree_id) DO NOTHING
                    """,
                    (repo_id, snapshot.tree_id, now),
                )
            if snapshot.branch:
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO git_refs (repo_id, ref_name, commit_id, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(repo_id, ref_name) DO UPDATE SET
                        commit_id=excluded.commit_id,
                        updated_at=excluded.updated_at
                    """,
                    (repo_id, snapshot.branch, snapshot.commit_id, now),
                )
            conn.commit()

    def save_relational_facts(self, snapshot: IRSnapshot) -> None:
        if self.db_runtime.backend != "postgres":
            return
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(conn, "DELETE FROM snapshot_documents WHERE snapshot_id=?", (snapshot.snapshot_id,))
            self.db_runtime.execute(conn, "DELETE FROM symbols WHERE snapshot_id=?", (snapshot.snapshot_id,))
            self.db_runtime.execute(conn, "DELETE FROM occurrences WHERE snapshot_id=?", (snapshot.snapshot_id,))
            self.db_runtime.execute(conn, "DELETE FROM edges WHERE snapshot_id=?", (snapshot.snapshot_id,))
            self.db_runtime.execute(conn, "DELETE FROM attachments WHERE snapshot_id=?", (snapshot.snapshot_id,))
            for doc in snapshot.documents:
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO snapshot_documents (snapshot_id, doc_id, path, language, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.snapshot_id,
                        doc.doc_id,
                        doc.path,
                        doc.language,
                        json.dumps(doc.to_dict(), ensure_ascii=False),
                    ),
                )
            for sym in snapshot.symbols:
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO symbols (
                        snapshot_id, symbol_id, path, display_name, qualified_name, kind,
                        language, source_priority, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.snapshot_id,
                        sym.symbol_id,
                        sym.path,
                        sym.display_name,
                        sym.qualified_name,
                        sym.kind,
                        sym.language,
                        sym.source_priority,
                        json.dumps(sym.to_dict(), ensure_ascii=False),
                    ),
                )
            for occ in snapshot.occurrences:
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO occurrences (
                        snapshot_id, occurrence_id, symbol_id, doc_id, role, start_line,
                        start_col, end_line, end_col, source, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.snapshot_id,
                        occ.occurrence_id,
                        occ.symbol_id,
                        occ.doc_id,
                        occ.role,
                        occ.start_line,
                        occ.start_col,
                        occ.end_line,
                        occ.end_col,
                        occ.source,
                        json.dumps(occ.to_dict(), ensure_ascii=False),
                    ),
                )
            seen_edge_ids: set[str] = set()
            for edge in snapshot.edges:
                if edge.edge_id in seen_edge_ids:
                    continue
                seen_edge_ids.add(edge.edge_id)
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO edges (
                        snapshot_id, edge_id, src_id, dst_id, edge_type, source, confidence,
                        doc_id, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.snapshot_id,
                        edge.edge_id,
                        edge.src_id,
                        edge.dst_id,
                        edge.edge_type,
                        edge.source,
                        edge.confidence,
                        edge.doc_id,
                        json.dumps(edge.to_dict(), ensure_ascii=False),
                    ),
                )
            for attachment in snapshot.attachments:
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO attachments (
                        snapshot_id, attachment_id, target_id, target_type, attachment_type,
                        source, confidence, payload_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.snapshot_id,
                        attachment.attachment_id,
                        attachment.target_id,
                        attachment.target_type,
                        attachment.attachment_type,
                        attachment.source,
                        attachment.confidence,
                        json.dumps(attachment.payload, ensure_ascii=False),
                        json.dumps(attachment.metadata, ensure_ascii=False),
                    ),
                )
            conn.commit()

    def stage_snapshot(self, snapshot: IRSnapshot, metadata: dict[str, Any] | None = None) -> str:
        stage_id = f"stage_{uuid.uuid4().hex[:16]}"
        if self.db_runtime.backend != "postgres":
            return stage_id
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO snapshot_staging (stage_id, snapshot_id, status, metadata_json, created_at)
                VALUES (?, ?, 'staged', ?, ?)
                ON CONFLICT(stage_id) DO NOTHING
                """,
                (stage_id, snapshot.snapshot_id, json.dumps(metadata or {}, ensure_ascii=False), utc_now()),
            )
            conn.commit()
        return stage_id

    def promote_staged_snapshot(self, snapshot_id: str, stage_id: str) -> None:
        if self.db_runtime.backend != "postgres":
            return
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                UPDATE snapshot_staging
                SET status='published', promoted_at=?
                WHERE stage_id=? AND snapshot_id=?
                """,
                (utc_now(), stage_id, snapshot_id),
            )
            conn.commit()

    def acquire_lock(self, lock_name: str, owner_id: str, ttl_seconds: int = 300) -> int | None:
        if self.db_runtime.backend != "postgres":
            return 1
        now = datetime.now(UTC)
        expires_at = (now.timestamp() + ttl_seconds)
        expires_iso = datetime.fromtimestamp(expires_at, tz=UTC).isoformat()
        now_iso = now.isoformat()
        with self.db_runtime.connect() as conn:
            # Atomic compare-and-swap: INSERT or UPDATE only if expired or same owner
            row = self.db_runtime.execute(
                conn,
                """
                INSERT INTO resource_locks (lock_name, owner_id, expires_at, updated_at, fencing_token)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(lock_name) DO UPDATE SET
                    owner_id=excluded.owner_id,
                    expires_at=excluded.expires_at,
                    updated_at=excluded.updated_at,
                    fencing_token=resource_locks.fencing_token + 1
                WHERE resource_locks.expires_at < ? OR resource_locks.owner_id = ?
                RETURNING fencing_token
                """,
                (lock_name, owner_id, expires_iso, utc_now(), now_iso, owner_id),
            ).fetchone()
            conn.commit()
        if row is None:
            return None
        return int(row["fencing_token"])

    def validate_fencing_token(self, lock_name: str, expected_token: int) -> bool:
        if self.db_runtime.backend != "postgres":
            return True
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                "SELECT fencing_token FROM resource_locks WHERE lock_name=?",
                (lock_name,),
            ).fetchone()
        if not row:
            return False
        return int(row.get("fencing_token") or 0) == int(expected_token)

    def release_lock(self, lock_name: str, owner_id: str) -> None:
        if self.db_runtime.backend != "postgres":
            return
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                "DELETE FROM resource_locks WHERE lock_name=? AND owner_id=?",
                (lock_name, owner_id),
            )
            conn.commit()

    def enqueue_redo_task(self, task_type: str, payload: dict[str, Any], error: str | None = None) -> str:
        task_id = f"redo_{uuid.uuid4().hex[:16]}"
        if self.db_runtime.backend != "postgres":
            return task_id
        now = utc_now()
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO redo_tasks (
                    task_id, task_type, payload_json, status, attempts, last_error, next_attempt_at, created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', 0, ?, ?, ?, ?)
                """,
                (task_id, task_type, json.dumps(payload, ensure_ascii=False), error, now, now, now),
            )
            conn.commit()
        return task_id

    def save_design_documents(
        self,
        snapshot_id: str,
        repo_name: str,
        chunks: list[dict[str, Any]],
        mentions: list[dict[str, Any]],
    ) -> None:
        if self.db_runtime.backend != "postgres":
            return
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(conn, "DELETE FROM design_documents WHERE snapshot_id=?", (snapshot_id,))
            self.db_runtime.execute(conn, "DELETE FROM design_doc_mentions WHERE snapshot_id=?", (snapshot_id,))
            for chunk in chunks:
                if not chunk.get("chunk_id"):
                    logging.getLogger(__name__).warning("Skipping design document chunk without chunk_id: %s", chunk.get("path", "<unknown>"))
                    continue
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO design_documents (
                        snapshot_id, chunk_id, repo_name, path, title, heading, doc_type, content, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        chunk.get("chunk_id"),
                        repo_name,
                        chunk.get("path"),
                        chunk.get("title"),
                        chunk.get("heading"),
                        chunk.get("doc_type"),
                        chunk.get("content", ""),
                        json.dumps(chunk, ensure_ascii=False),
                    ),
                )
            for mention in mentions:
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO design_doc_mentions (
                        snapshot_id, chunk_id, symbol_id, symbol_name, confidence, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(snapshot_id, chunk_id, symbol_id) DO UPDATE SET
                        symbol_name=excluded.symbol_name,
                        confidence=excluded.confidence,
                        metadata_json=excluded.metadata_json
                    """,
                    (
                        snapshot_id,
                        mention.get("chunk_id"),
                        mention.get("symbol_id"),
                        mention.get("symbol_name"),
                        mention.get("confidence"),
                        json.dumps(mention, ensure_ascii=False),
                    ),
                )
            conn.commit()

    def get_doc_mentions(self, snapshot_id: str) -> list[dict[str, Any]]:
        """Return all design doc mentions for a snapshot."""
        with self.db_runtime.connect() as conn:
            rows = self.db_runtime.execute(
                conn,
                """
                SELECT chunk_id, symbol_id, symbol_name, confidence, metadata_json
                FROM design_doc_mentions
                WHERE snapshot_id = ?
                """,
                (snapshot_id,),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            entry = {
                "chunk_id": r["chunk_id"],
                "symbol_id": r["symbol_id"],
                "symbol_name": r["symbol_name"],
                "confidence": r["confidence"],
            }
            metadata_json = r["metadata_json"]
            if metadata_json:
                try:
                    meta = json.loads(metadata_json)
                    entry.update({k: v for k, v in meta.items() if k not in entry})
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(entry)
        return results

    def claim_redo_task(self) -> dict[str, Any] | None:
        if self.db_runtime.backend != "postgres":
            return None
        now = utc_now()
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM redo_tasks
                WHERE status='pending'
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                ORDER BY created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if not row:
                conn.commit()
                return None
            task = self.db_runtime.row_to_dict(row) or {}
            self.db_runtime.execute(
                conn,
                """
                UPDATE redo_tasks
                SET status='running', attempts=attempts+1, updated_at=?
                WHERE task_id=?
                """,
                (now, task.get("task_id")),
            )
            conn.commit()
        task["status"] = "running"
        task["attempts"] = int(task.get("attempts") or 0) + 1
        return task

    def mark_redo_task_done(self, task_id: str) -> None:
        if self.db_runtime.backend != "postgres":
            return
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                UPDATE redo_tasks
                SET status='completed', updated_at=?
                WHERE task_id=?
                """,
                (utc_now(), task_id),
            )
            conn.commit()

    def mark_redo_task_failed(self, task_id: str, error: str, max_attempts: int = 5) -> None:
        if self.db_runtime.backend != "postgres":
            return
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                "SELECT attempts FROM redo_tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            attempts = int((row or {}).get("attempts") or 0)
            if attempts >= max_attempts:
                self.db_runtime.execute(
                    conn,
                    """
                    UPDATE redo_tasks
                    SET status='dead', last_error=?, updated_at=?
                    WHERE task_id=?
                    """,
                    (error, utc_now(), task_id),
                )
            else:
                backoff_seconds = max(1, 2 ** attempts)
                next_attempt_at = (datetime.now(UTC) + timedelta(seconds=backoff_seconds)).isoformat()
                self.db_runtime.execute(
                    conn,
                    """
                    UPDATE redo_tasks
                    SET status='pending', last_error=?, next_attempt_at=?, updated_at=?
                    WHERE task_id=?
                    """,
                    (error, next_attempt_at, utc_now(), task_id),
                )
            conn.commit()
