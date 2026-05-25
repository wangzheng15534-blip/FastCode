"""Real PostgreSQL storage semantics gate for P0.6.

This suite intentionally uses the opt-in ``require_postgres_e2e`` fixture
instead of fakes. It exercises the production storage primitives that are
otherwise hard to validate with SQLite or test doubles: idempotent schema
initialization, manifests, lock fencing, staging, redo/outbox queues, SCIP
lineage refs, and relational facts.
"""

from __future__ import annotations

import json
import pathlib
import uuid
from contextlib import suppress
from typing import Any

import pytest

from fastcode.ir.types import (
    IRCodeUnit,
    IRRelation,
    IRSnapshot,
    IRUnitEmbedding,
    IRUnitSupport,
)
from fastcode.store.infrastructure.runtime import DBRuntime
from fastcode.store.manifest import ManifestStore
from fastcode.store.snapshot import SnapshotStore

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def _postgres_runtime(dsn: str) -> DBRuntime:
    return DBRuntime(
        backend="postgres",
        postgres_dsn=dsn,
        pool_min=1,
        pool_max=2,
    )


def _pg_fetchall(dsn: str, sql: str, params: tuple[Any, ...] = ()) -> list[Any]:
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        if cur.description is None:
            conn.commit()
            return []
        return list(cur.fetchall())


def _pg_execute(dsn: str, sql: str, params: tuple[Any, ...] = ()) -> None:
    _pg_fetchall(dsn, sql, params)


def _cleanup_pg_storage_semantics_rows(dsn: str, prefix: str) -> None:
    for table, column in (
        ("manifest_heads", "repo_name"),
        ("manifests", "repo_name"),
        ("snapshot_refs", "repo_name"),
        ("repositories", "repo_name"),
    ):
        with suppress(Exception):
            _pg_execute(dsn, f"DELETE FROM {table} WHERE {column} = %s", (prefix,))
    for table in (
        "attachments",
        "edges",
        "occurrences",
        "symbols",
        "snapshot_documents",
        "scip_artifact_entries",
        "scip_artifacts",
        "snapshot_staging",
        "publish_outbox",
        "snapshots",
    ):
        with suppress(Exception):
            _pg_execute(
                dsn,
                f"DELETE FROM {table} WHERE snapshot_id LIKE %s",
                (f"{prefix}:%",),
            )
    with suppress(Exception):
        _pg_execute(
            dsn,
            "DELETE FROM redo_tasks WHERE task_id LIKE %s",
            (f"redo_{prefix}_%",),
        )
    with suppress(Exception):
        _pg_execute(
            dsn,
            "DELETE FROM resource_locks WHERE lock_name LIKE %s",
            (f"{prefix}:%",),
        )


def _snapshot(
    *,
    repo_name: str,
    snapshot_id: str,
    commit_id: str,
    file_suffix: str = "v1",
    function_signature: str = "def work() -> int",
) -> IRSnapshot:
    path = "pkg/a.py"
    file_unit_id = f"file:{path}"
    function_unit_id = "sym:pkg.a.work"
    return IRSnapshot(
        repo_name=repo_name,
        snapshot_id=snapshot_id,
        branch="main",
        commit_id=commit_id,
        tree_id=f"tree:{commit_id}",
        units=[
            IRCodeUnit(
                unit_id=file_unit_id,
                kind="file",
                path=path,
                language="python",
                display_name=path,
                source_set={"fc_structure"},
                metadata={
                    "content_hash": f"hash:{file_suffix}",
                    "blob_oid": f"blob:{file_suffix}",
                },
            ),
            IRCodeUnit(
                unit_id=function_unit_id,
                kind="function",
                path=path,
                language="python",
                display_name="work",
                qualified_name="pkg.a.work",
                signature=function_signature,
                start_line=1,
                start_col=0,
                end_line=2,
                end_col=12,
                primary_anchor_symbol_id="scip-python pkg/a.py work().",
                anchor_symbol_ids=["scip-python pkg/a.py work()."],
                anchor_coverage=1.0,
                source_set={"fc_structure", "scip"},
                metadata={"semantic_surface_hash": f"surface:{file_suffix}"},
            ),
        ],
        supports=[
            IRUnitSupport(
                support_id=f"occ:{file_suffix}",
                unit_id=function_unit_id,
                source="scip",
                support_kind="occurrence",
                role="definition",
                path=path,
                start_line=1,
                start_col=0,
                end_line=1,
                end_col=10,
                metadata={"doc_id": file_unit_id},
            )
        ],
        relations=[
            IRRelation(
                relation_id=f"rel:contain:{file_suffix}",
                src_unit_id=file_unit_id,
                dst_unit_id=function_unit_id,
                relation_type="contain",
                resolution_state="structural",
                support_sources={"fc_structure"},
                metadata={"doc_id": file_unit_id, "source": "fc_structure"},
            )
        ],
        embeddings=[
            IRUnitEmbedding(
                embedding_id=f"emb:{file_suffix}",
                unit_id=function_unit_id,
                source="fc_embedding",
                vector=[0.1, 0.2, 0.3],
                embedding_text=f"embedding text {file_suffix}",
                model_id="test-model",
                metadata={"embedding_fingerprint": {"model": "test-model"}},
            )
        ],
        metadata={"test": "pg-storage-semantics"},
    )


def test_e2e_postgres_storage_semantics_gate(
    tmp_path: pathlib.Path,
    require_postgres_e2e: str,
) -> None:
    """Verify P0.6 storage semantics against a real PostgreSQL database."""
    dsn = require_postgres_e2e
    prefix = f"pg_storage_{uuid.uuid4().hex[:10]}"
    snapshot_id_v1 = f"{prefix}:snap:v1"
    snapshot_id_v2 = f"{prefix}:snap:v2"
    lock_name = f"{prefix}:index-lock"

    _cleanup_pg_storage_semantics_rows(dsn, prefix)
    runtime = _postgres_runtime(dsn)
    store: SnapshotStore | None = None
    try:
        store = SnapshotStore(
            str(tmp_path / "snapshots"),
            db_runtime=runtime,
        )
        # Re-instantiation must be migration/idempotency-safe on an existing DB.
        runtime_again = _postgres_runtime(dsn)
        try:
            SnapshotStore(
                str(tmp_path / "snapshots_again"),
                db_runtime=runtime_again,
            )
        finally:
            runtime_again.close()
        manifest_store = ManifestStore(runtime)
        ManifestStore(runtime)

        snapshot_v1 = _snapshot(
            repo_name=prefix,
            snapshot_id=snapshot_id_v1,
            commit_id="commit-v1",
            file_suffix="v1",
        )
        snapshot_v2 = _snapshot(
            repo_name=prefix,
            snapshot_id=snapshot_id_v2,
            commit_id="commit-v2",
            file_suffix="v2",
            function_signature="def work(value: int) -> int",
        )

        store.save_snapshot(snapshot_v1, metadata={"version": 1})
        loaded = store.load_snapshot(snapshot_id_v1)
        assert loaded is not None
        assert loaded.snapshot_id == snapshot_id_v1
        assert len(loaded.units) == 2

        first_manifest = manifest_store.publish_record(
            prefix,
            "main",
            snapshot_id_v1,
            "run-v1",
        )
        second_manifest = manifest_store.publish_record(
            prefix,
            "main",
            snapshot_id_v2,
            "run-v2",
        )
        head = manifest_store.get_branch_manifest_record(prefix, "main")
        assert head is not None
        assert head.snapshot_id == snapshot_id_v2
        assert second_manifest.previous_manifest_id == first_manifest.manifest_id

        migration_rows = _pg_fetchall(
            dsn,
            """
            SELECT component, version, COUNT(*)
            FROM schema_migrations
            WHERE component IN ('core_metadata', 'pg_full_spec_alignment', 'manifest_store')
            GROUP BY component, version
            """,
        )
        assert {row[0] for row in migration_rows} >= {
            "core_metadata",
            "pg_full_spec_alignment",
            "manifest_store",
        }
        assert all(row[2] == 1 for row in migration_rows)

        stage_id = store.stage_snapshot(snapshot_v1, metadata={"gate": "p0.6"})
        store.promote_staged_snapshot(snapshot_id_v1, stage_id)
        staging_rows = _pg_fetchall(
            dsn,
            "SELECT status, promoted_at FROM snapshot_staging WHERE stage_id = %s",
            (stage_id,),
        )
        assert len(staging_rows) == 1
        assert staging_rows[0][0] == "published"
        assert staging_rows[0][1] is not None

        token_1 = store.acquire_lock(lock_name, "owner-a", ttl_seconds=60)
        assert token_1 == 1
        assert store.acquire_lock(lock_name, "owner-b", ttl_seconds=60) is None
        token_1_refresh = store.acquire_lock(lock_name, "owner-a", ttl_seconds=60)
        assert token_1_refresh == token_1
        assert store.validate_fencing_token(lock_name, token_1)
        _pg_execute(
            dsn,
            "UPDATE resource_locks SET expires_at = %s WHERE lock_name = %s",
            ("1970-01-01T00:00:00+00:00", lock_name),
        )
        token_2 = store.acquire_lock(lock_name, "owner-b", ttl_seconds=60)
        assert token_2 == 2
        assert not store.validate_fencing_token(lock_name, token_1)
        assert store.validate_fencing_token(lock_name, token_2)

        redo_id = store.enqueue_redo_task(
            "index_run_recovery",
            {"snapshot_id": snapshot_id_v1, "repo_name": prefix},
        )
        _pg_execute(
            dsn,
            """
            UPDATE redo_tasks
            SET task_id = %s,
                created_at = %s,
                updated_at = %s,
                next_attempt_at = NULL
            WHERE task_id = %s
            """,
            (
                f"redo_{prefix}_1",
                "1900-01-01T00:00:00+00:00",
                "1900-01-01T00:00:00+00:00",
                redo_id,
            ),
        )
        redo_id = f"redo_{prefix}_1"
        claimed_redo = store.claim_redo_task_record()
        assert claimed_redo is not None
        assert claimed_redo.task_id == redo_id
        assert claimed_redo.status == "running"
        assert claimed_redo.attempts == 1
        store.mark_redo_task_failed(redo_id, "retry", max_attempts=5)
        pending_rows = _pg_fetchall(
            dsn,
            "SELECT status, attempts, last_error FROM redo_tasks WHERE task_id = %s",
            (redo_id,),
        )
        assert pending_rows == [("pending", 1, "retry")]
        _pg_execute(
            dsn,
            "UPDATE redo_tasks SET next_attempt_at = NULL WHERE task_id = %s",
            (redo_id,),
        )
        claimed_again = store.claim_redo_task_record()
        assert claimed_again is not None
        assert claimed_again.attempts == 2
        store.mark_redo_task_done(redo_id)
        done_rows = _pg_fetchall(
            dsn,
            "SELECT status FROM redo_tasks WHERE task_id = %s",
            (redo_id,),
        )
        assert done_rows == [("completed",)]

        event_id = f"{prefix}:event:1"
        assert store.enqueue_outbox_event(
            event_id,
            "snapshot_published",
            json.dumps({"snapshot_id": snapshot_id_v1}),
            snapshot_id_v1,
            max_attempts=2,
        )
        _pg_execute(
            dsn,
            "UPDATE publish_outbox SET created_at = %s WHERE event_id = %s",
            ("1900-01-01T00:00:00+00:00", event_id),
        )
        assert not store.enqueue_outbox_event(
            event_id,
            "snapshot_published",
            "{}",
            snapshot_id_v1,
        )
        claimed_events = store.claim_outbox_event_records(limit=1)
        assert [event.event_id for event in claimed_events] == [event_id]
        assert claimed_events[0].status == "in_progress"
        store.mark_outbox_event_failed(event_id, "publish failed")
        assert store.get_outbox_pending_count() >= 1
        retryable_rows = _pg_fetchall(
            dsn,
            """
            SELECT status, attempts, max_attempts, error_message
            FROM publish_outbox
            WHERE event_id = %s
            """,
            (event_id,),
        )
        assert retryable_rows == [("failed", 1, 2, "publish failed")]
        claimed_events = store.claim_outbox_event_records(limit=1)
        assert [event.event_id for event in claimed_events] == [event_id]
        store.mark_outbox_event_done(event_id)
        outbox_rows = _pg_fetchall(
            dsn,
            "SELECT status, attempts, error_message FROM publish_outbox WHERE event_id = %s",
            (event_id,),
        )
        assert outbox_rows == [("published", 1, "publish failed")]

        artifacts = store.save_scip_artifact_refs(
            snapshot_id_v1,
            artifacts=[
                {
                    "indexer_name": "scip-python",
                    "indexer_version": "1.0",
                    "artifact_path": "/tmp/a.scip",
                    "checksum": "sha256:a",
                    "language": "python",
                    "metadata": {"paths": ["pkg/a.py"]},
                },
                {
                    "indexer_name": "scip-typescript",
                    "indexer_version": "2.0",
                    "artifact_path": "/tmp/b.scip",
                    "checksum": "sha256:b",
                    "language": "typescript",
                },
            ],
        )
        assert [artifact["sequence_no"] for artifact in artifacts] == [0, 1]
        listed_artifacts = store.list_scip_artifact_refs(snapshot_id_v1)
        assert [artifact["indexer_name"] for artifact in listed_artifacts] == [
            "scip-python",
            "scip-typescript",
        ]
        assert listed_artifacts[0]["metadata"]["language"] == "python"

        store.save_relational_facts(snapshot_v1)
        counts_v1 = _relational_fact_counts(dsn, snapshot_id_v1)
        assert counts_v1 == {
            "snapshot_documents": 1,
            "symbols": 1,
            "occurrences": 1,
            "edges": 1,
            "attachments": 1,
        }

        assert store.save_relational_facts_delta(
            snapshot_v2,
            previous_snapshot_id=snapshot_id_v1,
            changed_paths=["pkg/a.py"],
        )
        counts_v2 = _relational_fact_counts(dsn, snapshot_id_v2)
        assert counts_v2 == counts_v1
        symbol_rows = _pg_fetchall(
            dsn,
            "SELECT metadata_json FROM symbols WHERE snapshot_id = %s",
            (snapshot_id_v2,),
        )
        symbol_metadata = json.loads(symbol_rows[0][0])
        assert symbol_metadata["metadata"]["semantic_surface_hash"] == "surface:v2"
    finally:
        if store is not None:
            store.db_runtime.close()
        runtime.close()
        _cleanup_pg_storage_semantics_rows(dsn, prefix)


def _relational_fact_counts(dsn: str, snapshot_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in (
        "snapshot_documents",
        "symbols",
        "occurrences",
        "edges",
        "attachments",
    ):
        rows = _pg_fetchall(
            dsn,
            f"SELECT COUNT(*) FROM {table} WHERE snapshot_id = %s",
            (snapshot_id,),
        )
        counts[table] = int(rows[0][0])
    return counts
