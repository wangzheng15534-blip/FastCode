import json
import os
import pickle
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
from git import Repo

from fastcode.indexing.pipeline import IndexPipeline
from fastcode.ir.element import CodeElement
from fastcode.ir.types import (
    IRAttachment,
    IRCodeUnit,
    IRRelation,
    IRSnapshot,
    IRUnitSupport,
)
from fastcode.scip.models import SCIPDocument, SCIPIndex
from fastcode.store.index_run import IndexRunStore
from fastcode.store.manifest import ManifestStore
from fastcode.store.snapshot import SnapshotStore
from fastcode.store.unit_artifacts import UnitArtifactStore
from fastcode.utils.materialization import (
    BOUNDARY_PICKLE_LOAD,
    increment_materialization_boundary,
)


def test_snapshot_store_persists_and_loads_snapshot():
    with tempfile.TemporaryDirectory(prefix="fc_snap_test_") as tmp:
        store = SnapshotStore(tmp)
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            branch="main",
            commit_id="abc",
            tree_id="tree123",
            attachments=[
                IRAttachment(
                    attachment_id="att:repo:abc",
                    target_id="snap:repo:abc",
                    target_type="snapshot",
                    attachment_type="summary",
                    source="fc_structure",
                    confidence="derived",
                    payload={"text": "Repository summary"},
                    metadata={"producer": "test"},
                )
            ],
        )
        meta = store.save_snapshot(snap, metadata={"x": 1})
        assert meta.artifact_key.startswith("snap_")

        loaded = store.load_snapshot("snap:repo:abc")
        assert loaded is not None
        assert loaded.repo_name == "repo"
        assert loaded.branch == "main"
        assert len(loaded.attachments) == 1
        assert loaded.attachments[0].payload["text"] == "Repository summary"


def test_manifest_head_points_to_latest_publish():
    with tempfile.TemporaryDirectory(prefix="fc_manifest_test_") as tmp:
        snapshot_store = SnapshotStore(tmp)
        manifest_store = ManifestStore(snapshot_store.db_path)

        m1 = manifest_store.publish("repo", "main", "snap:repo:1", "run_1")
        m2 = manifest_store.publish("repo", "main", "snap:repo:2", "run_2")
        head = manifest_store.get_branch_manifest("repo", "main")

        assert head is not None
        assert head["manifest_id"] == m2["manifest_id"]
        assert head["previous_manifest_id"] == m1["manifest_id"]


def test_index_run_idempotency_key_reuses_run():
    with tempfile.TemporaryDirectory(prefix="fc_run_test_") as tmp:
        snapshot_store = SnapshotStore(tmp)
        run_store = IndexRunStore(snapshot_store.db_path)

        run_1 = run_store.create_run(
            "repo", "snap:repo:1", "main", "c1", idempotency_key="k1"
        )
        run_2 = run_store.create_run(
            "repo", "snap:repo:1", "main", "c1", idempotency_key="k1"
        )

        assert run_1 == run_2


def test_snapshot_store_persists_scip_artifact_ref():
    with tempfile.TemporaryDirectory(prefix="fc_scip_artifact_test_") as tmp:
        store = SnapshotStore(tmp)
        artifact = store.save_scip_artifact_ref(
            snapshot_id="snap:repo:abc",
            indexer_name="scip-python",
            indexer_version="1.0.0",
            artifact_path="/tmp/index.scip.json",
            checksum="deadbeef",
        )
        assert artifact["snapshot_id"] == "snap:repo:abc"
        loaded = store.get_scip_artifact_ref("snap:repo:abc")
        assert loaded is not None
        assert loaded["indexer_name"] == "scip-python"


def test_snapshot_store_save_scip_artifact_ref_defaults():
    """Verify that save_scip_artifact_ref uses defaults when optional args are omitted."""
    with tempfile.TemporaryDirectory(prefix="fc_scip_defaults_test_") as tmp:
        store = SnapshotStore(tmp)
        # Only pass the required snapshot_id; everything else should use defaults
        artifact = store.save_scip_artifact_ref(snapshot_id="snap:repo:defaults")
        assert artifact["snapshot_id"] == "snap:repo:defaults"
        assert artifact["indexer_name"] == "unknown"
        assert artifact["indexer_version"] is None
        assert artifact["artifact_path"] == ""
        assert artifact["checksum"] == ""
        assert "created_at" in artifact

        # Round-trip through the store to confirm persistence
        loaded = store.get_scip_artifact_ref("snap:repo:defaults")
        assert loaded is not None
        assert loaded["indexer_name"] == "unknown"
        assert loaded["artifact_path"] == ""


def test_snapshot_store_save_scip_artifact_ref_upsert():
    """Verify that calling save_scip_artifact_ref twice with the same snapshot_id upserts."""
    with tempfile.TemporaryDirectory(prefix="fc_scip_upsert_test_") as tmp:
        store = SnapshotStore(tmp)
        store.save_scip_artifact_ref(
            snapshot_id="snap:repo:upsert",
            indexer_name="scip-python",
            artifact_path="/old/path.scip",
            checksum="old",
        )
        # Second call with different values should overwrite
        updated = store.save_scip_artifact_ref(
            snapshot_id="snap:repo:upsert",
            indexer_name="scip-java",
            artifact_path="/new/path.scip",
            checksum="new",
        )
        assert updated["indexer_name"] == "scip-java"
        assert updated["artifact_path"] == "/new/path.scip"
        assert updated["checksum"] == "new"

        loaded = store.get_scip_artifact_ref("snap:repo:upsert")
        assert loaded is not None
        assert loaded["indexer_name"] == "scip-java"
        assert loaded["checksum"] == "new"


def test_snapshot_store_persists_multiple_scip_artifact_refs():
    with tempfile.TemporaryDirectory(prefix="fc_scip_multi_test_") as tmp:
        store = SnapshotStore(tmp)
        artifacts = store.save_scip_artifact_refs(
            "snap:repo:multi",
            artifacts=[
                {
                    "indexer_name": "scip-python",
                    "indexer_version": "1.0",
                    "artifact_path": "/tmp/python.scip",
                    "checksum": "aaa",
                    "language": "python",
                },
                {
                    "indexer_name": "scip-go",
                    "indexer_version": "1.0",
                    "artifact_path": "/tmp/go.scip",
                    "checksum": "bbb",
                    "language": "go",
                },
            ],
        )

        assert len(artifacts) == 2
        assert artifacts[0]["role"] == "primary"
        assert artifacts[1]["role"] == "secondary"
        assert artifacts[1]["metadata"]["language"] == "go"

        primary = store.get_scip_artifact_ref("snap:repo:multi")
        assert primary is not None
        assert primary["artifact_path"] == "/tmp/python.scip"

        listed = store.list_scip_artifact_refs("snap:repo:multi")
        assert [artifact["artifact_path"] for artifact in listed] == [
            "/tmp/python.scip",
            "/tmp/go.scip",
        ]


def test_snapshot_store_lock_api_returns_fencing_token_shape():
    with tempfile.TemporaryDirectory(prefix="fc_lock_test_") as tmp:
        store = SnapshotStore(tmp)
        token = store.acquire_lock("index:snap:repo:1", owner_id="run1", ttl_seconds=60)
        assert token == 1
        assert store.validate_fencing_token("index:snap:repo:1", expected_token=token)


def test_sqlite_fencing_token_always_one():
    """SQLite lock implementation returns token=1 for all acquire calls (no PG-style increment)."""
    with tempfile.TemporaryDirectory(prefix="fc_fence_") as tmp:
        store = SnapshotStore(tmp)
        token1 = store.acquire_lock(
            "index:snap:repo:1", owner_id="run1", ttl_seconds=60
        )
        assert token1 == 1
        token2 = store.acquire_lock(
            "index:snap:repo:1", owner_id="run2", ttl_seconds=60
        )
        assert token2 == 1  # SQLite always returns 1, no PG-style increment


def test_sqlite_lock_release_does_not_raise():
    """SQLite release_lock and validate_fencing_token are no-ops but must not crash."""
    with tempfile.TemporaryDirectory(prefix="fc_fence_") as tmp:
        store = SnapshotStore(tmp)
        store.acquire_lock("index:snap:repo:1", owner_id="run1", ttl_seconds=60)
        store.release_lock("index:snap:repo:1", owner_id="run1")
        # validate on nonexistent lock also returns True (SQLite no-op)
        assert store.validate_fencing_token("nonexistent:lock", expected_token=1)


class _FakeCursor:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self._row = row

    def fetchone(self) -> dict[str, Any] | None:
        return self._row


class _FakePostgresLockRuntime:
    backend = "postgres"

    def __init__(self) -> None:
        self.locks: dict[str, dict[str, Any]] = {}

    def connect(self) -> Any:
        return self

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def commit(self) -> None:
        return None

    def execute(
        self, _conn: object, sql: str, params: tuple[Any, ...] = ()
    ) -> _FakeCursor:
        if "INSERT INTO resource_locks" in sql:
            lock_name, owner_id, expires_at, updated_at = params[:4]
            current = self.locks.get(str(lock_name))
            if current is None:
                token = 1
            elif current["owner_id"] == owner_id:
                token = (
                    current["fencing_token"]
                    if "CASE" in sql
                    else current["fencing_token"] + 1
                )
            else:
                return _FakeCursor(None)
            self.locks[str(lock_name)] = {
                "owner_id": owner_id,
                "expires_at": expires_at,
                "updated_at": updated_at,
                "fencing_token": token,
            }
            return _FakeCursor({"fencing_token": token})

        if "SELECT fencing_token FROM resource_locks" in sql:
            current = self.locks.get(str(params[0]))
            if current is None:
                return _FakeCursor(None)
            return _FakeCursor({"fencing_token": current["fencing_token"]})

        if "DELETE FROM resource_locks" in sql:
            lock_name, owner_id = params
            current = self.locks.get(str(lock_name))
            if current and current["owner_id"] == owner_id:
                del self.locks[str(lock_name)]
            return _FakeCursor(None)

        raise AssertionError(f"unexpected SQL: {sql}")


def test_postgres_lock_reacquire_by_same_owner_preserves_fencing_token():
    """Same-owner lock refresh must not invalidate in-flight work."""
    store = SnapshotStore.__new__(SnapshotStore)
    store.db_runtime = _FakePostgresLockRuntime()

    token1 = store.acquire_lock("index:snap:repo:1", owner_id="run1", ttl_seconds=60)
    token2 = store.acquire_lock("index:snap:repo:1", owner_id="run1", ttl_seconds=60)

    assert token1 == 1
    assert token2 == 1
    assert store.validate_fencing_token("index:snap:repo:1", token1)


def test_enqueue_redo_task_returns_id():
    with tempfile.TemporaryDirectory(prefix="fc_redo_") as tmp:
        store = SnapshotStore(tmp)
        task_id = store.enqueue_redo_task(
            task_type="index_run_recovery",
            payload={"run_id": "run1", "source": "/tmp/repo"},
        )
        assert task_id.startswith("redo_")
        assert len(task_id) > len("redo_")


def test_sqlite_redo_task_noops_do_not_raise():
    """SQLite redo task methods (claim, mark_done, mark_failed) are no-ops but must not crash."""
    with tempfile.TemporaryDirectory(prefix="fc_redo_") as tmp:
        store = SnapshotStore(tmp)
        assert store.claim_redo_task() is None
        store.mark_redo_task_done("redo_fake")
        store.mark_redo_task_failed(task_id="redo_fake", error="test error")


def _make_minimal_pipeline(tmp: str) -> IndexPipeline:
    store = SnapshotStore(tmp)
    registry = SimpleNamespace(
        applicable=lambda **kwargs: [],
        applicable_for_capabilities=lambda **kwargs: [],
    )
    return IndexPipeline(
        config={},
        logger=SimpleNamespace(
            info=lambda *a, **kw: None, warning=lambda *a, **kw: None
        ),
        loader=SimpleNamespace(
            repo_path=tmp,
            get_repository_info=lambda: {"name": "repo", "url": tmp},
            scan_files=lambda: [],
        ),
        snapshot_store=store,
        manifest_store=ManifestStore(store.db_runtime),
        index_run_store=IndexRunStore(store.db_runtime),
        unit_artifact_store=UnitArtifactStore(store.db_runtime),
        snapshot_symbol_index=SimpleNamespace(register_snapshot=lambda snapshot: None),
        vector_store=SimpleNamespace(persist_dir=tmp, load=lambda artifact_key: True),
        embedder=SimpleNamespace(embedding_dim=3),
        indexer=SimpleNamespace(extract_elements=lambda **kwargs: []),
        retriever=SimpleNamespace(
            load_bm25=lambda artifact_key: True,
            set_ir_graphs=lambda *a, **kw: None,
            build_repo_overview_bm25=lambda: None,
        ),
        graph_builder=SimpleNamespace(load=lambda artifact_key: True),
        ir_graph_builder=SimpleNamespace(
            build_graphs=lambda snapshot: SimpleNamespace()
        ),
        pg_retrieval_store=SimpleNamespace(upsert_elements=lambda **kwargs: None),
        terminus_publisher=SimpleNamespace(is_configured=lambda: False),
        doc_ingester=SimpleNamespace(),
        semantic_resolver_registry=registry,
        set_repo_indexed=lambda v: None,
        set_repo_loaded=lambda v: None,
        set_repo_info=lambda v: None,
    )


def test_pipeline_layer_contract_records_disabled_scip_non_silently() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_layers_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        element = CodeElement(
            id="file:a",
            type="file",
            name="a.py",
            file_path="a.py",
            relative_path="a.py",
            language="python",
            start_line=1,
            end_line=1,
            code="pass\n",
            signature=None,
            docstring=None,
            summary=None,
            metadata={"embedding": np.array([0.1, 0.2, 0.3])},
            repo_name="repo",
            repo_url=None,
        )
        ast_snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:test",
            branch="main",
            commit_id="c1",
            tree_id="t1",
            units=[
                IRCodeUnit(
                    unit_id="doc:snap:repo:test:a.py",
                    kind="file",
                    path="a.py",
                    language="python",
                    display_name="a.py",
                    source_set={"fc_structure"},
                    metadata={"source": "fc_structure"},
                )
            ],
        )

        temp_store = SimpleNamespace(
            metadata=[],
            initialize=lambda dim: None,
            add_vectors=lambda vectors, metadata: None,
            save=lambda artifact_key: None,
        )
        temp_graph = SimpleNamespace(
            dependency_graph=SimpleNamespace(
                number_of_nodes=lambda: 0, number_of_edges=lambda: 0
            ),
            inheritance_graph=SimpleNamespace(
                number_of_nodes=lambda: 0, number_of_edges=lambda: 0
            ),
            call_graph=SimpleNamespace(
                number_of_nodes=lambda: 0, number_of_edges=lambda: 0
            ),
            build_graphs=lambda elements, module_resolver, symbol_resolver: None,
            save=lambda artifact_key: None,
        )
        temp_retriever = SimpleNamespace(
            index_for_bm25=lambda elements: None,
            build_repo_overview_bm25=lambda: None,
            save_bm25=lambda artifact_key: None,
        )

        with (
            patch.object(
                pipeline,
                "_resolve_snapshot_ref",
                return_value={
                    "repo_name": "repo",
                    "branch": "main",
                    "commit_id": "c1",
                    "tree_id": "t1",
                    "snapshot_id": "snap:repo:test",
                },
            ),
            patch.object(pipeline, "_build_git_meta", return_value={}),
            patch.object(pipeline.indexer, "extract_elements", return_value=[element]),
            patch.object(
                CodeElement,
                "to_dict",
                autospec=True,
                side_effect=AssertionError(
                    "active indexing path must not call CodeElement.to_dict()"
                ),
            ),
            patch("fastcode.indexing.pipeline.VectorStore", return_value=temp_store),
            patch(
                "fastcode.indexing.pipeline.CodeGraphBuilder", return_value=temp_graph
            ),
            patch(
                "fastcode.indexing.pipeline.HybridRetriever",
                return_value=temp_retriever,
            ),
            patch(
                "fastcode.indexing.pipeline.build_ir_from_ast",
                return_value=ast_snapshot,
            ),
            patch("fastcode.indexing.pipeline.validate_snapshot", return_value=[]),
            patch.object(
                pipeline,
                "_apply_semantic_resolvers",
                side_effect=lambda **kwargs: kwargs["snapshot"],
            ),
            patch.object(pipeline.snapshot_store, "stage_snapshot", return_value=None),
            patch.object(
                pipeline.snapshot_store, "save_relational_facts", return_value=None
            ),
            patch.object(pipeline.snapshot_store, "save_ir_graphs", return_value=None),
            patch.object(
                pipeline.snapshot_store, "import_git_backbone", return_value=None
            ),
            patch.object(
                pipeline.snapshot_store, "update_snapshot_metadata", return_value=None
            ),
            patch.object(pipeline.snapshot_store, "release_lock", return_value=None),
            patch.object(
                pipeline.snapshot_store, "validate_fencing_token", return_value=True
            ),
            patch.object(pipeline, "_load_artifacts_by_key", return_value=True),
        ):
            result = pipeline.run_index_pipeline(
                source=tmp,
                is_url=False,
                enable_scip=False,
                publish=False,
            )

        layers = result["pipeline_layers"]
        assert [layer["name"] for layer in layers] == [
            "plain_ast_embedding",
            "unified_ir_scip_merge",
            "language_specific_semantic_upgrade",
        ]
        assert layers[0]["status"] == "succeeded"
        assert layers[1]["status"] == "skipped"
        assert layers[1]["reason"] == "disabled_by_config"
        assert layers[1]["warnings"] == ["layer_disabled: enable_scip=false"]
        assert result["pipeline_metrics"]["never_silent_fallback"] is True
        materialization_counts = result["pipeline_metrics"][
            "materialization_boundary_counts"
        ]
        assert materialization_counts["json_encode"] >= 1
        assert materialization_counts["vector_list_conversion"] == 1


def test_reused_snapshot_result_reports_materialization_metrics() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_reused_metrics_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        snapshot_id = "snap:repo:reused"
        snapshot = IRSnapshot(repo_name="repo", snapshot_id=snapshot_id)
        layers = pipeline._default_pipeline_layers(enable_scip=False)
        pipeline.snapshot_store.save_snapshot(
            snapshot,
            metadata={
                "pipeline_layers": layers,
                "pipeline_metrics": {
                    "never_silent_fallback": True,
                    "materialization_boundary_counts": {"json_encode": 99},
                },
            },
        )

        def _load_existing_artifacts(_artifact_key: str) -> bool:
            increment_materialization_boundary(BOUNDARY_PICKLE_LOAD)
            return True

        with (
            patch.object(
                pipeline,
                "_resolve_snapshot_ref",
                return_value={
                    "repo_name": "repo",
                    "branch": "main",
                    "commit_id": "c1",
                    "tree_id": "t1",
                    "snapshot_id": snapshot_id,
                },
            ),
            patch.object(pipeline, "_build_git_meta", return_value={}),
            patch.object(
                pipeline,
                "_load_artifacts_by_key",
                side_effect=_load_existing_artifacts,
            ),
        ):
            result = pipeline.run_index_pipeline(
                source=tmp,
                is_url=False,
                enable_scip=False,
                publish=False,
            )

        counts = result["pipeline_metrics"]["materialization_boundary_counts"]
        assert result["status"] == "reused"
        assert counts["pickle_load"] == 1
        assert counts["snapshot_full_load"] == 1
        assert counts["json_decode"] >= 1
        assert counts.get("json_encode") is None


def test_pipeline_layer_helpers_report_semantic_gap_metrics() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_layers_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        snapshot = IRSnapshot(repo_name="repo", snapshot_id="snap:1")
        layer3 = pipeline._make_layer_record(
            name="language_specific_semantic_upgrade",
            ordinal=3,
            enabled=True,
            source="language_specific_ast_resolvers",
            description="semantic layer",
            conditional=True,
        )
        pipeline._finalize_layer_metrics(
            snapshot,
            layer3,
            extra_metrics={
                "resolver_runs": 0,
                **pipeline._layer3_quality_metrics(snapshot),
            },
        )
        layer3["status"] = "degraded"
        layer3["reason"] = "no_semantic_resolver_runs_recorded"
        assert layer3["metrics"]["resolver_runs"] == 0
        assert layer3["metrics"]["semantic_relations"] == 0
        assert layer3["status"] == "degraded"


def test_pipeline_reused_result_keeps_existing_behavior_without_fake_layer_claims() -> (
    None
):
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_reuse_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        snapshot = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:test")
        pipeline.snapshot_store.save_snapshot(
            snapshot, metadata={"artifact_key": "snap_repo_test"}
        )
        pipeline.index_run_store.create_run(
            repo_name="repo",
            snapshot_id="snap:repo:test",
            branch="main",
            commit_id="c1",
            idempotency_key="reuse-key",
        )
        run_id = pipeline.index_run_store.create_run(
            repo_name="repo",
            snapshot_id="snap:repo:test",
            branch="main",
            commit_id="c1",
            idempotency_key="reuse-key",
        )
        run = pipeline.index_run_store.get_run(run_id)
        assert run is not None
        pipeline.index_run_store.mark_completed(
            run["run_id"], status="succeeded", warnings=[]
        )

        with (
            patch.object(
                pipeline,
                "_resolve_snapshot_ref",
                return_value={
                    "repo_name": "repo",
                    "branch": "main",
                    "commit_id": "c1",
                    "tree_id": "t1",
                    "snapshot_id": "snap:repo:test",
                },
            ),
            patch.object(pipeline, "_build_git_meta", return_value={}),
            patch.object(pipeline, "_load_artifacts_by_key", return_value=True),
        ):
            result = pipeline.run_index_pipeline(
                source=tmp, is_url=False, publish=False
            )

        assert result["status"] == "reused"
        assert [layer["name"] for layer in result["pipeline_layers"]] == [
            "plain_ast_embedding",
            "unified_ir_scip_merge",
            "language_specific_semantic_upgrade",
        ]
        assert result["pipeline_metrics"]["never_silent_fallback"] is True


def test_pipeline_layer3_succeeds_when_semantic_runs_recorded_even_without_new_relations() -> (
    None
):
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_layers_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        snapshot = IRSnapshot(repo_name="repo", snapshot_id="snap:1")
        snapshot.metadata["semantic_resolver_runs"] = [
            {"language": "python", "source": "python_resolver"}
        ]
        metrics = pipeline._layer3_quality_metrics(snapshot)
        assert metrics["semantic_relations"] == 0
        assert len(snapshot.metadata["semantic_resolver_runs"]) == 1


def test_pipeline_layer3_requires_upgrade_signal_not_just_run_metadata() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_layers_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        snapshot = IRSnapshot(repo_name="repo", snapshot_id="snap:1")
        snapshot.metadata["semantic_resolver_runs"] = [
            {"language": "python", "source": "python_resolver"}
        ]
        quality = pipeline._layer3_quality_metrics(snapshot)
        assert quality["semantic_relations"] == 0
        assert quality["anchored_relations"] == 0
        assert quality["relations_with_pending_capabilities"] == 0


def test_pipeline_backfill_reuses_full_scip_artifact_lineage_metadata() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_scip_meta_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        snapshot = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:test")
        pipeline.snapshot_store.save_snapshot(
            snapshot,
            metadata={
                "artifact_key": "snap_repo_test",
                "scip_artifact_ref": {
                    "snapshot_id": "snap:repo:test",
                    "artifact_path": "/tmp/a.scip",
                    "indexer_name": "scip-python",
                    "indexer_version": "1.0",
                    "checksum": "111",
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
                "scip_artifact_refs": [
                    {
                        "snapshot_id": "snap:repo:test",
                        "artifact_path": "/tmp/a.scip",
                        "indexer_name": "scip-python",
                        "indexer_version": "1.0",
                        "checksum": "111",
                        "created_at": "2026-01-01T00:00:00+00:00",
                    },
                    {
                        "snapshot_id": "snap:repo:test",
                        "artifact_path": "/tmp/b.scip",
                        "indexer_name": "scip-go",
                        "indexer_version": "1.0",
                        "checksum": "222",
                        "created_at": "2026-01-01T00:00:00+00:00",
                    },
                ],
            },
        )

        result = pipeline._backfill_result_layer_metadata(
            snapshot_id="snap:repo:test",
            enable_scip=True,
            result={"status": "reused", "snapshot_id": "snap:repo:test"},
        )

        assert result["scip_artifact_ref"]["artifact_path"] == "/tmp/a.scip"
        assert [
            artifact["artifact_path"] for artifact in result["scip_artifact_refs"]
        ] == [
            "/tmp/a.scip",
            "/tmp/b.scip",
        ]


def test_pipeline_backfill_persists_missing_layer_metadata_to_snapshot_store() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_backfill_persist_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        snapshot = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:legacy")
        pipeline.snapshot_store.save_snapshot(
            snapshot, metadata={"artifact_key": "legacy_key"}
        )

        result = pipeline._backfill_result_layer_metadata(
            snapshot_id="snap:repo:legacy",
            enable_scip=False,
            result={"status": "reused", "snapshot_id": "snap:repo:legacy"},
        )

        record = pipeline.snapshot_store.get_snapshot_record("snap:repo:legacy")
        assert record is not None
        stored_metadata = json.loads(record.metadata_json)
        assert stored_metadata["pipeline_metrics"]["never_silent_fallback"] is True
        assert stored_metadata["pipeline_layers"][1]["status"] == "skipped"
        assert result["pipeline_metrics"]["never_silent_fallback"] is True


def test_pipeline_layer2_records_experimental_scip_languages_non_silently() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_scip_experimental_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        element = SimpleNamespace(
            id="file:a",
            type="file",
            name="a.zig",
            file_path="a.zig",
            relative_path="a.zig",
            language="zig",
            start_line=1,
            end_line=1,
            signature=None,
            metadata={"embedding": np.array([0.1, 0.2, 0.3])},
            to_dict=lambda: {"id": "file:a", "relative_path": "a.zig", "metadata": {}},
        )
        ast_snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:test",
            branch="main",
            commit_id="c1",
            tree_id="t1",
            units=[
                IRCodeUnit(
                    unit_id="doc:snap:repo:test:a.zig",
                    kind="file",
                    path="a.zig",
                    language="zig",
                    display_name="a.zig",
                    source_set={"fc_structure"},
                    metadata={"source": "fc_structure"},
                )
            ],
        )
        scip_snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:test",
            branch="main",
            commit_id="c1",
            tree_id="t1",
            metadata={
                "scip_languages": ["zig"],
                "experimental_scip_languages": ["zig"],
            },
        )
        temp_store = SimpleNamespace(
            metadata=[],
            initialize=lambda dim: None,
            add_vectors=lambda vectors, metadata: None,
            save=lambda artifact_key: None,
        )
        temp_graph = SimpleNamespace(
            dependency_graph=SimpleNamespace(
                number_of_nodes=lambda: 0, number_of_edges=lambda: 0
            ),
            inheritance_graph=SimpleNamespace(
                number_of_nodes=lambda: 0, number_of_edges=lambda: 0
            ),
            call_graph=SimpleNamespace(
                number_of_nodes=lambda: 0, number_of_edges=lambda: 0
            ),
            build_graphs=lambda elements, module_resolver, symbol_resolver: None,
            save=lambda artifact_key: None,
        )
        temp_retriever = SimpleNamespace(
            index_for_bm25=lambda elements: None,
            build_repo_overview_bm25=lambda: None,
            save_bm25=lambda artifact_key: None,
        )

        def _fake_run_scip(language: str, repo_path: str, out_dir: str) -> SCIPIndex:
            assert language == "zig"
            artifact_path = os.path.join(out_dir, "zig.scip")
            with open(artifact_path, "wb") as handle:
                handle.write(b"fake")
            return SCIPIndex(indexer_name="zls", indexer_version="1.0")

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    pipeline,
                    "_resolve_snapshot_ref",
                    return_value={
                        "repo_name": "repo",
                        "branch": "main",
                        "commit_id": "c1",
                        "tree_id": "t1",
                        "snapshot_id": "snap:repo:test",
                    },
                )
            )
            stack.enter_context(
                patch.object(pipeline, "_build_git_meta", return_value={})
            )
            stack.enter_context(
                patch.object(
                    pipeline.indexer, "extract_elements", return_value=[element]
                )
            )
            stack.enter_context(
                patch("fastcode.indexing.pipeline.VectorStore", return_value=temp_store)
            )
            stack.enter_context(
                patch(
                    "fastcode.indexing.pipeline.CodeGraphBuilder",
                    return_value=temp_graph,
                )
            )
            stack.enter_context(
                patch(
                    "fastcode.indexing.pipeline.HybridRetriever",
                    return_value=temp_retriever,
                )
            )
            stack.enter_context(
                patch(
                    "fastcode.indexing.pipeline.build_ir_from_ast",
                    return_value=ast_snapshot,
                )
            )
            stack.enter_context(
                patch(
                    "fastcode.indexing.pipeline.detect_scip_languages",
                    return_value=["zig"],
                )
            )
            stack.enter_context(
                patch(
                    "fastcode.indexing.pipeline.run_scip_for_language",
                    side_effect=_fake_run_scip,
                )
            )
            stack.enter_context(
                patch(
                    "fastcode.indexing.pipeline.build_ir_from_scip",
                    return_value=scip_snapshot,
                )
            )
            stack.enter_context(
                patch("fastcode.indexing.pipeline.merge_ir", return_value=scip_snapshot)
            )
            stack.enter_context(
                patch("fastcode.indexing.pipeline.validate_snapshot", return_value=[])
            )
            stack.enter_context(
                patch.object(
                    pipeline,
                    "_apply_semantic_resolvers",
                    side_effect=lambda **kwargs: kwargs["snapshot"],
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "stage_snapshot", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "save_relational_facts", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "save_ir_graphs", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "import_git_backbone", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store,
                    "update_snapshot_metadata",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch.object(pipeline.snapshot_store, "release_lock", return_value=None)
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store,
                    "validate_fencing_token",
                    return_value=True,
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store,
                    "save_scip_artifact_refs",
                    return_value=[
                        {
                            "snapshot_id": "snap:repo:test",
                            "artifact_path": os.path.join(tmp, "zig.scip"),
                            "indexer_name": "zls",
                            "indexer_version": "1.0",
                            "checksum": "abc",
                            "created_at": "2026-01-01T00:00:00+00:00",
                        }
                    ],
                )
            )
            stack.enter_context(
                patch.object(pipeline, "_load_artifacts_by_key", return_value=True)
            )
            result = pipeline.run_index_pipeline(
                source=tmp,
                is_url=False,
                enable_scip=True,
                publish=False,
            )

        layer2 = result["pipeline_layers"][1]
        assert "experimental_scip_languages: zig" in layer2["warnings"]
        assert "experimental_scip_languages: zig" in result["warnings"]
        assert layer2["metrics"]["experimental_scip_languages"] == ["zig"]
        assert layer2["metrics"]["experimental_language_count"] == 1


def test_pipeline_incremental_prefilter_only_indexes_changed_files() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_incremental_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)

        unchanged_path = os.path.join(tmp, "a.py")
        changed_path = os.path.join(tmp, "b.py")
        with open(unchanged_path, "w", encoding="utf-8") as handle:
            handle.write("print('a')\n")
        with open(changed_path, "w", encoding="utf-8") as handle:
            handle.write("print('new')\n")

        unchanged_stat = os.stat(unchanged_path)
        changed_stat = os.stat(changed_path)

        previous_snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:prev",
            branch="main",
            commit_id="c1",
            tree_id="t1",
        )
        previous_record = pipeline.snapshot_store.save_snapshot(
            previous_snapshot, metadata={}
        )
        pipeline.manifest_store.publish(
            repo_name="repo",
            ref_name="main",
            snapshot_id="snap:repo:prev",
            index_run_id="run_prev",
            status="published",
        )

        with open(
            os.path.join(tmp, f"{previous_record.artifact_key}_metadata.pkl"),
            "wb",
        ) as handle:
            pickle.dump(
                {
                    "metadata": [
                        {
                            "id": "unchanged:1",
                            "type": "file",
                            "name": "a.py",
                            "file_path": unchanged_path,
                            "relative_path": "a.py",
                            "language": "python",
                            "start_line": 1,
                            "end_line": 1,
                            "code": "print('a')\n",
                            "signature": None,
                            "docstring": None,
                            "summary": None,
                            "metadata": {
                                "embedding": np.array([0.1, 0.2, 0.3]),
                                "embedding_text": "a",
                            },
                            "repo_name": "repo",
                            "repo_url": tmp,
                        }
                    ]
                },
                handle,
            )

        with open(
            os.path.join(tmp, f"{previous_record.artifact_key}_manifest.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            unchanged_fingerprint = pipeline._file_fingerprint(unchanged_path)
            changed_fingerprint = pipeline._file_fingerprint(changed_path)
            assert unchanged_fingerprint is not None
            assert changed_fingerprint is not None
            json.dump(
                {
                    "schema_version": 2,
                    "compatibility": pipeline._incremental_compatibility_payload(),
                    "compatibility_hash": pipeline._incremental_compatibility_hash(),
                    "files": {
                        "a.py": {
                            **unchanged_fingerprint,
                            "element_ids": ["unchanged:1"],
                        },
                        "b.py": {
                            "mtime": changed_stat.st_mtime - 10,
                            "size": changed_stat.st_size + 1,
                            "content_hash": "stale-content-hash",
                            "element_ids": ["stale:2"],
                        },
                    },
                },
                handle,
            )

        changed_element = SimpleNamespace(
            id="changed:2",
            type="file",
            name="b.py",
            file_path=changed_path,
            relative_path="b.py",
            language="python",
            start_line=1,
            end_line=1,
            code="print('new')\n",
            signature=None,
            docstring=None,
            summary=None,
            metadata={"embedding": np.array([0.4, 0.5, 0.6])},
            repo_name="repo",
            repo_url=tmp,
            to_dict=lambda: {
                "id": "changed:2",
                "type": "file",
                "name": "b.py",
                "file_path": changed_path,
                "relative_path": "b.py",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "code": "print('new')\n",
                "signature": None,
                "docstring": None,
                "summary": None,
                "metadata": {"embedding": np.array([0.4, 0.5, 0.6])},
                "repo_name": "repo",
                "repo_url": tmp,
            },
        )
        ast_snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:current",
            branch="main",
            commit_id="c2",
            tree_id="t2",
            units=[
                IRCodeUnit(
                    unit_id="doc:snap:repo:current:a.py",
                    kind="file",
                    path="a.py",
                    language="python",
                    display_name="a.py",
                    source_set={"fc_structure"},
                    metadata={"source": "fc_structure"},
                ),
                IRCodeUnit(
                    unit_id="doc:snap:repo:current:b.py",
                    kind="file",
                    path="b.py",
                    language="python",
                    display_name="b.py",
                    source_set={"fc_structure"},
                    metadata={"source": "fc_structure"},
                ),
            ],
        )

        temp_store_metadata: list[dict[str, Any]] = []
        vector_incremental_calls: list[dict[str, Any]] = []
        bm25_incremental_calls: list[dict[str, Any]] = []

        def _save_vector_incremental(
            artifact_key: str,
            *,
            previous_name: str,
            reusable_path_keys: set[str],
            snapshot_id: str | None = None,
        ) -> dict[str, int]:
            del snapshot_id
            vector_incremental_calls.append(
                {
                    "artifact_key": artifact_key,
                    "previous_name": previous_name,
                    "reusable_path_keys": sorted(reusable_path_keys),
                }
            )
            return {"vector_shards_reused": len(reusable_path_keys)}

        def _save_bm25_incremental(
            artifact_key: str,
            *,
            previous_name: str,
            reusable_path_keys: set[str],
        ) -> dict[str, int]:
            bm25_incremental_calls.append(
                {
                    "artifact_key": artifact_key,
                    "previous_name": previous_name,
                    "reusable_path_keys": sorted(reusable_path_keys),
                }
            )
            return {"bm25_shards_reused": len(reusable_path_keys)}

        temp_store = SimpleNamespace(
            metadata=temp_store_metadata,
            initialize=lambda dim: None,
            add_vectors=lambda vectors, metadata: temp_store_metadata.extend(metadata),
            save=lambda artifact_key: None,
            publish_delta=_save_vector_incremental,
        )
        temp_graph = SimpleNamespace(
            dependency_graph=SimpleNamespace(
                number_of_nodes=lambda: 0, number_of_edges=lambda: 0
            ),
            inheritance_graph=SimpleNamespace(
                number_of_nodes=lambda: 0, number_of_edges=lambda: 0
            ),
            call_graph=SimpleNamespace(
                number_of_nodes=lambda: 0, number_of_edges=lambda: 0
            ),
            build_graphs=lambda elements, module_resolver, symbol_resolver: None,
            save=lambda artifact_key: None,
        )
        temp_retriever = SimpleNamespace(
            index_for_bm25=lambda elements: None,
            build_repo_overview_bm25=lambda: None,
            save_bm25=lambda artifact_key: None,
            publish_bm25_delta=_save_bm25_incremental,
        )

        pipeline.loader.scan_files = lambda: [
            {
                "path": unchanged_path,
                "relative_path": "a.py",
                "size": unchanged_stat.st_size,
                "extension": ".py",
            },
            {
                "path": changed_path,
                "relative_path": "b.py",
                "size": changed_stat.st_size,
                "extension": ".py",
            },
        ]

        extract_mock = Mock(
            side_effect=AssertionError("extract_elements should not run")
        )
        index_files_mock = Mock(return_value=[changed_element])
        pipeline.indexer.extract_elements = extract_mock
        pipeline.indexer.index_files = index_files_mock

        seen_ast_element_ids: list[str] = []

        def _capture_ast(**kwargs: Any) -> IRSnapshot:
            seen_ast_element_ids.extend(elem.id for elem in kwargs["elements"])
            return ast_snapshot

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    pipeline,
                    "_resolve_snapshot_ref",
                    return_value={
                        "repo_name": "repo",
                        "branch": "main",
                        "commit_id": "c2",
                        "tree_id": "t2",
                        "snapshot_id": "snap:repo:current",
                    },
                )
            )
            stack.enter_context(
                patch.object(pipeline, "_build_git_meta", return_value={})
            )
            stack.enter_context(
                patch("fastcode.indexing.pipeline.VectorStore", return_value=temp_store)
            )
            stack.enter_context(
                patch(
                    "fastcode.indexing.pipeline.CodeGraphBuilder",
                    return_value=temp_graph,
                )
            )
            stack.enter_context(
                patch(
                    "fastcode.indexing.pipeline.HybridRetriever",
                    return_value=temp_retriever,
                )
            )
            stack.enter_context(
                patch(
                    "fastcode.indexing.pipeline.build_ir_from_ast",
                    side_effect=_capture_ast,
                )
            )
            stack.enter_context(
                patch("fastcode.indexing.pipeline.merge_ir", return_value=ast_snapshot)
            )
            stack.enter_context(
                patch("fastcode.indexing.pipeline.validate_snapshot", return_value=[])
            )
            stack.enter_context(
                patch.object(
                    pipeline,
                    "_apply_semantic_resolvers",
                    side_effect=lambda **kwargs: kwargs["snapshot"],
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "stage_snapshot", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "save_relational_facts", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "save_ir_graphs", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "import_git_backbone", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store,
                    "update_snapshot_metadata",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch.object(pipeline.snapshot_store, "release_lock", return_value=None)
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store,
                    "validate_fencing_token",
                    return_value=True,
                )
            )
            stack.enter_context(
                patch.object(pipeline, "_load_artifacts_by_key", return_value=True)
            )

            result = pipeline.run_index_pipeline(
                source=tmp,
                is_url=False,
                enable_scip=False,
                publish=False,
            )

        extract_mock.assert_not_called()
        index_files_mock.assert_called_once()
        changed_file_infos = index_files_mock.call_args.args[0]
        assert [file_info["relative_path"] for file_info in changed_file_infos] == [
            "b.py"
        ]
        assert set(seen_ast_element_ids) == {"changed:2"}
        assert {row["id"] for row in temp_store_metadata} == {"changed:2"}
        assert vector_incremental_calls == [
            {
                "artifact_key": pipeline.snapshot_store.artifact_key_for_snapshot(
                    "snap:repo:current"
                ),
                "previous_name": previous_record.artifact_key,
                "reusable_path_keys": ["a.py"],
            }
        ]
        assert bm25_incremental_calls == vector_incremental_calls
        assert result["incremental_prefilter"] == {
            "previous_snapshot_id": "snap:repo:prev",
            "previous_artifact_key": previous_record.artifact_key,
            "artifact_delta_mode": True,
            "added": 0,
            "modified": 1,
            "removed": 0,
            "unchanged": 1,
            "added_paths": [],
            "modified_paths": ["b.py"],
            "removed_paths": [],
            "unchanged_paths": ["a.py"],
            "changed_paths": ["b.py"],
            "artifact_delta_graph_fallback_reason": "edge_surface_changed",
            "ast_ir_rebuilt_elements": 1,
            "ast_ir_reused_files": 1,
            "reused_elements": 1,
            "reindexed_elements": 1,
            "reused_changed_embeddings": 0,
            "semantic_frontier_widened": 1,
            "api_frontier_changed": 1,
            "api_frontier_changed_paths": ["b.py"],
            "package_scope_roots": ["."],
            "change_kinds": [
                "api_surface_hash",
                "edge_surface_hash",
                "embedding_text_hash",
                "signature_hash",
            ],
            "artifact_shard_reuse": {
                "vector_shards_reused": 1,
                "bm25_shards_reused": 1,
                "graph_fallback_reason": "edge_or_delete_frontier_requires_full_graph",
            },
        }
        assert result["repair_queue"]["pending"] == 1
        assert result["repair_queue"]["task_type"] == "semantic_repair_frontier"
        assert result["repair_queue"]["scope_kind"] == "package"
        assert result["repair_queue"]["scope_roots"] == ["."]


def test_incremental_diff_reuses_inventory_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_fingerprints_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)

        def _boom_hash(_path: str) -> str:
            raise AssertionError("pre-fingerprinted inventory should not be rehashed")

        monkeypatch.setattr("fastcode.indexing.pipeline.compute_file_hash", _boom_hash)

        changes = pipeline._detect_file_changes(
            {
                "files": {
                    "a.py": {
                        "content_hash": "hash-a",
                        "blob_oid": "hash-a",
                        "element_ids": ["elem:a"],
                    }
                }
            },
            [
                {
                    "path": os.path.join(tmp, "a.py"),
                    "relative_path": "a.py",
                    "size": 12,
                    "mtime": 123.0,
                    "content_hash": "hash-a",
                    "blob_oid": "hash-a",
                }
            ],
        )

        assert changes["unchanged"] == ["a.py"]
        assert changes["added"] == []
        assert changes["modified"] == []


def test_snapshot_ref_reuses_inventory_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_snapshot_ref_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)

        def _boom_hash(_path: str) -> str:
            raise AssertionError("snapshot identity should use inventory fingerprints")

        monkeypatch.setattr("fastcode.indexing.pipeline.compute_file_hash", _boom_hash)

        snapshot_ref = pipeline._resolve_snapshot_ref(
            "repo",
            current_files=[
                {
                    "path": os.path.join(tmp, "a.py"),
                    "relative_path": "a.py",
                    "size": 12,
                    "mtime": 123.0,
                    "content_hash": "hash-a",
                    "blob_oid": "hash-a",
                }
            ],
        )

        assert snapshot_ref["snapshot_id"].startswith("snap:repo:")


def test_pipeline_incremental_prefilter_falls_back_on_compatibility_mismatch() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_incremental_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)

        unchanged_path = os.path.join(tmp, "a.py")
        with open(unchanged_path, "w", encoding="utf-8") as handle:
            handle.write("print('a')\n")
        fingerprint = pipeline._file_fingerprint(unchanged_path)
        assert fingerprint is not None

        previous_snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:prev",
            branch="main",
            commit_id="c1",
            tree_id="t1",
        )
        previous_record = pipeline.snapshot_store.save_snapshot(
            previous_snapshot, metadata={}
        )
        pipeline.manifest_store.publish(
            repo_name="repo",
            ref_name="main",
            snapshot_id="snap:repo:prev",
            index_run_id="run_prev",
            status="published",
        )

        with open(
            os.path.join(tmp, f"{previous_record.artifact_key}_metadata.pkl"),
            "wb",
        ) as handle:
            pickle.dump(
                {
                    "metadata": [
                        {
                            "id": "unchanged:1",
                            "type": "file",
                            "name": "a.py",
                            "file_path": unchanged_path,
                            "relative_path": "a.py",
                            "language": "python",
                            "start_line": 1,
                            "end_line": 1,
                            "code": "print('a')\n",
                            "signature": None,
                            "docstring": None,
                            "summary": None,
                            "metadata": {"embedding": np.array([0.1, 0.2, 0.3])},
                            "repo_name": "repo",
                            "repo_url": tmp,
                        }
                    ]
                },
                handle,
            )

        incompatible_payload = pipeline._incremental_compatibility_payload()
        incompatible_payload["embedding"] = {
            **incompatible_payload["embedding"],
            "model": "previous-model",
        }
        with open(
            os.path.join(tmp, f"{previous_record.artifact_key}_manifest.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "schema_version": 2,
                    "compatibility": incompatible_payload,
                    "compatibility_hash": "previous-compatibility-hash",
                    "files": {
                        "a.py": {
                            **fingerprint,
                            "element_ids": ["unchanged:1"],
                        }
                    },
                },
                handle,
            )

        pipeline.loader.scan_files = lambda: [
            {
                "path": unchanged_path,
                "relative_path": "a.py",
                "size": os.path.getsize(unchanged_path),
                "extension": ".py",
            }
        ]
        pipeline.indexer.index_files = Mock(
            side_effect=AssertionError("index_files should not run")
        )

        planned_elements, plan = pipeline._plan_incremental_elements(
            repo_name="repo",
            repo_url=tmp,
            snapshot_id="snap:repo:current",
            snapshot_ref={"branch": "main", "snapshot_id": "snap:repo:current"},
            ref=None,
        )

        assert planned_elements is None
        assert plan is None
        pipeline.indexer.index_files.assert_not_called()


def test_incremental_compatibility_uses_typed_embedding_fingerprint_payload() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_embedding_fp_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        payload = {
            "version": 2,
            "provider": "sentence_transformers",
            "model": "typed-model",
            "dimension": 3,
            "max_seq_length": 512,
            "normalize": True,
            "ollama_url": None,
            "cache_version": "v1",
        }

        class _Fingerprint:
            def to_payload(self) -> dict[str, Any]:
                return dict(payload)

        def _embedding_fingerprint_record() -> _Fingerprint:
            return _Fingerprint()

        pipeline.embedder = SimpleNamespace(
            embedding_fingerprint_record=_embedding_fingerprint_record,
            embedding_fingerprint=lambda: {"legacy": True},
        )

        assert pipeline._incremental_compatibility_payload()["embedding"] == payload


def test_file_manifest_persists_embedding_fingerprint_surface() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_file_manifest_fp_") as tmp:
        root = Path(tmp)
        source_path = root / "a.py"
        source_path.write_text("def a():\n    return 1\n", encoding="utf-8")
        pipeline = _make_minimal_pipeline(tmp)
        fingerprint = {
            "version": 2,
            "provider": "test",
            "model": "stub",
            "dimension": 3,
            "text_schema_version": 1,
        }
        pipeline.embedder = SimpleNamespace(
            embedding_fingerprint_record=lambda: SimpleNamespace(
                to_payload=lambda: dict(fingerprint)
            )
        )
        element = CodeElement(
            id="elem:a",
            type="function",
            name="a",
            file_path=str(source_path),
            relative_path="a.py",
            language="python",
            start_line=1,
            end_line=2,
            code="def a():\n    return 1\n",
            signature="def a()",
            docstring=None,
            summary=None,
            metadata={},
            repo_name="repo",
            repo_url=None,
        )

        manifest = pipeline._build_file_manifest([element], tmp)

        assert manifest["embedding_fingerprint"] == fingerprint
        assert manifest["compatibility"]["embedding"] == fingerprint
        assert manifest["files"]["a.py"]["embedding_fingerprint"] == fingerprint


def test_incremental_compatibility_does_not_touch_embedding_dim_property() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_embedding_lazy_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        payload = {
            "version": 2,
            "provider": "sentence_transformers",
            "model": "typed-model",
            "dimension": None,
            "max_seq_length": 512,
            "normalize": True,
            "text_schema_version": 1,
            "ollama_url": None,
            "cache_version": None,
        }

        class _Fingerprint:
            def to_payload(self) -> dict[str, Any]:
                return dict(payload)

        class _LazyEmbedder:
            provider = "sentence_transformers"
            model_name = "typed-model"
            max_seq_length = 512
            normalize = True

            @property
            def embedding_dim(self) -> int:
                raise AssertionError(
                    "compatibility planning should not probe dimension"
                )

            def embedding_fingerprint_record(self) -> _Fingerprint:
                return _Fingerprint()

        pipeline.embedder = _LazyEmbedder()

        assert pipeline._incremental_compatibility_payload()["embedding"] == payload


def test_plan_incremental_elements_disables_when_manifest_lacks_fingerprint() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_incremental_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)

        unchanged_path = os.path.join(tmp, "a.py")
        with open(unchanged_path, "w", encoding="utf-8") as handle:
            handle.write("print('a')\n")
        stat = os.stat(unchanged_path)

        previous_snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:prev",
            branch="main",
            commit_id="c1",
            tree_id="t1",
        )
        previous_record = pipeline.snapshot_store.save_snapshot(
            previous_snapshot, metadata={}
        )
        pipeline.manifest_store.publish(
            repo_name="repo",
            ref_name="main",
            snapshot_id="snap:repo:prev",
            index_run_id="run_prev",
            status="published",
        )

        with open(
            os.path.join(tmp, f"{previous_record.artifact_key}_manifest.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "schema_version": 2,
                    "compatibility": pipeline._incremental_compatibility_payload(),
                    "compatibility_hash": pipeline._incremental_compatibility_hash(),
                    "files": {
                        "a.py": {
                            "mtime": stat.st_mtime,
                            "size": stat.st_size,
                            "content_hash": None,
                            "element_ids": ["unchanged:1"],
                        }
                    },
                },
                handle,
            )

        pipeline.loader.scan_files = lambda: [
            {
                "path": unchanged_path,
                "relative_path": "a.py",
                "size": stat.st_size,
                "extension": ".py",
            }
        ]
        pipeline.indexer.index_files = Mock(
            side_effect=AssertionError("missing fingerprints must disable reuse")
        )

        planned_elements, plan = pipeline._plan_incremental_elements(
            repo_name="repo",
            repo_url=tmp,
            snapshot_id="snap:repo:current",
            snapshot_ref={"branch": "main", "snapshot_id": "snap:repo:current"},
            ref=None,
        )

        assert planned_elements is None
        assert plan is None
        pipeline.indexer.index_files.assert_not_called()


def test_plan_incremental_elements_reuses_changed_unit_embedding_when_text_hash_matches() -> (
    None
):
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_incremental_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)

        changed_path = os.path.join(tmp, "b.py")
        with open(changed_path, "w", encoding="utf-8") as handle:
            handle.write("print('new')\n")

        changed_stat = os.stat(changed_path)
        previous_snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:prev",
            branch="main",
            commit_id="c1",
            tree_id="t1",
        )
        previous_record = pipeline.snapshot_store.save_snapshot(
            previous_snapshot, metadata={}
        )
        pipeline.manifest_store.publish(
            repo_name="repo",
            ref_name="main",
            snapshot_id="snap:repo:prev",
            index_run_id="run_prev",
            status="published",
        )

        previous_embedding_text = (
            "Type: function\nName: foo\nSignature: def foo()\nCode:\nreturn 1"
        )
        previous_embedding_hash = (
            __import__("hashlib")
            .sha256(previous_embedding_text.encode("utf-8"))
            .hexdigest()
        )
        previous_embedding_fingerprint = pipeline._incremental_compatibility_payload()[
            "embedding"
        ]
        with open(
            os.path.join(tmp, f"{previous_record.artifact_key}_metadata.pkl"),
            "wb",
        ) as handle:
            pickle.dump(
                {
                    "metadata": [
                        {
                            "id": "changed:1",
                            "type": "function",
                            "name": "foo",
                            "file_path": changed_path,
                            "relative_path": "b.py",
                            "language": "python",
                            "start_line": 1,
                            "end_line": 1,
                            "code": "return 1",
                            "signature": "def foo()",
                            "docstring": None,
                            "summary": None,
                            "metadata": {
                                "stable_unit_id": "unit:function:stable",
                                "embedding": np.array([0.9, 0.8, 0.7]),
                                "embedding_text": previous_embedding_text,
                                "embedding_text_hash": previous_embedding_hash,
                                "embedding_fingerprint": previous_embedding_fingerprint,
                            },
                            "repo_name": "repo",
                            "repo_url": tmp,
                        }
                    ]
                },
                handle,
            )

        with open(
            os.path.join(tmp, f"{previous_record.artifact_key}_manifest.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            changed_fingerprint = pipeline._file_fingerprint(changed_path)
            assert changed_fingerprint is not None
            json.dump(
                {
                    "schema_version": 2,
                    "compatibility": pipeline._incremental_compatibility_payload(),
                    "compatibility_hash": pipeline._incremental_compatibility_hash(),
                    "files": {
                        "b.py": {
                            "mtime": changed_stat.st_mtime - 10,
                            "size": changed_stat.st_size + 1,
                            "content_hash": "stale-content-hash",
                            "element_ids": ["changed:1"],
                        },
                    },
                },
                handle,
            )

        changed_element = SimpleNamespace(
            id="changed:2",
            type="function",
            name="foo",
            file_path=changed_path,
            relative_path="b.py",
            language="python",
            start_line=1,
            end_line=1,
            code="return 1",
            signature="def foo()",
            docstring=None,
            summary=None,
            metadata={
                "stable_unit_id": "unit:function:stable",
                "embedding_text_hash": previous_embedding_hash,
            },
            repo_name="repo",
            repo_url=tmp,
        )

        pipeline.loader.scan_files = lambda: [
            {
                "path": changed_path,
                "relative_path": "b.py",
                "size": changed_stat.st_size,
                "extension": ".py",
            }
        ]
        pipeline.indexer.index_files = Mock(return_value=[changed_element])

        planned_elements, plan = pipeline._plan_incremental_elements(
            repo_name="repo",
            repo_url=tmp,
            snapshot_id="snap:repo:current",
            snapshot_ref={"branch": "main", "snapshot_id": "snap:repo:current"},
            ref=None,
        )

        assert planned_elements is not None
        assert plan is not None
        assert plan["reused_changed_embeddings"] == 1
        assert plan["semantic_frontier_widened"] == 0
        assert plan["api_frontier_changed"] == 0
        assert plan["api_frontier_changed_paths"] == []
        assert plan["changed_paths"] == ["b.py"]
        assert pipeline._preservable_incremental_sources(plan) == {"scip"}
        assert planned_elements[0].metadata["embedding"] == pytest.approx(
            [0.9, 0.8, 0.7]
        )
        assert planned_elements[0].metadata["embedding_text"] == previous_embedding_text

        repair_task = pipeline._build_repair_frontier_task(
            snapshot_id="snap:repo:current",
            repo_name="repo",
            source=tmp,
            changed_paths=["b.py"],
            modified_count=1,
            widened=False,
            scope_kind="path",
            scope_roots=[],
        )
        assert repair_task is None


@pytest.mark.regression
@pytest.mark.audit_finding("PIPE-001")
def test_reuse_changed_unit_embeddings_propagates_fingerprint_and_artifact_ref() -> (
    None
):
    """PIPE-001: verify all 5 embedding fields propagate through reuse path.

    The reuse path in _reuse_changed_unit_embeddings must propagate:
    embedding, embedding_text, embedding_text_hash, embedding_artifact_ref,
    and embedding_fingerprint. Missing fields would cause downstream PG
    metadata to lack fingerprint provenance.
    """
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_pipe001_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)

        changed_path = os.path.join(tmp, "b.py")
        with open(changed_path, "w", encoding="utf-8") as handle:
            handle.write("print('new')\n")

        changed_stat = os.stat(changed_path)
        previous_snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:prev",
            branch="main",
            commit_id="c1",
            tree_id="t1",
        )
        previous_record = pipeline.snapshot_store.save_snapshot(
            previous_snapshot, metadata={}
        )
        pipeline.manifest_store.publish(
            repo_name="repo",
            ref_name="main",
            snapshot_id="snap:repo:prev",
            index_run_id="run_prev",
            status="published",
        )

        previous_embedding_text = (
            "Type: function\nName: foo\nSignature: def foo()\nCode:\nreturn 1"
        )
        previous_embedding_hash = (
            __import__("hashlib")
            .sha256(previous_embedding_text.encode("utf-8"))
            .hexdigest()
        )
        previous_embedding_fingerprint = pipeline._incremental_compatibility_payload()[
            "embedding"
        ]
        previous_artifact_ref = "embedding:changed:1:abc123"

        with open(
            os.path.join(tmp, f"{previous_record.artifact_key}_metadata.pkl"),
            "wb",
        ) as handle:
            pickle.dump(
                {
                    "metadata": [
                        {
                            "id": "changed:1",
                            "type": "function",
                            "name": "foo",
                            "file_path": changed_path,
                            "relative_path": "b.py",
                            "language": "python",
                            "start_line": 1,
                            "end_line": 1,
                            "code": "return 1",
                            "signature": "def foo()",
                            "docstring": None,
                            "summary": None,
                            "metadata": {
                                "stable_unit_id": "unit:function:stable",
                                "embedding": np.array([0.9, 0.8, 0.7]),
                                "embedding_text": previous_embedding_text,
                                "embedding_text_hash": previous_embedding_hash,
                                "embedding_artifact_ref": previous_artifact_ref,
                                "embedding_fingerprint": previous_embedding_fingerprint,
                            },
                            "repo_name": "repo",
                            "repo_url": tmp,
                        }
                    ]
                },
                handle,
            )

        with open(
            os.path.join(tmp, f"{previous_record.artifact_key}_manifest.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            changed_fingerprint = pipeline._file_fingerprint(changed_path)
            assert changed_fingerprint is not None
            json.dump(
                {
                    "schema_version": 2,
                    "compatibility": pipeline._incremental_compatibility_payload(),
                    "compatibility_hash": pipeline._incremental_compatibility_hash(),
                    "files": {
                        "b.py": {
                            "mtime": changed_stat.st_mtime - 10,
                            "size": changed_stat.st_size + 1,
                            "content_hash": "stale-content-hash",
                            "element_ids": ["changed:1"],
                        },
                    },
                },
                handle,
            )

        changed_element = SimpleNamespace(
            id="changed:2",
            type="function",
            name="foo",
            file_path=changed_path,
            relative_path="b.py",
            language="python",
            start_line=1,
            end_line=1,
            code="return 1",
            signature="def foo()",
            docstring=None,
            summary=None,
            metadata={
                "stable_unit_id": "unit:function:stable",
                "embedding_text_hash": previous_embedding_hash,
            },
            repo_name="repo",
            repo_url=tmp,
        )

        pipeline.loader.scan_files = lambda: [
            {
                "path": changed_path,
                "relative_path": "b.py",
                "size": changed_stat.st_size,
                "extension": ".py",
            }
        ]
        pipeline.indexer.index_files = Mock(return_value=[changed_element])

        planned_elements, plan = pipeline._plan_incremental_elements(
            repo_name="repo",
            repo_url=tmp,
            snapshot_id="snap:repo:current",
            snapshot_ref={"branch": "main", "snapshot_id": "snap:repo:current"},
            ref=None,
        )

        assert planned_elements is not None
        assert plan is not None
        assert plan["reused_changed_embeddings"] == 1

        reused_meta = planned_elements[0].metadata

        assert reused_meta["embedding"] == pytest.approx([0.9, 0.8, 0.7])
        assert reused_meta["embedding_text"] == previous_embedding_text
        assert reused_meta["embedding_text_hash"] == previous_embedding_hash
        assert reused_meta["embedding_artifact_ref"] == previous_artifact_ref, (
            "PIPE-001: embedding_artifact_ref must propagate through reuse"
        )
        assert reused_meta["embedding_fingerprint"] == previous_embedding_fingerprint, (
            "PIPE-001: embedding_fingerprint must propagate through reuse"
        )


def test_plan_incremental_elements_refuses_changed_unit_embedding_with_stale_fingerprint() -> (
    None
):
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_incremental_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)

        changed_path = os.path.join(tmp, "b.py")
        with open(changed_path, "w", encoding="utf-8") as handle:
            handle.write("print('new')\n")

        changed_stat = os.stat(changed_path)
        previous_snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:prev",
            branch="main",
            commit_id="c1",
            tree_id="t1",
        )
        previous_record = pipeline.snapshot_store.save_snapshot(
            previous_snapshot, metadata={}
        )
        pipeline.manifest_store.publish(
            repo_name="repo",
            ref_name="main",
            snapshot_id="snap:repo:prev",
            index_run_id="run_prev",
            status="published",
        )

        previous_embedding_text = (
            "Type: function\nName: foo\nSignature: def foo()\nCode:\nreturn 1"
        )
        previous_embedding_hash = (
            __import__("hashlib")
            .sha256(previous_embedding_text.encode("utf-8"))
            .hexdigest()
        )
        stale_embedding_fingerprint = dict(
            pipeline._incremental_compatibility_payload()["embedding"]
        )
        stale_embedding_fingerprint["model"] = "stale-model"
        with open(
            os.path.join(tmp, f"{previous_record.artifact_key}_metadata.pkl"),
            "wb",
        ) as handle:
            pickle.dump(
                {
                    "metadata": [
                        {
                            "id": "changed:1",
                            "type": "function",
                            "name": "foo",
                            "file_path": changed_path,
                            "relative_path": "b.py",
                            "language": "python",
                            "start_line": 1,
                            "end_line": 1,
                            "code": "return 1",
                            "signature": "def foo()",
                            "docstring": None,
                            "summary": None,
                            "metadata": {
                                "stable_unit_id": "unit:function:stable",
                                "embedding": np.array([0.9, 0.8, 0.7]),
                                "embedding_text": previous_embedding_text,
                                "embedding_text_hash": previous_embedding_hash,
                                "embedding_fingerprint": stale_embedding_fingerprint,
                            },
                            "repo_name": "repo",
                            "repo_url": tmp,
                        }
                    ]
                },
                handle,
            )

        with open(
            os.path.join(tmp, f"{previous_record.artifact_key}_manifest.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            changed_fingerprint = pipeline._file_fingerprint(changed_path)
            assert changed_fingerprint is not None
            json.dump(
                {
                    "schema_version": 2,
                    "compatibility": pipeline._incremental_compatibility_payload(),
                    "compatibility_hash": pipeline._incremental_compatibility_hash(),
                    "files": {
                        "b.py": {
                            "mtime": changed_stat.st_mtime - 10,
                            "size": changed_stat.st_size + 1,
                            "content_hash": "stale-content-hash",
                            "element_ids": ["changed:1"],
                        },
                    },
                },
                handle,
            )

        changed_element = SimpleNamespace(
            id="changed:2",
            type="function",
            name="foo",
            file_path=changed_path,
            relative_path="b.py",
            language="python",
            start_line=1,
            end_line=1,
            code="return 1",
            signature="def foo()",
            docstring=None,
            summary=None,
            metadata={
                "stable_unit_id": "unit:function:stable",
                "embedding_text_hash": previous_embedding_hash,
            },
            repo_name="repo",
            repo_url=tmp,
        )

        pipeline.loader.scan_files = lambda: [
            {
                "path": changed_path,
                "relative_path": "b.py",
                "size": changed_stat.st_size,
                "extension": ".py",
            }
        ]
        pipeline.indexer.index_files = Mock(return_value=[changed_element])

        planned_elements, plan = pipeline._plan_incremental_elements(
            repo_name="repo",
            repo_url=tmp,
            snapshot_id="snap:repo:current",
            snapshot_ref={"branch": "main", "snapshot_id": "snap:repo:current"},
            ref=None,
        )

        assert planned_elements is not None
        assert plan is not None
        assert plan["reused_changed_embeddings"] == 0
        assert "embedding" not in planned_elements[0].metadata


def test_plan_incremental_elements_marks_package_scope_when_api_surface_changes() -> (
    None
):
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_incremental_") as tmp:
        package_dir = os.path.join(tmp, "pkg")
        os.makedirs(package_dir, exist_ok=True)
        pyproject = os.path.join(tmp, "pyproject.toml")
        with open(pyproject, "w", encoding="utf-8") as handle:
            handle.write("[project]\nname='repo'\n")
        changed_path = os.path.join(package_dir, "b.py")
        with open(changed_path, "w", encoding="utf-8") as handle:
            handle.write("print('new')\n")

        pipeline = _make_minimal_pipeline(tmp)
        previous_snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:prev",
            branch="main",
            commit_id="c1",
            tree_id="t1",
        )
        previous_record = pipeline.snapshot_store.save_snapshot(
            previous_snapshot, metadata={}
        )
        pipeline.manifest_store.publish(
            repo_name="repo",
            ref_name="main",
            snapshot_id="snap:repo:prev",
            index_run_id="run_prev",
            status="published",
        )
        with open(
            os.path.join(tmp, f"{previous_record.artifact_key}_metadata.pkl"),
            "wb",
        ) as handle:
            pickle.dump(
                {
                    "metadata": [
                        {
                            "id": "changed:1",
                            "type": "function",
                            "name": "foo",
                            "file_path": changed_path,
                            "relative_path": "pkg/b.py",
                            "language": "python",
                            "start_line": 1,
                            "end_line": 1,
                            "code": "return 1",
                            "signature": "def foo()",
                            "docstring": None,
                            "summary": None,
                            "metadata": {
                                "stable_unit_id": "unit:function:stable",
                                "api_surface_hash": "old-hash",
                                "signature_hash": "old-sig",
                                "edge_surface_hash": "old-edge",
                                "embedding_text_hash": "old-embed",
                            },
                            "repo_name": "repo",
                            "repo_url": tmp,
                        }
                    ]
                },
                handle,
            )
        with open(
            os.path.join(tmp, f"{previous_record.artifact_key}_manifest.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            changed_fingerprint = pipeline._file_fingerprint(changed_path)
            assert changed_fingerprint is not None
            json.dump(
                {
                    "schema_version": 2,
                    "compatibility": pipeline._incremental_compatibility_payload(),
                    "compatibility_hash": pipeline._incremental_compatibility_hash(),
                    "files": {
                        "pkg/b.py": {
                            "mtime": changed_fingerprint["mtime"] - 10,
                            "size": changed_fingerprint["size"] + 1,
                            "content_hash": "stale-content-hash",
                            "element_ids": ["changed:1"],
                        }
                    },
                },
                handle,
            )
        changed_element = SimpleNamespace(
            id="changed:2",
            type="function",
            name="foo",
            file_path=changed_path,
            relative_path="pkg/b.py",
            language="python",
            start_line=1,
            end_line=1,
            code="return 1",
            signature="def foo()",
            docstring=None,
            summary=None,
            metadata={
                "stable_unit_id": "unit:function:stable",
                "api_surface_hash": "new-hash",
                "signature_hash": "new-sig",
                "edge_surface_hash": "new-edge",
                "embedding_text_hash": "new-embed",
            },
            repo_name="repo",
            repo_url=tmp,
        )
        pipeline.loader.scan_files = lambda: [
            {
                "path": changed_path,
                "relative_path": "pkg/b.py",
                "size": os.path.getsize(changed_path),
                "extension": ".py",
            }
        ]
        pipeline.indexer.index_files = Mock(return_value=[changed_element])

        _, plan = pipeline._plan_incremental_elements(
            repo_name="repo",
            repo_url=tmp,
            snapshot_id="snap:repo:current",
            snapshot_ref={"branch": "main", "snapshot_id": "snap:repo:current"},
            ref=None,
        )

        assert plan is not None
        assert plan["api_frontier_changed"] == 1
        assert plan["api_frontier_changed_paths"] == ["pkg/b.py"]
        assert plan["package_scope_roots"] == ["."]
        assert pipeline._preservable_incremental_sources(plan) is None
        assert pipeline._incremental_scip_scope(plan) == {
            "mode": "package",
            "reason": "incremental_surface_frontier",
            "target_paths": ["pkg/b.py"],
            "scope_roots": ["."],
        }


def test_incremental_scip_scope_skips_when_source_owned_evidence_preserved() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_scip_scope_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        plan = {
            "modified": 1,
            "removed": 0,
            "changed_paths": ["pkg/b.py"],
            "change_kinds": ["embedding_text_hash"],
        }

        assert pipeline._incremental_scip_scope(plan) == {
            "mode": "skip",
            "reason": "source_owned_evidence_preserved",
            "target_paths": ["pkg/b.py"],
            "scope_roots": [],
        }


def test_scip_degraded_reasons_capture_widened_and_full_tool_rerun_causes() -> None:
    plan = {
        "modified": 1,
        "removed": 1,
        "changed_paths": ["pkg/b.py"],
        "change_kinds": ["api_surface_hash", "edge_surface_hash"],
        "semantic_frontier_widened": 1,
    }

    reasons = IndexPipeline._scip_degraded_reasons(plan, scip_scope=None)

    assert reasons == [
        "scip_dependency_frontier_changed",
        "scip_frontier_widened",
        "scip_full_rerun:file_delete",
    ]


def test_scoped_scip_frontier_uses_repo_root_mode_and_cache() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_scoped_cache_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        os.makedirs(os.path.join(tmp, "pkg"), exist_ok=True)
        with open(os.path.join(tmp, "pyproject.toml"), "w", encoding="utf-8") as handle:
            handle.write("[project]\nname='repo'\n")
        with open(os.path.join(tmp, "pkg", "a.py"), "w", encoding="utf-8") as handle:
            handle.write("def a():\n    return 1\n")
        snapshot = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:scoped")
        scip_index = SCIPIndex(
            documents=[SCIPDocument(path="pkg/a.py", language="python")],
            indexer_name="scip-python",
            indexer_version="test",
        )
        warnings: list[str] = []
        run_roots: list[str] = []

        def _run_scip(_language: str, repo_path: str, _output_dir: str) -> SCIPIndex:
            run_roots.append(repo_path)
            return scip_index

        with (
            patch("fastcode.indexing.pipeline.run_scip_for_language", _run_scip),
            patch.object(
                pipeline,
                "_copy_scope_root",
                side_effect=AssertionError("scoped SCIP should not copy package roots"),
            ),
        ):
            first, first_languages = pipeline._run_scoped_scip_frontier(
                snapshot=snapshot,
                repo_name="repo",
                scope_kind="package",
                scope_roots=["pkg"],
                target_paths={"pkg/a.py"},
                warnings=warnings,
            )
            second, second_languages = pipeline._run_scoped_scip_frontier(
                snapshot=snapshot,
                repo_name="repo",
                scope_kind="package",
                scope_roots=["pkg"],
                target_paths={"pkg/a.py"},
                warnings=warnings,
            )

        assert first is not None
        assert second is not None
        assert first_languages == ["python"]
        assert second_languages == ["python"]
        assert run_roots == [tmp]
        assert warnings == []
        assert first.metadata["scoped_scip_cache"] == {
            "hits": 0,
            "misses": 1,
            "scope_copies": 0,
            "working_mode": "repo_root_filtered",
        }
        assert second.metadata["scoped_scip_cache"] == {
            "hits": 1,
            "misses": 0,
            "scope_copies": 0,
            "working_mode": "repo_root_filtered",
        }


def test_scoped_scip_cache_key_includes_package_marker_fingerprint() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_scoped_key_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        os.makedirs(os.path.join(tmp, "pkg"), exist_ok=True)
        with open(os.path.join(tmp, "pkg", "a.py"), "w", encoding="utf-8") as handle:
            handle.write("def a():\n    return 1\n")
        marker_path = os.path.join(tmp, "pyproject.toml")
        with open(marker_path, "w", encoding="utf-8") as handle:
            handle.write("[project]\nname='repo'\n")

        first = pipeline._scoped_scip_cache_entry(
            repo_root=tmp,
            language="python",
            scope_root="pkg",
            target_paths={"pkg/a.py"},
        )
        with open(marker_path, "w", encoding="utf-8") as handle:
            handle.write("[project]\nname='repo2'\n")
        second = pipeline._scoped_scip_cache_entry(
            repo_root=tmp,
            language="python",
            scope_root="pkg",
            target_paths={"pkg/a.py"},
        )

        assert first.key != second.key
        assert first.payload["package_markers"][0]["path"] == "pyproject.toml"


def test_resolve_snapshot_ref_uses_dirty_worktree_hash_for_local_changes() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_git_") as tmp:
        repo = Repo.init(tmp)
        file_path = os.path.join(tmp, "a.py")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("print('a')\n")
        repo.index.add(["a.py"])
        repo.index.commit("initial")
        repo.git.clean("-fd")

        pipeline = _make_minimal_pipeline(tmp)
        pipeline.loader.scan_files = lambda: [
            {
                "path": file_path,
                "relative_path": "a.py",
                "size": os.path.getsize(file_path),
                "extension": ".py",
            }
        ]
        with patch.object(Repo, "is_dirty", side_effect=[False, True]):
            clean_ref = pipeline._resolve_snapshot_ref("repo")
            with open(file_path, "w", encoding="utf-8") as handle:
                handle.write("print('b')\n")
            dirty_ref = pipeline._resolve_snapshot_ref("repo")

        assert clean_ref["snapshot_id"] != dirty_ref["snapshot_id"]
        assert ":dirty:" in dirty_ref["snapshot_id"]
        assert ":dirty:" in dirty_ref["tree_id"]


def _expand_repair_pipeline_with_graph(tmp: str) -> tuple[IndexPipeline, list[Any]]:
    pipeline = _make_minimal_pipeline(tmp)
    elements = [
        SimpleNamespace(id="elem:a", relative_path="pkg/a.py", file_path="pkg/a.py"),
        SimpleNamespace(id="elem:b", relative_path="pkg/b.py", file_path="pkg/b.py"),
        SimpleNamespace(id="elem:c", relative_path="pkg/c.py", file_path="pkg/c.py"),
        SimpleNamespace(id="elem:d", relative_path="pkg/d.py", file_path="pkg/d.py"),
    ]

    class _StubGraph:
        def __init__(self, edges: dict[str, list[str]]) -> None:
            self._edges = edges

        def __contains__(self, node_id: str) -> bool:
            return node_id in self._edges or any(
                node_id in succ for succ in self._edges.values()
            )

        def predecessors(self, node_id: str) -> list[str]:
            return [src for src, succs in self._edges.items() if node_id in succs]

    pipeline.graph_builder = SimpleNamespace(
        dependency_graph=_StubGraph(
            {"elem:b": ["elem:a"], "elem:c": ["elem:b"], "elem:d": ["elem:c"]}
        ),
        call_graph=_StubGraph({}),
        inheritance_graph=_StubGraph({}),
    )
    return pipeline, elements


def test_expand_repair_target_paths_signature_change_uses_full_bfs() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_expand_sig_") as tmp:
        pipeline, elements = _expand_repair_pipeline_with_graph(tmp)
        scope_paths = {"pkg/a.py", "pkg/b.py", "pkg/c.py", "pkg/d.py"}

        expanded = pipeline._expand_repair_target_paths(
            elements=elements,
            changed_paths=["pkg/a.py"],
            scope_paths=scope_paths,
            change_kinds={"signature_hash"},
        )

        assert expanded == scope_paths


def test_expand_repair_target_paths_edge_only_change_uses_one_hop() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_expand_edge_") as tmp:
        pipeline, elements = _expand_repair_pipeline_with_graph(tmp)
        scope_paths = {"pkg/a.py", "pkg/b.py", "pkg/c.py", "pkg/d.py"}

        expanded = pipeline._expand_repair_target_paths(
            elements=elements,
            changed_paths=["pkg/a.py"],
            scope_paths=scope_paths,
            change_kinds={"edge_surface_hash"},
        )

        assert expanded == {"pkg/a.py", "pkg/b.py"}


def test_expand_repair_target_paths_embedding_only_change_returns_scope_unchanged() -> (
    None
):
    with tempfile.TemporaryDirectory(prefix="fc_expand_embed_") as tmp:
        pipeline, elements = _expand_repair_pipeline_with_graph(tmp)
        scope_paths = {"pkg/a.py", "pkg/b.py", "pkg/c.py", "pkg/d.py"}

        expanded = pipeline._expand_repair_target_paths(
            elements=elements,
            changed_paths=["pkg/a.py"],
            scope_paths=scope_paths,
            change_kinds={"embedding_text_hash"},
        )

        assert expanded == scope_paths


def test_run_semantic_repair_frontier_uses_package_scope_paths() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_repair_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:repair",
            branch="main",
            commit_id="c1",
            tree_id="t1",
            metadata={"semantic_resolver_runs": []},
        )
        record = pipeline.snapshot_store.save_snapshot(snapshot, metadata={})
        pipeline.snapshot_store.save_relational_facts = Mock(return_value=None)
        pipeline.snapshot_store.save_ir_graphs = Mock(return_value=None)
        pipeline.snapshot_symbol_index.register_snapshot = Mock(return_value=None)
        pipeline.ir_graph_builder.build_graphs = Mock(return_value=SimpleNamespace())
        pipeline._load_artifacts_by_key = Mock(return_value=True)
        pipeline.loader.repo_path = tmp
        pipeline._reconstruct_elements_from_metadata = Mock(
            return_value=[
                SimpleNamespace(
                    id="elem:a",
                    relative_path="pkg/a.py",
                    file_path="pkg/a.py",
                    to_dict=lambda: {
                        "id": "elem:a",
                        "type": "function",
                        "relative_path": "pkg/a.py",
                        "metadata": {"stable_unit_id": "unit:function:a"},
                    },
                ),
                SimpleNamespace(
                    id="elem:b",
                    relative_path="pkg/sub/b.py",
                    file_path="pkg/sub/b.py",
                    to_dict=lambda: {
                        "id": "elem:b",
                        "type": "function",
                        "relative_path": "pkg/sub/b.py",
                        "metadata": {"stable_unit_id": "unit:function:b"},
                    },
                ),
                SimpleNamespace(
                    id="elem:c",
                    relative_path="other/c.py",
                    file_path="other/c.py",
                    to_dict=lambda: {
                        "id": "elem:c",
                        "type": "function",
                        "relative_path": "other/c.py",
                        "metadata": {"stable_unit_id": "unit:function:c"},
                    },
                ),
            ]
        )
        pipeline._apply_semantic_resolvers = Mock(return_value=snapshot)
        with open(os.path.join(tmp, "pyproject.toml"), "w", encoding="utf-8") as handle:
            handle.write("[project]\nname='repo'\n")
        os.makedirs(os.path.join(tmp, "pkg", "sub"), exist_ok=True)
        with open(os.path.join(tmp, "pkg", "a.py"), "w", encoding="utf-8") as handle:
            handle.write("print('a')\n")
        with open(
            os.path.join(tmp, "pkg", "sub", "b.py"), "w", encoding="utf-8"
        ) as handle:
            handle.write("print('b')\n")
        pipeline.unit_artifact_store.replace_snapshot_units(
            "snap:repo:repair",
            elements=[
                {
                    "type": "function",
                    "relative_path": "pkg/a.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:a",
                        "signature_hash": "sig:a:old",
                        "embedding_artifact_ref": "embedding:a:old",
                        "package_root": "pkg",
                    },
                },
                {
                    "type": "function",
                    "relative_path": "pkg/sub/b.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:b",
                        "signature_hash": "sig:b:old",
                        "embedding_artifact_ref": "embedding:b:old",
                        "package_root": "pkg",
                    },
                },
                {
                    "type": "function",
                    "relative_path": "other/c.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:c",
                        "signature_hash": "sig:c:keep",
                        "embedding_artifact_ref": "embedding:c",
                        "package_root": "other",
                    },
                },
            ],
        )
        with patch(
            "fastcode.indexing.pipeline.run_scip_for_language",
            return_value=None,
        ):
            result = pipeline.run_semantic_repair_frontier(
                snapshot_id=record.snapshot_id,
                scope_kind="package",
                scope_roots=["pkg"],
                changed_paths=["pkg/a.py"],
                repo_name="repo",
            )

        assert result["status"] == "repaired"
        assert sorted(result["repair_frontier"]["target_paths"]) == [
            "pkg/a.py",
            "pkg/sub/b.py",
        ]
        assert result["repair_frontier"]["tool_rerun_languages"] == ["python"]
        kwargs = pipeline._apply_semantic_resolvers.call_args.kwargs
        assert kwargs["budget"] == "repair_frontier"
        assert kwargs["target_paths"] == {"pkg/a.py", "pkg/sub/b.py"}
        stored_units = pipeline.unit_artifact_store.list_snapshot_units(
            "snap:repo:repair"
        )
        assert {row["relative_path"] for row in stored_units} == {
            "pkg/a.py",
            "pkg/sub/b.py",
            "other/c.py",
        }
        rows_by_path = {row["relative_path"]: row for row in stored_units}
        assert rows_by_path["pkg/a.py"]["scoped_tool_ref"]
        assert rows_by_path["pkg/a.py"]["metadata"]["repair_frontier_summary"]
        assert rows_by_path["other/c.py"]["metadata"]["signature_hash"] == "sig:c:keep"


def test_run_semantic_repair_frontier_refresh_drops_missing_target_units() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_repair_refresh_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:repair-refresh",
            branch="main",
            commit_id="c1",
            tree_id="t1",
            metadata={"semantic_resolver_runs": []},
        )
        record = pipeline.snapshot_store.save_snapshot(snapshot, metadata={})
        pipeline.snapshot_store.save_relational_facts = Mock(return_value=None)
        pipeline.snapshot_store.save_ir_graphs = Mock(return_value=None)
        pipeline.snapshot_symbol_index.register_snapshot = Mock(return_value=None)
        pipeline.ir_graph_builder.build_graphs = Mock(return_value=SimpleNamespace())
        pipeline._load_artifacts_by_key = Mock(return_value=True)
        pipeline.loader.repo_path = tmp
        pipeline._reconstruct_elements_from_metadata = Mock(
            return_value=[
                SimpleNamespace(
                    id="elem:a",
                    relative_path="pkg/a.py",
                    file_path="pkg/a.py",
                    to_dict=lambda: {
                        "id": "elem:a",
                        "type": "function",
                        "relative_path": "pkg/a.py",
                        "metadata": {
                            "stable_unit_id": "unit:function:a",
                            "signature_hash": "sig:a:new",
                        },
                    },
                ),
                SimpleNamespace(
                    id="elem:b",
                    relative_path="pkg/sub/b.py",
                    file_path="pkg/sub/b.py",
                    to_dict=lambda: {
                        "id": "elem:b",
                        "type": "function",
                        "relative_path": "pkg/sub/b.py",
                        "metadata": {
                            "stable_unit_id": "unit:function:b",
                            "signature_hash": "sig:b:new",
                        },
                    },
                ),
                SimpleNamespace(
                    id="elem:c",
                    relative_path="other/c.py",
                    file_path="other/c.py",
                    to_dict=lambda: {
                        "id": "elem:c",
                        "type": "function",
                        "relative_path": "other/c.py",
                        "metadata": {"stable_unit_id": "unit:function:c"},
                    },
                ),
            ]
        )
        repaired_snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id=record.snapshot_id,
            branch="main",
            commit_id="c1",
            tree_id="t1",
            units=[
                IRCodeUnit(
                    unit_id="unit:a:current",
                    kind="function",
                    path="pkg/a.py",
                    language="python",
                    display_name="a",
                    metadata={
                        "stable_unit_id": "unit:function:a",
                        "signature_hash": "sig:a:new",
                    },
                ),
                IRCodeUnit(
                    unit_id="unit:c:current",
                    kind="function",
                    path="other/c.py",
                    language="python",
                    display_name="c",
                    metadata={
                        "stable_unit_id": "unit:function:c",
                        "signature_hash": "sig:c:keep",
                    },
                ),
            ],
            metadata={"semantic_resolver_runs": []},
        )
        pipeline._apply_semantic_resolvers = Mock(return_value=repaired_snapshot)
        with open(os.path.join(tmp, "pyproject.toml"), "w", encoding="utf-8") as handle:
            handle.write("[project]\nname='repo'\n")
        os.makedirs(os.path.join(tmp, "pkg", "sub"), exist_ok=True)
        with open(os.path.join(tmp, "pkg", "a.py"), "w", encoding="utf-8") as handle:
            handle.write("print('a')\n")
        with open(
            os.path.join(tmp, "pkg", "sub", "b.py"), "w", encoding="utf-8"
        ) as handle:
            handle.write("print('b')\n")
        pipeline.unit_artifact_store.replace_snapshot_units(
            record.snapshot_id,
            elements=[
                {
                    "type": "function",
                    "relative_path": "pkg/a.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:a",
                        "signature_hash": "sig:a:old",
                    },
                },
                {
                    "type": "function",
                    "relative_path": "pkg/sub/b.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:b",
                        "signature_hash": "sig:b:old",
                    },
                },
                {
                    "type": "function",
                    "relative_path": "other/c.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:c",
                        "signature_hash": "sig:c:keep",
                    },
                },
            ],
        )

        with patch(
            "fastcode.indexing.pipeline.run_scip_for_language",
            return_value=None,
        ):
            result = pipeline.run_semantic_repair_frontier(
                snapshot_id=record.snapshot_id,
                scope_kind="package",
                scope_roots=["pkg"],
                changed_paths=["pkg/a.py"],
                repo_name="repo",
            )

        assert result["status"] == "repaired"
        stored_units = pipeline.unit_artifact_store.list_snapshot_units(
            record.snapshot_id
        )
        rows_by_id = {row["stable_unit_id"]: row for row in stored_units}
        assert "unit:function:b" not in rows_by_id
        assert rows_by_id["unit:function:a"]["signature_hash"] == "sig:a:new"
        assert rows_by_id["unit:function:c"]["signature_hash"] == "sig:c:keep"


def test_drop_owned_evidence_removes_scoped_scip_supports_only() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_support_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:supports",
            branch="main",
            commit_id="c1",
            tree_id="t1",
            units=[
                IRCodeUnit(
                    unit_id="unit:a",
                    kind="function",
                    path="pkg/a.py",
                    language="python",
                    display_name="a",
                )
            ],
            supports=[
                IRUnitSupport(
                    support_id="sup:scip",
                    unit_id="unit:a",
                    source="scip",
                    support_kind="occurrence",
                    path="pkg/a.py",
                ),
                IRUnitSupport(
                    support_id="sup:semantic",
                    unit_id="unit:a",
                    source="semantic_resolver",
                    support_kind="occurrence",
                    path="pkg/a.py",
                ),
            ],
            relations=[
                IRRelation(
                    relation_id="rel:scip",
                    src_unit_id="unit:a",
                    dst_unit_id="unit:a",
                    relation_type="calls",
                    resolution_state="derived",
                    support_sources={"scip"},
                    support_ids=["sup:scip"],
                ),
                IRRelation(
                    relation_id="rel:semantic",
                    src_unit_id="unit:a",
                    dst_unit_id="unit:a",
                    relation_type="calls",
                    resolution_state="derived",
                    support_sources={"semantic_resolver"},
                    support_ids=["sup:semantic"],
                ),
            ],
        )

        pruned = pipeline._drop_owned_evidence(
            snapshot=snapshot,
            target_paths={"pkg/a.py"},
            owned_sources={"scip"},
        )

        assert {support.support_id for support in pruned.supports} == {"sup:semantic"}
        assert {relation.relation_id for relation in pruned.relations} == {
            "rel:semantic"
        }


def _setup_repair_pipeline_with_one_changed_path(tmp: str) -> tuple[IndexPipeline, Any]:
    pipeline = _make_minimal_pipeline(tmp)
    snapshot = IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:repo:degraded",
        branch="main",
        commit_id="c1",
        tree_id="t1",
        metadata={"semantic_resolver_runs": []},
    )
    record = pipeline.snapshot_store.save_snapshot(snapshot, metadata={})
    pipeline.snapshot_store.save_relational_facts = Mock(return_value=None)
    pipeline.snapshot_store.save_ir_graphs = Mock(return_value=None)
    pipeline.snapshot_symbol_index.register_snapshot = Mock(return_value=None)
    pipeline.ir_graph_builder.build_graphs = Mock(return_value=SimpleNamespace())
    pipeline._load_artifacts_by_key = Mock(return_value=True)
    pipeline.loader.repo_path = tmp
    pipeline._reconstruct_elements_from_metadata = Mock(
        return_value=[
            SimpleNamespace(
                id="elem:a",
                relative_path="pkg/a.py",
                file_path="pkg/a.py",
                to_dict=lambda: {
                    "id": "elem:a",
                    "type": "function",
                    "relative_path": "pkg/a.py",
                    "metadata": {"stable_unit_id": "unit:function:a"},
                },
            )
        ]
    )
    pipeline._apply_semantic_resolvers = Mock(return_value=snapshot)
    with open(os.path.join(tmp, "pyproject.toml"), "w", encoding="utf-8") as handle:
        handle.write("[project]\nname='repo'\n")
    os.makedirs(os.path.join(tmp, "pkg"), exist_ok=True)
    with open(os.path.join(tmp, "pkg", "a.py"), "w", encoding="utf-8") as handle:
        handle.write("print('a')\n")
    return pipeline, record


def test_run_semantic_repair_frontier_records_scoped_scip_failed_degraded_reason() -> (
    None
):
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_repair_degraded_") as tmp:
        pipeline, record = _setup_repair_pipeline_with_one_changed_path(tmp)

        with patch(
            "fastcode.indexing.pipeline.run_scip_for_language",
            side_effect=RuntimeError("indexer crashed"),
        ):
            result = pipeline.run_semantic_repair_frontier(
                snapshot_id=record.snapshot_id,
                scope_kind="package",
                scope_roots=["pkg"],
                changed_paths=["pkg/a.py"],
                repo_name="repo",
            )

        assert result["status"] == "repaired"
        assert "scoped_scip_failed:pkg" in result["degraded_reasons"]


def test_run_semantic_repair_frontier_records_tooling_repo_fallback_for_path_scope() -> (
    None
):
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_repair_fallback_") as tmp:
        pipeline, record = _setup_repair_pipeline_with_one_changed_path(tmp)
        result = pipeline.run_semantic_repair_frontier(
            snapshot_id=record.snapshot_id,
            scope_kind="path",
            scope_roots=[],
            changed_paths=["pkg/a.py"],
            repo_name="repo",
            change_kinds=["signature_hash"],
        )

        assert result["status"] == "repaired"
        assert "tooling_repo_fallback" in result["degraded_reasons"]


def test_run_semantic_repair_frontier_no_fallback_for_embedding_only_change() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_repair_embed_") as tmp:
        pipeline, record = _setup_repair_pipeline_with_one_changed_path(tmp)
        result = pipeline.run_semantic_repair_frontier(
            snapshot_id=record.snapshot_id,
            scope_kind="path",
            scope_roots=[],
            changed_paths=["pkg/a.py"],
            repo_name="repo",
            change_kinds=["embedding_text_hash"],
        )

        assert result["status"] == "repaired"
        assert result["degraded_reasons"] == []


def test_run_semantic_repair_frontier_records_expansion_widened_past_package() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_pipeline_repair_expand_pkg_") as tmp:
        pipeline = _make_minimal_pipeline(tmp)
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:expand-package",
            branch="main",
            commit_id="c1",
            tree_id="t1",
            metadata={"semantic_resolver_runs": []},
        )
        record = pipeline.snapshot_store.save_snapshot(snapshot, metadata={})
        pipeline.snapshot_store.save_relational_facts = Mock(return_value=None)
        pipeline.snapshot_store.save_ir_graphs = Mock(return_value=None)
        pipeline.snapshot_symbol_index.register_snapshot = Mock(return_value=None)
        pipeline.ir_graph_builder.build_graphs = Mock(return_value=SimpleNamespace())
        pipeline._load_artifacts_by_key = Mock(return_value=True)
        pipeline.loader.repo_path = tmp
        elements = [
            SimpleNamespace(
                id="elem:a",
                relative_path="pkg/a.py",
                file_path="pkg/a.py",
                to_dict=lambda: {
                    "id": "elem:a",
                    "type": "function",
                    "relative_path": "pkg/a.py",
                    "metadata": {"stable_unit_id": "unit:function:a"},
                },
            ),
            SimpleNamespace(
                id="elem:other",
                relative_path="other/c.py",
                file_path="other/c.py",
                to_dict=lambda: {
                    "id": "elem:other",
                    "type": "function",
                    "relative_path": "other/c.py",
                    "metadata": {"stable_unit_id": "unit:function:other"},
                },
            ),
        ]
        pipeline._reconstruct_elements_from_metadata = Mock(return_value=elements)
        pipeline._apply_semantic_resolvers = Mock(return_value=snapshot)

        class _StubGraph:
            def __contains__(self, node_id: str) -> bool:
                return node_id in {"elem:a", "elem:other"}

            def predecessors(self, node_id: str) -> list[str]:
                return ["elem:other"] if node_id == "elem:a" else []

        pipeline.graph_builder = SimpleNamespace(
            dependency_graph=_StubGraph(),
            call_graph=_StubGraph(),
            inheritance_graph=_StubGraph(),
        )
        os.makedirs(os.path.join(tmp, "pkg"), exist_ok=True)
        os.makedirs(os.path.join(tmp, "other"), exist_ok=True)
        with open(os.path.join(tmp, "pkg", "a.py"), "w", encoding="utf-8") as handle:
            handle.write("print('a')\n")
        with open(os.path.join(tmp, "other", "c.py"), "w", encoding="utf-8") as handle:
            handle.write("print('c')\n")

        result = pipeline.run_semantic_repair_frontier(
            snapshot_id=record.snapshot_id,
            scope_kind="path",
            scope_roots=[],
            changed_paths=["pkg/a.py"],
            repo_name="repo",
            change_kinds=["signature_hash"],
        )

        assert result["status"] == "repaired"
        assert "expansion_widened_past_package" in result["degraded_reasons"]
        assert result["repair_frontier"]["target_paths"] == [
            "other/c.py",
            "pkg/a.py",
        ]


# ---------------------------------------------------------------------------
# Regression: stale fencing token must not leave artifact files
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_stale_fencing_token_writes_no_artifact_files() -> None:
    """When validate_fencing_token returns False, the artifact save sequence
    (temp_store.save, temp_retriever.save_bm25, temp_graph.save) must NOT
    have been called.  Artifact persistence must happen AFTER fencing
    validation, not before.
    """
    with tempfile.TemporaryDirectory(prefix="fc_fencing_") as tmp:
        element = SimpleNamespace(
            id="file:a",
            type="file",
            name="a.py",
            file_path="a.py",
            relative_path="a.py",
            language="python",
            start_line=1,
            end_line=1,
            signature=None,
            metadata={"embedding": np.array([0.1, 0.2, 0.3])},
            to_dict=lambda: {"id": "file:a", "relative_path": "a.py", "metadata": {}},
        )
        ast_snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:fence-test",
            branch="main",
            commit_id="c1",
            tree_id="t1",
            units=[
                IRCodeUnit(
                    unit_id="doc:snap:repo:fence-test:a.py",
                    kind="file",
                    path="a.py",
                    language="python",
                    display_name="a.py",
                    source_set={"fc_structure"},
                    metadata={"source": "fc_structure"},
                )
            ],
        )

        artifact_save_calls: list[str] = []
        temp_store = SimpleNamespace(
            metadata=[],
            initialize=lambda dim: None,
            add_vectors=lambda vectors, metadata: None,
            save=lambda key: artifact_save_calls.append(f"store:{key}"),
        )
        temp_graph = SimpleNamespace(
            dependency_graph=SimpleNamespace(
                number_of_nodes=lambda: 0, number_of_edges=lambda: 0
            ),
            inheritance_graph=SimpleNamespace(
                number_of_nodes=lambda: 0, number_of_edges=lambda: 0
            ),
            call_graph=SimpleNamespace(
                number_of_nodes=lambda: 0, number_of_edges=lambda: 0
            ),
            build_graphs=lambda elements, module_resolver, symbol_resolver: None,
            save=lambda key: artifact_save_calls.append(f"graph:{key}"),
        )
        temp_retriever = SimpleNamespace(
            index_for_bm25=lambda elements: None,
            build_repo_overview_bm25=lambda: None,
            save_bm25=lambda key: artifact_save_calls.append(f"bm25:{key}"),
        )

        pipeline = _make_minimal_pipeline(tmp)

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    pipeline.indexer,
                    "extract_elements",
                    return_value=[element],
                )
            )
            stack.enter_context(
                patch(
                    "fastcode.indexing.pipeline.build_ir_from_ast",
                    return_value=ast_snapshot,
                )
            )
            stack.enter_context(
                patch("fastcode.indexing.pipeline.merge_ir", return_value=ast_snapshot)
            )
            stack.enter_context(
                patch("fastcode.indexing.pipeline.validate_snapshot", return_value=[])
            )
            stack.enter_context(
                patch.object(
                    pipeline,
                    "_apply_semantic_resolvers",
                    side_effect=lambda **kwargs: kwargs["snapshot"],
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store,
                    "validate_fencing_token",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch.object(pipeline.snapshot_store, "acquire_lock", return_value=1)
            )
            stack.enter_context(
                patch.object(pipeline.snapshot_store, "release_lock", return_value=None)
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "save_snapshot", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "stage_snapshot", return_value="stage_1"
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "save_ir_graphs", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "import_git_backbone", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "save_relational_facts", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store,
                    "update_snapshot_metadata",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "save_design_documents", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline.snapshot_store, "find_by_artifact_key", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline,
                    "pg_retrieval_store",
                    SimpleNamespace(upsert_elements=lambda **kw: None),
                )
            )
            stack.enter_context(
                patch.object(pipeline, "_load_artifacts_by_key", return_value=True)
            )
            stack.enter_context(
                patch(
                    "fastcode.indexing.pipeline.HybridRetriever",
                    return_value=temp_retriever,
                )
            )
            stack.enter_context(
                patch(
                    "fastcode.indexing.pipeline.CodeGraphBuilder",
                    return_value=temp_graph,
                )
            )
            stack.enter_context(
                patch("fastcode.indexing.pipeline.VectorStore", return_value=temp_store)
            )

            with pytest.raises(RuntimeError, match="stale_lock_detected"):
                pipeline.run_index_pipeline(
                    source=tmp,
                    is_url=False,
                    enable_scip=False,
                    publish=False,
                )

        assert artifact_save_calls == [], (
            f"Artifact saves happened before/despite stale fencing: {artifact_save_calls}"
        )
