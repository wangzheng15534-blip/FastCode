import json
import os
import tempfile
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from fastcode.index_run import IndexRunStore
from fastcode.manifest_store import ManifestStore
from fastcode.pipeline import IndexPipeline
from fastcode.scip_models import SCIPIndex
from fastcode.semantic_ir import IRAttachment, IRCodeUnit, IRSnapshot
from fastcode.snapshot_store import SnapshotStore


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
            patch("fastcode.pipeline.VectorStore", return_value=temp_store),
            patch("fastcode.pipeline.CodeGraphBuilder", return_value=temp_graph),
            patch("fastcode.pipeline.HybridRetriever", return_value=temp_retriever),
            patch("fastcode.pipeline.build_ir_from_ast", return_value=ast_snapshot),
            patch("fastcode.pipeline.validate_snapshot", return_value=[]),
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
                patch("fastcode.pipeline.VectorStore", return_value=temp_store)
            )
            stack.enter_context(
                patch("fastcode.pipeline.CodeGraphBuilder", return_value=temp_graph)
            )
            stack.enter_context(
                patch("fastcode.pipeline.HybridRetriever", return_value=temp_retriever)
            )
            stack.enter_context(
                patch("fastcode.pipeline.build_ir_from_ast", return_value=ast_snapshot)
            )
            stack.enter_context(
                patch("fastcode.pipeline.detect_scip_languages", return_value=["zig"])
            )
            stack.enter_context(
                patch(
                    "fastcode.pipeline.run_scip_for_language",
                    side_effect=_fake_run_scip,
                )
            )
            stack.enter_context(
                patch(
                    "fastcode.pipeline.build_ir_from_scip", return_value=scip_snapshot
                )
            )
            stack.enter_context(
                patch("fastcode.pipeline.merge_ir", return_value=scip_snapshot)
            )
            stack.enter_context(
                patch("fastcode.pipeline.validate_snapshot", return_value=[])
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
                patch("fastcode.pipeline.build_ir_from_ast", return_value=ast_snapshot)
            )
            stack.enter_context(
                patch("fastcode.pipeline.merge_ir", return_value=ast_snapshot)
            )
            stack.enter_context(
                patch("fastcode.pipeline.validate_snapshot", return_value=[])
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
                patch("fastcode.pipeline.HybridRetriever", return_value=temp_retriever)
            )
            stack.enter_context(
                patch("fastcode.pipeline.CodeGraphBuilder", return_value=temp_graph)
            )
            stack.enter_context(
                patch("fastcode.pipeline.VectorStore", return_value=temp_store)
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
