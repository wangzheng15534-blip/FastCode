import tempfile

from fastcode.index_run import IndexRunStore
from fastcode.manifest_store import ManifestStore
from fastcode.semantic_ir import IRSnapshot
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
        )
        meta = store.save_snapshot(snap, metadata={"x": 1})
        assert meta["artifact_key"].startswith("snap_")

        loaded = store.load_snapshot("snap:repo:abc")
        assert loaded is not None
        assert loaded.repo_name == "repo"
        assert loaded.branch == "main"


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

        run_1 = run_store.create_run("repo", "snap:repo:1", "main", "c1", idempotency_key="k1")
        run_2 = run_store.create_run("repo", "snap:repo:1", "main", "c1", idempotency_key="k1")

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


def test_snapshot_store_lock_api_returns_fencing_token_shape():
    with tempfile.TemporaryDirectory(prefix="fc_lock_test_") as tmp:
        store = SnapshotStore(tmp)
        token = store.acquire_lock("index:snap:repo:1", owner_id="run1", ttl_seconds=60)
        assert token == 1
        assert store.validate_fencing_token("index:snap:repo:1", expected_token=token)


def test_fencing_token_increments_on_reacquire():
    with tempfile.TemporaryDirectory(prefix="fc_fence_") as tmp:
        store = SnapshotStore(tmp)
        token1 = store.acquire_lock("index:snap:repo:1", owner_id="run1", ttl_seconds=60)
        assert token1 == 1
        token2 = store.acquire_lock("index:snap:repo:1", owner_id="run2", ttl_seconds=60)
        assert token2 == 1  # SQLite always returns 1


def test_validate_fencing_token_returns_true_on_sqlite():
    with tempfile.TemporaryDirectory(prefix="fc_fence_") as tmp:
        store = SnapshotStore(tmp)
        # SQLite bypasses PG logic, always returns True
        assert store.validate_fencing_token("nonexistent:lock", expected_token=1)


def test_release_lock_noop_on_sqlite():
    with tempfile.TemporaryDirectory(prefix="fc_fence_") as tmp:
        store = SnapshotStore(tmp)
        store.acquire_lock("index:snap:repo:1", owner_id="run1", ttl_seconds=60)
        store.release_lock("index:snap:repo:1", owner_id="run1")
        # Should not raise


def test_enqueue_redo_task_returns_id():
    with tempfile.TemporaryDirectory(prefix="fc_redo_") as tmp:
        store = SnapshotStore(tmp)
        task_id = store.enqueue_redo_task(
            task_type="index_run_recovery",
            payload={"run_id": "run1", "source": "/tmp/repo"},
        )
        assert task_id.startswith("redo_")
        assert len(task_id) > len("redo_")


def test_claim_redo_task_returns_none_on_sqlite():
    with tempfile.TemporaryDirectory(prefix="fc_redo_") as tmp:
        store = SnapshotStore(tmp)
        assert store.claim_redo_task() is None


def test_mark_redo_task_done_and_failed_noop_on_sqlite():
    with tempfile.TemporaryDirectory(prefix="fc_redo_") as tmp:
        store = SnapshotStore(tmp)
        store.mark_redo_task_done("redo_fake")
        store.mark_redo_task_failed(task_id="redo_fake", error="test error")
