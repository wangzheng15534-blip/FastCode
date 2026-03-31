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
