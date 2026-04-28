"""
Behavior tests for FastCode API routes.

Uses a real SnapshotStore (SQLite in tmp_path) wired into the API via
monkeypatch so endpoints exercise actual database logic instead of
returning hardcoded mock data.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from fastcode import api
from fastcode.manifest_store import ManifestStore
from fastcode.semantic_ir import IRSnapshot
from fastcode.snapshot_store import SnapshotStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_snapshot(
    *,
    repo_name: str = "test-repo",
    snapshot_id: str = "snap:test-repo:abc123",
    branch: str | None = "main",
    commit_id: str | None = "abc123",
) -> IRSnapshot:
    """Create a valid IRSnapshot with no documents/symbols/edges."""
    return IRSnapshot(
        repo_name=repo_name,
        snapshot_id=snapshot_id,
        branch=branch,
        commit_id=commit_id,
        tree_id="tree_001",
    )


class _FakeFastCode:
    """Minimal stand-in for FastCode with real storage backends.

    Only the attributes accessed by the endpoints under test are provided.
    """

    def __init__(self, snapshot_store: SnapshotStore) -> None:
        self.snapshot_store = snapshot_store
        self.manifest_store = ManifestStore(snapshot_store.db_runtime)

    # -- Endpoints delegate to snapshot_store directly --

    def list_repo_refs(self, repo_name: str) -> list[dict[str, Any]]:
        with self.snapshot_store.db_runtime.connect() as conn:
            rows = self.snapshot_store.db_runtime.execute(
                conn,
                """
                SELECT branch, commit_id, tree_id, snapshot_id, created_at
                FROM snapshot_refs
                WHERE repo_name=?
                ORDER BY created_at DESC
                """,
                (repo_name,),
            ).fetchall()
        return [
            d
            for r in rows
            if r
            for d in [self.snapshot_store.db_runtime.row_to_dict(r)]
            if d is not None
        ]

    def get_scip_artifact_ref(self, snapshot_id: str) -> dict[str, Any] | None:
        return self.snapshot_store.get_scip_artifact_ref(snapshot_id)

    def get_branch_manifest(
        self, repo_name: str, ref_name: str
    ) -> dict[str, Any] | None:
        return self.manifest_store.get_branch_manifest(repo_name, ref_name)

    def get_snapshot_manifest(self, snapshot_id: str) -> dict[str, Any] | None:
        return self.manifest_store.get_snapshot_manifest(snapshot_id)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path: Any) -> Any:
    """Yield a TestClient with a real SnapshotStore wired in, restored after."""
    store = SnapshotStore(str(tmp_path / "store"))
    fake = _FakeFastCode(store)
    original = api.fastcode_instance
    api.fastcode_instance = fake
    try:
        yield TestClient(api.app), store
    finally:
        api.fastcode_instance = original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRepoRefs:
    """GET /repos/{repo_name}/refs"""

    def test_returns_empty_list_for_unknown_repo(self, client: Any) -> None:
        c, _store = client
        resp = c.get("/repos/unknown-repo/refs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["refs"] == []
        assert body["repo_name"] == "unknown-repo"

    def test_returns_refs_after_saving_snapshot(self, client: Any) -> None:
        c, store = client
        snap = _make_minimal_snapshot(
            repo_name="my-repo",
            snapshot_id="snap:my-repo:deadbeef",
            branch="develop",
            commit_id="deadbeef",
        )
        store.save_snapshot(snap)

        resp = c.get("/repos/my-repo/refs")
        assert resp.status_code == 200
        refs = resp.json()["refs"]
        assert len(refs) == 1
        assert refs[0]["branch"] == "develop"
        assert refs[0]["snapshot_id"] == "snap:my-repo:deadbeef"

    def test_multiple_refs_ordered_newest_first(self, client: Any) -> None:
        c, store = client
        store.save_snapshot(
            _make_minimal_snapshot(
                repo_name="r",
                snapshot_id="snap:r:first",
                branch="main",
                commit_id="c1",
            )
        )
        store.save_snapshot(
            _make_minimal_snapshot(
                repo_name="r",
                snapshot_id="snap:r:second",
                branch="feature",
                commit_id="c2",
            )
        )
        resp = c.get("/repos/r/refs")
        assert resp.status_code == 200
        refs = resp.json()["refs"]
        assert len(refs) == 2
        # Most recent first
        assert refs[0]["snapshot_id"] == "snap:r:second"
        assert refs[1]["snapshot_id"] == "snap:r:first"


class TestScipArtifacts:
    """GET /scip/artifacts/{snapshot_id}"""

    def test_returns_404_when_not_found(self, client: Any) -> None:
        c, _store = client
        resp = c.get("/scip/artifacts/snap:x:nonexistent")
        assert resp.status_code == 404

    def test_returns_artifact_after_save(self, client: Any) -> None:
        c, store = client
        snap = _make_minimal_snapshot(
            snapshot_id="snap:repo:with-scip",
        )
        store.save_snapshot(snap)
        store.save_scip_artifact_ref(
            "snap:repo:with-scip",
            indexer_name="scip-python",
            indexer_version="1.0",
            artifact_path="/tmp/scip.dump",
            checksum="abc123",
        )

        resp = c.get("/scip/artifacts/snap:repo:with-scip")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        artifact = body["artifact"]
        assert artifact["indexer_name"] == "scip-python"
        assert artifact["checksum"] == "abc123"


class TestManifests:
    """GET /manifests/{repo_name}/{ref_name} and GET /manifests/snapshot/{id}"""

    def test_branch_manifest_returns_404_when_not_found(self, client: Any) -> None:
        c, _store = client
        resp = c.get("/manifests/no-repo/main")
        assert resp.status_code == 404

    def test_snapshot_manifest_returns_404_when_not_found(self, client: Any) -> None:
        c, _store = client
        resp = c.get("/manifests/snapshot/snap:x:none")
        assert resp.status_code == 404

    def test_branch_manifest_returns_data_after_publish(self, client: Any) -> None:
        c, store = client
        snap = _make_minimal_snapshot(
            repo_name="manifest-repo",
            snapshot_id="snap:manifest-repo:m1",
            branch="main",
            commit_id="c_m1",
        )
        store.save_snapshot(snap)

        fake_fc: _FakeFastCode = api.fastcode_instance  # type: ignore[assignment]
        fake_fc.manifest_store.publish(
            repo_name="manifest-repo",
            ref_name="main",
            snapshot_id="snap:manifest-repo:m1",
            index_run_id="run_001",
        )

        resp = c.get("/manifests/manifest-repo/main")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["manifest"]["snapshot_id"] == "snap:manifest-repo:m1"
        assert body["manifest"]["repo_name"] == "manifest-repo"

    def test_snapshot_manifest_returns_data_after_publish(self, client: Any) -> None:
        c, store = client
        snap = _make_minimal_snapshot(
            repo_name="manifest-repo2",
            snapshot_id="snap:manifest-repo2:s1",
            branch="develop",
            commit_id="c_s1",
        )
        store.save_snapshot(snap)

        fake_fc: _FakeFastCode = api.fastcode_instance  # type: ignore[assignment]
        fake_fc.manifest_store.publish(
            repo_name="manifest-repo2",
            ref_name="develop",
            snapshot_id="snap:manifest-repo2:s1",
            index_run_id="run_002",
        )

        resp = c.get("/manifests/snapshot/snap:manifest-repo2:s1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["manifest"]["snapshot_id"] == "snap:manifest-repo2:s1"


class TestRootAndHealth:
    """Static endpoints that do not need storage."""

    def test_root_returns_metadata(self, client: Any) -> None:
        c, _store = client
        resp = c.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "FastCode API"
        assert body["version"] == "2.0.0"
        assert body["status"] == "running"
