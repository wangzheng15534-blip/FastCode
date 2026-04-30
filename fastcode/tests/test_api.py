"""
Behavior tests for FastCode API routes.

Uses a real SnapshotStore (SQLite in tmp_path) wired into the API via
monkeypatch so endpoints exercise actual database logic instead of
returning hardcoded mock data.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

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

    Only the attributes and methods accessed by the endpoints under test
    are provided.  Delegates to real SnapshotStore and ManifestStore so
    that database logic (SQL, ordering, 404s) is exercised.
    """

    def __init__(self, snapshot_store: SnapshotStore) -> None:
        self.snapshot_store = snapshot_store
        self.manifest_store = ManifestStore(snapshot_store.db_runtime)

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


@pytest.fixture(autouse=True)
def inline_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run API thread offloads inline to keep route tests deterministic."""

    async def _run_inline(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(api.asyncio, "to_thread", _run_inline)


@pytest.fixture
def stores(tmp_path: Any) -> Any:
    """Yield real storage wired into the module-global FastCode handle."""
    store = SnapshotStore(str(tmp_path / "store"))
    fake = _FakeFastCode(store)
    original = api.fastcode_instance
    api.fastcode_instance = fake
    try:
        yield fake, store
    finally:
        api.fastcode_instance = original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRepoRefs:
    """GET /repos/{repo_name}/refs"""

    def test_returns_empty_list_for_unknown_repo(self, stores: Any) -> None:
        body = asyncio.run(api.get_repo_refs("unknown-repo"))
        assert body["refs"] == []
        assert body["repo_name"] == "unknown-repo"

    def test_returns_refs_after_saving_snapshot(self, stores: Any) -> None:
        _fake_fc, store = stores
        snap = _make_minimal_snapshot(
            repo_name="my-repo",
            snapshot_id="snap:my-repo:deadbeef",
            branch="develop",
            commit_id="deadbeef",
        )
        store.save_snapshot(snap)

        refs = asyncio.run(api.get_repo_refs("my-repo"))["refs"]
        assert len(refs) == 1
        assert refs[0]["branch"] == "develop"
        assert refs[0]["snapshot_id"] == "snap:my-repo:deadbeef"

    def test_multiple_refs_ordered_newest_first(self, stores: Any) -> None:
        _fake_fc, store = stores
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
        refs = asyncio.run(api.get_repo_refs("r"))["refs"]
        assert len(refs) == 2
        assert refs[0]["snapshot_id"] == "snap:r:second"
        assert refs[1]["snapshot_id"] == "snap:r:first"


class TestScipArtifacts:
    """GET /scip/artifacts/{snapshot_id}"""

    def test_returns_404_when_not_found(self, stores: Any) -> None:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(api.get_scip_artifact("snap:x:nonexistent"))
        assert exc_info.value.status_code == 404

    def test_returns_artifact_after_save(self, stores: Any) -> None:
        _fake_fc, store = stores
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

        body = asyncio.run(api.get_scip_artifact("snap:repo:with-scip"))
        assert body["status"] == "success"
        artifact = body["artifact"]
        assert artifact["indexer_name"] == "scip-python"
        assert artifact["checksum"] == "abc123"


class TestManifests:
    """GET /manifests/{repo_name}/{ref_name}"""

    def test_branch_manifest_returns_404_when_not_found(self, stores: Any) -> None:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(api.get_branch_manifest("no-repo", "main"))
        assert exc_info.value.status_code == 404

    def test_branch_manifest_returns_data_after_publish(self, stores: Any) -> None:
        fake_fc, store = stores
        snap = _make_minimal_snapshot(
            repo_name="manifest-repo",
            snapshot_id="snap:manifest-repo:m1",
            branch="main",
            commit_id="c_m1",
        )
        store.save_snapshot(snap)

        fake_fc.manifest_store.publish(
            repo_name="manifest-repo",
            ref_name="main",
            snapshot_id="snap:manifest-repo:m1",
            index_run_id="run_001",
        )

        body = asyncio.run(api.get_branch_manifest("manifest-repo", "main"))
        assert body["status"] == "success"
        assert body["manifest"]["snapshot_id"] == "snap:manifest-repo:m1"
        assert body["manifest"]["repo_name"] == "manifest-repo"

    def test_snapshot_manifest_returns_data_after_publish(self, stores: Any) -> None:
        _fake_fc, store = stores
        snap = _make_minimal_snapshot(
            repo_name="manifest-repo2",
            snapshot_id="snap:manifest-repo2:s1",
            branch="develop",
            commit_id="c_s1",
        )
        store.save_snapshot(snap)

        fake_fc = api.fastcode_instance
        assert fake_fc is not None
        fake_fc.manifest_store.publish(
            repo_name="manifest-repo2",
            ref_name="develop",
            snapshot_id="snap:manifest-repo2:s1",
            index_run_id="run_002",
        )

        body = asyncio.run(api.get_snapshot_manifest("snap:manifest-repo2:s1"))
        assert body["manifest"]["snapshot_id"] == "snap:manifest-repo2:s1"


class TestRootAndHealth:
    """Static endpoints that do not need storage."""

    def test_root_returns_metadata(self) -> None:
        body = asyncio.run(api.root())
        assert body["name"] == "FastCode API"
        assert body["version"] == "2.0.0"
        assert body["status"] == "running"


class TestIndexMultiple:
    """POST /index-multiple explicit source mapping behavior."""

    def test_maps_sources_explicitly_without_model_dump(self) -> None:
        request = api.IndexMultipleRequest(
            sources=[
                api.LoadRepositoryRequest(
                    source="https://example.com/repo.git", is_url=True
                ),
                api.LoadRepositoryRequest(source="/tmp/local-repo", is_url=False),
            ]
        )
        fake_fastcode = MagicMock()
        fake_fastcode.get_repository_stats.return_value = {"repos": 2}

        with patch(
            "fastcode.api._ensure_fastcode_initialized", return_value=fake_fastcode
        ):
            result = asyncio.run(api.index_multiple(request))

        fake_fastcode.load_multiple_repositories.assert_called_once_with(
            [
                {"source": "https://example.com/repo.git", "is_url": True},
                {"source": "/tmp/local-repo", "is_url": False},
            ]
        )
        fake_fastcode.vector_store.invalidate_scan_cache.assert_called_once_with()
        assert result["status"] == "success"
