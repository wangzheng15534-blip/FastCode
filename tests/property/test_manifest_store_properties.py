"""Property-based tests for manifest_store module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.db_runtime import DBRuntime
from fastcode.manifest_store import ManifestStore


# --- Helpers ---

def _make_store() -> ManifestStore:
    import tempfile, os, uuid
    tmpdir = tempfile.mkdtemp(prefix=f"mfst_{uuid.uuid4().hex[:12]}_")
    path = os.path.join(tmpdir, "test.db")
    return ManifestStore(path)

small_id = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)


# --- Properties ---


@pytest.mark.property
class TestManifestStoreProperties:

    @pytest.mark.happy
    def test_publish_returns_valid_manifest(self):
        """HAPPY: publish returns dict with all required keys."""
        store = _make_store()
        result = store.publish("repo", "main", "snap1", "run1")
        assert result["repo_name"] == "repo"
        assert result["ref_name"] == "main"
        assert result["snapshot_id"] == "snap1"
        assert result["index_run_id"] == "run1"
        assert result["manifest_id"].startswith("manifest_")
        assert result["status"] == "published"
        assert result["published_at"] is not None

    @pytest.mark.happy
    def test_get_branch_manifest_after_publish(self):
        """HAPPY: get_branch_manifest returns published manifest."""
        store = _make_store()
        store.publish("repo", "main", "snap1", "run1")
        result = store.get_branch_manifest("repo", "main")
        assert result is not None
        assert result["snapshot_id"] == "snap1"
        assert result["repo_name"] == "repo"

    @pytest.mark.edge
    def test_get_branch_manifest_missing_returns_none(self):
        """EDGE: get_branch_manifest returns None for unknown repo/ref."""
        store = _make_store()
        result = store.get_branch_manifest("nope", "nope")
        assert result is None

    @pytest.mark.edge
    def test_get_snapshot_manifest_missing_returns_none(self):
        """EDGE: get_snapshot_manifest returns None for unknown snapshot."""
        store = _make_store()
        result = store.get_snapshot_manifest("nope")
        assert result is None

    @pytest.mark.happy
    def test_get_snapshot_manifest_after_publish(self):
        """HAPPY: get_snapshot_manifest returns manifest by snapshot_id."""
        store = _make_store()
        store.publish("repo", "main", "snap1", "run1")
        result = store.get_snapshot_manifest("snap1")
        assert result is not None
        assert result["snapshot_id"] == "snap1"

    @pytest.mark.happy
    def test_publish_overwrites_branch_head(self):
        """HAPPY: second publish to same repo/ref updates branch head."""
        store = _make_store()
        store.publish("repo1", "main", "snap_v1", "run_1")
        store.publish("repo1", "main", "snap_v2", "run_2")
        result = store.get_branch_manifest("repo1", "main")
        assert result is not None
        assert result["snapshot_id"] == "snap_v2"

    @pytest.mark.happy
    def test_publish_chains_previous_manifest(self):
        """HAPPY: second publish links to previous manifest."""
        store = _make_store()
        first = store.publish("repo", "main", "snap_v1", "run1")
        second = store.publish("repo", "main", "snap_v2", "run1")
        assert second["previous_manifest_id"] == first["manifest_id"]
        assert first["previous_manifest_id"] is None

    @pytest.mark.happy
    def test_publish_custom_status(self):
        """HAPPY: publish with custom status."""
        store = _make_store()
        result = store.publish("repo", "main", "snap1", "run1", status="draft")
        assert result["status"] == "draft"

    @pytest.mark.edge
    def test_init_with_string_path(self):
        """EDGE: ManifestStore accepts string path (not just DBRuntime)."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "manifest.db")
            store = ManifestStore(path)
            result = store.publish("repo", "main", "snap1", "run1")
            assert result["manifest_id"] is not None

    @pytest.mark.edge
    def test_init_with_dbruntime(self):
        """EDGE: ManifestStore accepts DBRuntime object directly."""
        rt = DBRuntime(backend="sqlite", sqlite_path=":memory:")
        store = ManifestStore(rt)
        assert store.db_runtime is rt

    @pytest.mark.edge
    def test_first_publish_no_previous(self):
        """EDGE: first publish has no previous_manifest_id."""
        store = _make_store()
        result = store.publish("repo", "main", "snap1", "run1")
        assert result["previous_manifest_id"] is None

    @pytest.mark.edge
    def test_multiple_refs_independent(self):
        """EDGE: different refs track independent manifests."""
        store = _make_store()
        store.publish("repo", "main", "snap_main", "run1")
        store.publish("repo", "dev", "snap_dev", "run2")
        main = store.get_branch_manifest("repo", "main")
        dev = store.get_branch_manifest("repo", "dev")
        assert main["snapshot_id"] == "snap_main"
        assert dev["snapshot_id"] == "snap_dev"

    @pytest.mark.edge
    def test_different_repos_independent(self):
        """EDGE: different repos track independent manifests."""
        store = _make_store()
        store.publish("repo_a", "main", "snap_a", "run1")
        store.publish("repo_b", "main", "snap_b", "run2")
        a = store.get_branch_manifest("repo_a", "main")
        b = store.get_branch_manifest("repo_b", "main")
        assert a["snapshot_id"] == "snap_a"
        assert b["snapshot_id"] == "snap_b"
