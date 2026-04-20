"""Property-based tests for manifest_store module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from fastcode.db_runtime import DBRuntime
from fastcode.manifest_store import ManifestStore


# --- Helpers ---

def _make_store() -> ManifestStore:
    import tempfile, os, uuid
    tmpdir = tempfile.mkdtemp(prefix=f"manifest_{uuid.uuid4().hex[:8]}_")
    path = os.path.join(tmpdir, "test.db")
    return ManifestStore(path)

small_id = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)


# --- Properties ---


@pytest.mark.property
class TestManifestStoreProperties:

    @given(repo=small_id, ref=small_id, snap_id=small_id, run_id=small_id)
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_publish_returns_valid_manifest(self, repo, ref, snap_id, run_id):
        """HAPPY: publish returns dict with all required keys."""
        store = _make_store()
        result = store.publish(repo, ref, snap_id, run_id)
        assert result["repo_name"] == repo
        assert result["ref_name"] == ref
        assert result["snapshot_id"] == snap_id
        assert result["index_run_id"] == run_id
        assert result["manifest_id"].startswith("manifest_")
        assert result["status"] == "published"
        assert result["published_at"] is not None

    @given(repo=small_id, ref=small_id, snap_id=small_id, run_id=small_id)
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_get_branch_manifest_after_publish(self, repo, ref, snap_id, run_id):
        """HAPPY: get_branch_manifest returns published manifest."""
        store = _make_store()
        store.publish(repo, ref, snap_id, run_id)
        result = store.get_branch_manifest(repo, ref)
        assert result is not None
        assert result["snapshot_id"] == snap_id
        assert result["repo_name"] == repo

    @given(repo=small_id, ref=small_id)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_get_branch_manifest_missing_returns_none(self, repo, ref):
        """EDGE: get_branch_manifest returns None for unknown repo/ref."""
        store = _make_store()
        result = store.get_branch_manifest(repo, ref)
        assert result is None

    @given(snap_id=small_id)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_get_snapshot_manifest_missing_returns_none(self, snap_id):
        """EDGE: get_snapshot_manifest returns None for unknown snapshot."""
        store = _make_store()
        result = store.get_snapshot_manifest(snap_id)
        assert result is None

    @given(repo=small_id, ref=small_id, snap_id=small_id, run_id=small_id)
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_get_snapshot_manifest_after_publish(self, repo, ref, snap_id, run_id):
        """HAPPY: get_snapshot_manifest returns manifest by snapshot_id."""
        store = _make_store()
        store.publish(repo, ref, snap_id, run_id)
        result = store.get_snapshot_manifest(snap_id)
        assert result is not None
        assert result["snapshot_id"] == snap_id

    @pytest.mark.happy
    def test_publish_overwrites_branch_head(self):
        """HAPPY: second publish to same repo/ref updates branch head."""
        store = _make_store()
        store.publish("repo1", "main", "snap_v1", "run_1")
        store.publish("repo1", "main", "snap_v2", "run_2")
        result = store.get_branch_manifest("repo1", "main")
        assert result is not None
        assert result["snapshot_id"] == "snap_v2"

    @given(repo=small_id, ref=small_id, snap1=small_id, snap2=small_id, run_id=small_id)
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_publish_chains_previous_manifest(self, repo, ref, snap1, snap2, run_id):
        """HAPPY: second publish links to previous manifest."""
        store = _make_store()
        first = store.publish(repo, ref, snap1, run_id)
        second = store.publish(repo, ref, snap2, run_id)
        assert second["previous_manifest_id"] == first["manifest_id"]
        assert first["previous_manifest_id"] is None

    @given(repo=small_id, ref=small_id, snap_id=small_id, run_id=small_id)
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_publish_custom_status(self, repo, ref, snap_id, run_id):
        """HAPPY: publish with custom status."""
        store = _make_store()
        result = store.publish(repo, ref, snap_id, run_id, status="draft")
        assert result["status"] == "draft"

    @given(repo=small_id, ref=small_id, snap_id=small_id, run_id=small_id)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_init_with_string_path(self, repo, ref, snap_id, run_id):
        """EDGE: ManifestStore accepts string path (not just DBRuntime)."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "manifest.db")
            store = ManifestStore(path)
            result = store.publish(repo, ref, snap_id, run_id)
            assert result["manifest_id"] is not None
