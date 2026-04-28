"""Tests for manifest_store module."""

from __future__ import annotations

import os
import re
import tempfile

import pytest
from hypothesis import strategies as st

from fastcode.db_runtime import DBRuntime
from fastcode.manifest_store import ManifestStore

# --- Helpers ---


def _make_store() -> ManifestStore:
    tmpdir = tempfile.mkdtemp(prefix="mfst_prop_")
    path = os.path.join(tmpdir, "test.db")
    return ManifestStore(path)


small_id = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)


# --- Properties ---


class TestManifestStoreProperties:
    def test_publish_returns_valid_manifest_property(self):
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

    def test_get_branch_manifest_after_publish_property(self):
        """HAPPY: get_branch_manifest returns published manifest."""
        store = _make_store()
        store.publish("repo", "main", "snap1", "run1")
        result = store.get_branch_manifest("repo", "main")
        assert result is not None
        assert result["snapshot_id"] == "snap1"
        assert result["repo_name"] == "repo"

    @pytest.mark.edge
    def test_get_branch_manifest_missing_returns_none_property(self):
        """EDGE: get_branch_manifest returns None for unknown repo/ref."""
        store = _make_store()
        result = store.get_branch_manifest("nope", "nope")
        assert result is None

    @pytest.mark.edge
    def test_get_snapshot_manifest_missing_returns_none_property(self):
        """EDGE: get_snapshot_manifest returns None for unknown snapshot."""
        store = _make_store()
        result = store.get_snapshot_manifest("nope")
        assert result is None

    def test_get_snapshot_manifest_after_publish_property(self):
        """HAPPY: get_snapshot_manifest returns manifest by snapshot_id."""
        store = _make_store()
        store.publish("repo", "main", "snap1", "run1")
        result = store.get_snapshot_manifest("snap1")
        assert result is not None
        assert result["snapshot_id"] == "snap1"

    def test_publish_overwrites_branch_head_property(self):
        """HAPPY: second publish to same repo/ref updates branch head."""
        store = _make_store()
        store.publish("repo1", "main", "snap_v1", "run_1")
        store.publish("repo1", "main", "snap_v2", "run_2")
        result = store.get_branch_manifest("repo1", "main")
        assert result is not None
        assert result["snapshot_id"] == "snap_v2"

    def test_publish_chains_previous_manifest_property(self):
        """HAPPY: second publish links to previous manifest."""
        store = _make_store()
        first = store.publish("repo", "main", "snap_v1", "run1")
        second = store.publish("repo", "main", "snap_v2", "run1")
        assert second["previous_manifest_id"] == first["manifest_id"]
        assert first["previous_manifest_id"] is None

    def test_publish_custom_status_property(self):
        """HAPPY: publish with custom status."""
        store = _make_store()
        result = store.publish("repo", "main", "snap1", "run1", status="draft")
        assert result["status"] == "draft"

    @pytest.mark.edge
    def test_init_with_string_path_property(self):
        """EDGE: ManifestStore accepts string path (not just DBRuntime)."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "manifest.db")
            store = ManifestStore(path)
            result = store.publish("repo", "main", "snap1", "run1")
            assert result["manifest_id"] is not None

    @pytest.mark.edge
    def test_init_with_dbruntime_property(self):
        """EDGE: ManifestStore accepts DBRuntime object directly."""
        rt = DBRuntime(backend="sqlite", sqlite_path=":memory:")
        store = ManifestStore(rt)
        assert store.db_runtime is rt

    @pytest.mark.edge
    def test_first_publish_no_previous_property(self):
        """EDGE: first publish has no previous_manifest_id."""
        store = _make_store()
        result = store.publish("repo", "main", "snap1", "run1")
        assert result["previous_manifest_id"] is None

    @pytest.mark.edge
    def test_multiple_refs_independent_property(self):
        """EDGE: different refs track independent manifests."""
        store = _make_store()
        store.publish("repo", "main", "snap_main", "run1")
        store.publish("repo", "dev", "snap_dev", "run2")
        main = store.get_branch_manifest("repo", "main")
        dev = store.get_branch_manifest("repo", "dev")
        assert main["snapshot_id"] == "snap_main"
        assert dev["snapshot_id"] == "snap_dev"

    @pytest.mark.edge
    def test_different_repos_independent_property(self):
        """EDGE: different repos track independent manifests."""
        store = _make_store()
        store.publish("repo_a", "main", "snap_a", "run1")
        store.publish("repo_b", "main", "snap_b", "run2")
        a = store.get_branch_manifest("repo_a", "main")
        b = store.get_branch_manifest("repo_b", "main")
        assert a["snapshot_id"] == "snap_a"
        assert b["snapshot_id"] == "snap_b"

    @pytest.mark.edge
    def test_publish_same_snapshot_different_refs_property(self):
        """EDGE: same snapshot_id published to different refs."""
        store = _make_store()
        store.publish("repo", "main", "snap1", "run1")
        store.publish("repo", "dev", "snap1", "run2")
        main = store.get_branch_manifest("repo", "main")
        dev = store.get_branch_manifest("repo", "dev")
        assert main["snapshot_id"] == "snap1"
        assert dev["snapshot_id"] == "snap1"
        assert main["ref_name"] == "main"
        assert dev["ref_name"] == "dev"

    @pytest.mark.edge
    def test_publish_generates_unique_manifest_ids_property(self):
        """EDGE: each publish gets unique manifest_id."""
        store = _make_store()
        m1 = store.publish("repo", "main", "snap1", "run1")
        m2 = store.publish("repo", "main", "snap2", "run2")
        assert m1["manifest_id"] != m2["manifest_id"]

    @pytest.mark.edge
    def test_publish_published_at_is_iso_format_property(self):
        """EDGE: published_at is a valid timestamp string."""

        store = _make_store()
        result = store.publish("repo", "main", "snap1", "run1")
        ts = result["published_at"]
        assert isinstance(ts, str)
        assert re.match(r"\d{4}-\d{2}-\d{2}", ts)

    @pytest.mark.edge
    def test_chain_across_three_publishes_property(self):
        """EDGE: chain links across 3 sequential publishes."""
        store = _make_store()
        m1 = store.publish("repo", "main", "s1", "r1")
        m2 = store.publish("repo", "main", "s2", "r2")
        m3 = store.publish("repo", "main", "s3", "r3")
        assert m2["previous_manifest_id"] == m1["manifest_id"]
        assert m3["previous_manifest_id"] == m2["manifest_id"]
        assert m1["previous_manifest_id"] is None


# ---------------------------------------------------------------------------
# Database-level invariants
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_init_idempotent_reinstantiation_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "manifest.db")
            ManifestStore(path)
            store2 = ManifestStore(path)
            result = store2.publish("repo", "main", "snap1", "run1")
            assert result["manifest_id"] is not None

    @pytest.mark.edge
    def test_init_db_creates_required_tables_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "manifest.db")
            store = ManifestStore(path)
            with store.db_runtime.connect() as conn:
                tables = [
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                ]
            assert "manifests" in tables
            assert "manifest_heads" in tables
            assert "schema_migrations" in tables

    def test_schema_migration_row_written_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "manifest.db")
            store = ManifestStore(path)
            with store.db_runtime.connect() as conn:
                row = conn.execute(
                    "SELECT component, version FROM schema_migrations WHERE component=?",
                    ("manifest_store",),
                ).fetchone()
            assert row is not None
            assert row[0] == "manifest_store"
            assert row[1] == "v1"


class TestHeadsTablePointsToLatest:
    def test_heads_points_to_last_published_property(self):
        store = _make_store()
        store.publish("repo", "main", "snap_v1", "run1")
        store.publish("repo", "main", "snap_v2", "run2")
        store.publish("repo", "main", "snap_v3", "run3")
        result = store.get_branch_manifest("repo", "main")
        assert result is not None
        assert result["snapshot_id"] == "snap_v3"

    def test_heads_updates_per_publish_property(self):
        store = _make_store()
        m1 = store.publish("repo", "main", "snap1", "r1")
        head = store.get_branch_manifest("repo", "main")
        assert head["manifest_id"] == m1["manifest_id"]

        m2 = store.publish("repo", "main", "snap2", "r2")
        head = store.get_branch_manifest("repo", "main")
        assert head["manifest_id"] == m2["manifest_id"]


class TestPublishReturnStructure:
    def test_publish_returns_all_required_fields_property(self):
        store = _make_store()
        result = store.publish("repo", "main", "snap1", "run1")
        expected_keys = {
            "manifest_id",
            "repo_name",
            "ref_name",
            "snapshot_id",
            "index_run_id",
            "published_at",
            "previous_manifest_id",
            "status",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_publish_fields_match_input_property(self):
        store = _make_store()
        result = store.publish(
            "my_repo", "develop", "snap_42", "run_99", status="draft"
        )
        assert result["repo_name"] == "my_repo"
        assert result["ref_name"] == "develop"
        assert result["snapshot_id"] == "snap_42"
        assert result["index_run_id"] == "run_99"
        assert result["status"] == "draft"

    @pytest.mark.edge
    def test_manifest_ids_are_unique_property(self):
        store = _make_store()
        ids = set()
        for i in range(5):
            result = store.publish("repo", "main", f"snap{i}", f"run{i}")
            ids.add(result["manifest_id"])
        assert len(ids) == 5


class TestCrossRepoSnapshotLookup:
    @pytest.mark.edge
    def test_snapshot_manifest_across_repos_property(self):
        store = _make_store()
        store.publish("repo_a", "main", "snap_unique_a", "r1")
        store.publish("repo_b", "main", "snap_unique_b", "r2")
        result = store.get_snapshot_manifest("snap_unique_b")
        assert result is not None
        assert result["repo_name"] == "repo_b"
        assert result["snapshot_id"] == "snap_unique_b"

    def test_cross_repo_publish_does_not_interfere_property(self):
        store = _make_store()
        a1 = store.publish("repo_a", "main", "snap_a1", "r1")
        store.publish("repo_b", "main", "snap_b1", "r2")
        store.publish("repo_b", "main", "snap_b2", "r3")
        a2 = store.publish("repo_a", "main", "snap_a2", "r4")
        assert a2["previous_manifest_id"] == a1["manifest_id"]
        a_head = store.get_branch_manifest("repo_a", "main")
        assert a_head["snapshot_id"] == "snap_a2"
