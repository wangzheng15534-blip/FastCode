"""Property-based tests for ManifestStore database-level invariants.

Covers schema migration idempotency, manifest chain linking, heads table
correctness, publish return structure, missing lookups, ref independence,
and multi-repo isolation.
"""

from __future__ import annotations

import os
import re
import tempfile

import pytest

from fastcode.manifest_store import ManifestStore

# --- Helpers ---


def _make_store() -> ManifestStore:
    tmpdir = tempfile.mkdtemp(prefix="mfst_db_prop_")
    path = os.path.join(tmpdir, "manifest.db")
    return ManifestStore(path)


# --- Properties ---


@pytest.mark.property
class TestSchemaMigration:
    @pytest.mark.happy
    def test_init_idempotent_reinstantiation(self):
        """HAPPY: re-instantiating ManifestStore on same DB path does not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "manifest.db")
            ManifestStore(path)
            # Second instantiation should succeed without error
            store2 = ManifestStore(path)
            # Functional check: publish still works
            result = store2.publish("repo", "main", "snap1", "run1")
            assert result["manifest_id"] is not None

    @pytest.mark.edge
    def test_init_db_creates_required_tables(self):
        """EDGE: all required tables are created during construction."""
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

    @pytest.mark.happy
    def test_schema_migration_row_written(self):
        """HAPPY: schema_migrations table records the component version."""
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


@pytest.mark.property
class TestManifestChain:
    @pytest.mark.happy
    def test_chain_links_two_publishes(self):
        """HAPPY: two publishes for same repo/ref create linked chain."""
        store = _make_store()
        first = store.publish("repo", "main", "snap_v1", "run1")
        second = store.publish("repo", "main", "snap_v2", "run2")
        assert first["previous_manifest_id"] is None
        assert second["previous_manifest_id"] == first["manifest_id"]

    @pytest.mark.happy
    def test_chain_links_three_publishes(self):
        """HAPPY: three publishes create a three-element chain."""
        store = _make_store()
        m1 = store.publish("repo", "main", "snap1", "run1")
        m2 = store.publish("repo", "main", "snap2", "run2")
        m3 = store.publish("repo", "main", "snap3", "run3")
        assert m1["previous_manifest_id"] is None
        assert m2["previous_manifest_id"] == m1["manifest_id"]
        assert m3["previous_manifest_id"] == m2["manifest_id"]

    @pytest.mark.edge
    def test_chain_with_custom_status(self):
        """EDGE: chain links are maintained regardless of status."""
        store = _make_store()
        m1 = store.publish("repo", "main", "snap1", "run1", status="draft")
        m2 = store.publish("repo", "main", "snap2", "run2", status="published")
        assert m2["previous_manifest_id"] == m1["manifest_id"]
        assert m1["status"] == "draft"
        assert m2["status"] == "published"


@pytest.mark.property
class TestHeadsTablePointsToLatest:
    @pytest.mark.happy
    def test_heads_points_to_last_published(self):
        """HAPPY: get_branch_manifest returns the most recently published manifest."""
        store = _make_store()
        store.publish("repo", "main", "snap_v1", "run1")
        store.publish("repo", "main", "snap_v2", "run2")
        store.publish("repo", "main", "snap_v3", "run3")
        result = store.get_branch_manifest("repo", "main")
        assert result is not None
        assert result["snapshot_id"] == "snap_v3"

    @pytest.mark.happy
    def test_heads_updates_per_publish(self):
        """HAPPY: each publish moves the head forward."""
        store = _make_store()
        m1 = store.publish("repo", "main", "snap1", "r1")
        head = store.get_branch_manifest("repo", "main")
        assert head["manifest_id"] == m1["manifest_id"]

        m2 = store.publish("repo", "main", "snap2", "r2")
        head = store.get_branch_manifest("repo", "main")
        assert head["manifest_id"] == m2["manifest_id"]


@pytest.mark.property
class TestPublishReturnStructure:
    @pytest.mark.happy
    def test_manifest_id_starts_with_prefix(self):
        """HAPPY: manifest_id starts with 'manifest_'."""
        store = _make_store()
        result = store.publish("repo", "main", "snap1", "run1")
        assert result["manifest_id"].startswith("manifest_")

    @pytest.mark.happy
    def test_publish_returns_all_required_fields(self):
        """HAPPY: publish result contains all expected keys."""
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

    @pytest.mark.happy
    def test_publish_fields_match_input(self):
        """HAPPY: publish result fields reflect the input arguments."""
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
    def test_published_at_is_iso_format(self):
        """EDGE: published_at is a valid ISO-style timestamp string."""
        store = _make_store()
        result = store.publish("repo", "main", "snap1", "run1")
        assert isinstance(result["published_at"], str)
        assert re.match(r"\d{4}-\d{2}-\d{2}", result["published_at"])

    @pytest.mark.edge
    def test_manifest_ids_are_unique(self):
        """EDGE: each publish produces a unique manifest_id."""
        store = _make_store()
        ids = set()
        for i in range(5):
            result = store.publish("repo", "main", f"snap{i}", f"run{i}")
            ids.add(result["manifest_id"])
        assert len(ids) == 5


@pytest.mark.property
class TestMissingLookups:
    @pytest.mark.edge
    def test_get_branch_manifest_unknown_returns_none(self):
        """EDGE: get_branch_manifest returns None for unknown repo/ref."""
        store = _make_store()
        assert store.get_branch_manifest("nonexistent", "main") is None

    @pytest.mark.edge
    def test_get_snapshot_manifest_unknown_returns_none(self):
        """EDGE: get_snapshot_manifest returns None for unknown snapshot_id."""
        store = _make_store()
        assert store.get_snapshot_manifest("snap:ghost:0000000") is None

    @pytest.mark.edge
    def test_lookup_after_unpublished_repo(self):
        """EDGE: querying a repo that was never published returns None."""
        store = _make_store()
        store.publish("repo_a", "main", "snap1", "run1")
        assert store.get_branch_manifest("repo_b", "main") is None


@pytest.mark.property
class TestDifferentRefsIndependent:
    @pytest.mark.happy
    def test_same_repo_different_refs_independent_chains(self):
        """HAPPY: different refs for same repo maintain independent chains."""
        store = _make_store()
        m_main = store.publish("repo", "main", "snap_main", "run1")
        m_dev = store.publish("repo", "dev", "snap_dev", "run2")
        assert m_main["previous_manifest_id"] is None
        assert m_dev["previous_manifest_id"] is None

    @pytest.mark.happy
    def test_different_refs_evolve_independently(self):
        """HAPPY: publishing to one ref does not affect another ref's head."""
        store = _make_store()
        store.publish("repo", "main", "snap_m1", "r1")
        store.publish("repo", "dev", "snap_d1", "r2")
        store.publish("repo", "main", "snap_m2", "r3")

        main_head = store.get_branch_manifest("repo", "main")
        dev_head = store.get_branch_manifest("repo", "dev")
        assert main_head["snapshot_id"] == "snap_m2"
        assert dev_head["snapshot_id"] == "snap_d1"

    @pytest.mark.edge
    def test_same_snapshot_different_refs(self):
        """EDGE: same snapshot_id can be published to different refs."""
        store = _make_store()
        store.publish("repo", "main", "snap1", "run1")
        store.publish("repo", "dev", "snap1", "run2")
        main = store.get_branch_manifest("repo", "main")
        dev = store.get_branch_manifest("repo", "dev")
        assert main["snapshot_id"] == "snap1"
        assert dev["snapshot_id"] == "snap1"
        assert main["ref_name"] == "main"
        assert dev["ref_name"] == "dev"


@pytest.mark.property
class TestMultipleRepos:
    @pytest.mark.happy
    def test_different_repos_tracked_independently(self):
        """HAPPY: different repos each track their own manifest chains."""
        store = _make_store()
        store.publish("repo_a", "main", "snap_a1", "run1")
        store.publish("repo_b", "main", "snap_b1", "run2")
        a = store.get_branch_manifest("repo_a", "main")
        b = store.get_branch_manifest("repo_b", "main")
        assert a["snapshot_id"] == "snap_a1"
        assert b["snapshot_id"] == "snap_b1"

    @pytest.mark.happy
    def test_cross_repo_publish_does_not_interfere(self):
        """HAPPY: publishing to repo_b does not affect repo_a's chain."""
        store = _make_store()
        a1 = store.publish("repo_a", "main", "snap_a1", "r1")
        store.publish("repo_b", "main", "snap_b1", "r2")
        store.publish("repo_b", "main", "snap_b2", "r3")
        a2 = store.publish("repo_a", "main", "snap_a2", "r4")
        assert a2["previous_manifest_id"] == a1["manifest_id"]
        # repo_a chain not affected by repo_b publishes
        a_head = store.get_branch_manifest("repo_a", "main")
        assert a_head["snapshot_id"] == "snap_a2"

    @pytest.mark.edge
    def test_snapshot_manifest_across_repos(self):
        """EDGE: get_snapshot_manifest finds the right manifest across repos."""
        store = _make_store()
        store.publish("repo_a", "main", "snap_unique_a", "r1")
        store.publish("repo_b", "main", "snap_unique_b", "r2")
        result = store.get_snapshot_manifest("snap_unique_b")
        assert result is not None
        assert result["repo_name"] == "repo_b"
        assert result["snapshot_id"] == "snap_unique_b"
