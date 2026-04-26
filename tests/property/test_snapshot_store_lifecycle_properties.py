"""Property-based lifecycle tests for SnapshotStore.

Covers full save/load/update cycles, snapshot isolation, upsert semantics,
SCIP artifact ref lifecycle, IR graphs persistence, end-to-end workflows,
multi-snapshot queries, and metadata variant roundtrips.
"""

from __future__ import annotations

import json
import tempfile

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from fastcode.semantic_ir import (
    IRDocument,
    IROccurrence,
    IRSnapshot,
    IRSymbol,
)
from fastcode.snapshot_store import SnapshotStore

# --- Strategies (self-contained, matching conftest patterns) ---

identifier = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=40,
)

_repo_name_st = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=12,
)
_commit_st = st.text(alphabet="0123456789abcdef", min_size=7, max_size=40)
_branch_st = st.sampled_from(["main", "dev", "feature", "release"])


# --- Helpers ---


def _make_store() -> SnapshotStore:
    tmpdir = tempfile.mkdtemp(prefix="ss_lc_prop_")
    return SnapshotStore(tmpdir)


def _build_snapshot(
    repo: str = "test_repo",
    commit: str = "abc1234",
    branch: str = "main",
    n_docs: int = 1,
    n_symbols: int = 1,
) -> IRSnapshot:
    """Build a minimal connected snapshot for lifecycle tests."""
    docs = []
    syms = []
    occs = []
    for i in range(n_docs):
        doc_id = f"doc:f{i}"
        docs.append(IRDocument(
            doc_id=doc_id, path=f"src/f{i}.py",
            language="python", source_set={"ast"},
        ))
        for j in range(n_symbols):
            sym_id = f"sym:f{i}_s{j}"
            syms.append(IRSymbol(
                symbol_id=sym_id, external_symbol_id=None,
                path=f"src/f{i}.py",
                display_name=f"fn_{j}", kind="function",
                language="python", source_priority=10,
                source_set={"ast"}, start_line=j + 1,
            ))
            occs.append(IROccurrence(
                occurrence_id=f"occ:f{i}_s{j}",
                symbol_id=sym_id, doc_id=doc_id,
                role="definition", start_line=j + 1,
                start_col=0, end_line=j + 1, end_col=0,
                source="ast",
            ))
    return IRSnapshot(
        repo_name=repo,
        snapshot_id=f"snap:{repo}:{commit}",
        branch=branch,
        commit_id=commit,
        documents=docs,
        symbols=syms,
        occurrences=occs,
        metadata={"source_modes": ["ast"]},
    )


# --- Properties ---


@pytest.mark.property
class TestFullLifecycle:

    @pytest.mark.happy
    def test_save_load_update_load_cycle(self):
        """HAPPY: save -> load -> update_metadata -> load verifies metadata update."""
        store = _make_store()
        snap = _build_snapshot("repo_lc", "c001", "main")
        original_meta = {"version": 1, "author": "bot"}
        store.save_snapshot(snap, metadata=original_meta)

        # Load and verify initial metadata
        record = store.get_snapshot_record(snap.snapshot_id)
        assert record is not None
        stored = json.loads(record["metadata_json"])
        assert stored == original_meta

        # Update metadata
        updated_meta = {"version": 2, "author": "human", "reviewed": True}
        store.update_snapshot_metadata(snap.snapshot_id, updated_meta)

        # Load again and verify update
        record2 = store.get_snapshot_record(snap.snapshot_id)
        stored2 = json.loads(record2["metadata_json"])
        assert stored2 == updated_meta
        assert stored2["version"] == 2

    @pytest.mark.happy
    def test_save_load_snapshot_ir_preserved(self):
        """HAPPY: IRSnapshot data is fully preserved across save/load."""
        store = _make_store()
        snap = _build_snapshot("repo_ir", "c002", "dev", n_docs=2, n_symbols=3)
        store.save_snapshot(snap)

        loaded = store.load_snapshot(snap.snapshot_id)
        assert loaded is not None
        assert loaded.repo_name == snap.repo_name
        assert loaded.snapshot_id == snap.snapshot_id
        assert loaded.commit_id == snap.commit_id
        assert len(loaded.documents) == 2
        assert len(loaded.symbols) == 6  # 2 docs * 3 symbols
        assert len(loaded.occurrences) == 6
        assert len(loaded.edges) == 6

    @pytest.mark.happy
    def test_load_nonexistent_returns_none(self):
        """HAPPY: loading a nonexistent snapshot returns None."""
        store = _make_store()
        assert store.load_snapshot("snap:ghost:000") is None

    @pytest.mark.edge
    def test_update_metadata_nonexistent_does_not_raise(self):
        """EDGE: updating metadata for nonexistent snapshot does not raise."""
        store = _make_store()
        store.update_snapshot_metadata("snap:ghost:000", {"x": 1})
        # Should silently update zero rows
        assert store.get_snapshot_record("snap:ghost:000") is None


@pytest.mark.property
class TestSnapshotIsolation:

    @pytest.mark.happy
    def test_two_snapshots_different_repos_no_interference(self):
        """HAPPY: two snapshots for different repos are fully isolated."""
        store = _make_store()
        snap_a = _build_snapshot("repo_alpha", "aaa1111", "main", n_docs=1)
        snap_b = _build_snapshot("repo_beta", "bbb2222", "main", n_docs=3)
        store.save_snapshot(snap_a)
        store.save_snapshot(snap_b)

        loaded_a = store.load_snapshot(snap_a.snapshot_id)
        loaded_b = store.load_snapshot(snap_b.snapshot_id)
        assert loaded_a is not None
        assert loaded_b is not None
        assert loaded_a.repo_name == "repo_alpha"
        assert loaded_b.repo_name == "repo_beta"
        assert len(loaded_a.documents) == 1
        assert len(loaded_b.documents) == 3

    @pytest.mark.edge
    def test_metadata_update_does_not_affect_other_snapshot(self):
        """EDGE: updating metadata for one snapshot doesn't affect another."""
        store = _make_store()
        snap_a = _build_snapshot("repo_a", "c111", "main")
        snap_b = _build_snapshot("repo_b", "c222", "main")
        store.save_snapshot(snap_a, metadata={"owner": "a"})
        store.save_snapshot(snap_b, metadata={"owner": "b"})

        store.update_snapshot_metadata(snap_a.snapshot_id, {"owner": "a_updated"})

        rec_a = store.get_snapshot_record(snap_a.snapshot_id)
        rec_b = store.get_snapshot_record(snap_b.snapshot_id)
        assert json.loads(rec_a["metadata_json"])["owner"] == "a_updated"
        assert json.loads(rec_b["metadata_json"])["owner"] == "b"


@pytest.mark.property
class TestConcurrentSaveSameSnapshotId:

    @pytest.mark.happy
    def test_second_save_wins_upsert(self):
        """HAPPY: saving same snapshot_id twice uses upsert; load returns second."""
        store = _make_store()
        snap_id = "snap:repo:deadbeef"
        snap_v1 = IRSnapshot(
            repo_name="repo",
            snapshot_id=snap_id,
            commit_id="c1",
            documents=[IRDocument(doc_id="d1", path="a.py", language="python")],
        )
        snap_v2 = IRSnapshot(
            repo_name="repo",
            snapshot_id=snap_id,
            commit_id="c2",
            documents=[
                IRDocument(doc_id="d1", path="a.py", language="python"),
                IRDocument(doc_id="d2", path="b.py", language="python"),
            ],
        )
        store.save_snapshot(snap_v1)
        store.save_snapshot(snap_v2)

        loaded = store.load_snapshot(snap_id)
        assert loaded is not None
        assert loaded.commit_id == "c2"
        assert len(loaded.documents) == 2

    @pytest.mark.edge
    def test_upsert_replaces_metadata(self):
        """EDGE: upsert replaces the metadata with the latest value."""
        store = _make_store()
        snap = _build_snapshot("repo", "c001", "main")
        store.save_snapshot(snap, metadata={"v": 1})
        store.save_snapshot(snap, metadata={"v": 2})

        record = store.get_snapshot_record(snap.snapshot_id)
        stored = json.loads(record["metadata_json"])
        assert stored["v"] == 2


@pytest.mark.property
class TestArtifactRefLifecycle:

    @pytest.mark.happy
    def test_save_get_artifact_ref(self):
        """HAPPY: save and retrieve a SCIP artifact ref."""
        store = _make_store()
        snap = _build_snapshot("repo_art", "c001", "main")
        store.save_snapshot(snap)
        store.save_scip_artifact_ref(
            snap.snapshot_id,
            indexer_name="scip-python",
            indexer_version="2.0",
            artifact_path="/data/repo.scip",
            checksum="abc123",
        )

        ref = store.get_scip_artifact_ref(snap.snapshot_id)
        assert ref is not None
        assert ref["indexer_name"] == "scip-python"
        assert ref["indexer_version"] == "2.0"
        assert ref["artifact_path"] == "/data/repo.scip"
        assert ref["checksum"] == "abc123"

    @pytest.mark.happy
    def test_upsert_artifact_ref_updates_indexer(self):
        """HAPPY: upserting artifact ref with different indexer_name updates it."""
        store = _make_store()
        snap = _build_snapshot("repo_art2", "c002", "main")
        store.save_snapshot(snap)

        store.save_scip_artifact_ref(
            snap.snapshot_id,
            indexer_name="scip-python",
            checksum="v1",
        )
        store.save_scip_artifact_ref(
            snap.snapshot_id,
            indexer_name="scip-java",
            checksum="v2",
        )

        ref = store.get_scip_artifact_ref(snap.snapshot_id)
        assert ref["indexer_name"] == "scip-java"
        assert ref["checksum"] == "v2"

    @pytest.mark.edge
    def test_get_artifact_ref_nonexistent_returns_none(self):
        """EDGE: getting artifact ref for unknown snapshot returns None."""
        store = _make_store()
        assert store.get_scip_artifact_ref("snap:ghost:000") is None


@pytest.mark.property
class TestIRGraphsLifecycle:

    @pytest.mark.happy
    def test_save_load_graphs(self):
        """HAPPY: save and load IR graphs via pickle."""
        store = _make_store()
        snap = _build_snapshot("repo_g", "c001", "main")
        store.save_snapshot(snap)

        graphs = {"dependency": {"nodes": [1, 2, 3]}, "call": {"edges": [(1, 2)]}}
        path = store.save_ir_graphs(snap.snapshot_id, graphs)
        assert path.endswith(".pkl")

        loaded = store.load_ir_graphs(snap.snapshot_id)
        assert loaded == graphs

    @pytest.mark.happy
    def test_overwrite_graphs(self):
        """HAPPY: saving graphs twice overwrites the previous version."""
        store = _make_store()
        snap = _build_snapshot("repo_g2", "c002", "dev")
        store.save_snapshot(snap)

        store.save_ir_graphs(snap.snapshot_id, {"version": 1, "nodes": 10})
        store.save_ir_graphs(snap.snapshot_id, {"version": 2, "nodes": 20})

        loaded = store.load_ir_graphs(snap.snapshot_id)
        assert loaded["version"] == 2
        assert loaded["nodes"] == 20

    @pytest.mark.edge
    def test_load_graphs_nonexistent_returns_none(self):
        """EDGE: loading graphs for nonexistent snapshot returns None."""
        store = _make_store()
        assert store.load_ir_graphs("snap:ghost:000") is None

    @pytest.mark.edge
    def test_load_graphs_without_saved_returns_none(self):
        """EDGE: loading graphs when snapshot exists but no graphs saved returns None."""
        store = _make_store()
        snap = _build_snapshot()
        store.save_snapshot(snap)
        assert store.load_ir_graphs(snap.snapshot_id) is None


@pytest.mark.property
class TestFullWorkflow:

    @pytest.mark.happy
    def test_save_snapshot_scip_ref_graphs_query_all(self):
        """HAPPY: full workflow saves snapshot, scip ref, and graphs; all queryable."""
        store = _make_store()
        snap = _build_snapshot("repo_wf", "c_workflow", "main", n_docs=2, n_symbols=2)
        meta = {"pipeline": "full", "step": "final"}
        store.save_snapshot(snap, metadata=meta)

        store.save_scip_artifact_ref(
            snap.snapshot_id,
            indexer_name="scip-python",
            indexer_version="3.0",
            artifact_path="/out/repo.scip",
            checksum="beefdead",
        )

        graphs = {"dependency": [[0, 1], [1, 2]], "call": [[0, 2]]}
        store.save_ir_graphs(snap.snapshot_id, graphs)

        # Query snapshot
        loaded_snap = store.load_snapshot(snap.snapshot_id)
        assert loaded_snap is not None
        assert loaded_snap.repo_name == "repo_wf"
        assert len(loaded_snap.documents) == 2

        # Query record
        record = store.get_snapshot_record(snap.snapshot_id)
        assert record is not None
        stored_meta = json.loads(record["metadata_json"])
        assert stored_meta["pipeline"] == "full"

        # Query scip ref
        ref = store.get_scip_artifact_ref(snap.snapshot_id)
        assert ref is not None
        assert ref["indexer_name"] == "scip-python"
        assert ref["checksum"] == "beefdead"

        # Query graphs
        loaded_graphs = store.load_ir_graphs(snap.snapshot_id)
        assert loaded_graphs == graphs

        # Query by repo+commit
        found = store.find_by_repo_commit("repo_wf", "c_workflow")
        assert found is not None
        assert found["snapshot_id"] == snap.snapshot_id

    @pytest.mark.edge
    def test_workflow_find_by_repo_commit_unknown(self):
        """EDGE: find_by_repo_commit returns None for unknown repo/commit."""
        store = _make_store()
        assert store.find_by_repo_commit("no_repo", "no_commit") is None


@pytest.mark.property
class TestMultipleSnapshotsSameRepoDifferentCommits:

    @given(
        repo=_repo_name_st,
        commits=st.lists(_commit_st, min_size=2, max_size=5, unique=True),
        branch=_branch_st,
    )
    @settings(max_examples=10, deadline=None)
    @pytest.mark.happy
    def test_all_snapshots_findable_individually(self, repo, commits, branch):
        """HAPPY: multiple snapshots for same repo are findable individually."""
        store = _make_store()
        for commit in commits:
            snap = _build_snapshot(repo, commit, branch)
            store.save_snapshot(snap)

        for commit in commits:
            snap_id = f"snap:{repo}:{commit}"
            loaded = store.load_snapshot(snap_id)
            assert loaded is not None
            assert loaded.commit_id == commit
            assert loaded.repo_name == repo

    @given(
        repo=_repo_name_st,
        commit1=_commit_st,
        commit2=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_find_by_repo_commit_distinguishes(self, repo, commit1, commit2, branch):
        """HAPPY: find_by_repo_commit returns the correct record for each commit."""
        assume(commit1 != commit2)
        store = _make_store()
        snap1 = _build_snapshot(repo, commit1, branch)
        snap2 = _build_snapshot(repo, commit2, branch)
        store.save_snapshot(snap1)
        store.save_snapshot(snap2)

        r1 = store.find_by_repo_commit(repo, commit1)
        r2 = store.find_by_repo_commit(repo, commit2)
        assert r1["commit_id"] == commit1
        assert r2["commit_id"] == commit2
        assert r1["snapshot_id"] != r2["snapshot_id"]

    @pytest.mark.edge
    def test_snapshots_do_not_leak_metadata(self):
        """EDGE: saving multiple snapshots keeps metadata independent."""
        store = _make_store()
        snap_a = _build_snapshot("repo_x", "c001", "main")
        snap_b = _build_snapshot("repo_x", "c002", "main")
        store.save_snapshot(snap_a, metadata={"tag": "alpha"})
        store.save_snapshot(snap_b, metadata={"tag": "beta"})

        rec_a = store.get_snapshot_record(snap_a.snapshot_id)
        rec_b = store.get_snapshot_record(snap_b.snapshot_id)
        assert json.loads(rec_a["metadata_json"])["tag"] == "alpha"
        assert json.loads(rec_b["metadata_json"])["tag"] == "beta"


@pytest.mark.property
class TestMetadataVariants:

    @pytest.mark.happy
    def test_empty_dict_metadata_roundtrip(self):
        """HAPPY: empty dict metadata roundtrips correctly."""
        store = _make_store()
        snap = _build_snapshot("repo_meta", "c_empty", "main")
        store.save_snapshot(snap, metadata={})

        record = store.get_snapshot_record(snap.snapshot_id)
        stored = json.loads(record["metadata_json"])
        assert stored == {}

    @pytest.mark.happy
    def test_nested_dict_metadata_roundtrip(self):
        """HAPPY: nested dict metadata roundtrips correctly."""
        store = _make_store()
        snap = _build_snapshot("repo_meta", "c_nested", "main")
        nested = {
            "level1": {
                "level2": {
                    "level3": [1, 2, 3],
                    "flag": True,
                },
                "count": 42,
            },
        }
        store.save_snapshot(snap, metadata=nested)

        record = store.get_snapshot_record(snap.snapshot_id)
        stored = json.loads(record["metadata_json"])
        assert stored == nested

    @pytest.mark.happy
    def test_large_dict_metadata_roundtrip(self):
        """HAPPY: large metadata dict (~10KB) roundtrips correctly."""
        store = _make_store()
        snap = _build_snapshot("repo_meta", "c_large", "main")
        large_meta = {f"key_{i}": f"value_{i}" + "x" * 100 for i in range(50)}
        store.save_snapshot(snap, metadata=large_meta)

        record = store.get_snapshot_record(snap.snapshot_id)
        stored = json.loads(record["metadata_json"])
        assert len(stored) == 50
        for k, v in large_meta.items():
            assert stored[k] == v

    @pytest.mark.edge
    def test_none_metadata_treated_as_empty(self):
        """EDGE: passing None as metadata stores empty dict."""
        store = _make_store()
        snap = _build_snapshot("repo_meta", "c_none", "main")
        store.save_snapshot(snap, metadata=None)

        record = store.get_snapshot_record(snap.snapshot_id)
        stored = json.loads(record["metadata_json"])
        assert stored == {}

    @pytest.mark.edge
    def test_update_metadata_to_empty_dict(self):
        """EDGE: updating metadata to empty dict replaces previous value."""
        store = _make_store()
        snap = _build_snapshot("repo_meta", "c_upd", "main")
        store.save_snapshot(snap, metadata={"rich": True, "data": [1, 2, 3]})

        store.update_snapshot_metadata(snap.snapshot_id, {})

        record = store.get_snapshot_record(snap.snapshot_id)
        stored = json.loads(record["metadata_json"])
        assert stored == {}
