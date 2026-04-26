"""Property-based tests for SnapshotStore database-level invariants.

Covers schema migration idempotency, primary key enforcement, UNIQUE
constraints, upsert semantics, large/unicode metadata roundtrips,
pickle-based IR graphs persistence, and query correctness.
"""

from __future__ import annotations

import json
import tempfile

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from fastcode.semantic_ir import (
    IRDocument,
    IREdge,
    IROccurrence,
    IRSnapshot,
    IRSymbol,
)
from fastcode.snapshot_store import SnapshotStore

# --- Strategies (self-contained) ---

identifier = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=40,
)


# --- Strategies ---

_repo_name_st = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz",
    min_size=1,
    max_size=12,
)
_commit_st = st.text(alphabet="0123456789abcdef", min_size=7, max_size=40)
_branch_st = st.sampled_from(["main", "dev", "feature", "release"])
_snapshot_id_st = st.builds(
    lambda repo, commit: f"snap:{repo}:{commit}",
    _repo_name_st,
    _commit_st,
)
_unicode_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "\u00e0\u00e1\u00e2\u00e3\u00e4\u00e5\u00e6\u00e7\u00e8\u00e9"
    "\u4e00\u4e01\u4e02\u4e03\u4e04\u4e05"
    "\u0410\u0411\u0412\u0413\u0414",
    min_size=1,
    max_size=50,
)


# --- Helpers ---


def _make_store() -> SnapshotStore:
    tmpdir = tempfile.mkdtemp(prefix="ss_db_prop_")
    return SnapshotStore(tmpdir)


def _build_snapshot(
    repo: str = "test_repo",
    commit: str = "abc1234",
    branch: str = "main",
    n_docs: int = 1,
    n_symbols: int = 1,
) -> IRSnapshot:
    """Build a minimal connected snapshot for DB tests."""
    docs = []
    syms = []
    occs = []
    edges = []
    for i in range(n_docs):
        doc_id = f"doc:f{i}"
        docs.append(
            IRDocument(
                doc_id=doc_id,
                path=f"src/f{i}.py",
                language="python",
                source_set={"ast"},
            )
        )
        for j in range(n_symbols):
            sym_id = f"sym:f{i}_s{j}"
            syms.append(
                IRSymbol(
                    symbol_id=sym_id,
                    external_symbol_id=None,
                    path=f"src/f{i}.py",
                    display_name=f"fn_{j}",
                    kind="function",
                    language="python",
                    source_priority=10,
                    source_set={"ast"},
                    start_line=j + 1,
                )
            )
            occs.append(
                IROccurrence(
                    occurrence_id=f"occ:f{i}_s{j}",
                    symbol_id=sym_id,
                    doc_id=doc_id,
                    role="definition",
                    start_line=j + 1,
                    start_col=0,
                    end_line=j + 1,
                    end_col=0,
                    source="ast",
                )
            )
            edges.append(
                IREdge(
                    edge_id=f"edge:contain:{doc_id}:{sym_id}",
                    src_id=doc_id,
                    dst_id=sym_id,
                    edge_type="contain",
                    source="ast",
                    confidence="resolved",
                )
            )
    return IRSnapshot(
        repo_name=repo,
        snapshot_id=f"snap:{repo}:{commit}",
        branch=branch,
        commit_id=commit,
        documents=docs,
        symbols=syms,
        occurrences=occs,
        edges=edges,
        metadata={"source_modes": ["ast"]},
    )


# --- Properties ---


@pytest.mark.property
class TestSchemaMigration:
    @pytest.mark.happy
    def test_init_db_idempotent(self):
        """HAPPY: calling _init_db twice does not raise."""
        store = _make_store()
        store._init_db()  # Second call
        # Verify tables still exist
        record = store.get_snapshot_record("nonexistent")
        assert record is None

    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_init_db_after_save_preserves_data(self, repo, commit, branch):
        """HAPPY: re-running _init_db after saves preserves all data."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap)

        # Re-run init
        store._init_db()

        loaded = store.load_snapshot(snap.snapshot_id)
        assert loaded is not None
        assert loaded.repo_name == repo

    @pytest.mark.edge
    def test_init_db_creates_required_tables(self):
        """EDGE: all required tables are created during _init_db."""
        store = _make_store()
        with store.db_runtime.connect() as conn:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
        assert "snapshots" in tables
        assert "snapshot_refs" in tables
        assert "scip_artifacts" in tables
        assert "schema_migrations" in tables


@pytest.mark.property
class TestSnapshotsTableConstraints:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_duplicate_snapshot_id_upserts(self, repo, commit, branch):
        """HAPPY: saving same snapshot_id twice uses upsert (last wins)."""
        store = _make_store()
        snap_id = f"snap:{repo}:{commit}"
        snap1 = _build_snapshot(repo, commit, branch)
        snap2 = _build_snapshot(repo, commit, branch, n_docs=2, n_symbols=2)

        store.save_snapshot(snap1)
        store.save_snapshot(snap2)

        record = store.get_snapshot_record(snap_id)
        assert record is not None
        # Only one record in DB (upsert)
        with store.db_runtime.connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM snapshots WHERE snapshot_id=?",
                (snap_id,),
            ).fetchone()[0]
        assert count == 1

    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_upsert_preserves_latest_data(self, repo, commit, branch):
        """EDGE: upsert overwrites with latest snapshot data."""
        store = _make_store()
        snap_id = f"snap:{repo}:{commit}"
        snap1 = _build_snapshot(repo, commit, branch, n_docs=1)
        store.save_snapshot(snap1)

        # Save with different content
        snap2 = _build_snapshot(repo, commit, branch, n_docs=3)
        store.save_snapshot(snap2)

        loaded = store.load_snapshot(snap_id)
        assert loaded is not None
        assert len(loaded.documents) == 3


@pytest.mark.property
class TestSnapshotRefsConstraints:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_duplicate_ref_does_not_raise(self, repo, commit, branch):
        """HAPPY: saving same (repo, commit, snapshot_id) ref twice is no-op."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap)
        # Saving same snapshot again should not raise
        store.save_snapshot(snap)

    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_resolve_snapshot_for_ref(self, repo, commit, branch):
        """HAPPY: resolve_snapshot_for_ref returns record for known branch."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap)

        result = store.resolve_snapshot_for_ref(repo, branch)
        assert result is not None
        assert result["repo_name"] == repo
        assert result["branch"] == branch
        assert result["snapshot_id"] == snap.snapshot_id

    @given(
        repo=_repo_name_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_resolve_snapshot_for_unknown_ref_returns_none(self, repo, branch):
        """EDGE: resolving unknown branch returns None."""
        store = _make_store()
        result = store.resolve_snapshot_for_ref(repo, branch)
        assert result is None


@pytest.mark.property
class TestScipArtifactsTable:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
        indexer_name=st.sampled_from(["scip-python", "scip-java", "scip-go"]),
    )
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_scip_artifact_upsert(self, repo, commit, branch, indexer_name):
        """HAPPY: saving SCIP artifact ref twice uses upsert."""
        store = _make_store()
        snap_id = f"snap:{repo}:{commit}"
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap)

        store.save_scip_artifact_ref(snap_id, indexer_name=indexer_name, checksum="v1")
        store.save_scip_artifact_ref(snap_id, indexer_name="other", checksum="v2")

        ref = store.get_scip_artifact_ref(snap_id)
        assert ref is not None
        assert ref["indexer_name"] == "other"
        assert ref["checksum"] == "v2"

    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_scip_artifact_fields_roundtrip(self, repo, commit, branch):
        """HAPPY: all SCIP artifact fields survive save/load."""
        store = _make_store()
        snap_id = f"snap:{repo}:{commit}"
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap)

        store.save_scip_artifact_ref(
            snap_id,
            indexer_name="scip-python",
            indexer_version="2.1.0",
            artifact_path="/data/test.scip",
            checksum="deadbeef",
        )

        ref = store.get_scip_artifact_ref(snap_id)
        assert ref["indexer_name"] == "scip-python"
        assert ref["indexer_version"] == "2.1.0"
        assert ref["artifact_path"] == "/data/test.scip"
        assert ref["checksum"] == "deadbeef"
        assert ref["created_at"] is not None


@pytest.mark.property
class TestMultipleSnapshotsSameRepo:
    @given(
        repo=_repo_name_st,
        commits=st.lists(_commit_st, min_size=2, max_size=4, unique=True),
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_multiple_snapshots_all_retrievable(self, repo, commits, branch):
        """HAPPY: multiple snapshots for same repo all retrievable independently."""
        store = _make_store()
        for commit in commits:
            snap = _build_snapshot(repo, commit, branch)
            store.save_snapshot(snap)

        for commit in commits:
            snap_id = f"snap:{repo}:{commit}"
            loaded = store.load_snapshot(snap_id)
            assert loaded is not None
            assert loaded.repo_name == repo
            assert loaded.commit_id == commit

    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_find_by_repo_commit_returns_latest(self, repo, commit, branch):
        """HAPPY: find_by_repo_commit returns a record for the given repo+commit."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap)

        result = store.find_by_repo_commit(repo, commit)
        assert result is not None
        assert result["snapshot_id"] == snap.snapshot_id


@pytest.mark.property
class TestLargeMetadataRoundtrip:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=5)
    @pytest.mark.edge
    def test_large_metadata_roundtrip(self, repo, commit, branch):
        """EDGE: metadata of ~10KB roundtrips correctly."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        large_meta = {f"key_{i}": f"value_{i}" + "x" * 100 for i in range(50)}
        store.save_snapshot(snap, metadata=large_meta)

        record = store.get_snapshot_record(snap.snapshot_id)
        assert record is not None
        stored_meta = json.loads(record.get("metadata_json", "{}"))
        assert len(stored_meta) == 50
        for k, v in large_meta.items():
            assert stored_meta[k] == v


@pytest.mark.property
class TestUnicodeMetadata:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
        unicode_key=_unicode_text,
        unicode_val=_unicode_text,
    )
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_unicode_metadata_preserved(
        self, repo, commit, branch, unicode_key, unicode_val
    ):
        """EDGE: Unicode characters in metadata survive save/load."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        meta = {unicode_key: unicode_val}
        store.save_snapshot(snap, metadata=meta)

        record = store.get_snapshot_record(snap.snapshot_id)
        assert record is not None
        stored_meta = json.loads(record.get("metadata_json", "{}"))
        assert stored_meta.get(unicode_key) == unicode_val


@pytest.mark.property
class TestIRGraphsPickleRoundtrip:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
        n_nodes=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_ir_graphs_pickle_roundtrip(self, repo, commit, branch, n_nodes):
        """HAPPY: save_ir_graphs/load_ir_graphs roundtrips arbitrary data."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap)

        # Use a simple dict as "graphs"
        graphs = {f"node_{i}": {"edges": list(range(i))} for i in range(n_nodes)}
        path = store.save_ir_graphs(snap.snapshot_id, graphs)
        assert path.endswith(".pkl")

        loaded = store.load_ir_graphs(snap.snapshot_id)
        assert loaded == graphs

    @pytest.mark.edge
    def test_load_ir_graphs_nonexistent_returns_none(self):
        """EDGE: loading graphs for nonexistent snapshot returns None."""
        store = _make_store()
        result = store.load_ir_graphs("snap:ghost:0000000")
        assert result is None

    @pytest.mark.edge
    def test_load_ir_graphs_no_saved_graphs_returns_none(self):
        """EDGE: loading graphs when only snapshot exists (no graphs) returns None."""
        store = _make_store()
        snap = _build_snapshot()
        store.save_snapshot(snap)
        result = store.load_ir_graphs(snap.snapshot_id)
        assert result is None

    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=5)
    @pytest.mark.happy
    def test_ir_graphs_overwrite(self, repo, commit, branch):
        """HAPPY: saving graphs twice overwrites previous version."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap)

        store.save_ir_graphs(snap.snapshot_id, {"version": 1})
        store.save_ir_graphs(snap.snapshot_id, {"version": 2})

        loaded = store.load_ir_graphs(snap.snapshot_id)
        assert loaded == {"version": 2}


@pytest.mark.property
class TestSnapshotWithNoneFields:
    @pytest.mark.edge
    def test_snapshot_all_optional_fields_none(self):
        """EDGE: snapshot with all optional fields None still works."""
        store = _make_store()
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:deadbeef",
            branch=None,
            commit_id=None,
            tree_id=None,
            documents=[],
            symbols=[],
            occurrences=[],
            edges=[],
            metadata={"source_modes": ["ast"]},
        )
        result = store.save_snapshot(snap)
        assert result["snapshot_id"] == "snap:repo:deadbeef"

        loaded = store.load_snapshot("snap:repo:deadbeef")
        assert loaded is not None
        assert loaded.branch is None
        assert loaded.commit_id is None
        assert loaded.tree_id is None

    @pytest.mark.edge
    def test_snapshot_empty_metadata(self):
        """EDGE: snapshot with empty metadata dict works."""
        store = _make_store()
        snap = _build_snapshot()
        store.save_snapshot(snap, metadata={})

        record = store.get_snapshot_record(snap.snapshot_id)
        assert record is not None
        stored_meta = json.loads(record.get("metadata_json", "{}"))
        assert stored_meta == {}


@pytest.mark.property
class TestFindLatestBehavior:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_find_by_repo_commit_returns_matching(self, repo, commit, branch):
        """HAPPY: find_by_repo_commit returns a snapshot matching the query."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap)

        result = store.find_by_repo_commit(repo, commit)
        assert result is not None
        assert result["repo_name"] == repo
        assert result["commit_id"] == commit

    @given(
        repo=_repo_name_st,
        commit1=_commit_st,
        commit2=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_find_by_repo_commit_distinguishes_commits(
        self, repo, commit1, commit2, branch
    ):
        """HAPPY: different commits for same repo return distinct records."""
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

    @given(
        repo=_repo_name_st,
        commit1=_commit_st,
        commit2=_commit_st,
        branch1=_branch_st,
        branch2=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_resolve_snapshot_for_ref_distinguishes_branches(
        self, repo, commit1, commit2, branch1, branch2
    ):
        """HAPPY: resolve_snapshot_for_ref distinguishes branches."""
        assume(branch1 != branch2)
        assume(commit1 != commit2)
        store = _make_store()
        snap1 = _build_snapshot(repo, commit1, branch1)
        snap2 = _build_snapshot(repo, commit2, branch2)
        store.save_snapshot(snap1)
        store.save_snapshot(snap2)

        r1 = store.resolve_snapshot_for_ref(repo, branch1)
        r2 = store.resolve_snapshot_for_ref(repo, branch2)
        assert r1 is not None
        assert r2 is not None
        assert r1["branch"] == branch1
        assert r2["branch"] == branch2


@pytest.mark.property
class TestArtifactKeyGeneration:
    @given(
        snap_id=st.builds(lambda x: f"snap:{x}", identifier),
    )
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_artifact_key_deterministic(self, snap_id):
        """HAPPY: artifact_key_for_snapshot is deterministic."""
        store = _make_store()
        key1 = store.artifact_key_for_snapshot(snap_id)
        key2 = store.artifact_key_for_snapshot(snap_id)
        assert key1 == key2
        assert key1.startswith("snap_")

    @given(
        snap_id1=st.builds(lambda x: f"snap:{x}", identifier),
        snap_id2=st.builds(lambda x: f"snap:{x}", identifier),
    )
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_different_snapshot_ids_different_keys(self, snap_id1, snap_id2):
        """HAPPY: different snapshot_ids produce different artifact keys."""
        assume(snap_id1 != snap_id2)
        store = _make_store()
        key1 = store.artifact_key_for_snapshot(snap_id1)
        key2 = store.artifact_key_for_snapshot(snap_id2)
        assert key1 != key2


@pytest.mark.property
class TestSnapshotDir:
    @given(
        snap_id=st.builds(lambda x: f"snap:{x}", identifier),
    )
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_snapshot_dir_exists_after_call(self, snap_id):
        """HAPPY: snapshot_dir creates the directory if needed."""
        store = _make_store()
        d = store.snapshot_dir(snap_id)
        import os

        assert os.path.isdir(d)

    @given(
        snap_id=st.builds(lambda x: f"snap:{x}", identifier),
    )
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_snapshot_dir_idempotent(self, snap_id):
        """HAPPY: calling snapshot_dir twice returns same path."""
        store = _make_store()
        d1 = store.snapshot_dir(snap_id)
        d2 = store.snapshot_dir(snap_id)
        assert d1 == d2


@pytest.mark.property
class TestFindbyArtifactKey:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_find_by_artifact_key_after_save(self, repo, commit, branch):
        """HAPPY: find_by_artifact_key retrieves saved snapshot."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        result = store.save_snapshot(snap)

        found = store.find_by_artifact_key(result["artifact_key"])
        assert found is not None
        assert found["snapshot_id"] == snap.snapshot_id

    @pytest.mark.edge
    def test_find_by_artifact_key_unknown_returns_none(self):
        """EDGE: querying unknown artifact key returns None."""
        store = _make_store()
        result = store.find_by_artifact_key("snap_nonexistent")
        assert result is None


@pytest.mark.property
class TestSnapshotRecordFields:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_snapshot_record_has_all_fields(self, repo, commit, branch):
        """HAPPY: get_snapshot_record returns all expected columns."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap)

        record = store.get_snapshot_record(snap.snapshot_id)
        assert record is not None
        expected_keys = {
            "snapshot_id",
            "repo_name",
            "branch",
            "commit_id",
            "tree_id",
            "artifact_key",
            "ir_path",
            "created_at",
            "metadata_json",
        }
        assert expected_keys.issubset(set(record.keys()))

    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_ir_path_points_to_real_file(self, repo, commit, branch):
        """HAPPY: ir_path in snapshot record points to an existing file."""
        import os

        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap)

        record = store.get_snapshot_record(snap.snapshot_id)
        assert os.path.isfile(record["ir_path"])


@pytest.mark.property
class TestEdgeCases:
    @pytest.mark.edge
    def test_save_load_snapshot_preserves_documents(self):
        """EDGE: documents survive full save/load cycle."""
        store = _make_store()
        snap = _build_snapshot(n_docs=3, n_symbols=2)
        store.save_snapshot(snap)

        loaded = store.load_snapshot(snap.snapshot_id)
        assert loaded is not None
        assert len(loaded.documents) == 3
        assert len(loaded.symbols) == 6  # 3 docs * 2 symbols each

    @pytest.mark.edge
    def test_save_load_preserves_edges(self):
        """EDGE: edges survive full save/load cycle."""
        store = _make_store()
        snap = _build_snapshot(n_docs=2, n_symbols=1)
        store.save_snapshot(snap)

        loaded = store.load_snapshot(snap.snapshot_id)
        assert loaded is not None
        edge_types = {e.edge_type for e in loaded.edges}
        assert "contain" in edge_types

    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
        meta_val=st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.integers(min_value=0, max_value=1000),
            max_size=5,
        ),
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_metadata_json_roundtrip_via_save(self, repo, commit, branch, meta_val):
        """HAPPY: metadata passed to save_snapshot roundtrips through DB."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap, metadata=meta_val)

        record = store.get_snapshot_record(snap.snapshot_id)
        stored = json.loads(record["metadata_json"])
        assert stored == meta_val
