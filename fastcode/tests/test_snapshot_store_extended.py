"""Tests for snapshot_store module (database-level, lifecycle, and stateful)."""

from __future__ import annotations

import json
import tempfile
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

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

_repo_name_st = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz",
    min_size=1,
    max_size=12,
)
_commit_st = st.text(alphabet="0123456789abcdef", min_size=7, max_size=40)
_branch_st = st.sampled_from(["main", "dev", "feature", "release", "hotfix"])
_unicode_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "àáâãäåæçèé"
    "一丁丂七丄丅"
    "АБВГД",
    min_size=1,
    max_size=50,
)
_metadata_key_st = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10
)
_metadata_val_st = st.one_of(
    st.integers(min_value=0, max_value=100),
    st.text(alphabet="abc", min_size=0, max_size=5),
    st.booleans(),
)


# --- Helpers ---


def _make_store() -> SnapshotStore:
    tmpdir = tempfile.mkdtemp(prefix="ss_ext_prop_")
    return SnapshotStore(tmpdir)


def _build_snapshot(
    repo: str = "test_repo",
    commit: str = "abc1234",
    branch: str = "main",
    n_docs: int = 1,
    n_symbols: int = 1,
) -> IRSnapshot:
    """Build a minimal connected snapshot."""
    docs = []
    syms = []
    occs = []
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
    return IRSnapshot(
        repo_name=repo,
        snapshot_id=f"snap:{repo}:{commit}",
        branch=branch,
        commit_id=commit,
        documents=docs,
        symbols=syms,
        occurrences=occs,
        edges=[],
        metadata={"source_modes": ["ast"]},
    )


def _build_connected_snapshot(
    repo_name: str,
    snap_id: str,
    branch: str,
    commit_id: str,
) -> IRSnapshot:
    """Build a minimal connected IRSnapshot for stateful testing."""
    doc_id = f"doc:{commit_id[:8]}"
    sym_id = f"sym:{commit_id[:8]}"
    docs = [
        IRDocument(
            doc_id=doc_id, path="src/main.py", language="python", source_set={"ast"}
        )
    ]
    syms = [
        IRSymbol(
            symbol_id=sym_id,
            external_symbol_id=None,
            path="src/main.py",
            display_name="main_fn",
            kind="function",
            language="python",
            source_priority=10,
            source_set={"ast"},
            start_line=1,
        )
    ]
    occs = [
        IROccurrence(
            occurrence_id=f"occ:{commit_id[:8]}",
            symbol_id=sym_id,
            doc_id=doc_id,
            role="definition",
            start_line=1,
            start_col=0,
            end_line=1,
            end_col=0,
            source="ast",
        )
    ]
    edges = [
        IREdge(
            edge_id=f"edge:{commit_id[:8]}",
            src_id=doc_id,
            dst_id=sym_id,
            edge_type="contain",
            source="ast",
            confidence="resolved",
        )
    ]
    return IRSnapshot(
        repo_name=repo_name,
        snapshot_id=snap_id,
        branch=branch,
        commit_id=commit_id,
        documents=docs,
        symbols=syms,
        occurrences=occs,
        edges=edges,
        metadata={"source_modes": ["ast"]},
    )


# ============================================================
# Database-level invariants
# ============================================================


class TestSchemaMigration:
    def test_init_db_idempotent_property(self):
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
    def test_init_db_after_save_preserves_data_property(
        self, repo: str, commit: str, branch: str
    ):
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
    def test_init_db_creates_required_tables_property(self):
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


class TestSnapshotsTableConstraints:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=15)
    def test_duplicate_snapshot_id_upserts_property(
        self, repo: str, commit: str, branch: str
    ):
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
    def test_upsert_preserves_latest_data_property(
        self, repo: str, commit: str, branch: str
    ):
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


class TestSnapshotRefsConstraints:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=15)
    def test_duplicate_ref_does_not_raise_property(
        self, repo: str, commit: str, branch: str
    ):
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
    def test_resolve_snapshot_for_ref_property(
        self, repo: str, commit: str, branch: str
    ):
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
    def test_resolve_snapshot_for_unknown_ref_returns_none_property(
        self, repo: str, branch: str
    ):
        """EDGE: resolving unknown branch returns None."""
        store = _make_store()
        result = store.resolve_snapshot_for_ref(repo, branch)
        assert result is None


class TestScipArtifactsTable:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
        indexer_name=st.sampled_from(["scip-python", "scip-java", "scip-go"]),
    )
    @settings(max_examples=15)
    def test_scip_artifact_upsert_property(
        self, repo: str, commit: str, branch: str, indexer_name: str
    ):
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
    def test_scip_artifact_fields_roundtrip_property(
        self, repo: str, commit: str, branch: str
    ):
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


class TestMultipleSnapshotsSameRepo:
    @given(
        repo=_repo_name_st,
        commits=st.lists(_commit_st, min_size=2, max_size=4, unique=True),
        branch=_branch_st,
    )
    @settings(max_examples=10)
    def test_multiple_snapshots_all_retrievable_property(
        self, repo: str, commits: list[str], branch: str
    ):
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
    def test_find_by_repo_commit_returns_latest_property(
        self, repo: str, commit: str, branch: str
    ):
        """HAPPY: find_by_repo_commit returns a record for the given repo+commit."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap)

        result = store.find_by_repo_commit(repo, commit)
        assert result is not None
        assert result["snapshot_id"] == snap.snapshot_id


class TestLargeMetadataRoundtrip:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=5)
    @pytest.mark.edge
    def test_large_metadata_roundtrip_property(
        self, repo: str, commit: str, branch: str
    ):
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
    def test_unicode_metadata_preserved_property(
        self, repo: str, commit: str, branch: str, unicode_key: Any, unicode_val: Any
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


class TestIRGraphsPickleRoundtrip:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
        n_nodes=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=10)
    def test_ir_graphs_pickle_roundtrip_property(
        self, repo: str, commit: str, branch: str, n_nodes: Any
    ):
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
    def test_load_ir_graphs_nonexistent_returns_none_property(self):
        """EDGE: loading graphs for nonexistent snapshot returns None."""
        store = _make_store()
        result = store.load_ir_graphs("snap:ghost:0000000")
        assert result is None

    @pytest.mark.edge
    def test_load_ir_graphs_no_saved_graphs_returns_none_property(self):
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
    def test_ir_graphs_overwrite_property(self, repo: str, commit: str, branch: str):
        """HAPPY: saving graphs twice overwrites previous version."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap)

        store.save_ir_graphs(snap.snapshot_id, {"version": 1})
        store.save_ir_graphs(snap.snapshot_id, {"version": 2})

        loaded = store.load_ir_graphs(snap.snapshot_id)
        assert loaded == {"version": 2}


class TestSnapshotWithNoneFields:
    @pytest.mark.edge
    def test_snapshot_all_optional_fields_none_property(self):
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
    def test_snapshot_empty_metadata_property(self):
        """EDGE: snapshot with empty metadata dict works."""
        store = _make_store()
        snap = _build_snapshot()
        store.save_snapshot(snap, metadata={})

        record = store.get_snapshot_record(snap.snapshot_id)
        assert record is not None
        stored_meta = json.loads(record.get("metadata_json", "{}"))
        assert stored_meta == {}


class TestFindLatestBehavior:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    def test_find_by_repo_commit_returns_matching_property(
        self, repo: str, commit: str, branch: str
    ):
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
    def test_find_by_repo_commit_distinguishes_commits_property(
        self, repo: str, commit1: Any, commit2: Any, branch: str
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
    def test_resolve_snapshot_for_ref_distinguishes_branches_property(
        self, repo: str, commit1: Any, commit2: Any, branch1: Any, branch2: Any
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


class TestArtifactKeyGeneration:
    @given(
        snap_id=st.builds(lambda x: f"snap:{x}", identifier),
    )
    @settings(max_examples=20)
    def test_artifact_key_deterministic_property(self, snap_id: str):
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
    def test_different_snapshot_ids_different_keys_property(
        self, snap_id1: Any, snap_id2: Any
    ):
        """HAPPY: different snapshot_ids produce different artifact keys."""
        assume(snap_id1 != snap_id2)
        store = _make_store()
        key1 = store.artifact_key_for_snapshot(snap_id1)
        key2 = store.artifact_key_for_snapshot(snap_id2)
        assert key1 != key2


class TestSnapshotDir:
    @given(
        snap_id=st.builds(lambda x: f"snap:{x}", identifier),
    )
    @settings(max_examples=15)
    def test_snapshot_dir_exists_after_call_property(self, snap_id: str):
        """HAPPY: snapshot_dir creates the directory if needed."""
        store = _make_store()
        d = store.snapshot_dir(snap_id)
        import os

        assert os.path.isdir(d)

    @given(
        snap_id=st.builds(lambda x: f"snap:{x}", identifier),
    )
    @settings(max_examples=15)
    def test_snapshot_dir_idempotent_property(self, snap_id: str):
        """HAPPY: calling snapshot_dir twice returns same path."""
        store = _make_store()
        d1 = store.snapshot_dir(snap_id)
        d2 = store.snapshot_dir(snap_id)
        assert d1 == d2


class TestFindbyArtifactKey:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    def test_find_by_artifact_key_after_save_property(
        self, repo: str, commit: str, branch: str
    ):
        """HAPPY: find_by_artifact_key retrieves saved snapshot."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        result = store.save_snapshot(snap)

        found = store.find_by_artifact_key(result["artifact_key"])
        assert found is not None
        assert found["snapshot_id"] == snap.snapshot_id

    @pytest.mark.edge
    def test_find_by_artifact_key_unknown_returns_none_property(self):
        """EDGE: querying unknown artifact key returns None."""
        store = _make_store()
        result = store.find_by_artifact_key("snap_nonexistent")
        assert result is None


class TestSnapshotRecordFields:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    def test_snapshot_record_has_all_fields_property(
        self, repo: str, commit: str, branch: str
    ):
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
    def test_ir_path_points_to_real_file_property(
        self, repo: str, commit: str, branch: str
    ):
        """HAPPY: ir_path in snapshot record points to an existing file."""
        import os

        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap)

        record = store.get_snapshot_record(snap.snapshot_id)
        assert os.path.isfile(record["ir_path"])


class TestEdgeCases:
    @pytest.mark.edge
    def test_save_load_snapshot_preserves_documents_property(self):
        """EDGE: documents survive full save/load cycle."""
        store = _make_store()
        snap = _build_snapshot(n_docs=3, n_symbols=2)
        store.save_snapshot(snap)

        loaded = store.load_snapshot(snap.snapshot_id)
        assert loaded is not None
        assert len(loaded.documents) == 3
        assert len(loaded.symbols) == 6  # 3 docs * 2 symbols each

    @pytest.mark.edge
    def test_save_load_preserves_edges_property(self):
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
    def test_metadata_json_roundtrip_via_save_property(
        self, repo: str, commit: str, branch: str, meta_val: Any
    ):
        """HAPPY: metadata passed to save_snapshot roundtrips through DB."""
        store = _make_store()
        snap = _build_snapshot(repo, commit, branch)
        store.save_snapshot(snap, metadata=meta_val)

        record = store.get_snapshot_record(snap.snapshot_id)
        stored = json.loads(record["metadata_json"])
        assert stored == meta_val


# ============================================================
# Lifecycle state machine tests
# ============================================================


class TestFullLifecycle:
    def test_save_load_update_load_cycle_property(self):
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

    def test_save_load_snapshot_ir_preserved_property(self):
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

    def test_load_nonexistent_returns_none_property(self):
        """HAPPY: loading a nonexistent snapshot returns None."""
        store = _make_store()
        assert store.load_snapshot("snap:ghost:000") is None

    @pytest.mark.edge
    def test_update_metadata_nonexistent_does_not_raise_property(self):
        """EDGE: updating metadata for nonexistent snapshot does not raise."""
        store = _make_store()
        store.update_snapshot_metadata("snap:ghost:000", {"x": 1})
        # Should silently update zero rows
        assert store.get_snapshot_record("snap:ghost:000") is None


class TestSnapshotIsolation:
    def test_two_snapshots_different_repos_no_interference_property(self):
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
    def test_metadata_update_does_not_affect_other_snapshot_property(self):
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


class TestConcurrentSaveSameSnapshotId:
    def test_second_save_wins_upsert_property(self):
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
    def test_upsert_replaces_metadata_property(self):
        """EDGE: upsert replaces the metadata with the latest value."""
        store = _make_store()
        snap = _build_snapshot("repo", "c001", "main")
        store.save_snapshot(snap, metadata={"v": 1})
        store.save_snapshot(snap, metadata={"v": 2})

        record = store.get_snapshot_record(snap.snapshot_id)
        stored = json.loads(record["metadata_json"])
        assert stored["v"] == 2


class TestArtifactRefLifecycle:
    def test_save_get_artifact_ref_property(self):
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

    def test_upsert_artifact_ref_updates_indexer_property(self):
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

    def test_list_artifact_refs_returns_ordered_lineage_property(self):
        """HAPPY: multi-artifact lineage remains ordered and primary-stable."""
        store = _make_store()
        snap = _build_snapshot("repo_art3", "c003", "main")
        store.save_snapshot(snap)

        store.save_scip_artifact_refs(
            snap.snapshot_id,
            artifacts=[
                {
                    "indexer_name": "scip-python",
                    "artifact_path": "/data/a.scip",
                    "checksum": "a",
                    "language": "python",
                },
                {
                    "indexer_name": "scip-go",
                    "artifact_path": "/data/b.scip",
                    "checksum": "b",
                    "language": "go",
                },
            ],
        )

        refs = store.list_scip_artifact_refs(snap.snapshot_id)
        assert [ref["role"] for ref in refs] == ["primary", "secondary"]
        assert refs[0]["artifact_path"] == "/data/a.scip"
        assert refs[1]["metadata"]["language"] == "go"
        assert (
            store.get_scip_artifact_ref(snap.snapshot_id)["artifact_path"]
            == "/data/a.scip"
        )

    @pytest.mark.edge
    def test_get_artifact_ref_nonexistent_returns_none_property(self):
        """EDGE: getting artifact ref for unknown snapshot returns None."""
        store = _make_store()
        assert store.get_scip_artifact_ref("snap:ghost:000") is None


class TestIRGraphsLifecycle:
    def test_save_load_graphs_property(self):
        """HAPPY: save and load IR graphs via pickle."""
        store = _make_store()
        snap = _build_snapshot("repo_g", "c001", "main")
        store.save_snapshot(snap)

        graphs = {"dependency": {"nodes": [1, 2, 3]}, "call": {"edges": [(1, 2)]}}
        path = store.save_ir_graphs(snap.snapshot_id, graphs)
        assert path.endswith(".pkl")

        loaded = store.load_ir_graphs(snap.snapshot_id)
        assert loaded == graphs

    def test_overwrite_graphs_property(self):
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
    def test_load_graphs_nonexistent_returns_none_property(self):
        """EDGE: loading graphs for nonexistent snapshot returns None."""
        store = _make_store()
        assert store.load_ir_graphs("snap:ghost:000") is None

    @pytest.mark.edge
    def test_load_graphs_without_saved_returns_none_property(self):
        """EDGE: loading graphs when snapshot exists but no graphs saved returns None."""
        store = _make_store()
        snap = _build_snapshot()
        store.save_snapshot(snap)
        assert store.load_ir_graphs(snap.snapshot_id) is None


class TestFullWorkflow:
    def test_save_snapshot_scip_ref_graphs_query_all_property(self):
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
    def test_workflow_find_by_repo_commit_unknown_property(self):
        """EDGE: find_by_repo_commit returns None for unknown repo/commit."""
        store = _make_store()
        assert store.find_by_repo_commit("no_repo", "no_commit") is None


class TestMultipleSnapshotsSameRepoDifferentCommits:
    @given(
        repo=_repo_name_st,
        commits=st.lists(_commit_st, min_size=2, max_size=5, unique=True),
        branch=_branch_st,
    )
    @settings(max_examples=10, deadline=None)
    def test_all_snapshots_findable_individually_property(
        self, repo: str, commits: list[str], branch: str
    ):
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
    def test_find_by_repo_commit_distinguishes_property(
        self, repo: str, commit1: Any, commit2: Any, branch: str
    ):
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
    def test_snapshots_do_not_leak_metadata_property(self):
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


class TestMetadataVariants:
    def test_empty_dict_metadata_roundtrip_property(self):
        """HAPPY: empty dict metadata roundtrips correctly."""
        store = _make_store()
        snap = _build_snapshot("repo_meta", "c_empty", "main")
        store.save_snapshot(snap, metadata={})

        record = store.get_snapshot_record(snap.snapshot_id)
        stored = json.loads(record["metadata_json"])
        assert stored == {}

    def test_nested_dict_metadata_roundtrip_property(self):
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

    def test_large_dict_metadata_roundtrip_property(self):
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
    def test_none_metadata_treated_as_empty_property(self):
        """EDGE: passing None as metadata stores empty dict."""
        store = _make_store()
        snap = _build_snapshot("repo_meta", "c_none", "main")
        store.save_snapshot(snap, metadata=None)

        record = store.get_snapshot_record(snap.snapshot_id)
        stored = json.loads(record["metadata_json"])
        assert stored == {}

    @pytest.mark.edge
    def test_update_metadata_to_empty_dict_property(self):
        """EDGE: updating metadata to empty dict replaces previous value."""
        store = _make_store()
        snap = _build_snapshot("repo_meta", "c_upd", "main")
        store.save_snapshot(snap, metadata={"rich": True, "data": [1, 2, 3]})

        store.update_snapshot_metadata(snap.snapshot_id, {})

        record = store.get_snapshot_record(snap.snapshot_id)
        stored = json.loads(record["metadata_json"])
        assert stored == {}


# ============================================================
# Stateful property-based tests (hypothesis RuleBasedStateMachine)
# ============================================================


class SnapshotStoreMachine(RuleBasedStateMachine):
    """Models the SnapshotStore lifecycle as a state machine."""

    def __init__(self) -> None:
        super().__init__()
        self.tmpdir = tempfile.mkdtemp(prefix="ss_stateful_")
        self.store = SnapshotStore(self.tmpdir)
        self.saved_snapshots: dict[str, IRSnapshot] = {}
        self.saved_refs: dict[str, dict[str, Any]] = {}
        self.metadata_state: dict[str, dict[str, Any]] = {}

    @initialize()
    def init_store(self):
        """Initialize with an empty store."""
        assert self.store is not None

    @rule(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    def save_snapshot(self, repo: str, commit: str, branch: str):
        """Save a connected snapshot to the store."""
        snap_id = f"snap:{repo}:{commit}"
        snapshot = _build_connected_snapshot(repo, snap_id, branch, commit)
        result = self.store.save_snapshot(snapshot)
        assert result["snapshot_id"] == snap_id
        self.saved_snapshots[snap_id] = snapshot
        self.metadata_state[snap_id] = {}

    @rule(data=st.data())
    def load_snapshot(self, data: st.DataObject):
        """Load a previously saved snapshot and verify identity."""
        if not self.saved_snapshots:
            return
        snap_id = data.draw(st.sampled_from(sorted(self.saved_snapshots.keys())))
        loaded = self.store.load_snapshot(snap_id)
        assert loaded is not None
        original = self.saved_snapshots[snap_id]
        assert loaded.snapshot_id == original.snapshot_id
        assert loaded.repo_name == original.repo_name
        assert loaded.commit_id == original.commit_id

    @rule(data=st.data())
    def save_scip_artifact_ref(self, data: st.DataObject):
        """Save a SCIP artifact reference for a stored snapshot."""
        if not self.saved_snapshots:
            return
        snap_id = data.draw(st.sampled_from(sorted(self.saved_snapshots.keys())))
        ref = self.store.save_scip_artifact_ref(
            snap_id,
            indexer_name="scip-python",
            indexer_version="1.0.0",
            artifact_path=f"/tmp/{snap_id}.scip",
            checksum="abc123",
        )
        assert ref["snapshot_id"] == snap_id
        self.saved_refs[snap_id] = ref

    @rule(data=st.data())
    def get_scip_artifact_ref(self, data: st.DataObject):
        """Retrieve a SCIP artifact ref and verify it matches last save."""
        if not self.saved_refs:
            return
        snap_id = data.draw(st.sampled_from(sorted(self.saved_refs.keys())))
        ref = self.store.get_scip_artifact_ref(snap_id)
        assert ref is not None
        assert ref["snapshot_id"] == snap_id
        assert ref["indexer_name"] == self.saved_refs[snap_id]["indexer_name"]

    @rule(
        data=st.data(),
        key=_metadata_key_st,
        value=_metadata_val_st,
    )
    def update_snapshot_metadata(self, data: st.DataObject, key: str, value: Any):
        """Update snapshot metadata and track the change."""
        if not self.saved_snapshots:
            return
        snap_id = data.draw(st.sampled_from(sorted(self.saved_snapshots.keys())))
        meta = dict(self.metadata_state.get(snap_id, {}))
        meta[key] = value
        self.store.update_snapshot_metadata(snap_id, meta)
        self.metadata_state[snap_id] = meta

    @rule(data=st.data())
    def find_by_repo_commit(self, data: st.DataObject):
        """Query by repo+commit and verify result exists."""
        if not self.saved_snapshots:
            return
        snap_id = data.draw(st.sampled_from(sorted(self.saved_snapshots.keys())))
        snap = self.saved_snapshots[snap_id]
        result = self.store.find_by_repo_commit(snap.repo_name, snap.commit_id)
        assert result is not None
        assert result["repo_name"] == snap.repo_name
        assert result["commit_id"] == snap.commit_id

    @invariant()
    def load_never_raises_on_known_id(self):
        """INVARIANT: loading any known snapshot_id never raises."""
        for snap_id in self.saved_snapshots:
            try:
                result = self.store.load_snapshot(snap_id)
                assert result is not None
                assert result.snapshot_id == snap_id
            except Exception:
                raise AssertionError(f"load_snapshot({snap_id}) raised unexpectedly")

    @invariant()
    def get_scip_ref_matches_last_save(self):
        """INVARIANT: get_scip_artifact_ref matches last saved ref."""
        for snap_id, saved_ref in self.saved_refs.items():
            ref = self.store.get_scip_artifact_ref(snap_id)
            assert ref is not None, f"Missing ref for {snap_id}"
            assert ref["snapshot_id"] == saved_ref["snapshot_id"]

    @invariant()
    def find_by_repo_commit_returns_record(self):
        """INVARIANT: find_by_repo_commit always returns a record for saved data."""
        for _snap_id, snap in self.saved_snapshots.items():
            result = self.store.find_by_repo_commit(snap.repo_name, snap.commit_id)
            assert result is not None, (
                f"No record for {snap.repo_name}:{snap.commit_id}"
            )

    @invariant()
    def snapshot_record_exists(self):
        """INVARIANT: get_snapshot_record returns a row for every saved snapshot."""
        for snap_id in self.saved_snapshots:
            record = self.store.get_snapshot_record(snap_id)
            assert record is not None, f"Missing record for {snap_id}"
            assert record["snapshot_id"] == snap_id


# --- Registration ---

TestSnapshotStoreStateMachine = SnapshotStoreMachine.TestCase
TestSnapshotStoreStateMachine.settings = settings(
    max_examples=20,
    stateful_step_count=20,
    deadline=None,
)
pytest.mark.timeout(120)(TestSnapshotStoreStateMachine.runTest)


# --- Additional upsert tests ---


class TestSnapshotStoreUpsert:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=15)
    def test_double_save_upsert_semantics_property(
        self, repo: str, commit: str, branch: str
    ):
        """HAPPY: saving the same snapshot twice uses upsert (last wins)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            snap_id = f"snap:{repo}:{commit}"
            snap1 = _build_connected_snapshot(repo, snap_id, branch, commit)

            store.save_snapshot(snap1)
            # Mutate a field and save again
            snap2 = _build_connected_snapshot(repo, snap_id, branch, commit)
            store.save_snapshot(snap2)

            loaded = store.load_snapshot(snap_id)
            assert loaded is not None
            assert loaded.snapshot_id == snap_id

    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_scip_artifact_ref_upsert_property(
        self, repo: str, commit: str, branch: str
    ):
        """EDGE: saving SCIP artifact ref twice uses upsert (last wins)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            snap_id = f"snap:{repo}:{commit}"
            snap = _build_connected_snapshot(repo, snap_id, branch, commit)
            store.save_snapshot(snap)

            store.save_scip_artifact_ref(
                snap_id, indexer_name="scip-python", checksum="aaa"
            )
            ref2 = store.save_scip_artifact_ref(
                snap_id, indexer_name="scip-go", checksum="bbb"
            )
            assert ref2["indexer_name"] == "scip-go"

            loaded_ref = store.get_scip_artifact_ref(snap_id)
            assert loaded_ref["indexer_name"] == "scip-go"
            assert loaded_ref["checksum"] == "bbb"

    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_load_unknown_snapshot_returns_none_property(
        self, repo: str, commit: str, branch: str
    ):
        """EDGE: loading a never-saved snapshot_id returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            result = store.load_snapshot("snap:nonexistent:0000000")
            assert result is None

    @given(
        repo=_repo_name_st,
        commit=_commit_st,
    )
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_find_by_repo_commit_unknown_returns_none_property(
        self, repo: str, commit: str
    ):
        """EDGE: querying unknown repo+commit returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            result = store.find_by_repo_commit(repo, commit)
            assert result is None

    @given(
        snap_id=st.builds(lambda x: f"snap:{x}", identifier),
    )
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_get_scip_artifact_ref_unknown_returns_none_property(self, snap_id: str):
        """EDGE: getting SCIP artifact ref for unknown snapshot returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            result = store.get_scip_artifact_ref(snap_id)
            assert result is None

    @pytest.mark.edge
    def test_update_metadata_on_nonexistent_no_error_property(self):
        """EDGE: updating metadata on nonexistent snapshot_id does not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            store.update_snapshot_metadata("snap:ghost:000", {"key": "val"})
            # Should not raise

    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
        key=_metadata_key_st,
        value=_metadata_val_st,
    )
    @settings(max_examples=10)
    def test_metadata_update_persists_property(
        self, repo: str, commit: str, branch: str, key: str, value: Any
    ):
        """HAPPY: metadata update roundtrips through get_snapshot_record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            snap_id = f"snap:{repo}:{commit}"
            snap = _build_connected_snapshot(repo, snap_id, branch, commit)
            store.save_snapshot(snap)

            meta = {key: value}
            store.update_snapshot_metadata(snap_id, meta)

            record = store.get_snapshot_record(snap_id)
            assert record is not None
            import json

            stored_meta = json.loads(record.get("metadata_json", "{}"))
            assert stored_meta.get(key) == value
