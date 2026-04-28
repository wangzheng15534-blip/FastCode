"""Stateful property-based tests for SnapshotStore lifecycle invariants.

Uses hypothesis RuleBasedStateMachine to model the full snapshot store
lifecycle: save -> load -> update -> save_scip_artifact_ref ->
get_scip_artifact_ref -> find_by_repo_commit.

Verifies that all operations maintain internal consistency and that
upsert semantics hold for repeated saves.
"""

from __future__ import annotations

import tempfile
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from fastcode.semantic_ir import IRSnapshot
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
_branch_st = st.sampled_from(["main", "dev", "feature", "release", "hotfix"])
_commit_st = st.text(alphabet="0123456789abcdef", min_size=7, max_size=40)
_metadata_key_st = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10
)
_metadata_val_st = st.one_of(
    st.integers(min_value=0, max_value=100),
    st.text(alphabet="abc", min_size=0, max_size=5),
    st.booleans(),
)


# --- Stateful model ---


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


# --- Helpers ---


def _build_connected_snapshot(
    repo_name: str,
    snap_id: str,
    branch: str,
    commit_id: str,
) -> IRSnapshot:
    """Build a minimal connected IRSnapshot for stateful testing."""
    from fastcode.semantic_ir import IRDocument, IREdge, IROccurrence, IRSymbol

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


# --- Registration ---

TestSnapshotStoreStateMachine = SnapshotStoreMachine.TestCase


# --- Additional upsert tests ---


class TestSnapshotStoreUpsert:
    @given(
        repo=_repo_name_st,
        commit=_commit_st,
        branch=_branch_st,
    )
    @settings(max_examples=15)
    @pytest.mark.basic
    def test_double_save_upsert_semantics_property(self, repo: str, commit: str, branch: str):
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
    def test_scip_artifact_ref_upsert_property(self, repo: str, commit: str, branch: str):
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
    def test_find_by_repo_commit_unknown_returns_none_property(self, repo: str, commit: str):
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
    @pytest.mark.basic
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
