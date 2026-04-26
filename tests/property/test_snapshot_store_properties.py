"""Property-based tests for snapshot_store.SnapshotStore invariants.

Covers: save/load roundtrip, query methods, SCIP artifact refs,
locks (SQLite stubs), redo tasks (SQLite stubs), IR graphs pickle,
staging, and relational fact no-op paths on SQLite backend.
"""

from __future__ import annotations

import tempfile
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from fastcode.semantic_ir import (
    IRAttachment,
    IRDocument,
    IREdge,
    IROccurrence,
    IRSnapshot,
    IRSymbol,
)
from fastcode.snapshot_store import SnapshotStore

# --- Strategies (mirrored from tests/conftest.py) ---

identifier = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=40,
)

file_path_st = st.tuples(identifier, identifier).map(lambda t: f"{t[0]}/{t[1]}.py")

role_st = st.sampled_from(["definition", "reference", "import", "implementation"])

edge_type_st = st.sampled_from(
    ["dependency", "call", "inheritance", "reference", "contain"]
)

source_st = st.sampled_from(["ast", "fc_structure", "scip"])
attachment_source_st = st.sampled_from(
    ["fc_structure", "fc_embedding", "llm_annotation"]
)

kind_st = st.sampled_from(
    [
        "function",
        "method",
        "class",
        "variable",
        "module",
        "interface",
        "enum",
        "constant",
    ]
)

language_st = st.sampled_from(
    ["python", "javascript", "typescript", "go", "java", "rust", "c", "cpp"]
)

line_number_st = st.integers(min_value=1, max_value=10000)

ir_document_st = st.builds(
    IRDocument,
    doc_id=st.builds(lambda x: f"doc:{x}", identifier),
    path=file_path_st,
    language=language_st,
    blob_oid=st.none() | st.text(alphabet="0123456789abcdef", min_size=40, max_size=40),
    content_hash=st.none()
    | st.text(alphabet="0123456789abcdef", min_size=40, max_size=40),
    source_set=st.sets(source_st, max_size=2),
)

ir_symbol_st = st.builds(
    IRSymbol,
    symbol_id=st.builds(lambda x: f"sym:{x}", identifier),
    external_symbol_id=st.none() | identifier,
    path=file_path_st,
    display_name=identifier,
    kind=kind_st,
    language=language_st,
    qualified_name=st.none() | st.builds(lambda x: f"pkg.{x}", identifier),
    signature=st.none() | st.just("def foo(x: int) -> str"),
    start_line=st.none() | line_number_st,
    start_col=st.none() | st.integers(min_value=0, max_value=120),
    end_line=st.none() | line_number_st,
    end_col=st.none() | st.integers(min_value=0, max_value=120),
    source_priority=st.integers(min_value=0, max_value=200),
    source_set=st.sets(source_st, max_size=2),
    metadata=st.dictionaries(st.text(min_size=1, max_size=10), st.integers()),
)

ir_occurrence_st = st.builds(
    IROccurrence,
    occurrence_id=st.builds(lambda x: f"occ:{x}", identifier),
    symbol_id=st.builds(lambda x: f"sym:{x}", identifier),
    doc_id=st.builds(lambda x: f"doc:{x}", identifier),
    role=role_st,
    start_line=line_number_st,
    start_col=st.integers(min_value=0, max_value=120),
    end_line=line_number_st,
    end_col=st.integers(min_value=0, max_value=120),
    source=source_st,
    metadata=st.dictionaries(st.text(min_size=1, max_size=10), st.integers()),
)

ir_edge_st = st.builds(
    IREdge,
    edge_id=st.builds(lambda x: f"edge:{x}", identifier),
    src_id=identifier,
    dst_id=identifier,
    edge_type=edge_type_st,
    source=source_st,
    confidence=st.sampled_from(["precise", "heuristic", "resolved", ""]),
    doc_id=st.none() | st.builds(lambda x: f"doc:{x}", identifier),
    metadata=st.dictionaries(st.text(min_size=1, max_size=10), st.integers()),
)

ir_attachment_st = st.builds(
    IRAttachment,
    attachment_id=st.builds(lambda x: f"att:{x}", identifier),
    target_id=st.builds(lambda x: f"sym:{x}", identifier),
    target_type=st.sampled_from(["document", "symbol", "snapshot"]),
    attachment_type=st.sampled_from(["embedding", "summary", "semantic_note"]),
    source=attachment_source_st,
    confidence=st.sampled_from(["derived", "precise", "heuristic"]),
    payload=st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(
            st.integers(),
            st.text(min_size=0, max_size=20),
            st.lists(st.floats(allow_nan=False, allow_infinity=False), max_size=4),
        ),
        max_size=3,
    ),
    metadata=st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(st.integers(), st.text(min_size=0, max_size=20)),
        max_size=3,
    ),
)


def snapshot_st(
    max_docs: int = 3,
    max_syms: int = 5,
    max_occs: int = 8,
    max_edges: int = 4,
) -> st.SearchStrategy[IRSnapshot]:
    """Build an IRSnapshot strategy with controlled size."""
    return st.builds(
        IRSnapshot,
        repo_name=identifier,
        snapshot_id=st.builds(lambda x: f"snap:{x}", identifier),
        branch=st.none() | st.just("main"),
        commit_id=st.none()
        | st.text(alphabet="0123456789abcdef", min_size=7, max_size=40),
        tree_id=st.none() | identifier,
        documents=st.lists(ir_document_st, max_size=max_docs),
        symbols=st.lists(ir_symbol_st, max_size=max_syms),
        occurrences=st.lists(ir_occurrence_st, max_size=max_occs),
        edges=st.lists(ir_edge_st, max_size=max_edges),
        attachments=st.lists(ir_attachment_st, max_size=4),
        metadata=st.dictionaries(
            st.sampled_from(["source_modes", "version", "tool"]),
            st.one_of(
                st.just(["ast"]), st.just(["scip"]), st.just(1), st.just("fastcode")
            ),
        ),
    )


@st.composite
def connected_snapshot_st(
    draw: st.DataObject,
    n_docs: int | None = None,
    n_symbols_per_doc: int | None = None,
):
    """Generate IRSnapshot where all references are valid."""
    nd = n_docs or draw(st.integers(min_value=1, max_value=4))
    ns = n_symbols_per_doc or draw(st.integers(min_value=1, max_value=4))

    repo = draw(identifier)
    snap_id = f"snap:{draw(identifier)}"
    docs = []
    symbols = []
    occurrences = []
    edges = []

    for i in range(nd):
        path = f"dir{i % 3}/file{i}.py"
        doc_id = f"doc:{draw(identifier)}"
        docs.append(
            IRDocument(
                doc_id=doc_id,
                path=path,
                language=draw(language_st),
                source_set={"fc_structure"},
            )
        )

        for j in range(ns):
            sym_id = f"sym:{draw(identifier)}"
            symbols.append(
                IRSymbol(
                    symbol_id=sym_id,
                    external_symbol_id=None,
                    path=path,
                    display_name=f"func_{j}",
                    kind=draw(kind_st),
                    language=docs[-1].language,
                    source_priority=50,
                    source_set={"fc_structure"},
                    start_line=draw(st.integers(min_value=1, max_value=500)),
                )
            )
            occurrences.append(
                IROccurrence(
                    occurrence_id=f"occ:{draw(identifier)}",
                    symbol_id=sym_id,
                    doc_id=doc_id,
                    role="definition",
                    start_line=symbols[-1].start_line or 1,
                    start_col=0,
                    end_line=symbols[-1].start_line or 1,
                    end_col=0,
                    source="fc_structure",
                )
            )
            edges.append(
                IREdge(
                    edge_id=f"edge:{draw(identifier)}",
                    src_id=doc_id,
                    dst_id=sym_id,
                    edge_type="contain",
                    source="fc_structure",
                    confidence="resolved",
                )
            )

    return IRSnapshot(
        repo_name=repo,
        snapshot_id=snap_id,
        branch="main",
        commit_id=draw(st.text(alphabet="0123456789abcdef", min_size=7, max_size=40)),
        documents=docs,
        symbols=symbols,
        occurrences=occurrences,
        edges=edges,
        attachments=[
            IRAttachment(
                attachment_id=f"att:{draw(identifier)}",
                target_id=snap_id,
                target_type="snapshot",
                attachment_type="summary",
                source="fc_structure",
                confidence="derived",
                payload={"text": "snapshot summary"},
                metadata={},
            )
        ],
        metadata={"source_modes": ["fc_structure"]},
    )


# --- Helpers ---


def _make_store() -> SnapshotStore:
    tmpdir = tempfile.mkdtemp(prefix="snap_prop_")
    return SnapshotStore(tmpdir)


metadata_st = st.dictionaries(
    st.text(min_size=1, max_size=10),
    st.one_of(st.integers(), st.text(min_size=0, max_size=20), st.booleans()),
    max_size=4,
)


# --- TestSnapshotSaveLoadProperties ---


@pytest.mark.property
class TestSnapshotSaveLoadProperties:
    @given(snap=snapshot_st())
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_save_load_roundtrip(self, snap: IRSnapshot):
        """HAPPY: save_snapshot then load_snapshot preserves IRSnapshot fields."""
        store = _make_store()
        store.save_snapshot(snap, metadata={"test": True})
        loaded = store.load_snapshot(snap.snapshot_id)
        assert loaded is not None
        assert loaded.repo_name == snap.repo_name
        assert loaded.snapshot_id == snap.snapshot_id
        assert loaded.branch == snap.branch
        assert loaded.commit_id == snap.commit_id
        assert loaded.tree_id == snap.tree_id
        assert len(loaded.documents) == len(snap.documents)
        assert len(loaded.symbols) == len(snap.symbols)
        assert len(loaded.occurrences) == len(snap.occurrences)
        assert len(loaded.edges) == len(snap.edges)
        assert len(loaded.attachments) == len(snap.attachments)
        assert loaded.metadata == snap.metadata

    @given(snap=snapshot_st())
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_save_returns_artifact_key(self, snap: IRSnapshot):
        """HAPPY: save_snapshot returns dict with artifact_key and snapshot_id."""
        store = _make_store()
        result = store.save_snapshot(snap)
        assert "artifact_key" in result
        assert "snapshot_id" in result
        assert result["snapshot_id"] == snap.snapshot_id
        assert result["artifact_key"].startswith("snap_")

    @given(snap=snapshot_st())
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_save_idempotent_upsert(self, snap: IRSnapshot):
        """HAPPY: saving same snapshot twice does not raise (ON CONFLICT DO UPDATE)."""
        store = _make_store()
        r1 = store.save_snapshot(snap)
        r2 = store.save_snapshot(snap)
        assert r1["snapshot_id"] == r2["snapshot_id"]
        assert r1["artifact_key"] == r2["artifact_key"]

    @given(snap=snapshot_st())
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_get_snapshot_record_after_save(self, snap: IRSnapshot):
        """HAPPY: get_snapshot_record returns dict with all DB columns."""
        store = _make_store()
        store.save_snapshot(snap)
        record = store.get_snapshot_record(snap.snapshot_id)
        assert record is not None
        assert record["snapshot_id"] == snap.snapshot_id
        assert record["repo_name"] == snap.repo_name
        assert record["artifact_key"] is not None
        assert record["ir_path"] is not None
        assert record["created_at"] is not None

    @given(snap=snapshot_st())
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_load_snapshot_roundtrip_children(self, snap: IRSnapshot):
        """HAPPY: load_snapshot preserves nested document/symbol/edge fields."""
        store = _make_store()
        store.save_snapshot(snap)
        loaded = store.load_snapshot(snap.snapshot_id)
        assert loaded is not None
        for orig, rest in zip(snap.documents, loaded.documents, strict=True):
            assert orig.doc_id == rest.doc_id
            assert orig.path == rest.path
            assert orig.language == rest.language
            assert orig.source_set == rest.source_set
        for orig, rest in zip(snap.symbols, loaded.symbols, strict=True):
            assert orig.symbol_id == rest.symbol_id
            assert orig.display_name == rest.display_name
            assert orig.kind == rest.kind
        for orig, rest in zip(snap.attachments, loaded.attachments, strict=True):
            assert orig.attachment_id == rest.attachment_id
            assert orig.payload == rest.payload

    @given(snapshot_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_load_snapshot_missing_returns_none(self, snapshot_id: str):
        """EDGE: load_snapshot for non-existent ID returns None."""
        store = _make_store()
        assert store.load_snapshot(snapshot_id) is None

    @given(snapshot_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_get_snapshot_record_missing_returns_none(self, snapshot_id: str):
        """EDGE: get_snapshot_record for non-existent ID returns None."""
        store = _make_store()
        assert store.get_snapshot_record(snapshot_id) is None

    @given(snap=connected_snapshot_st())
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_resolve_snapshot_for_ref(self, snap: IRSnapshot):
        """HAPPY: resolve_snapshot_for_ref finds snapshot by repo+branch after save."""
        assume(snap.branch is not None)
        store = _make_store()
        store.save_snapshot(snap)
        result = store.resolve_snapshot_for_ref(snap.repo_name, snap.branch)
        assert result is not None
        assert result["snapshot_id"] == snap.snapshot_id
        assert result["repo_name"] == snap.repo_name
        assert result["branch"] == snap.branch

    @given(repo=identifier, branch=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_resolve_snapshot_for_ref_missing_returns_none(
        self, repo: str, branch: str
    ):
        """EDGE: resolve_snapshot_for_ref returns None for unknown ref."""
        store = _make_store()
        assert store.resolve_snapshot_for_ref(repo, branch) is None

    @given(snap=snapshot_st(), metadata=metadata_st)
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_save_with_metadata(self, snap: IRSnapshot, metadata: dict):
        """HAPPY: save_snapshot stores metadata_json and get_snapshot_record returns it."""
        store = _make_store()
        store.save_snapshot(snap, metadata=metadata)
        record = store.get_snapshot_record(snap.snapshot_id)
        assert record is not None
        assert record["metadata_json"] is not None


# --- TestSnapshotStoreQueries ---


@pytest.mark.property
class TestSnapshotStoreQueries:
    @given(snap=snapshot_st())
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_find_by_repo_commit(self, snap: IRSnapshot):
        """HAPPY: find_by_repo_commit returns record after save (requires commit_id)."""
        assume(snap.commit_id is not None)
        store = _make_store()
        store.save_snapshot(snap)
        result = store.find_by_repo_commit(snap.repo_name, snap.commit_id)
        assert result is not None
        assert result["snapshot_id"] == snap.snapshot_id
        assert result["repo_name"] == snap.repo_name

    @given(repo=identifier, commit=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_find_by_repo_commit_missing_returns_none(self, repo: str, commit: str):
        """EDGE: find_by_repo_commit returns None for unknown repo/commit."""
        store = _make_store()
        assert store.find_by_repo_commit(repo, commit) is None

    @given(snap=snapshot_st())
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_find_by_artifact_key(self, snap: IRSnapshot):
        """HAPPY: find_by_artifact_key returns record after save."""
        store = _make_store()
        result = store.save_snapshot(snap)
        found = store.find_by_artifact_key(result["artifact_key"])
        assert found is not None
        assert found["snapshot_id"] == snap.snapshot_id

    @given(key=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_find_by_artifact_key_missing_returns_none(self, key: str):
        """EDGE: find_by_artifact_key returns None for unknown key."""
        store = _make_store()
        assert store.find_by_artifact_key(key) is None

    @given(
        snap1=connected_snapshot_st(n_docs=1, n_symbols_per_doc=1),
        snap2=connected_snapshot_st(n_docs=1, n_symbols_per_doc=1),
    )
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_different_repos_independent(self, snap1: IRSnapshot, snap2: IRSnapshot):
        """EDGE: snapshots from different repos queried independently."""
        from hypothesis import assume

        assume(snap1.snapshot_id != snap2.snapshot_id)
        # Force different repo names to test isolation
        snap2 = IRSnapshot(
            repo_name=snap2.repo_name
            if snap2.repo_name != snap1.repo_name
            else snap2.repo_name + "_b",
            snapshot_id=snap2.snapshot_id,
            branch=snap2.branch,
            commit_id=snap2.commit_id,
            tree_id=snap2.tree_id,
            documents=snap2.documents,
            symbols=snap2.symbols,
            occurrences=snap2.occurrences,
            edges=snap2.edges,
            metadata=snap2.metadata,
        )
        store = _make_store()
        store.save_snapshot(snap1)
        store.save_snapshot(snap2)
        r1 = store.find_by_repo_commit(snap1.repo_name, snap1.commit_id)
        r2 = store.find_by_repo_commit(snap2.repo_name, snap2.commit_id)
        assert r1 is not None
        assert r2 is not None
        assert r1["snapshot_id"] == snap1.snapshot_id
        assert r2["snapshot_id"] == snap2.snapshot_id

    @given(
        snap=snapshot_st(),
        metadata1=metadata_st,
        metadata2=metadata_st,
    )
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_update_snapshot_metadata(
        self, snap: IRSnapshot, metadata1: dict, metadata2: dict
    ):
        """HAPPY: update_snapshot_metadata overwrites stored metadata."""
        store = _make_store()
        store.save_snapshot(snap, metadata=metadata1)
        store.update_snapshot_metadata(snap.snapshot_id, metadata2)
        record = store.get_snapshot_record(snap.snapshot_id)
        assert record is not None

    @given(snap=snapshot_st())
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_artifact_key_deterministic(self, snap: IRSnapshot):
        """HAPPY: artifact_key_for_snapshot is deterministic for same snapshot_id."""
        store = _make_store()
        key1 = store.artifact_key_for_snapshot(snap.snapshot_id)
        key2 = store.artifact_key_for_snapshot(snap.snapshot_id)
        assert key1 == key2

    @given(sid1=identifier, sid2=identifier)
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_artifact_key_differs_for_different_ids(self, sid1: str, sid2: str):
        """EDGE: different snapshot_ids produce different artifact keys (probabilistic)."""
        store = _make_store()
        key1 = store.artifact_key_for_snapshot(sid1)
        key2 = store.artifact_key_for_snapshot(sid2)
        if sid1 != sid2:
            assert key1 != key2


# --- TestScipArtifactRefProperties ---


@pytest.mark.property
class TestScipArtifactRefProperties:
    @given(
        snapshot_id=identifier,
        indexer_name=identifier,
        indexer_version=st.none() | st.just("1.0.0"),
        artifact_path=st.just("/tmp/scip.dump"),
        checksum=st.just("abc123"),
    )
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_save_and_get_scip_artifact_ref(
        self,
        snapshot_id: str,
        indexer_name: str,
        indexer_version: Any,
        artifact_path: str,
        checksum: str,
    ):
        """HAPPY: save_scip_artifact_ref then get_scip_artifact_ref roundtrip."""
        store = _make_store()
        result = store.save_scip_artifact_ref(
            snapshot_id,
            indexer_name=indexer_name,
            indexer_version=indexer_version,
            artifact_path=artifact_path,
            checksum=checksum,
        )
        assert result["snapshot_id"] == snapshot_id
        assert result["indexer_name"] == indexer_name
        assert result["artifact_path"] == artifact_path
        assert result["checksum"] == checksum
        assert result["created_at"] is not None

        loaded = store.get_scip_artifact_ref(snapshot_id)
        assert loaded is not None
        assert loaded["snapshot_id"] == snapshot_id
        assert loaded["indexer_name"] == indexer_name

    @given(snapshot_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_get_scip_artifact_ref_missing_returns_none(self, snapshot_id: str):
        """EDGE: get_scip_artifact_ref returns None when not saved."""
        store = _make_store()
        assert store.get_scip_artifact_ref(snapshot_id) is None

    @given(snapshot_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_save_scip_artifact_ref_defaults(self, snapshot_id: str):
        """HAPPY: save_scip_artifact_ref with defaults uses 'unknown' indexer_name."""
        store = _make_store()
        result = store.save_scip_artifact_ref(snapshot_id)
        assert result["indexer_name"] == "unknown"
        assert result["indexer_version"] is None
        assert result["artifact_path"] == ""
        assert result["checksum"] == ""

    @given(snapshot_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_save_scip_artifact_ref_upsert(self, snapshot_id: str):
        """HAPPY: saving SCIP artifact ref twice updates the record."""
        store = _make_store()
        store.save_scip_artifact_ref(snapshot_id, indexer_name="v1")
        r2 = store.save_scip_artifact_ref(snapshot_id, indexer_name="v2")
        assert r2["indexer_name"] == "v2"
        loaded = store.get_scip_artifact_ref(snapshot_id)
        assert loaded["indexer_name"] == "v2"


# --- TestSnapshotStoreRelationalFacts ---


@pytest.mark.property
class TestSnapshotStoreRelationalFacts:
    @given(snap=snapshot_st())
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_save_relational_facts_sqlite_noop(self, snap: IRSnapshot):
        """EDGE: save_relational_facts is no-op on SQLite (returns early)."""
        store = _make_store()
        # Should not raise
        store.save_relational_facts(snap)

    @given(snap=snapshot_st())
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_import_git_backbone_sqlite_noop(self, snap: IRSnapshot):
        """EDGE: import_git_backbone is no-op on SQLite (returns early)."""
        store = _make_store()
        store.import_git_backbone(snap, git_meta={"parent_commit_id": "deadbeef"})


# --- TestSnapshotStoreStaging ---


@pytest.mark.property
class TestSnapshotStoreStaging:
    @given(snap=snapshot_st())
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_stage_snapshot_returns_stage_id(self, snap: IRSnapshot):
        """HAPPY: stage_snapshot returns a stage_id starting with 'stage_'."""
        store = _make_store()
        stage_id = store.stage_snapshot(snap)
        assert stage_id.startswith("stage_")
        assert len(stage_id) > len("stage_")

    @given(snap=snapshot_st())
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_stage_snapshot_unique_ids(self, snap: IRSnapshot):
        """EDGE: stage_snapshot returns unique stage_ids each call."""
        store = _make_store()
        s1 = store.stage_snapshot(snap)
        s2 = store.stage_snapshot(snap)
        assert s1 != s2

    @given(snap=snapshot_st())
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_promote_staged_snapshot_sqlite_noop(self, snap: IRSnapshot):
        """EDGE: promote_staged_snapshot is no-op on SQLite (returns early)."""
        store = _make_store()
        stage_id = store.stage_snapshot(snap)
        store.promote_staged_snapshot(snap.snapshot_id, stage_id)
        # No error means success for SQLite no-op path


# --- TestSnapshotStoreLockProperties ---


@pytest.mark.property
class TestSnapshotStoreLockProperties:
    @given(lock_name=identifier, owner_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_acquire_lock_returns_one(self, lock_name: str, owner_id: str):
        """HAPPY: acquire_lock returns 1 on SQLite backend."""
        store = _make_store()
        result = store.acquire_lock(lock_name, owner_id)
        assert result == 1

    @given(lock_name=identifier, owner_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_acquire_lock_with_custom_ttl(self, lock_name: str, owner_id: str):
        """HAPPY: acquire_lock with custom TTL still returns 1 on SQLite."""
        store = _make_store()
        result = store.acquire_lock(lock_name, owner_id, ttl_seconds=600)
        assert result == 1

    @given(lock_name=identifier, token=st.integers(min_value=0, max_value=1000))
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_validate_fencing_token_returns_true(self, lock_name: str, token: int):
        """HAPPY: validate_fencing_token returns True on SQLite backend."""
        store = _make_store()
        assert store.validate_fencing_token(lock_name, token) is True

    @given(lock_name=identifier, owner_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_release_lock_noop(self, lock_name: str, owner_id: str):
        """HAPPY: release_lock is no-op on SQLite (returns None, no error)."""
        store = _make_store()
        store.release_lock(lock_name, owner_id)

    @given(lock_name=identifier, owner_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_lock_acquire_validate_release_sequence(
        self, lock_name: str, owner_id: str
    ):
        """EDGE: acquire, validate, release sequence completes without error on SQLite."""
        store = _make_store()
        token = store.acquire_lock(lock_name, owner_id)
        assert token == 1
        assert store.validate_fencing_token(lock_name, token) is True
        store.release_lock(lock_name, owner_id)


# --- TestSnapshotStoreRedoProperties ---


@pytest.mark.property
class TestSnapshotStoreRedoProperties:
    @given(
        task_type=identifier,
        payload=st.dictionaries(identifier, st.integers(), max_size=3),
    )
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_enqueue_redo_task_returns_redo_id(self, task_type: str, payload: dict):
        """HAPPY: enqueue_redo_task returns ID starting with 'redo_'."""
        store = _make_store()
        task_id = store.enqueue_redo_task(task_type, payload)
        assert task_id.startswith("redo_")

    @given(
        task_type=identifier,
        payload=st.dictionaries(identifier, st.integers(), max_size=2),
        error=st.none() | st.just("test error"),
    )
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_enqueue_redo_task_with_error(
        self, task_type: str, payload: dict, error: Exception
    ):
        """HAPPY: enqueue_redo_task with optional error still returns redo_ ID."""
        store = _make_store()
        task_id = store.enqueue_redo_task(task_type, payload, error=error)
        assert task_id.startswith("redo_")

    @given(
        task_type=identifier,
        payload=st.dictionaries(identifier, st.integers(), max_size=2),
    )
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_enqueue_redo_task_unique_ids(self, task_type: str, payload: dict):
        """EDGE: each enqueue call returns a unique task_id."""
        store = _make_store()
        t1 = store.enqueue_redo_task(task_type, payload)
        t2 = store.enqueue_redo_task(task_type, payload)
        assert t1 != t2

    @pytest.mark.edge
    def test_claim_redo_task_sqlite_noop(self):
        """EDGE: claim_redo_task returns None on SQLite backend."""
        store = _make_store()
        result = store.claim_redo_task()
        assert result is None

    @pytest.mark.edge
    def test_mark_redo_task_done_sqlite_noop(self):
        """EDGE: mark_redo_task_done is no-op on SQLite (returns None, no error)."""
        store = _make_store()
        store.mark_redo_task_done("redo_test123")

    @pytest.mark.edge
    def test_mark_redo_task_failed_sqlite_noop(self):
        """EDGE: mark_redo_task_failed is no-op on SQLite (returns None, no error)."""
        store = _make_store()
        store.mark_redo_task_failed(
            task_id="redo_test123", error="fail", max_attempts=3
        )


# --- TestIRGraphsRoundtrip ---


@pytest.mark.property
class TestIRGraphsRoundtrip:
    @given(
        snap=snapshot_st(),
        graph_data=st.dictionaries(
            identifier,
            st.lists(st.integers(min_value=0, max_value=100), max_size=5),
            max_size=3,
        ),
    )
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_save_load_ir_graphs_roundtrip(self, snap: IRSnapshot, graph_data: dict):
        """HAPPY: save_ir_graphs then load_ir_graphs roundtrip via pickle."""
        store = _make_store()
        store.save_snapshot(snap)
        path = store.save_ir_graphs(snap.snapshot_id, graph_data)
        assert path.endswith(".pkl")
        loaded = store.load_ir_graphs(snap.snapshot_id)
        assert loaded == graph_data

    @given(snapshot_id=identifier)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_load_ir_graphs_missing_returns_none(self, snapshot_id: str):
        """EDGE: load_ir_graphs returns None when snapshot not found."""
        store = _make_store()
        assert store.load_ir_graphs(snapshot_id) is None

    @given(snap=snapshot_st())
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_load_ir_graphs_none_when_not_saved(self, snap: IRSnapshot):
        """EDGE: load_ir_graphs returns None when snapshot exists but graphs not saved."""
        store = _make_store()
        store.save_snapshot(snap)
        assert store.load_ir_graphs(snap.snapshot_id) is None

    @given(snap=snapshot_st())
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_save_ir_graphs_returns_pkl_path(self, snap: IRSnapshot):
        """HAPPY: save_ir_graphs returns path ending with ir_graphs.pkl."""
        store = _make_store()
        store.save_snapshot(snap)
        path = store.save_ir_graphs(snap.snapshot_id, {"nodes": [1, 2, 3]})
        assert "ir_graphs.pkl" in path

    @given(
        snap=snapshot_st(),
        graph_obj=st.builds(
            lambda a, b: {a: [b, b + 1]},
            identifier,
            st.integers(min_value=0, max_value=50),
        ),
    )
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_ir_graphs_pickle_preserves_types(self, snap: IRSnapshot, graph_obj: dict):
        """HAPPY: pickle serialization preserves exact Python types in graphs."""
        store = _make_store()
        store.save_snapshot(snap)
        store.save_ir_graphs(snap.snapshot_id, graph_obj)
        loaded = store.load_ir_graphs(snap.snapshot_id)
        assert type(loaded) is dict
        for key, value in graph_obj.items():
            assert key in loaded
            assert loaded[key] == value

    @given(snap=snapshot_st())
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_save_ir_graphs_overwrite(self, snap: IRSnapshot):
        """HAPPY: saving IR graphs twice overwrites previous data."""
        store = _make_store()
        store.save_snapshot(snap)
        store.save_ir_graphs(snap.snapshot_id, {"v": 1})
        store.save_ir_graphs(snap.snapshot_id, {"v": 2})
        loaded = store.load_ir_graphs(snap.snapshot_id)
        assert loaded == {"v": 2}
