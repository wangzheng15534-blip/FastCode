"""Property-based tests for ir_validators.validate_snapshot invariants."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.ir_validators import validate_snapshot
from fastcode.semantic_ir import (
    IRDocument,
    IREdge,
    IROccurrence,
    IRSnapshot,
    IRSymbol,
)

# --- Local strategies (reused from conftest patterns) ---

edge_type_st = st.sampled_from(["dependency", "call", "inheritance", "reference", "contain"])


def _snapshot_st(
    max_docs: int = 3,
    max_syms: int = 5,
    max_occs: int = 8,
    max_edges: int = 4,
) -> st.SearchStrategy[IRSnapshot]:
    """Build an IRSnapshot strategy with controlled size."""
    return st.builds(
        IRSnapshot,
        repo_name=st.just("repo"),
        snapshot_id=st.just("snap:repo:abc"),
        documents=st.lists(
            st.builds(
                _doc,
                st.builds(lambda x: f"doc:{x}", small_id),
                st.builds(lambda x: f"{x}/{x}.py", small_id),
            ),
            max_size=max_docs,
        ),
        symbols=st.lists(
            st.builds(
                _sym,
                st.builds(lambda x: f"sym:{x}", small_id),
                st.builds(lambda x: f"{x}.py", small_id),
                st.builds(lambda x: f"fn_{x}", small_id),
                source=st.just("ast"),
            ),
            max_size=max_syms,
        ),
        occurrences=st.lists(
            st.builds(
                _occ,
                st.builds(lambda x: f"occ:{x}", small_id),
                st.builds(lambda x: f"sym:{x}", small_id),
                st.builds(lambda x: f"doc:{x}", small_id),
                source=st.just("ast"),
            ),
            max_size=max_occs,
        ),
        edges=st.lists(
            st.builds(
                _edge,
                st.builds(lambda x: f"edge:{x}", small_id),
                small_id,
                small_id,
                source=st.just("ast"),
            ),
            max_size=max_edges,
        ),
    )


# --- Helpers ---


def _doc(doc_id: str, path: str, source: str = "ast") -> IRDocument:
    return IRDocument(doc_id=doc_id, path=path, language="python", source_set={source})


def _sym(
    symbol_id: str,
    path: str,
    display_name: str,
    kind: str = "function",
    start_line: int = 1,
    source_priority: int = 10,
    source: str = "ast",
) -> IRSymbol:
    return IRSymbol(
        symbol_id=symbol_id,
        external_symbol_id=None,
        path=path,
        display_name=display_name,
        kind=kind,
        language="python",
        start_line=start_line,
        source_priority=source_priority,
        source_set={source},
        metadata={"source": source},
    )


def _occ(
    occ_id: str,
    symbol_id: str,
    doc_id: str,
    role: str = "definition",
    start_line: int = 1,
    start_col: int = 0,
    end_line: int = 1,
    end_col: int = 10,
    source: str = "ast",
) -> IROccurrence:
    return IROccurrence(
        occurrence_id=occ_id,
        symbol_id=symbol_id,
        doc_id=doc_id,
        role=role,
        start_line=start_line,
        start_col=start_col,
        end_line=end_line,
        end_col=end_col,
        source=source,
        metadata={},
    )


def _edge(
    edge_id: str,
    src: str,
    dst: str,
    edge_type: str = "call",
    source: str = "ast",
    confidence: str = "heuristic",
) -> IREdge:
    return IREdge(
        edge_id=edge_id,
        src_id=src,
        dst_id=dst,
        edge_type=edge_type,
        source=source,
        confidence=confidence,
    )


small_id = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)

# Strategy for a well-formed snapshot (valid by construction)
@st.composite
def valid_snapshot_st(draw):
    """Build a valid snapshot with unique doc IDs, paths, and symbol IDs."""
    n_docs = draw(st.integers(min_value=1, max_value=3))
    n_syms = draw(st.integers(min_value=1, max_value=5))
    doc_ids = [f"d{i}" for i in range(n_docs)]
    paths = [f"f{i}.py" for i in range(n_docs)]
    docs = [_doc(doc_ids[i], paths[i]) for i in range(n_docs)]
    sym_ids = [f"sym:{i}" for i in range(n_syms)]
    syms = [
        _sym(sym_ids[i], paths[i % n_docs], f"fn_{i}", source="ast")
        for i in range(n_syms)
    ]
    return IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:repo:abc",
        documents=docs,
        symbols=syms,
    )


# --- Properties ---


@pytest.mark.property
class TestValidateSnapshotProperties:

    # --- HAPPY path: valid snapshots produce no errors ---

    @given(doc_id=small_id, path=st.builds(lambda x: f"{x}.py", small_id))
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_valid_minimal_snapshot_passes(self, doc_id: str, path: str):
        """HAPPY: a snapshot with one doc and one symbol produces no errors."""
        doc = _doc(doc_id, path)
        sym = _sym(f"sym:{doc_id}", path, f"fn_{doc_id}")
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[sym],
        )
        errors = validate_snapshot(snap)
        assert errors == []

    @given(snapshot=valid_snapshot_st())
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_valid_snapshot_no_errors(self, snapshot: IRSnapshot):
        """HAPPY: well-formed snapshots have empty error lists."""
        errors = validate_snapshot(snapshot)
        assert errors == []

    @given(
        doc_id=small_id,
        path=st.builds(lambda x: f"{x}.py", small_id),
        sym_id=st.builds(lambda x: f"sym:{x}", small_id),
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_valid_occurrences_no_errors(self, doc_id: str, path: str, sym_id: str):
        """HAPPY: occurrences referencing existing doc and symbol produce no errors."""
        doc = _doc(doc_id, path)
        sym = _sym(sym_id, path, f"fn_{sym_id}")
        occ = _occ(f"occ:{doc_id}", sym_id, doc_id)
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[sym],
            occurrences=[occ],
        )
        errors = validate_snapshot(snap)
        assert errors == []

    @given(
        doc_id=small_id,
        path=st.builds(lambda x: f"{x}.py", small_id),
        sym_id=st.builds(lambda x: f"sym:{x}", small_id),
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_valid_edges_no_errors(self, doc_id: str, path: str, sym_id: str):
        """HAPPY: edges referencing valid nodes produce no errors."""
        doc = _doc(doc_id, path)
        sym = _sym(sym_id, path, f"fn_{sym_id}")
        edge = _edge(f"e:{doc_id}", doc_id, sym_id)
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[sym],
            edges=[edge],
        )
        errors = validate_snapshot(snap)
        assert errors == []

    # --- EDGE cases: missing/empty collections trigger errors ---

    @pytest.mark.edge
    def test_empty_documents_no_error(self):
        """EDGE: snapshot with no documents is valid if units exist."""
        sym = _sym("sym:1", "a.py", "foo")
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[],
            symbols=[sym],
        )
        errors = validate_snapshot(snap)
        assert len(errors) == 0

    @pytest.mark.edge
    def test_empty_symbols_no_error(self):
        """EDGE: snapshot with no symbols is valid if units exist."""
        doc = _doc("d1", "a.py")
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[],
        )
        errors = validate_snapshot(snap)
        assert len(errors) == 0

    @pytest.mark.edge
    def test_duplicate_doc_ids_error(self):
        """EDGE: duplicate document IDs (as unit IDs) are detected."""
        doc1 = _doc("d1", "a.py")
        doc2 = _doc("d1", "b.py")
        sym = _sym("sym:1", "a.py", "foo")
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc1, doc2],
            symbols=[sym],
        )
        errors = validate_snapshot(snap)
        assert any("duplicate unit IDs" in e for e in errors)

    @pytest.mark.edge
    def test_duplicate_symbol_ids_error(self):
        """EDGE: duplicate symbol IDs (as unit IDs) are detected."""
        doc = _doc("d1", "a.py")
        sym1 = _sym("sym:1", "a.py", "foo")
        sym2 = _sym("sym:1", "a.py", "bar")
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[sym1, sym2],
        )
        errors = validate_snapshot(snap)
        assert any("duplicate unit IDs" in e for e in errors)

    @pytest.mark.edge
    def test_occurrence_missing_doc_id_no_error(self):
        """EDGE: occurrence doc_id is stored in metadata, no longer validated separately."""
        sym = _sym("sym:1", "a.py", "foo")
        occ = _occ("occ:1", "sym:1", "nonexistent_doc")
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[_doc("d1", "a.py")],
            symbols=[sym],
            occurrences=[occ],
        )
        errors = validate_snapshot(snap)
        # doc_id for occurrence is in support metadata, not separately validated
        assert len(errors) == 0

    @pytest.mark.edge
    def test_occurrence_missing_symbol_id_error(self):
        """EDGE: occurrence referencing nonexistent symbol_id is flagged as support with missing unit."""
        doc = _doc("d1", "a.py")
        occ = _occ("occ:1", "nonexistent_sym", "d1")
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[_sym("sym:1", "a.py", "foo")],
            occurrences=[occ],
        )
        errors = validate_snapshot(snap)
        assert any("support references missing unit_id" in e for e in errors)

    @pytest.mark.edge
    def test_edge_src_not_found_error(self):
        """EDGE: edge with unknown src_id is flagged as relation src not found."""
        doc = _doc("d1", "a.py")
        sym = _sym("sym:1", "a.py", "foo")
        edge = _edge("e:1", "unknown_src", sym.symbol_id)
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[sym],
            edges=[edge],
        )
        errors = validate_snapshot(snap)
        assert any("relation src not found" in e for e in errors)

    @pytest.mark.edge
    def test_edge_dst_not_found_error(self):
        """EDGE: edge with unknown dst_id is flagged as relation dst not found."""
        doc = _doc("d1", "a.py")
        sym = _sym("sym:1", "a.py", "foo")
        edge = _edge("e:1", sym.symbol_id, "unknown_dst")
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[sym],
            edges=[edge],
        )
        errors = validate_snapshot(snap)
        assert any("relation dst not found" in e for e in errors)

    @pytest.mark.edge
    def test_edge_source_missing_error(self):
        """EDGE: edge with empty source string is flagged as relation source missing."""
        doc = _doc("d1", "a.py")
        sym = _sym("sym:1", "a.py", "foo")
        edge = _edge("e:1", doc.doc_id, sym.symbol_id, source="")
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[sym],
            edges=[edge],
        )
        errors = validate_snapshot(snap)
        assert any("relation source missing" in e for e in errors)

    @pytest.mark.edge
    def test_edge_confidence_mapped_to_resolution(self):
        """EDGE: edge confidence is now derived from relation resolution_state, not separately validated."""
        doc = _doc("d1", "a.py")
        sym = _sym("sym:1", "a.py", "foo")
        edge = _edge("e:1", doc.doc_id, sym.symbol_id, confidence="")
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[sym],
            edges=[edge],
        )
        errors = validate_snapshot(snap)
        # Confidence is now derived from resolution_state; empty confidence maps to "structural"
        assert len(errors) == 0

    @pytest.mark.edge
    def test_duplicate_doc_paths_error(self):
        """EDGE: duplicate document paths are detected."""
        doc1 = _doc("d1", "same.py")
        doc2 = _doc("d2", "same.py")
        sym = _sym("sym:1", "same.py", "foo")
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc1, doc2],
            symbols=[sym],
        )
        errors = validate_snapshot(snap)
        assert any("duplicate file paths" in e for e in errors)

    @pytest.mark.edge
    def test_symbol_provenance_missing_error(self):
        """EDGE: symbol with no source metadata and empty source_set is flagged."""
        doc = _doc("d1", "a.py")
        sym = IRSymbol(
            symbol_id="sym:1",
            external_symbol_id=None,
            path="a.py",
            display_name="foo",
            kind="function",
            language="python",
            source_set=set(),
            metadata={},
        )
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[sym],
        )
        errors = validate_snapshot(snap)
        assert any("provenance missing" in e for e in errors)

    @pytest.mark.edge
    def test_fully_invalid_snapshot_multiple_errors(self):
        """EDGE: a snapshot violating many rules collects all errors."""
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[],
            symbols=[],
            occurrences=[_occ("occ:1", "missing_sym", "missing_doc")],
            edges=[_edge("e:1", "x", "y", source="", confidence="")],
        )
        errors = validate_snapshot(snap)
        assert len(errors) >= 4

    # --- Property-based EDGE tests ---

    @given(snapshot=_snapshot_st(max_docs=0, max_syms=0, max_occs=0, max_edges=0))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_empty_snapshot_is_valid(self, snapshot: IRSnapshot):
        """EDGE: an empty snapshot (no units, supports, relations) has no errors."""
        errors = validate_snapshot(snapshot)
        assert len(errors) == 0

    @given(
        doc_id=st.builds(lambda x: f"d{x}", small_id),
        path=st.builds(lambda x: f"{x}.py", small_id),
        n=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_duplicate_doc_ids_always_flagged(self, doc_id: str, path: str, n: int):
        """EDGE: N documents sharing the same ID always triggers duplicate error."""
        docs = [_doc(doc_id, f"file{i}.py") for i in range(n)]
        sym = _sym("sym:1", path, "foo")
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=docs,
            symbols=[sym],
        )
        errors = validate_snapshot(snap)
        assert any("duplicate unit IDs" in e for e in errors)

    @given(
        sym_id=st.builds(lambda x: f"sym:{x}", small_id),
        n=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_duplicate_symbol_ids_always_flagged(self, sym_id: str, n: int):
        """EDGE: N symbols sharing the same ID always triggers duplicate error."""
        doc = _doc("d1", "a.py")
        syms = [_sym(sym_id, f"f{i}.py", f"fn_{i}") for i in range(n)]
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=syms,
        )
        errors = validate_snapshot(snap)
        assert any("duplicate unit IDs" in e for e in errors)

    @given(
        bad_doc_id=st.builds(lambda x: f"bad_{x}", small_id),
        bad_sym_id=st.builds(lambda x: f"bad_{x}", small_id),
    )
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_occurrence_dangling_refs_always_flagged(self, bad_doc_id: str, bad_sym_id: str):
        """EDGE: occurrences referencing nonexistent IDs always flagged."""
        doc = _doc("d1", "a.py")
        sym = _sym("sym:1", "a.py", "foo")
        occ = _occ("occ:1", bad_sym_id, bad_doc_id)
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[sym],
            occurrences=[occ],
        )
        errors = validate_snapshot(snap)
        # Occurrence converts to support; unit_id (= symbol_id) is checked against known units
        assert any("support references missing unit_id" in e for e in errors)

    @given(
        bad_src=st.builds(lambda x: f"no_{x}", small_id),
        bad_dst=st.builds(lambda x: f"no_{x}", small_id),
    )
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_edge_dangling_refs_always_flagged(self, bad_src: str, bad_dst: str):
        """EDGE: edges referencing nonexistent nodes always flagged."""
        doc = _doc("d1", "a.py")
        sym = _sym("sym:1", "a.py", "foo")
        edge = _edge("e:1", bad_src, bad_dst)
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[sym],
            edges=[edge],
        )
        errors = validate_snapshot(snap)
        assert any("relation src not found" in e for e in errors)
        assert any("relation dst not found" in e for e in errors)

    @given(
        doc_id=small_id,
        path=st.builds(lambda x: f"{x}.py", small_id),
        sym_id=st.builds(lambda x: f"sym:{x}", small_id),
    )
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_edge_src_can_be_doc_or_sym(self, doc_id: str, path: str, sym_id: str):
        """HAPPY: edge src_id may be a doc_id or symbol_id."""
        doc = _doc(doc_id, path)
        sym = _sym(sym_id, path, f"fn_{sym_id}")
        edge = _edge("e:1", doc_id, sym_id)
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[sym],
            edges=[edge],
        )
        errors = validate_snapshot(snap)
        assert not any("edge src not found" in e for e in errors)
        assert not any("edge dst not found" in e for e in errors)

    @given(
        doc_id=small_id,
        path=st.builds(lambda x: f"{x}.py", small_id),
        sym_id=st.builds(lambda x: f"sym:{x}", small_id),
        edge_type=edge_type_st,
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_all_edge_types_valid(self, doc_id: str, path: str, sym_id: str, edge_type: str):
        """HAPPY: all valid edge_types pass validation with correct node refs."""
        doc = _doc(doc_id, path)
        sym = _sym(sym_id, path, f"fn_{sym_id}")
        edge = _edge("e:1", doc_id, sym_id, edge_type=edge_type)
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[sym],
            edges=[edge],
        )
        errors = validate_snapshot(snap)
        assert errors == []
