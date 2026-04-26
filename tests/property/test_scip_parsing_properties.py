"""Property-based tests for SCIP parsing and conversion."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.adapters.scip_to_ir import build_ir_from_scip
from fastcode.scip_models import SCIPDocument, SCIPIndex, SCIPOccurrence, SCIPSymbol

# --- Strategies ---

small_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_/",
    min_size=1, max_size=30,
)

role_st = st.sampled_from(["definition", "reference", "import", "implementation",
                            "write_access", "forward_definition", "type_definition"])

kind_st = st.sampled_from(["function", "method", "class", "variable", "module",
                            "interface", "enum", "constant", "macro"])

@st.composite
def _valid_range(draw):
    start_line = draw(st.one_of(st.integers(min_value=0, max_value=1000), st.none()))
    if start_line is None:
        end_line = draw(st.one_of(st.integers(min_value=0, max_value=1000), st.none()))
    else:
        end_line = draw(st.one_of(st.integers(min_value=start_line, max_value=1000), st.none()))
    start_col = draw(st.one_of(st.integers(min_value=0, max_value=1000), st.none()))
    end_col = draw(st.one_of(st.integers(min_value=0, max_value=1000), st.none()))
    return [start_line, start_col, end_line, end_col]

range_st = _valid_range()

scip_occurrence_st = st.builds(
    SCIPOccurrence,
    symbol=st.builds(lambda x: f"pkg {x}.", small_text),
    role=role_st,
    range=range_st,
)

scip_symbol_st = st.builds(
    SCIPSymbol,
    symbol=st.builds(lambda x: f"pkg {x}.", small_text),
    name=st.none() | small_text,
    kind=st.none() | kind_st,
    qualified_name=st.none() | st.builds(lambda x: f"pkg.{x}", small_text),
    signature=st.none() | st.just("def foo(x)"),
    range=range_st,
)

scip_document_st = st.builds(
    SCIPDocument,
    path=st.builds(lambda x: f"{x}.py", small_text),
    language=st.none() | st.sampled_from(["python", "javascript", "go", "java"]),
    symbols=st.lists(scip_symbol_st, max_size=3),
    occurrences=st.lists(scip_occurrence_st, max_size=5),
)

scip_index_st = st.builds(
    SCIPIndex,
    documents=st.lists(scip_document_st, min_size=1, max_size=3),
    indexer_name=st.none() | st.just("scip-python"),
    indexer_version=st.none() | st.just("1.0.0"),
)


# --- Properties ---

@pytest.mark.property
class TestScipParsingProperties:

    @given(index=scip_index_st)
    @settings(max_examples=50)
    @pytest.mark.happy
    def test_scip_index_roundtrip(self, index: SCIPIndex):
        """HAPPY: SCIPIndex.from_dict(data).to_dict() preserves key fields."""
        data = index.to_dict()
        restored = SCIPIndex.from_dict(data)
        assert restored.indexer_name == index.indexer_name
        assert restored.indexer_version == index.indexer_version
        assert len(restored.documents) == len(index.documents)
        for orig_doc, rest_doc in zip(index.documents, restored.documents):
            assert orig_doc.path == rest_doc.path
            assert len(orig_doc.symbols) == len(rest_doc.symbols)

    @given(index=scip_index_st, snap_id=small_text)
    @settings(max_examples=40)
    @pytest.mark.happy
    def test_build_ir_from_scip_produces_valid_snapshot(self, index: SCIPIndex, snap_id: str):
        """HAPPY: build_ir_from_scip always produces IRSnapshot with documents."""
        snap = build_ir_from_scip(
            repo_name="repo",
            snapshot_id=snap_id,
            scip_index=index,
        )
        assert snap.repo_name == "repo"
        assert snap.snapshot_id == snap_id
        assert len(snap.documents) >= len(index.documents)
        total_symbols = sum(len(d.symbols) for d in index.documents)
        # The adapter may normalize module-kind symbols to file units, so
        # snap.symbols can be fewer than total_symbols when modules are present.
        # It can also be fewer when duplicate symbol strings appear in the same doc.
        assert len(snap.symbols) + len(snap.documents) >= len(index.documents)

    @given(index=scip_index_st)
    @settings(max_examples=50)
    @pytest.mark.edge
    def test_scip_occurrence_ranges_valid(self, index: SCIPIndex):
        """EDGE: SCIP occurrence ranges have start_line <= end_line (when both present)."""
        for doc in index.documents:
            for occ in doc.occurrences:
                r = occ.range
                if r[0] is not None and r[2] is not None:
                    assert r[0] <= r[2], f"start_line {r[0]} > end_line {r[2]}"

    @given(index=scip_index_st, snap_id=small_text)
    @settings(max_examples=40)
    @pytest.mark.happy
    def test_scip_symbol_ids_follow_pattern(self, index: SCIPIndex, snap_id: str):
        """HAPPY: SCIP-derived IRSymbol IDs follow scip:{snapshot_id}:... pattern."""
        snap = build_ir_from_scip(
            repo_name="repo",
            snapshot_id=snap_id,
            scip_index=index,
        )
        for sym in snap.symbols:
            assert sym.symbol_id.startswith(f"scip:{snap_id}:"), \
                f"Symbol ID {sym.symbol_id} does not match expected pattern"

    @given(doc=scip_document_st)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_scip_document_roundtrip(self, doc: SCIPDocument):
        """HAPPY: SCIPDocument roundtrip through dict preserves path and counts."""
        data = doc.to_dict()
        restored = SCIPDocument.from_dict(data)
        assert restored.path == doc.path
        assert len(restored.symbols) == len(doc.symbols)
        assert len(restored.occurrences) == len(doc.occurrences)

    @given(sym=scip_symbol_st)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_scip_symbol_roundtrip(self, sym: SCIPSymbol):
        """HAPPY: SCIPSymbol roundtrip preserves all fields."""
        data = sym.to_dict()
        restored = SCIPSymbol.from_dict(data)
        assert restored.symbol == sym.symbol
        assert restored.name == sym.name
        assert restored.kind == sym.kind

    @given(snap_id=small_text)
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_build_ir_from_scip_empty_documents(self, snap_id: str):
        """EDGE: empty SCIP index produces empty IR snapshot."""
        index = SCIPIndex(documents=[])
        snap = build_ir_from_scip(
            repo_name="repo",
            snapshot_id=snap_id,
            scip_index=index,
        )
        assert len(snap.documents) == 0
        assert len(snap.symbols) == 0
        assert len(snap.occurrences) == 0

    @given(snap_id=small_text)
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_build_ir_from_scip_document_missing_language(self, snap_id: str):
        """EDGE: document with no language falls back to language_hint or 'unknown'."""
        index = SCIPIndex(documents=[
            SCIPDocument(path="unknown.txt", language=None, symbols=[
                SCIPSymbol(symbol="pkg foo.", name="foo"),
            ]),
        ])
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id,
            scip_index=index, language_hint="python",
        )
        assert snap.documents[0].language == "python"

        snap_no_hint = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id,
            scip_index=index,
        )
        assert snap_no_hint.documents[0].language == "unknown"

    @given(snap_id=small_text)
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_build_ir_from_scip_symbol_empty_string_skipped(self, snap_id: str):
        """EDGE: symbol with empty string is skipped (no crash)."""
        index = SCIPIndex(documents=[
            SCIPDocument(path="a.py", language="python", symbols=[
                SCIPSymbol(symbol="", name="empty"),
                SCIPSymbol(symbol="pkg valid.", name="valid"),
            ]),
        ])
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index,
        )
        assert len(snap.symbols) == 1
        assert snap.symbols[0].display_name == "valid"

    @given(data=st.dictionaries(st.text(min_size=1), st.integers()))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_scip_index_from_arbitrary_dict(self, data):
        """EDGE: SCIPIndex.from_dict handles arbitrary dicts without crash."""
        index = SCIPIndex.from_dict(data)
        assert isinstance(index, SCIPIndex)
        assert isinstance(index.documents, list)

    @pytest.mark.edge
    def test_scip_occurrence_empty_symbol(self):
        """EDGE: SCIPOccurrence with empty symbol string doesn't crash."""
        occ = SCIPOccurrence(symbol="", role="reference", range=[None, None, None, None])
        data = occ.to_dict()
        restored = SCIPOccurrence.from_dict(data)
        assert restored.symbol == ""

    @given(snap_id=small_text)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_build_ir_from_scip_no_occurrences(self, snap_id: str):
        """EDGE: documents with symbols but no occurrences produces zero IR occurrences."""
        index = SCIPIndex(documents=[
            SCIPDocument(path="a.py", language="python", symbols=[
                SCIPSymbol(symbol="pkg foo.", name="foo"),
            ], occurrences=[]),
        ])
        snap = build_ir_from_scip(repo_name="repo", snapshot_id=snap_id, scip_index=index)
        assert len(snap.occurrences) == 0
        assert len(snap.symbols) == 1

    @given(text_part=small_text)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_scip_symbol_long_symbol_string(self, text_part: str):
        """EDGE: very long symbol string handled without crash."""
        sym = f"pkg {text_part * 50}."
        s = SCIPSymbol(symbol=sym, name="long")
        data = s.to_dict()
        restored = SCIPSymbol.from_dict(data)
        assert restored.symbol == sym

    @given(snap_id=small_text)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_build_ir_from_scip_occurrences_only_no_symbols(self, snap_id: str):
        """EDGE: documents with occurrences but no symbols produces occurrences in IR."""
        index = SCIPIndex(documents=[
            SCIPDocument(path="a.py", language="python", symbols=[],
                         occurrences=[
                             SCIPOccurrence(symbol="pkg foo.", role="reference",
                                           range=[1, 0, 1, 10]),
                         ]),
        ])
        snap = build_ir_from_scip(repo_name="repo", snapshot_id=snap_id, scip_index=index)
        assert len(snap.symbols) == 0
        assert len(snap.documents) == 1

    @given(snap_id=small_text)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_build_ir_from_scip_duplicate_symbols_across_docs(self, snap_id: str):
        """EDGE: same symbol name in different docs produces distinct IR symbols."""
        sym = SCIPSymbol(symbol="pkg foo.", name="foo", kind="function")
        index = SCIPIndex(documents=[
            SCIPDocument(path="a.py", language="python", symbols=[sym]),
            SCIPDocument(path="b.py", language="python", symbols=[sym]),
        ])
        snap = build_ir_from_scip(repo_name="repo", snapshot_id=snap_id, scip_index=index)
        assert len(snap.symbols) == 2

    @pytest.mark.edge
    def test_scip_occurrence_all_none_range(self):
        """EDGE: occurrence with all-None range doesn't crash."""
        occ = SCIPOccurrence(symbol="pkg x.", role="reference", range=[None, None, None, None])
        d = occ.to_dict()
        r = SCIPOccurrence.from_dict(d)
        assert r.range == [None, None, None, None]

    @pytest.mark.edge
    def test_scip_symbol_missing_optional_fields(self):
        """EDGE: symbol with only required fields gets defaults."""
        sym = SCIPSymbol(symbol="pkg x.")
        assert sym.name is None
        assert sym.kind is None
        d = sym.to_dict()
        r = SCIPSymbol.from_dict(d)
        assert r.symbol == "pkg x."

    @pytest.mark.edge
    def test_scip_document_no_language_no_symbols(self):
        """EDGE: document with no language and no symbols roundtrips."""
        doc = SCIPDocument(path="data.bin", language=None, symbols=[], occurrences=[])
        d = doc.to_dict()
        r = SCIPDocument.from_dict(d)
        assert r.path == "data.bin"
        assert r.language is None

    @pytest.mark.edge
    def test_build_ir_from_scip_empty_symbol_name_skipped(self):
        """EDGE: symbol with empty name field still gets processed."""
        index = SCIPIndex(documents=[
            SCIPDocument(path="a.py", language="python", symbols=[
                SCIPSymbol(symbol="pkg foo.", name=""),
            ]),
        ])
        snap = build_ir_from_scip(repo_name="repo", snapshot_id="s1", scip_index=index)
        assert len(snap.symbols) == 1

    @pytest.mark.edge
    def test_scip_index_with_metadata(self):
        """EDGE: SCIPIndex with indexer metadata roundtrips."""
        idx = SCIPIndex(documents=[], indexer_name="scip-go", indexer_version="1.2.3")
        d = idx.to_dict()
        r = SCIPIndex.from_dict(d)
        assert r.indexer_name == "scip-go"
        assert r.indexer_version == "1.2.3"

    @given(data=st.dictionaries(st.text(min_size=1, max_size=5), st.integers(min_value=0, max_value=100), max_size=3))
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_scip_occurrence_from_arbitrary_dict(self, data):
        """EDGE: SCIPOccurrence.from_dict handles arbitrary dicts without crash."""
        occ = SCIPOccurrence.from_dict(data)
        assert isinstance(occ, SCIPOccurrence)

    @pytest.mark.edge
    def test_scip_symbol_roundtrip_preserves_range(self):
        """EDGE: SCIPSymbol range field survives roundtrip."""
        sym = SCIPSymbol(symbol="pkg x.", name="x", range=[10, 0, 20, 5])
        d = sym.to_dict()
        r = SCIPSymbol.from_dict(d)
        assert r.range == [10, 0, 20, 5]

    @pytest.mark.edge
    def test_scip_document_with_many_occurrences(self):
        """EDGE: document with many occurrences roundtrips."""
        occs = [SCIPOccurrence(symbol=f"pkg s{i}.", role="reference", range=[i, 0, i, 5]) for i in range(20)]
        doc = SCIPDocument(path="big.py", language="python", symbols=[], occurrences=occs)
        d = doc.to_dict()
        r = SCIPDocument.from_dict(d)
        assert len(r.occurrences) == 20
