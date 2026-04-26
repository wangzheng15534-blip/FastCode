"""Property-based tests for adapters/scip_to_ir.build_ir_from_scip invariants.

Covers: valid IRSnapshot production, source_priority=100, source_set={"scip"},
symbol ID prefix, containment edges, empty payloads, metadata, raw dict inputs,
range coercion, and occurrence source field.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.adapters.scip_to_ir import build_ir_from_scip
from fastcode.scip_models import SCIPDocument, SCIPIndex, SCIPOccurrence, SCIPSymbol
from fastcode.semantic_ir import IRSnapshot

# --- Strategies (mirrored from tests/conftest.py) ---

identifier = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=40,
)

file_path_st = st.tuples(identifier, identifier).map(lambda t: f"{t[0]}/{t[1]}.py")

language_st = st.sampled_from(
    ["python", "javascript", "typescript", "go", "java", "rust", "c", "cpp"]
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
        "parameter",
        "type",
        "macro",
        "field",
    ]
)

scip_occurrence_st = st.builds(
    SCIPOccurrence,
    symbol=st.builds(lambda x: f"scip python pkg {x} method()", identifier),
    role=st.sampled_from(
        [
            "definition",
            "reference",
            "implementation",
            "import",
            "write_access",
            "type_definition",
            "forward_definition",
        ]
    ),
    range=st.lists(
        st.none() | st.integers(min_value=0, max_value=500), min_size=0, max_size=4
    ),
)

scip_symbol_st = st.builds(
    SCIPSymbol,
    symbol=st.builds(lambda x: f"scip python pkg {x} func()", identifier),
    name=st.none() | identifier,
    kind=st.sampled_from(
        [
            "function",
            "method",
            "class",
            "variable",
            "module",
            "interface",
            "enum",
            "constant",
            "parameter",
            "type",
            "macro",
            "field",
        ]
    ),
    qualified_name=st.none() | st.builds(lambda x: f"pkg.{x}", identifier),
    signature=st.none() | st.just("def foo(x: int) -> str"),
    range=st.lists(
        st.none() | st.integers(min_value=0, max_value=500), min_size=0, max_size=4
    ),
)

scip_document_st = st.builds(
    SCIPDocument,
    path=file_path_st,
    language=st.none() | language_st,
    symbols=st.lists(scip_symbol_st, max_size=3),
    occurrences=st.lists(scip_occurrence_st, max_size=4),
)

scip_index_st = st.builds(
    SCIPIndex,
    documents=st.lists(scip_document_st, max_size=3),
    indexer_name=st.none()
    | st.sampled_from(["scip-python", "scip-java", "scip-go", "scip-typescript"]),
    indexer_version=st.none() | st.just("1.0.0"),
    metadata=st.dictionaries(
        identifier, st.integers(min_value=0, max_value=100), max_size=3
    ),
)

scip_raw_payload_st = st.builds(
    lambda docs, iname, iver: {
        "documents": docs,
        "indexer_name": iname,
        "indexer_version": iver,
    },
    docs=st.lists(
        st.fixed_dictionaries(
            {
                "path": file_path_st,
                "language": st.none() | language_st,
                "symbols": st.lists(
                    st.fixed_dictionaries(
                        {
                            "symbol": st.builds(
                                lambda x: f"scip python pkg {x} func()", identifier
                            ),
                            "name": st.none() | identifier,
                            "kind": st.sampled_from(
                                ["function", "method", "class", "variable"]
                            ),
                            "range": st.lists(
                                st.none() | st.integers(0, 500), min_size=0, max_size=4
                            ),
                        },
                        optional={
                            "signature": st.just("def foo()"),
                            "qualified_name": st.builds(
                                lambda x: f"pkg.{x}", identifier
                            ),
                        },
                    ),
                    max_size=3,
                ),
                "occurrences": st.lists(
                    st.fixed_dictionaries(
                        {
                            "symbol": st.builds(
                                lambda x: f"scip python pkg {x} func()", identifier
                            ),
                            "role": st.sampled_from(
                                ["definition", "reference", "implementation"]
                            ),
                            "range": st.lists(
                                st.none() | st.integers(0, 500), min_size=0, max_size=4
                            ),
                        }
                    ),
                    max_size=3,
                ),
            }
        ),
        max_size=2,
    ),
    iname=st.none() | st.just("scip-python"),
    iver=st.none() | st.just("1.0.0"),
)


# --- TestBuildIrFromScipValidSnapshot ---


@pytest.mark.property
class TestBuildIrFromScipValidSnapshot:
    @given(index=scip_index_st, repo=identifier, snap_id=identifier)
    @settings(max_examples=40)
    @pytest.mark.happy
    def test_produces_valid_ir_snapshot(
        self, index: SCIPIndex, repo: str, snap_id: str
    ):
        """HAPPY: every SCIPIndex produces a valid IRSnapshot."""
        snap = build_ir_from_scip(repo_name=repo, snapshot_id=snap_id, scip_index=index)
        assert isinstance(snap, IRSnapshot)
        assert snap.repo_name == repo
        assert snap.snapshot_id == snap_id

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_source_priority_is_100(self, index: SCIPIndex, snap_id: str):
        """HAPPY: all SCIP-derived symbols have source_priority=100."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        for sym in snap.symbols:
            assert sym.source_priority == 100

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_source_set_is_scip(self, index: SCIPIndex, snap_id: str):
        """HAPPY: all SCIP-derived documents and symbols have source_set={"scip"}."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        for doc in snap.documents:
            assert doc.source_set == {"scip"}
        for sym in snap.symbols:
            assert sym.source_set == {"scip"}

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_symbol_ids_prefixed(self, index: SCIPIndex, snap_id: str):
        """HAPPY: every symbol ID starts with 'scip:{snapshot_id}:'."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        expected_prefix = f"scip:{snap_id}:"
        for sym in snap.symbols:
            assert sym.symbol_id.startswith(expected_prefix)

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_metadata_source_modes_scip(self, index: SCIPIndex, snap_id: str):
        """HAPPY: snapshot metadata contains source_modes=["scip"]."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        assert snap.metadata["source_modes"] == ["scip"]

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_documents_count_includes_all_scip_docs(
        self, index: SCIPIndex, snap_id: str
    ):
        """HAPPY: output IR documents include at least one per SCIP document."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        assert len(snap.documents) >= len(index.documents)

    @given(
        index=scip_index_st,
        snap_id=identifier,
        branch=identifier,
        commit_id=st.text(alphabet="0123456789abcdef", min_size=7, max_size=40),
        tree_id=identifier,
    )
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_branch_commit_tree_passed_through(
        self, index: SCIPIndex, snap_id: str, branch: str, commit_id: str, tree_id: str
    ):
        """HAPPY: branch, commit_id, tree_id are stored in the snapshot."""
        snap = build_ir_from_scip(
            repo_name="repo",
            snapshot_id=snap_id,
            scip_index=index,
            branch=branch,
            commit_id=commit_id,
            tree_id=tree_id,
        )
        assert snap.branch == branch
        assert snap.commit_id == commit_id
        assert snap.tree_id == tree_id


# --- TestContainmentEdges ---


@pytest.mark.property
class TestContainmentEdges:
    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_every_symbol_gets_containment_edge(self, index: SCIPIndex, snap_id: str):
        """HAPPY: every symbol unit has a containment edge from its document.

        The adapter normalises kind='module' to kind='file', so those units
        appear in snap.documents rather than snap.symbols. Containment relations
        are created for *all* non-document symbol units, so the edge count is
        always >= the symbol count.
        """
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        containment_edges = [e for e in snap.edges if e.edge_type == "contain"]
        assert len(containment_edges) >= len(snap.symbols)

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_containment_edge_source_is_scip(self, index: SCIPIndex, snap_id: str):
        """HAPPY: containment edges have source='scip'."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        for edge in snap.edges:
            if edge.edge_type == "contain":
                assert edge.source == "scip"

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_containment_edge_confidence_precise(self, index: SCIPIndex, snap_id: str):
        """HAPPY: containment edges have confidence='precise'."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        for edge in snap.edges:
            if edge.edge_type == "contain":
                assert edge.confidence == "precise"


# --- TestEmptyAndEdgeCases ---


@pytest.mark.property
class TestEmptyAndEdgeCases:
    @given(snap_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_empty_payload_produces_empty_snapshot(self, snap_id: str):
        """EDGE: empty SCIPIndex produces snapshot with 0 docs, 0 symbols."""
        index = SCIPIndex(documents=[])
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        assert len(snap.documents) == 0
        assert len(snap.symbols) == 0
        assert len(snap.occurrences) == 0
        assert len(snap.edges) == 0
        assert snap.metadata["source_modes"] == ["scip"]

    @given(payload=scip_raw_payload_st, snap_id=identifier)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_raw_dict_input_produces_valid_snapshot(
        self, payload: dict[str, Any], snap_id: str
    ):
        """HAPPY: raw dict input (not SCIPIndex) produces valid IRSnapshot."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=payload
        )
        assert isinstance(snap, IRSnapshot)
        assert snap.repo_name == "repo"
        assert snap.snapshot_id == snap_id

    @given(snap_id=identifier)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_document_missing_language_falls_back_to_hint(self, snap_id: str):
        """EDGE: document with no language uses language_hint."""
        index = SCIPIndex(
            documents=[
                SCIPDocument(
                    path="f.txt",
                    language=None,
                    symbols=[
                        SCIPSymbol(symbol="pkg foo.", name="foo"),
                    ],
                ),
            ]
        )
        snap = build_ir_from_scip(
            repo_name="repo",
            snapshot_id=snap_id,
            scip_index=index,
            language_hint="rust",
        )
        assert snap.documents[0].language == "rust"

    @given(snap_id=identifier)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_document_missing_language_no_hint_falls_back_to_unknown(
        self, snap_id: str
    ):
        """EDGE: document with no language and no hint falls back to 'unknown'."""
        index = SCIPIndex(
            documents=[
                SCIPDocument(
                    path="f.txt",
                    language=None,
                    symbols=[
                        SCIPSymbol(symbol="pkg foo.", name="foo"),
                    ],
                ),
            ]
        )
        snap = build_ir_from_scip(
            repo_name="repo",
            snapshot_id=snap_id,
            scip_index=index,
        )
        assert snap.documents[0].language == "unknown"

    @given(snap_id=identifier)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_symbol_empty_string_skipped(self, snap_id: str):
        """EDGE: symbol with empty string is skipped."""
        index = SCIPIndex(
            documents=[
                SCIPDocument(
                    path="a.py",
                    language="python",
                    symbols=[
                        SCIPSymbol(symbol="", name="empty"),
                        SCIPSymbol(symbol="pkg valid.", name="valid"),
                    ],
                ),
            ]
        )
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        assert len(snap.symbols) == 1

    @given(snap_id=identifier)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_occurrence_empty_symbol_skipped(self, snap_id: str):
        """EDGE: occurrence with empty symbol string is skipped."""
        index = SCIPIndex(
            documents=[
                SCIPDocument(
                    path="a.py",
                    language="python",
                    occurrences=[
                        SCIPOccurrence(
                            symbol="", role="reference", range=[1, 0, 1, 10]
                        ),
                        SCIPOccurrence(
                            symbol="pkg foo.", role="definition", range=[2, 0, 2, 5]
                        ),
                    ],
                ),
            ]
        )
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        assert len(snap.occurrences) == 1


# --- TestOccurrenceProperties ---


@pytest.mark.property
class TestOccurrenceProperties:
    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_all_occurrences_source_is_scip(self, index: SCIPIndex, snap_id: str):
        """HAPPY: all IR occurrences have source='scip'."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        for occ in snap.occurrences:
            assert occ.source == "scip"

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_occurrence_range_none_coerced_to_zero(
        self, index: SCIPIndex, snap_id: str
    ):
        """HAPPY: None range values are coerced to 0 in occurrences."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        for occ in snap.occurrences:
            assert occ.start_line is not None
            assert occ.start_col is not None
            assert occ.end_line is not None
            assert occ.end_col is not None
            assert isinstance(occ.start_line, int)
            assert isinstance(occ.start_col, int)

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_occurrence_symbol_ids_prefixed(self, index: SCIPIndex, snap_id: str):
        """HAPPY: every occurrence symbol_id starts with 'scip:{snapshot_id}:'."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        prefix = f"scip:{snap_id}:"
        for occ in snap.occurrences:
            assert occ.symbol_id.startswith(prefix)

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_occurrence_metadata_has_scip_source(self, index: SCIPIndex, snap_id: str):
        """HAPPY: every occurrence metadata contains source='scip'."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        for occ in snap.occurrences:
            assert occ.metadata.get("source") == "scip"
            assert occ.metadata.get("confidence") == "precise"

    @given(snap_id=identifier)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_occurrence_ref_role_produces_ref_edge(self, snap_id: str):
        """EDGE: occurrences with 'reference' role produce ref edges."""
        index = SCIPIndex(
            documents=[
                SCIPDocument(
                    path="a.py",
                    language="python",
                    occurrences=[
                        SCIPOccurrence(
                            symbol="pkg foo.", role="reference", range=[1, 0, 1, 10]
                        ),
                    ],
                ),
            ]
        )
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        ref_edges = [e for e in snap.edges if e.edge_type == "ref"]
        assert len(ref_edges) >= 1

    @given(snap_id=identifier)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_symbol_metadata_scip_true(self, snap_id: str):
        """EDGE: SCIP-derived symbol metadata has scip=True."""
        index = SCIPIndex(
            documents=[
                SCIPDocument(
                    path="a.py",
                    language="python",
                    symbols=[
                        SCIPSymbol(symbol="pkg foo.", name="foo", kind="function"),
                    ],
                ),
            ]
        )
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        assert snap.symbols[0].metadata.get("scip") is True
        assert snap.symbols[0].metadata.get("source") == "scip"
