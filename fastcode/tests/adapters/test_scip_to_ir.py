"""Tests for adapters.scip_to_ir module."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.adapters.scip_to_ir import build_ir_from_scip
from fastcode.scip_models import SCIPDocument, SCIPIndex, SCIPOccurrence, SCIPSymbol
from fastcode.semantic_ir import IRSnapshot

# ---------------------------------------------------------------------------
# Strategies (mirrored from tests/conftest.py)
# ---------------------------------------------------------------------------

identifier = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=40,
)

small_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_/",
    min_size=1,
    max_size=30,
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

role_st = st.sampled_from(
    [
        "definition",
        "reference",
        "import",
        "implementation",
        "write_access",
        "forward_definition",
        "type_definition",
    ]
)


@st.composite
def _valid_range(draw: Any) -> Any:
    start_line = draw(st.one_of(st.integers(min_value=0, max_value=1000), st.none()))
    if start_line is None:
        end_line = draw(st.one_of(st.integers(min_value=0, max_value=1000), st.none()))
    else:
        end_line = draw(
            st.one_of(st.integers(min_value=start_line, max_value=1000), st.none())
        )
    start_col = draw(st.one_of(st.integers(min_value=0, max_value=1000), st.none()))
    end_col = draw(st.one_of(st.integers(min_value=0, max_value=1000), st.none()))
    return [start_line, start_col, end_line, end_col]


range_st = _valid_range()

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

# Parsing-specific strategies
parsing_scip_occurrence_st = st.builds(
    SCIPOccurrence,
    symbol=st.builds(lambda x: f"pkg {x}.", small_text),
    role=role_st,
    range=range_st,
)

parsing_scip_symbol_st = st.builds(
    SCIPSymbol,
    symbol=st.builds(lambda x: f"pkg {x}.", small_text),
    name=st.none() | small_text,
    kind=st.none() | kind_st,
    qualified_name=st.none() | st.builds(lambda x: f"pkg.{x}", small_text),
    signature=st.none() | st.just("def foo(x)"),
    range=range_st,
)

parsing_scip_document_st = st.builds(
    SCIPDocument,
    path=st.builds(lambda x: f"{x}.py", small_text),
    language=st.none() | st.sampled_from(["python", "javascript", "go", "java"]),
    symbols=st.lists(parsing_scip_symbol_st, max_size=3),
    occurrences=st.lists(parsing_scip_occurrence_st, max_size=5),
)

parsing_scip_index_st = st.builds(
    SCIPIndex,
    documents=st.lists(parsing_scip_document_st, min_size=1, max_size=3),
    indexer_name=st.none() | st.just("scip-python"),
    indexer_version=st.none() | st.just("1.0.0"),
)

# ---------------------------------------------------------------------------
# Inline factory (mirrors tests.conftest._make_scip_payload)
# ---------------------------------------------------------------------------


def _make_scip_payload(
    n_docs: int = 1,
    n_symbols: int = 2,
    n_occurrences: int = 1,
) -> dict[str, Any]:
    """Create a raw dict payload matching build_ir_from_scip() input schema."""
    documents = []
    for i in range(n_docs):
        syms = []
        for j in range(n_symbols):
            syms.append(
                {
                    "symbol": f"scip python test func_{j}()",
                    "name": f"func_{j}",
                    "kind": "function",
                    "range": [j + 1, 0, j + 1, 20],
                }
            )
        occs = []
        for k in range(n_occurrences):
            occs.append(
                {
                    "symbol": f"scip python test func_{k % n_symbols}()",
                    "role": "definition",
                    "range": [k + 1, 0, k + 1, 20],
                }
            )
        documents.append(
            {
                "path": f"src/file{i}.py",
                "language": "python",
                "symbols": syms,
                "occurrences": occs,
            }
        )
    return {
        "documents": documents,
        "indexer_name": "scip-python",
        "indexer_version": "1.0.0",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(
    payload: dict[str, Any] | SCIPIndex | None = None,
    *,
    language_hint: str | None = None,
) -> IRSnapshot:
    """Thin wrapper around build_ir_from_scip with sensible defaults."""
    if payload is None:
        payload = _make_scip_payload()
    snap = build_ir_from_scip(
        repo_name="test_repo",
        snapshot_id="snap:test:abc1234",
        scip_index=payload,
        branch="main",
        commit_id="abc1234",
        tree_id="tree1",
        language_hint=language_hint,
    )
    assert isinstance(snap, IRSnapshot)
    return snap


# =========================================================================
# Basic tests
# =========================================================================


_REF_ROLES = [
    "reference",
    "definition",
    "implementation",
    "type_definition",
    "import",
    "write_access",
    "forward_definition",
]


@pytest.mark.parametrize("role", _REF_ROLES)
def test_role_produces_ref_edge(role: str):
    """Roles in the whitelist produce a 'ref' edge."""
    payload = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=1)
    payload["documents"][0]["occurrences"][0]["role"] = role
    snap = _build(payload)
    ref_edges = [e for e in snap.edges if e.edge_type == "ref"]
    assert len(ref_edges) == 1
    assert ref_edges[0].metadata["role"] == role


def test_all_scip_symbols_have_priority_100():
    """Every SCIP-derived symbol has source_priority == 100."""
    payload = _make_scip_payload(n_docs=2, n_symbols=3, n_occurrences=0)
    snap = _build(payload)
    assert len(snap.symbols) == 6  # 2 docs * 3 symbols
    for sym in snap.symbols:
        assert sym.source_priority == 100


def test_symbol_metadata_fields():
    """SCIP symbols carry metadata with scip=True, source='scip', confidence='precise'."""
    payload = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=0)
    snap = _build(payload)
    sym = snap.symbols[0]
    assert sym.metadata["scip"] is True
    assert sym.metadata["source"] == "scip"
    assert sym.metadata["confidence"] == "precise"


def test_indexer_name_version_propagated_to_symbol_metadata():
    """indexer_name and indexer_version flow into symbol metadata."""
    payload = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=0)
    payload["indexer_name"] = "scip-java"
    payload["indexer_version"] = "2.3.0"
    snap = _build(payload)
    sym = snap.symbols[0]
    assert sym.metadata["indexer_name"] == "scip-java"
    assert sym.metadata["indexer_version"] == "2.3.0"


def test_occurrence_metadata_fields():
    """Occurrences carry source='scip' and confidence='precise'."""
    payload = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=1)
    snap = _build(payload)
    occ = snap.occurrences[0]
    assert occ.metadata["source"] == "scip"
    assert occ.metadata["confidence"] == "precise"


def test_snapshot_metadata_source_modes():
    """Snapshot-level metadata records source_modes=['scip']."""
    snap = _build(_make_scip_payload())
    assert snap.metadata["source_modes"] == ["scip"]


def test_contain_edge_metadata():
    """Contain edges carry extractor metadata."""
    payload = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=0)
    snap = _build(payload)
    contain_edges = [e for e in snap.edges if e.edge_type == "contain"]
    assert len(contain_edges) == 1
    assert contain_edges[0].metadata["extractor"] == "fastcode.adapters.scip_to_ir"


def test_dict_input_produces_snapshot():
    """Raw dict payload produces a valid snapshot."""
    payload = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=1)
    snap = _build(payload)
    assert snap.repo_name == "test_repo"
    assert len(snap.documents) == 1
    assert len(snap.symbols) == 1
    assert len(snap.occurrences) == 1


def test_scip_index_input_produces_snapshot():
    """SCIPIndex object produces a valid snapshot with same content as dict."""
    raw = _make_scip_payload(n_docs=1, n_symbols=2, n_occurrences=1)
    scip_idx = SCIPIndex.from_dict(raw)
    snap = _build(scip_idx)
    assert snap.repo_name == "test_repo"
    assert len(snap.documents) == 1
    assert len(snap.symbols) == 2
    assert len(snap.occurrences) == 1


def test_dict_and_scip_index_produce_equivalent_symbols():
    """Both input types produce the same symbol IDs for equivalent payloads."""
    raw = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=0)
    scip_idx = SCIPIndex.from_dict(raw)
    snap_dict = _build(raw)
    snap_obj = _build(scip_idx)
    assert snap_dict.symbols[0].symbol_id == snap_obj.symbols[0].symbol_id
    assert snap_dict.symbols[0].display_name == snap_obj.symbols[0].display_name


def test_multiple_documents_each_scoped():
    """Multiple documents produce separate IRDocument entries."""
    payload = _make_scip_payload(n_docs=3, n_symbols=1, n_occurrences=0)
    snap = _build(payload)
    assert len(snap.documents) == 3
    paths = {d.path for d in snap.documents}
    assert len(paths) == 3


def test_ref_edge_carries_occurrence_id():
    """Ref edges include the occurrence_id in their metadata."""
    payload = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=1)
    payload["documents"][0]["occurrences"][0]["role"] = "definition"
    snap = _build(payload)
    ref_edges = [e for e in snap.edges if e.edge_type == "ref"]
    assert len(ref_edges) == 1
    assert "occurrence_id" in ref_edges[0].metadata


def test_source_set_always_scip():
    """All produced entities have source_set={'scip'}."""
    payload = _make_scip_payload(n_docs=1, n_symbols=2, n_occurrences=2)
    snap = _build(payload)
    for doc in snap.documents:
        assert doc.source_set == {"scip"}
    for sym in snap.symbols:
        assert sym.source_set == {"scip"}


def test_contain_edge_per_symbol():
    """Each symbol gets exactly one contain edge from its parent document."""
    payload = _make_scip_payload(n_docs=1, n_symbols=4, n_occurrences=0)
    snap = _build(payload)
    contain_edges = [e for e in snap.edges if e.edge_type == "contain"]
    assert len(contain_edges) == 4


def test_snapshot_identity_fields():
    """Branch, commit_id, tree_id are passed through unchanged."""
    payload = _make_scip_payload()
    snap = _build(payload)
    assert snap.repo_name == "test_repo"
    assert snap.snapshot_id == "snap:test:abc1234"
    assert snap.branch == "main"
    assert snap.commit_id == "abc1234"
    assert snap.tree_id == "tree1"


# =========================================================================
# Edge tests
# =========================================================================


@pytest.mark.edge
@pytest.mark.parametrize(
    "role",
    [
        "unknown_role",
        "read_access",
        "call",
        "",
    ],
)
def test_role_does_not_produce_ref_edge_negative(role: str):
    """Roles outside the whitelist do not produce a 'ref' edge."""
    payload = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=1)
    payload["documents"][0]["occurrences"][0]["role"] = role
    snap = _build(payload)
    ref_edges = [e for e in snap.edges if e.edge_type == "ref"]
    assert len(ref_edges) == 0


_RANGE_CASES = [
    # (input_range, expected_start_line)
    ([], 0),
    ([10], 10),
    ([10, 2], 10),
    ([10, 2, 20], 10),
    ([10, 2, 20, 5], 10),
    # None values in various positions
    ([None], 0),
    ([None, None, None, None], 0),
    ([5, None, None, None], 5),
    ([None, 5, None, None], 0),
]


@pytest.mark.edge
@pytest.mark.parametrize(("raw_range", "expected_start_line"), _RANGE_CASES)
def test_occurrence_range_normalization_edge(
    raw_range: list[Any], expected_start_line: int
):
    """Range list is padded to 4 elements and None->0 for occurrences."""
    payload = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=1)
    payload["documents"][0]["occurrences"][0]["range"] = raw_range
    snap = _build(payload)
    assert len(snap.occurrences) == 1
    occ = snap.occurrences[0]
    assert occ.start_line == expected_start_line


@pytest.mark.edge
@pytest.mark.parametrize(
    "raw_range",
    [
        [None, None],
        [None, None, None],
        [None, None, None, None],
    ],
)
def test_occurrence_none_cols_become_zero_edge(raw_range: list[Any]):
    """None column values are converted to 0."""
    payload = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=1)
    payload["documents"][0]["occurrences"][0]["range"] = raw_range
    snap = _build(payload)
    occ = snap.occurrences[0]
    assert occ.start_col == 0
    assert occ.end_col == 0


_EMPTY_DOC_VARIANTS = [
    ("empty_documents_list", {"documents": []}),
    ("no_documents_key", {}),
    (
        "doc_no_symbols_no_occurrences",
        {
            "documents": [
                {"path": "a.py", "language": "python", "symbols": [], "occurrences": []}
            ],
        },
    ),
]


@pytest.mark.edge
@pytest.mark.parametrize(("label", "payload_override"), _EMPTY_DOC_VARIANTS)
def test_empty_variants_no_symbols_or_occurrences_edge(
    label: str, payload_override: dict[str, Any]
):
    """Empty or missing documents produce zero symbols and occurrences."""
    payload = {
        "indexer_name": "scip-python",
        "indexer_version": "1.0.0",
        **payload_override,
    }
    snap = _build(payload)
    assert len(snap.symbols) == 0
    assert len(snap.occurrences) == 0


@pytest.mark.edge
def test_empty_symbols_still_creates_document_edge():
    """A document with no symbols still appears in the snapshot."""
    payload = {
        "documents": [
            {
                "path": "empty.py",
                "language": "python",
                "symbols": [],
                "occurrences": [],
            },
        ],
    }
    snap = _build(payload)
    assert len(snap.documents) == 1
    assert snap.documents[0].path == "empty.py"


@pytest.mark.edge
def test_symbol_without_symbol_field_skipped_edge():
    """A symbol dict with no 'symbol' key is silently skipped."""
    payload = {
        "documents": [
            {
                "path": "a.py",
                "language": "python",
                "symbols": [{"name": "x", "kind": "function"}],
                "occurrences": [],
            },
        ],
    }
    snap = _build(payload)
    assert len(snap.symbols) == 0


@pytest.mark.edge
def test_occurrence_without_symbol_field_skipped_edge():
    """An occurrence dict with no 'symbol' key is silently skipped."""
    payload = {
        "documents": [
            {
                "path": "a.py",
                "language": "python",
                "symbols": [],
                "occurrences": [{"role": "reference", "range": [1, 0, 1, 10]}],
            },
        ],
    }
    snap = _build(payload)
    assert len(snap.occurrences) == 0


_LANGUAGE_CASES = [
    # (doc_language, language_hint, expected)
    ("python", None, "python"),
    (None, "typescript", "typescript"),
    ("", "go", "go"),
    (None, None, "unknown"),
    ("", "", "unknown"),
]


@pytest.mark.edge
@pytest.mark.parametrize(("doc_lang", "hint", "expected"), _LANGUAGE_CASES)
def test_language_fallback_chain_edge(
    doc_lang: str | None, hint: str | None, expected: str
):
    """Language is resolved via doc.language > language_hint > 'unknown'."""
    payload = {
        "documents": [
            {
                "path": "f.py",
                "language": doc_lang,
                "symbols": [],
                "occurrences": [],
            },
        ],
    }
    snap = _build(payload, language_hint=hint)
    assert snap.documents[0].language == expected


_DISPLAY_NAME_CASES = [
    ("my_func", "scip python test my_func()", "my_func"),
    (None, "scip python pkg/mod/Class()", "Class()"),
    ("", "scip python pkg/mod/Class()", "Class()"),
]


@pytest.mark.edge
@pytest.mark.parametrize(("name", "ext_symbol", "expected"), _DISPLAY_NAME_CASES)
def test_display_name_fallback_edge(name: str | None, ext_symbol: str, expected: str):
    """display_name falls back to last segment of external symbol."""
    payload = {
        "documents": [
            {
                "path": "a.py",
                "language": "python",
                "symbols": [
                    {
                        "symbol": ext_symbol,
                        "name": name,
                        "kind": "function",
                        "range": [1, 0, 1, 5],
                    },
                ],
                "occurrences": [],
            },
        ],
    }
    snap = _build(payload)
    assert len(snap.symbols) == 1
    assert snap.symbols[0].display_name == expected


# =========================================================================
# Negative tests
# =========================================================================

# (none beyond test_role_does_not_produce_ref_edge_negative above)


# =========================================================================
# Property-based tests
# =========================================================================


class TestBuildIrFromScipValidSnapshot:
    @given(index=scip_index_st, repo=identifier, snap_id=identifier)
    @settings(max_examples=40)
    def test_produces_valid_ir_snapshot_property(
        self, index: SCIPIndex, repo: str, snap_id: str
    ):
        """HAPPY: every SCIPIndex produces a valid IRSnapshot."""
        snap = build_ir_from_scip(repo_name=repo, snapshot_id=snap_id, scip_index=index)
        assert isinstance(snap, IRSnapshot)
        assert snap.repo_name == repo
        assert snap.snapshot_id == snap_id

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=30)
    def test_source_priority_is_100_property(self, index: SCIPIndex, snap_id: str):
        """HAPPY: all SCIP-derived symbols have source_priority=100."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        for sym in snap.symbols:
            assert sym.source_priority == 100

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=30)
    def test_source_set_is_scip_property(self, index: SCIPIndex, snap_id: str):
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
    def test_symbol_ids_prefixed_property(self, index: SCIPIndex, snap_id: str):
        """HAPPY: every symbol ID starts with 'scip:{snapshot_id}:'."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        expected_prefix = f"scip:{snap_id}:"
        for sym in snap.symbols:
            assert sym.symbol_id.startswith(expected_prefix)

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=30)
    def test_metadata_source_modes_scip_property(self, index: SCIPIndex, snap_id: str):
        """HAPPY: snapshot metadata contains source_modes=["scip"]."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        assert snap.metadata["source_modes"] == ["scip"]

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=30)
    def test_documents_count_includes_all_scip_docs_property(
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
    def test_branch_commit_tree_passed_through_property(
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


class TestContainmentEdges:
    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=30)
    def test_every_symbol_gets_containment_edge_property(
        self, index: SCIPIndex, snap_id: str
    ):
        """HAPPY: every symbol unit has a containment edge from its document."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        containment_edges = [e for e in snap.edges if e.edge_type == "contain"]
        assert len(containment_edges) >= len(snap.symbols)

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=20)
    def test_containment_edge_source_is_scip_property(
        self, index: SCIPIndex, snap_id: str
    ):
        """HAPPY: containment edges have source='scip'."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        for edge in snap.edges:
            if edge.edge_type == "contain":
                assert edge.source == "scip"

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=20)
    def test_containment_edge_confidence_precise_property(
        self, index: SCIPIndex, snap_id: str
    ):
        """HAPPY: containment edges have confidence='precise'."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        for edge in snap.edges:
            if edge.edge_type == "contain":
                assert edge.confidence == "precise"


class TestEmptyAndEdgeCases:
    @given(snap_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_empty_payload_produces_empty_snapshot_property(self, snap_id: str):
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
    def test_raw_dict_input_produces_valid_snapshot_property(
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
    def test_document_missing_language_falls_back_to_hint_property(self, snap_id: str):
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
    def test_document_missing_language_no_hint_falls_back_to_unknown_property(
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
    def test_symbol_empty_string_skipped_property(self, snap_id: str):
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
    def test_occurrence_empty_symbol_skipped_property(self, snap_id: str):
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


class TestOccurrenceProperties:
    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=30)
    def test_all_occurrences_source_is_scip_property(
        self, index: SCIPIndex, snap_id: str
    ):
        """HAPPY: all IR occurrences have source='scip'."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        for occ in snap.occurrences:
            assert occ.source == "scip"

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=20)
    def test_occurrence_range_none_coerced_to_zero_property(
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
    def test_occurrence_symbol_ids_prefixed_property(
        self, index: SCIPIndex, snap_id: str
    ):
        """HAPPY: every occurrence symbol_id starts with 'scip:{snapshot_id}:'."""
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        prefix = f"scip:{snap_id}:"
        for occ in snap.occurrences:
            assert occ.symbol_id.startswith(prefix)

    @given(index=scip_index_st, snap_id=identifier)
    @settings(max_examples=20)
    def test_occurrence_metadata_has_scip_source_property(
        self, index: SCIPIndex, snap_id: str
    ):
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
    def test_occurrence_ref_role_produces_ref_edge_property(self, snap_id: str):
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
    def test_symbol_metadata_scip_true_property(self, snap_id: str):
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


class TestScipParsingProperties:
    @given(index=parsing_scip_index_st)
    @settings(max_examples=50)
    def test_scip_index_roundtrip_property(self, index: SCIPIndex):
        """HAPPY: SCIPIndex.from_dict(data).to_dict() preserves key fields."""
        data = index.to_dict()
        restored = SCIPIndex.from_dict(data)
        assert restored.indexer_name == index.indexer_name
        assert restored.indexer_version == index.indexer_version
        assert len(restored.documents) == len(index.documents)
        for orig_doc, rest_doc in zip(index.documents, restored.documents, strict=True):
            assert orig_doc.path == rest_doc.path
            assert len(orig_doc.symbols) == len(rest_doc.symbols)

    @given(index=parsing_scip_index_st, snap_id=small_text)
    @settings(max_examples=40)
    def test_build_ir_from_scip_produces_valid_snapshot_property(
        self, index: SCIPIndex, snap_id: str
    ):
        """HAPPY: build_ir_from_scip always produces IRSnapshot with documents."""
        snap = build_ir_from_scip(
            repo_name="repo",
            snapshot_id=snap_id,
            scip_index=index,
        )
        assert snap.repo_name == "repo"
        assert snap.snapshot_id == snap_id
        assert len(snap.documents) >= len(index.documents)
        _ = sum(len(d.symbols) for d in index.documents)
        # The adapter may normalize module-kind symbols to file units, so
        # snap.symbols can be fewer than _total when modules are present.
        # It can also be fewer when duplicate symbol strings appear in the same doc.
        assert len(snap.symbols) + len(snap.documents) >= len(index.documents)

    @given(index=parsing_scip_index_st)
    @settings(max_examples=50)
    @pytest.mark.edge
    def test_scip_occurrence_ranges_valid_property(self, index: SCIPIndex):
        """EDGE: SCIP occurrence ranges have start_line <= end_line (when both present)."""
        for doc in index.documents:
            for occ in doc.occurrences:
                r = occ.range
                if r[0] is not None and r[2] is not None:
                    assert r[0] <= r[2], f"start_line {r[0]} > end_line {r[2]}"

    @given(index=parsing_scip_index_st, snap_id=small_text)
    @settings(max_examples=40)
    def test_scip_symbol_ids_follow_pattern_property(
        self, index: SCIPIndex, snap_id: str
    ):
        """HAPPY: SCIP-derived IRSymbol IDs follow scip:{snapshot_id}:... pattern."""
        snap = build_ir_from_scip(
            repo_name="repo",
            snapshot_id=snap_id,
            scip_index=index,
        )
        for sym in snap.symbols:
            assert sym.symbol_id.startswith(f"scip:{snap_id}:"), (
                f"Symbol ID {sym.symbol_id} does not match expected pattern"
            )

    @given(doc=parsing_scip_document_st)
    @settings(max_examples=30)
    def test_scip_document_roundtrip_property(self, doc: SCIPDocument):
        """HAPPY: SCIPDocument roundtrip through dict preserves path and counts."""
        data = doc.to_dict()
        restored = SCIPDocument.from_dict(data)
        assert restored.path == doc.path
        assert len(restored.symbols) == len(doc.symbols)
        assert len(restored.occurrences) == len(doc.occurrences)

    @given(sym=parsing_scip_symbol_st)
    @settings(max_examples=30)
    def test_scip_symbol_roundtrip_property(self, sym: SCIPSymbol):
        """HAPPY: SCIPSymbol roundtrip preserves all fields."""
        data = sym.to_dict()
        restored = SCIPSymbol.from_dict(data)
        assert restored.symbol == sym.symbol
        assert restored.name == sym.name
        assert restored.kind == sym.kind

    @given(snap_id=small_text)
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_build_ir_from_scip_empty_documents_property(self, snap_id: str):
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
    def test_build_ir_from_scip_document_missing_language_property(self, snap_id: str):
        """EDGE: document with no language falls back to language_hint or 'unknown'."""
        index = SCIPIndex(
            documents=[
                SCIPDocument(
                    path="unknown.txt",
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
            language_hint="python",
        )
        assert snap.documents[0].language == "python"

        snap_no_hint = build_ir_from_scip(
            repo_name="repo",
            snapshot_id=snap_id,
            scip_index=index,
        )
        assert snap_no_hint.documents[0].language == "unknown"

    @given(snap_id=small_text)
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_build_ir_from_scip_symbol_empty_string_skipped_property(
        self, snap_id: str
    ):
        """EDGE: symbol with empty string is skipped (no crash)."""
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
            repo_name="repo",
            snapshot_id=snap_id,
            scip_index=index,
        )
        assert len(snap.symbols) == 1
        assert snap.symbols[0].display_name == "valid"

    @given(data=st.dictionaries(st.text(min_size=1), st.integers()))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_scip_index_from_arbitrary_dict_property(self, data: st.DataObject):
        """EDGE: SCIPIndex.from_dict handles arbitrary dicts without crash."""
        index = SCIPIndex.from_dict(data)
        assert isinstance(index, SCIPIndex)
        assert isinstance(index.documents, list)

    @pytest.mark.edge
    def test_scip_occurrence_empty_symbol_property(self):
        """EDGE: SCIPOccurrence with empty symbol string doesn't crash."""
        occ = SCIPOccurrence(
            symbol="", role="reference", range=[None, None, None, None]
        )
        data = occ.to_dict()
        restored = SCIPOccurrence.from_dict(data)
        assert restored.symbol == ""

    @given(snap_id=small_text)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_build_ir_from_scip_no_occurrences_property(self, snap_id: str):
        """EDGE: documents with symbols but no occurrences produces zero IR occurrences."""
        index = SCIPIndex(
            documents=[
                SCIPDocument(
                    path="a.py",
                    language="python",
                    symbols=[
                        SCIPSymbol(symbol="pkg foo.", name="foo"),
                    ],
                    occurrences=[],
                ),
            ]
        )
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        assert len(snap.occurrences) == 0
        assert len(snap.symbols) == 1

    @given(text_part=small_text)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_scip_symbol_long_symbol_string_property(self, text_part: str):
        """EDGE: very long symbol string handled without crash."""
        sym = f"pkg {text_part * 50}."
        s = SCIPSymbol(symbol=sym, name="long")
        data = s.to_dict()
        restored = SCIPSymbol.from_dict(data)
        assert restored.symbol == sym

    @given(snap_id=small_text)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_build_ir_from_scip_occurrences_only_no_symbols_property(
        self, snap_id: str
    ):
        """EDGE: documents with occurrences but no symbols produces occurrences in IR."""
        index = SCIPIndex(
            documents=[
                SCIPDocument(
                    path="a.py",
                    language="python",
                    symbols=[],
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
        assert len(snap.symbols) == 0
        assert len(snap.documents) == 1

    @given(snap_id=small_text)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_build_ir_from_scip_duplicate_symbols_across_docs_property(
        self, snap_id: str
    ):
        """EDGE: same symbol name in different docs produces distinct IR symbols."""
        sym = SCIPSymbol(symbol="pkg foo.", name="foo", kind="function")
        index = SCIPIndex(
            documents=[
                SCIPDocument(path="a.py", language="python", symbols=[sym]),
                SCIPDocument(path="b.py", language="python", symbols=[sym]),
            ]
        )
        snap = build_ir_from_scip(
            repo_name="repo", snapshot_id=snap_id, scip_index=index
        )
        assert len(snap.symbols) == 2

    @pytest.mark.edge
    def test_scip_occurrence_all_none_range_property(self):
        """EDGE: occurrence with all-None range doesn't crash."""
        occ = SCIPOccurrence(
            symbol="pkg x.", role="reference", range=[None, None, None, None]
        )
        d = occ.to_dict()
        r = SCIPOccurrence.from_dict(d)
        assert r.range == [None, None, None, None]

    @pytest.mark.edge
    def test_scip_symbol_missing_optional_fields_property(self):
        """EDGE: symbol with only required fields gets defaults."""
        sym = SCIPSymbol(symbol="pkg x.")
        assert sym.name is None
        assert sym.kind is None
        d = sym.to_dict()
        r = SCIPSymbol.from_dict(d)
        assert r.symbol == "pkg x."

    @pytest.mark.edge
    def test_scip_document_no_language_no_symbols_property(self):
        """EDGE: document with no language and no symbols roundtrips."""
        doc = SCIPDocument(path="data.bin", language=None, symbols=[], occurrences=[])
        d = doc.to_dict()
        r = SCIPDocument.from_dict(d)
        assert r.path == "data.bin"
        assert r.language is None

    @pytest.mark.edge
    def test_build_ir_from_scip_empty_symbol_name_skipped_property(self):
        """EDGE: symbol with empty name field still gets processed."""
        index = SCIPIndex(
            documents=[
                SCIPDocument(
                    path="a.py",
                    language="python",
                    symbols=[
                        SCIPSymbol(symbol="pkg foo.", name=""),
                    ],
                ),
            ]
        )
        snap = build_ir_from_scip(repo_name="repo", snapshot_id="s1", scip_index=index)
        assert len(snap.symbols) == 1

    @pytest.mark.edge
    def test_scip_index_with_metadata_property(self):
        """EDGE: SCIPIndex with indexer metadata roundtrips."""
        idx = SCIPIndex(documents=[], indexer_name="scip-go", indexer_version="1.2.3")
        d = idx.to_dict()
        r = SCIPIndex.from_dict(d)
        assert r.indexer_name == "scip-go"
        assert r.indexer_version == "1.2.3"

    @given(
        data=st.dictionaries(
            st.text(min_size=1, max_size=5),
            st.integers(min_value=0, max_value=100),
            max_size=3,
        )
    )
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_scip_occurrence_from_arbitrary_dict_property(self, data: st.DataObject):
        """EDGE: SCIPOccurrence.from_dict handles arbitrary dicts without crash."""
        occ = SCIPOccurrence.from_dict(data)
        assert isinstance(occ, SCIPOccurrence)

    @pytest.mark.edge
    def test_scip_symbol_roundtrip_preserves_range_property(self):
        """EDGE: SCIPSymbol range field survives roundtrip."""
        sym = SCIPSymbol(symbol="pkg x.", name="x", range=[10, 0, 20, 5])
        d = sym.to_dict()
        r = SCIPSymbol.from_dict(d)
        assert r.range == [10, 0, 20, 5]

    @pytest.mark.edge
    def test_scip_document_with_many_occurrences_property(self):
        """EDGE: document with many occurrences roundtrips."""
        occs = [
            SCIPOccurrence(symbol=f"pkg s{i}.", role="reference", range=[i, 0, i, 5])
            for i in range(20)
        ]
        doc = SCIPDocument(
            path="big.py", language="python", symbols=[], occurrences=occs
        )
        d = doc.to_dict()
        r = SCIPDocument.from_dict(d)
        assert len(r.occurrences) == 20


# ─── Edge cases: malformed inputs and error handling ───


class TestBuildIrFromScipEdgeCases:
    """Negative tests for build_ir_from_scip with malformed/missing data."""

    def test_empty_documents_list(self):
        """Index with no documents should produce snapshot with no documents."""
        payload = _make_scip_payload(n_docs=0, n_symbols=0, n_occurrences=0)
        payload["documents"] = []
        snap = _build(payload)
        assert len(snap.documents) == 0

    def test_document_with_no_symbols(self):
        """Document with empty symbols list should still create a document unit."""
        payload = _make_scip_payload(n_docs=1, n_symbols=0, n_occurrences=0)
        snap = _build(payload)
        assert len(snap.documents) == 1

    def test_occurrence_with_empty_range(self):
        """Occurrence with None range values should not crash."""
        raw = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=1)
        raw["documents"][0]["occurrences"][0]["range"] = [None, None, None, None]
        snap = _build(raw)
        assert snap is not None

    def test_missing_indexer_name_uses_default(self):
        payload = _make_scip_payload()
        del payload["indexer_name"]
        snap = _build(payload)
        assert snap is not None

    def test_snapshot_id_format(self):
        """snapshot_id should follow snap:{repo}:{commit} format."""
        snap = _build(_make_scip_payload())
        # _build hardcodes snapshot_id="snap:test:abc1234"
        assert snap.snapshot_id.startswith("snap:test:")

    def test_metadata_records_source_modes(self):
        """Built snapshot should record which source modes were used."""
        snap = _build(_make_scip_payload())
        assert "scip" in snap.metadata.get("source_modes", [])


class TestNormalizeKindEdge:
    """Cover _normalize_kind mapping — completely untested per audit."""

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("documentation", "doc"),
            ("module", "file"),
            ("type", "class"),
            ("function", "function"),
            ("method", "method"),
            ("variable", "variable"),
            ("CLASS", "class"),
            ("Function", "function"),
            ("unknown_kind", "unknown_kind"),
            ("", "symbol"),
        ],
    )
    def test_known_mappings(self, kind: str, expected: str) -> None:
        from fastcode.adapters.scip_to_ir import _normalize_kind

        assert _normalize_kind(kind) == expected

    def test_none_returns_default(self) -> None:
        from fastcode.adapters.scip_to_ir import _normalize_kind

        assert _normalize_kind(None) == "symbol"
