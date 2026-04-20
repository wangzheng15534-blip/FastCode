"""
Parametrized tests for fastcode/adapters/scip_to_ir.py.

Covers: role-based ref edges, range normalization, language fallback,
display name fallback, source priority, metadata fields, and dual input types.
"""

from __future__ import annotations

import pytest

from fastcode.adapters.scip_to_ir import build_ir_from_scip
from fastcode.scip_models import SCIPDocument, SCIPIndex, SCIPOccurrence, SCIPSymbol


# ---------------------------------------------------------------------------
# Inline factory (mirrors tests.conftest._make_scip_payload)
# ---------------------------------------------------------------------------

def _make_scip_payload(
    n_docs: int = 1,
    n_symbols: int = 2,
    n_occurrences: int = 1,
) -> dict:
    """Create a raw dict payload matching build_ir_from_scip() input schema."""
    documents = []
    for i in range(n_docs):
        syms = []
        for j in range(n_symbols):
            syms.append({
                "symbol": f"scip python test func_{j}()",
                "name": f"func_{j}",
                "kind": "function",
                "range": [j + 1, 0, j + 1, 20],
            })
        occs = []
        for k in range(n_occurrences):
            occs.append({
                "symbol": f"scip python test func_{k % n_symbols}()",
                "role": "definition",
                "range": [k + 1, 0, k + 1, 20],
            })
        documents.append({
            "path": f"src/file{i}.py",
            "language": "python",
            "symbols": syms,
            "occurrences": occs,
        })
    return {
        "documents": documents,
        "indexer_name": "scip-python",
        "indexer_version": "1.0.0",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(
    payload: dict | SCIPIndex | None = None,
    *,
    language_hint: str | None = None,
) -> "IRSnapshot":
    """Thin wrapper around build_ir_from_scip with sensible defaults."""
    from fastcode.semantic_ir import IRSnapshot

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


# ---------------------------------------------------------------------------
# 1. Roles that produce ref edges vs not
# ---------------------------------------------------------------------------

_REF_ROLES = [
    "reference",
    "definition",
    "implementation",
    "type_definition",
    "import",
    "write_access",
    "forward_definition",
]

_NO_REF_ROLES = [
    "unknown_role",
    "read_access",
    "call",
    "",
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


@pytest.mark.parametrize("role", _NO_REF_ROLES)
def test_role_does_not_produce_ref_edge(role: str):
    """Roles outside the whitelist do not produce a 'ref' edge."""
    payload = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=1)
    payload["documents"][0]["occurrences"][0]["role"] = role
    snap = _build(payload)
    ref_edges = [e for e in snap.edges if e.edge_type == "ref"]
    assert len(ref_edges) == 0


# ---------------------------------------------------------------------------
# 2. Range normalization (padding, None→0)
# ---------------------------------------------------------------------------

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


@pytest.mark.parametrize("raw_range,expected_start_line", _RANGE_CASES)
def test_occurrence_range_normalization(raw_range: list, expected_start_line: int):
    """Range list is padded to 4 elements and None→0 for occurrences."""
    payload = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=1)
    payload["documents"][0]["occurrences"][0]["range"] = raw_range
    snap = _build(payload)
    assert len(snap.occurrences) == 1
    occ = snap.occurrences[0]
    assert occ.start_line == expected_start_line


@pytest.mark.parametrize("raw_range", [
    [None, None],
    [None, None, None],
    [None, None, None, None],
])
def test_occurrence_none_cols_become_zero(raw_range: list):
    """None column values are converted to 0."""
    payload = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=1)
    payload["documents"][0]["occurrences"][0]["range"] = raw_range
    snap = _build(payload)
    occ = snap.occurrences[0]
    assert occ.start_col == 0
    assert occ.end_col == 0


# ---------------------------------------------------------------------------
# 3. Empty documents variants
# ---------------------------------------------------------------------------

_EMPTY_DOC_VARIANTS = [
    ("empty_documents_list", {"documents": []}),
    ("no_documents_key", {}),
    ("doc_no_symbols_no_occurrences", {
        "documents": [{"path": "a.py", "language": "python", "symbols": [], "occurrences": []}],
    }),
]


@pytest.mark.parametrize("label,payload_override", _EMPTY_DOC_VARIANTS)
def test_empty_variants_no_symbols_or_occurrences(label: str, payload_override: dict):
    """Empty or missing documents produce zero symbols and occurrences."""
    payload = {"indexer_name": "scip-python", "indexer_version": "1.0.0", **payload_override}
    snap = _build(payload)
    assert len(snap.symbols) == 0
    assert len(snap.occurrences) == 0


def test_empty_symbols_still_creates_document():
    """A document with no symbols still appears in the snapshot."""
    payload = {
        "documents": [
            {"path": "empty.py", "language": "python", "symbols": [], "occurrences": []},
        ],
    }
    snap = _build(payload)
    assert len(snap.documents) == 1
    assert snap.documents[0].path == "empty.py"


def test_symbol_without_symbol_field_skipped():
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


def test_occurrence_without_symbol_field_skipped():
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


# ---------------------------------------------------------------------------
# 4. Language fallback chain
# ---------------------------------------------------------------------------

_LANGUAGE_CASES = [
    # (doc_language, language_hint, expected)
    ("python", None, "python"),
    (None, "typescript", "typescript"),
    ("", "go", "go"),
    (None, None, "unknown"),
    ("", "", "unknown"),
]


@pytest.mark.parametrize("doc_lang,hint,expected", _LANGUAGE_CASES)
def test_language_fallback_chain(doc_lang: str | None, hint: str | None, expected: str):
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


# ---------------------------------------------------------------------------
# 5. Display name fallback
# ---------------------------------------------------------------------------

_DISPLAY_NAME_CASES = [
    ("my_func", "scip python test my_func()", "my_func"),
    (None, "scip python pkg/mod/Class()", "Class()"),
    ("", "scip python pkg/mod/Class()", "Class()"),
]


@pytest.mark.parametrize("name,ext_symbol,expected", _DISPLAY_NAME_CASES)
def test_display_name_fallback(name: str | None, ext_symbol: str, expected: str):
    """display_name falls back to last segment of external symbol."""
    payload = {
        "documents": [
            {
                "path": "a.py",
                "language": "python",
                "symbols": [
                    {"symbol": ext_symbol, "name": name, "kind": "function", "range": [1, 0, 1, 5]},
                ],
                "occurrences": [],
            },
        ],
    }
    snap = _build(payload)
    assert len(snap.symbols) == 1
    assert snap.symbols[0].display_name == expected


# ---------------------------------------------------------------------------
# 6. Source priority constant
# ---------------------------------------------------------------------------

def test_all_scip_symbols_have_priority_100():
    """Every SCIP-derived symbol has source_priority == 100."""
    payload = _make_scip_payload(n_docs=2, n_symbols=3, n_occurrences=0)
    snap = _build(payload)
    assert len(snap.symbols) == 6  # 2 docs * 3 symbols
    for sym in snap.symbols:
        assert sym.source_priority == 100


# ---------------------------------------------------------------------------
# 7. Metadata fields present
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 8. SCIPIndex vs dict input both work
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

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
