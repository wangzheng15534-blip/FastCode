"""Shared hypothesis strategies for FastCode IR models."""

from __future__ import annotations

from hypothesis import strategies as st

from fastcode.semantic_ir import IRDocument, IREdge, IROccurrence, IRSnapshot, IRSymbol

# --- Primitive strategies ---

identifier = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=40,
)

file_path_st = st.tuples(identifier, identifier).map(lambda t: f"{t[0]}/{t[1]}.py")

role_st = st.sampled_from(["definition", "reference", "import", "implementation"])

edge_type_st = st.sampled_from(["dependency", "call", "inheritance", "reference", "contain"])

source_st = st.sampled_from(["ast", "scip"])

kind_st = st.sampled_from(
    ["function", "method", "class", "variable", "module", "interface", "enum", "constant"]
)

language_st = st.sampled_from(["python", "javascript", "typescript", "go", "java", "rust", "c", "cpp"])

line_number_st = st.integers(min_value=1, max_value=10000)

# --- Model strategies ---

ir_document_st = st.builds(
    IRDocument,
    doc_id=st.builds(lambda x: f"doc:{x}", identifier),
    path=file_path_st,
    language=language_st,
    blob_oid=st.none() | st.text(alphabet="0123456789abcdef", min_size=40, max_size=40),
    content_hash=st.none() | st.text(alphabet="0123456789abcdef", min_size=40, max_size=40),
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
        commit_id=st.none() | st.text(alphabet="0123456789abcdef", min_size=7, max_size=40),
        tree_id=st.none() | identifier,
        documents=st.lists(ir_document_st, max_size=max_docs),
        symbols=st.lists(ir_symbol_st, max_size=max_syms),
        occurrences=st.lists(ir_occurrence_st, max_size=max_occs),
        edges=st.lists(ir_edge_st, max_size=max_edges),
        metadata=st.dictionaries(
            st.sampled_from(["source_modes", "version", "tool"]),
            st.one_of(st.just(["ast"]), st.just(["scip"]), st.just(1), st.just("fastcode")),
        ),
    )
