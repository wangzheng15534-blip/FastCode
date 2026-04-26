"""Shared hypothesis strategies and fixtures for FastCode tests."""

from __future__ import annotations

import pathlib

import pytest
from hypothesis import strategies as st

from fastcode.db_runtime import DBRuntime
from fastcode.indexer import CodeElement
from fastcode.scip_models import SCIPDocument, SCIPIndex, SCIPOccurrence, SCIPSymbol
from fastcode.semantic_ir import IRDocument, IREdge, IROccurrence, IRSnapshot, IRSymbol

# --- Primitive strategies ---

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

source_st = st.sampled_from(["ast", "scip"])

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

# --- IR Model strategies ---

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
        metadata=st.dictionaries(
            st.sampled_from(["source_modes", "version", "tool"]),
            st.one_of(
                st.just(["ast"]), st.just(["scip"]), st.just(1), st.just("fastcode")
            ),
        ),
    )


# --- SCIP Model strategies ---

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


# --- CodeElement strategy ---

code_element_st = st.builds(
    CodeElement,
    id=st.builds(lambda x: f"elem_{x}", identifier),
    type=st.sampled_from(
        ["function", "class", "variable", "method", "file", "documentation"]
    ),
    name=identifier,
    file_path=file_path_st,
    relative_path=file_path_st,
    language=language_st,
    start_line=st.integers(min_value=1, max_value=1000),
    end_line=st.integers(min_value=1, max_value=1000),
    code=st.just("def foo(): pass"),
    signature=st.none() | st.just("def foo(x: int) -> str"),
    docstring=st.none() | st.just("A function."),
    summary=st.none() | st.just("Does stuff."),
    metadata=st.fixed_dictionaries(
        {},
        optional={
            "class_name": identifier,
            "start_col": st.integers(min_value=0, max_value=80),
            "imports": st.lists(
                st.fixed_dictionaries({"module": identifier}),
                max_size=2,
            ),
            "bases": st.lists(identifier, max_size=2),
        },
    ),
)


# --- Connected snapshot composite ---


@st.composite
def connected_snapshot_st(
    draw: st.DataObject,
    n_docs: int | None = None,
    n_symbols_per_doc: int | None = None,
):
    """Generate IRSnapshot where all references are valid (occurrences/edges point to real entities)."""
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
                source_set={"ast"},
            )
        )

        for j in range(ns):
            sym_id = f"sym:{draw(identifier)}"
            symbols.append(
                IRSymbol(
                    symbol_id=sym_id,
                    path=path,
                    display_name=f"func_{j}",
                    kind=draw(kind_st),
                    language=docs[-1].language,
                    source_priority=10,
                    source_set={"ast"},
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
                    source="ast",
                )
            )
            edges.append(
                IREdge(
                    edge_id=f"edge:{draw(identifier)}",
                    src_id=doc_id,
                    dst_id=sym_id,
                    edge_type="contain",
                    source="ast",
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
        metadata={"source_modes": ["ast"]},
    )


# --- Pytest fixtures ---


@pytest.fixture
def snapshot_store(tmp_path: pathlib.Path):
    """Create a SnapshotStore backed by a temp directory."""
    from fastcode.snapshot_store import SnapshotStore

    return SnapshotStore(str(tmp_path))


@pytest.fixture
def sqlite_runtime():
    """Create an in-memory SQLite DBRuntime."""
    return DBRuntime(backend="sqlite", sqlite_path=":memory:")


# --- Factory functions ---


def _make_snapshot(
    repo_name: str = "test_repo",
    commit_id: str = "abc1234",
    n_docs: int = 2,
    n_symbols: int = 3,
) -> IRSnapshot:
    """Create a deterministic connected snapshot for testing."""
    docs = []
    symbols = []
    occurrences = []
    edges = []
    for i in range(n_docs):
        doc_id = f"doc:file{i}"
        docs.append(
            IRDocument(
                doc_id=doc_id,
                path=f"src/file{i}.py",
                language="python",
                source_set={"ast"},
            )
        )
        for j in range(n_symbols):
            sym_id = f"sym:file{i}_func{j}"
            symbols.append(
                IRSymbol(
                    symbol_id=sym_id,
                    path=f"src/file{i}.py",
                    display_name=f"func_{j}",
                    kind="function",
                    language="python",
                    source_priority=10,
                    source_set={"ast"},
                    start_line=j + 1,
                )
            )
            occurrences.append(
                IROccurrence(
                    occurrence_id=f"occ:file{i}_func{j}",
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
        repo_name=repo_name,
        snapshot_id=f"snap:{repo_name}:{commit_id}",
        branch="main",
        commit_id=commit_id,
        documents=docs,
        symbols=symbols,
        occurrences=occurrences,
        edges=edges,
        metadata={"source_modes": ["ast"]},
    )


def _make_scip_payload(
    n_docs: int = 1,
    n_symbols: int = 2,
    n_occurrences: int = 1,
) -> dict:
    """Create a raw dict payload matching build_ir_from_scip() input schema."""
    documents = []
    for i in range(n_docs):
        symbols = []
        for j in range(n_symbols):
            symbols.append(
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
                "symbols": symbols,
                "occurrences": occs,
            }
        )
    return {
        "documents": documents,
        "indexer_name": "scip-python",
        "indexer_version": "1.0.0",
    }


def _make_code_elements(n: int = 3) -> list[CodeElement]:
    """Create a list of CodeElement instances for AST adapter tests."""
    elements = []
    for i in range(n):
        elements.append(
            CodeElement(
                id=f"elem_{i}",
                type="function" if i % 2 == 0 else "class",
                name=f"func_{i}" if i % 2 == 0 else f"Class_{i}",
                file_path=f"src/file{i % 2}.py",
                relative_path=f"src/file{i % 2}.py",
                language="python",
                start_line=i + 1,
                end_line=i + 5,
                code=f"def func_{i}(): pass",
                signature=None,
                docstring=None,
                summary=None,
                metadata={},
            )
        )
    return elements
