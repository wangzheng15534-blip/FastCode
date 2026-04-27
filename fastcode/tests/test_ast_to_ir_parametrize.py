"""
Parametrized tests for fastcode/adapters/ast_to_ir.py.

Covers: element type filtering, qualified name construction, source set,
line clamping, source priority, metadata fields, and empty input.
"""

from __future__ import annotations

from typing import Any

import pytest

from fastcode.adapters.ast_to_ir import build_ir_from_ast
from fastcode.indexer import CodeElement
from fastcode.semantic_ir import IRSnapshot

# ---------------------------------------------------------------------------
# Inline factory (mirrors tests.conftest._make_code_elements)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _elem(
    *,
    type: str = "function",
    name: str = "my_func",
    start_line: int = 1,
    end_line: int = 5,
    relative_path: str = "src/a.py",
    language: str = "python",
    summary: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CodeElement:
    """Create a single CodeElement with sensible defaults."""
    return CodeElement(
        id=f"elem_{name}",
        type=type,
        name=name,
        file_path=relative_path,
        relative_path=relative_path,
        language=language,
        start_line=start_line,
        end_line=end_line,
        code=f"def {name}(): pass",
        signature=None,
        docstring=None,
        summary=summary,
        metadata=metadata or {},
    )


def _build(elements: list[CodeElement] | None = None) -> IRSnapshot:
    """Build an IRSnapshot from elements with default kwargs."""
    if elements is None:
        elements = _make_code_elements()
    return build_ir_from_ast(
        repo_name="test_repo",
        snapshot_id="snap:test:abc1234",
        elements=elements,
        repo_root="/tmp/repo",
        branch="main",
        commit_id="abc1234",
        tree_id="tree1",
    )


# ---------------------------------------------------------------------------
# 1. Element types that produce vs skip symbols
# ---------------------------------------------------------------------------

_SYMBOL_TYPES = ["function", "class", "variable", "method", "interface", "enum"]

_SKIP_TYPES = ["file", "documentation"]


@pytest.mark.basic
@pytest.mark.parametrize("etype", _SYMBOL_TYPES)
def test_type_produces_symbol(etype: str):
    """Non-file/documentation types produce an IRSymbol."""
    elements = [_elem(type=etype, name=f"item_{etype}")]
    snap = _build(elements)
    assert len(snap.symbols) == 1
    assert snap.symbols[0].kind == etype


@pytest.mark.basic
@pytest.mark.parametrize("etype", _SKIP_TYPES)
def test_type_skips_symbol(etype: str):
    """'file' and 'documentation' types produce no symbol."""
    elements = [_elem(type=etype, name=f"item_{etype}")]
    snap = _build(elements)
    assert len(snap.symbols) == 0


@pytest.mark.basic
@pytest.mark.parametrize("etype", _SKIP_TYPES)
def test_type_skips_symbol_but_still_creates_document(etype: str):
    """Skipped types still create a document for the file path."""
    elements = [_elem(type=etype, name=f"item_{etype}", relative_path="src/x.py")]
    snap = _build(elements)
    assert len(snap.symbols) == 0
    assert len(snap.documents) == 1
    assert snap.documents[0].path == "src/x.py"


# ---------------------------------------------------------------------------
# 2. Qualified name with/without class_name
# ---------------------------------------------------------------------------

_QUALIFIED_NAME_CASES = [
    # (metadata, expected_qualified_name)
    ({}, "my_func"),
    ({"class_name": None}, "my_func"),
    ({"class_name": "MyClass"}, "MyClass.my_func"),
    ({"class_name": ""}, "my_func"),
]


@pytest.mark.edge
@pytest.mark.parametrize(("meta", "expected_qname"), _QUALIFIED_NAME_CASES)
def test_qualified_name_construction(meta: dict[str, Any], expected_qname: str):
    """qualified_name uses class_name.name when class_name is truthy."""
    elements = [_elem(name="my_func", metadata=meta)]
    snap = _build(elements)
    assert len(snap.symbols) == 1
    assert snap.symbols[0].qualified_name == expected_qname


def test_qualified_name_with_class():
    """When class_name is present, qualified_name is class.method."""
    elements = [
        _elem(type="method", name="do_stuff", metadata={"class_name": "Worker"}),
    ]
    snap = _build(elements)
    assert snap.symbols[0].qualified_name == "Worker.do_stuff"


def test_qualified_name_without_class():
    """When class_name is absent, qualified_name equals the element name."""
    elements = [_elem(type="function", name="standalone")]
    snap = _build(elements)
    assert snap.symbols[0].qualified_name == "standalone"


# ---------------------------------------------------------------------------
# 3. Source set always {"fc_structure"}
# ---------------------------------------------------------------------------


def test_source_set_on_symbols():
    """All structure-derived symbols have source_set={'fc_structure'}."""
    elements = _make_code_elements(n=4)
    snap = _build(elements)
    for sym in snap.symbols:
        assert sym.source_set == {"fc_structure"}


def test_source_set_on_documents():
    """All structure-derived documents have source_set={'fc_structure'}."""
    elements = _make_code_elements(n=4)
    snap = _build(elements)
    for doc in snap.documents:
        assert "fc_structure" in doc.source_set


# ---------------------------------------------------------------------------
# 4. Line clamping (start_line=0, None, -1, 1 → >= 1)
# ---------------------------------------------------------------------------

_LINE_CLAMP_CASES = [
    # (start_line, expected_occurrence_start_line)
    (0, 1),
    (1, 1),
    (5, 5),
    (100, 100),
]


@pytest.mark.edge
@pytest.mark.parametrize(("start_line", "expected"), _LINE_CLAMP_CASES)
def test_start_line_clamping(start_line: int, expected: int):
    """Occurrence start_line is clamped to >= 1 via max(... or 1, 1)."""
    elements = [_elem(name="f", start_line=start_line, end_line=start_line + 5)]
    snap = _build(elements)
    assert len(snap.occurrences) == 1
    assert snap.occurrences[0].start_line == expected


@pytest.mark.edge
def test_start_line_none_clamps_to_one():
    """start_line=None clamps to 1."""
    elem = _elem(name="f", start_line=10, end_line=20)
    elem.start_line = None  # type: ignore[assignment]
    snap = _build([elem])
    assert snap.occurrences[0].start_line == 1


@pytest.mark.edge
def test_start_line_negative_clamps_to_one():
    """Negative start_line would clamp to 1 (max(-1 or 1, 1) = max(-1, 1) = 1)."""
    elem = _elem(name="f", start_line=10, end_line=20)
    elem.start_line = -1  # type: ignore[assignment]
    snap = _build([elem])
    assert snap.occurrences[0].start_line == 1


# ---------------------------------------------------------------------------
# 5. Source priority always 50
# ---------------------------------------------------------------------------


def test_source_priority_constant():
    """All structure-derived symbols have source_priority == 50."""
    elements = _make_code_elements(n=5)
    snap = _build(elements)
    for sym in snap.symbols:
        assert sym.source_priority == 50


# ---------------------------------------------------------------------------
# 6. Metadata fields
# ---------------------------------------------------------------------------


def test_symbol_metadata_contains_ast_fields():
    """Symbol metadata includes ast_element_id, source, confidence, extractor."""
    elements = [_elem(type="function", name="compute", metadata={"key1": "val1"})]
    snap = _build(elements)
    meta = snap.symbols[0].metadata
    assert meta["ast_element_id"] == "elem_compute"
    assert meta["source"] == "fc_structure"
    assert meta["confidence"] == "resolved"
    assert meta["extractor"] == "fastcode.adapters.ast_to_ir"
    assert meta["key1"] == "val1"


def test_symbol_metadata_excludes_embedding_keys():
    """embedding and embedding_text are stripped from symbol metadata."""
    elements = [
        _elem(
            name="f",
            metadata={
                "embedding": [0.1, 0.2],
                "embedding_text": "code",
                "visible": True,
            },
        ),
    ]
    snap = _build(elements)
    meta = snap.symbols[0].metadata
    assert "embedding" not in meta
    assert "embedding_text" not in meta
    assert meta["visible"] is True


def test_embedding_attachment_created_from_metadata():
    """Embedding metadata is promoted into a first-class attachment."""
    elements = [
        _elem(
            name="embed_me",
            metadata={"embedding": [0.1, 0.2], "embedding_text": "vector text"},
        ),
    ]
    snap = _build(elements)
    assert len(snap.attachments) == 1
    attachment = snap.attachments[0]
    assert attachment.target_type == "symbol"
    assert attachment.attachment_type == "embedding"
    assert attachment.source == "fc_embedding"
    assert attachment.payload["vector"] == [0.1, 0.2]
    assert attachment.payload["text"] == "vector text"


def test_summary_attachment_created_from_element_summary():
    """Element summaries become summary attachments on the symbol."""
    elements = [
        _elem(name="summarized", summary="Important entry point"),
    ]
    snap = _build(elements)
    assert len(snap.attachments) == 1
    attachment = snap.attachments[0]
    assert attachment.target_type == "symbol"
    assert attachment.attachment_type == "summary"
    assert attachment.source == "fc_structure"
    assert attachment.payload["text"] == "Important entry point"


def test_occurrence_metadata_kind():
    """Occurrence metadata records the element kind."""
    elements = [_elem(type="class", name="Foo")]
    snap = _build(elements)
    assert snap.occurrences[0].metadata["kind"] == "class"


def test_snapshot_metadata_source_modes():
    """Snapshot metadata has source_modes=['fc_structure']."""
    snap = _build(_make_code_elements())
    assert snap.metadata["source_modes"] == ["fc_structure"]


def test_contain_edge_metadata():
    """Contain edges carry extractor and source metadata."""
    elements = [_elem(name="f")]
    snap = _build(elements)
    contain_edges = [e for e in snap.edges if e.edge_type == "contain"]
    assert len(contain_edges) == 1
    assert contain_edges[0].metadata["extractor"] == "fastcode.adapters.ast_to_ir"
    assert contain_edges[0].metadata["source"] == "fc_structure"


# ---------------------------------------------------------------------------
# 7. Empty elements list produces empty snapshot
# ---------------------------------------------------------------------------


@pytest.mark.edge
def test_empty_elements_no_symbols_or_docs():
    """An empty elements list produces an empty snapshot."""
    snap = _build([])
    assert len(snap.documents) == 0
    assert len(snap.symbols) == 0
    assert len(snap.occurrences) == 0
    assert len(snap.edges) == 0


def test_empty_elements_identity_fields():
    """Empty snapshot still carries repo_name, snapshot_id, etc."""
    snap = _build([])
    assert snap.repo_name == "test_repo"
    assert snap.snapshot_id == "snap:test:abc1234"
    assert snap.branch == "main"
    assert snap.commit_id == "abc1234"
    assert snap.tree_id == "tree1"


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_multiple_elements_same_file_one_document():
    """Multiple elements in the same file produce a single IRDocument."""
    elements = [
        _elem(name="f1", relative_path="src/mod.py"),
        _elem(name="f2", relative_path="src/mod.py"),
        _elem(name="f3", relative_path="src/mod.py"),
    ]
    snap = _build(elements)
    assert len(snap.documents) == 1
    assert len(snap.symbols) == 3


def test_multiple_files_multiple_documents():
    """Elements across different files produce separate documents."""
    elements = [
        _elem(name="a", relative_path="src/a.py"),
        _elem(name="b", relative_path="src/b.py"),
        _elem(name="c", relative_path="lib/c.py"),
    ]
    snap = _build(elements)
    assert len(snap.documents) == 3


def test_contain_edge_per_symbol():
    """Each symbol gets exactly one contain edge."""
    elements = _make_code_elements(n=4)
    snap = _build(elements)
    contain_edges = [e for e in snap.edges if e.edge_type == "contain"]
    assert len(contain_edges) == len(snap.symbols)


def test_occurrence_role_always_definition():
    """All structure-derived occurrences have role='definition'."""
    elements = _make_code_elements(n=3)
    snap = _build(elements)
    for occ in snap.occurrences:
        assert occ.role == "definition"


def test_occurrence_source_always_ast():
    """All structure-derived occurrences have source='fc_structure'."""
    elements = _make_code_elements(n=3)
    snap = _build(elements)
    for occ in snap.occurrences:
        assert occ.source == "fc_structure"


def test_class_element_creates_inheritance_edge_from_bases():
    """A class with 'bases' metadata produces inheritance edges."""
    elements = [
        _elem(type="class", name="Child", metadata={"bases": ["Parent"]}),
        _elem(type="class", name="Parent", metadata={}),
    ]
    snap = _build(elements)
    inherit_edges = [e for e in snap.edges if e.edge_type == "inherit"]
    assert len(inherit_edges) == 1
    assert inherit_edges[0].metadata["base"] == "Parent"


def test_file_element_with_imports_creates_import_edge():
    """A file element with 'imports' metadata can produce import edges."""
    elements = [
        _elem(type="file", name="module_a", relative_path="src/a.py", metadata={}),
        _elem(
            type="file",
            name="module_b",
            relative_path="src/b.py",
            metadata={"imports": [{"module": "src.a"}]},
        ),
    ]
    snap = _build(elements)
    import_edges = [e for e in snap.edges if e.edge_type == "import"]
    assert len(import_edges) == 1
    assert import_edges[0].metadata["module"] == "src.a"


@pytest.mark.edge
def test_no_self_import_edge():
    """A file importing itself does not produce an import edge."""
    elements = [
        _elem(
            type="file",
            name="mod",
            relative_path="src/mod.py",
            metadata={"imports": [{"module": "src.mod"}]},
        ),
    ]
    snap = _build(elements)
    import_edges = [e for e in snap.edges if e.edge_type == "import"]
    assert len(import_edges) == 0


@pytest.mark.edge
def test_language_fallback_to_unknown():
    """An element with empty language falls back to 'unknown'."""
    elem = _elem(name="f", language="")
    snap = _build([elem])
    assert snap.symbols[0].language == "unknown"


def test_symbol_id_is_deterministic():
    """Same element inputs always produce the same symbol_id."""
    elements = [
        _elem(name="compute", type="function", start_line=10, metadata={}),
    ]
    snap1 = _build(elements)
    snap2 = _build(elements)
    assert snap1.symbols[0].symbol_id == snap2.symbols[0].symbol_id
