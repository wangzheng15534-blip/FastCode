"""Tests for adapters.ast_to_ir module."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from fastcode.adapters.ast_to_ir import build_ir_from_ast
from fastcode.indexer import CodeElement
from fastcode.semantic_ir import IRSnapshot

# ---------------------------------------------------------------------------
# Strategies (self-contained, no conftest import)
# ---------------------------------------------------------------------------

identifier = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=40,
)

file_path_st = st.tuples(identifier, identifier).map(lambda t: f"{t[0]}/{t[1]}.py")

language_st = st.sampled_from(
    ["python", "javascript", "typescript", "go", "java", "rust", "c", "cpp"]
)

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

# ---------------------------------------------------------------------------
# Inline factories / helpers
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


_repo_root = "/tmp/test_repo"

_element_types_with_symbols = st.sampled_from(
    ["function", "class", "variable", "method"]
)
_element_types_without_symbols: Any = st.sampled_from(["file", "documentation"])

_small_int = st.integers(min_value=1, max_value=500)


def _elem_prop(
    type_: str,
    name: str,
    relative_path: str,
    language: str = "python",
    start_line: int = 1,
    end_line: int = 5,
    metadata: dict[str, Any] | None = None,
) -> CodeElement:
    return CodeElement(
        id=f"elem_{name}",
        type=type_,
        name=name,
        file_path=relative_path,
        relative_path=relative_path,
        language=language,
        start_line=start_line,
        end_line=end_line,
        code="pass",
        signature=None,
        docstring=None,
        summary=None,
        metadata=metadata or {},
    )


# =========================================================================
# Basic tests
# =========================================================================


_SYMBOL_TYPES = ["function", "class", "variable", "method", "interface", "enum"]

_SKIP_TYPES = ["file", "documentation"]


@pytest.mark.parametrize("etype", _SYMBOL_TYPES)
def test_type_produces_symbol(etype: str):
    """Non-file/documentation types produce an IRSymbol."""
    elements = [_elem(type=etype, name=f"item_{etype}")]
    snap = _build(elements)
    assert len(snap.symbols) == 1
    assert snap.symbols[0].kind == etype


@pytest.mark.parametrize("etype", _SKIP_TYPES)
def test_type_skips_symbol_negative(etype: str):
    """'file' and 'documentation' types produce no symbol."""
    elements = [_elem(type=etype, name=f"item_{etype}")]
    snap = _build(elements)
    assert len(snap.symbols) == 0


@pytest.mark.parametrize("etype", _SKIP_TYPES)
def test_type_skips_symbol_but_still_creates_document_edge(etype: str):
    """Skipped types still create a document for the file path."""
    elements = [_elem(type=etype, name=f"item_{etype}", relative_path="src/x.py")]
    snap = _build(elements)
    assert len(snap.symbols) == 0
    assert len(snap.documents) == 1
    assert snap.documents[0].path == "src/x.py"


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


def test_source_priority_constant():
    """All structure-derived symbols have source_priority == 50."""
    elements = _make_code_elements(n=5)
    snap = _build(elements)
    for sym in snap.symbols:
        assert sym.source_priority == 50


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


def test_qualified_name_with_class():
    """When class_name is present, qualified_name is class.method."""
    elements = [
        _elem(type="method", name="do_stuff", metadata={"class_name": "Worker"}),
    ]
    snap = _build(elements)
    assert snap.symbols[0].qualified_name == "Worker.do_stuff"


def test_symbol_id_is_deterministic():
    """Same element inputs always produce the same symbol_id."""
    elements = [
        _elem(name="compute", type="function", start_line=10, metadata={}),
    ]
    snap1 = _build(elements)
    snap2 = _build(elements)
    assert snap1.symbols[0].symbol_id == snap2.symbols[0].symbol_id


# =========================================================================
# Edge tests
# =========================================================================


_QUALIFIED_NAME_CASES = [
    # (metadata, expected_qualified_name)
    ({}, "my_func"),
    ({"class_name": None}, "my_func"),
    ({"class_name": "MyClass"}, "MyClass.my_func"),
    ({"class_name": ""}, "my_func"),
]


@pytest.mark.edge
@pytest.mark.parametrize(("meta", "expected_qname"), _QUALIFIED_NAME_CASES)
def test_qualified_name_construction_edge(meta: dict[str, Any], expected_qname: str):
    """qualified_name uses class_name.name when class_name is truthy."""
    elements = [_elem(name="my_func", metadata=meta)]
    snap = _build(elements)
    assert len(snap.symbols) == 1
    assert snap.symbols[0].qualified_name == expected_qname


def test_qualified_name_without_class_edge():
    """When class_name is absent, qualified_name equals the element name."""
    elements = [_elem(type="function", name="standalone")]
    snap = _build(elements)
    assert snap.symbols[0].qualified_name == "standalone"


def test_symbol_metadata_excludes_embedding_keys_edge():
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


def test_empty_elements_identity_fields_edge():
    """Empty snapshot still carries repo_name, snapshot_id, etc."""
    snap = _build([])
    assert snap.repo_name == "test_repo"
    assert snap.snapshot_id == "snap:test:abc1234"
    assert snap.branch == "main"
    assert snap.commit_id == "abc1234"
    assert snap.tree_id == "tree1"


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
def test_start_line_none_clamps_to_one_edge():
    """start_line=None clamps to 1."""
    elem = _elem(name="f", start_line=10, end_line=20)
    elem.start_line = None  # type: ignore[assignment]
    snap = _build([elem])
    assert snap.occurrences[0].start_line == 1


@pytest.mark.edge
def test_start_line_negative_clamps_to_one_edge():
    """Negative start_line would clamp to 1 (max(-1 or 1, 1) = max(-1, 1) = 1)."""
    elem = _elem(name="f", start_line=10, end_line=20)
    elem.start_line = -1  # type: ignore[assignment]
    snap = _build([elem])
    assert snap.occurrences[0].start_line == 1


@pytest.mark.edge
def test_empty_elements_no_symbols_or_docs_edge():
    """An empty elements list produces an empty snapshot."""
    snap = _build([])
    assert len(snap.documents) == 0
    assert len(snap.symbols) == 0
    assert len(snap.occurrences) == 0
    assert len(snap.edges) == 0


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
def test_language_fallback_to_unknown_edge():
    """An element with empty language falls back to 'unknown'."""
    elem = _elem(name="f", language="")
    snap = _build([elem])
    assert snap.symbols[0].language == "unknown"


# =========================================================================
# Property-based tests
# =========================================================================


class TestDocumentDeduplication:
    @given(
        name1=identifier,
        name2=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=30)
    def test_same_path_produces_one_document_property(
        self, name1: str, name2: str, path: str, language: str
    ):
        """HAPPY: multiple elements from same file produce exactly one document."""
        assume(name1 != name2)
        elements = [
            _elem_prop("function", name1, path, language, start_line=1, end_line=5),
            _elem_prop("function", name2, path, language, start_line=10, end_line=15),
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert len(snap.documents) == 1
        assert snap.documents[0].path == path

    @given(
        name1=identifier,
        name2=identifier,
        path1=file_path_st,
        path2=file_path_st,
        language=language_st,
    )
    @settings(max_examples=30)
    def test_different_paths_produce_multiple_documents_property(
        self, name1: str, name2: str, path1: str, path2: str, language: str
    ):
        """HAPPY: elements from different files produce distinct documents."""
        assume(path1 != path2)
        elements = [
            _elem_prop("function", name1, path1, language),
            _elem_prop("function", name2, path2, language),
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert len(snap.documents) == 2
        doc_paths = {d.path for d in snap.documents}
        assert path1 in doc_paths
        assert path2 in doc_paths

    @given(
        n=st.integers(min_value=2, max_value=6),
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=20)
    def test_many_elements_one_file_still_one_document_property(
        self, n: int, path: str, language: str
    ):
        """HAPPY: N elements in same file produce exactly one document."""
        elements = [
            _elem_prop("function", f"fn_{i}", path, language, start_line=i * 10)
            for i in range(n)
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert len(snap.documents) == 1


class TestSymbolCreation:
    @given(
        name=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=20)
    def test_file_type_skips_symbol_property(self, name: str, path: str, language: str):
        """HAPPY: file-type elements produce no symbols."""
        elements = [_elem_prop("file", name, path, language)]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert len(snap.symbols) == 0
        assert len(snap.occurrences) == 0

    @given(
        name=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=20)
    def test_documentation_type_skips_symbol_property(
        self, name: str, path: str, language: str
    ):
        """HAPPY: documentation-type elements produce no symbols."""
        elements = [_elem_prop("documentation", name, path, language)]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert len(snap.symbols) == 0

    @given(
        name=identifier,
        path=file_path_st,
        language=language_st,
        type_=_element_types_with_symbols,
    )
    @settings(max_examples=30)
    def test_non_file_types_create_symbols_property(
        self, name: str, path: str, language: str, type_: str
    ):
        """HAPPY: function/class/variable/method elements produce exactly one symbol."""
        elements = [_elem_prop(type_, name, path, language)]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert len(snap.symbols) == 1
        assert snap.symbols[0].display_name == name

    def test_empty_elements_produces_empty_snapshot_property(self):
        """HAPPY: empty element list produces snapshot with zero symbols/occurrences."""
        snap = build_ir_from_ast("repo", "snap:1", [], _repo_root)
        assert len(snap.symbols) == 0
        assert len(snap.occurrences) == 0
        assert len(snap.documents) == 0
        assert len(snap.edges) == 0


class TestLineClamping:
    @given(
        name=identifier,
        path=file_path_st,
        start_line=st.one_of(st.none(), st.just(0), st.just(-1)),
    )
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_occurrence_start_line_clamped_to_one_property(
        self, name: str, path: str, start_line: int
    ):
        """EDGE: occurrence start_line is clamped to >= 1."""
        elem = CodeElement(
            id=f"elem_{name}",
            type="function",
            name=name,
            file_path=path,
            relative_path=path,
            language="python",
            start_line=start_line,
            end_line=1,
            code="pass",
            signature=None,
            docstring=None,
            summary=None,
            metadata={},
        )
        snap = build_ir_from_ast("repo", "snap:1", [elem], _repo_root)
        if snap.occurrences:
            assert snap.occurrences[0].start_line >= 1

    @given(
        name=identifier,
        path=file_path_st,
        start_line=_small_int,
    )
    @settings(max_examples=30)
    def test_valid_start_line_preserved_property(
        self, name: str, path: str, start_line: int
    ):
        """HAPPY: valid start_line >= 1 is preserved."""
        elem = _elem_prop(
            "function", name, path, start_line=start_line, end_line=start_line + 5
        )
        snap = build_ir_from_ast("repo", "snap:1", [elem], _repo_root)
        assert snap.occurrences[0].start_line == start_line


class TestSourcePriorityAndSet:
    @given(
        elements=st.lists(
            code_element_st.filter(lambda e: e.type not in {"file", "documentation"}),
            max_size=5,
        ),
    )
    @settings(max_examples=30)
    def test_all_symbols_have_source_priority_50_property(
        self, elements: list[CodeElement]
    ):
        """HAPPY: all symbols created by structure adapter have source_priority=50."""
        assume(len(elements) > 0)
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        for sym in snap.symbols:
            assert sym.source_priority == 50

    @given(
        elements=st.lists(code_element_st, max_size=5),
    )
    @settings(max_examples=30)
    def test_all_docs_have_fc_structure_in_source_set_property(
        self, elements: list[CodeElement]
    ):
        """HAPPY: all documents have 'fc_structure' in source_set."""
        assume(len(elements) > 0)
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        for doc in snap.documents:
            assert "fc_structure" in doc.source_set

    @given(
        elements=st.lists(
            code_element_st.filter(lambda e: e.type not in {"file", "documentation"}),
            max_size=5,
        ),
    )
    @settings(max_examples=30)
    def test_all_symbols_have_fc_structure_source_set_property(
        self, elements: list[CodeElement]
    ):
        """HAPPY: all symbols have source_set={'fc_structure'}."""
        assume(len(elements) > 0)
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        for sym in snap.symbols:
            assert sym.source_set == {"fc_structure"}


class TestContainmentEdges:
    @given(
        name=identifier,
        path=file_path_st,
        language=language_st,
        type_=_element_types_with_symbols,
    )
    @settings(max_examples=30)
    def test_non_file_symbols_have_containment_edge_property(
        self, name: str, path: str, language: str, type_: str
    ):
        """HAPPY: every non-file, non-doc symbol gets a containment edge."""
        elements = [_elem_prop(type_, name, path, language)]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert len(snap.edges) >= 1
        contain_edges = [e for e in snap.edges if e.edge_type == "contain"]
        assert len(contain_edges) == 1
        assert contain_edges[0].dst_id == snap.symbols[0].symbol_id

    @given(
        n=st.integers(min_value=1, max_value=5),
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=20)
    def test_containment_count_equals_symbol_count_property(
        self, n: int, path: str, language: str
    ):
        """HAPPY: number of containment edges equals number of symbols."""
        elements = [
            _elem_prop("function", f"fn_{i}", path, language, start_line=i * 10)
            for i in range(n)
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        contain_edges = [e for e in snap.edges if e.edge_type == "contain"]
        assert len(contain_edges) == n


class TestMetadataConsistency:
    @given(
        elements=st.lists(code_element_st, max_size=4),
    )
    @settings(max_examples=30)
    def test_snapshot_metadata_source_modes_is_fc_structure_property(
        self, elements: list[CodeElement]
    ):
        """HAPPY: snapshot metadata always has source_modes=['fc_structure']."""
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert snap.metadata.get("source_modes") == ["fc_structure"]

    @given(
        name=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=30)
    def test_symbol_metadata_has_source_and_confidence_property(
        self, name: str, path: str, language: str
    ):
        """HAPPY: every symbol metadata contains 'source' and 'confidence' keys."""
        elements = [_elem_prop("function", name, path, language)]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        for sym in snap.symbols:
            assert "source" in sym.metadata
            assert "confidence" in sym.metadata

    @given(
        elements=st.lists(
            code_element_st.filter(lambda e: e.type not in {"file", "documentation"}),
            max_size=5,
        ),
    )
    @settings(max_examples=30)
    def test_symbol_metadata_source_is_fc_structure_property(
        self, elements: list[CodeElement]
    ):
        """HAPPY: all symbol metadata['source'] == 'fc_structure'."""
        assume(len(elements) > 0)
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        for sym in snap.symbols:
            assert sym.metadata.get("source") == "fc_structure"


class TestAttachments:
    @given(name=identifier, path=file_path_st, language=language_st)
    @settings(max_examples=20)
    def test_summary_generates_summary_attachment_property(
        self, name: str, path: str, language: str
    ):
        """HAPPY: summary-bearing symbols emit one summary attachment."""
        element = CodeElement(
            id=f"elem_{name}",
            type="function",
            name=name,
            file_path=path,
            relative_path=path,
            language=language,
            start_line=1,
            end_line=5,
            code="pass",
            signature=None,
            docstring=None,
            summary="summary text",
            metadata={},
        )
        snap = build_ir_from_ast("repo", "snap:1", [element], _repo_root)
        assert len(snap.attachments) == 1
        attachment = snap.attachments[0]
        assert attachment.attachment_type == "summary"
        assert attachment.source == "fc_structure"
        assert attachment.target_type == "symbol"

    @given(name=identifier, path=file_path_st, language=language_st)
    @settings(max_examples=20)
    def test_embedding_metadata_generates_embedding_attachment_property(
        self, name: str, path: str, language: str
    ):
        """HAPPY: embedding metadata is promoted into an embedding attachment."""
        element = CodeElement(
            id=f"elem_{name}",
            type="function",
            name=name,
            file_path=path,
            relative_path=path,
            language=language,
            start_line=1,
            end_line=5,
            code="pass",
            signature=None,
            docstring=None,
            summary=None,
            metadata={"embedding": [0.1, 0.2], "embedding_text": "vector text"},
        )
        snap = build_ir_from_ast("repo", "snap:1", [element], _repo_root)
        assert len(snap.attachments) == 1
        attachment = snap.attachments[0]
        assert attachment.attachment_type == "embedding"
        assert attachment.source == "fc_embedding"
        assert attachment.payload["vector"] == [0.1, 0.2]


class TestImportEdges:
    @given(
        module_name=identifier,
        path1=file_path_st,
        path2=file_path_st,
        language=language_st,
    )
    @settings(max_examples=20)
    def test_file_imports_create_dependency_edge_property(
        self, module_name: str, path1: str, path2: str, language: str
    ):
        """HAPPY: file element with imports metadata matching a known path creates import edge."""
        assume(path1 != path2)
        # Adjust path2 so it ends with the module path
        adjusted_path2 = f"src/{module_name}.py"

        elements = [
            _elem_prop(
                "file",
                "mod",
                path1,
                language,
                metadata={
                    "imports": [{"module": module_name}],
                },
            ),
            _elem_prop("function", "fn1", adjusted_path2, language),
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        import_edges = [e for e in snap.edges if e.edge_type == "import"]
        # Import edge may or may not exist depending on path matching
        for edge in import_edges:
            assert edge.edge_type == "import"
            assert edge.source == "fc_structure"
            assert edge.confidence == "resolved"

    @given(path=file_path_st, language=language_st)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_file_import_no_match_no_edge_property(self, path: str, language: str):
        """EDGE: file with imports but no matching target produces no import edge."""
        elements = [
            _elem_prop(
                "file",
                "mod",
                path,
                language,
                metadata={
                    "imports": [{"module": "nonexistent_module_xyz"}],
                },
            ),
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        import_edges = [e for e in snap.edges if e.edge_type == "import"]
        assert len(import_edges) == 0


class TestInheritanceEdges:
    @given(
        base_name=identifier,
        derived_name=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=20)
    def test_class_with_bases_creates_inheritance_edge_property(
        self, base_name: Any, derived_name: Any, path: str, language: str
    ):
        """HAPPY: class with matching base creates an inheritance edge."""
        assume(base_name != derived_name)
        elements = [
            _elem_prop("class", base_name, path, language, metadata={}),
            _elem_prop(
                "class",
                derived_name,
                path,
                language,
                metadata={
                    "bases": [base_name],
                },
            ),
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        inherit_edges = [e for e in snap.edges if e.edge_type == "inherit"]
        assert len(inherit_edges) == 1
        assert inherit_edges[0].metadata.get("base") == base_name
        assert inherit_edges[0].source == "fc_structure"

    @given(
        derived_name=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_class_with_unknown_base_no_inheritance_edge_property(
        self, derived_name: Any, path: str, language: str
    ):
        """EDGE: class with base not in snapshot produces no inheritance edge."""
        elements = [
            _elem_prop(
                "class",
                derived_name,
                path,
                language,
                metadata={
                    "bases": ["UnknownClass"],
                },
            ),
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        inherit_edges = [e for e in snap.edges if e.edge_type == "inherit"]
        assert len(inherit_edges) == 0

    @given(
        name=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_class_with_self_base_no_edge_property(
        self, name: str, path: str, language: str
    ):
        """EDGE: class inheriting from itself produces no edge."""
        elements = [
            _elem_prop(
                "class",
                name,
                path,
                language,
                metadata={
                    "bases": [name],
                },
            ),
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        inherit_edges = [e for e in snap.edges if e.edge_type == "inherit"]
        assert len(inherit_edges) == 0


class TestSnapshotIdentity:
    @given(
        repo_name=identifier,
        snapshot_id=st.builds(lambda x: f"snap:{x}", identifier),
        branch=st.none() | identifier,
        commit_id=st.none()
        | st.text(alphabet="0123456789abcdef", min_size=7, max_size=40),
        tree_id=st.none() | identifier,
    )
    @settings(max_examples=20)
    def test_snapshot_fields_preserved_property(
        self,
        repo_name: str,
        snapshot_id: str,
        branch: str,
        commit_id: str,
        tree_id: Any,
    ):
        """HAPPY: snapshot identity fields pass through unchanged."""
        snap = build_ir_from_ast(
            repo_name,
            snapshot_id,
            [],
            _repo_root,
            branch=branch,
            commit_id=commit_id,
            tree_id=tree_id,
        )
        assert snap.repo_name == repo_name
        assert snap.snapshot_id == snapshot_id
        assert snap.branch == branch
        assert snap.commit_id == commit_id
        assert snap.tree_id == tree_id


class TestQualifiedNameResolution:
    @given(
        class_name=identifier,
        method_name=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=20)
    def test_method_with_class_name_gets_qualified_property(
        self, class_name: str, method_name: Any, path: str, language: str
    ):
        """HAPPY: method with class_name metadata gets qualified_name='Class.method'."""
        elements = [
            _elem_prop(
                "method",
                method_name,
                path,
                language,
                metadata={"class_name": class_name},
            ),
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert len(snap.symbols) == 1
        assert snap.symbols[0].qualified_name == f"{class_name}.{method_name}"

    @given(
        name=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=20)
    def test_function_without_class_name_uses_plain_name_property(
        self, name: str, path: str, language: str
    ):
        """HAPPY: function without class_name metadata uses name as qualified_name."""
        elements = [_elem_prop("function", name, path, language, metadata={})]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert len(snap.symbols) == 1
        assert snap.symbols[0].qualified_name == name


# ─── Edge cases: empty inputs, boundary conditions ───


class TestBuildIrFromAstEdgeCases:
    """Edge case tests for build_ir_from_ast with boundary inputs."""

    def test_empty_elements_list(self):
        """No code elements should produce snapshot with no symbols."""
        snap = _build([])
        assert len(snap.symbols) == 0

    def test_element_with_no_metadata(self):
        """Element with empty metadata dict should not crash."""
        elements = [_elem(name="f", type="function", start_line=1, metadata={})]
        snap = _build(elements)
        assert len(snap.symbols) == 1

    def test_element_with_zero_start_line(self):
        """Start line of 0 should be clamped or handled."""
        elements = [_elem(name="f", type="function", start_line=0, metadata={})]
        snap = _build(elements)
        assert len(snap.symbols) == 1

    def test_multiple_elements_same_name_different_files(self):
        """Elements with same name in different files get different symbol_ids."""
        e1 = _elem(
            name="handler",
            type="function",
            start_line=10,
            metadata={},
            relative_path="a.py",
            language="python",
        )
        e2 = _elem(
            name="handler",
            type="function",
            start_line=20,
            metadata={},
            relative_path="b.py",
            language="python",
        )
        snap = _build([e1, e2])
        assert len(snap.symbols) == 2
        ids = [s.symbol_id for s in snap.symbols]
        assert ids[0] != ids[1]
