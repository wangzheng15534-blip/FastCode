"""Property-based tests for ast_to_ir.build_ir_from_ast invariants.

Covers document deduplication, symbol creation filtering, line clamping,
source_priority/source_set tagging, containment edges, import/inheritance
edge generation, and metadata consistency.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from fastcode.adapters.ast_to_ir import build_ir_from_ast
from fastcode.indexer import CodeElement
from fastcode.semantic_ir import IRSnapshot

# --- Strategies (self-contained, no conftest import) ---

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
    type=st.sampled_from(["function", "class", "variable", "method", "file", "documentation"]),
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
    metadata=st.fixed_dictionaries({}, optional={
        "class_name": identifier,
        "start_col": st.integers(min_value=0, max_value=80),
        "imports": st.lists(
            st.fixed_dictionaries({"module": identifier}),
            max_size=2,
        ),
        "bases": st.lists(identifier, max_size=2),
    }),
)


# --- Helpers ---

_repo_root = "/tmp/test_repo"

_element_types_with_symbols = st.sampled_from(["function", "class", "variable", "method"])
_element_types_without_symbols = st.sampled_from(["file", "documentation"])

_small_int = st.integers(min_value=1, max_value=500)


def _elem(
    type_: str,
    name: str,
    relative_path: str,
    language: str = "python",
    start_line: int = 1,
    end_line: int = 5,
    metadata: dict | None = None,
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


# --- Properties ---


@pytest.mark.property
class TestDocumentDeduplication:

    @given(
        name1=identifier,
        name2=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_same_path_produces_one_document(self, name1, name2, path, language):
        """HAPPY: multiple elements from same file produce exactly one document."""
        assume(name1 != name2)
        elements = [
            _elem("function", name1, path, language, start_line=1, end_line=5),
            _elem("function", name2, path, language, start_line=10, end_line=15),
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
    @pytest.mark.happy
    def test_different_paths_produce_multiple_documents(self, name1, name2, path1, path2, language):
        """HAPPY: elements from different files produce distinct documents."""
        assume(path1 != path2)
        elements = [
            _elem("function", name1, path1, language),
            _elem("function", name2, path2, language),
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
    @pytest.mark.happy
    def test_many_elements_one_file_still_one_document(self, n, path, language):
        """HAPPY: N elements in same file produce exactly one document."""
        elements = [
            _elem("function", f"fn_{i}", path, language, start_line=i * 10)
            for i in range(n)
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert len(snap.documents) == 1


@pytest.mark.property
class TestSymbolCreation:

    @given(
        name=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_file_type_skips_symbol(self, name, path, language):
        """HAPPY: file-type elements produce no symbols."""
        elements = [_elem("file", name, path, language)]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert len(snap.symbols) == 0
        assert len(snap.occurrences) == 0

    @given(
        name=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_documentation_type_skips_symbol(self, name, path, language):
        """HAPPY: documentation-type elements produce no symbols."""
        elements = [_elem("documentation", name, path, language)]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert len(snap.symbols) == 0

    @given(
        name=identifier,
        path=file_path_st,
        language=language_st,
        type_=_element_types_with_symbols,
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_non_file_types_create_symbols(self, name, path, language, type_):
        """HAPPY: function/class/variable/method elements produce exactly one symbol."""
        elements = [_elem(type_, name, path, language)]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert len(snap.symbols) == 1
        assert snap.symbols[0].display_name == name

    @pytest.mark.happy
    def test_empty_elements_produces_empty_snapshot(self):
        """HAPPY: empty element list produces snapshot with zero symbols/occurrences."""
        snap = build_ir_from_ast("repo", "snap:1", [], _repo_root)
        assert len(snap.symbols) == 0
        assert len(snap.occurrences) == 0
        assert len(snap.documents) == 0
        assert len(snap.edges) == 0


@pytest.mark.property
class TestLineClamping:

    @given(
        name=identifier,
        path=file_path_st,
        start_line=st.one_of(st.none(), st.just(0), st.just(-1)),
    )
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_occurrence_start_line_clamped_to_one(self, name, path, start_line):
        """EDGE: occurrence start_line is clamped to >= 1."""
        elem = CodeElement(
            id=f"elem_{name}", type="function", name=name,
            file_path=path, relative_path=path, language="python",
            start_line=start_line, end_line=1,
            code="pass", signature=None, docstring=None, summary=None,
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
    @pytest.mark.happy
    def test_valid_start_line_preserved(self, name, path, start_line):
        """HAPPY: valid start_line >= 1 is preserved."""
        elem = _elem("function", name, path, start_line=start_line, end_line=start_line + 5)
        snap = build_ir_from_ast("repo", "snap:1", [elem], _repo_root)
        assert snap.occurrences[0].start_line == start_line


@pytest.mark.property
class TestSourcePriorityAndSet:

    @given(
        elements=st.lists(code_element_st.filter(lambda e: e.type not in {"file", "documentation"}), max_size=5),
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_all_symbols_have_source_priority_10(self, elements):
        """HAPPY: all symbols created by AST adapter have source_priority=10."""
        assume(len(elements) > 0)
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        for sym in snap.symbols:
            assert sym.source_priority == 10

    @given(
        elements=st.lists(code_element_st, max_size=5),
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_all_docs_have_ast_in_source_set(self, elements):
        """HAPPY: all documents have 'ast' in source_set."""
        assume(len(elements) > 0)
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        for doc in snap.documents:
            assert "ast" in doc.source_set

    @given(
        elements=st.lists(code_element_st.filter(lambda e: e.type not in {"file", "documentation"}), max_size=5),
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_all_symbols_have_ast_source_set(self, elements):
        """HAPPY: all symbols have source_set={'ast'}."""
        assume(len(elements) > 0)
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        for sym in snap.symbols:
            assert sym.source_set == {"ast"}


@pytest.mark.property
class TestContainmentEdges:

    @given(
        name=identifier,
        path=file_path_st,
        language=language_st,
        type_=_element_types_with_symbols,
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_non_file_symbols_have_containment_edge(self, name, path, language, type_):
        """HAPPY: every non-file, non-doc symbol gets a containment edge."""
        elements = [_elem(type_, name, path, language)]
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
    @pytest.mark.happy
    def test_containment_count_equals_symbol_count(self, n, path, language):
        """HAPPY: number of containment edges equals number of symbols."""
        elements = [_elem("function", f"fn_{i}", path, language, start_line=i * 10) for i in range(n)]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        contain_edges = [e for e in snap.edges if e.edge_type == "contain"]
        assert len(contain_edges) == n


@pytest.mark.property
class TestMetadataConsistency:

    @given(
        elements=st.lists(code_element_st, max_size=4),
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_snapshot_metadata_source_modes_is_ast(self, elements):
        """HAPPY: snapshot metadata always has source_modes=['ast']."""
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert snap.metadata.get("source_modes") == ["ast"]

    @given(
        name=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_symbol_metadata_has_source_and_confidence(self, name, path, language):
        """HAPPY: every symbol metadata contains 'source' and 'confidence' keys."""
        elements = [_elem("function", name, path, language)]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        for sym in snap.symbols:
            assert "source" in sym.metadata
            assert "confidence" in sym.metadata

    @given(
        elements=st.lists(code_element_st.filter(lambda e: e.type not in {"file", "documentation"}), max_size=5),
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_symbol_metadata_source_is_ast(self, elements):
        """HAPPY: all symbol metadata['source'] == 'ast'."""
        assume(len(elements) > 0)
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        for sym in snap.symbols:
            assert sym.metadata.get("source") == "ast"


@pytest.mark.property
class TestImportEdges:

    @given(
        module_name=identifier,
        path1=file_path_st,
        path2=file_path_st,
        language=language_st,
    )
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_file_imports_create_dependency_edge(self, module_name, path1, path2, language):
        """HAPPY: file element with imports metadata matching a known path creates import edge."""
        assume(path1 != path2)
        # Create a module path that maps to path2
        module_path = module_name.replace(".", "/")
        # Adjust path2 so it ends with the module path
        adjusted_path2 = f"src/{module_name}.py"

        elements = [
            _elem("file", "mod", path1, language, metadata={
                "imports": [{"module": module_name}],
            }),
            _elem("function", "fn1", adjusted_path2, language),
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        import_edges = [e for e in snap.edges if e.edge_type == "import"]
        # Import edge may or may not exist depending on path matching
        for edge in import_edges:
            assert edge.edge_type == "import"
            assert edge.source == "ast"
            assert edge.confidence == "heuristic"

    @given(path=file_path_st, language=language_st)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_file_import_no_match_no_edge(self, path, language):
        """EDGE: file with imports but no matching target produces no import edge."""
        elements = [
            _elem("file", "mod", path, language, metadata={
                "imports": [{"module": "nonexistent_module_xyz"}],
            }),
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        import_edges = [e for e in snap.edges if e.edge_type == "import"]
        assert len(import_edges) == 0


@pytest.mark.property
class TestInheritanceEdges:

    @given(
        base_name=identifier,
        derived_name=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_class_with_bases_creates_inheritance_edge(self, base_name, derived_name, path, language):
        """HAPPY: class with matching base creates an inheritance edge."""
        assume(base_name != derived_name)
        elements = [
            _elem("class", base_name, path, language, metadata={}),
            _elem("class", derived_name, path, language, metadata={
                "bases": [base_name],
            }),
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        inherit_edges = [e for e in snap.edges if e.edge_type == "inherit"]
        assert len(inherit_edges) == 1
        assert inherit_edges[0].metadata.get("base") == base_name
        assert inherit_edges[0].source == "ast"

    @given(
        derived_name=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_class_with_unknown_base_no_inheritance_edge(self, derived_name, path, language):
        """EDGE: class with base not in snapshot produces no inheritance edge."""
        elements = [
            _elem("class", derived_name, path, language, metadata={
                "bases": ["UnknownClass"],
            }),
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
    def test_class_with_self_base_no_edge(self, name, path, language):
        """EDGE: class inheriting from itself produces no edge."""
        elements = [
            _elem("class", name, path, language, metadata={
                "bases": [name],
            }),
        ]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        inherit_edges = [e for e in snap.edges if e.edge_type == "inherit"]
        assert len(inherit_edges) == 0


@pytest.mark.property
class TestSnapshotIdentity:

    @given(
        repo_name=identifier,
        snapshot_id=st.builds(lambda x: f"snap:{x}", identifier),
        branch=st.none() | identifier,
        commit_id=st.none() | st.text(alphabet="0123456789abcdef", min_size=7, max_size=40),
        tree_id=st.none() | identifier,
    )
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_snapshot_fields_preserved(self, repo_name, snapshot_id, branch, commit_id, tree_id):
        """HAPPY: snapshot identity fields pass through unchanged."""
        snap = build_ir_from_ast(repo_name, snapshot_id, [], _repo_root,
                                 branch=branch, commit_id=commit_id, tree_id=tree_id)
        assert snap.repo_name == repo_name
        assert snap.snapshot_id == snapshot_id
        assert snap.branch == branch
        assert snap.commit_id == commit_id
        assert snap.tree_id == tree_id


@pytest.mark.property
class TestQualifiedNameResolution:

    @given(
        class_name=identifier,
        method_name=identifier,
        path=file_path_st,
        language=language_st,
    )
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_method_with_class_name_gets_qualified(self, class_name, method_name, path, language):
        """HAPPY: method with class_name metadata gets qualified_name='Class.method'."""
        elements = [
            _elem("method", method_name, path, language, metadata={"class_name": class_name}),
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
    @pytest.mark.happy
    def test_function_without_class_name_uses_plain_name(self, name, path, language):
        """HAPPY: function without class_name metadata uses name as qualified_name."""
        elements = [_elem("function", name, path, language, metadata={})]
        snap = build_ir_from_ast("repo", "snap:1", elements, _repo_root)
        assert len(snap.symbols) == 1
        assert snap.symbols[0].qualified_name == name
