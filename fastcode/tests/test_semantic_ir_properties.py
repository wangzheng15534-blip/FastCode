"""Property-based tests for semantic_ir models."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.semantic_ir import (
    IRAttachment,
    IRDocument,
    IREdge,
    IROccurrence,
    IRSnapshot,
    IRSymbol,
)

# --- Strategies (mirrored from tests/conftest.py) ---

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

source_st = st.sampled_from(["ast", "fc_structure", "scip"])
attachment_source_st = st.sampled_from(
    ["fc_structure", "fc_embedding", "llm_annotation"]
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
    ]
)

language_st = st.sampled_from(
    ["python", "javascript", "typescript", "go", "java", "rust", "c", "cpp"]
)

line_number_st = st.integers(min_value=1, max_value=10000)

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

ir_attachment_st = st.builds(
    IRAttachment,
    attachment_id=st.builds(lambda x: f"att:{x}", identifier),
    target_id=st.builds(lambda x: f"sym:{x}", identifier),
    target_type=st.sampled_from(["document", "symbol", "snapshot"]),
    attachment_type=st.sampled_from(["embedding", "summary", "semantic_note"]),
    source=attachment_source_st,
    confidence=st.sampled_from(["derived", "precise", "heuristic"]),
    payload=st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(
            st.integers(),
            st.text(min_size=0, max_size=20),
            st.lists(st.floats(allow_nan=False, allow_infinity=False), max_size=4),
        ),
        max_size=3,
    ),
    metadata=st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(st.integers(), st.text(min_size=0, max_size=20)),
        max_size=3,
    ),
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
        attachments=st.lists(ir_attachment_st, max_size=4),
        metadata=st.dictionaries(
            st.sampled_from(["source_modes", "version", "tool"]),
            st.one_of(
                st.just(["ast"]), st.just(["scip"]), st.just(1), st.just("fastcode")
            ),
        ),
    )


# --- IRDocument Properties ---


class TestIRDocumentProperties:
    @given(doc=ir_document_st)
    @settings(max_examples=50)
    @pytest.mark.basic
    def test_document_roundtrip_property(self, doc: IRDocument):
        """HAPPY: IRDocument.to_dict() -> from_dict() roundtrip preserves all fields."""
        data = doc.to_dict()
        restored = IRDocument.from_dict(data)
        assert restored.doc_id == doc.doc_id
        assert restored.path == doc.path
        assert restored.language == doc.language
        assert restored.blob_oid == doc.blob_oid
        assert restored.content_hash == doc.content_hash
        assert restored.source_set == doc.source_set

    @given(doc=ir_document_st)
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_document_source_set_serialized_as_sorted_list_property(self, doc: IRDocument):
        """HAPPY: to_dict serializes source_set as a sorted list."""
        data = doc.to_dict()
        assert isinstance(data["source_set"], list)
        assert data["source_set"] == sorted(data["source_set"])

    @given(doc_id=identifier, path=st.just("a/b.py"), language=language_st)
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_document_defaults_property(self, doc_id: str, path: str, language: str):
        """HAPPY: IRDocument with only required fields gets proper defaults."""
        doc = IRDocument(doc_id=doc_id, path=path, language=language)
        assert doc.blob_oid is None
        assert doc.content_hash is None
        assert doc.source_set == set()

    @given(doc_id=identifier, path=st.just("a/b.py"), language=language_st)
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_document_from_dict_missing_source_set_property(
        self, doc_id: str, path: str, language: str
    ):
        """EDGE: from_dict with missing source_set key defaults to empty set (line 28)."""
        data = {"doc_id": doc_id, "path": path, "language": language}
        doc = IRDocument.from_dict(data)
        assert doc.source_set == set()

    @given(
        doc_id=identifier,
        path=file_path_st,
        language=language_st,
        source_set=st.sets(source_st, max_size=2),
    )
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_document_from_dict_with_source_set_present_property(
        self, doc_id: str, path: str, language: str, source_set: Any
    ):
        """HAPPY: from_dict with source_set present exercises lines 27-29 (dict copy + set conversion + cls construction)."""
        data = {
            "doc_id": doc_id,
            "path": path,
            "language": language,
            "source_set": list(source_set),
        }
        restored = IRDocument.from_dict(data)
        assert restored.source_set == source_set
        assert restored.doc_id == doc_id

    @given(
        doc_id=identifier,
        path=file_path_st,
        language=language_st,
        extra_key=identifier.filter(
            lambda k: (
                k
                not in {
                    "doc_id",
                    "path",
                    "language",
                    "blob_oid",
                    "content_hash",
                    "source_set",
                }
            )
        ),
        extra_val=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=20)
    @pytest.mark.negative
    def test_document_from_dict_extra_keys_raise_typeerror_property(
        self, doc_id: str, path: str, language: str, extra_key: Any, extra_val: Any
    ):
        """EDGE: from_dict with extra unknown key raises TypeError (dataclass rejects unexpected kwargs)."""
        data = {
            "doc_id": doc_id,
            "path": path,
            "language": language,
            extra_key: extra_val,
        }
        with pytest.raises(TypeError):
            IRDocument.from_dict(data)

    @given(doc=ir_document_st)
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_document_from_dict_does_not_mutate_input_property(self, doc: IRDocument):
        """HAPPY: from_dict copies data (line 27: payload = dict(data)) so original dict is not mutated."""
        data = doc.to_dict()
        original_source_set = list(data["source_set"])
        IRDocument.from_dict(data)
        assert data["source_set"] == original_source_set
        assert isinstance(
            data["source_set"], list
        )  # Still a list, not converted to set


# --- IRSymbol Properties ---


class TestIRSymbolProperties:
    @given(sym=ir_symbol_st)
    @settings(max_examples=50)
    @pytest.mark.basic
    def test_symbol_roundtrip_property(self, sym: IRSymbol):
        """HAPPY: IRSymbol.to_dict() -> from_dict() roundtrip preserves all fields."""
        data = sym.to_dict()
        restored = IRSymbol.from_dict(data)
        assert restored.symbol_id == sym.symbol_id
        assert restored.external_symbol_id == sym.external_symbol_id
        assert restored.path == sym.path
        assert restored.display_name == sym.display_name
        assert restored.kind == sym.kind
        assert restored.language == sym.language
        assert restored.qualified_name == sym.qualified_name
        assert restored.signature == sym.signature
        assert restored.start_line == sym.start_line
        assert restored.start_col == sym.start_col
        assert restored.end_line == sym.end_line
        assert restored.end_col == sym.end_col
        assert restored.source_priority == sym.source_priority
        assert restored.source_set == sym.source_set
        assert restored.metadata == sym.metadata

    @given(sym=ir_symbol_st)
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_symbol_source_set_serialized_as_sorted_list_property(self, sym: IRSymbol):
        """HAPPY: to_dict serializes source_set as a sorted list."""
        data = sym.to_dict()
        assert isinstance(data["source_set"], list)
        assert data["source_set"] == sorted(data["source_set"])

    @given(
        symbol_id=st.builds(lambda x: f"sym:{x}", identifier),
        path=st.just("a/b.py"),
        display_name=identifier,
        kind=kind_st,
        language=language_st,
    )
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_symbol_defaults_property(
        self, symbol_id: str, path: str, display_name: str, kind: str, language: str
    ):
        """HAPPY: IRSymbol with only required fields gets proper defaults."""
        sym = IRSymbol(
            symbol_id=symbol_id,
            external_symbol_id=None,
            path=path,
            display_name=display_name,
            kind=kind,
            language=language,
        )
        assert sym.qualified_name is None
        assert sym.signature is None
        assert sym.start_line is None
        assert sym.start_col is None
        assert sym.end_line is None
        assert sym.end_col is None
        assert sym.source_priority == 0
        assert sym.source_set == set()
        assert sym.metadata == {}

    @given(
        symbol_id=st.builds(lambda x: f"sym:{x}", identifier),
        path=st.just("a/b.py"),
        display_name=identifier,
        kind=kind_st,
        language=language_st,
    )
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_symbol_from_dict_missing_source_set_property(
        self, symbol_id: str, path: str, display_name: str, kind: str, language: str
    ):
        """EDGE: from_dict with missing source_set defaults to empty set (line 58)."""
        data = {
            "symbol_id": symbol_id,
            "external_symbol_id": None,
            "path": path,
            "display_name": display_name,
            "kind": kind,
            "language": language,
        }
        sym = IRSymbol.from_dict(data)
        assert sym.source_set == set()

    @given(
        symbol_id=st.builds(lambda x: f"sym:{x}", identifier),
        external_symbol_id=st.none() | identifier,
        path=file_path_st,
        display_name=identifier,
        kind=kind_st,
        language=language_st,
        source_set=st.sets(source_st, max_size=2),
        metadata=st.dictionaries(st.text(min_size=1, max_size=10), st.integers()),
    )
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_symbol_from_dict_with_source_set_present_property(
        self,
        symbol_id: str,
        external_symbol_id: Any,
        path: str,
        display_name: Any,
        kind: bool,
        language: str,
        source_set: Any,
        metadata: dict[str, Any],
    ):
        """HAPPY: from_dict with source_set present exercises lines 57-59 (dict copy + set conversion + cls construction)."""
        data = {
            "symbol_id": symbol_id,
            "external_symbol_id": external_symbol_id,
            "path": path,
            "display_name": display_name,
            "kind": kind,
            "language": language,
            "source_set": list(source_set),
            "metadata": metadata,
        }
        restored = IRSymbol.from_dict(data)
        assert restored.source_set == source_set
        assert restored.metadata == metadata
        assert restored.symbol_id == symbol_id

    @given(sym=ir_symbol_st)
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_symbol_from_dict_does_not_mutate_input_property(self, sym: IRSymbol):
        """HAPPY: from_dict copies data (line 57: payload = dict(data)) so original dict is not mutated."""
        data = sym.to_dict()
        original_source_set = list(data["source_set"])
        IRSymbol.from_dict(data)
        assert data["source_set"] == original_source_set
        assert isinstance(data["source_set"], list)  # Still a list, not a set

    @given(
        symbol_id=st.builds(lambda x: f"sym:{x}", identifier),
        external_symbol_id=st.none() | identifier,
        path=file_path_st,
        display_name=identifier,
        kind=kind_st,
        language=language_st,
    )
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_symbol_from_dict_empty_source_set_list_property(
        self,
        symbol_id: str,
        external_symbol_id: Any,
        path: str,
        display_name: Any,
        kind: bool,
        language: str,
    ):
        """EDGE: from_dict with empty list for source_set produces empty set."""
        data = {
            "symbol_id": symbol_id,
            "external_symbol_id": external_symbol_id,
            "path": path,
            "display_name": display_name,
            "kind": kind,
            "language": language,
            "source_set": [],
        }
        restored = IRSymbol.from_dict(data)
        assert restored.source_set == set()


# --- IROccurrence Properties ---


class TestIROccurrenceProperties:
    @given(occ=ir_occurrence_st)
    @settings(max_examples=50)
    @pytest.mark.basic
    def test_occurrence_roundtrip_property(self, occ: IROccurrence):
        """HAPPY: IROccurrence.to_dict() -> from_dict() roundtrip preserves all fields."""
        data = occ.to_dict()
        restored = IROccurrence.from_dict(data)
        assert restored.occurrence_id == occ.occurrence_id
        assert restored.symbol_id == occ.symbol_id
        assert restored.doc_id == occ.doc_id
        assert restored.role == occ.role
        assert restored.start_line == occ.start_line
        assert restored.start_col == occ.start_col
        assert restored.end_line == occ.end_line
        assert restored.end_col == occ.end_col
        assert restored.source == occ.source
        assert restored.metadata == occ.metadata

    @given(
        occurrence_id=st.builds(lambda x: f"occ:{x}", identifier),
        symbol_id=st.builds(lambda x: f"sym:{x}", identifier),
        doc_id=st.builds(lambda x: f"doc:{x}", identifier),
        role=role_st,
        source=source_st,
    )
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_occurrence_defaults_property(
        self, occurrence_id: str, symbol_id: str, doc_id: str, role: str, source: str
    ):
        """HAPPY: IROccurrence with only required fields gets empty metadata."""
        occ = IROccurrence(
            occurrence_id=occurrence_id,
            symbol_id=symbol_id,
            doc_id=doc_id,
            role=role,
            start_line=1,
            start_col=0,
            end_line=1,
            end_col=10,
            source=source,
        )
        assert occ.metadata == {}

    @given(occ=ir_occurrence_st)
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_occurrence_to_dict_is_asdict_property(self, occ: IROccurrence):
        """HAPPY: to_dict output matches dataclasses.asdict for IROccurrence (line 76)."""
        from dataclasses import asdict

        assert occ.to_dict() == asdict(occ)

    @given(
        occurrence_id=st.builds(lambda x: f"occ:{x}", identifier),
        symbol_id=st.builds(lambda x: f"sym:{x}", identifier),
        doc_id=st.builds(lambda x: f"doc:{x}", identifier),
        role=role_st,
        source=source_st,
    )
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_occurrence_from_dict_passes_through_property(
        self, occurrence_id: str, symbol_id: str, doc_id: str, role: str, source: str
    ):
        """EDGE: from_dict directly passes data dict to constructor (line 80)."""
        data = {
            "occurrence_id": occurrence_id,
            "symbol_id": symbol_id,
            "doc_id": doc_id,
            "role": role,
            "start_line": 5,
            "start_col": 2,
            "end_line": 5,
            "end_col": 20,
            "source": source,
        }
        occ = IROccurrence.from_dict(data)
        assert occ.occurrence_id == occurrence_id
        assert occ.start_line == 5

    @given(
        occurrence_id=st.builds(lambda x: f"occ:{x}", identifier),
        symbol_id=st.builds(lambda x: f"sym:{x}", identifier),
        doc_id=st.builds(lambda x: f"doc:{x}", identifier),
        role=role_st,
        source=source_st,
        metadata=st.dictionaries(st.text(min_size=1, max_size=10), st.integers()),
    )
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_occurrence_from_dict_with_metadata_property(
        self,
        occurrence_id: str,
        symbol_id: str,
        doc_id: str,
        role: str,
        source: str,
        metadata: dict[str, Any],
    ):
        """HAPPY: from_dict with metadata exercises line 80 (cls(**data)) including metadata field."""
        data = {
            "occurrence_id": occurrence_id,
            "symbol_id": symbol_id,
            "doc_id": doc_id,
            "role": role,
            "start_line": 1,
            "start_col": 0,
            "end_line": 1,
            "end_col": 10,
            "source": source,
            "metadata": metadata,
        }
        occ = IROccurrence.from_dict(data)
        assert occ.metadata == metadata
        assert occ.occurrence_id == occurrence_id

    @given(
        occurrence_id=st.builds(lambda x: f"occ:{x}", identifier),
        symbol_id=st.builds(lambda x: f"sym:{x}", identifier),
        doc_id=st.builds(lambda x: f"doc:{x}", identifier),
        role=role_st,
        source=source_st,
    )
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_occurrence_from_dict_no_metadata_key_property(
        self, occurrence_id: str, symbol_id: str, doc_id: str, role: str, source: str
    ):
        """EDGE: from_dict without metadata key uses dataclass default (line 80), which is empty dict."""
        data = {
            "occurrence_id": occurrence_id,
            "symbol_id": symbol_id,
            "doc_id": doc_id,
            "role": role,
            "start_line": 1,
            "start_col": 0,
            "end_line": 1,
            "end_col": 10,
            "source": source,
        }
        occ = IROccurrence.from_dict(data)
        assert occ.metadata == {}


# --- IREdge Properties ---


class TestIREdgeProperties:
    @given(edge=ir_edge_st)
    @settings(max_examples=50)
    @pytest.mark.basic
    def test_edge_roundtrip_property(self, edge: IREdge):
        """HAPPY: IREdge.to_dict() -> from_dict() roundtrip preserves all fields."""
        data = edge.to_dict()
        restored = IREdge.from_dict(data)
        assert restored.edge_id == edge.edge_id
        assert restored.src_id == edge.src_id
        assert restored.dst_id == edge.dst_id
        assert restored.edge_type == edge.edge_type
        assert restored.source == edge.source
        assert restored.confidence == edge.confidence
        assert restored.doc_id == edge.doc_id
        assert restored.metadata == edge.metadata

    @given(
        edge_id=st.builds(lambda x: f"edge:{x}", identifier),
        src_id=identifier,
        dst_id=identifier,
        edge_type=edge_type_st,
        source=source_st,
        confidence=st.just("precise"),
    )
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_edge_defaults_property(
        self,
        edge_id: str,
        src_id: str,
        dst_id: str,
        edge_type: str,
        source: str,
        confidence: str,
    ):
        """HAPPY: IREdge with only required fields gets None doc_id and empty metadata."""
        edge = IREdge(
            edge_id=edge_id,
            src_id=src_id,
            dst_id=dst_id,
            edge_type=edge_type,
            source=source,
            confidence=confidence,
        )
        assert edge.doc_id is None
        assert edge.metadata == {}

    @given(edge=ir_edge_st)
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_edge_to_dict_is_asdict_property(self, edge: IREdge):
        """HAPPY: to_dict output matches dataclasses.asdict for IREdge (line 95)."""
        from dataclasses import asdict

        assert edge.to_dict() == asdict(edge)

    @given(
        edge_id=st.builds(lambda x: f"edge:{x}", identifier),
        src_id=identifier,
        dst_id=identifier,
        edge_type=edge_type_st,
        source=source_st,
    )
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_edge_from_dict_passes_through_property(
        self, edge_id: str, src_id: str, dst_id: str, edge_type: str, source: str
    ):
        """EDGE: from_dict directly passes data dict to constructor (line 99)."""
        data = {
            "edge_id": edge_id,
            "src_id": src_id,
            "dst_id": dst_id,
            "edge_type": edge_type,
            "source": source,
            "confidence": "heuristic",
        }
        edge = IREdge.from_dict(data)
        assert edge.edge_id == edge_id
        assert edge.confidence == "heuristic"
        assert edge.doc_id is None
        assert edge.metadata == {}

    @given(
        edge_id=st.builds(lambda x: f"edge:{x}", identifier),
        src_id=identifier,
        dst_id=identifier,
        edge_type=edge_type_st,
        source=source_st,
        confidence=st.sampled_from(["precise", "heuristic", "resolved", ""]),
        doc_id=st.none() | st.builds(lambda x: f"doc:{x}", identifier),
        metadata=st.dictionaries(st.text(min_size=1, max_size=10), st.integers()),
    )
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_edge_from_dict_with_all_fields_property(
        self,
        edge_id: str,
        src_id: str,
        dst_id: str,
        edge_type: str,
        source: str,
        confidence: Any,
        doc_id: str,
        metadata: dict[str, Any],
    ):
        """HAPPY: from_dict with all fields exercises line 99 (cls(**data)) completely."""
        data = {
            "edge_id": edge_id,
            "src_id": src_id,
            "dst_id": dst_id,
            "edge_type": edge_type,
            "source": source,
            "confidence": confidence,
            "doc_id": doc_id,
            "metadata": metadata,
        }
        edge = IREdge.from_dict(data)
        assert edge.edge_id == edge_id
        assert edge.src_id == src_id
        assert edge.dst_id == dst_id
        assert edge.edge_type == edge_type
        assert edge.source == source
        assert edge.confidence == confidence
        assert edge.doc_id == doc_id
        assert edge.metadata == metadata

    @given(
        edge_id=st.builds(lambda x: f"edge:{x}", identifier),
        src_id=identifier,
        dst_id=identifier,
        edge_type=edge_type_st,
        source=source_st,
    )
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_edge_from_dict_no_optional_keys_property(
        self, edge_id: str, src_id: str, dst_id: str, edge_type: str, source: str
    ):
        """EDGE: from_dict without doc_id/metadata keys uses dataclass defaults (line 99)."""
        data = {
            "edge_id": edge_id,
            "src_id": src_id,
            "dst_id": dst_id,
            "edge_type": edge_type,
            "source": source,
            "confidence": "precise",
        }
        edge = IREdge.from_dict(data)
        assert edge.doc_id is None
        assert edge.metadata == {}


# --- IRSnapshot Properties ---


class TestIRAttachmentProperties:
    @given(attachment=ir_attachment_st)
    @settings(max_examples=40)
    @pytest.mark.basic
    def test_attachment_roundtrip_property(self, attachment: IRAttachment):
        """HAPPY: IRAttachment.to_dict() -> from_dict() preserves all fields."""
        data = attachment.to_dict()
        restored = IRAttachment.from_dict(data)
        assert restored.attachment_id == attachment.attachment_id
        assert restored.target_id == attachment.target_id
        assert restored.target_type == attachment.target_type
        assert restored.attachment_type == attachment.attachment_type
        assert restored.source == attachment.source
        assert restored.confidence == attachment.confidence
        assert restored.payload == attachment.payload
        assert restored.metadata == attachment.metadata

    @given(attachment=ir_attachment_st)
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_attachment_payload_is_jsonable_after_to_dict_property(
        self, attachment: IRAttachment
    ):
        """HAPPY: serialization normalizes payload and metadata to JSON-safe values."""
        data = attachment.to_dict()
        assert isinstance(data["payload"], dict)
        assert isinstance(data["metadata"], dict)


class TestIRSnapshotProperties:
    @given(snap=snapshot_st())
    @settings(max_examples=40)
    @pytest.mark.basic
    def test_snapshot_roundtrip_property(self, snap: IRSnapshot):
        """HAPPY: IRSnapshot.to_dict() -> from_dict() roundtrip preserves all fields."""
        data = snap.to_dict()
        restored = IRSnapshot.from_dict(data)
        assert restored.repo_name == snap.repo_name
        assert restored.snapshot_id == snap.snapshot_id
        assert restored.branch == snap.branch
        assert restored.commit_id == snap.commit_id
        assert restored.tree_id == snap.tree_id
        assert len(restored.documents) == len(snap.documents)
        assert len(restored.symbols) == len(snap.symbols)
        assert len(restored.occurrences) == len(snap.occurrences)
        assert len(restored.edges) == len(snap.edges)
        assert len(restored.attachments) == len(snap.attachments)
        assert restored.metadata == snap.metadata

    @given(snap=snapshot_st())
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_snapshot_roundtrip_documents_preserve_fields_property(self, snap: IRSnapshot):
        """HAPPY: nested IRDocument roundtrips preserve all fields."""
        data = snap.to_dict()
        restored = IRSnapshot.from_dict(data)
        for orig, rest in zip(snap.documents, restored.documents, strict=True):
            assert orig.doc_id == rest.doc_id
            assert orig.path == rest.path
            assert orig.language == rest.language
            assert orig.source_set == rest.source_set

    @given(snap=snapshot_st())
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_snapshot_roundtrip_symbols_preserve_fields_property(self, snap: IRSnapshot):
        """HAPPY: nested IRSymbol roundtrips preserve all fields."""
        data = snap.to_dict()
        restored = IRSnapshot.from_dict(data)
        for orig, rest in zip(snap.symbols, restored.symbols, strict=True):
            assert orig.symbol_id == rest.symbol_id
            assert orig.display_name == rest.display_name
            assert orig.kind == rest.kind
            assert orig.source_set == rest.source_set

    @given(snap=snapshot_st())
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_snapshot_roundtrip_edges_preserve_fields_property(self, snap: IRSnapshot):
        """HAPPY: nested IREdge roundtrips preserve all fields."""
        data = snap.to_dict()
        restored = IRSnapshot.from_dict(data)
        for orig, rest in zip(snap.edges, restored.edges, strict=True):
            assert orig.edge_id == rest.edge_id
            assert orig.edge_type == rest.edge_type
            assert orig.confidence == rest.confidence

    @given(snap=snapshot_st())
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_snapshot_roundtrip_attachments_preserve_fields_property(self, snap: IRSnapshot):
        """HAPPY: nested IRAttachment roundtrips preserve all fields."""
        data = snap.to_dict()
        restored = IRSnapshot.from_dict(data)
        for orig, rest in zip(snap.attachments, restored.attachments, strict=True):
            assert orig.attachment_id == rest.attachment_id
            assert orig.target_id == rest.target_id
            assert orig.attachment_type == rest.attachment_type
            assert orig.payload == rest.payload

    @given(repo=identifier, snap_id=st.builds(lambda x: f"snap:{x}", identifier))
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_snapshot_defaults_property(self, repo: str, snap_id: str):
        """HAPPY: IRSnapshot with only required fields gets proper defaults."""
        snap = IRSnapshot(repo_name=repo, snapshot_id=snap_id)
        assert snap.branch is None
        assert snap.commit_id is None
        assert snap.tree_id is None
        assert snap.documents == []
        assert snap.symbols == []
        assert snap.occurrences == []
        assert snap.edges == []
        assert snap.attachments == []
        assert snap.metadata == {}

    @given(repo=identifier, snap_id=st.builds(lambda x: f"snap:{x}", identifier))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_snapshot_from_dict_missing_optional_keys_property(self, repo: str, snap_id: str):
        """EDGE: from_dict with minimal data uses .get() defaults for optional fields."""
        data = {"repo_name": repo, "snapshot_id": snap_id}
        snap = IRSnapshot.from_dict(data)
        assert snap.repo_name == repo
        assert snap.snapshot_id == snap_id
        assert snap.branch is None
        assert snap.commit_id is None
        assert snap.tree_id is None
        assert snap.documents == []
        assert snap.symbols == []
        assert snap.occurrences == []
        assert snap.edges == []
        assert snap.attachments == []
        assert snap.metadata == {}

    @given(repo=identifier, snap_id=st.builds(lambda x: f"snap:{x}", identifier))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_snapshot_from_dict_empty_lists_property(self, repo: str, snap_id: str):
        """EDGE: from_dict with explicitly empty nested lists."""
        data = {
            "repo_name": repo,
            "snapshot_id": snap_id,
            "documents": [],
            "symbols": [],
            "occurrences": [],
            "edges": [],
            "attachments": [],
            "metadata": {},
        }
        snap = IRSnapshot.from_dict(data)
        assert snap.documents == []
        assert snap.symbols == []
        assert snap.occurrences == []
        assert snap.edges == []
        assert snap.attachments == []

    @given(snap=snapshot_st())
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_snapshot_to_dict_lists_not_mutated_property(self, snap: IRSnapshot):
        """HAPPY: calling to_dict does not mutate the snapshot's internal lists."""
        original_doc_count = len(snap.documents)
        original_sym_count = len(snap.symbols)
        original_attachment_count = len(snap.attachments)
        _ = snap.to_dict()
        assert len(snap.documents) == original_doc_count
        assert len(snap.symbols) == original_sym_count
        assert len(snap.attachments) == original_attachment_count

    @given(snap=snapshot_st())
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_snapshot_to_dict_keys_are_stable_property(self, snap: IRSnapshot):
        """HAPPY: to_dict always produces the same set of top-level keys."""
        data = snap.to_dict()
        expected_keys = {
            "schema_version",
            "repo_name",
            "snapshot_id",
            "branch",
            "commit_id",
            "tree_id",
            "units",
            "supports",
            "relations",
            "embeddings",
            "metadata",
        }
        assert set(data.keys()) == expected_keys

    @given(snap=snapshot_st())
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_snapshot_roundtrip_occurrences_preserve_fields_property(self, snap: IRSnapshot):
        """HAPPY: nested IROccurrence roundtrips preserve all fields."""
        data = snap.to_dict()
        restored = IRSnapshot.from_dict(data)
        for orig, rest in zip(snap.occurrences, restored.occurrences, strict=True):
            assert orig.occurrence_id == rest.occurrence_id
            assert orig.role == rest.role
            assert orig.metadata == rest.metadata

    @given(snap=snapshot_st())
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_snapshot_to_dict_documents_are_dicts_property(self, snap: IRSnapshot):
        """HAPPY: to_dict serializes each unit as a plain dict, not a dataclass."""
        data = snap.to_dict()
        for unit_data in data["units"]:
            assert isinstance(unit_data, dict)
            assert "source_set" in unit_data
            assert isinstance(unit_data["source_set"], list)

    @given(snap=snapshot_st())
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_snapshot_to_dict_symbols_are_dicts_property(self, snap: IRSnapshot):
        """HAPPY: to_dict serializes each unit (including symbols) as a plain dict with source_set as list."""
        data = snap.to_dict()
        symbol_units = [
            u for u in data["units"] if u.get("kind") not in ("file", "doc")
        ]
        for sym_data in symbol_units:
            assert isinstance(sym_data, dict)
            assert "source_set" in sym_data
            assert isinstance(sym_data["source_set"], list)

    @given(snap=snapshot_st())
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_snapshot_metadata_roundtrip_property(self, snap: IRSnapshot):
        """HAPPY: snapshot-level metadata survives roundtrip."""
        data = snap.to_dict()
        restored = IRSnapshot.from_dict(data)
        assert restored.metadata == snap.metadata

    @given(repo=identifier, snap_id=st.builds(lambda x: f"snap:{x}", identifier))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_snapshot_empty_roundtrip_property(self, repo: str, snap_id: str):
        """EDGE: snapshot with no children roundtrips cleanly."""
        snap = IRSnapshot(repo_name=repo, snapshot_id=snap_id)
        data = snap.to_dict()
        restored = IRSnapshot.from_dict(data)
        assert restored.documents == []
        assert restored.symbols == []
        assert restored.occurrences == []
        assert restored.edges == []
        assert restored.metadata == {}
