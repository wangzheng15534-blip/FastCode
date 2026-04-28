"""Property-based tests for scip_models dataclass invariants.

Covers: SCIPOccurrence, SCIPSymbol, SCIPDocument, SCIPIndex, SCIPArtifactRef.
Tests roundtrip serialization, defaults, None coercion, filtering, and metadata.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.scip_models import (
    SCIPArtifactRef,
    SCIPDocument,
    SCIPIndex,
    SCIPOccurrence,
    SCIPSymbol,
)

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


# --- TestSCIPOccurrenceRoundtrip ---


class TestSCIPOccurrenceRoundtrip:
    @given(occ=scip_occurrence_st)
    @settings(max_examples=50)
    @pytest.mark.basic
    def test_occurrence_roundtrip_property(self, occ: SCIPOccurrence):
        """HAPPY: SCIPOccurrence.to_dict() -> from_dict() roundtrip is stable."""
        data = occ.to_dict()
        restored = SCIPOccurrence.from_dict(data)
        # Second roundtrip must be identical (handles empty list normalization)
        data2 = restored.to_dict()
        restored2 = SCIPOccurrence.from_dict(data2)
        assert restored2.symbol == restored.symbol
        assert restored2.role == restored.role
        assert restored2.range == restored.range

    @pytest.mark.edge
    def test_from_dict_empty_dict_property(self):
        """EDGE: from_dict with empty dict produces valid defaults."""
        occ = SCIPOccurrence.from_dict({})
        assert occ.symbol == ""
        assert occ.role == "reference"
        assert occ.range == [None, None, None, None]

    @given(symbol=st.none() | st.just(""), role=st.none(), range_val=st.none())
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_from_dict_none_values_coerced_property(
        self, symbol: str, role: str, range_val: Any
    ):
        """EDGE: from_dict with None values coerces string fields to empty/default."""
        data = {"symbol": symbol, "role": role, "range": range_val}
        occ = SCIPOccurrence.from_dict(data)
        assert isinstance(occ.symbol, str)
        assert isinstance(occ.role, str)

    @given(
        symbol=identifier,
        role=st.sampled_from(["definition", "reference", "implementation"]),
    )
    @settings(max_examples=15)
    @pytest.mark.basic
    def test_range_defaults_to_none_tuple_property(self, symbol: str, role: str):
        """HAPPY: SCIPOccurrence with no range gets [None, None, None, None]."""
        occ = SCIPOccurrence(symbol=symbol, role=role)
        assert occ.range == [None, None, None, None]

    @given(occ=scip_occurrence_st)
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_to_dict_keys_stable_property(self, occ: SCIPOccurrence):
        """HAPPY: to_dict always produces the same set of keys."""
        data = occ.to_dict()
        assert set(data.keys()) == {"symbol", "role", "range"}


# --- TestSCIPSymbolRoundtrip ---


class TestSCIPSymbolRoundtrip:
    @given(sym=scip_symbol_st)
    @settings(max_examples=50)
    @pytest.mark.basic
    def test_symbol_roundtrip_property(self, sym: SCIPSymbol):
        """HAPPY: SCIPSymbol.to_dict() -> from_dict() roundtrip is stable."""
        data = sym.to_dict()
        restored = SCIPSymbol.from_dict(data)
        # Second roundtrip must be identical (handles empty list normalization)
        data2 = restored.to_dict()
        restored2 = SCIPSymbol.from_dict(data2)
        assert restored2.symbol == restored.symbol
        assert restored2.name == restored.name
        assert restored2.kind == restored.kind
        assert restored2.qualified_name == restored.qualified_name
        assert restored2.signature == restored.signature
        assert restored2.range == restored.range

    @pytest.mark.edge
    def test_from_dict_empty_dict_property(self):
        """EDGE: from_dict with empty dict produces valid defaults."""
        sym = SCIPSymbol.from_dict({})
        assert sym.symbol == ""
        assert sym.name is None
        assert sym.kind is None
        assert sym.qualified_name is None
        assert sym.signature is None
        assert sym.range == [None, None, None, None]

    @pytest.mark.edge
    def test_from_dict_none_symbol_coerced_property(self):
        """EDGE: from_dict with None symbol coerces to empty string."""
        sym = SCIPSymbol.from_dict({"symbol": None})
        assert sym.symbol == ""

    @given(sym=scip_symbol_st)
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_to_dict_keys_stable_property(self, sym: SCIPSymbol):
        """HAPPY: to_dict always produces the same set of keys."""
        data = sym.to_dict()
        assert set(data.keys()) == {
            "symbol",
            "name",
            "kind",
            "qualified_name",
            "signature",
            "range",
        }

    @given(symbol=identifier)
    @settings(max_examples=10)
    @pytest.mark.basic
    def test_defaults_with_symbol_only_property(self, symbol: str):
        """HAPPY: SCIPSymbol with only required field gets None defaults."""
        sym = SCIPSymbol(symbol=symbol)
        assert sym.name is None
        assert sym.kind is None
        assert sym.qualified_name is None
        assert sym.signature is None
        assert sym.range == [None, None, None, None]


# --- TestSCIPDocumentRoundtrip ---


class TestSCIPDocumentRoundtrip:
    @given(doc=scip_document_st)
    @settings(max_examples=50)
    @pytest.mark.basic
    def test_document_roundtrip_property(self, doc: SCIPDocument):
        """HAPPY: SCIPDocument.to_dict() -> from_dict() roundtrip preserves all fields."""
        data = doc.to_dict()
        restored = SCIPDocument.from_dict(data)
        assert restored.path == doc.path
        assert restored.language == doc.language
        assert len(restored.symbols) == len(doc.symbols)
        assert len(restored.occurrences) == len(doc.occurrences)

    @pytest.mark.edge
    def test_from_dict_empty_dict_property(self):
        """EDGE: from_dict with empty dict produces valid defaults."""
        doc = SCIPDocument.from_dict({})
        assert doc.path == ""
        assert doc.language is None
        assert doc.symbols == []
        assert doc.occurrences == []

    @given(doc=scip_document_st)
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_nested_symbols_roundtrip_property(self, doc: SCIPDocument):
        """HAPPY: nested symbols survive document roundtrip."""
        data = doc.to_dict()
        restored = SCIPDocument.from_dict(data)
        for orig, rest in zip(doc.symbols, restored.symbols, strict=True):
            assert orig.symbol == rest.symbol
            assert orig.name == rest.name
            assert orig.kind == rest.kind

    @given(doc=scip_document_st)
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_nested_occurrences_roundtrip_property(self, doc: SCIPDocument):
        """HAPPY: nested occurrences survive document roundtrip."""
        data = doc.to_dict()
        restored = SCIPDocument.from_dict(data)
        for orig, rest in zip(doc.occurrences, restored.occurrences, strict=True):
            assert orig.symbol == rest.symbol
            assert orig.role == rest.role

    @given(doc=scip_document_st)
    @settings(max_examples=15)
    @pytest.mark.basic
    def test_to_dict_keys_stable_property(self, doc: SCIPDocument):
        """HAPPY: to_dict always produces the same set of keys."""
        data = doc.to_dict()
        assert set(data.keys()) == {"path", "language", "symbols", "occurrences"}

    @given(path=st.none())
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_from_dict_none_path_coerced_property(self, path: str):
        """EDGE: from_dict with None path coerces to empty string."""
        doc = SCIPDocument.from_dict({"path": path})
        assert doc.path == ""


# --- TestSCIPIndexRoundtrip ---


class TestSCIPIndexRoundtrip:
    @given(index=scip_index_st)
    @settings(max_examples=50)
    @pytest.mark.basic
    def test_index_roundtrip_property(self, index: SCIPIndex):
        """HAPPY: SCIPIndex.to_dict() -> from_dict() roundtrip preserves structure."""
        data = index.to_dict()
        restored = SCIPIndex.from_dict(data)
        assert restored.indexer_name == index.indexer_name
        assert restored.indexer_version == index.indexer_version
        assert len(restored.documents) == len(index.documents)
        assert restored.metadata == index.metadata

    @pytest.mark.edge
    def test_from_dict_empty_dict_property(self):
        """EDGE: from_dict with empty dict produces valid defaults."""
        index = SCIPIndex.from_dict({})
        assert index.documents == []
        assert index.indexer_name is None
        assert index.indexer_version is None
        assert index.metadata == {}

    @given(
        docs=st.lists(st.just("not_a_dict"), max_size=3),
        valid_docs=st.lists(scip_document_st, max_size=2),
    )
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_from_dict_filters_non_dict_entries_property(
        self, docs: list[Any], valid_docs: list[dict[str, Any]]
    ):
        """EDGE: SCIPIndex.from_dict filters non-dict entries from documents list."""
        raw_docs = docs + [d.to_dict() for d in valid_docs]
        index = SCIPIndex.from_dict({"documents": raw_docs})
        assert len(index.documents) == len(valid_docs)

    @given(
        extra_key=identifier.filter(
            lambda k: k not in {"documents", "indexer_name", "indexer_version"}
        ),
        extra_val=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=15)
    @pytest.mark.basic
    def test_from_dict_sends_extra_keys_to_metadata_property(
        self, extra_key: str, extra_val: int
    ):
        """HAPPY: SCIPIndex.from_dict sends extra keys to metadata dict."""
        index = SCIPIndex.from_dict({extra_key: extra_val})
        assert extra_key in index.metadata
        assert index.metadata[extra_key] == extra_val

    @given(index=scip_index_st)
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_to_dict_includes_metadata_property(self, index: SCIPIndex):
        """HAPPY: to_dict merges metadata keys into output dict."""
        data = index.to_dict()
        assert "documents" in data
        assert "indexer_name" in data
        assert "indexer_version" in data

    @given(index=scip_index_st)
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_roundtrip_preserves_document_count_property(self, index: SCIPIndex):
        """HAPPY: document count is preserved through roundtrip."""
        data = index.to_dict()
        restored = SCIPIndex.from_dict(data)
        assert len(restored.documents) == len(index.documents)

    @given(metadata=st.dictionaries(identifier, st.integers(), max_size=3))
    @settings(max_examples=15)
    @pytest.mark.basic
    def test_metadata_roundtrip_property(self, metadata: dict[str, int]):
        """HAPPY: SCIPIndex metadata survives roundtrip."""
        index = SCIPIndex(metadata=metadata)
        data = index.to_dict()
        restored = SCIPIndex.from_dict(data)
        assert restored.metadata == metadata


# --- TestSCIPArtifactRefRoundtrip ---


class TestSCIPArtifactRefRoundtrip:
    @given(
        snapshot_id=identifier,
        indexer_name=identifier,
        indexer_version=st.none() | st.just("1.0.0"),
        artifact_path=identifier,
        checksum=identifier,
        created_at=identifier,
    )
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_artifact_ref_roundtrip_property(
        self,
        snapshot_id: str,
        indexer_name: str,
        indexer_version: Any,
        artifact_path: str,
        checksum: str,
        created_at: str,
    ):
        """HAPPY: SCIPArtifactRef.to_dict() -> from_dict() roundtrip preserves all fields."""
        ref = SCIPArtifactRef(
            snapshot_id=snapshot_id,
            indexer_name=indexer_name,
            indexer_version=indexer_version,
            artifact_path=artifact_path,
            checksum=checksum,
            created_at=created_at,
        )
        data = ref.to_dict()
        restored = SCIPArtifactRef.from_dict(data)
        assert restored.snapshot_id == snapshot_id
        assert restored.indexer_name == indexer_name
        assert restored.indexer_version == indexer_version
        assert restored.artifact_path == artifact_path
        assert restored.checksum == checksum
        assert restored.created_at == created_at

    @pytest.mark.edge
    def test_from_dict_empty_dict_property(self):
        """EDGE: from_dict with empty dict produces empty string defaults."""
        ref = SCIPArtifactRef.from_dict({})
        assert ref.snapshot_id == ""
        assert ref.indexer_name == ""
        assert ref.indexer_version is None
        assert ref.artifact_path == ""
        assert ref.checksum == ""
        assert ref.created_at == ""

    @pytest.mark.edge
    def test_from_dict_none_values_coerced_to_empty_property(self):
        """EDGE: from_dict with None values coerces string fields to empty string."""
        data = {
            "snapshot_id": None,
            "indexer_name": None,
            "indexer_version": None,
            "artifact_path": None,
            "checksum": None,
            "created_at": None,
        }
        ref = SCIPArtifactRef.from_dict(data)
        assert ref.snapshot_id == ""
        assert ref.indexer_name == ""
        assert ref.artifact_path == ""
        assert ref.checksum == ""
        assert ref.created_at == ""

    @given(
        snapshot_id=identifier,
        indexer_name=identifier,
        artifact_path=identifier,
        checksum=identifier,
        created_at=identifier,
    )
    @settings(max_examples=15)
    @pytest.mark.basic
    def test_to_dict_keys_stable_property(
        self,
        snapshot_id: str,
        indexer_name: str,
        artifact_path: Any,
        checksum: Any,
        created_at: Any,
    ):
        """HAPPY: to_dict always produces the same set of keys."""
        ref = SCIPArtifactRef(
            snapshot_id=snapshot_id,
            indexer_name=indexer_name,
            indexer_version=None,
            artifact_path=artifact_path,
            checksum=checksum,
            created_at=created_at,
        )
        data = ref.to_dict()
        expected_keys = {
            "snapshot_id",
            "indexer_name",
            "indexer_version",
            "artifact_path",
            "checksum",
            "created_at",
        }
        assert set(data.keys()) == expected_keys
