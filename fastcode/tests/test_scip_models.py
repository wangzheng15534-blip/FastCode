"""Tests for fastcode.scip_models — metadata extraction, nested conversion, edge cases.

Previous version: 619 lines of to_dict/from_dict round-trips.
This version: tests actual conversion logic, reserved field separation, edge handling.
"""

from __future__ import annotations

from typing import Any

from fastcode.scip_models import (
    SCIPArtifactRef,
    SCIPDocument,
    SCIPIndex,
    SCIPOccurrence,
)


def _make_symbol(symbol_name: str = "my_func()", **overrides) -> dict[str, Any]:
    base = {
        "symbol": symbol_name,
        "name": symbol_name,
        "kind": "function",
    }
    base.update(overrides)
    return base


def _make_occurrence(**overrides) -> dict[str, Any]:
    base = {
        "symbol": "my_func()",
        "range": [1, 0, 1, 10],
        "role": "definition",
    }
    base.update(overrides)
    return base


def _make_document(
    path: str = "src/main.py",
    symbols: list[dict[str, Any]] | None = None,
    occurrences: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _sym = symbols if symbols is not None else [_make_symbol()]
    _occ = occurrences if occurrences is not None else [_make_occurrence()]
    return {
        "path": path,
        "language": "python",
        "symbols": _sym,
        "occurrences": _occ,
    }


def _make_index(
    documents: list[dict[str, Any]] | None = None, **overrides
) -> dict[str, Any]:
    _docs = documents if documents is not None else [_make_document()]
    base = {
        "indexer_name": "scip-python",
        "indexer_version": "1.2.3",
        "documents": _docs,
    }
    base.update(overrides)
    return base


# --- SCIPIndex metadata separation ---


class TestSCIPIndexMetadata:
    def test_reserved_fields_not_in_metadata(self):
        raw = _make_index(custom_field="extra_value")
        index = SCIPIndex.from_dict(raw)
        assert "indexer_name" not in index.metadata
        assert "indexer_version" not in index.metadata
        assert "documents" not in index.metadata

    def test_unknown_fields_go_to_metadata(self):
        raw = _make_index(custom_tool="cargo-scip", schema_version="3")
        index = SCIPIndex.from_dict(raw)
        assert index.metadata["custom_tool"] == "cargo-scip"
        assert index.metadata["schema_version"] == "3"

    def test_metadata_roundtrips_through_to_dict(self):
        raw = _make_index(extra="value")
        index = SCIPIndex.from_dict(raw)
        out = index.to_dict()
        assert out["extra"] == "value"
        assert out["indexer_name"] == "scip-python"

    def test_empty_documents_list(self):
        index = SCIPIndex.from_dict(_make_index(documents=[]))
        assert len(index.documents) == 0

    def test_non_dict_items_in_documents_filtered(self):
        raw = _make_index(documents=[_make_document(), "not_a_dict", None, 42])
        index = SCIPIndex.from_dict(raw)
        assert len(index.documents) == 1


# --- SCIPDocument nested conversion ---


class TestSCIPDocumentConversion:
    def test_converts_nested_symbols(self):
        raw = _make_document(
            symbols=[
                _make_symbol("func_a()"),
                _make_symbol("func_b()"),
            ]
        )
        doc = SCIPDocument.from_dict(raw)
        assert len(doc.symbols) == 2
        assert doc.symbols[0].symbol == "func_a()"
        assert doc.symbols[1].symbol == "func_b()"

    def test_converts_nested_occurrences(self):
        raw = _make_document(
            occurrences=[
                _make_occurrence(symbol="f()", role="definition"),
                _make_occurrence(symbol="f()", role="reference"),
            ]
        )
        doc = SCIPDocument.from_dict(raw)
        assert len(doc.occurrences) == 2
        assert doc.occurrences[0].role == "definition"
        assert doc.occurrences[1].role == "reference"

    def test_empty_symbols_and_occurrences(self):
        raw = _make_document(symbols=[], occurrences=[])
        doc = SCIPDocument.from_dict(raw)
        assert len(doc.symbols) == 0
        assert len(doc.occurrences) == 0


# --- Range handling ---


class TestRangeHandling:
    def test_range_serialized_as_list(self):
        occ = SCIPOccurrence.from_dict(_make_occurrence(range=[5, 0, 5, 20]))
        out = occ.to_dict()
        assert isinstance(out["range"], list)
        assert out["range"] == [5, 0, 5, 20]

    def test_none_range_handled(self):
        occ = SCIPOccurrence.from_dict(_make_occurrence())
        occ.range = None
        out = occ.to_dict()
        # to_dict normalizes None range to empty tuple
        assert out["range"] == [None, None, None, None]


# --- SCIPArtifactRef ---


class TestSCIPArtifactRef:
    def test_from_dict_roundtrip(self):
        ref = SCIPArtifactRef.from_dict(
            {
                "snapshot_id": "snap:repo:1",
                "indexer_name": "scip-python",
                "indexer_version": "1.0.0",
                "artifact_path": "/tmp/index.scip.json",
                "checksum": "abc",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
        assert ref.snapshot_id == "snap:repo:1"
        assert ref.indexer_name == "scip-python"
        assert ref.artifact_path == "/tmp/index.scip.json"

    def test_from_dict_empty_dict_coerces_to_empty_strings(self):
        ref = SCIPArtifactRef.from_dict({})
        assert ref.snapshot_id == ""
        assert ref.indexer_name == ""
        assert ref.indexer_version is None
        assert ref.artifact_path == ""
        assert ref.checksum == ""
        assert ref.created_at == ""


# --- Smoke round-trip (keep exactly 1) ---


class TestSmokeRoundTrip:
    def test_full_index_roundtrip_smoke(self):
        """One smoke test: full index survives to_dict -> from_dict cycle."""
        raw = _make_index(extra_meta="test")
        index = SCIPIndex.from_dict(raw)
        restored = SCIPIndex.from_dict(index.to_dict())
        assert restored.indexer_name == index.indexer_name
        assert len(restored.documents) == len(index.documents)
        assert restored.metadata["extra_meta"] == "test"
