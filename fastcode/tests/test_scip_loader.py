"""Tests for scip_loader module."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import tempfile
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

try:
    _has_protobuf = importlib.util.find_spec("google.protobuf") is not None
except ModuleNotFoundError:
    _has_protobuf = False

from fastcode.scip_loader import (
    _scip_kind_to_str,
    _symbol_role_to_str,
    load_scip_artifact,
)
from fastcode.scip_models import SCIPIndex

requires_protobuf = pytest.mark.skipif(
    not _has_protobuf, reason="protobuf not installed"
)

# --- Strategies ---

small_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_/",
    min_size=1,
    max_size=30,
)


# --- Helpers ---


def _write_json_file(tmpdir: str, name: str, data: dict[str, Any]) -> str:
    path = os.path.join(tmpdir, name)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _minimal_scip_dict() -> dict[str, Any]:
    return {
        "documents": [
            {
                "path": "a.py",
                "language": "python",
                "symbols": [
                    {"symbol": "pkg foo.", "name": "foo", "kind": "function"},
                ],
                "occurrences": [
                    {
                        "symbol": "pkg foo.",
                        "role": "definition",
                        "range": [1, 0, 1, 10],
                    },
                ],
            }
        ],
    }


# --- Properties ---


class TestSymbolRoleToStr:
    @given(role_bitmask=st.integers(min_value=0, max_value=255))
    @settings(max_examples=30)
    def test_always_returns_valid_role_property(self, role_bitmask: int):
        """HAPPY: _symbol_role_to_str always returns a known role string."""
        result = _symbol_role_to_str(role_bitmask)
        assert result in (
            "definition",
            "import",
            "write_access",
            "forward_definition",
            "reference",
        )

    def test_definition_role_property(self):
        assert _symbol_role_to_str(1) == "definition"

    def test_import_role_property(self):
        assert _symbol_role_to_str(2) == "import"

    def test_write_access_role_property(self):
        assert _symbol_role_to_str(4) == "write_access"

    def test_forward_definition_role_property(self):
        assert _symbol_role_to_str(64) == "forward_definition"

    def test_reference_fallback_property(self):
        assert _symbol_role_to_str(0) == "reference"

    def test_reference_fallback_other_bits_property(self):
        assert _symbol_role_to_str(8) == "reference"

    @given(bitmask=st.integers(min_value=0, max_value=255))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_definition_has_priority_over_import_property(self, bitmask: int):
        """EDGE: definition bit (1) takes priority when multiple bits set."""
        if bitmask & 1:
            assert _symbol_role_to_str(bitmask) == "definition"

    @given(bitmask=st.integers(min_value=0, max_value=255))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_import_priority_over_write_property(self, bitmask: int):
        """EDGE: import bit (2) has priority over write_access (4)."""
        if not (bitmask & 1) and (bitmask & 2):
            assert _symbol_role_to_str(bitmask) == "import"


class TestScipKindToStr:
    @given(kind_value=st.integers(min_value=0, max_value=30))
    @settings(max_examples=30)
    def test_always_returns_string_property(self, kind_value: int):
        """HAPPY: _scip_kind_to_str always returns a string."""
        result = _scip_kind_to_str(kind_value)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_kind_returns_symbol_property(self):
        """HAPPY: unknown kind value returns 'symbol'."""
        assert _scip_kind_to_str(999) == "symbol"

    @given(kind_value=st.integers(min_value=0, max_value=30))
    @settings(max_examples=30)
    @pytest.mark.edge
    def test_result_is_known_kind_or_symbol_property(self, kind_value: int):
        """EDGE: result is always from the known kind set or 'symbol'."""
        known = {
            "function",
            "method",
            "class",
            "interface",
            "enum",
            "enum_member",
            "variable",
            "constant",
            "property",
            "type",
            "macro",
            "module",
            "namespace",
            "package",
            "parameter",
            "type_parameter",
            "constructor",
            "struct",
            "symbol",
        }
        assert _scip_kind_to_str(kind_value) in known


class TestLoadScipArtifactJson:
    def test_load_valid_json_property(self):
        """HAPPY: loading a valid .json SCIP artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json_file(tmpdir, "index.json", _minimal_scip_dict())
            result = load_scip_artifact(path)
            assert isinstance(result, SCIPIndex)
            assert len(result.documents) == 1
            assert result.documents[0].path == "a.py"

    def test_load_scip_json_extension_property(self):
        """HAPPY: loading a .scip.json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json_file(tmpdir, "index.scip.json", _minimal_scip_dict())
            result = load_scip_artifact(path)
            assert isinstance(result, SCIPIndex)

    def test_load_empty_documents_property(self):
        """HAPPY: loading JSON with empty documents list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json_file(tmpdir, "empty.json", {"documents": []})
            result = load_scip_artifact(path)
            assert len(result.documents) == 0

    def test_load_preserves_symbol_data_property(self):
        """HAPPY: symbol fields survive JSON roundtrip."""
        data = {
            "documents": [
                {
                    "path": "a.py",
                    "language": "python",
                    "symbols": [
                        {"symbol": "pkg foo.", "name": "foo", "kind": "function"},
                    ],
                    "occurrences": [],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json_file(tmpdir, "sym.json", data)
            result = load_scip_artifact(path)
            assert result.documents[0].symbols[0].symbol == "pkg foo."
            assert result.documents[0].symbols[0].name == "foo"
            assert result.documents[0].symbols[0].kind == "function"

    def test_load_preserves_occurrence_data_property(self):
        """HAPPY: occurrence fields survive JSON roundtrip."""
        data = {
            "documents": [
                {
                    "path": "a.py",
                    "language": "python",
                    "symbols": [],
                    "occurrences": [
                        {
                            "symbol": "pkg bar.",
                            "role": "reference",
                            "range": [10, 0, 10, 5],
                        },
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json_file(tmpdir, "occ.json", data)
            result = load_scip_artifact(path)
            assert result.documents[0].occurrences[0].symbol == "pkg bar."
            assert result.documents[0].occurrences[0].role == "reference"

    @pytest.mark.negative
    def test_load_missing_file_raises_property(self):
        """EDGE: loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_scip_artifact("/nonexistent/path/index.json")

    @pytest.mark.negative
    def test_load_unsupported_extension_raises_property(self):
        """EDGE: loading file with unsupported extension raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.xyz")
            with open(path, "w") as f:
                f.write("{}")
            with pytest.raises(ValueError, match="Unsupported"):
                load_scip_artifact(path)

    @pytest.mark.negative
    def test_load_invalid_json_raises_property(self):
        """EDGE: loading file with invalid JSON raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.json")
            with open(path, "w") as f:
                f.write("{not valid json")
            with pytest.raises(Exception, match=r".*"):
                load_scip_artifact(path)

    @pytest.mark.negative
    def test_load_scip_extension_without_protobuf_raises_property(self):
        """EDGE: .scip file without protobuf support or CLI raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.scip")
            with open(path, "wb") as f:
                f.write(b"\x00\x01\x02")
            # Will fail since protobuf can't parse random bytes and scip CLI not in PATH
            with pytest.raises((ValueError, RuntimeError, ImportError, OSError)):
                load_scip_artifact(path)

    @given(
        n_docs=st.integers(min_value=0, max_value=3),
        n_syms=st.integers(min_value=0, max_value=3),
    )
    @settings(max_examples=15)
    def test_load_variable_document_count_property(self, n_docs: int, n_syms: int):
        """HAPPY: loading JSON with varying document/symbol counts."""
        data = {
            "documents": [
                {
                    "path": f"f{i}.py",
                    "language": "python",
                    "symbols": [
                        {"symbol": f"pkg s{j}.", "name": f"s{j}", "kind": "function"}
                        for j in range(n_syms)
                    ],
                    "occurrences": [],
                }
                for i in range(n_docs)
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json_file(tmpdir, "var.json", data)
            result = load_scip_artifact(path)
            assert len(result.documents) == n_docs
            for doc in result.documents:
                assert len(doc.symbols) == n_syms

    @given(text=small_text)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_load_with_minimal_dict_property(self, text: str):
        """EDGE: loading JSON with minimal valid structure."""
        data = {"documents": [{"path": f"{text}.py", "symbols": [], "occurrences": []}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json_file(tmpdir, "min.json", data)
            result = load_scip_artifact(path)
            assert len(result.documents) == 1

    @pytest.mark.edge
    def test_role_bitmask_combined_property(self):
        """EDGE: combined definition + import bit returns definition (priority)."""
        assert _symbol_role_to_str(1 | 2) == "definition"

    @pytest.mark.edge
    def test_role_high_bits_only_property(self):
        """EDGE: bits above defined roles return reference."""
        assert _symbol_role_to_str(128) == "reference"

    @pytest.mark.edge
    def test_kind_zero_property(self):
        """EDGE: kind value 0 returns mapped value or symbol."""
        result = _scip_kind_to_str(0)
        assert isinstance(result, str)

    @pytest.mark.edge
    def test_load_json_with_no_language_property(self):
        """EDGE: document without language field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "documents": [{"path": "unknown.xyz", "symbols": [], "occurrences": []}]
            }
            path = _write_json_file(tmpdir, "nolang.json", data)
            result = load_scip_artifact(path)
            assert result.documents[0].language is None

    @pytest.mark.edge
    def test_load_json_with_extra_fields_property(self):
        """EDGE: extra fields in JSON are ignored gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"documents": [], "extra_field": "ignored", "version": 999}
            path = _write_json_file(tmpdir, "extra.json", data)
            result = load_scip_artifact(path)
            assert len(result.documents) == 0

    @pytest.mark.edge
    def test_load_empty_json_object_property(self):
        """EDGE: empty JSON object produces empty index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json_file(tmpdir, "empty.json", {})
            result = load_scip_artifact(path)
            assert isinstance(result, SCIPIndex)


# --- Binary SCIP protobuf tests (from test_scip_binary.py) ---


@requires_protobuf
def test_scip_pb2_module_importable():
    """Protobuf bindings module must be importable."""
    from fastcode.scip_pb2 import Index

    idx = Index()
    assert idx.metadata.tool_info.name == ""


@requires_protobuf
def test_load_binary_scip_artifact(tmp_path: pathlib.Path):
    """Binary .scip files parse without external CLI."""
    from fastcode.scip_pb2 import Index

    # Build a minimal binary SCIP index
    idx = Index()
    idx.metadata.version = 0
    idx.metadata.tool_info.name = "test-indexer"
    idx.metadata.tool_info.version = "1.0.0"
    doc = idx.documents.add()
    doc.relative_path = "src/main.py"
    doc.language = "python"
    sym = doc.symbols.add()
    sym.symbol = "scip test src/main.py main()`"
    sym.display_name = "main"

    # Write binary
    scip_path = tmp_path / "index.scip"
    scip_path.write_bytes(idx.SerializeToString())

    result = load_scip_artifact(str(scip_path))

    assert len(result.documents) == 1
    assert result.documents[0].path == "src/main.py"
    assert result.documents[0].language == "python"
    assert len(result.documents[0].symbols) == 1
    assert result.documents[0].symbols[0].name == "main"


@requires_protobuf
def test_binary_scip_with_occurrences(tmp_path: pathlib.Path):
    """Binary SCIP with occurrences converts correctly."""
    from fastcode.scip_pb2 import Index, SymbolInformation

    idx = Index()
    idx.metadata.version = 0
    idx.metadata.tool_info.name = "scip-java"
    idx.metadata.tool_info.version = "2.1.0"

    doc = idx.documents.add()
    doc.relative_path = "src/Main.java"
    doc.language = "java"

    sym = doc.symbols.add()
    sym.symbol = "scip java com/example Main main(java.lang.String[])`"
    sym.display_name = "main"
    sym.kind = SymbolInformation.Kind.Method

    # Definition occurrence (symbol_roles bit 0 = definition)
    occ = doc.occurrences.add()
    occ.symbol = sym.symbol
    occ.range.extend([10, 4, 10, 20])
    occ.symbol_roles = 1  # Definition

    # Reference occurrence
    occ2 = doc.occurrences.add()
    occ2.symbol = sym.symbol
    occ2.range.extend([20, 2, 20, 8])
    occ2.symbol_roles = 0  # Reference

    scip_path = tmp_path / "index.scip"
    scip_path.write_bytes(idx.SerializeToString())

    result = load_scip_artifact(str(scip_path))

    assert result.indexer_name == "scip-java"
    assert result.indexer_version == "2.1.0"
    assert len(result.documents) == 1

    doc_result = result.documents[0]
    assert doc_result.path == "src/Main.java"
    assert doc_result.language == "java"
    assert len(doc_result.symbols) == 1
    assert doc_result.symbols[0].name == "main"
    assert doc_result.symbols[0].kind == "method"

    assert len(doc_result.occurrences) == 2
    assert doc_result.occurrences[0].role == "definition"
    assert doc_result.occurrences[0].range == [10, 4, 10, 20]
    assert doc_result.occurrences[1].role == "reference"


@requires_protobuf
def test_binary_scip_empty_index(tmp_path: pathlib.Path):
    """Empty SCIP index produces empty SCIPIndex."""
    from fastcode.scip_pb2 import Index

    idx = Index()
    idx.metadata.version = 0

    scip_path = tmp_path / "empty.scip"
    scip_path.write_bytes(idx.SerializeToString())

    result = load_scip_artifact(str(scip_path))

    assert len(result.documents) == 0
    assert result.indexer_name is None or result.indexer_name == ""


@requires_protobuf
def test_binary_scip_to_ir_round_trip(tmp_path: pathlib.Path):
    """Binary SCIP -> SCIPIndex -> IRSnapshot produces valid IR."""
    from fastcode.adapters.scip_to_ir import build_ir_from_scip
    from fastcode.scip_pb2 import Index, SymbolInformation

    idx = Index()
    idx.metadata.version = 0
    idx.metadata.tool_info.name = "scip-go"
    idx.metadata.tool_info.version = "0.2.0"

    doc = idx.documents.add()
    doc.relative_path = "main.go"
    doc.language = "go"

    sym = doc.symbols.add()
    sym.symbol = "scip golang example/main main()`"
    sym.display_name = "main"
    sym.kind = SymbolInformation.Kind.Function

    occ = doc.occurrences.add()
    occ.symbol = sym.symbol
    occ.range.extend([5, 0, 5, 12])
    occ.symbol_roles = 1  # Definition

    scip_path = tmp_path / "index.scip"
    scip_path.write_bytes(idx.SerializeToString())

    scip_index = load_scip_artifact(str(scip_path))
    snapshot = build_ir_from_scip(
        repo_name="example",
        snapshot_id="snap:example:abc123",
        scip_index=scip_index,
    )

    assert len(snapshot.documents) == 1
    assert snapshot.documents[0].path == "main.go"
    assert len(snapshot.symbols) == 1
    assert snapshot.symbols[0].display_name == "main"
    assert snapshot.symbols[0].kind == "function"
    assert snapshot.symbols[0].source_priority == 100
    assert len(snapshot.occurrences) == 1
    assert snapshot.occurrences[0].role == "definition"
    assert len(snapshot.edges) == 2  # containment + ref edge


def test_run_scip_python_index_delegates_to_scip_indexers(tmp_path: pathlib.Path):
    """run_scip_python_index delegates to scip_indexers.run_scip_indexer."""
    from fastcode.scip_loader import run_scip_python_index

    with patch(
        "fastcode.scip_indexers.run_scip_indexer", return_value="/fake/output.scip"
    ) as mock_run:
        result = run_scip_python_index(str(tmp_path), "/fake/output.scip")

    mock_run.assert_called_once_with("python", str(tmp_path), "/fake/output.scip")
    assert result == "/fake/output.scip"
