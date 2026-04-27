"""Property-based tests for scip_loader module."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.scip_loader import (
    _scip_kind_to_str,
    _symbol_role_to_str,
    load_scip_artifact,
)
from fastcode.scip_models import SCIPIndex

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


@pytest.mark.property
class TestSymbolRoleToStr:
    @given(role_bitmask=st.integers(min_value=0, max_value=255))
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_always_returns_valid_role(self, role_bitmask: int):
        """HAPPY: _symbol_role_to_str always returns a known role string."""
        result = _symbol_role_to_str(role_bitmask)
        assert result in (
            "definition",
            "import",
            "write_access",
            "forward_definition",
            "reference",
        )

    @pytest.mark.basic
    def test_definition_role(self):
        assert _symbol_role_to_str(1) == "definition"

    @pytest.mark.basic
    def test_import_role(self):
        assert _symbol_role_to_str(2) == "import"

    @pytest.mark.basic
    def test_write_access_role(self):
        assert _symbol_role_to_str(4) == "write_access"

    @pytest.mark.basic
    def test_forward_definition_role(self):
        assert _symbol_role_to_str(64) == "forward_definition"

    @pytest.mark.basic
    def test_reference_fallback(self):
        assert _symbol_role_to_str(0) == "reference"

    @pytest.mark.basic
    def test_reference_fallback_other_bits(self):
        assert _symbol_role_to_str(8) == "reference"

    @given(bitmask=st.integers(min_value=0, max_value=255))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_definition_has_priority_over_import(self, bitmask: int):
        """EDGE: definition bit (1) takes priority when multiple bits set."""
        if bitmask & 1:
            assert _symbol_role_to_str(bitmask) == "definition"

    @given(bitmask=st.integers(min_value=0, max_value=255))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_import_priority_over_write(self, bitmask: int):
        """EDGE: import bit (2) has priority over write_access (4)."""
        if not (bitmask & 1) and (bitmask & 2):
            assert _symbol_role_to_str(bitmask) == "import"


@pytest.mark.property
class TestScipKindToStr:
    @given(kind_value=st.integers(min_value=0, max_value=30))
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_always_returns_string(self, kind_value: int):
        """HAPPY: _scip_kind_to_str always returns a string."""
        result = _scip_kind_to_str(kind_value)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.basic
    def test_unknown_kind_returns_symbol(self):
        """HAPPY: unknown kind value returns 'symbol'."""
        assert _scip_kind_to_str(999) == "symbol"

    @given(kind_value=st.integers(min_value=0, max_value=30))
    @settings(max_examples=30)
    @pytest.mark.edge
    def test_result_is_known_kind_or_symbol(self, kind_value: int):
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


@pytest.mark.property
class TestLoadScipArtifactJson:
    @pytest.mark.basic
    def test_load_valid_json(self):
        """HAPPY: loading a valid .json SCIP artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json_file(tmpdir, "index.json", _minimal_scip_dict())
            result = load_scip_artifact(path)
            assert isinstance(result, SCIPIndex)
            assert len(result.documents) == 1
            assert result.documents[0].path == "a.py"

    @pytest.mark.basic
    def test_load_scip_json_extension(self):
        """HAPPY: loading a .scip.json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json_file(tmpdir, "index.scip.json", _minimal_scip_dict())
            result = load_scip_artifact(path)
            assert isinstance(result, SCIPIndex)

    @pytest.mark.basic
    def test_load_empty_documents(self):
        """HAPPY: loading JSON with empty documents list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json_file(tmpdir, "empty.json", {"documents": []})
            result = load_scip_artifact(path)
            assert len(result.documents) == 0

    @pytest.mark.basic
    def test_load_preserves_symbol_data(self):
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

    @pytest.mark.basic
    def test_load_preserves_occurrence_data(self):
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
    def test_load_missing_file_raises(self):
        """EDGE: loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_scip_artifact("/nonexistent/path/index.json")

    @pytest.mark.negative
    def test_load_unsupported_extension_raises(self):
        """EDGE: loading file with unsupported extension raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.xyz")
            with open(path, "w") as f:
                f.write("{}")
            with pytest.raises(ValueError, match="Unsupported"):
                load_scip_artifact(path)

    @pytest.mark.negative
    def test_load_invalid_json_raises(self):
        """EDGE: loading file with invalid JSON raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.json")
            with open(path, "w") as f:
                f.write("{not valid json")
            with pytest.raises(Exception, match=r".*"):
                load_scip_artifact(path)

    @pytest.mark.negative
    def test_load_scip_extension_without_protobuf_raises(self):
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
    @pytest.mark.basic
    def test_load_variable_document_count(self, n_docs: int, n_syms: int):
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
    def test_load_with_minimal_dict(self, text: str):
        """EDGE: loading JSON with minimal valid structure."""
        data = {"documents": [{"path": f"{text}.py", "symbols": [], "occurrences": []}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json_file(tmpdir, "min.json", data)
            result = load_scip_artifact(path)
            assert len(result.documents) == 1

    @pytest.mark.edge
    def test_role_bitmask_combined(self):
        """EDGE: combined definition + import bit returns definition (priority)."""
        assert _symbol_role_to_str(1 | 2) == "definition"

    @pytest.mark.edge
    def test_role_high_bits_only(self):
        """EDGE: bits above defined roles return reference."""
        assert _symbol_role_to_str(128) == "reference"

    @pytest.mark.edge
    def test_kind_zero(self):
        """EDGE: kind value 0 returns mapped value or symbol."""
        result = _scip_kind_to_str(0)
        assert isinstance(result, str)

    @pytest.mark.edge
    def test_load_json_with_no_language(self):
        """EDGE: document without language field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "documents": [{"path": "unknown.xyz", "symbols": [], "occurrences": []}]
            }
            path = _write_json_file(tmpdir, "nolang.json", data)
            result = load_scip_artifact(path)
            assert result.documents[0].language is None

    @pytest.mark.edge
    def test_load_json_with_extra_fields(self):
        """EDGE: extra fields in JSON are ignored gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"documents": [], "extra_field": "ignored", "version": 999}
            path = _write_json_file(tmpdir, "extra.json", data)
            result = load_scip_artifact(path)
            assert len(result.documents) == 0

    @pytest.mark.edge
    def test_load_empty_json_object(self):
        """EDGE: empty JSON object produces empty index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json_file(tmpdir, "empty.json", {})
            result = load_scip_artifact(path)
            assert isinstance(result, SCIPIndex)
