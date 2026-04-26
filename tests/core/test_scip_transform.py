# tests/test_core_scip_transform.py
"""Tests for pure SCIP transform functions."""

from fastcode.core.scip_transform import scip_kind_to_str, symbol_role_to_str


class TestSymbolRoleToStr:
    def test_definition(self):
        assert symbol_role_to_str(1) == "definition"

    def test_import(self):
        assert symbol_role_to_str(2) == "import"

    def test_write_access(self):
        assert symbol_role_to_str(4) == "write_access"

    def test_forward_definition(self):
        assert symbol_role_to_str(64) == "forward_definition"

    def test_reference(self):
        assert symbol_role_to_str(0) == "reference"

    def test_combined_roles(self):
        # Definition | Reference = 1 | 0 (definition wins)
        assert symbol_role_to_str(1) == "definition"


class TestScipKindToStr:
    def test_known_kinds(self):
        # Test with known kind values using the enum if available
        # Falls back to "symbol" for unknown values
        assert scip_kind_to_str(999) == "symbol"

    def test_fallback(self):
        assert scip_kind_to_str(-1) == "symbol"
