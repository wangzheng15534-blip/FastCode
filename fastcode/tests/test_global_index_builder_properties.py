"""Property-based tests for global_index_builder module."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.global_index_builder import GlobalIndexBuilder

# --- Helpers ---

segment_st = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)

module_path_st = st.lists(segment_st, min_size=1, max_size=4).map(".".join)


def _make_builder() -> Any:
    """Create a builder pre-populated with some test data."""
    builder = GlobalIndexBuilder()
    builder.file_map = {
        "/repo/app/auth.py": "file:auth",
        "/repo/app/models.py": "file:models",
        "/repo/app/__init__.py": "file:init",
    }
    builder.module_map = {
        "app.auth": "file:auth",
        "app.models": "file:models",
        "app": "file:init",
    }
    builder.export_map = {
        "app.auth": {"login": "sym:login", "logout": "sym:logout"},
        "app.models": {"User": "sym:User"},
    }
    builder.stats = {
        "files_processed": 3,
        "modules_created": 3,
        "symbols_exported": 3,
        "errors": 0,
    }
    return builder


# --- Properties ---


class TestGetFileIdByPath:
    @pytest.mark.basic
    def test_existing_path_property(self):
        """HAPPY: known path returns file ID."""
        builder = _make_builder()
        result = builder.get_file_id_by_path("/repo/app/auth.py")
        assert result == "file:auth"

    @pytest.mark.edge
    def test_unknown_path_returns_none_property(self):
        """EDGE: unknown path returns None."""
        builder = _make_builder()
        assert builder.get_file_id_by_path("/repo/unknown.py") is None

    @pytest.mark.edge
    def test_empty_path_returns_none_property(self):
        """EDGE: empty path returns None."""
        builder = _make_builder()
        assert builder.get_file_id_by_path("") is None


class TestGetFileIdByModule:
    @pytest.mark.basic
    def test_existing_module_property(self):
        """HAPPY: known module returns file ID."""
        builder = _make_builder()
        assert builder.get_file_id_by_module("app.auth") == "file:auth"

    @pytest.mark.edge
    def test_unknown_module_returns_none_property(self):
        """EDGE: unknown module returns None."""
        builder = _make_builder()
        assert builder.get_file_id_by_module("unknown.module") is None

    @given(mod=module_path_st)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_arbitrary_module_returns_none_or_id_property(self, mod: Any):
        """EDGE: arbitrary module returns None or valid ID."""
        builder = _make_builder()
        result = builder.get_file_id_by_module(mod)
        assert result is None or isinstance(result, str)


class TestContainsFile:
    @pytest.mark.basic
    def test_known_file_property(self):
        """HAPPY: known file returns True."""
        builder = _make_builder()
        assert builder.contains_file("/repo/app/auth.py") is True

    @pytest.mark.edge
    def test_unknown_file_property(self):
        """EDGE: unknown file returns False."""
        builder = _make_builder()
        assert builder.contains_file("/repo/nope.py") is False


class TestContainsModule:
    @pytest.mark.basic
    def test_known_module_property(self):
        """HAPPY: known module returns True."""
        builder = _make_builder()
        assert builder.contains_module("app.auth") is True

    @pytest.mark.edge
    def test_unknown_module_property(self):
        """EDGE: unknown module returns False."""
        builder = _make_builder()
        assert builder.contains_module("nope") is False


class TestGetAllFileIds:
    @pytest.mark.basic
    def test_returns_all_ids_property(self):
        """HAPPY: returns all file IDs."""
        builder = _make_builder()
        ids = builder.get_all_file_ids()
        assert len(ids) == 3
        assert "file:auth" in ids

    @pytest.mark.edge
    def test_empty_builder_returns_empty_property(self):
        """EDGE: empty builder returns empty list."""
        builder = GlobalIndexBuilder()
        assert builder.get_all_file_ids() == []


class TestGetAllModules:
    @pytest.mark.basic
    def test_returns_all_modules_property(self):
        """HAPPY: returns all module paths."""
        builder = _make_builder()
        mods = builder.get_all_modules()
        assert "app.auth" in mods
        assert "app.models" in mods

    @pytest.mark.edge
    def test_empty_builder_returns_empty_property(self):
        """EDGE: empty builder returns empty list."""
        builder = GlobalIndexBuilder()
        assert builder.get_all_modules() == []


class TestGetStats:
    @pytest.mark.basic
    def test_stats_keys_property(self):
        """HAPPY: stats contains expected keys."""
        builder = _make_builder()
        stats = builder.get_stats()
        assert "files_processed" in stats
        assert "modules_created" in stats
        assert "symbols_exported" in stats
        assert "file_map_size" in stats
        assert "module_map_size" in stats
        assert "module_coverage" in stats

    @pytest.mark.edge
    def test_stats_empty_builder_property(self):
        """EDGE: empty builder stats are zeroed."""
        builder = GlobalIndexBuilder()
        stats = builder.get_stats()
        assert stats["files_processed"] == 0
        assert stats["module_coverage"] == 0.0


class TestValidateMaps:
    @pytest.mark.basic
    def test_valid_maps_no_errors_property(self):
        """HAPPY: consistent maps produce no errors."""
        builder = _make_builder()
        errors = builder.validate_maps()
        assert errors == []

    @pytest.mark.edge
    def test_empty_maps_no_errors_property(self):
        """EDGE: empty maps produce no errors."""
        builder = GlobalIndexBuilder()
        assert builder.validate_maps() == []


class TestGetExportedSymbolId:
    @pytest.mark.basic
    def test_known_symbol_property(self):
        """HAPPY: known symbol returns node ID."""
        builder = _make_builder()
        assert builder.get_exported_symbol_id("app.auth", "login") == "sym:login"

    @pytest.mark.edge
    def test_unknown_module_returns_none_property(self):
        """EDGE: unknown module returns None."""
        builder = _make_builder()
        assert builder.get_exported_symbol_id("no.module", "foo") is None

    @pytest.mark.edge
    def test_unknown_symbol_returns_none_property(self):
        """EDGE: unknown symbol in known module returns None."""
        builder = _make_builder()
        assert builder.get_exported_symbol_id("app.auth", "nonexistent") is None


class TestGetModuleExports:
    @pytest.mark.basic
    def test_known_module_property(self):
        """HAPPY: returns copy of export dict."""
        builder = _make_builder()
        exports = builder.get_module_exports("app.auth")
        assert "login" in exports
        assert "logout" in exports

    @pytest.mark.edge
    def test_unknown_module_returns_empty_property(self):
        """EDGE: unknown module returns empty dict."""
        builder = _make_builder()
        assert builder.get_module_exports("no.module") == {}

    @pytest.mark.edge
    def test_returns_copy_property(self):
        """EDGE: returned dict is a copy, not the original."""
        builder = _make_builder()
        exports = builder.get_module_exports("app.auth")
        exports["new"] = "value"
        assert "new" not in builder.export_map["app.auth"]


class TestClear:
    @pytest.mark.basic
    def test_clear_resets_maps_property(self):
        """HAPPY: clear empties all maps."""
        builder = _make_builder()
        builder.clear()
        assert builder.file_map == {}
        assert builder.module_map == {}
        assert builder.export_map == {}

    @pytest.mark.edge
    def test_clear_resets_stats_property(self):
        """EDGE: clear resets stats to zero."""
        builder = _make_builder()
        builder.clear()
        assert builder.stats["files_processed"] == 0
        assert builder.stats["errors"] == 0

    @pytest.mark.edge
    def test_clear_idempotent_property(self):
        """EDGE: clearing twice is safe."""
        builder = _make_builder()
        builder.clear()
        builder.clear()
        assert builder.file_map == {}
