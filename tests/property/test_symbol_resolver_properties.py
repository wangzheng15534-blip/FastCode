"""Property-based tests for symbol_resolver module."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.global_index_builder import GlobalIndexBuilder
from fastcode.symbol_resolver import SymbolResolver

# --- Helpers ---


class _FakeModuleResolver:
    """Minimal fake module resolver."""

    def __init__(self, mappings: Any = None) -> None:
        self._mappings = mappings or {}

    def resolve_import(
        self,
        current_module_path: str,
        import_name: str,
        level: int = 0,
        is_package: bool = False,
    ) -> Any:
        key = (current_module_path, import_name, level)
        return self._mappings.get(key)


def _make_index(module_map: dict | None = None, export_map: dict | None = None) -> Any:
    """Build a GlobalIndexBuilder with pre-populated maps."""
    idx = GlobalIndexBuilder()
    idx.module_map = module_map or {}
    idx.export_map = export_map or {}
    idx.file_map = {f"file:{k}": f"file:{k}" for k in (module_map or {})}
    return idx


def _make_resolver(
    module_map: dict | None = None,
    export_map: dict | None = None,
    import_mappings: dict | None = None,
) -> Any:
    idx = _make_index(module_map, export_map)
    mr = _FakeModuleResolver(import_mappings)
    return SymbolResolver(idx, mr)


name_st = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)


# --- Properties ---


@pytest.mark.property
class TestResolveSymbol:
    @pytest.mark.happy
    def test_local_resolution(self):
        """HAPPY: symbol defined in current file resolved locally."""
        resolver = _make_resolver(
            module_map={"app.auth": "file:auth"},
            export_map={"app.auth": {"login": "sym:login"}},
        )
        result = resolver.resolve_symbol("login", "file:auth", [])
        assert result == "sym:login"

    @pytest.mark.happy
    def test_imported_resolution(self):
        """HAPPY: symbol resolved through import."""
        resolver = _make_resolver(
            module_map={"app.utils": "file:utils", "app.main": "file:main"},
            export_map={"app.utils": {"helper": "sym:helper"}},
            import_mappings={("app.main", "app.utils", 0): "file:utils"},
        )
        imports = [
            {"module": "app.utils", "names": ["helper"], "alias": None, "level": 0}
        ]
        result = resolver.resolve_symbol("helper", "file:main", imports)
        assert result == "sym:helper"

    @pytest.mark.edge
    def test_empty_symbol_returns_none(self):
        """EDGE: empty symbol name returns None."""
        resolver = _make_resolver()
        assert resolver.resolve_symbol("", "file:1", []) is None

    @pytest.mark.edge
    def test_none_file_id_returns_none(self):
        """EDGE: None file_id returns None."""
        resolver = _make_resolver()
        assert resolver.resolve_symbol("foo", None, []) is None

    @pytest.mark.edge
    def test_empty_file_id_returns_none(self):
        """EDGE: empty file_id returns None."""
        resolver = _make_resolver()
        assert resolver.resolve_symbol("foo", "", []) is None

    @pytest.mark.edge
    def test_unknown_symbol_returns_none(self):
        """EDGE: unknown symbol returns None."""
        resolver = _make_resolver(
            module_map={"app.auth": "file:auth"},
            export_map={"app.auth": {"login": "sym:login"}},
        )
        assert resolver.resolve_symbol("nonexistent", "file:auth", []) is None

    @pytest.mark.edge
    def test_no_imports_returns_none_for_external(self):
        """EDGE: no imports, symbol not local returns None."""
        resolver = _make_resolver(
            module_map={"app.auth": "file:auth"},
            export_map={"app.auth": {}},
        )
        assert resolver.resolve_symbol("external_func", "file:auth", []) is None


@pytest.mark.property
class TestMatchesImport:
    @pytest.mark.happy
    def test_direct_name_match(self):
        """HAPPY: symbol directly in import names."""
        resolver = _make_resolver()
        assert (
            resolver._matches_import(
                "helper",
                {
                    "names": ["helper"],
                    "module": "utils",
                    "alias": None,
                },
            )
            is True
        )

    @pytest.mark.happy
    def test_alias_match(self):
        """HAPPY: symbol matches import alias."""
        resolver = _make_resolver()
        assert (
            resolver._matches_import(
                "h",
                {
                    "names": ["helper"],
                    "module": "utils",
                    "alias": "h",
                },
            )
            is True
        )

    @pytest.mark.happy
    def test_module_prefix_match(self):
        """HAPPY: symbol starts with module prefix."""
        resolver = _make_resolver()
        assert (
            resolver._matches_import(
                "utils.helper",
                {
                    "names": [],
                    "module": "utils",
                    "alias": None,
                },
            )
            is True
        )

    @pytest.mark.edge
    def test_no_match(self):
        """EDGE: symbol doesn't match any import pattern."""
        resolver = _make_resolver()
        assert (
            resolver._matches_import(
                "other",
                {
                    "names": ["helper"],
                    "module": "utils",
                    "alias": None,
                },
            )
            is False
        )

    @pytest.mark.edge
    def test_empty_import_info(self):
        """EDGE: empty import info returns False."""
        resolver = _make_resolver()
        assert resolver._matches_import("foo", {}) is False

    @pytest.mark.edge
    def test_member_prefix_match(self):
        """EDGE: Class.method matches imported Class name."""
        resolver = _make_resolver()
        assert (
            resolver._matches_import(
                "MyClass.method",
                {
                    "names": ["MyClass"],
                    "module": "mod",
                    "alias": None,
                },
            )
            is True
        )


@pytest.mark.property
class TestGetModulePathByFileId:
    @pytest.mark.happy
    def test_known_file_id(self):
        """HAPPY: known file_id returns module path."""
        resolver = _make_resolver(module_map={"app.auth": "file:auth"})
        assert resolver._get_module_path_by_file_id("file:auth") == "app.auth"

    @pytest.mark.edge
    def test_unknown_file_id(self):
        """EDGE: unknown file_id returns None."""
        resolver = _make_resolver(module_map={"app.auth": "file:auth"})
        assert resolver._get_module_path_by_file_id("file:unknown") is None

    @pytest.mark.edge
    def test_empty_file_id(self):
        """EDGE: empty file_id returns None."""
        resolver = _make_resolver()
        assert resolver._get_module_path_by_file_id("") is None


@pytest.mark.property
class TestGetCurrentModulePathForImports:
    @pytest.mark.happy
    def test_known_file(self):
        """HAPPY: returns module path for known file_id."""
        resolver = _make_resolver(module_map={"app.main": "file:main"})
        assert resolver._get_current_module_path_for_imports("file:main") == "app.main"

    @pytest.mark.edge
    def test_unknown_file_returns_empty(self):
        """EDGE: unknown file_id returns empty string."""
        resolver = _make_resolver()
        assert resolver._get_current_module_path_for_imports("file:unknown") == ""

    @pytest.mark.edge
    def test_none_returns_empty(self):
        """EDGE: None returns empty string."""
        resolver = _make_resolver()
        assert resolver._get_current_module_path_for_imports(None) == ""


@pytest.mark.property
class TestGetResolutionStats:
    @pytest.mark.happy
    def test_stats_keys(self):
        """HAPPY: stats contains expected keys."""
        resolver = _make_resolver(
            module_map={"a": "1", "b": "2"},
            export_map={"a": {"x": "sx"}},
        )
        stats = resolver.get_resolution_stats()
        assert "modules_available" in stats
        assert "exports_available" in stats
        assert "files_mapped" in stats

    @pytest.mark.edge
    def test_empty_stats(self):
        """EDGE: empty resolver stats are zeroed."""
        resolver = _make_resolver()
        stats = resolver.get_resolution_stats()
        assert stats["modules_available"] == 0
        assert stats["exports_available"] == 0

    @given(symbol=name_st)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_resolve_always_returns_none_or_str(self, symbol: str):
        """EDGE: resolve_symbol always returns None or str."""
        resolver = _make_resolver()
        result = resolver.resolve_symbol(symbol, "file:unknown", [])
        assert result is None or isinstance(result, str)
