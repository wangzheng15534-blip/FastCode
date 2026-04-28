"""Property-based tests for module_resolver module."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.module_resolver import ModuleResolver

# --- Helpers ---

segment_st = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz",
    min_size=1,
    max_size=5,
)

module_path_st = st.lists(segment_st, min_size=1, max_size=4).map(".".join)


class _FakeIndex:
    """Minimal fake GlobalIndexBuilder for testing."""

    def __init__(self, module_map: dict[str, list[str]] | None = None) -> None:
        self.module_map = module_map or {}


def _make_resolver(module_map: dict[str, list[str]] | None = None) -> Any:
    return ModuleResolver(_FakeIndex(module_map))


# --- Properties ---


class TestResolveAbsoluteImport:
    @pytest.mark.basic
    def test_known_module_found_property(self):
        """HAPPY: absolute import of known module returns file ID."""
        resolver = _make_resolver({"utils": "file:utils.py"})
        result = resolver.resolve_import("app.main", "utils", level=0)
        assert result == "file:utils.py"

    @pytest.mark.basic
    def test_nested_module_found_property(self):
        """HAPPY: nested absolute import resolves."""
        resolver = _make_resolver({"app.services": "file:services.py"})
        result = resolver.resolve_import("app.main", "app.services", level=0)
        assert result == "file:services.py"

    @pytest.mark.edge
    def test_unknown_module_returns_none_property(self):
        """EDGE: unknown third-party module returns None."""
        resolver = _make_resolver({"utils": "file:utils.py"})
        result = resolver.resolve_import("app.main", "numpy", level=0)
        assert result is None

    @pytest.mark.edge
    def test_empty_import_name_returns_none_property(self):
        """EDGE: empty import name returns None."""
        resolver = _make_resolver({})
        result = resolver.resolve_import("app.main", "", level=0)
        assert result is None

    @pytest.mark.edge
    def test_empty_module_map_returns_none_property(self):
        """EDGE: empty module map always returns None."""
        resolver = _make_resolver({})
        result = resolver.resolve_import("app.main", "anything", level=0)
        assert result is None


class TestResolveRelativeImport:
    @pytest.mark.basic
    def test_level_one_import_property(self):
        """HAPPY: single-dot import resolves to sibling module."""
        resolver = _make_resolver({"app.utils": "file:utils.py"})
        result = resolver.resolve_import("app.main", "utils", level=1)
        assert result == "file:utils.py"

    @pytest.mark.basic
    def test_level_two_import_property(self):
        """HAPPY: double-dot import resolves to parent package."""
        resolver = _make_resolver({"app.utils": "file:utils.py"})
        result = resolver.resolve_import("app.services.auth", "utils", level=2)
        assert result == "file:utils.py"

    @pytest.mark.basic
    def test_package_init_level_one_property(self):
        """HAPPY: __init__.py level=1 stays in current package."""
        resolver = _make_resolver({"app.utils": "file:utils.py"})
        result = resolver.resolve_import("app", "utils", level=1, is_package=True)
        assert result == "file:utils.py"

    @pytest.mark.edge
    def test_level_exceeds_depth_property(self):
        """EDGE: relative import beyond package root returns None."""
        resolver = _make_resolver({"utils": "file:utils.py"})
        result = resolver.resolve_import("app", "utils", level=3)
        assert result is None

    @pytest.mark.edge
    def test_level_exactly_depth_property(self):
        """EDGE: relative import exactly at root."""
        resolver = _make_resolver({"utils": "file:utils.py"})
        result = resolver.resolve_import("app", "utils", level=1)
        assert result == "file:utils.py"

    @pytest.mark.edge
    def test_no_import_name_returns_package_property(self):
        """EDGE: level import with no name resolves to parent package."""
        resolver = _make_resolver({"app": "file:__init__.py"})
        result = resolver.resolve_import("app.services", "", level=1)
        assert result == "file:__init__.py"

    @pytest.mark.edge
    def test_level_zero_with_empty_map_property(self):
        """EDGE: absolute import with no entries returns None."""
        resolver = _make_resolver()
        result = resolver.resolve_import("a.b", "c", level=0)
        assert result is None

    @given(
        current=module_path_st,
        target=segment_st,
    )
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_relative_always_checks_map_property(self, current: bool, target: Any):
        """HAPPY: relative import always looks up in module map."""
        resolver = _make_resolver()
        result = resolver.resolve_import(current, target, level=1)
        assert result is None  # empty map always returns None

    @pytest.mark.edge
    def test_single_segment_level_one_property(self):
        """EDGE: single-segment path with level=1."""
        resolver = _make_resolver({"": "file:root.py"})
        result = resolver.resolve_import("app", "utils", level=1)
        # level=1 strips last segment from "app" -> "", then appends "utils" -> "utils"
        assert result is None  # "utils" not in map


class TestModuleResolverEdge:
    @pytest.mark.edge
    def test_is_package_false_by_default_property(self):
        """EDGE: is_package defaults to False."""
        resolver = _make_resolver({"app.utils": "f.py"})
        # Regular file: level=1 strips "auth" from "app.auth" -> "app", appends "utils"
        result = resolver.resolve_import("app.auth", "utils", level=1, is_package=False)
        assert result == "f.py"

    @pytest.mark.edge
    def test_is_package_true_strips_less_property(self):
        """EDGE: is_package=True strips one less level."""
        resolver = _make_resolver({"app.utils": "f.py"})
        # Package: level=1 from "app" strips 0 -> "app", appends "utils" -> "app.utils"
        result = resolver.resolve_import("app", "utils", level=1, is_package=True)
        assert result == "f.py"

    @pytest.mark.edge
    def test_resolve_absolute_with_dotted_name_property(self):
        """EDGE: dotted absolute import resolves."""
        resolver = _make_resolver({"a.b.c": "file:c.py"})
        result = resolver.resolve_import("x.y", "a.b.c", level=0)
        assert result == "file:c.py"

    @pytest.mark.edge
    def test_level_two_package_import_property(self):
        """EDGE: level=2 from package."""
        resolver = _make_resolver({"top.utils": "f.py"})
        result = resolver.resolve_import("top.sub", "utils", level=2, is_package=True)
        assert result == "f.py"
