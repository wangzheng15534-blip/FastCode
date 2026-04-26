"""Tests for fastcode.utils.paths."""

from __future__ import annotations

from fastcode.utils.paths import get_language_from_extension, projection_scope_key


class TestGetLanguageFromExtension:
    def test_python(self) -> None:
        assert get_language_from_extension(".py") == "python"

    def test_javascript(self) -> None:
        assert get_language_from_extension(".js") == "javascript"

    def test_unknown_returns_unknown(self) -> None:
        result = get_language_from_extension(".xyz")
        assert result == "unknown"


class TestProjectionScopeKey:
    def test_deterministic(self) -> None:
        k1 = projection_scope_key("global", "snap:repo:abc", None, None, None)
        k2 = projection_scope_key("global", "snap:repo:abc", None, None, None)
        assert k1 == k2

    def test_different_kinds_differ(self) -> None:
        k1 = projection_scope_key("global", "snap:repo:abc", None, None, None)
        k2 = projection_scope_key("targeted", "snap:repo:abc", None, None, None)
        assert k1 != k2

    def test_with_query(self) -> None:
        k1 = projection_scope_key("global", "snap:repo:abc", None, None, None)
        k2 = projection_scope_key(
            "global", "snap:repo:abc", "how does X work?", None, None
        )
        assert k1 != k2
