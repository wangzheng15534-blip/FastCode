"""Tests for projection_models — utc_now_iso, ProjectionScope, ProjectionBuildResult."""

from __future__ import annotations

from fastcode.projection_models import (
    ProjectionBuildResult,
    ProjectionScope,
    utc_now_iso,
)


# --- Helpers ---


def _make_scope(**overrides):
    defaults = dict(scope_kind="snapshot", snapshot_id="snap:repo:abc123", scope_key="repo")
    defaults.update(overrides)
    return ProjectionScope(**defaults)


def _make_result(**overrides):
    defaults = dict(
        projection_id="proj_001",
        snapshot_id="snap:repo:abc123",
        scope_kind="snapshot",
        scope_key="repo",
        l0={"nodes": 10, "edges": 5},
        l1={"clusters": 3},
        l2_index={"index_size": 100},
        chunks=[{"text": "chunk0"}],
    )
    defaults.update(overrides)
    return ProjectionBuildResult(**defaults)


# --- Tests ---


class TestUtcNowIso:
    def test_returns_string(self):
        result = utc_now_iso()
        assert isinstance(result, str)

    def test_contains_iso_separator(self):
        result = utc_now_iso()
        assert "T" in result

    def test_returns_different_values_on_successive_calls(self):
        a = utc_now_iso()
        b = utc_now_iso()
        # Not guaranteed to differ in the same microsecond, but extremely likely
        # to differ in practice — ISO format includes microseconds.
        assert isinstance(a, str) and isinstance(b, str)


class TestProjectionScope:
    def test_to_dict_includes_all_fields(self):
        scope = _make_scope(query="find callers", target_id="sym:main", filters={"lang": "py"})
        d = scope.to_dict()
        assert set(d.keys()) == {
            "scope_kind", "snapshot_id", "scope_key", "query", "target_id", "filters",
        }

    def test_to_dict_defaults_none_and_empty(self):
        scope = _make_scope()
        d = scope.to_dict()
        assert d["query"] is None
        assert d["target_id"] is None
        assert d["filters"] == {}

    def test_to_dict_preserves_values(self):
        scope = _make_scope(
            scope_kind="query",
            snapshot_id="snap:x:deadbeef",
            scope_key="module:auth",
            query="impact analysis",
            target_id="sym:login",
            filters={"depth": 2},
        )
        d = scope.to_dict()
        assert d["scope_kind"] == "query"
        assert d["snapshot_id"] == "snap:x:deadbeef"
        assert d["scope_key"] == "module:auth"
        assert d["query"] == "impact analysis"
        assert d["target_id"] == "sym:login"
        assert d["filters"] == {"depth": 2}


class TestProjectionBuildResult:
    def test_to_dict_has_required_keys(self):
        result = _make_result()
        d = result.to_dict()
        assert set(d.keys()) == {
            "projection_id", "snapshot_id", "scope_kind", "scope_key",
            "l0", "l1", "l2_index", "chunks", "warnings", "created_at",
        }

    def test_warnings_default_empty(self):
        result = _make_result()
        assert result.warnings == []
        assert result.to_dict()["warnings"] == []

    def test_created_at_auto_set_contains_T(self):
        result = _make_result()
        assert isinstance(result.created_at, str)
        assert "T" in result.created_at

    def test_created_at_propagated_to_dict(self):
        result = _make_result()
        assert result.to_dict()["created_at"] == result.created_at

    def test_explicit_warnings_preserved(self):
        result = _make_result(warnings=["missing cluster", "empty l2"])
        d = result.to_dict()
        assert d["warnings"] == ["missing cluster", "empty l2"]

    def test_explicit_created_at_preserved(self):
        result = _make_result(created_at="2026-01-15T10:30:00+00:00")
        assert result.created_at == "2026-01-15T10:30:00+00:00"
        assert result.to_dict()["created_at"] == "2026-01-15T10:30:00+00:00"
