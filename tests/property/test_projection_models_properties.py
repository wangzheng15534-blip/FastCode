"""Property-based tests for projection_models module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.projection_models import (
    ProjectionBuildResult,
    ProjectionScope,
    utc_now_iso,
)

# --- Strategies ---

small_text = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)


# --- Properties ---


@pytest.mark.property
class TestUtcNowIso:

    @pytest.mark.happy
    def test_returns_iso_string(self):
        """HAPPY: utc_now_iso returns valid ISO format string."""
        result = utc_now_iso()
        assert isinstance(result, str)
        assert "T" in result
        assert "+" in result or "-" in result.split("T")[1] or "Z" in result


@pytest.mark.property
class TestProjectionScope:

    @given(kind=small_text, snap_id=small_text, key=small_text)
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_to_dict_roundtrip(self, kind, snap_id, key):
        """HAPPY: ProjectionScope to_dict preserves all fields."""
        scope = ProjectionScope(scope_kind=kind, snapshot_id=snap_id, scope_key=key)
        d = scope.to_dict()
        assert d["scope_kind"] == kind
        assert d["snapshot_id"] == snap_id
        assert d["scope_key"] == key
        assert d["query"] is None
        assert d["target_id"] is None
        assert d["filters"] == {}

    @given(kind=small_text, snap_id=small_text, key=small_text, query=st.none() | small_text)
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_scope_with_query(self, kind, snap_id, key, query):
        """HAPPY: ProjectionScope with optional query."""
        scope = ProjectionScope(scope_kind=kind, snapshot_id=snap_id, scope_key=key, query=query)
        d = scope.to_dict()
        assert d["query"] == query

    @given(kind=small_text, snap_id=small_text, key=small_text)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_scope_default_filters_empty(self, kind, snap_id, key):
        """EDGE: default filters is empty dict."""
        scope = ProjectionScope(scope_kind=kind, snapshot_id=snap_id, scope_key=key)
        assert scope.filters == {}
        assert isinstance(scope.filters, dict)


@pytest.mark.property
class TestProjectionBuildResult:

    @pytest.mark.happy
    def test_to_dict_has_all_keys(self):
        """HAPPY: ProjectionBuildResult.to_dict has all required keys."""
        result = ProjectionBuildResult(
            projection_id="proj_1", snapshot_id="snap_1",
            scope_kind="snapshot", scope_key="repo",
            l0={}, l1={}, l2_index={}, chunks=[],
        )
        d = result.to_dict()
        for key in ("projection_id", "snapshot_id", "scope_kind", "scope_key",
                     "l0", "l1", "l2_index", "chunks", "warnings", "created_at"):
            assert key in d

    @pytest.mark.happy
    def test_default_warnings_empty(self):
        """HAPPY: default warnings is empty list."""
        result = ProjectionBuildResult(
            projection_id="proj_1", snapshot_id="snap_1",
            scope_kind="snapshot", scope_key="repo",
            l0={}, l1={}, l2_index={}, chunks=[],
        )
        assert result.warnings == []

    @pytest.mark.happy
    def test_created_at_auto_set(self):
        """HAPPY: created_at is auto-set to current time."""
        result = ProjectionBuildResult(
            projection_id="proj_1", snapshot_id="snap_1",
            scope_kind="snapshot", scope_key="repo",
            l0={}, l1={}, l2_index={}, chunks=[],
        )
        assert result.created_at is not None
        assert "T" in result.created_at
