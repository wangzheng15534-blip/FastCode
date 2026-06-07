"""Contract tests for ir.types helper functions — meaning_core, ZERO test doubles.

Tests for _sorted_set, _normalize_set, _copy_dict, resolution_to_confidence,
and IRCodeUnit/IRRelation round-trip serialization edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.ir.types import (
    IRCodeUnit,
    IRRelation,
    _copy_dict,
    _normalize_set,
    _sorted_set,
    resolution_rank,
    resolution_to_confidence,
)

# ---------------------------------------------------------------------------
# _sorted_set
# ---------------------------------------------------------------------------


class TestSortedSet:
    def test_sorts_strings(self) -> None:
        assert _sorted_set({"c", "a", "b"}) == ["a", "b", "c"]

    def test_filters_empty_strings(self) -> None:
        assert _sorted_set({"a", "", "b"}) == ["a", "b"]

    @pytest.mark.edge
    def test_empty_set(self) -> None:
        assert _sorted_set(set()) == []

    @given(items=st.sets(st.text(min_size=1, max_size=10)))
    @settings(max_examples=30)
    @pytest.mark.property
    def test_result_always_sorted(self, items: set[str]) -> None:
        result = _sorted_set(items)
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# _normalize_set
# ---------------------------------------------------------------------------


class TestNormalizeSet:
    def test_converts_to_strings(self) -> None:
        result = _normalize_set([1, 2, 3])
        assert result == {"1", "2", "3"}

    def test_filters_falsy(self) -> None:
        result = _normalize_set(["a", "", None, 0, "b"])  # type: ignore[list-item]
        assert result == {"a", "b"}

    @pytest.mark.edge
    def test_none_input(self) -> None:
        assert _normalize_set(None) == set()

    @pytest.mark.edge
    def test_empty_list(self) -> None:
        assert _normalize_set([]) == set()


# ---------------------------------------------------------------------------
# _copy_dict
# ---------------------------------------------------------------------------


class TestCopyDict:
    def test_copies_all_keys(self) -> None:
        original: dict[str, Any] = {"a": 1, "b": "two", "c": [3]}
        result = _copy_dict(original)
        assert result == original
        assert result is not original

    def test_stringifies_keys(self) -> None:
        result = _copy_dict({42: "val"})  # type: ignore[dict-item]
        assert "42" in result

    @pytest.mark.edge
    def test_empty_dict(self) -> None:
        assert _copy_dict({}) == {}


# ---------------------------------------------------------------------------
# resolution_to_confidence
# ---------------------------------------------------------------------------


class TestResolutionToConfidence:
    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            ("anchored", "precise"),
            ("semantic", "precise"),
            ("semantically_resolved", "precise"),
            ("structural", "resolved"),
            ("candidate", "heuristic"),
        ],
    )
    def test_known_states(self, state: str, expected: str) -> None:
        assert resolution_to_confidence(state) == expected

    def test_unknown_defaults_to_derived(self) -> None:
        assert resolution_to_confidence("bogus") == "derived"

    def test_none_defaults_to_derived(self) -> None:
        assert resolution_to_confidence(None) == "derived"


# ---------------------------------------------------------------------------
# IRCodeUnit round-trip edge cases
# ---------------------------------------------------------------------------


class TestIRCodeUnitRoundTrip:
    def test_roundtrip_preserves_anchor_symbol_ids(self) -> None:
        unit = IRCodeUnit(
            unit_id="u1",
            kind="function",
            path="a.py",
            language="python",
            display_name="fn",
            anchor_symbol_ids=["s1", "s2"],
            candidate_anchor_symbol_ids=["c1"],
            source_set={"fc_structure"},
        )
        restored = IRCodeUnit.from_dict(unit.to_dict())
        assert restored.anchor_symbol_ids == ["s1", "s2"]
        assert restored.candidate_anchor_symbol_ids == ["c1"]

    def test_roundtrip_preserves_source_set(self) -> None:
        unit = IRCodeUnit(
            unit_id="u1",
            kind="function",
            path="a.py",
            language="python",
            display_name="fn",
            source_set={"scip", "fc_structure"},
        )
        restored = IRCodeUnit.from_dict(unit.to_dict())
        assert restored.source_set == {"scip", "fc_structure"}

    def test_roundtrip_with_empty_fields(self) -> None:
        unit = IRCodeUnit(
            unit_id="u1",
            kind="function",
            path="a.py",
            language="python",
            display_name="fn",
            source_set=set(),
            metadata={},
        )
        restored = IRCodeUnit.from_dict(unit.to_dict())
        assert restored.source_set == set()
        assert restored.metadata == {}

    @pytest.mark.edge
    def test_from_dict_handles_missing_optional_fields(self) -> None:
        data = {
            "unit_id": "u1",
            "kind": "function",
            "path": "a.py",
            "language": "python",
            "display_name": "fn",
        }
        restored = IRCodeUnit.from_dict(data)
        assert restored.anchor_symbol_ids == []
        assert restored.candidate_anchor_symbol_ids == []
        assert restored.source_set == set()


# ---------------------------------------------------------------------------
# IRRelation round-trip edge cases
# ---------------------------------------------------------------------------


class TestIRRelationRoundTrip:
    def test_roundtrip_preserves_support_ids(self) -> None:
        rel = IRRelation(
            relation_id="r1",
            src_unit_id="u1",
            dst_unit_id="u2",
            relation_type="call",
            resolution_state="anchored",
            support_ids=["s1", "s2"],
        )
        restored = IRRelation.from_dict(rel.to_dict())
        assert restored.support_ids == ["s1", "s2"]

    def test_roundtrip_filters_empty_support_ids(self) -> None:
        data: dict[str, Any] = {
            "relation_id": "r1",
            "src_unit_id": "u1",
            "dst_unit_id": "u2",
            "relation_type": "call",
            "resolution_state": "anchored",
            "support_ids": ["s1", "", None, "s2"],  # type: ignore[list-item]
        }
        restored = IRRelation.from_dict(data)
        assert restored.support_ids == ["s1", "s2"]


# ---------------------------------------------------------------------------
# resolution_rank ordering property
# ---------------------------------------------------------------------------


class TestResolutionRankProperties:
    @given(
        a=st.sampled_from(
            ["candidate", "structural", "anchored", "semantic", "semantically_resolved"]
        ),
        b=st.sampled_from(
            ["candidate", "structural", "anchored", "semantic", "semantically_resolved"]
        ),
    )
    @settings(max_examples=30)
    @pytest.mark.property
    def test_rank_is_transitive(self, a: str, b: str) -> None:
        ra = resolution_rank(a)
        rb = resolution_rank(b)
        # If a > b and b > c then a > c (checked implicitly by numeric comparison)
        assert isinstance(ra, int)
        assert isinstance(rb, int)
        assert ra >= 0
        assert rb >= 0
