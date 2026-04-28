"""Tests for pure combination function extracted from retriever."""

from typing import Any

import pytest

from fastcode.core.combination import combine_results


def _mk_meta(elem_id: str, **extra) -> dict[str, Any]:
    meta = {"id": elem_id, "type": "function", "name": elem_id}
    meta.update(extra)
    return meta


class TestCombineResults:
    def test_merges_semantic_and_keyword(self):
        sem = [(_mk_meta("a"), 0.8)]
        kw = [(_mk_meta("b"), 5.0)]
        result = combine_results(sem, kw)
        ids = [r["element"]["id"] for r in result]
        assert "a" in ids
        assert "b" in ids

    def test_merges_same_element(self):
        sem = [(_mk_meta("a"), 0.8)]
        kw = [(_mk_meta("a"), 5.0)]
        result = combine_results(sem, kw)
        assert len(result) == 1
        assert result[0]["semantic_score"] > 0
        assert result[0]["keyword_score"] > 0

    def test_pseudocode_results(self):
        sem = [(_mk_meta("a"), 0.8)]
        pseudo = [(_mk_meta("b"), 0.6)]
        result = combine_results(sem, [], pseudo)
        assert len(result) == 2

    def test_source_priority_boost(self):
        sem_high = [(_mk_meta("a", metadata={"source_priority": 100}), 0.8)]
        sem_low = [(_mk_meta("b", metadata={"source_priority": 0}), 0.8)]
        result = combine_results(sem_high + sem_low, [])
        assert result[0]["element"]["id"] == "a"

    def test_empty_inputs(self):
        result = combine_results([], [])
        assert result == []

    def test_sorted_by_total_score(self):
        sem = [(_mk_meta("low"), 0.3), (_mk_meta("high"), 0.9)]
        result = combine_results(sem, [])
        assert result[0]["element"]["id"] == "high"

    def test_bm25_normalization(self):
        kw = [(_mk_meta("a"), 5.0), (_mk_meta("b"), 2.5)]
        result = combine_results([], kw)
        assert result[0]["keyword_score"] >= result[1]["keyword_score"]

    def test_semantic_weight_applied(self):
        sem = [(_mk_meta("a"), 1.0)]
        result = combine_results(sem, [], semantic_weight=0.5)
        assert abs(result[0]["semantic_score"] - 0.5) < 1e-9

    def test_keyword_weight_applied(self):
        kw = [(_mk_meta("a"), 2.0)]
        result = combine_results([], kw, keyword_weight=0.3)
        assert abs(result[0]["keyword_score"] - 0.3) < 1e-9


class TestSourcePriorityBoostExact:
    """Verify exact boost = 1 + min(max(priority, 0), 100) / 200."""

    def test_priority_100_boost(self) -> None:
        """priority=100: boost = 1 + 100/200 = 1.5, total = 0.8 * 1.5 = 1.2."""
        sem = [(_mk_meta("a", metadata={"source_priority": 100}), 0.8)]
        result = combine_results(sem, [])
        assert result[0]["total_score"] == pytest.approx(1.2)

    def test_priority_0_no_boost(self) -> None:
        """priority=0: boost = 1 + 0/200 = 1.0, total = 0.8 * 1.0 = 0.8."""
        sem = [(_mk_meta("a", metadata={"source_priority": 0}), 0.8)]
        result = combine_results(sem, [])
        assert result[0]["total_score"] == pytest.approx(0.8)

    def test_priority_50_half_boost(self) -> None:
        """priority=50: boost = 1 + 50/200 = 1.25, total = 0.8 * 1.25 = 1.0."""
        sem = [(_mk_meta("a", metadata={"source_priority": 50}), 0.8)]
        result = combine_results(sem, [])
        assert result[0]["total_score"] == pytest.approx(1.0)

    def test_negative_priority_clamped_to_zero(self) -> None:
        """priority=-10: clamped to 0, boost = 1.0, total = 0.8."""
        sem = [(_mk_meta("a", metadata={"source_priority": -10}), 0.8)]
        result = combine_results(sem, [])
        assert result[0]["total_score"] == pytest.approx(0.8)

    def test_priority_above_100_clamped(self) -> None:
        """priority=200: clamped to 100, boost = 1.5, total = 0.8 * 1.5 = 1.2."""
        sem = [(_mk_meta("a", metadata={"source_priority": 200}), 0.8)]
        result = combine_results(sem, [])
        assert result[0]["total_score"] == pytest.approx(1.2)
