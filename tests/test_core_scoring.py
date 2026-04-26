"""Tests for fastcode.core.scoring — pure scoring functions.

Each test is independent, self-contained, and requires no HybridRetriever.
"""

from __future__ import annotations

from typing import Any

import pytest

from fastcode.core.scoring import (
    clone_result_row,
    normalized_query_entropy,
    normalized_totals,
    sigmoid,
    tokenize_signal,
    trace_confidence_weight,
    weighted_keyword_affinity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_row(elem_id: str, total: float) -> dict[str, Any]:
    """Create a minimal retrieval result row matching retriever.py's format."""
    return {
        "element": {"id": elem_id, "type": "function", "name": elem_id},
        "semantic_score": total,
        "keyword_score": total * 0.5,
        "pseudocode_score": 0.0,
        "graph_score": 0.0,
        "total_score": total,
    }


# ---------------------------------------------------------------------------
# clone_result_row
# ---------------------------------------------------------------------------


class TestCloneResultRow:
    def test_clones_dicts_and_lists(self) -> None:
        row = _mk_row("a", 1.0)
        row["element"]["metadata"] = {"key": "value"}
        row["tags"] = ["x", "y"]

        clone = clone_result_row(row)

        # Same values
        assert clone["element"]["id"] == "a"
        assert clone["element"]["metadata"] == {"key": "value"}
        assert clone["tags"] == ["x", "y"]

        # Top-level dict and list mutations are independent
        clone["element"]["id"] = "b"
        clone["element"]["metadata"] = {"other": "thing"}
        clone["tags"].append("z")

        assert row["element"]["id"] == "a"
        assert row["element"]["metadata"] == {"key": "value"}
        assert row["tags"] == ["x", "y"]

    def test_preserves_scalar_values(self) -> None:
        row: dict[str, Any] = {"score": 0.85, "name": "foo", "count": 3}
        clone = clone_result_row(row)
        assert clone is not row
        assert clone["score"] == 0.85

    def test_empty_row(self) -> None:
        assert clone_result_row({}) == {}


# ---------------------------------------------------------------------------
# normalized_totals
# ---------------------------------------------------------------------------


class TestNormalizedTotals:
    def test_empty_results(self) -> None:
        assert normalized_totals([]) == {}

    def test_positive_scores_normalized_by_max(self) -> None:
        results = [_mk_row("a", 0.5), _mk_row("b", 1.0), _mk_row("c", 0.25)]
        out = normalized_totals(results)
        assert out["a"] == pytest.approx(0.5)
        assert out["b"] == pytest.approx(1.0)
        assert out["c"] == pytest.approx(0.25)

    def test_zero_scores_use_rank_fallback(self) -> None:
        results = [_mk_row("a", 0.0), _mk_row("b", 0.0), _mk_row("c", 0.0)]
        out = normalized_totals(results)
        assert out["a"] == pytest.approx(1.0)
        assert out["b"] == pytest.approx(0.5)
        assert out["c"] == pytest.approx(1.0 / 3.0)

    def test_missing_total_score_treated_as_zero(self) -> None:
        row: dict[str, Any] = {"element": {"id": "x", "type": "f", "name": "x"}}
        results = [_mk_row("a", 0.0), row, _mk_row("b", 0.0)]
        out = normalized_totals(results)
        assert "x" in out
        # rank-based: a=1.0, x=0.5, b=1/3
        assert out["x"] == pytest.approx(0.5)

    def test_none_total_score_treated_as_zero(self) -> None:
        row: dict[str, Any] = {
            "element": {"id": "x", "type": "f", "name": "x"},
            "total_score": None,
        }
        results = [row]
        out = normalized_totals(results)
        assert out["x"] == pytest.approx(1.0)

    def test_skips_rows_without_element_id(self) -> None:
        row: dict[str, Any] = {"total_score": 1.0}
        results = [row, _mk_row("a", 0.8)]
        out = normalized_totals(results)
        assert "a" in out
        assert len(out) == 1

    def test_scores_clamped_to_zero_one(self) -> None:
        # total_score > max_score shouldn't happen in practice, but clamp is safe
        results = [_mk_row("a", 1.0), _mk_row("b", 1.0)]
        out = normalized_totals(results)
        assert out["a"] == pytest.approx(1.0)
        assert out["b"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# trace_confidence_weight
# ---------------------------------------------------------------------------


class TestTraceConfidenceWeight:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("precise", 1.0),
            ("anchored", 1.0),
            ("resolved", 0.8),
            ("derived", 0.7),
            ("heuristic", 0.6),
            ("candidate", 0.5),
        ],
    )
    def test_known_labels(self, label: str, expected: float) -> None:
        assert trace_confidence_weight(label) == pytest.approx(expected)

    def test_unknown_label_defaults_to_0_6(self) -> None:
        assert trace_confidence_weight("unknown") == pytest.approx(0.6)

    def test_none_defaults_to_0_6(self) -> None:
        assert trace_confidence_weight(None) == pytest.approx(0.6)

    def test_case_insensitive(self) -> None:
        assert trace_confidence_weight("Precise") == pytest.approx(1.0)
        assert trace_confidence_weight("CANDIDATE") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------


class TestSigmoid:
    def test_zero(self) -> None:
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self) -> None:
        assert sigmoid(30.0) == pytest.approx(1.0)

    def test_large_negative(self) -> None:
        assert sigmoid(-30.0) == pytest.approx(0.0)

    def test_clamping(self) -> None:
        # Values beyond [-30, 30] are clamped, so result stays stable
        assert sigmoid(1000.0) == pytest.approx(sigmoid(30.0))
        assert sigmoid(-1000.0) == pytest.approx(sigmoid(-30.0))

    def test_symmetry(self) -> None:
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            assert sigmoid(x) + sigmoid(-x) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# tokenize_signal
# ---------------------------------------------------------------------------


class TestTokenizeSignal:
    def test_basic(self) -> None:
        assert tokenize_signal("hello world") == ["hello", "world"]

    def test_numbers_and_underscores(self) -> None:
        assert tokenize_signal("foo_bar 123") == ["foo_bar", "123"]

    def test_strips_uppercase(self) -> None:
        assert tokenize_signal("HelloWorld") == ["helloworld"]

    def test_special_chars_removed(self) -> None:
        result = tokenize_signal("foo@bar#baz!")
        assert result == ["foo", "bar", "baz"]

    def test_empty_string(self) -> None:
        assert tokenize_signal("") == []

    def test_none_treated_as_empty(self) -> None:
        assert tokenize_signal(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# normalized_query_entropy
# ---------------------------------------------------------------------------


class TestNormalizedQueryEntropy:
    def test_empty_tokens(self) -> None:
        assert normalized_query_entropy([]) == pytest.approx(0.0)

    def test_single_token(self) -> None:
        assert normalized_query_entropy(["hello"]) == pytest.approx(0.0)

    def test_uniform_distribution(self) -> None:
        # Two distinct tokens, each appearing once -> max entropy -> 1.0
        assert normalized_query_entropy(["a", "b"]) == pytest.approx(1.0)

    def test_skewed_distribution(self) -> None:
        # ["a", "a", "b"] -> entropy < 1.0
        e = normalized_query_entropy(["a", "a", "b"])
        assert 0.0 < e < 1.0

    def test_all_same_token(self) -> None:
        assert normalized_query_entropy(["a", "a", "a"]) == pytest.approx(0.0)

    def test_returns_value_in_0_to_1(self) -> None:
        e = normalized_query_entropy(["foo", "bar", "baz", "qux"])
        assert 0.0 <= e <= 1.0


# ---------------------------------------------------------------------------
# weighted_keyword_affinity
# ---------------------------------------------------------------------------


class TestWeightedKeywordAffinity:
    def test_empty_tokens(self) -> None:
        assert weighted_keyword_affinity([], {"auth": 1.0}) == pytest.approx(0.0)

    def test_empty_weights(self) -> None:
        assert weighted_keyword_affinity(["auth"], {}) == pytest.approx(0.0)

    def test_full_match(self) -> None:
        assert weighted_keyword_affinity(
            ["auth", "login"],
            {"auth": 1.0, "login": 1.0},
        ) == pytest.approx(1.0)

    def test_partial_match(self) -> None:
        result = weighted_keyword_affinity(
            ["auth", "login"],
            {"auth": 1.0, "logout": 1.0, "session": 1.0},
        )
        assert result == pytest.approx(1.0 / 3.0)

    def test_no_match(self) -> None:
        result = weighted_keyword_affinity(
            ["database", "query"],
            {"auth": 1.0, "login": 1.0},
        )
        assert result == pytest.approx(0.0)

    def test_negative_weights_ignored(self) -> None:
        # Negative weights are treated as zero
        result = weighted_keyword_affinity(
            ["auth"],
            {"auth": 1.0, "spam": -5.0},
        )
        # total = max(0, 1.0) + max(0, -5.0) = 1.0
        # matched = 1.0
        assert result == pytest.approx(1.0)

    def test_all_negative_weights_returns_zero(self) -> None:
        result = weighted_keyword_affinity(
            ["auth"],
            {"auth": -1.0},
        )
        assert result == pytest.approx(0.0)

    def test_clamped_to_one(self) -> None:
        # Matching tokens whose combined weight exceeds total (shouldn't happen
        # with non-negative weights, but guard exists)
        result = weighted_keyword_affinity(
            ["a", "b", "c"],
            {"a": 0.5, "b": 0.5, "c": 0.5},
        )
        assert result == pytest.approx(1.0)
