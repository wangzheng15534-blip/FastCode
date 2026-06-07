"""Property-based contract tests for retrieval.ranking.scoring — meaning_core, ZERO test doubles.

These tests exercise open input spaces with Hypothesis to verify mathematical
invariants that must hold for all inputs.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.retrieval.ranking.scoring import (
    normalized_query_entropy,
    sigmoid,
    tokenize_signal,
    trace_confidence_weight,
    weighted_keyword_affinity,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_token_st = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=10),
    min_size=0,
    max_size=20,
)

_weights_st = st.dictionaries(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=5),
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    max_size=10,
)

_text_st = st.text(min_size=0, max_size=200)


# ---------------------------------------------------------------------------
# normalized_query_entropy properties
# ---------------------------------------------------------------------------


class TestNormalizedQueryEntropyProperties:
    @given(tokens=_token_st)
    @settings(max_examples=50)
    @pytest.mark.property
    def test_output_always_in_zero_to_one(self, tokens: list[str]) -> None:
        result = normalized_query_entropy(tokens)
        assert 0.0 <= result <= 1.0

    @given(tokens=_token_st)
    @settings(max_examples=50)
    @pytest.mark.property
    def test_empty_or_single_always_zero(self, tokens: list[str]) -> None:
        """0 or 1 tokens always produce entropy 0."""
        if len(tokens) <= 1:
            assert normalized_query_entropy(tokens) == pytest.approx(0.0)

    @given(tokens=st.lists(st.just("same"), min_size=2, max_size=20))
    @settings(max_examples=20)
    @pytest.mark.property
    def test_all_identical_tokens_zero_entropy(self, tokens: list[str]) -> None:
        assert normalized_query_entropy(tokens) == pytest.approx(0.0)

    @given(
        tokens=st.lists(st.sampled_from(["a", "b"]), min_size=2, max_size=10).filter(
            lambda ts: len(set(ts)) >= 2
        )
    )
    @settings(max_examples=30)
    @pytest.mark.property
    def test_two_distinct_tokens_approaching_one(self, tokens: list[str]) -> None:
        """With exactly 2 distinct tokens, entropy should be close to 1.0."""
        result = normalized_query_entropy(tokens)
        assert result > 0.0
        # Max entropy for 2 types is log2(2) = 1.0, so normalized should be <= 1.0
        assert result <= 1.0

    @given(tokens=_token_st)
    @settings(max_examples=30)
    @pytest.mark.property
    def test_monotonic_with_distinct_count(self, tokens: list[str]) -> None:
        """All-same has lower entropy than mixed tokens (when len > 1)."""
        if len(tokens) <= 1:
            return
        uniform = [tokens[0]] * len(tokens)
        e_uniform = normalized_query_entropy(uniform)
        e_mixed = normalized_query_entropy(tokens)
        assert e_uniform <= e_mixed + 1e-9


# ---------------------------------------------------------------------------
# sigmoid properties
# ---------------------------------------------------------------------------


class TestSigmoidProperties:
    @given(x=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False))
    @settings(max_examples=50)
    @pytest.mark.property
    def test_output_always_in_zero_to_one(self, x: float) -> None:
        result = sigmoid(x)
        assert 0.0 < result < 1.0  # strict: sigmoid never reaches 0 or 1 exactly

    @given(x=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False))
    @settings(max_examples=50)
    @pytest.mark.property
    def test_symmetry(self, x: float) -> None:
        assert sigmoid(x) + sigmoid(-x) == pytest.approx(1.0, abs=1e-10)

    @given(
        x=st.floats(min_value=-29.0, max_value=29.0, allow_nan=False),
    )
    @settings(max_examples=30)
    @pytest.mark.property
    def test_monotonic(self, x: float) -> None:
        """sigmoid(x + 0.1) > sigmoid(x) for finite inputs inside clamp range."""
        assert sigmoid(x + 0.1) > sigmoid(x)

    @given(x=st.floats(allow_nan=False, allow_infinity=False))
    @settings(max_examples=30)
    @pytest.mark.property
    def test_clamping_prevents_overflow(self, x: float) -> None:
        """Result is always a finite float."""
        result = sigmoid(x)
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# weighted_keyword_affinity properties
# ---------------------------------------------------------------------------


class TestWeightedKeywordAffinityProperties:
    @given(tokens=_token_st, weights=_weights_st)
    @settings(max_examples=50)
    @pytest.mark.property
    def test_output_always_in_zero_to_one(
        self, tokens: list[str], weights: dict[str, float]
    ) -> None:
        result = weighted_keyword_affinity(tokens, weights)
        assert 0.0 <= result <= 1.0

    @given(
        tokens=st.lists(
            st.text(alphabet="abc", min_size=1, max_size=3),
            min_size=0,
            max_size=5,
        ),
        weights=st.dictionaries(
            st.text(alphabet="abc", min_size=1, max_size=3),
            st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
            max_size=5,
        ),
    )
    @settings(max_examples=30)
    @pytest.mark.property
    def test_no_match_when_no_overlap(
        self, tokens: list[str], weights: dict[str, float]
    ) -> None:
        """If no token matches any weight key, result must be 0.0."""
        token_set = set(tokens)
        if not any(k in token_set for k in weights):
            assert weighted_keyword_affinity(tokens, weights) == pytest.approx(0.0)

    @given(
        terms=st.lists(
            st.text(alphabet="xyz", min_size=1, max_size=3),
            min_size=1,
            max_size=5,
            unique=True,
        ),
        weight_val=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
    )
    @settings(max_examples=30)
    @pytest.mark.property
    def test_full_match_equals_one(self, terms: list[str], weight_val: float) -> None:
        """If all tokens match all weights exactly, result must be 1.0."""
        weights = dict.fromkeys(terms, weight_val)
        assert weighted_keyword_affinity(terms, weights) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# tokenize_signal properties
# ---------------------------------------------------------------------------


class TestTokenizeSignalProperties:
    @given(text=_text_st)
    @settings(max_examples=50)
    @pytest.mark.property
    def test_output_always_lowercase_alphanum(self, text: str) -> None:
        tokens = tokenize_signal(text)
        for token in tokens:
            assert token == token.lower()
            assert token.isalnum() or "_" in token

    @given(text=_text_st)
    @settings(max_examples=50)
    @pytest.mark.property
    def test_no_empty_tokens(self, text: str) -> None:
        tokens = tokenize_signal(text)
        assert all(len(t) > 0 for t in tokens)

    @given(text=_text_st)
    @settings(max_examples=30)
    @pytest.mark.property
    def test_concatenation_roundtrip_within_vocab(self, text: str) -> None:
        """Tokens joined by space should re-tokenize to same tokens."""
        tokens = tokenize_signal(text)
        rejoined = " ".join(tokens)
        retokenized = tokenize_signal(rejoined)
        assert retokenized == tokens


# ---------------------------------------------------------------------------
# trace_confidence_weight properties
# ---------------------------------------------------------------------------


class TestTraceConfidenceWeightProperties:
    @given(label=st.text(min_size=0, max_size=50))
    @settings(max_examples=30)
    @pytest.mark.property
    def test_always_returns_positive_float(self, label: str) -> None:
        result = trace_confidence_weight(label)
        assert isinstance(result, float)
        assert result > 0.0

    @given(label=st.one_of(st.none(), st.just("")))
    @settings(max_examples=5)
    @pytest.mark.property
    def test_none_or_empty_returns_default(self, label: str | None) -> None:
        assert trace_confidence_weight(label) == pytest.approx(0.6)
