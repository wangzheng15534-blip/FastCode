"""Property-based tests for repo_selector pure functions."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.repo_selector import RepositorySelector


def _make_selector() -> Any:
    """Create minimal selector with logger for testing pure methods."""
    selector = RepositorySelector.__new__(RepositorySelector)
    selector.logger = logging.getLogger(__name__)
    return selector


# --- Strategies ---

name_st = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
    min_size=1,
    max_size=20,
)

names_list_st = st.lists(name_st, min_size=1, max_size=5, unique=True)


# --- Properties ---


class TestNormalize:
    @given(name=st.text(min_size=1, max_size=20))
    @settings(max_examples=30)
    def test_always_returns_lowercase_property(self, name: str):
        """HAPPY: _normalize always returns lowercase regardless of input case."""
        result = RepositorySelector._normalize(name)
        assert result == result.lower()

    def test_strips_whitespace_property(self):
        """HAPPY: leading/trailing whitespace stripped."""
        assert RepositorySelector._normalize("  hello  ") == "hello"

    def test_strips_backticks_property(self):
        """HAPPY: backticks stripped."""
        assert RepositorySelector._normalize("`repo`") == "repo"

    def test_strips_asterisks_property(self):
        """HAPPY: asterisks stripped."""
        assert RepositorySelector._normalize("*repo*") == "repo"

    def test_strips_quotes_property(self):
        """HAPPY: single and double quotes stripped."""
        assert RepositorySelector._normalize("'repo'") == "repo"
        assert RepositorySelector._normalize('"repo"') == "repo"

    @pytest.mark.edge
    def test_empty_after_strip_property(self):
        """EDGE: string of only special chars returns empty."""
        result = RepositorySelector._normalize("``**''")
        assert result == ""

    @pytest.mark.edge
    def test_mixed_special_chars_property(self):
        """EDGE: mixed special chars all stripped."""
        result = RepositorySelector._normalize("*`'hello\"`*")
        assert "hello" in result

    @given(name=st.text(min_size=0, max_size=20))
    @settings(max_examples=20)
    def test_always_returns_string_property(self, name: str):
        """HAPPY: _normalize always returns a string."""
        result = RepositorySelector._normalize(name)
        assert isinstance(result, str)


class TestFuzzyMatchRepo:
    def test_exact_match_property(self):
        """HAPPY: exact name match returns the name."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("myrepo", ["myrepo", "other"])
        assert result == "myrepo"

    def test_case_insensitive_match_property(self):
        """HAPPY: case-insensitive exact match."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("MyRepo", ["myrepo", "other"])
        assert result == "myrepo"

    def test_substring_candidate_in_name_property(self):
        """HAPPY: candidate is substring of available name."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("repo", ["myrepo", "other"])
        assert result == "myrepo"

    def test_substring_name_in_candidate_property(self):
        """HAPPY: available name is substring of candidate."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("myrepo", ["repo", "other"])
        assert result == "repo"

    @pytest.mark.edge
    def test_no_match_returns_none_property(self):
        """EDGE: no match returns None."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("xyz", ["abc", "def"])
        assert result is None

    @pytest.mark.edge
    def test_empty_candidate_returns_none_property(self):
        """EDGE: empty candidate after normalization returns None."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("``", ["repo"])
        assert result is None

    @pytest.mark.edge
    def test_empty_available_returns_none_property(self):
        """EDGE: empty available list returns None."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("repo", [])
        assert result is None

    def test_jaccard_token_match_property(self):
        """HAPPY: token overlap >= 0.5 returns match."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("fast-code", ["fast_code", "other"])
        # "fast code" tokens overlap with "fast code" -> Jaccard 1.0
        assert result == "fast_code"

    @pytest.mark.edge
    def test_jaccard_below_threshold_property(self):
        """EDGE: token overlap < 0.5 returns None."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("abc", ["xyz", "def"])
        assert result is None

    @given(candidate=name_st, available=names_list_st)
    @settings(max_examples=20)
    def test_result_is_none_or_member_property(self, candidate: Any, available: Any):
        """HAPPY: result is always None or one of available names."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo(candidate, available)
        assert result is None or result in available
