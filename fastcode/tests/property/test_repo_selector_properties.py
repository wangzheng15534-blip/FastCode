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


@pytest.mark.property
class TestNormalize:
    @given(name=st.text(min_size=1, max_size=20))
    @settings(max_examples=30)
    @pytest.mark.basic
    def test_always_returns_lowercase(self, name: str):
        """HAPPY: _normalize always returns lowercase regardless of input case."""
        result = RepositorySelector._normalize(name)
        assert result == result.lower()

    @pytest.mark.basic
    def test_strips_whitespace(self):
        """HAPPY: leading/trailing whitespace stripped."""
        assert RepositorySelector._normalize("  hello  ") == "hello"

    @pytest.mark.basic
    def test_strips_backticks(self):
        """HAPPY: backticks stripped."""
        assert RepositorySelector._normalize("`repo`") == "repo"

    @pytest.mark.basic
    def test_strips_asterisks(self):
        """HAPPY: asterisks stripped."""
        assert RepositorySelector._normalize("*repo*") == "repo"

    @pytest.mark.basic
    def test_strips_quotes(self):
        """HAPPY: single and double quotes stripped."""
        assert RepositorySelector._normalize("'repo'") == "repo"
        assert RepositorySelector._normalize('"repo"') == "repo"

    @pytest.mark.edge
    def test_empty_after_strip(self):
        """EDGE: string of only special chars returns empty."""
        result = RepositorySelector._normalize("``**''")
        assert result == ""

    @pytest.mark.edge
    def test_mixed_special_chars(self):
        """EDGE: mixed special chars all stripped."""
        result = RepositorySelector._normalize("*`'hello\"`*")
        assert "hello" in result

    @given(name=st.text(min_size=0, max_size=20))
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_always_returns_string(self, name: str):
        """HAPPY: _normalize always returns a string."""
        result = RepositorySelector._normalize(name)
        assert isinstance(result, str)


@pytest.mark.property
class TestFuzzyMatchRepo:
    @pytest.mark.basic
    def test_exact_match(self):
        """HAPPY: exact name match returns the name."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("myrepo", ["myrepo", "other"])
        assert result == "myrepo"

    @pytest.mark.basic
    def test_case_insensitive_match(self):
        """HAPPY: case-insensitive exact match."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("MyRepo", ["myrepo", "other"])
        assert result == "myrepo"

    @pytest.mark.basic
    def test_substring_candidate_in_name(self):
        """HAPPY: candidate is substring of available name."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("repo", ["myrepo", "other"])
        assert result == "myrepo"

    @pytest.mark.basic
    def test_substring_name_in_candidate(self):
        """HAPPY: available name is substring of candidate."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("myrepo", ["repo", "other"])
        assert result == "repo"

    @pytest.mark.edge
    def test_no_match_returns_none(self):
        """EDGE: no match returns None."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("xyz", ["abc", "def"])
        assert result is None

    @pytest.mark.edge
    def test_empty_candidate_returns_none(self):
        """EDGE: empty candidate after normalization returns None."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("``", ["repo"])
        assert result is None

    @pytest.mark.edge
    def test_empty_available_returns_none(self):
        """EDGE: empty available list returns None."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("repo", [])
        assert result is None

    @pytest.mark.basic
    def test_jaccard_token_match(self):
        """HAPPY: token overlap >= 0.5 returns match."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("fast-code", ["fast_code", "other"])
        # "fast code" tokens overlap with "fast code" -> Jaccard 1.0
        assert result == "fast_code"

    @pytest.mark.edge
    def test_jaccard_below_threshold(self):
        """EDGE: token overlap < 0.5 returns None."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo("abc", ["xyz", "def"])
        assert result is None

    @given(candidate=name_st, available=names_list_st)
    @settings(max_examples=20)
    @pytest.mark.basic
    def test_result_is_none_or_member(self, candidate: Any, available: Any):
        """HAPPY: result is always None or one of available names."""
        selector = _make_selector()
        result = selector._fuzzy_match_repo(candidate, available)
        assert result is None or result in available
