"""Contract tests for fastcode.utils.atoms — meaning_seed, ZERO test doubles.

NonEmptyString and PositiveInt are value objects used throughout the codebase.
These tests guard their construction, validation, and comparison contracts.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.utils.atoms.non_empty_string import NonEmptyString
from fastcode.utils.atoms.positive_int import PositiveInt

# ---------------------------------------------------------------------------
# NonEmptyString
# ---------------------------------------------------------------------------


class TestNonEmptyStringContracts:
    def test_parse_strips_whitespace(self) -> None:
        assert NonEmptyString.parse("  hello  ").as_str() == "hello"

    def test_parse_preserves_inner_content(self) -> None:
        assert NonEmptyString.parse("hello world").as_str() == "hello world"

    @pytest.mark.negative
    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            NonEmptyString.parse("")

    @pytest.mark.negative
    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            NonEmptyString.parse("   \n\t  ")

    def test_frozen(self) -> None:
        ns = NonEmptyString.parse("hello")
        with pytest.raises(AttributeError):
            ns.value = "other"  # type: ignore[misc]

    @given(s=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()))
    @settings(max_examples=50)
    @pytest.mark.property
    def test_non_blank_strings_always_accepted(self, s: str) -> None:
        result = NonEmptyString.parse(s)
        assert result.as_str() == s.strip()

    @given(s=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=20))
    @settings(max_examples=30)
    @pytest.mark.property
    def test_parse_roundtrip(self, s: str) -> None:
        parsed = NonEmptyString.parse(s)
        assert NonEmptyString.parse(parsed.as_str()).as_str() == parsed.as_str()


# ---------------------------------------------------------------------------
# PositiveInt
# ---------------------------------------------------------------------------


class TestPositiveIntContracts:
    def test_parse_accepts_positive(self) -> None:
        assert PositiveInt.parse(5).as_int() == 5

    def test_as_int_matches_value(self) -> None:
        p = PositiveInt.parse(42)
        assert p.as_int() == p.value

    @pytest.mark.negative
    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            PositiveInt.parse(0)

    @pytest.mark.negative
    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            PositiveInt.parse(-1)

    def test_coerce_from_int(self) -> None:
        p = PositiveInt.coerce(10)
        assert p.as_int() == 10

    def test_coerce_from_positive_int(self) -> None:
        original = PositiveInt.parse(7)
        coerced = PositiveInt.coerce(original)
        assert coerced is original  # identity: already a PositiveInt

    def test_dunder_int(self) -> None:
        p = PositiveInt.parse(99)
        assert int(p) == 99

    def test_ordering(self) -> None:
        a = PositiveInt.parse(3)
        b = PositiveInt.parse(5)
        assert a < b
        assert b > a
        assert a != b

    def test_equality(self) -> None:
        a = PositiveInt.parse(5)
        b = PositiveInt.parse(5)
        assert a == b

    @given(n=st.integers(min_value=1, max_value=10000))
    @settings(max_examples=50)
    @pytest.mark.property
    def test_positive_integers_always_accepted(self, n: int) -> None:
        assert PositiveInt.parse(n).as_int() == n

    @given(n=st.integers(max_value=0))
    @settings(max_examples=30)
    @pytest.mark.property
    def test_non_positive_always_rejected(self, n: int) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            PositiveInt.parse(n)
