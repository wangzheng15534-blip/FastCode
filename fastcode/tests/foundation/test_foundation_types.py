from __future__ import annotations

import pytest

from fastcode.utils.atoms.byte_count import ByteCount
from fastcode.utils.atoms.non_empty_string import NonEmptyString
from fastcode.utils.atoms.positive_int import PositiveInt


def test_byte_count_from_mib() -> None:
    assert ByteCount.from_mib(1).as_bytes() == 1024 * 1024


def test_byte_count_rejects_negative() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        ByteCount(-1)


def test_non_empty_string_trims() -> None:
    assert NonEmptyString.parse("  repo  ").as_str() == "repo"


def test_non_empty_string_rejects_blank() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        NonEmptyString.parse("   ")


def test_positive_int_rejects_zero() -> None:
    with pytest.raises(ValueError, match="> 0"):
        PositiveInt.parse(0)


def test_positive_int_coerce_keeps_value() -> None:
    value = PositiveInt.coerce(7)
    assert value.as_int() == 7
