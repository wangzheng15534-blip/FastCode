"""Small stdlib-only ID helpers."""

from __future__ import annotations

import uuid

from ..foundation.non_empty_string import NonEmptyString
from ..foundation.positive_int import PositiveInt


def new_prefixed_id(prefix: str, *, length: int = 16) -> str:
    """Return an opaque ID with a stable prefix and random hex suffix."""
    validated_prefix = NonEmptyString.parse(prefix)
    validated_length = PositiveInt.parse(length)
    return (
        f"{validated_prefix.as_str()}_{uuid.uuid4().hex[: validated_length.as_int()]}"
    )


class PrefixedIdGenerator:
    """Stdlib-backed opaque ID generator implementation."""

    def new_id(self, prefix: str, *, length: int = 16) -> str:
        return new_prefixed_id(prefix, length=length)
