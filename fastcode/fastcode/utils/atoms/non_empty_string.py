"""Generic non-empty string value type."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NonEmptyString:
    """A trimmed string guaranteed to be non-empty."""

    value: str

    def __post_init__(self) -> None:
        normalized = self.value.strip()
        if not normalized:
            msg = "string must be non-empty"
            raise ValueError(msg)
        object.__setattr__(self, "value", normalized)

    @classmethod
    def parse(cls, value: str) -> NonEmptyString:
        return cls(str(value))

    def as_str(self) -> str:
        return self.value
