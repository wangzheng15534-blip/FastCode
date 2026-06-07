"""Generic strictly-positive integer value type."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class PositiveInt:
    """A validated integer greater than zero."""

    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            msg = "value must be > 0"
            raise ValueError(msg)

    @classmethod
    def parse(cls, value: int) -> PositiveInt:
        return cls(int(value))

    @classmethod
    def coerce(cls, value: PositiveInt | int) -> PositiveInt:
        if isinstance(value, cls):
            return value
        return cls.parse(int(value))

    def as_int(self) -> int:
        return self.value

    def __int__(self) -> int:
        return self.value
