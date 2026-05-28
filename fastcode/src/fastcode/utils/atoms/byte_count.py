"""Generic byte-count value type."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class ByteCount:
    """A validated non-negative byte count."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("byte count must be >= 0")

    @classmethod
    def from_bytes(cls, value: int) -> ByteCount:
        return cls(int(value))

    @classmethod
    def from_kib(cls, value: int) -> ByteCount:
        return cls(int(value) * 1024)

    @classmethod
    def from_mib(cls, value: int) -> ByteCount:
        return cls(int(value) * 1024 * 1024)

    @classmethod
    def coerce(cls, value: ByteCount | int) -> ByteCount:
        if isinstance(value, cls):
            return value
        return cls(int(value))

    def as_bytes(self) -> int:
        return self.value

    def __int__(self) -> int:
        return self.value
