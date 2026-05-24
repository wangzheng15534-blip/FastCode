"""Runtime determinism capability ports."""

from __future__ import annotations

from typing import Protocol


class Clock(Protocol):
    """UTC timestamp capability for persisted app/runtime records."""

    def utc_now(self) -> str: ...


class IdGenerator(Protocol):
    """Opaque ID generation capability for app/runtime stores."""

    def new_id(self, prefix: str, *, length: int = 16) -> str: ...
