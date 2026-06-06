"""Small stdlib-only time helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


class SystemClock:
    """Stdlib-backed clock implementation."""

    def utc_now(self) -> str:
        return utc_now()
