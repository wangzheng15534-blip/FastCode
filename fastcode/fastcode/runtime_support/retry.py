"""Generic retry/backoff helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Generic exponential backoff policy."""

    minimum_seconds: int = 1
    base: int = 2
    maximum_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.minimum_seconds < 1:
            raise ValueError("minimum_seconds must be >= 1")
        if self.base < 2:
            raise ValueError("base must be >= 2")
        if self.maximum_seconds is not None and self.maximum_seconds < self.minimum_seconds:
            raise ValueError("maximum_seconds must be >= minimum_seconds")


DEFAULT_RETRY_POLICY = RetryPolicy()


def exponential_backoff_seconds(
    attempts: int,
    *,
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
) -> int:
    """Return a bounded exponential backoff delay for a retry attempt count."""
    bounded_attempts = max(0, int(attempts))
    delay = max(policy.minimum_seconds, policy.base**bounded_attempts)
    if policy.maximum_seconds is not None:
        delay = min(delay, policy.maximum_seconds)
    return delay
