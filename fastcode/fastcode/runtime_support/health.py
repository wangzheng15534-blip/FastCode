"""Generic runtime health helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HealthStatus:
    """Generic runtime health result."""

    healthy: bool
    status: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "status": self.status,
            "details": dict(self.details),
        }


def readiness_health(
    *,
    repo_loaded: bool,
    repo_indexed: bool,
    details: dict[str, Any] | None = None,
) -> HealthStatus:
    status = "healthy" if repo_loaded else "not_ready"
    if repo_loaded and not repo_indexed:
        status = "degraded"
    return HealthStatus(
        healthy=repo_loaded,
        status=status,
        details={
            "repo_loaded": bool(repo_loaded),
            "repo_indexed": bool(repo_indexed),
            **dict(details or {}),
        },
    )
