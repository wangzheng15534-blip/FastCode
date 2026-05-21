"""Lifecycle event contracts for pipeline and agent context flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _empty_payload() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class PipelineStageEvent:
    """A lifecycle event emitted by indexing/query shell stages."""

    stage: str
    status: str
    snapshot_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=_empty_payload)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentContextEvent:
    """A lifecycle event emitted by agent-context workflows."""

    session_id: str
    turn_number: int
    event_type: str
    payload: dict[str, Any] = field(default_factory=_empty_payload)
