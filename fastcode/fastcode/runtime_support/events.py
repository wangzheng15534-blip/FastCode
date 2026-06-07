"""Runtime lifecycle event contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


def _empty_payload() -> dict[str, Any]:
    return {}


class PipelineStageStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentContextEventType(StrEnum):
    TURN_STARTED = "turn_started"
    OBSERVATION_APPENDED = "observation_appended"
    TURN_COMPLETED = "turn_completed"


def _pipeline_stage_status(value: PipelineStageStatus | str) -> PipelineStageStatus:
    if isinstance(value, PipelineStageStatus):
        return value
    return PipelineStageStatus(str(value))


def _agent_context_event_type(
    value: AgentContextEventType | str,
) -> AgentContextEventType:
    if isinstance(value, AgentContextEventType):
        return value
    return AgentContextEventType(str(value))


@dataclass(frozen=True)
class PipelineStageEvent:
    """A lifecycle event emitted by indexing/query shell stages."""

    stage: str
    status: PipelineStageStatus
    snapshot_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=_empty_payload)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _pipeline_stage_status(self.status))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))


@dataclass(frozen=True)
class AgentContextEvent:
    """A lifecycle event emitted by agent-context workflows."""

    session_id: str
    turn_number: int
    event_type: AgentContextEventType
    payload: dict[str, Any] = field(default_factory=_empty_payload)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "event_type", _agent_context_event_type(self.event_type)
        )
        if self.turn_number < 0:
            msg = "agent_context.turn_number must be >= 0"
            raise ValueError(msg)
