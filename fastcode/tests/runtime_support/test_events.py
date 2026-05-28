"""Tests for runtime lifecycle event contracts."""

from __future__ import annotations

import pytest

from fastcode.runtime_support.events import (
    AgentContextEvent,
    AgentContextEventType,
    PipelineStageEvent,
    PipelineStageStatus,
)


def test_pipeline_stage_event_uses_explicit_status_enum() -> None:
    event = PipelineStageEvent(
        stage="parse",
        status="started",
        warnings=["legacy warning"],
    )

    assert event.status is PipelineStageStatus.STARTED
    assert event.status == "started"
    assert event.warnings == ("legacy warning",)


def test_agent_context_event_uses_explicit_event_type_enum() -> None:
    event = AgentContextEvent(
        session_id="session-1",
        turn_number=1,
        event_type="observation_appended",
    )

    assert event.event_type is AgentContextEventType.OBSERVATION_APPENDED
    assert event.event_type == "observation_appended"


def test_runtime_events_reject_invalid_state_values() -> None:
    with pytest.raises(ValueError, match=r"unknown"):
        PipelineStageEvent(stage="parse", status="unknown")
    with pytest.raises(ValueError, match=r"turn_number"):
        AgentContextEvent(
            session_id="session-1",
            turn_number=-1,
            event_type=AgentContextEventType.TURN_STARTED,
        )
