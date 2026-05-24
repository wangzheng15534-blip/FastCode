"""Outbound publishing capability ports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from fastcode.ir.types import IRSnapshot


class OutboxEventView(Protocol):
    """Read-only publish outbox event shape used across app/runtime boundaries."""

    @property
    def event_id(self) -> str: ...

    @property
    def event_type(self) -> str: ...

    @property
    def payload(self) -> str: ...

    @property
    def snapshot_id(self) -> str: ...

    @property
    def status(self) -> str: ...

    @property
    def attempts(self) -> int: ...

    @property
    def max_attempts(self) -> int: ...

    @property
    def created_at(self) -> str: ...

    @property
    def last_attempt_at(self) -> str | None: ...

    @property
    def error_message(self) -> str | None: ...


class EventSink(Protocol):
    """Durable event sink capability for publish outbox workflows."""

    def enqueue_outbox_event(
        self,
        event_id: str,
        event_type: str,
        payload: str,
        snapshot_id: str,
        max_attempts: int = 5,
    ) -> bool: ...

    def claim_outbox_event_records(
        self, limit: int = 10
    ) -> Sequence[OutboxEventView]: ...

    def claim_outbox_event(self, limit: int = 10) -> Sequence[Mapping[str, Any]]: ...

    def mark_outbox_event_done(self, event_id: str) -> None: ...

    def mark_outbox_event_failed(self, event_id: str, error: str) -> None: ...

    def get_outbox_pending_count(self) -> int: ...


class LineagePublisher(Protocol):
    """External lineage publication capability used by indexing workflows."""

    def is_configured(self) -> bool:
        """Return True when lineage publication is available."""
        ...

    def publish_snapshot_lineage_for_snapshot(
        self,
        snapshot: IRSnapshot,
        manifest: Any,
        git_meta: dict[str, Any],
        previous_snapshot_symbols: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        """Publish lineage for a typed snapshot without forcing dict materialization."""
        ...

    def flush_outbox(
        self, snapshot_store: EventSink, limit: int = 10
    ) -> dict[str, int]:
        """Flush queued lineage publication events."""
        ...
