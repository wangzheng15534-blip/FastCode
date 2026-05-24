"""Durable job queue capability ports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class IndexRunView(Protocol):
    """Read-only index run shape used across app/runtime boundaries."""

    @property
    def run_id(self) -> str: ...

    @property
    def repo_name(self) -> str: ...

    @property
    def snapshot_id(self) -> str: ...

    @property
    def branch(self) -> str | None: ...

    @property
    def commit_id(self) -> str | None: ...

    @property
    def idempotency_key(self) -> str | None: ...

    @property
    def status(self) -> str: ...

    @property
    def error_message(self) -> str | None: ...

    @property
    def warnings_json(self) -> str | None: ...

    @property
    def created_at(self) -> str: ...

    @property
    def started_at(self) -> str | None: ...

    @property
    def completed_at(self) -> str | None: ...


class RedoTaskView(Protocol):
    """Read-only redo task shape used across app/runtime boundaries."""

    @property
    def task_id(self) -> str: ...

    @property
    def task_type(self) -> str: ...

    @property
    def payload_json(self) -> str: ...

    @property
    def status(self) -> str: ...

    @property
    def attempts(self) -> int: ...

    @property
    def last_error(self) -> str | None: ...

    @property
    def next_attempt_at(self) -> str | None: ...

    @property
    def created_at(self) -> str: ...

    @property
    def updated_at(self) -> str: ...


class PublishRetryTaskView(Protocol):
    """Read-only publish retry task shape used across app/runtime boundaries."""

    @property
    def task_id(self) -> str: ...

    @property
    def run_id(self) -> str: ...

    @property
    def snapshot_id(self) -> str: ...

    @property
    def manifest_id(self) -> str | None: ...

    @property
    def status(self) -> str: ...

    @property
    def attempts(self) -> int: ...

    @property
    def last_error(self) -> str | None: ...

    @property
    def created_at(self) -> str: ...

    @property
    def updated_at(self) -> str: ...


class RedoJobQueue(Protocol):
    """Durable redo job queue capability."""

    def enqueue_redo_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        error: str | None = None,
    ) -> str: ...

    def claim_redo_task_record(self) -> RedoTaskView | None: ...

    def claim_redo_task(self) -> Mapping[str, Any] | None: ...

    def mark_redo_task_done(self, task_id: str) -> None: ...

    def mark_redo_task_failed(
        self, task_id: str, error: str, max_attempts: int = 5
    ) -> None: ...


class PublishRetryQueue(Protocol):
    """Durable publish retry job queue capability."""

    def enqueue_publish_retry(
        self,
        run_id: str,
        snapshot_id: str,
        manifest_id: str | None,
        error_message: str,
    ) -> str: ...

    def claim_next_publish_task_record(self) -> PublishRetryTaskView | None: ...

    def mark_publish_task_done(self, task_id: str) -> None: ...

    def mark_publish_task_failed(self, task_id: str, error: str) -> None: ...


class IndexRunStore(PublishRetryQueue, Protocol):
    """Index-run state capability plus its durable publish retry queue."""

    def get_run_record(self, run_id: str) -> IndexRunView | None: ...

    def mark_completed(
        self,
        run_id: str,
        status: str = "succeeded",
        warnings: list[str] | None = None,
    ) -> None: ...
