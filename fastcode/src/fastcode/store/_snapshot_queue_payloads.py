"""Private row and payload adapters for snapshot retry queues."""
# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import Any

from .snapshot_contracts import OutboxEventRecord, RedoTaskRecord


def _row_value(row: Any, index: int, key: str) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (IndexError, KeyError, TypeError):
        try:
            return row[index]
        except (IndexError, KeyError, TypeError):
            return None


def _row_to_redo_task_record(row: Any) -> RedoTaskRecord | None:
    task_id = _row_value(row, 0, "task_id")
    if task_id is None:
        return None
    return RedoTaskRecord(
        task_id=str(task_id),
        task_type=str(_row_value(row, 1, "task_type") or ""),
        payload_json=str(_row_value(row, 2, "payload_json") or ""),
        status=str(_row_value(row, 3, "status") or ""),
        attempts=int(_row_value(row, 4, "attempts") or 0),
        last_error=(
            str(last_error)
            if (last_error := _row_value(row, 5, "last_error")) is not None
            else None
        ),
        next_attempt_at=(
            str(next_attempt_at)
            if (next_attempt_at := _row_value(row, 6, "next_attempt_at")) is not None
            else None
        ),
        created_at=str(_row_value(row, 7, "created_at") or ""),
        updated_at=str(_row_value(row, 8, "updated_at") or ""),
    )


def _redo_task_payload(record: RedoTaskRecord) -> dict[str, Any]:
    return {
        "task_id": record.task_id,
        "task_type": record.task_type,
        "payload_json": record.payload_json,
        "status": record.status,
        "attempts": record.attempts,
        "last_error": record.last_error,
        "next_attempt_at": record.next_attempt_at,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _row_to_outbox_event_record(row: Any) -> OutboxEventRecord | None:
    event_id = _row_value(row, 0, "event_id")
    if event_id is None:
        return None
    return OutboxEventRecord(
        event_id=str(event_id),
        event_type=str(_row_value(row, 1, "event_type") or ""),
        payload=str(_row_value(row, 2, "payload") or ""),
        snapshot_id=str(_row_value(row, 3, "snapshot_id") or ""),
        status=str(_row_value(row, 4, "status") or ""),
        attempts=int(_row_value(row, 5, "attempts") or 0),
        max_attempts=int(_row_value(row, 6, "max_attempts") or 0),
        created_at=str(_row_value(row, 7, "created_at") or ""),
        last_attempt_at=(
            str(last_attempt_at)
            if (last_attempt_at := _row_value(row, 8, "last_attempt_at")) is not None
            else None
        ),
        error_message=(
            str(error_message)
            if (error_message := _row_value(row, 9, "error_message")) is not None
            else None
        ),
    )


def _outbox_event_payload(record: OutboxEventRecord) -> dict[str, Any]:
    return {
        "event_id": record.event_id,
        "event_type": record.event_type,
        "payload": record.payload,
        "snapshot_id": record.snapshot_id,
        "status": record.status,
        "attempts": record.attempts,
        "max_attempts": record.max_attempts,
        "created_at": record.created_at,
        "last_attempt_at": record.last_attempt_at,
        "error_message": record.error_message,
    }
