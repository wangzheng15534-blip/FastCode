"""
Tests for RedoWorker background task processing.
"""

import json
from unittest.mock import MagicMock

import pytest

from fastcode.redo_worker import RedoWorker

pytestmark = [pytest.mark.test_double]


class _FakeFastCode:
    def __init__(self) -> None:
        self.snapshot_store = MagicMock()


def test_process_once_status_returns_none_when_no_tasks_double():
    fc = _FakeFastCode()
    fc.snapshot_store.claim_redo_task.return_value = None
    worker = RedoWorker(fc)
    assert worker.process_once_status() == "none"


def test_process_once_status_succeeds_on_valid_task_double():
    fc = _FakeFastCode()
    fc.snapshot_store.claim_redo_task.return_value = {
        "task_id": "redo_abc",
        "task_type": "index_run_recovery",
        "payload_json": json.dumps({"run_id": "run1", "source": "/tmp/repo"}),
    }
    fc.retry_index_run_recovery = MagicMock(return_value={"status": "published"})
    worker = RedoWorker(fc)
    assert worker.process_once_status() == "succeeded"
    fc.snapshot_store.mark_redo_task_done.assert_called_once_with("redo_abc")


def test_process_once_status_fails_and_marks_failed_double():
    fc = _FakeFastCode()
    fc.snapshot_store.claim_redo_task.return_value = {
        "task_id": "redo_xyz",
        "task_type": "index_run_recovery",
        "payload_json": json.dumps({"run_id": "run2", "source": "/tmp/repo"}),
    }
    fc.retry_index_run_recovery = MagicMock(side_effect=RuntimeError("boom"))
    worker = RedoWorker(fc)
    assert worker.process_once_status() == "failed"
    fc.snapshot_store.mark_redo_task_failed.assert_called_once()
    call_args = fc.snapshot_store.mark_redo_task_failed.call_args
    assert call_args[1]["task_id"] == "redo_xyz"
    assert "boom" in call_args[1]["error"]


def test_dispatch_task_raises_on_missing_run_id_double():
    fc = _FakeFastCode()
    worker = RedoWorker(fc)
    task = {
        "task_id": "redo_bad",
        "task_type": "index_run_recovery",
        "payload_json": "{}",
    }
    with pytest.raises(RuntimeError, match="missing run_id"):
        worker._dispatch_task(task)


def test_dispatch_task_raises_on_unsupported_type_double():
    fc = _FakeFastCode()
    worker = RedoWorker(fc)
    task = {"task_id": "redo_bad", "task_type": "unknown_type", "payload_json": "{}"}
    with pytest.raises(RuntimeError, match="unsupported redo task type"):
        worker._dispatch_task(task)


def test_stop_sets_event_and_joins_double():
    fc = _FakeFastCode()
    worker = RedoWorker(fc, poll_interval_seconds=1)
    worker._thread = MagicMock()
    worker.stop()
    assert worker._stop_event.is_set()
    worker._thread.join.assert_called_once_with(timeout=10)


def test_start_creates_daemon_thread_double():
    fc = _FakeFastCode()
    worker = RedoWorker(fc)
    worker.start()
    assert worker._thread is not None
    assert worker._thread.daemon is True
    assert worker._thread.name == "fastcode-redo-worker"
    worker.stop()


def test_start_idempotent_if_thread_alive_double():
    fc = _FakeFastCode()
    worker = RedoWorker(fc)
    worker.start()
    original_thread = worker._thread
    worker.start()  # should not create a second thread
    assert worker._thread is original_thread
    worker.stop()
