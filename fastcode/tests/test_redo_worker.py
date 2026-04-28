"""Tests for redo_worker module."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.redo_worker import RedoWorker

pytestmark = [pytest.mark.test_double]

# --- Helpers ---


class _FakeSnapshotStore:
    """Fake snapshot store for testing RedoWorker."""

    def __init__(self, tasks: Any = None) -> None:
        self._tasks = list(tasks or [])
        self.done_ids = []
        self.failed_ids = []

    def claim_redo_task(self) -> Any:
        if self._tasks:
            return self._tasks.pop(0)
        return None

    def mark_redo_task_done(self, task_id: str) -> None:
        self.done_ids.append(task_id)

    def mark_redo_task_failed(self, task_id: str, error: Exception) -> None:
        self.failed_ids.append((task_id, error))


class _FakeFastCode:
    """Minimal FastCode fake for RedoWorker (MagicMock-based)."""

    def __init__(self) -> None:
        self.snapshot_store = MagicMock()


class _FakeFastCodeWithStore:
    """Fake FastCode with real snapshot store for property tests."""

    def __init__(self, snapshot_store: Any = None, retry_raises: Any = None) -> None:
        self.snapshot_store = snapshot_store or _FakeSnapshotStore()
        self._retry_raises = retry_raises
        self.retried_runs = []

    def retry_index_run_recovery(self, run_id: str, payload: Any) -> None:
        self.retried_runs.append(run_id)
        if self._retry_raises:
            raise self._retry_raises


def _make_worker(fastcode: Any = None, poll: int = 30) -> Any:
    if fastcode is None:
        fastcode = _FakeFastCodeWithStore()
    return RedoWorker(fastcode, poll_interval_seconds=poll)


task_st = st.dictionaries(
    st.sampled_from(["task_id", "task_type", "payload_json"]),
    st.one_of(st.text(min_size=1, max_size=10), st.none()),
    max_size=3,
)


# --- MagicMock-based tests ---


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


# --- Property-based tests ---


class TestRedoWorkerInit:
    def test_poll_interval_minimum_property(self):
        """HAPPY: poll interval clamped to minimum 1."""
        w = _make_worker(poll=0)
        assert w.poll_interval_seconds == 1

    @pytest.mark.edge
    def test_poll_interval_negative_clamped_property(self):
        """EDGE: negative poll interval clamped to 1."""
        w = _make_worker(poll=-10)
        assert w.poll_interval_seconds == 1

    @given(poll=st.integers(min_value=1, max_value=300))
    @settings(max_examples=10)
    def test_poll_interval_preserved_property(self, poll: int):
        """HAPPY: valid poll interval preserved."""
        w = _make_worker(poll=poll)
        assert w.poll_interval_seconds == poll


class TestProcessOnce:
    def test_no_task_returns_false_property(self):
        """HAPPY: no pending task returns False."""
        w = _make_worker()
        assert w.process_once() is False

    def test_successful_task_returns_true_property(self):
        """HAPPY: successful task returns True."""
        store = _FakeSnapshotStore(
            [
                {
                    "task_id": "t1",
                    "task_type": "index_run_recovery",
                    "payload_json": json.dumps({"run_id": "r1"}),
                }
            ]
        )
        fc = _FakeFastCodeWithStore(store)
        w = _make_worker(fc)
        assert w.process_once() is True
        assert "r1" in fc.retried_runs

    @pytest.mark.edge
    def test_task_without_id_returns_false_property(self):
        """EDGE: task without task_id returns False."""
        store = _FakeSnapshotStore([{"task_type": "other"}])
        fc = _FakeFastCodeWithStore(store)
        w = _make_worker(fc)
        assert w.process_once() is False

    @pytest.mark.edge
    def test_failed_task_returns_false_property(self):
        """EDGE: task that raises returns False."""
        store = _FakeSnapshotStore(
            [
                {
                    "task_id": "t1",
                    "task_type": "index_run_recovery",
                    "payload_json": json.dumps({"run_id": "r1"}),
                }
            ]
        )
        fc = _FakeFastCodeWithStore(store, retry_raises=RuntimeError("boom"))
        w = _make_worker(fc)
        assert w.process_once() is False

    @pytest.mark.edge
    def test_malformed_payload_json_raises_property(self):
        """EDGE: malformed JSON in payload raises."""
        store = _FakeSnapshotStore(
            [
                {
                    "task_id": "t1",
                    "task_type": "index_run_recovery",
                    "payload_json": "{not valid json",
                }
            ]
        )
        fc = _FakeFastCodeWithStore(store)
        w = _make_worker(fc)
        # Should return False because exception is caught
        assert w.process_once() is False

    @pytest.mark.edge
    def test_unsupported_task_type_fails_property(self):
        """EDGE: unsupported task type causes failure."""
        store = _FakeSnapshotStore(
            [
                {
                    "task_id": "t1",
                    "task_type": "unknown_type",
                    "payload_json": "{}",
                }
            ]
        )
        fc = _FakeFastCodeWithStore(store)
        w = _make_worker(fc)
        assert w.process_once() is False

    @pytest.mark.edge
    def test_missing_run_id_fails_property(self):
        """EDGE: index_run_recovery without run_id fails."""
        store = _FakeSnapshotStore(
            [
                {
                    "task_id": "t1",
                    "task_type": "index_run_recovery",
                    "payload_json": json.dumps({}),
                }
            ]
        )
        fc = _FakeFastCodeWithStore(store)
        w = _make_worker(fc)
        assert w.process_once() is False


class TestProcessOnceStatus:
    def test_none_status_property(self):
        """HAPPY: no task returns 'none'."""
        w = _make_worker()
        assert w.process_once_status() == "none"

    def test_succeeded_status_property(self):
        """HAPPY: successful task returns 'succeeded'."""
        store = _FakeSnapshotStore(
            [
                {
                    "task_id": "t1",
                    "task_type": "index_run_recovery",
                    "payload_json": json.dumps({"run_id": "r1"}),
                }
            ]
        )
        fc = _FakeFastCodeWithStore(store)
        w = _make_worker(fc)
        assert w.process_once_status() == "succeeded"

    @pytest.mark.edge
    def test_failed_status_property(self):
        """EDGE: failed task returns 'failed'."""
        store = _FakeSnapshotStore(
            [
                {
                    "task_id": "t1",
                    "task_type": "index_run_recovery",
                    "payload_json": json.dumps({"run_id": "r1"}),
                }
            ]
        )
        fc = _FakeFastCodeWithStore(store, retry_raises=RuntimeError("fail"))
        w = _make_worker(fc)
        assert w.process_once_status() == "failed"

    @pytest.mark.edge
    def test_task_marked_done_on_success_property(self):
        """EDGE: successful task marked done in store."""
        store = _FakeSnapshotStore(
            [
                {
                    "task_id": "t1",
                    "task_type": "index_run_recovery",
                    "payload_json": json.dumps({"run_id": "r1"}),
                }
            ]
        )
        fc = _FakeFastCodeWithStore(store)
        w = _make_worker(fc)
        w.process_once_status()
        assert "t1" in store.done_ids

    @pytest.mark.edge
    def test_task_marked_failed_on_error_property(self):
        """EDGE: failed task marked failed in store."""
        store = _FakeSnapshotStore(
            [
                {
                    "task_id": "t1",
                    "task_type": "index_run_recovery",
                    "payload_json": json.dumps({"run_id": "r1"}),
                }
            ]
        )
        fc = _FakeFastCodeWithStore(store, retry_raises=RuntimeError("err"))
        w = _make_worker(fc)
        w.process_once_status()
        assert len(store.failed_ids) == 1
        assert store.failed_ids[0][0] == "t1"

    @pytest.mark.edge
    def test_dict_payload_without_json_property(self):
        """EDGE: payload as dict (not JSON string) works."""
        store = _FakeSnapshotStore(
            [
                {
                    "task_id": "t1",
                    "task_type": "index_run_recovery",
                    "payload_json": {"run_id": "r1"},
                }
            ]
        )
        fc = _FakeFastCodeWithStore(store)
        w = _make_worker(fc)
        assert w.process_once_status() == "succeeded"

    @pytest.mark.edge
    def test_none_payload_treated_as_empty_property(self):
        """EDGE: None payload treated as empty dict."""
        store = _FakeSnapshotStore(
            [
                {
                    "task_id": "t1",
                    "task_type": "index_run_recovery",
                    "payload_json": None,
                }
            ]
        )
        fc = _FakeFastCodeWithStore(store)
        w = _make_worker(fc)
        # Will fail because run_id missing from empty dict
        assert w.process_once_status() == "failed"
