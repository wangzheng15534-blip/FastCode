"""Property-based tests for redo_worker module."""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.redo_worker import RedoWorker

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
    """Minimal FastCode fake for RedoWorker."""

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
        fastcode = _FakeFastCode()
    return RedoWorker(fastcode, poll_interval_seconds=poll)


task_st = st.dictionaries(
    st.sampled_from(["task_id", "task_type", "payload_json"]),
    st.one_of(st.text(min_size=1, max_size=10), st.none()),
    max_size=3,
)


# --- Properties ---


@pytest.mark.property
class TestRedoWorkerInit:
    @pytest.mark.happy
    def test_poll_interval_minimum(self):
        """HAPPY: poll interval clamped to minimum 1."""
        w = _make_worker(poll=0)
        assert w.poll_interval_seconds == 1

    @pytest.mark.edge
    def test_poll_interval_negative_clamped(self):
        """EDGE: negative poll interval clamped to 1."""
        w = _make_worker(poll=-10)
        assert w.poll_interval_seconds == 1

    @given(poll=st.integers(min_value=1, max_value=300))
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_poll_interval_preserved(self, poll: int):
        """HAPPY: valid poll interval preserved."""
        w = _make_worker(poll=poll)
        assert w.poll_interval_seconds == poll


@pytest.mark.property
class TestProcessOnce:
    @pytest.mark.happy
    def test_no_task_returns_false(self):
        """HAPPY: no pending task returns False."""
        w = _make_worker()
        assert w.process_once() is False

    @pytest.mark.happy
    def test_successful_task_returns_true(self):
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
        fc = _FakeFastCode(store)
        w = _make_worker(fc)
        assert w.process_once() is True
        assert "r1" in fc.retried_runs

    @pytest.mark.edge
    def test_task_without_id_returns_false(self):
        """EDGE: task without task_id returns False."""
        store = _FakeSnapshotStore([{"task_type": "other"}])
        fc = _FakeFastCode(store)
        w = _make_worker(fc)
        assert w.process_once() is False

    @pytest.mark.edge
    def test_failed_task_returns_false(self):
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
        fc = _FakeFastCode(store, retry_raises=RuntimeError("boom"))
        w = _make_worker(fc)
        assert w.process_once() is False

    @pytest.mark.edge
    def test_malformed_payload_json_raises(self):
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
        fc = _FakeFastCode(store)
        w = _make_worker(fc)
        # Should return False because exception is caught
        assert w.process_once() is False

    @pytest.mark.edge
    def test_unsupported_task_type_fails(self):
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
        fc = _FakeFastCode(store)
        w = _make_worker(fc)
        assert w.process_once() is False

    @pytest.mark.edge
    def test_missing_run_id_fails(self):
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
        fc = _FakeFastCode(store)
        w = _make_worker(fc)
        assert w.process_once() is False


@pytest.mark.property
class TestProcessOnceStatus:
    @pytest.mark.happy
    def test_none_status(self):
        """HAPPY: no task returns 'none'."""
        w = _make_worker()
        assert w.process_once_status() == "none"

    @pytest.mark.happy
    def test_succeeded_status(self):
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
        fc = _FakeFastCode(store)
        w = _make_worker(fc)
        assert w.process_once_status() == "succeeded"

    @pytest.mark.edge
    def test_failed_status(self):
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
        fc = _FakeFastCode(store, retry_raises=RuntimeError("fail"))
        w = _make_worker(fc)
        assert w.process_once_status() == "failed"

    @pytest.mark.edge
    def test_task_marked_done_on_success(self):
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
        fc = _FakeFastCode(store)
        w = _make_worker(fc)
        w.process_once_status()
        assert "t1" in store.done_ids

    @pytest.mark.edge
    def test_task_marked_failed_on_error(self):
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
        fc = _FakeFastCode(store, retry_raises=RuntimeError("err"))
        w = _make_worker(fc)
        w.process_once_status()
        assert len(store.failed_ids) == 1
        assert store.failed_ids[0][0] == "t1"

    @pytest.mark.edge
    def test_dict_payload_without_json(self):
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
        fc = _FakeFastCode(store)
        w = _make_worker(fc)
        assert w.process_once_status() == "succeeded"

    @pytest.mark.edge
    def test_none_payload_treated_as_empty(self):
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
        fc = _FakeFastCode(store)
        w = _make_worker(fc)
        # Will fail because run_id missing from empty dict
        assert w.process_once_status() == "failed"
