"""
Tests for the publish outbox pattern in TerminusPublisher + SnapshotStore.
"""

from __future__ import annotations

import json
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fastcode.redo_worker import RedoWorker
from fastcode.snapshot_store import SnapshotStore
from fastcode.terminus_publisher import TerminusPublisher

pytestmark = [pytest.mark.test_double]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_publisher(**overrides: Any) -> TerminusPublisher:
    cfg = {"terminus": {"endpoint": "http://localhost:6363/api/publish"}}
    cfg["terminus"].update(overrides)
    return TerminusPublisher(cfg)


def _minimal_snapshot() -> dict[str, Any]:
    return {
        "repo_name": "repo",
        "snapshot_id": "snap:repo:abc",
        "branch": "main",
        "commit_id": "abc123",
        "documents": [],
        "symbols": [],
    }


def _minimal_manifest() -> dict[str, Any]:
    return {
        "manifest_id": "manifest_001",
        "ref_name": "main",
        "status": "published",
        "published_at": "2026-01-01T00:00:00Z",
        "index_run_id": "run_001",
    }


def _minimal_git_meta() -> dict[str, Any]:
    return {"repo_name": "repo", "branch": "main", "commit_id": "abc123"}


# ---------------------------------------------------------------------------
# TerminusPublisher.enqueue_publish tests
# ---------------------------------------------------------------------------


class TestEnqueuePublish:
    def test_enqueue_returns_event_id_double(self):
        """enqueue_publish returns an event_id when snapshot_store accepts."""
        pub = _make_publisher()
        store = MagicMock()
        store.enqueue_outbox_event.return_value = True
        payload = {"version": "v1", "snapshot_id": "snap:repo:abc"}
        result = pub.enqueue_publish(
            snapshot_id="snap:repo:abc",
            payload=payload,
            snapshot_store=store,
        )
        assert result is not None
        assert result.startswith("outbox:snap:repo:abc:")
        store.enqueue_outbox_event.assert_called_once()
        call_kwargs = store.enqueue_outbox_event.call_args
        assert call_kwargs.kwargs.get("event_type") == "lineage_publish"
        assert call_kwargs.kwargs.get("snapshot_id") == "snap:repo:abc"

    def test_enqueue_uses_idempotency_key_when_provided_double(self):
        """enqueue_publish uses the provided idempotency_key as event_id."""
        pub = _make_publisher()
        store = MagicMock()
        store.enqueue_outbox_event.return_value = True
        payload = {"version": "v1"}
        result = pub.enqueue_publish(
            snapshot_id="snap:repo:abc",
            payload=payload,
            snapshot_store=store,
            idempotency_key="lineage:run1:snap:repo:abc",
        )
        assert result == "lineage:run1:snap:repo:abc"
        call_kwargs = store.enqueue_outbox_event.call_args.kwargs
        assert call_kwargs["event_id"] == "lineage:run1:snap:repo:abc"

    def test_enqueue_deterministic_event_id_double(self):
        """Same snapshot_id + payload always produces the same event_id."""
        pub = _make_publisher()
        store = MagicMock()
        store.enqueue_outbox_event.return_value = True
        payload = {"version": "v1", "snapshot_id": "snap:repo:abc", "data": "test"}
        result1 = pub.enqueue_publish("snap:repo:abc", payload, store)
        result2 = pub.enqueue_publish("snap:repo:abc", payload, store)
        assert result1 == result2

    def test_enqueue_different_payloads_different_ids_double(self):
        """Different payloads produce different event_ids."""
        pub = _make_publisher()
        store = MagicMock()
        store.enqueue_outbox_event.return_value = True
        result1 = pub.enqueue_publish("snap:repo:abc", {"v": 1}, store)
        result2 = pub.enqueue_publish("snap:repo:abc", {"v": 2}, store)
        assert result1 != result2

    def test_enqueue_returns_none_when_store_not_postgres_double(self):
        """enqueue_publish returns None when store rejects (non-postgres)."""
        pub = _make_publisher()
        store = MagicMock()
        store.enqueue_outbox_event.return_value = False
        payload = {"version": "v1"}
        result = pub.enqueue_publish("snap:repo:abc", payload, store)
        # Still returns event_id even when not inserted (duplicate case)
        assert result is not None


# ---------------------------------------------------------------------------
# TerminusPublisher.flush_outbox tests
# ---------------------------------------------------------------------------


class TestFlushOutbox:
    def test_flush_empty_outbox_double(self):
        """flush_outbox returns zeros when no events to process."""
        pub = _make_publisher()
        store = MagicMock()
        store.claim_outbox_event.return_value = []
        result = pub.flush_outbox(store)
        assert result == {"processed": 0, "succeeded": 0, "failed": 0}

    def test_flush_successful_event_double(self):
        """flush_outbox marks event done on successful POST."""
        pub = _make_publisher()
        store = MagicMock()
        payload = {"version": "v1", "snapshot_id": "snap:repo:abc"}
        store.claim_outbox_event.return_value = [
            {
                "event_id": "evt1",
                "payload": json.dumps(payload),
                "snapshot_id": "snap:repo:abc",
            },
        ]
        with patch.object(pub, "_do_post") as mock_post:
            result = pub.flush_outbox(store)
        assert result["processed"] == 1
        assert result["succeeded"] == 1
        assert result["failed"] == 0
        mock_post.assert_called_once_with(payload)
        store.mark_outbox_event_done.assert_called_once_with("evt1")

    def test_flush_failed_event_double(self):
        """flush_outbox marks event failed on POST error."""
        pub = _make_publisher()
        store = MagicMock()
        payload = {"version": "v1"}
        store.claim_outbox_event.return_value = [
            {
                "event_id": "evt2",
                "payload": json.dumps(payload),
                "snapshot_id": "snap:repo:abc",
            },
        ]
        with patch.object(
            pub, "_do_post", side_effect=RuntimeError("connection refused")
        ):
            result = pub.flush_outbox(store)
        assert result["processed"] == 1
        assert result["succeeded"] == 0
        assert result["failed"] == 1
        store.mark_outbox_event_failed.assert_called_once()
        call_args = store.mark_outbox_event_failed.call_args
        assert call_args[0][0] == "evt2"
        assert "connection refused" in call_args[0][1]

    def test_flush_malformed_payload_double(self):
        """flush_outbox marks event failed when payload is not valid JSON."""
        pub = _make_publisher()
        store = MagicMock()
        store.claim_outbox_event.return_value = [
            {
                "event_id": "evt3",
                "payload": "not-json{{{",
                "snapshot_id": "snap:repo:abc",
            },
        ]
        result = pub.flush_outbox(store)
        assert result["processed"] == 1
        assert result["succeeded"] == 0
        assert result["failed"] == 1
        store.mark_outbox_event_failed.assert_called_once_with(
            "evt3", "malformed payload JSON"
        )

    def test_flush_multiple_events_double(self):
        """flush_outbox processes up to limit events."""
        pub = _make_publisher()
        store = MagicMock()
        events = [
            {
                "event_id": f"evt{i}",
                "payload": json.dumps({"v": i}),
                "snapshot_id": "snap:repo:abc",
            }
            for i in range(3)
        ]
        store.claim_outbox_event.return_value = events
        with patch.object(pub, "_do_post"):
            result = pub.flush_outbox(store, limit=10)
        assert result["processed"] == 3
        assert result["succeeded"] == 3

    def test_flush_respects_limit_double(self):
        """flush_outbox passes limit to claim_outbox_event."""
        pub = _make_publisher()
        store = MagicMock()
        store.claim_outbox_event.return_value = []
        pub.flush_outbox(store, limit=5)
        store.claim_outbox_event.assert_called_once_with(limit=5)


# ---------------------------------------------------------------------------
# TerminusPublisher.get_pending_count tests
# ---------------------------------------------------------------------------


class TestGetPendingCount:
    def test_delegates_to_store_double(self):
        """get_pending_count delegates to snapshot_store."""
        pub = _make_publisher()
        store = MagicMock()
        store.get_outbox_pending_count.return_value = 7
        assert pub.get_pending_count(store) == 7
        store.get_outbox_pending_count.assert_called_once()


# ---------------------------------------------------------------------------
# TerminusPublisher.publish_snapshot_lineage unchanged behavior
# ---------------------------------------------------------------------------


class TestPublishSnapshotLineageUnchanged:
    def test_publish_still_raises_without_endpoint_double(self):
        """publish_snapshot_lineage still raises when endpoint not configured."""
        pub = TerminusPublisher({"terminus": {}})
        with pytest.raises(RuntimeError, match="not configured"):
            pub.publish_snapshot_lineage(
                snapshot=_minimal_snapshot(),
                manifest=_minimal_manifest(),
                git_meta=_minimal_git_meta(),
            )

    def test_publish_still_does_direct_post_double(self):
        """publish_snapshot_lineage still calls _do_post directly."""
        pub = _make_publisher()
        with patch.object(pub, "_do_post") as mock_post:
            pub.publish_snapshot_lineage(
                snapshot=_minimal_snapshot(),
                manifest=_minimal_manifest(),
                git_meta=_minimal_git_meta(),
                idempotency_key="key-123",
            )
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs.get("idempotency_key") == "key-123"


# ---------------------------------------------------------------------------
# RedoWorker outbox flush integration
# ---------------------------------------------------------------------------


class TestRedoWorkerOutboxFlush:
    def test_dispatch_publish_outbox_flush_double(self):
        """RedoWorker dispatches publish_outbox_flush task type."""
        fc = MagicMock()
        publisher = MagicMock()
        publisher.is_configured.return_value = True
        publisher.flush_outbox.return_value = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
        }
        fc.terminus_publisher = publisher
        fc.snapshot_store = MagicMock()
        fc.snapshot_store.claim_outbox_event.return_value = []
        worker = RedoWorker(fc)
        task = {
            "task_id": "redo_flush1",
            "task_type": "publish_outbox_flush",
            "payload_json": "{}",
        }
        worker._dispatch_task(task)  # should not raise
        publisher.flush_outbox.assert_called_once_with(fc.snapshot_store)

    def test_dispatch_publish_outbox_flush_not_configured_double(self):
        """RedoWorker skips outbox flush when publisher not configured."""
        fc = MagicMock()
        fc.terminus_publisher = TerminusPublisher({"terminus": {}})
        worker = RedoWorker(fc)
        task = {
            "task_id": "redo_flush2",
            "task_type": "publish_outbox_flush",
            "payload_json": "{}",
        }
        worker._dispatch_task(task)  # should not raise
        fc.snapshot_store.claim_outbox_event.assert_not_called()

    def test_flush_outbox_no_publisher_double(self):
        """_flush_outbox is safe when terminus_publisher attribute is missing."""
        fc = MagicMock(spec=[])
        worker = RedoWorker(fc)
        worker._flush_outbox()  # should not raise

    def test_run_loop_includes_outbox_flush_double(self):
        """_run_loop calls _flush_outbox after process_once_status."""
        fc = MagicMock()
        worker = RedoWorker(fc, poll_interval_seconds=60)
        flush_mock = MagicMock()
        with (
            patch.object(worker, "process_once_status"),
            patch.object(worker, "_flush_outbox", flush_mock),
            patch.object(worker._stop_event, "wait"),
            patch.object(worker._stop_event, "is_set", side_effect=[False, True]),
        ):
            worker._run_loop()
        flush_mock.assert_called_once()


# ---------------------------------------------------------------------------
# SnapshotStore outbox methods (unit tests with mocked DB)
# ---------------------------------------------------------------------------


class TestSnapshotStoreOutbox:
    def test_enqueue_outbox_event_non_postgres_double(self):
        """enqueue_outbox_event returns False for non-postgres."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            assert store.db_runtime.backend == "sqlite"
            result = store.enqueue_outbox_event(
                "evt1", "lineage_publish", "{}", "snap:1"
            )
            assert result is False

    def test_claim_outbox_event_non_postgres_double(self):
        """claim_outbox_event returns empty list for non-postgres."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            assert store.claim_outbox_event() == []

    def test_mark_outbox_event_done_non_postgres_double(self):
        """mark_outbox_event_done is no-op for non-postgres."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            store.mark_outbox_event_done("evt1")  # should not raise

    def test_mark_outbox_event_failed_non_postgres_double(self):
        """mark_outbox_event_failed is no-op for non-postgres."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            store.mark_outbox_event_failed("evt1", "error")  # should not raise

    def test_get_outbox_pending_count_non_postgres_double(self):
        """get_outbox_pending_count returns 0 for non-postgres."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            assert store.get_outbox_pending_count() == 0

    def test_deterministic_event_id_stability_double(self):
        """_deterministic_event_id is stable for same inputs."""
        pub = _make_publisher()
        id1 = pub._deterministic_event_id("snap:repo:abc", '{"v":1}')
        id2 = pub._deterministic_event_id("snap:repo:abc", '{"v":1}')
        assert id1 == id2
        # Different payload -> different id
        id3 = pub._deterministic_event_id("snap:repo:abc", '{"v":2}')
        assert id1 != id3
        # Different snapshot -> different id
        id4 = pub._deterministic_event_id("snap:repo:def", '{"v":1}')
        assert id1 != id4
