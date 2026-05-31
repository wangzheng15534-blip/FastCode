"""
Background redo task worker.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from typing import Any, cast

from fastcode.ports.jobs import RedoJobQueue
from fastcode.ports.publishing import EventSink


class RedoWorker:
    def __init__(self, fastcode: Any, poll_interval_seconds: int = 30) -> None:
        self.fastcode = fastcode
        self.poll_interval_seconds = max(1, int(poll_interval_seconds))
        self.logger = logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _field(payload: Any, name: str) -> Any:
        if isinstance(payload, Mapping):
            return cast(Mapping[str, Any], payload).get(name)
        return getattr(payload, name, None)

    def _redo_queue(self) -> RedoJobQueue:
        return cast(RedoJobQueue, self.fastcode.snapshot_store)

    def _event_sink(self) -> EventSink:
        return cast(EventSink, self.fastcode.snapshot_store)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name="fastcode-redo-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def process_once(self) -> bool:
        return self.process_once_status() == "succeeded"

    def process_once_status(self) -> str:
        task = self._redo_queue().claim_redo_task_record()
        if not task:
            return "none"
        task_id = str(self._field(task, "task_id") or "")
        if not task_id:
            self.logger.warning("Redo task claimed without task_id, skipping")
            return "none"
        try:
            self._dispatch_task(task)
            self._redo_queue().mark_redo_task_done(task_id)
            return "succeeded"
        except Exception as e:
            self.logger.exception("Redo task failed: %s", task_id)
            self._redo_queue().mark_redo_task_failed(task_id=task_id, error=str(e))
            return "failed"

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.process_once_status()
                self._flush_outbox()
            except Exception:
                self.logger.exception("Redo worker loop error")
            self._stop_event.wait(self.poll_interval_seconds)

    def _dispatch_task(self, task: Any) -> None:
        task_type = str(self._field(task, "task_type") or "")
        payload = self._field(task, "payload_json")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    "redo task "
                    f"{self._field(task, 'task_id')!r} "
                    f"has malformed payload_json: {exc}"
                ) from exc
        if not isinstance(payload, dict):
            payload = {}

        if task_type == "index_run_recovery":
            run_id = payload.get("run_id")
            if not run_id:
                raise RuntimeError("redo task missing run_id")
            self.fastcode.publishing.retry_index_run_recovery(
                run_id=str(run_id), payload=payload
            )
            return
        if task_type == "semantic_repair_frontier":
            result = self.fastcode.publishing.process_semantic_repair_frontier(
                payload=payload
            )
            if self._projection_rebuild_enabled(payload):
                self._rebuild_dirty_projections_after_repair(payload, result)
            return
        if task_type == "projection_dirty_rebuild":
            snapshot_id = str(payload.get("snapshot_id") or "")
            if not snapshot_id:
                raise RuntimeError("projection_dirty_rebuild task missing snapshot_id")
            self._rebuild_dirty_projections(snapshot_id)
            return
        if task_type == "publish_outbox_flush":
            self._flush_outbox()
            return
        raise RuntimeError(f"unsupported redo task type: {task_type}")

    def _projection_rebuild_enabled(self, payload: dict[str, Any]) -> bool:
        if "rebuild_dirty_projections" in payload:
            return bool(payload.get("rebuild_dirty_projections"))
        config = getattr(self.fastcode, "config", {}) or {}
        projection_cfg: Any = {}
        if isinstance(config, dict):
            projection_cfg = config.get("projection", {})
        else:
            projection_cfg = getattr(config, "projection", {})
        if isinstance(projection_cfg, dict):
            return bool(projection_cfg.get("rebuild_dirty_after_redo", True))
        return bool(getattr(projection_cfg, "rebuild_dirty_after_redo", True))

    def _rebuild_dirty_projections_after_repair(
        self, payload: dict[str, Any], result: Any
    ) -> None:
        if not isinstance(result, dict):
            return
        projection_dirty = result.get("projection_dirty")
        if not isinstance(projection_dirty, dict) or not projection_dirty.get("marked"):
            return
        repair_frontier = result.get("repair_frontier", {})
        if not isinstance(repair_frontier, dict):
            repair_frontier = {}
        snapshot_id = str(
            projection_dirty.get("snapshot_id")
            or repair_frontier.get("snapshot_id")
            or payload.get("snapshot_id")
            or ""
        )
        if snapshot_id:
            self._rebuild_dirty_projections(snapshot_id)

    def _rebuild_dirty_projections(self, snapshot_id: str) -> None:
        projection_service = getattr(self.fastcode, "projection_service", None)
        rebuild = getattr(projection_service, "rebuild_dirty_projections", None)
        if not callable(rebuild):
            self.logger.debug(
                "projection service unavailable, skipping dirty rebuild for %s",
                snapshot_id,
            )
            return
        result = rebuild(snapshot_id)
        rebuilt = result.get("rebuilt", 0) if isinstance(result, dict) else 0
        if rebuilt:
            self.logger.info(
                "Rebuilt %s dirty projection(s) for snapshot %s",
                rebuilt,
                snapshot_id,
            )

    def _flush_outbox(self) -> None:
        """Flush the TerminusDB publish outbox."""
        publisher = getattr(self.fastcode, "terminus_publisher", None)
        if not publisher or not publisher.is_configured():
            self.logger.debug(
                "TerminusDB publisher not configured, skipping outbox flush"
            )
            return
        result = publisher.flush_outbox(self._event_sink())
        if result["processed"] > 0:
            self.logger.info(
                "Outbox flush: processed=%d succeeded=%d failed=%d",
                result["processed"],
                result["succeeded"],
                result["failed"],
            )
