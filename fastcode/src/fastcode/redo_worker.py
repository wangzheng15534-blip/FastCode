"""
Background redo task worker.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .main import FastCode


class RedoWorker:
    def __init__(self, fastcode: FastCode, poll_interval_seconds: int = 30) -> None:
        self.fastcode = fastcode
        self.poll_interval_seconds = max(1, int(poll_interval_seconds))
        self.logger = logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

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
        task = self.fastcode.snapshot_store.claim_redo_task()
        if not task:
            return "none"
        task_id = str(task.get("task_id") or "")
        if not task_id:
            self.logger.warning("Redo task claimed without task_id, skipping")
            return "none"
        try:
            self._dispatch_task(task)
            self.fastcode.snapshot_store.mark_redo_task_done(task_id)
            return "succeeded"
        except Exception as e:
            self.logger.exception("Redo task failed: %s", task_id)
            self.fastcode.snapshot_store.mark_redo_task_failed(
                task_id=task_id, error=str(e)
            )
            return "failed"

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.process_once_status()
                self._flush_outbox()
            except Exception:
                self.logger.exception("Redo worker loop error")
            self._stop_event.wait(self.poll_interval_seconds)

    def _dispatch_task(self, task: dict[str, Any]) -> None:
        task_type = str(task.get("task_type") or "")
        payload = task.get("payload_json")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"redo task {task.get('task_id')!r} has malformed payload_json: {exc}"
                ) from exc
        if not isinstance(payload, dict):
            payload = {}

        if task_type == "index_run_recovery":
            run_id = payload.get("run_id")
            if not run_id:
                raise RuntimeError("redo task missing run_id")
            self.fastcode.retry_index_run_recovery(run_id=str(run_id), payload=payload)
            return
        if task_type == "publish_outbox_flush":
            self._flush_outbox()
            return
        raise RuntimeError(f"unsupported redo task type: {task_type}")

    def _flush_outbox(self) -> None:
        """Flush the TerminusDB publish outbox."""
        publisher = getattr(self.fastcode, "terminus_publisher", None)
        if not publisher or not publisher.is_configured():
            self.logger.debug(
                "TerminusDB publisher not configured, skipping outbox flush"
            )
            return
        result = publisher.flush_outbox(self.fastcode.snapshot_store)
        if result["processed"] > 0:
            self.logger.info(
                "Outbox flush: processed=%d succeeded=%d failed=%d",
                result["processed"],
                result["succeeded"],
                result["failed"],
            )
