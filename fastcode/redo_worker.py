"""
Background redo task worker.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from .main import FastCode


class RedoWorker:
    def __init__(self, fastcode: "FastCode", poll_interval_seconds: int = 30):
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
        try:
            self._dispatch_task(task)
            self.fastcode.snapshot_store.mark_redo_task_done(task_id)
            return "succeeded"
        except Exception as e:
            self.logger.exception("Redo task failed: %s", task_id)
            self.fastcode.snapshot_store.mark_redo_task_failed(task_id=task_id, error=str(e))
            return "failed"

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.process_once_status()
            except Exception:
                self.logger.exception("Redo worker loop error")
            self._stop_event.wait(self.poll_interval_seconds)

    def _dispatch_task(self, task: Dict[str, Any]) -> None:
        task_type = str(task.get("task_type") or "")
        payload = task.get("payload_json")
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)
        if not isinstance(payload, dict):
            payload = {}

        if task_type == "index_run_recovery":
            run_id = payload.get("run_id")
            if not run_id:
                raise RuntimeError("redo task missing run_id")
            self.fastcode.retry_index_run_recovery(run_id=str(run_id), payload=payload)
            return
        raise RuntimeError(f"unsupported redo task type: {task_type}")
