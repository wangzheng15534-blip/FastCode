"""
PublishingService — TerminusDB publish, retry, and redo task processing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .index_run import IndexRunStore
    from .manifest_store import ManifestStore
    from .redo_worker import RedoWorker
    from .snapshot_store import SnapshotStore
    from .terminus_publisher import TerminusPublisher


class PublishingService:
    """Handles TerminusDB publishing, retry, and redo task processing."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        logger: logging.Logger,
        index_run_store: IndexRunStore,
        manifest_store: ManifestStore,
        snapshot_store: SnapshotStore,
        terminus_publisher: TerminusPublisher,
        redo_worker: RedoWorker | None,
        build_git_meta: Callable[[dict[str, Any]], dict[str, Any]],
        previous_snapshot_symbol_versions: Callable[
            [str, str, str], dict[str, str] | None
        ],
        run_index_pipeline_cb: Callable[..., dict[str, Any]],
    ) -> None:
        self.config = config
        self.logger = logger
        self.index_run_store = index_run_store
        self.manifest_store = manifest_store
        self.snapshot_store = snapshot_store
        self.terminus_publisher = terminus_publisher
        self._redo_worker = redo_worker
        self._build_git_meta = build_git_meta
        self._previous_snapshot_symbol_versions = previous_snapshot_symbol_versions
        self._run_index_pipeline = run_index_pipeline_cb

    def get_index_run(self, run_id: str) -> dict[str, Any] | None:
        return self.index_run_store.get_run(run_id)

    def publish_index_run(
        self, run_id: str, ref_name: str | None = None
    ) -> dict[str, Any]:
        run = self.index_run_store.get_run(run_id)
        if not run:
            raise RuntimeError(f"index run not found: {run_id}")
        snapshot = self.snapshot_store.load_snapshot(run["snapshot_id"])
        if not snapshot:
            raise RuntimeError(f"snapshot not found for run: {run_id}")

        manifest = self.manifest_store.publish_record(
            repo_name=run["repo_name"],
            ref_name=ref_name or run.get("branch") or "HEAD",
            snapshot_id=run["snapshot_id"],
            index_run_id=run_id,
            status="published",
        )
        status = "published"
        if self.terminus_publisher.is_configured():
            try:
                git_meta = self._build_git_meta(
                    {
                        "repo_name": run["repo_name"],
                        "branch": run.get("branch"),
                        "commit_id": run.get("commit_id"),
                    }
                )
                branch_name = manifest.ref_name or run.get("branch") or "HEAD"
                previous_snapshot_symbols = self._previous_snapshot_symbol_versions(
                    run["repo_name"], branch_name, run["snapshot_id"]
                )
                self.terminus_publisher.publish_snapshot_lineage(
                    snapshot=snapshot.to_dict(),
                    manifest=manifest.to_dict(),
                    git_meta=git_meta,
                    previous_snapshot_symbols=previous_snapshot_symbols,
                    idempotency_key=f"lineage:{run_id}:{run['snapshot_id']}",
                )
            except Exception as e:
                self.index_run_store.enqueue_publish_retry(
                    run_id=run_id,
                    snapshot_id=run["snapshot_id"],
                    manifest_id=manifest.manifest_id,
                    error_message=str(e),
                )
                status = "publish_pending"
        self.index_run_store.mark_completed(run_id, status=status)
        return {"status": status, "manifest": manifest.to_dict(), "run_id": run_id}

    def retry_pending_publishes(self, limit: int = 10) -> dict[str, Any]:
        if not self.terminus_publisher.is_configured():
            return {
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "message": "terminus_not_configured",
            }

        processed = 0
        succeeded = 0
        failed = 0

        while processed < limit:
            task = self.index_run_store.claim_next_publish_task()
            if not task:
                break
            processed += 1
            task_id = task["task_id"]
            run_id = task["run_id"]
            try:
                run = self.index_run_store.get_run(run_id)
                if not run:
                    raise RuntimeError(f"run not found: {run_id}")
                snapshot = self.snapshot_store.load_snapshot(run["snapshot_id"])
                if not snapshot:
                    raise RuntimeError(f"snapshot not found: {run['snapshot_id']}")

                ref_name = run.get("branch") or "HEAD"
                manifest = self.manifest_store.get_branch_manifest_record(
                    run["repo_name"], ref_name
                )
                if not manifest:
                    manifest = self.manifest_store.publish_record(
                        repo_name=run["repo_name"],
                        ref_name=ref_name,
                        snapshot_id=run["snapshot_id"],
                        index_run_id=run_id,
                        status="published",
                    )

                git_meta = self._build_git_meta(
                    {
                        "repo_name": run["repo_name"],
                        "branch": run.get("branch"),
                        "commit_id": run.get("commit_id"),
                    }
                )
                previous_snapshot_symbols = self._previous_snapshot_symbol_versions(
                    run["repo_name"], ref_name, run["snapshot_id"]
                )
                self.terminus_publisher.publish_snapshot_lineage(
                    snapshot=snapshot.to_dict(),
                    manifest=manifest.to_dict(),
                    git_meta=git_meta,
                    previous_snapshot_symbols=previous_snapshot_symbols,
                    idempotency_key=f"lineage:{run_id}:{run['snapshot_id']}",
                )
                self.index_run_store.mark_publish_task_done(task_id)
                self.index_run_store.mark_completed(run_id, status="published")
                succeeded += 1
            except Exception as e:
                self.index_run_store.mark_publish_task_failed(task_id, str(e))
                failed += 1

        return {
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed,
        }

    def retry_index_run_recovery(
        self, run_id: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = payload or {}
        source = payload.get("source")
        if not source:
            raise RuntimeError(f"redo recovery payload missing source for run {run_id}")
        return self._run_index_pipeline(
            source=source,
            is_url=payload.get("is_url"),
            ref=payload.get("ref"),
            commit=payload.get("commit"),
            force=True,
            publish=bool(payload.get("publish", True)),
            scip_artifact_path=payload.get("scip_artifact_path"),
            enable_scip=bool(payload.get("enable_scip", True)),
        )

    def process_redo_tasks(self, limit: int = 10) -> dict[str, Any]:
        if not self._redo_worker:
            return {
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "message": "redo_worker_disabled",
            }
        processed = 0
        succeeded = 0
        failed = 0
        while processed < max(1, min(int(limit), 100)):
            status = self._redo_worker.process_once_status()
            if status == "none":
                break
            processed += 1
            if status == "succeeded":
                succeeded += 1
            elif status == "failed":
                failed += 1
        return {"processed": processed, "succeeded": succeeded, "failed": failed}
