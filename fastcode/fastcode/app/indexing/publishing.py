"""
PublishingService — TerminusDB publish, retry, and redo task processing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastcode.ports.jobs import IndexRunStore as IndexRunStorePort
from fastcode.ports.publishing import LineagePublisher

if TYPE_CHECKING:
    from fastcode.app.indexing.pipeline.redo_worker import RedoWorker
    from fastcode.app.store.snapshots.manifest import ManifestStore
    from fastcode.app.store.snapshots.snapshot import SnapshotStore
    from fastcode.ports.jobs import IndexRunView


class PublishingService:
    """Handles TerminusDB publishing, retry, and redo task processing."""

    _MANIFEST_FIELDS = (
        "manifest_id",
        "repo_name",
        "ref_name",
        "snapshot_id",
        "index_run_id",
        "published_at",
        "previous_manifest_id",
        "status",
    )

    def __init__(
        self,
        *,
        config: dict[str, Any],
        logger: logging.Logger,
        index_run_store: IndexRunStorePort,
        manifest_store: ManifestStore,
        snapshot_store: SnapshotStore,
        terminus_publisher: LineagePublisher,
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
        run = self.index_run_store.get_run_record(run_id)
        return self._run_payload(run) if run is not None else None

    @classmethod
    def _manifest_payload(cls, manifest: Any) -> dict[str, Any]:
        return {
            field_name: getattr(manifest, field_name, None)
            for field_name in cls._MANIFEST_FIELDS
        }

    @staticmethod
    def _run_payload(run: IndexRunView) -> dict[str, Any]:
        return {
            "run_id": run.run_id,
            "repo_name": run.repo_name,
            "snapshot_id": run.snapshot_id,
            "branch": run.branch,
            "commit_id": run.commit_id,
            "idempotency_key": run.idempotency_key,
            "status": run.status,
            "error_message": run.error_message,
            "warnings_json": run.warnings_json,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
        }

    def publish_index_run(
        self, run_id: str, ref_name: str | None = None
    ) -> dict[str, Any]:
        run = self.index_run_store.get_run_record(run_id)
        if not run:
            msg = f"index run not found: {run_id}"
            raise RuntimeError(msg)
        snapshot_id = run.snapshot_id
        repo_name = run.repo_name
        branch = run.branch
        commit_id = run.commit_id
        snapshot = self.snapshot_store.load_snapshot(snapshot_id)
        if not snapshot:
            msg = f"snapshot not found for run: {run_id}"
            raise RuntimeError(msg)

        manifest = self.manifest_store.publish_record(
            repo_name=repo_name,
            ref_name=ref_name or branch or "HEAD",
            snapshot_id=snapshot_id,
            index_run_id=run_id,
            status="published",
        )
        status = "published"
        if self.terminus_publisher.is_configured():
            try:
                git_meta = self._build_git_meta(
                    {
                        "repo_name": repo_name,
                        "branch": branch,
                        "commit_id": commit_id,
                    }
                )
                branch_name = manifest.ref_name or branch or "HEAD"
                previous_snapshot_symbols = self._previous_snapshot_symbol_versions(
                    repo_name, branch_name, snapshot_id
                )
                self.terminus_publisher.publish_snapshot_lineage_for_snapshot(
                    snapshot=snapshot,
                    manifest=manifest,
                    git_meta=git_meta,
                    previous_snapshot_symbols=previous_snapshot_symbols,
                    idempotency_key=f"lineage:{run_id}:{snapshot_id}",
                )
            except Exception as e:
                self.index_run_store.enqueue_publish_retry(
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    manifest_id=manifest.manifest_id,
                    error_message=str(e),
                )
                status = "publish_pending"
        self.index_run_store.mark_completed(run_id, status=status)
        return {
            "status": status,
            "manifest": self._manifest_payload(manifest),
            "run_id": run_id,
        }

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
            task = self.index_run_store.claim_next_publish_task_record()
            if not task:
                break
            processed += 1
            task_id = task.task_id
            run_id = task.run_id
            try:
                run = self.index_run_store.get_run_record(run_id)
                if not run:
                    msg = f"run not found: {run_id}"
                    raise RuntimeError(msg)
                snapshot_id = run.snapshot_id
                repo_name = run.repo_name
                branch = run.branch
                commit_id = run.commit_id
                snapshot = self.snapshot_store.load_snapshot(snapshot_id)
                if not snapshot:
                    msg = f"snapshot not found: {snapshot_id}"
                    raise RuntimeError(msg)

                ref_name = branch or "HEAD"
                manifest = self.manifest_store.get_branch_manifest_record(
                    repo_name, ref_name
                )
                if not manifest:
                    manifest = self.manifest_store.publish_record(
                        repo_name=repo_name,
                        ref_name=ref_name,
                        snapshot_id=snapshot_id,
                        index_run_id=run_id,
                        status="published",
                    )

                git_meta = self._build_git_meta(
                    {
                        "repo_name": repo_name,
                        "branch": branch,
                        "commit_id": commit_id,
                    }
                )
                previous_snapshot_symbols = self._previous_snapshot_symbol_versions(
                    repo_name, ref_name, snapshot_id
                )
                self.terminus_publisher.publish_snapshot_lineage_for_snapshot(
                    snapshot=snapshot,
                    manifest=manifest,
                    git_meta=git_meta,
                    previous_snapshot_symbols=previous_snapshot_symbols,
                    idempotency_key=f"lineage:{run_id}:{snapshot_id}",
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
            msg = f"redo recovery payload missing source for run {run_id}"
            raise RuntimeError(msg)
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
