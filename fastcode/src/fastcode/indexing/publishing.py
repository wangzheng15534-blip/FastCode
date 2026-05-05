"""
PublishingService — TerminusDB publish, retry, and redo task processing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..store.index_run import IndexRunStore
    from ..store.manifest import ManifestStore
    from ..store.snapshot import SnapshotStore
    from .redo_worker import RedoWorker
    from .terminus import TerminusPublisher


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

    def _get_run_record_or_payload(self, run_id: str) -> Any:
        get_record = getattr(self.index_run_store, "get_run_record", None)
        if callable(get_record):
            record = get_record(run_id)
            if record is not None:
                return record
        return self.index_run_store.get_run(run_id)

    def _claim_publish_task_record_or_payload(self) -> Any:
        claim_record = getattr(
            self.index_run_store, "claim_next_publish_task_record", None
        )
        if callable(claim_record):
            record = claim_record()
            if record is not None:
                return record
        return self.index_run_store.claim_next_publish_task()

    @staticmethod
    def _field(payload: Any, name: str) -> Any:
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload).get(name)
        return getattr(payload, name, None)

    @classmethod
    def _manifest_payload(cls, manifest: Any) -> dict[str, Any]:
        return {
            field_name: cls._field(manifest, field_name)
            for field_name in cls._MANIFEST_FIELDS
        }

    def publish_index_run(
        self, run_id: str, ref_name: str | None = None
    ) -> dict[str, Any]:
        run = self._get_run_record_or_payload(run_id)
        if not run:
            raise RuntimeError(f"index run not found: {run_id}")
        snapshot_id = str(self._field(run, "snapshot_id") or "")
        repo_name = str(self._field(run, "repo_name") or "")
        branch = self._field(run, "branch")
        commit_id = self._field(run, "commit_id")
        snapshot = self.snapshot_store.load_snapshot(snapshot_id)
        if not snapshot:
            raise RuntimeError(f"snapshot not found for run: {run_id}")

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
            task = self._claim_publish_task_record_or_payload()
            if not task:
                break
            processed += 1
            task_id = str(self._field(task, "task_id") or "")
            run_id = str(self._field(task, "run_id") or "")
            try:
                run = self._get_run_record_or_payload(run_id)
                if not run:
                    raise RuntimeError(f"run not found: {run_id}")
                snapshot_id = str(self._field(run, "snapshot_id") or "")
                repo_name = str(self._field(run, "repo_name") or "")
                branch = self._field(run, "branch")
                commit_id = self._field(run, "commit_id")
                snapshot = self.snapshot_store.load_snapshot(snapshot_id)
                if not snapshot:
                    raise RuntimeError(f"snapshot not found: {snapshot_id}")

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
