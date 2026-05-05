"""Tests for typed publishing service boundaries."""

from __future__ import annotations

import logging
from typing import Any, cast

import pytest

from fastcode.indexing.publishing import PublishingService
from fastcode.ir.types import IRSnapshot
from fastcode.store.records import IndexRunRecord


class NoDictSnapshot(IRSnapshot):
    def to_dict(self) -> dict[str, Any]:
        raise AssertionError("PublishingService must publish typed IRSnapshot objects")


class ManifestDouble:
    manifest_id = "manifest1"
    repo_name = "repo"
    ref_name = "main"
    snapshot_id = "snap1"
    index_run_id = "run1"
    published_at = "2026-01-01T00:00:00Z"
    previous_manifest_id: str | None = None
    status = "published"

    def to_dict(self) -> dict[str, Any]:
        raise AssertionError("PublishingService must not call manifest.to_dict()")


class IndexRunStoreDouble:
    def __init__(self) -> None:
        self.completed: tuple[str, str] | None = None
        self.retries: list[dict[str, Any]] = []

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        if run_id != "run1":
            return None
        return {
            "run_id": "run1",
            "repo_name": "repo",
            "branch": "main",
            "commit_id": "abc",
            "snapshot_id": "snap1",
        }

    def get_run_record(self, run_id: str) -> IndexRunRecord | None:
        if run_id != "run1":
            return None
        return IndexRunRecord(
            run_id="run1",
            repo_name="repo",
            snapshot_id="snap1",
            branch="main",
            commit_id="abc",
            idempotency_key=None,
            status="queued",
            error_message=None,
            warnings_json=None,
            created_at="2026-01-01T00:00:00Z",
            started_at=None,
            completed_at=None,
        )

    def enqueue_publish_retry(self, **kwargs: Any) -> None:
        self.retries.append(dict(kwargs))

    def mark_completed(self, run_id: str, status: str) -> None:
        self.completed = (run_id, status)


class ManifestStoreDouble:
    def __init__(self, manifest: ManifestDouble) -> None:
        self.manifest = manifest

    def publish_record(
        self,
        *,
        repo_name: str,
        ref_name: str,
        snapshot_id: str,
        index_run_id: str,
        status: str,
    ) -> ManifestDouble:
        assert (repo_name, ref_name, snapshot_id, index_run_id, status) == (
            "repo",
            "main",
            "snap1",
            "run1",
            "published",
        )
        return self.manifest


class SnapshotStoreDouble:
    def __init__(self, snapshot: IRSnapshot) -> None:
        self.snapshot = snapshot

    def load_snapshot(self, snapshot_id: str) -> IRSnapshot | None:
        if snapshot_id == "snap1":
            return self.snapshot
        return None


class TerminusPublisherDouble:
    def __init__(self) -> None:
        self.published: dict[str, Any] | None = None

    def is_configured(self) -> bool:
        return True

    def publish_snapshot_lineage(self, **_: Any) -> None:
        raise AssertionError("PublishingService used dict-based lineage publishing")

    def publish_snapshot_lineage_for_snapshot(
        self,
        *,
        snapshot: IRSnapshot,
        manifest: ManifestDouble,
        git_meta: dict[str, Any],
        previous_snapshot_symbols: dict[str, str] | None,
        idempotency_key: str | None,
    ) -> None:
        self.published = {
            "snapshot": snapshot,
            "manifest": manifest,
            "git_meta": git_meta,
            "previous_snapshot_symbols": previous_snapshot_symbols,
            "idempotency_key": idempotency_key,
        }


def test_publish_index_run_uses_typed_lineage_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = NoDictSnapshot(
        repo_name="repo",
        snapshot_id="snap1",
        branch="main",
        commit_id="abc",
    )
    manifest = ManifestDouble()
    index_run_store = IndexRunStoreDouble()
    publisher = TerminusPublisherDouble()
    monkeypatch.setattr(
        index_run_store,
        "get_run",
        lambda _run_id: (_ for _ in ()).throw(
            AssertionError("PublishingService should prefer get_run_record()")
        ),
    )
    service = PublishingService(
        config={},
        logger=logging.getLogger(__name__),
        index_run_store=cast(Any, index_run_store),
        manifest_store=cast(Any, ManifestStoreDouble(manifest)),
        snapshot_store=cast(Any, SnapshotStoreDouble(snapshot)),
        terminus_publisher=cast(Any, publisher),
        redo_worker=None,
        build_git_meta=lambda _: {
            "repo_name": "repo",
            "branch": "main",
            "commit_id": "abc",
        },
        previous_snapshot_symbol_versions=lambda *_: None,
        run_index_pipeline_cb=lambda **_: {},
    )

    result = service.publish_index_run("run1")

    assert result["status"] == "published"
    assert result["manifest"]["manifest_id"] == "manifest1"
    assert result["manifest"]["ref_name"] == "main"
    assert index_run_store.completed == ("run1", "published")
    assert publisher.published is not None
    assert publisher.published["snapshot"] is snapshot
    assert publisher.published["manifest"] is manifest
    assert publisher.published["idempotency_key"] == "lineage:run1:snap1"
