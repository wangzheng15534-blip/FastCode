from __future__ import annotations

from pathlib import Path

import pytest

from fastcode.store.index_run import IndexRunStore
from fastcode.store.index_run_contracts import IndexRunRecord, PublishTaskRecord
from fastcode.store.infrastructure.runtime import DBRuntime


def _make_store(tmp_path: Path) -> IndexRunStore:
    return IndexRunStore(
        DBRuntime(backend="sqlite", sqlite_path=str(tmp_path / "index_runs.db"))
    )


def test_get_run_avoids_generic_row_to_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _make_store(tmp_path)
    run_id = store.create_run(
        repo_name="repo",
        snapshot_id="snap1",
        branch="main",
        commit_id="abc123",
    )

    def _boom(_: object) -> dict[str, object]:
        raise AssertionError("index run store must not call row_to_dict()")

    monkeypatch.setattr(store.db_runtime, "row_to_dict", _boom, raising=False)

    run = store.get_run(run_id)

    assert run is not None
    assert run["run_id"] == run_id
    assert run["status"] == "queued"
    assert run["snapshot_id"] == "snap1"


def test_get_run_record_returns_typed_record(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    run_id = store.create_run(
        repo_name="repo",
        snapshot_id="snap1",
        branch="main",
        commit_id="abc123",
    )

    record = store.get_run_record(run_id)

    assert isinstance(record, IndexRunRecord)
    assert record.run_id == run_id
    assert record.status == "queued"
    assert record.snapshot_id == "snap1"


def test_get_latest_run_record_returns_newest_typed_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _make_store(tmp_path)
    first_run_id = store.create_run(
        repo_name="repo",
        snapshot_id="snap:first",
        branch="main",
        commit_id="abc123",
    )
    latest_run_id = store.create_run(
        repo_name="repo",
        snapshot_id="snap:latest",
        branch="main",
        commit_id="def456",
    )
    with store.db_runtime.connect() as conn:
        conn.execute(
            "UPDATE index_runs SET created_at=? WHERE run_id=?",
            ("2026-05-20T00:00:00+00:00", first_run_id),
        )
        conn.execute(
            "UPDATE index_runs SET created_at=? WHERE run_id=?",
            ("2026-05-20T00:00:01+00:00", latest_run_id),
        )
        conn.commit()

    def _boom(_: object) -> dict[str, object]:
        raise AssertionError("latest run lookup must not call row_to_dict()")

    monkeypatch.setattr(store.db_runtime, "row_to_dict", _boom, raising=False)

    record = store.get_latest_run_record()
    payload = store.get_latest_run()

    assert isinstance(record, IndexRunRecord)
    assert record.run_id == latest_run_id
    assert record.snapshot_id == "snap:latest"
    assert payload is not None
    assert payload["run_id"] == latest_run_id
    assert payload["snapshot_id"] == "snap:latest"


def test_claim_next_publish_task_returns_running_payload_after_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _make_store(tmp_path)
    run_id = store.create_run(
        repo_name="repo",
        snapshot_id="snap1",
        branch="main",
        commit_id="abc123",
    )
    task_id = store.enqueue_publish_retry(
        run_id=run_id,
        snapshot_id="snap1",
        manifest_id=None,
        error_message="publish failed",
    )

    def _boom(_: object) -> dict[str, object]:
        raise AssertionError("index run store must not call row_to_dict()")

    monkeypatch.setattr(store.db_runtime, "row_to_dict", _boom, raising=False)

    task = store.claim_next_publish_task()

    assert task is not None
    assert task["task_id"] == task_id
    assert task["run_id"] == run_id
    assert task["status"] == "running"
    assert task["attempts"] == 1
    assert task["last_error"] == "publish failed"
    assert task["updated_at"] is not None
    assert store.claim_next_publish_task() is None

    with store.db_runtime.connect() as conn:
        row = conn.execute(
            "SELECT status, attempts FROM publish_tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()

    assert row is not None
    assert row["status"] == "running"
    assert row["attempts"] == 1


def test_claim_next_publish_task_record_returns_running_record_after_claim(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    run_id = store.create_run(
        repo_name="repo",
        snapshot_id="snap1",
        branch="main",
        commit_id="abc123",
    )
    task_id = store.enqueue_publish_retry(
        run_id=run_id,
        snapshot_id="snap1",
        manifest_id=None,
        error_message="publish failed",
    )

    task = store.claim_next_publish_task_record()

    assert isinstance(task, PublishTaskRecord)
    assert task.task_id == task_id
    assert task.run_id == run_id
    assert task.status == "running"
    assert task.attempts == 1
    assert task.last_error == "publish failed"


def test_index_run_payload_helpers_do_not_call_record_to_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _make_store(tmp_path)
    run_id = store.create_run(
        repo_name="repo",
        snapshot_id="snap1",
        branch="main",
        commit_id="abc123",
    )
    store.enqueue_publish_retry(
        run_id=run_id,
        snapshot_id="snap1",
        manifest_id=None,
        error_message="publish failed",
    )

    def _boom_run(_: IndexRunRecord) -> dict[str, object]:
        raise AssertionError(
            "index run payload helper must not call IndexRunRecord.to_dict()"
        )

    def _boom_task(_: PublishTaskRecord) -> dict[str, object]:
        raise AssertionError(
            "publish task payload helper must not call PublishTaskRecord.to_dict()"
        )

    def _boom_from_dict(
        cls: type[IndexRunRecord] | type[PublishTaskRecord],
        _: dict[str, object],
    ) -> IndexRunRecord | PublishTaskRecord:
        raise AssertionError(
            f"index run store must not call {cls.__name__}.from_dict()"
        )

    monkeypatch.setattr(IndexRunRecord, "to_dict", _boom_run)
    monkeypatch.setattr(PublishTaskRecord, "to_dict", _boom_task)
    monkeypatch.setattr(IndexRunRecord, "from_dict", classmethod(_boom_from_dict))
    monkeypatch.setattr(PublishTaskRecord, "from_dict", classmethod(_boom_from_dict))

    run = store.get_run(run_id)
    task = store.claim_next_publish_task()

    assert run is not None
    assert run["run_id"] == run_id
    assert task is not None
    assert task["run_id"] == run_id
    assert task["status"] == "running"
