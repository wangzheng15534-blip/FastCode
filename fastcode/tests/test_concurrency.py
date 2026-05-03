"""Concurrency regression tests for shared-state safety."""
from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from fastcode.query_handler import QueryPipeline
from fastcode.store_records import SnapshotRecord


class _NoOpContext:
    """Context manager that does nothing — used to disable the lock."""

    def __enter__(self) -> _NoOpContext:
        return self

    def __exit__(self, *_args: object) -> None:
        pass


def _build_shared_pipeline(
    fake_load: Any,
    fake_query: Any,
) -> QueryPipeline:
    """Build a single QueryPipeline with injectable load/query callbacks."""
    p = QueryPipeline.__new__(QueryPipeline)
    p.config = {"generation": {"enable_multi_turn": False}}
    p.logger = MagicMock()
    p.retriever = MagicMock()
    p.retriever.enable_agency_mode = False
    p.retriever.iterative_agent = None
    p.retriever.retrieve.return_value = [{"element": {"relative_path": "a.py"}}]
    p.query_processor = MagicMock()
    p.answer_generator = MagicMock()
    p.answer_generator.generate.return_value = {"answer": "ok", "sources": []}
    p.cache_manager = MagicMock()
    p.cache_manager.get_recent_summaries.return_value = []
    p.cache_manager.get_dialogue_history.return_value = []
    p.manifest_store = MagicMock()
    p.snapshot_store = MagicMock()
    p.snapshot_symbol_index = MagicMock()
    p.snapshot_symbol_index.has_snapshot.return_value = True
    p.is_repo_indexed = lambda: True
    p.load_artifacts_by_key = fake_load
    p.query = fake_query
    p.semantic_escalation_cb = None
    p._snapshot_query_lock = threading.Lock()
    return p


def _snapshot_record_for(sid: str) -> SnapshotRecord:
    return SnapshotRecord(
        snapshot_id=sid,
        repo_name="repo",
        branch="main",
        commit_id=sid,
        tree_id="t1",
        artifact_key=f"art_{sid}",
        ir_path=f"/tmp/{sid}/ir.json",
        ir_graphs_path=None,
        created_at="2026-01-01",
        metadata_json=None,
    )


def _make_shared_tracking() -> tuple[dict[str, str], Any, Any]:
    """Create shared state + fake callbacks for artifact-load tracking."""
    shared: dict[str, str] = {"loaded_key": ""}

    def fake_load(artifact_key: str) -> bool:
        shared["loaded_key"] = artifact_key
        return True

    def fake_query(**kwargs: Any) -> dict[str, Any]:
        return {
            "answer": "ok",
            "query": kwargs.get("question", ""),
            "context_elements": 1,
            "sources": [],
            "_loaded_key": shared["loaded_key"],
        }

    return shared, fake_load, fake_query


@pytest.mark.regression
@pytest.mark.parametrize("n_threads", [2, 3])
def test_concurrent_queries_isolate_artifacts_with_shared_lock(n_threads: int) -> None:
    """N concurrent query_snapshot calls on a SINGLE pipeline must each
    see their own loaded artifacts when the lock serializes load+query."""
    _, fake_load, fake_query = _make_shared_tracking()
    results: dict[str, dict[str, Any]] = {}
    errors: list[Exception] = []
    barrier = threading.Barrier(n_threads, timeout=5)

    p = _build_shared_pipeline(fake_load, fake_query)
    p.snapshot_store.get_snapshot_record.side_effect = _snapshot_record_for

    def run_query(sid: str) -> None:
        try:
            barrier.wait(timeout=5)
            r = p.query_snapshot(question="test", snapshot_id=sid)
            results[sid] = r
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=run_query, args=(f"snap_{i}",))
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"threads raised: {errors}"
    for i in range(n_threads):
        key = results[f"snap_{i}"]["_loaded_key"]
        assert key == f"art_snap_{i}", f"snap_{i} got wrong artifacts: {key}"


@pytest.mark.regression
def test_forced_interleaving_without_lock_causes_cross_contamination() -> None:
    """Negative control: without the serialization lock, forced interleaving
    causes Thread A to observe Thread B's loaded artifacts.

    Event ordering (deterministic regardless of which thread starts first):
      1. A loads art_snap_A  -> shared = art_snap_A, signal a_loaded
      2. B waits for a_loaded, then loads art_snap_B  -> shared = art_snap_B
      3. A waits for b_loaded, then both query -> both see art_snap_B
    """
    shared: dict[str, str] = {"loaded_key": ""}
    a_loaded = threading.Event()
    b_loaded = threading.Event()
    results: dict[str, dict[str, Any]] = {}
    errors: list[Exception] = []

    def fake_load(artifact_key: str) -> bool:
        shared["loaded_key"] = artifact_key
        if "snap_A" in artifact_key:
            a_loaded.set()
            b_loaded.wait(timeout=5)
        else:
            a_loaded.wait(timeout=5)
            b_loaded.set()
        return True

    def fake_query(**kwargs: Any) -> dict[str, Any]:
        return {
            "answer": "ok",
            "query": kwargs.get("question", ""),
            "context_elements": 1,
            "sources": [],
            "_loaded_key": shared["loaded_key"],
        }

    p = _build_shared_pipeline(fake_load, fake_query)
    p.snapshot_store.get_snapshot_record.side_effect = _snapshot_record_for
    # Disable serialization lock
    p._snapshot_query_lock = _NoOpContext()

    def run_query(sid: str) -> None:
        try:
            r = p.query_snapshot(question="test", snapshot_id=sid)
            results[sid] = r
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=run_query, args=("snap_A",))
    t2 = threading.Thread(target=run_query, args=("snap_B",))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"threads raised: {errors}"
    # A loaded first, but B overwrote shared state before A queried.
    # A sees B's artifact — proving the race the lock prevents.
    assert results["snap_A"]["_loaded_key"] == "art_snap_B"
    assert results["snap_B"]["_loaded_key"] == "art_snap_B"
