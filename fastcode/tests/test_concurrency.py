"""Concurrency regression tests for shared-state safety."""
from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from fastcode.query_handler import QueryPipeline
from fastcode.store_records import SnapshotRecord


def _make_pipeline_with_artifact_tracking(
    snapshot_id: str, artifact_key: str
) -> tuple[QueryPipeline, dict[str, str]]:
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
    # Initialize the lock like __init__ does
    import threading

    p._snapshot_query_lock = threading.Lock()

    record = SnapshotRecord(
        snapshot_id=snapshot_id,
        repo_name="repo",
        branch="main",
        commit_id="c1",
        tree_id="t1",
        artifact_key=artifact_key,
        ir_path="/tmp/ir.json",
        ir_graphs_path=None,
        created_at="2026-01-01",
        metadata_json=None,
    )
    p.snapshot_store.get_snapshot_record.return_value = record
    return p, shared


@pytest.mark.regression
def test_two_concurrent_query_snapshot_calls_isolate_artifacts():
    """Two concurrent query_snapshot calls with different snapshots
    must not return each other's loaded artifacts."""
    barrier = threading.Barrier(2, timeout=5)
    results: dict[str, dict[str, Any]] = {}
    errors: list[Exception] = []

    def run_query(sid: str, akey: str) -> None:
        p, _ = _make_pipeline_with_artifact_tracking(sid, akey)
        try:
            barrier.wait(timeout=5)
            r = p.query_snapshot(question="test", snapshot_id=sid)
            results[sid] = r
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=run_query, args=("snap:A", "art_A"))
    t2 = threading.Thread(target=run_query, args=("snap:B", "art_B"))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"threads raised: {errors}"
    assert results.get("snap:A", {}).get("_loaded_key") == "art_A"
    assert results.get("snap:B", {}).get("_loaded_key") == "art_B"


@pytest.mark.regression
def test_three_concurrent_query_snapshot_calls_isolate_artifacts():
    """Three concurrent query_snapshot calls must each see their own artifacts."""
    barrier = threading.Barrier(3, timeout=5)
    results: dict[str, dict[str, Any]] = {}
    errors: list[Exception] = []

    def run_query(sid: str, akey: str) -> None:
        p, _ = _make_pipeline_with_artifact_tracking(sid, akey)
        try:
            barrier.wait(timeout=5)
            r = p.query_snapshot(question="test", snapshot_id=sid)
            results[sid] = r
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=run_query, args=(f"snap:{i}", f"art_{i}"))
        for i in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"threads raised: {errors}"
    for i in range(3):
        key = results.get(f"snap:{i}", {}).get("_loaded_key")
        assert key == f"art_{i}", f"snap:{i} got wrong artifacts: {key}"
