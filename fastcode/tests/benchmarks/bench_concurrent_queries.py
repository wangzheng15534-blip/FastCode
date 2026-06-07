"""Benchmarks for concurrent public query read sections."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from typing import Any
from unittest.mock import patch

import pytest

import fastcode.api.routes as api
from fastcode.runtime_support.runtime_state import _ReadWriteStateLock

pytestmark = [pytest.mark.perf]


class _EndpointBenchmarkFastCode:
    def __init__(self, *, delay_seconds: float, mutation_delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.mutation_delay_seconds = mutation_delay_seconds
        self._state_lock = _ReadWriteStateLock()
        self.current_queries = 0
        self.current_mutations = 0
        self.max_concurrent_queries = 0
        self.query_write_overlaps = 0
        self.mutation_count = 0
        self.snapshot_ids: list[str] = []
        self._lock = Lock()

    def query_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        with self._state_lock.read_lock():
            return self._query_snapshot_unlocked(**kwargs)

    def _query_snapshot_unlocked(self, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            self.current_queries += 1
            self.max_concurrent_queries = max(
                self.max_concurrent_queries,
                self.current_queries,
            )
            if self.current_mutations:
                self.query_write_overlaps += 1
            snapshot_id = kwargs.get("snapshot_id")
            if isinstance(snapshot_id, str):
                self.snapshot_ids.append(snapshot_id)
        time.sleep(self.delay_seconds)
        with self._lock:
            self.current_queries -= 1
        return {
            "answer": "ok",
            "query": kwargs["question"],
            "context_elements": 0,
            "sources": [],
        }

    def run_background_mutations(
        self,
        *,
        mutation_count: int,
        start_event: Event,
    ) -> int:
        start_event.wait(timeout=1)
        completed = 0
        for _ in range(mutation_count):
            with self._state_lock.write_lock():
                with self._lock:
                    self.current_mutations += 1
                    if self.current_queries:
                        self.query_write_overlaps += 1
                time.sleep(self.mutation_delay_seconds)
                with self._lock:
                    self.current_mutations -= 1
                    self.mutation_count += 1
                completed += 1
        return completed


def _run_query_lock_workload(
    *,
    query_count: int,
    workers: int,
    background_mutations: bool,
) -> dict[str, int | float | bool]:
    lock = _ReadWriteStateLock()
    started = time.perf_counter()

    def query_task(_index: int) -> int:
        with lock.read_lock():
            time.sleep(0.001)
            return 1

    def mutation_task() -> int:
        mutations = 0
        for _ in range(max(1, query_count // workers)):
            with lock.write_lock():
                mutations += 1
                time.sleep(0.001)
        return mutations

    with ThreadPoolExecutor(max_workers=workers + int(background_mutations)) as pool:
        query_futures = [pool.submit(query_task, index) for index in range(query_count)]
        mutation_future = pool.submit(mutation_task) if background_mutations else None
        completed_queries = sum(future.result() for future in query_futures)
        completed_mutations = mutation_future.result() if mutation_future else 0

    return {
        "query_count": completed_queries,
        "workers": workers,
        "background_mutations": background_mutations,
        "mutation_count": completed_mutations,
        "wall_time_ms": round((time.perf_counter() - started) * 1000, 3),
    }


async def _run_query_endpoint_requests(query_count: int) -> int:
    from types import SimpleNamespace

    mock_req = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(fastcode=None))
    )
    responses = await asyncio.gather(
        *(
            api.query_repository(
                mock_req,
                api.QueryRequest(
                    question=f"Where is component {index}?",
                    snapshot_id=f"snap:{index}",
                    repo_name=None,
                    ref_name=None,
                    filters=None,
                    repo_filter=None,
                    multi_turn=False,
                    session_id=f"bench-{index}",
                ),
            )
            for index in range(query_count)
        )
    )
    return len(responses)


def _run_query_endpoint_workload(
    *,
    query_count: int,
    delay_seconds: float,
    background_mutations: bool = False,
) -> dict[str, int | float]:
    mutation_count = max(1, query_count // 4) if background_mutations else 0
    fastcode = _EndpointBenchmarkFastCode(
        delay_seconds=delay_seconds,
        mutation_delay_seconds=delay_seconds,
    )
    start_event = Event()
    started = time.perf_counter()
    with (
        patch(
            "fastcode.api.routes._fc",
            return_value=fastcode,
        ),
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        mutation_future = (
            pool.submit(
                fastcode.run_background_mutations,
                mutation_count=mutation_count,
                start_event=start_event,
            )
            if background_mutations
            else None
        )
        start_event.set()
        completed_queries = asyncio.run(_run_query_endpoint_requests(query_count))
        completed_mutations = mutation_future.result() if mutation_future else 0

    return {
        "query_count": completed_queries,
        "unique_snapshots": len(set(fastcode.snapshot_ids)),
        "max_concurrent_queries": fastcode.max_concurrent_queries,
        "background_mutations": background_mutations,
        "mutation_count": completed_mutations,
        "query_write_overlaps": fastcode.query_write_overlaps,
        "wall_time_ms": round((time.perf_counter() - started) * 1000, 3),
    }


@pytest.mark.parametrize("background_mutations", [False, True])
def test_concurrent_query_read_lock_perf(
    background_mutations: bool,
    benchmark: Any,
) -> None:
    result = benchmark(
        _run_query_lock_workload,
        query_count=32,
        workers=8,
        background_mutations=background_mutations,
    )

    assert result["query_count"] == 32
    assert result["workers"] == 8
    if background_mutations:
        assert result["mutation_count"] > 0
    else:
        assert result["mutation_count"] == 0


@pytest.mark.parametrize("background_mutations", [False, True])
def test_query_endpoint_snapshot_concurrency_perf(
    background_mutations: bool,
    benchmark: Any,
) -> None:
    result = benchmark(
        _run_query_endpoint_workload,
        query_count=8,
        delay_seconds=0.003,
        background_mutations=background_mutations,
    )

    assert result["query_count"] == 8
    assert result["unique_snapshots"] == 8
    assert result["max_concurrent_queries"] > 1
    assert result["query_write_overlaps"] == 0
    if background_mutations:
        assert result["mutation_count"] == 2
    else:
        assert result["mutation_count"] == 0
