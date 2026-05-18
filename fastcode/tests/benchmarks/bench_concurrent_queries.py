"""Benchmarks for concurrent public query read sections."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from fastcode.main.fastcode import _ReadWriteStateLock

pytestmark = [pytest.mark.perf]


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
