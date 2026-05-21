"""Benchmarks for copy/materialization boundary costs."""

from __future__ import annotations

import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fastcode.ir.graph import IRGraphBuilder
from fastcode.ir.types import IRCodeUnit, IRRelation, IRSnapshot
from fastcode.store.vector import VectorStore
from fastcode.store.vector_math import as_float32_matrix
from fastcode.utils.materialization import (
    BOUNDARY_VECTOR_LIST_CONVERSION,
    collect_materialization_counters,
)

pytestmark = [pytest.mark.perf]


def _vector_metadata(count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"elem:{index}",
            "type": "function",
            "relative_path": f"pkg/{index % 8}.py",
            "repo_name": "repo",
        }
        for index in range(count)
    ]


def _disk_vector_store(tmp_path: Path, **config: Any) -> VectorStore:
    return VectorStore({"vector_store": {"persist_directory": str(tmp_path), **config}})


def _snapshot(size: int) -> IRSnapshot:
    units = [
        IRCodeUnit(
            unit_id=f"unit:{index}",
            kind="function",
            path="pkg/a.py",
            language="python",
            display_name=f"f{index}",
            source_set={"fc_structure"},
        )
        for index in range(size)
    ]
    relations = [
        IRRelation(
            relation_id=f"rel:{index}",
            src_unit_id=f"unit:{index}",
            dst_unit_id=f"unit:{(index + 1) % size}",
            relation_type="call",
            resolution_state="structural",
            support_sources={"fc_structure"},
        )
        for index in range(size)
    ]
    return IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:materialization",
        units=units,
        relations=relations,
    )


def test_vector_matrix_staging_allocations_perf(benchmark: Any) -> None:
    vectors = [np.arange(128, dtype=np.float32) for _ in range(1000)]

    def _stage() -> tuple[tuple[int, int], int]:
        tracemalloc.start()
        matrix = as_float32_matrix(vectors, copy_policy="contiguous")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return matrix.shape, peak

    shape, peak = benchmark(_stage)

    assert shape == (1000, 128)
    assert peak > 0


def test_vector_matrix_materialization_counter_perf(benchmark: Any) -> None:
    vectors = [np.arange(128, dtype=np.float32) for _ in range(1000)]

    def _stage() -> tuple[tuple[int, int], int]:
        with collect_materialization_counters() as counters:
            matrix = as_float32_matrix(vectors, copy_policy="contiguous")
        return (
            matrix.shape,
            counters.counts[BOUNDARY_VECTOR_LIST_CONVERSION],
        )

    shape, conversions = benchmark(_stage)

    assert shape == (1000, 128)
    assert conversions == 1


def test_ir_graph_view_build_allocations_perf(benchmark: Any) -> None:
    snapshot = _snapshot(1000)
    builder = IRGraphBuilder()

    def _build() -> tuple[int, int]:
        tracemalloc.start()
        graphs = builder.build_graphs(snapshot)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return graphs.call_graph.number_of_edges(), peak

    edge_count, peak = benchmark(_build)

    assert edge_count == 1000
    assert peak > 0


def test_vector_store_append_allocations_perf(benchmark: Any) -> None:
    vectors = np.eye(64, dtype=np.float32).repeat(8, axis=0)
    metadata = _vector_metadata(len(vectors))

    def _append() -> tuple[int, int]:
        store = VectorStore({"vector_store": {"in_memory": True}})
        store.initialize(vectors.shape[1])
        tracemalloc.start()
        store.add_vectors(vectors, metadata)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return store.get_count(), peak

    count, peak = benchmark(_append)

    assert count == len(vectors)
    assert peak > 0


def test_vector_store_incremental_save_allocations_perf(
    tmp_path: Path, benchmark: Any
) -> None:
    previous = _disk_vector_store(tmp_path, shard_storage="npy")
    previous.initialize(32)
    base_vectors = np.eye(32, dtype=np.float32).repeat(4, axis=0)
    previous.add_vectors(base_vectors, _vector_metadata(len(base_vectors)))
    previous.save("prev")

    changed_vectors = base_vectors.copy()
    changed_vectors[-1] = np.arange(32, dtype=np.float32)
    metadata = _vector_metadata(len(changed_vectors))

    def _save_incremental() -> tuple[dict[str, int], int]:
        current = _disk_vector_store(tmp_path, shard_storage="npy")
        current.initialize(32)
        current.add_vectors(changed_vectors, metadata)
        tracemalloc.start()
        stats = current.save_incremental(
            "next",
            previous_name="prev",
            reusable_path_keys={f"pkg/{index}.py" for index in range(7)},
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return stats, peak

    stats, peak = benchmark(_save_incremental)

    assert stats["vector_shards_reused"] > 0
    assert stats["vector_shards_written"] > 0
    assert peak > 0


def test_vector_store_lazy_load_search_allocations_perf(
    tmp_path: Path, benchmark: Any
) -> None:
    source = _disk_vector_store(tmp_path, shard_storage="npy")
    source.initialize(32)
    vectors = np.eye(32, dtype=np.float32).repeat(4, axis=0)
    source.add_vectors(vectors, _vector_metadata(len(vectors)))
    source.save("repo")

    def _load_and_search() -> tuple[str, int]:
        loaded = _disk_vector_store(
            tmp_path,
            shard_storage="npy",
            lazy_shard_search=True,
        )
        tracemalloc.start()
        assert loaded.load("repo") is True
        results = loaded.search(vectors[0], k=1)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return str(results[0][0]["id"]), peak

    element_id, peak = benchmark(_load_and_search)

    assert element_id == "elem:0"
    assert peak > 0
