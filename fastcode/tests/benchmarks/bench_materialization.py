"""Benchmarks for copy/materialization boundary costs."""

from __future__ import annotations

import tracemalloc
from typing import Any

import numpy as np
import pytest

from fastcode.ir.graph import IRGraphBuilder
from fastcode.ir.types import IRCodeUnit, IRRelation, IRSnapshot
from fastcode.utils import as_float32_matrix

pytestmark = [pytest.mark.perf]


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
