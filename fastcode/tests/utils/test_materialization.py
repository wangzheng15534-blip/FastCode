from __future__ import annotations

import numpy as np

from fastcode.store.vector_math import as_float32_matrix
from fastcode.utils.materialization import (
    BOUNDARY_JSON_ENCODE,
    BOUNDARY_VECTOR_LIST_CONVERSION,
    collect_materialization_counters,
    increment_materialization_boundary,
)


def test_materialization_counters_are_scoped() -> None:
    increment_materialization_boundary(BOUNDARY_JSON_ENCODE)

    with collect_materialization_counters() as counters:
        increment_materialization_boundary(
            BOUNDARY_JSON_ENCODE,
            items=3,
            bytes_count=12,
        )

    metrics = counters.as_metrics()
    assert metrics["materialization_boundary_counts"] == {BOUNDARY_JSON_ENCODE: 1}
    assert metrics["materialization_boundary_items"] == {BOUNDARY_JSON_ENCODE: 3}
    assert metrics["materialization_boundary_bytes"] == {BOUNDARY_JSON_ENCODE: 12}


def test_vector_matrix_boundary_counts_list_materialization() -> None:
    vectors = [np.array([1.0, 2.0], dtype=np.float32)]

    with collect_materialization_counters() as counters:
        matrix = as_float32_matrix(vectors)

    assert matrix.shape == (1, 2)
    metrics = counters.as_metrics()
    assert (
        metrics["materialization_boundary_counts"][BOUNDARY_VECTOR_LIST_CONVERSION] == 1
    )
    assert (
        metrics["materialization_boundary_items"][BOUNDARY_VECTOR_LIST_CONVERSION] == 1
    )
