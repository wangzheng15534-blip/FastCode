from __future__ import annotations

from typing import Any

import numpy as np

from fastcode.store.vector_math import as_float32_matrix, as_float32_vector


def test_as_float32_vector_view_does_not_mutate_nonfinite_input() -> None:
    source = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float64)

    vector = as_float32_vector(source, copy_policy="view")

    assert vector is not None
    assert vector.dtype == np.float32
    assert vector.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert np.isnan(source[1])
    assert np.isposinf(source[2])
    assert np.isneginf(source[3])


def test_as_float32_matrix_view_does_not_mutate_nonfinite_input() -> None:
    source = np.array([[1.0, np.nan], [np.inf, -np.inf]], dtype=np.float64)

    matrix = as_float32_matrix(source, copy_policy="view")

    assert matrix.dtype == np.float32
    assert matrix.tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert np.isnan(source[0, 1])
    assert np.isposinf(source[1, 0])
    assert np.isneginf(source[1, 1])


def test_as_float32_matrix_contiguous_owns_backend_ready_layout() -> None:
    source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64).T

    matrix = as_float32_matrix(source, copy_policy="contiguous")

    assert matrix.dtype == np.float32
    assert matrix.flags.c_contiguous
    assert matrix.tolist() == [[1.0, 3.0], [2.0, 4.0]]


def test_as_float32_matrix_sequence_preallocates_without_vstack(
    monkeypatch: Any,
) -> None:
    def _boom_vstack(_values: object) -> np.ndarray:
        raise AssertionError("sequence conversion should preallocate rows")

    monkeypatch.setattr("fastcode.store.vector_math.np.vstack", _boom_vstack)

    matrix = as_float32_matrix(
        [
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0], dtype=np.float64),
        ]
    )

    assert matrix.dtype == np.float32
    assert matrix.tolist() == [[1.0, 2.0], [3.0, 4.0]]
