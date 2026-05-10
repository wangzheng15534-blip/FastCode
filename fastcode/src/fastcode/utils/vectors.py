"""NumPy vector boundary helpers.

These helpers make copy intent explicit at native/vector boundaries.  The
default path returns float32 views where NumPy can provide them; callers request
contiguous or mutable storage only when a downstream backend requires it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

VectorCopyPolicy = Literal["view", "contiguous", "mutable"]


def as_float32_vector(
    value: Any,
    *,
    copy_policy: VectorCopyPolicy = "view",
) -> np.ndarray | None:
    """Return a 1-D float32 vector, or None for invalid/empty input."""
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size == 0:
        return None
    if not bool(np.isfinite(vector).all()):
        vector = np.nan_to_num(vector, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if copy_policy == "contiguous":
        return np.ascontiguousarray(vector, dtype=np.float32)
    if copy_policy == "mutable":
        return np.array(vector, dtype=np.float32, copy=True)
    return vector.astype(np.float32, copy=False)


def as_float32_matrix(
    values: Sequence[Any] | np.ndarray,
    *,
    copy_policy: VectorCopyPolicy = "view",
) -> np.ndarray:
    """Return a 2-D float32 matrix with explicit copy behavior."""
    if isinstance(values, np.ndarray):
        try:
            matrix = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError):
            return np.empty((0, 0), dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        elif matrix.ndim != 2:
            return np.empty((0, 0), dtype=np.float32)
    else:
        vectors = [as_float32_vector(value, copy_policy="view") for value in values]
        valid_vectors = [vector for vector in vectors if vector is not None]
        if not valid_vectors:
            return np.empty((0, 0), dtype=np.float32)
        try:
            matrix = np.vstack(valid_vectors).astype(np.float32, copy=False)
        except ValueError:
            return np.empty((0, 0), dtype=np.float32)

    if matrix.size and not bool(np.isfinite(matrix).all()):
        matrix = np.nan_to_num(matrix, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if copy_policy == "contiguous":
        return np.ascontiguousarray(matrix, dtype=np.float32)
    if copy_policy == "mutable":
        return np.array(matrix, dtype=np.float32, copy=True)
    return matrix.astype(np.float32, copy=False)
