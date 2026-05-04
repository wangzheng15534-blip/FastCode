from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fastcode.store.vector import VectorStore

pytestmark = [pytest.mark.test_double]


class _MetadataBomb(dict[str, Any]):
    def __getitem__(self, key: str) -> Any:
        if key == "metadata":
            raise AssertionError("metadata for unreturned overview was materialized")
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "metadata":
            raise AssertionError("metadata for unreturned overview was materialized")
        return super().get(key, default)


def _store() -> VectorStore:
    return VectorStore({"vector_store": {"in_memory": True}})


def test_repository_overview_search_only_materializes_returned_metadata_double() -> (
    None
):
    store = _store()
    store._in_memory_repo_overviews = {
        "best": {
            "repo_name": "best",
            "content": "",
            "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
            "metadata": {"summary": "best match"},
        },
        "other": _MetadataBomb(
            {
                "repo_name": "other",
                "content": "",
                "embedding": np.asarray([0.0, 1.0], dtype=np.float32),
                "metadata": {"summary": "should stay cold"},
            }
        ),
    }

    results = store.search_repository_overviews(
        np.asarray([1.0, 0.0], dtype=np.float32),
        k=1,
    )

    assert results == [
        (
            {
                "repo_name": "best",
                "type": "repository_overview",
                "summary": "best match",
            },
            1.0,
        )
    ]


def test_repository_overview_search_does_not_mutate_stored_embeddings_double() -> None:
    store = _store()
    original = np.asarray([3.0, 4.0], dtype=np.float32)
    store.save_repo_overview(
        "repo",
        "content",
        original,
        {"summary": "stored overview"},
    )
    stored_embedding = store._in_memory_repo_overviews["repo"]["embedding"]
    before = stored_embedding.copy()

    results = store.search_repository_overviews(
        np.asarray([1.0, 0.0], dtype=np.float32),
        k=1,
    )

    assert results[0][0]["repo_name"] == "repo"
    assert np.array_equal(stored_embedding, before)
    assert np.array_equal(stored_embedding, original)
