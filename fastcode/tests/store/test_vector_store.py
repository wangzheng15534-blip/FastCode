from __future__ import annotations

import json
from pathlib import Path
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


class _OpaqueValue:
    def __repr__(self) -> str:
        return "<opaque>"


def _store() -> VectorStore:
    return VectorStore({"vector_store": {"in_memory": True}})


def _disk_store(tmp_path: Path) -> VectorStore:
    return VectorStore({"vector_store": {"persist_directory": str(tmp_path)}})


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


def test_repository_overview_persists_as_explicit_manifest_and_embedding_archive_double(
    tmp_path: Path,
) -> None:
    store = _disk_store(tmp_path)

    store.save_repo_overview(
        "repo",
        "content",
        np.asarray([3.0, 4.0], dtype=np.float64),
        {"summary": "stored overview", "opaque": _OpaqueValue()},
    )

    manifest_path = tmp_path / "repo_overviews.json"
    embeddings_path = tmp_path / "repo_overviews_embeddings.npz"

    assert manifest_path.exists()
    assert embeddings_path.exists()
    assert not (tmp_path / "repo_overviews.pkl").exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_entry = manifest["repos"]["repo"]
    assert repo_entry["content"] == "content"
    assert json.loads(repo_entry["metadata_json"]) == {
        "opaque": "<opaque>",
        "summary": "stored overview",
    }

    loaded = store.load_repo_overviews()
    assert loaded["repo"]["metadata"] == {
        "opaque": "<opaque>",
        "summary": "stored overview",
    }
    assert loaded["repo"]["embedding"].dtype == np.float32
    assert np.array_equal(
        loaded["repo"]["embedding"],
        np.asarray([3.0, 4.0], dtype=np.float32),
    )


def test_repository_overview_metadata_load_skips_embedding_archive_double(
    tmp_path: Path,
) -> None:
    store = _disk_store(tmp_path)
    store.save_repo_overview(
        "repo",
        "content",
        np.asarray([1.0, 2.0], dtype=np.float32),
        {"summary": "stored overview"},
    )

    embeddings_path = tmp_path / "repo_overviews_embeddings.npz"
    embeddings_path.unlink()
    embeddings_path.write_text("corrupt archive", encoding="utf-8")

    loaded = store.load_repo_overviews(include_embeddings=False)

    assert loaded == {
        "repo": {
            "repo_name": "repo",
            "content": "content",
            "metadata": {"summary": "stored overview"},
        }
    }
