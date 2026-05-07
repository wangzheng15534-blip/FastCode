from __future__ import annotations

import json
import time
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


def _metadata_row(path: str, *, element_id: str, summary: str) -> dict[str, Any]:
    return {
        "id": element_id,
        "type": "function",
        "name": element_id,
        "file_path": f"/repo/{path}",
        "relative_path": path,
        "language": "python",
        "start_line": 1,
        "end_line": 1,
        "code": "pass\n",
        "signature": None,
        "docstring": None,
        "summary": summary,
        "metadata": {"stable_unit_id": f"unit:{element_id}"},
        "repo_name": "repo",
        "repo_url": None,
    }


def test_vector_store_save_load_uses_sharded_metadata_bundle(tmp_path: Path) -> None:
    store = _disk_store(tmp_path)
    store.initialize(3)
    store.add_vectors(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        [
            _metadata_row("pkg/a.py", element_id="elem:a", summary="a"),
            _metadata_row("pkg/b.py", element_id="elem:b", summary="b"),
        ],
    )
    (tmp_path / "repo.faiss").write_bytes(b"legacy")
    (tmp_path / "repo_metadata.pkl").write_bytes(b"legacy")

    store.save("repo")
    loaded = _disk_store(tmp_path)
    assert loaded.load("repo") is True

    assert not (tmp_path / "repo.faiss").exists()
    assert (tmp_path / "repo_vector_manifest.json").exists()
    assert (tmp_path / "repo_vector_shards").is_dir()
    assert (tmp_path / "repo_metadata_manifest.json").exists()
    assert (tmp_path / "repo_metadata_shards").is_dir()
    assert not (tmp_path / "repo_metadata.pkl").exists()
    assert [row["id"] for row in loaded.metadata] == ["elem:a", "elem:b"]
    assert loaded.get_count() == 2


def test_vector_store_sharded_load_keeps_raw_rows_hot_for_search(
    tmp_path: Path,
) -> None:
    store = _disk_store(tmp_path)
    store.initialize(3)
    store.add_vectors(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        [
            _metadata_row("pkg/a.py", element_id="elem:a", summary="a"),
            _metadata_row("pkg/b.py", element_id="elem:b", summary="b"),
        ],
    )
    store.save("repo")

    loaded = _disk_store(tmp_path)

    assert loaded.load("repo") is True
    assert loaded.index is None
    assert loaded._vector_rows is not None

    results = loaded.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), k=1)

    assert loaded.index is None
    assert results[0][0]["id"] == "elem:a"
    assert results[0][1] == pytest.approx(1.0)


def test_vector_store_raw_row_search_does_not_mutate_query_inputs(
    tmp_path: Path,
) -> None:
    store = _disk_store(tmp_path)
    store.initialize(3)
    store.add_vectors(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        [
            _metadata_row("pkg/a.py", element_id="elem:a", summary="a"),
            _metadata_row("pkg/b.py", element_id="elem:b", summary="b"),
        ],
    )
    store.save("repo")

    loaded = _disk_store(tmp_path)
    assert loaded.load("repo") is True

    query = np.asarray([3.0, 4.0, 0.0], dtype=np.float32)
    query_before = query.copy()
    batch_query = np.asarray([[3.0, 4.0, 0.0]], dtype=np.float32)
    batch_before = batch_query.copy()

    loaded.search(query, k=1)
    loaded.search_batch(batch_query, k=1)

    assert np.array_equal(query, query_before)
    assert np.array_equal(batch_query, batch_before)


def test_vector_store_lazy_load_can_rebuild_faiss_on_demand(tmp_path: Path) -> None:
    store = _disk_store(tmp_path)
    store.initialize(3)
    store.add_vectors(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        [
            _metadata_row("pkg/a.py", element_id="elem:a", summary="a"),
            _metadata_row("pkg/b.py", element_id="elem:b", summary="b"),
        ],
    )
    store.save("repo")

    loaded = VectorStore(
        {
            "vector_store": {
                "persist_directory": str(tmp_path),
                "persist_faiss_binary": True,
                "vector_rows_search_threshold": 1,
            }
        }
    )

    assert loaded.load("repo") is True
    assert loaded.index is None

    results = loaded.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), k=1)

    assert loaded.index is not None
    assert results[0][0]["id"] == "elem:a"

    loaded.save("repo")

    assert (tmp_path / "repo.faiss").exists()


def test_vector_store_save_reuses_unchanged_metadata_shards(tmp_path: Path) -> None:
    store = _disk_store(tmp_path)
    store.initialize(3)
    store.add_vectors(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        [
            _metadata_row("pkg/a.py", element_id="elem:a", summary="a1"),
            _metadata_row("pkg/b.py", element_id="elem:b", summary="b1"),
        ],
    )
    store.save("repo")

    metadata_manifest = json.loads(
        (tmp_path / "repo_metadata_manifest.json").read_text(encoding="utf-8")
    )
    vector_manifest = json.loads(
        (tmp_path / "repo_vector_manifest.json").read_text(encoding="utf-8")
    )
    metadata_shards = {
        entry["path_key"]: tmp_path / "repo_metadata_shards" / entry["shard_file"]
        for entry in metadata_manifest["shards"]
    }
    vector_shards = {
        entry["path_key"]: tmp_path / "repo_vector_shards" / entry["shard_file"]
        for entry in vector_manifest["shards"]
    }
    a_metadata_before = metadata_shards["pkg/a.py"].stat().st_mtime_ns
    b_metadata_before = metadata_shards["pkg/b.py"].stat().st_mtime_ns
    a_vector_before = vector_shards["pkg/a.py"].stat().st_mtime_ns
    b_vector_before = vector_shards["pkg/b.py"].stat().st_mtime_ns

    time.sleep(0.01)
    store.initialize(3)
    store.add_vectors(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=np.float32),
        [
            _metadata_row("pkg/a.py", element_id="elem:a", summary="a1"),
            _metadata_row("pkg/b.py", element_id="elem:b", summary="b2"),
        ],
    )
    store.save("repo")

    assert metadata_shards["pkg/a.py"].stat().st_mtime_ns == a_metadata_before
    assert metadata_shards["pkg/b.py"].stat().st_mtime_ns > b_metadata_before
    assert vector_shards["pkg/a.py"].stat().st_mtime_ns == a_vector_before
    assert vector_shards["pkg/b.py"].stat().st_mtime_ns > b_vector_before


def test_vector_store_merge_from_sharded_vectors_without_faiss(tmp_path: Path) -> None:
    source = _disk_store(tmp_path)
    source.initialize(3)
    source.add_vectors(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        [
            _metadata_row("pkg/a.py", element_id="elem:a", summary="a"),
            _metadata_row("pkg/b.py", element_id="elem:b", summary="b"),
        ],
    )
    source.save("repo")

    target = _disk_store(tmp_path)
    target.initialize(3)
    assert target.merge_from_index("repo") is True
    assert target.get_count() == 2
    assert [row["id"] for row in target.metadata] == ["elem:a", "elem:b"]


def test_vector_store_scan_finds_sharded_indexes_without_faiss(tmp_path: Path) -> None:
    store = _disk_store(tmp_path)
    store.initialize(3)
    store.add_vectors(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        [_metadata_row("pkg/a.py", element_id="elem:a", summary="a")],
    )
    store.save("repo")

    available = store.scan_available_indexes(use_cache=False)

    assert not (tmp_path / "repo.faiss").exists()
    assert len(available) == 1
    assert available[0]["name"] == "repo"
    assert available[0]["element_count"] == 1
    assert available[0]["file_count"] == 1
    assert available[0]["size_mb"] >= 0
    assert available[0]["url"] == "N/A"
