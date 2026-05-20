from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pytest

from fastcode.store.records import RepositoryOverviewRecord
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


def _disk_store_with_vector_config(tmp_path: Path, **config: Any) -> VectorStore:
    return VectorStore({"vector_store": {"persist_directory": str(tmp_path), **config}})


def _meta(element_id: str, path: str) -> dict[str, Any]:
    return {
        "id": element_id,
        "type": "function",
        "name": element_id,
        "file_path": f"/repo/{path}",
        "relative_path": path,
        "language": "python",
        "start_line": 1,
        "end_line": 1,
        "code": "return 1",
        "signature": None,
        "docstring": None,
        "summary": None,
        "metadata": {},
        "repo_name": "repo",
        "repo_url": None,
    }


def test_add_vectors_uses_growing_row_buffer_without_vstack_double(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store()
    store.initialize(2)

    def _boom_vstack(_values: object) -> np.ndarray:
        raise AssertionError("vector row appends should not use np.vstack")

    monkeypatch.setattr("fastcode.store.vector.np.vstack", _boom_vstack)

    store.add_vectors(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [_meta("a", "a.py"), _meta("b", "b.py")],
    )
    store.add_vectors(
        np.asarray([[0.5, 0.5]], dtype=np.float32),
        [_meta("c", "c.py")],
    )

    rows = store._vector_matrix_for_persist()
    assert rows.shape == (3, 2)
    assert store._vector_rows is not None
    assert store._vector_rows.shape[0] >= 3
    assert store.search(np.asarray([1.0, 0.0], dtype=np.float32), k=1)


def test_search_filters_same_dimension_stale_embedding_fingerprint_double() -> None:
    store = _store()
    store.initialize(2)
    stale_fingerprint = {
        "version": 2,
        "provider": "test",
        "model": "old",
        "dimension": 2,
        "text_schema_version": 1,
    }
    current_fingerprint = {**stale_fingerprint, "model": "current"}
    stale = _meta("stale", "stale.py")
    stale["metadata"]["embedding_fingerprint"] = stale_fingerprint
    current = _meta("current", "current.py")
    current["embedding_fingerprint"] = current_fingerprint
    store.add_vectors(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [stale, current],
    )

    results = store.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        k=1,
        query_embedding_fingerprint=current_fingerprint,
    )

    assert [metadata["id"] for metadata, _score in results] == ["current"]


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


def test_repository_overview_search_filters_stale_embedding_fingerprint_double() -> (
    None
):
    store = _store()
    stale_fingerprint = {
        "version": 2,
        "provider": "test",
        "model": "old",
        "dimension": 2,
        "text_schema_version": 1,
    }
    current_fingerprint = {**stale_fingerprint, "model": "current"}
    store.save_repo_overview(
        "stale",
        "stale content",
        np.asarray([1.0, 0.0], dtype=np.float32),
        {"summary": "stale", "embedding_fingerprint": stale_fingerprint},
    )
    store.save_repo_overview(
        "current",
        "current content",
        np.asarray([0.0, 1.0], dtype=np.float32),
        {"summary": "current", "embedding_fingerprint": current_fingerprint},
    )

    results = store.search_repository_overviews(
        np.asarray([1.0, 0.0], dtype=np.float32),
        k=1,
        query_embedding_fingerprint=current_fingerprint,
    )

    assert [metadata["repo_name"] for metadata, _score in results] == ["current"]


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
    fingerprint = {
        "version": 2,
        "provider": "test",
        "model": "overview",
        "dimension": 2,
        "text_schema_version": 1,
    }

    store.save_repo_overview(
        "repo",
        "content",
        np.asarray([3.0, 4.0], dtype=np.float64),
        {
            "summary": "stored overview",
            "opaque": _OpaqueValue(),
            "embedding_fingerprint": fingerprint,
        },
    )

    manifest_path = tmp_path / "repo_overviews.json"
    embeddings_path = tmp_path / "repo_overviews_embeddings.npz"

    assert manifest_path.exists()
    assert embeddings_path.exists()
    assert not (tmp_path / "repo_overviews.pkl").exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_entry = manifest["repos"]["repo"]
    assert repo_entry["content"] == "content"
    assert repo_entry["embedding_fingerprint"] == fingerprint
    assert json.loads(repo_entry["metadata_json"]) == {
        "embedding_fingerprint": fingerprint,
        "opaque": "<opaque>",
        "summary": "stored overview",
    }

    loaded = store.load_repo_overviews()
    assert loaded["repo"]["metadata"] == {
        "embedding_fingerprint": fingerprint,
        "opaque": "<opaque>",
        "summary": "stored overview",
    }
    assert loaded["repo"]["embedding"].dtype == np.float32
    assert np.array_equal(
        loaded["repo"]["embedding"],
        np.asarray([3.0, 4.0], dtype=np.float32),
    )

    records = store.load_repo_overview_records()
    assert set(records) == {"repo"}
    assert records["repo"].repo_name == "repo"
    assert records["repo"].content == "content"
    assert records["repo"].metadata_json == repo_entry["metadata_json"]
    assert records["repo"].embedding_fingerprint == fingerprint
    assert isinstance(records["repo"].embedding, np.ndarray)
    assert records["repo"].embedding.dtype == np.float32


def test_repository_overview_legacy_load_avoids_record_to_dict_double(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _disk_store(tmp_path)
    store.save_repo_overview(
        "repo",
        "content",
        np.asarray([1.0, 2.0], dtype=np.float32),
        {"summary": "stored overview"},
    )

    def _boom(_: RepositoryOverviewRecord) -> dict[str, Any]:
        raise AssertionError("repo overview compatibility load must be explicit")

    monkeypatch.setattr(RepositoryOverviewRecord, "to_dict", _boom)

    loaded = store.load_repo_overviews()

    assert loaded["repo"]["metadata"] == {"summary": "stored overview"}
    assert loaded["repo"]["embedding"].dtype == np.float32


def test_repository_overview_search_records_return_typed_records_double() -> None:
    store = _store()
    store.save_repo_overview(
        "repo",
        "content",
        np.asarray([1.0, 0.0], dtype=np.float32),
        {"summary": "stored overview"},
    )

    results = store.search_repository_overview_records(
        np.asarray([1.0, 0.0], dtype=np.float32),
        k=1,
    )

    assert len(results) == 1
    record, score = results[0]
    assert record.repo_name == "repo"
    assert record.content == "content"
    assert json.loads(record.metadata_json) == {"summary": "stored overview"}
    assert isinstance(record.embedding, np.ndarray)
    assert score == 1.0


def test_repository_overview_legacy_search_avoids_record_to_dict_double(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store()
    store.save_repo_overview(
        "repo",
        "content",
        np.asarray([1.0, 0.0], dtype=np.float32),
        {"summary": "stored overview"},
    )

    def _boom(_: RepositoryOverviewRecord) -> dict[str, Any]:
        raise AssertionError("repo overview compatibility search must be explicit")

    monkeypatch.setattr(RepositoryOverviewRecord, "to_dict", _boom)

    results = store.search_repository_overviews(
        np.asarray([1.0, 0.0], dtype=np.float32),
        k=1,
    )

    assert results == [
        (
            {
                "repo_name": "repo",
                "type": "repository_overview",
                "summary": "stored overview",
            },
            1.0,
        )
    ]


def test_save_incremental_reuses_unchanged_vector_shards_double(
    tmp_path: Path,
) -> None:
    previous = _disk_store(tmp_path)
    previous.initialize(2)
    previous.add_vectors(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [
            {
                "id": "elem:a",
                "type": "file",
                "name": "a.py",
                "file_path": "/repo/pkg/a.py",
                "relative_path": "pkg/a.py",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "code": "a",
                "signature": None,
                "docstring": None,
                "summary": None,
                "metadata": {"snapshot_id": "snap:old"},
                "repo_name": "repo",
                "repo_url": None,
                "snapshot_id": "snap:old",
            },
            {
                "id": "elem:b",
                "type": "file",
                "name": "b.py",
                "file_path": "/repo/pkg/b.py",
                "relative_path": "pkg/b.py",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "code": "b",
                "signature": None,
                "docstring": None,
                "summary": None,
                "metadata": {"snapshot_id": "snap:old"},
                "repo_name": "repo",
                "repo_url": None,
                "snapshot_id": "snap:old",
            },
        ],
    )
    previous.save("prev")
    prev_manifest = json.loads(
        (tmp_path / "prev_vector_manifest.json").read_text(encoding="utf-8")
    )
    prev_shards = {
        entry["path_key"]: tmp_path / "prev_vector_shards" / entry["shard_file"]
        for entry in prev_manifest["shards"]
    }

    current = _disk_store(tmp_path)
    current.initialize(2)
    current.add_vectors(
        np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
        [
            {
                "id": "elem:a",
                "type": "file",
                "name": "a.py",
                "file_path": "/repo/pkg/a.py",
                "relative_path": "pkg/a.py",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "code": "a",
                "signature": None,
                "docstring": None,
                "summary": None,
                "metadata": {"snapshot_id": "snap:new"},
                "repo_name": "repo",
                "repo_url": None,
                "snapshot_id": "snap:new",
            },
            {
                "id": "elem:b",
                "type": "file",
                "name": "b.py",
                "file_path": "/repo/pkg/b.py",
                "relative_path": "pkg/b.py",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "code": "b changed",
                "signature": None,
                "docstring": None,
                "summary": None,
                "metadata": {"snapshot_id": "snap:new"},
                "repo_name": "repo",
                "repo_url": None,
                "snapshot_id": "snap:new",
            },
        ],
    )

    stats = current.save_incremental(
        "next",
        previous_name="prev",
        reusable_path_keys={"pkg/a.py"},
    )

    assert stats["vector_shards_reused"] == 1
    next_manifest = json.loads(
        (tmp_path / "next_vector_manifest.json").read_text(encoding="utf-8")
    )
    next_shards = {
        entry["path_key"]: tmp_path / "next_vector_shards" / entry["shard_file"]
        for entry in next_manifest["shards"]
    }
    assert next_shards["pkg/a.py"].read_bytes() == prev_shards["pkg/a.py"].read_bytes()
    assert next_shards["pkg/b.py"].read_bytes() != prev_shards["pkg/b.py"].read_bytes()

    loaded = _disk_store(tmp_path)
    assert loaded.load("next") is True
    assert {row.get("snapshot_id") for row in loaded.metadata} == {"snap:new"}


def test_save_incremental_refuses_incompatible_previous_vector_manifest_double(
    tmp_path: Path,
) -> None:
    previous = _disk_store(tmp_path)
    previous.initialize(2)
    previous.add_vectors(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        [
            {
                "id": "elem:a",
                "type": "file",
                "name": "a.py",
                "file_path": "/repo/pkg/a.py",
                "relative_path": "pkg/a.py",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "code": "a",
                "signature": None,
                "docstring": None,
                "summary": None,
                "metadata": {"snapshot_id": "snap:old"},
                "repo_name": "repo",
                "repo_url": None,
                "snapshot_id": "snap:old",
            }
        ],
    )
    previous.save("prev")

    current = _disk_store(tmp_path)
    current.initialize(3)
    current.add_vectors(
        np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        [
            {
                "id": "elem:a",
                "type": "file",
                "name": "a.py",
                "file_path": "/repo/pkg/a.py",
                "relative_path": "pkg/a.py",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "code": "a",
                "signature": None,
                "docstring": None,
                "summary": None,
                "metadata": {"snapshot_id": "snap:new"},
                "repo_name": "repo",
                "repo_url": None,
                "snapshot_id": "snap:new",
            }
        ],
    )

    stats = current.save_incremental(
        "next",
        previous_name="prev",
        reusable_path_keys={"pkg/a.py"},
    )

    assert stats["vector_shards_reused"] == 0
    assert stats["vector_shards_written"] == 1
    loaded = _disk_store(tmp_path)
    assert loaded.load("next") is True
    assert loaded.dimension == 3
    assert loaded._vector_rows is not None
    assert loaded._vector_rows.shape == (1, 3)


def test_save_incremental_faiss_sidecar_matches_sequence_order_double(
    tmp_path: Path,
) -> None:
    config = {
        "vector_store": {
            "persist_directory": str(tmp_path),
            "persist_faiss_binary": True,
            "index_type": "Flat",
            "distance_metric": "l2",
        }
    }
    previous = VectorStore(config)
    previous.initialize(2)
    previous.add_vectors(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [
            _metadata_row("pkg/a.py", element_id="elem:a", summary="a"),
            _metadata_row("pkg/b.py", element_id="elem:b", summary="b"),
        ],
    )
    previous.save("prev")

    current = VectorStore(config)
    current.initialize(2)
    current.add_vectors(
        np.asarray([[0.25, 0.75], [1.0, 0.0]], dtype=np.float32),
        [
            _metadata_row("pkg/b.py", element_id="elem:b", summary="b2"),
            _metadata_row("pkg/a.py", element_id="elem:a", summary="a"),
        ],
    )

    stats = current.save_incremental(
        "next",
        previous_name="prev",
        reusable_path_keys={"pkg/a.py"},
    )

    assert stats["vector_shards_reused"] == 1
    metadata_payload = current.load_metadata_payload("next")
    assert metadata_payload is not None
    assert [row["id"] for row in metadata_payload["metadata"]] == [
        "elem:a",
        "elem:b",
    ]
    index = faiss.read_index(str(tmp_path / "next.faiss"))
    row0 = np.zeros(2, dtype=np.float32)
    row1 = np.zeros(2, dtype=np.float32)
    index.reconstruct(0, row0)
    index.reconstruct(1, row1)
    assert np.allclose(row0, np.asarray([1.0, 0.0], dtype=np.float32))
    assert np.allclose(row1, np.asarray([0.25, 0.75], dtype=np.float32))


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
    records = store.load_repo_overview_records(include_embeddings=False)

    assert loaded == {
        "repo": {
            "repo_name": "repo",
            "content": "content",
            "metadata": {"summary": "stored overview"},
        }
    }
    assert records["repo"].embedding is None
    assert records["repo"].metadata_json == '{"summary": "stored overview"}'


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


def _metadata_row_with_fingerprint(
    path: str,
    *,
    element_id: str,
    summary: str,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    row = _metadata_row(path, element_id=element_id, summary=summary)
    row["embedding_fingerprint"] = dict(fingerprint)
    row["metadata"]["embedding_fingerprint"] = dict(fingerprint)
    return row


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


def test_vector_and_metadata_manifests_persist_embedding_fingerprint(
    tmp_path: Path,
) -> None:
    fingerprint = {
        "version": 2,
        "provider": "test",
        "model": "stub",
        "dimension": 3,
        "text_schema_version": 1,
    }
    store = _disk_store(tmp_path)
    store.initialize(3)
    store.add_vectors(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        [
            _metadata_row_with_fingerprint(
                "pkg/a.py",
                element_id="elem:a",
                summary="a",
                fingerprint=fingerprint,
            )
        ],
    )

    store.save("repo")

    vector_manifest = json.loads(
        (tmp_path / "repo_vector_manifest.json").read_text(encoding="utf-8")
    )
    metadata_manifest = json.loads(
        (tmp_path / "repo_metadata_manifest.json").read_text(encoding="utf-8")
    )
    assert vector_manifest["embedding_fingerprint"] == fingerprint
    assert metadata_manifest["embedding_fingerprint"] == fingerprint


def test_incremental_vector_save_refuses_reuse_on_embedding_fingerprint_change(
    tmp_path: Path,
) -> None:
    previous_fingerprint = {
        "version": 2,
        "provider": "test",
        "model": "old",
        "dimension": 3,
        "text_schema_version": 1,
    }
    current_fingerprint = {**previous_fingerprint, "model": "new"}
    previous = _disk_store(tmp_path)
    previous.initialize(3)
    previous.add_vectors(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        [
            _metadata_row_with_fingerprint(
                "pkg/a.py",
                element_id="elem:a",
                summary="a",
                fingerprint=previous_fingerprint,
            )
        ],
    )
    previous.save("old")

    current = _disk_store(tmp_path)
    current.initialize(3)
    current.add_vectors(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        [
            _metadata_row_with_fingerprint(
                "pkg/a.py",
                element_id="elem:a",
                summary="a",
                fingerprint=current_fingerprint,
            )
        ],
    )

    stats = current.save_incremental(
        "new",
        previous_name="old",
        reusable_path_keys={"pkg/a.py"},
    )

    assert stats["vector_shards_reused"] == 0
    assert stats["vector_shards_written"] == 1


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


def test_vector_store_can_persist_mmap_capable_npy_vector_shards(
    tmp_path: Path,
) -> None:
    store = _disk_store_with_vector_config(tmp_path, shard_storage="npy")
    store.initialize(3)
    store.add_vectors(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        [
            _metadata_row("pkg/a.py", element_id="elem:a", summary="a"),
            _metadata_row("pkg/b.py", element_id="elem:b", summary="b"),
        ],
    )

    store.save("repo")

    manifest = json.loads(
        (tmp_path / "repo_vector_manifest.json").read_text(encoding="utf-8")
    )
    shard_entry = manifest["shards"][0]
    assert shard_entry["storage_format"] == "npy"
    assert "shard_file" not in shard_entry
    sequence_path = tmp_path / "repo_vector_shards" / shard_entry["sequence_file"]
    vector_path = tmp_path / "repo_vector_shards" / shard_entry["vector_file"]
    assert sequence_path.suffix == ".npy"
    assert vector_path.suffix == ".npy"
    assert sequence_path.exists()
    assert vector_path.exists()

    loaded_arrays = store._load_vector_shard_arrays(
        shard_dir=str(tmp_path / "repo_vector_shards"),
        entry=shard_entry,
        mmap_mode="r",
    )
    assert loaded_arrays is not None
    loaded_sequences, loaded_vectors = loaded_arrays
    assert isinstance(loaded_sequences, np.memmap)
    assert isinstance(loaded_vectors, np.memmap)

    loaded = _disk_store(tmp_path)
    assert loaded.load("repo") is True
    assert loaded._vector_rows is not None
    assert np.array_equal(
        loaded._vector_rows,
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    )
    results = loaded.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert results[0][0]["id"] == "elem:a"


def test_incremental_npy_publish_reads_sequence_shards_without_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = _disk_store_with_vector_config(tmp_path, shard_storage="npy")
    previous.initialize(3)
    previous.add_vectors(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        [
            _metadata_row("pkg/a.py", element_id="elem:a", summary="a"),
            _metadata_row("pkg/b.py", element_id="elem:b", summary="b"),
        ],
    )
    previous.save("prev")

    current = _disk_store_with_vector_config(tmp_path, shard_storage="npy")
    current.initialize(3)
    current.add_vectors(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=np.float32),
        [
            _metadata_row("pkg/a.py", element_id="elem:a", summary="a"),
            _metadata_row("pkg/b.py", element_id="elem:b", summary="b2"),
        ],
    )
    monkeypatch.setattr(
        current,
        "_load_vector_shard_arrays",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("publication should read sequence shards only")
        ),
    )

    stats = current.save_incremental(
        "next",
        previous_name="prev",
        reusable_path_keys={"pkg/a.py"},
    )

    assert stats["vector_shards_reused"] == 1
    assert stats["vector_shards_written"] == 1


def test_vector_store_lazy_shard_search_avoids_eager_vector_payload_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _disk_store_with_vector_config(tmp_path, shard_storage="npy")
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
                "lazy_shard_search": True,
                "shard_storage": "npy",
            }
        }
    )
    monkeypatch.setattr(
        loaded,
        "load_vector_payload",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("lazy shard search should not eager-load vectors")
        ),
    )

    assert loaded.load("repo") is True
    assert loaded._vector_rows is None
    assert loaded._vector_shard_handles is not None

    results = loaded.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), k=1)

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
