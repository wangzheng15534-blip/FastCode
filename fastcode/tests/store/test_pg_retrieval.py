from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pytest

from fastcode.store.pg_retrieval import PgRetrievalStore
from fastcode.store.records import PgRetrievalElementRecord, PgRetrievalResultRecord

pytestmark = [pytest.mark.test_double]


class _FakeCursor:
    def __init__(self, rows: Any) -> None:
        self.rows = rows
        self.execute_calls = 0

    def execute(self, sql: Any, params: dict[str, Any] | None = None) -> None:
        self.execute_calls += 1
        if self.execute_calls == 1:
            raise RuntimeError("force fallback path")

    def fetchall(self) -> Any:
        return list(self.rows)


class _StaticCursor:
    def __init__(self, rows: Any) -> None:
        self.rows = rows
        self.calls: list[tuple[Any, Any]] = []

    def execute(self, sql: Any, params: Any = ()) -> None:
        self.calls.append((sql, params))

    def fetchall(self) -> Any:
        return list(self.rows)


class _RecordingCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def execute(self, sql: Any, params: Any = ()) -> None:
        self.calls.append((sql, params))

    def executemany(self, sql: Any, params_seq: Any) -> None:
        self.calls.append((sql, list(params_seq)))


class _FakeConn:
    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    def cursor(self) -> Any:
        return self._cursor

    def commit(self) -> None:
        return None

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> bool:
        return False


class _FakeDBRuntime:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def connect(self) -> Any:
        return self._conn


class _ExplodingMetadata:
    def __str__(self) -> str:
        raise AssertionError("metadata for unreturned fallback row was materialized")


class _OpaqueValue:
    def __repr__(self) -> str:
        return "<opaque>"


def test_semantic_fallback_rechecks_repo_and_element_type_filters_double():
    rows = [
        ({"id": "ok", "type": "design_document", "repo_name": "repoA"}, [1.0, 0.0]),
        ({"id": "wrong_type", "type": "function", "repo_name": "repoA"}, [1.0, 0.0]),
        (
            {"id": "wrong_repo", "type": "design_document", "repo_name": "repoB"},
            [1.0, 0.0],
        ),
    ]
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(_FakeCursor(rows)))

    out = store.semantic_search(
        snapshot_id="snap:1",
        query_embedding=[1.0, 0.0],
        repo_filter=["repoA"],
        element_types=["design_document"],
        top_k=5,
    )
    assert len(out) == 1
    assert out[0][0]["id"] == "ok"


def test_semantic_fallback_only_materializes_ranked_metadata_double():
    rows = [
        (
            json.dumps({"id": "best", "type": "function", "repo_name": "repoA"}),
            None,
            [1.0, 0.0],
            "repoA",
            "function",
        ),
        (_ExplodingMetadata(), None, [0.0, 1.0], "repoA", "function"),
    ]
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(_FakeCursor(rows)))

    out = store.semantic_search(
        snapshot_id="snap:1",
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        repo_filter=["repoA"],
        element_types=["function"],
        top_k=1,
    )

    assert out == [({"id": "best", "type": "function", "repo_name": "repoA"}, 1.0)]


def test_semantic_search_filters_same_dimension_stale_embedding_fingerprint_double():
    stale_fingerprint = {
        "version": 2,
        "provider": "test",
        "model": "old",
        "dimension": 2,
        "text_schema_version": 1,
    }
    current_fingerprint = {**stale_fingerprint, "model": "current"}
    cursor = _StaticCursor(
        [
            (
                json.dumps(
                    {
                        "id": "stale",
                        "type": "function",
                        "repo_name": "repoA",
                        "embedding_fingerprint": stale_fingerprint,
                    }
                ),
                0.99,
            ),
            (
                json.dumps(
                    {
                        "id": "current",
                        "type": "function",
                        "repo_name": "repoA",
                        "metadata": {
                            "embedding_fingerprint": current_fingerprint,
                        },
                    }
                ),
                0.2,
            ),
        ]
    )
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(cursor))

    out = store.semantic_search(
        snapshot_id="snap:1",
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=1,
        query_embedding_fingerprint=current_fingerprint,
    )

    assert [metadata["id"] for metadata, _score in out] == ["current"]


def test_semantic_search_records_return_typed_pg_results_double():
    cursor = _StaticCursor(
        [
            (
                json.dumps(
                    {
                        "id": "elem:typed",
                        "type": "function",
                        "repo_name": "repoA",
                        "relative_path": "pkg/a.py",
                        "metadata": {"embedding_text_hash": "hash"},
                    }
                ),
                0.75,
            ),
        ]
    )
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(cursor))

    out = store.semantic_search_records(
        snapshot_id="snap:1",
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=1,
    )

    assert len(out) == 1
    assert isinstance(out[0], PgRetrievalResultRecord)
    assert out[0].score == pytest.approx(0.75)
    assert out[0].element.id == "elem:typed"
    assert out[0].element.element_type == "function"
    assert out[0].element.repo_name == "repoA"
    assert out[0].element.metadata["embedding_text_hash"] == "hash"


def test_semantic_search_compatibility_serializes_records_explicitly_double(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = _StaticCursor(
        [
            (
                json.dumps(
                    {
                        "id": "elem:compat",
                        "type": "function",
                        "repo_name": "repoA",
                        "metadata": {"embedding_text_hash": "hash"},
                    }
                ),
                0.5,
            ),
        ]
    )
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(cursor))

    def _boom_element(_: PgRetrievalElementRecord) -> dict[str, Any]:
        raise AssertionError(
            "pg retrieval compatibility must not call element.to_dict()"
        )

    def _boom_result(_: PgRetrievalResultRecord) -> dict[str, Any]:
        raise AssertionError(
            "pg retrieval compatibility must not call result.to_dict()"
        )

    monkeypatch.setattr(PgRetrievalElementRecord, "to_dict", _boom_element)
    monkeypatch.setattr(PgRetrievalResultRecord, "to_dict", _boom_result)

    out = store.semantic_search(
        snapshot_id="snap:1",
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=1,
    )

    assert out == [
        (
            {
                "id": "elem:compat",
                "type": "function",
                "repo_name": "repoA",
                "metadata": {"embedding_text_hash": "hash"},
            },
            0.5,
        )
    ]


def test_semantic_fallback_filters_stale_embedding_fingerprint_double():
    stale_fingerprint = {
        "version": 2,
        "provider": "test",
        "model": "old",
        "dimension": 2,
        "text_schema_version": 1,
    }
    current_fingerprint = {**stale_fingerprint, "model": "current"}
    rows = [
        (
            json.dumps(
                {
                    "id": "stale",
                    "type": "function",
                    "repo_name": "repoA",
                    "embedding_fingerprint": stale_fingerprint,
                }
            ),
            None,
            [1.0, 0.0],
            "repoA",
            "function",
        ),
        (
            json.dumps(
                {
                    "id": "current",
                    "type": "function",
                    "repo_name": "repoA",
                    "metadata": {
                        "embedding_fingerprint": current_fingerprint,
                    },
                }
            ),
            None,
            [0.0, 1.0],
            "repoA",
            "function",
        ),
    ]
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(_FakeCursor(rows)))

    out = store.semantic_search(
        snapshot_id="snap:1",
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=1,
        query_embedding_fingerprint=current_fingerprint,
    )

    assert [metadata["id"] for metadata, _score in out] == ["current"]


def test_semantic_fallback_records_keep_ranked_payloads_typed_double():
    rows = [
        (
            json.dumps({"id": "best", "type": "function", "repo_name": "repoA"}),
            None,
            [1.0, 0.0],
            "repoA",
            "function",
        ),
        (_ExplodingMetadata(), None, [0.0, 1.0], "repoA", "function"),
    ]
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(_FakeCursor(rows)))

    out = store.semantic_search_records(
        snapshot_id="snap:1",
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        repo_filter=["repoA"],
        element_types=["function"],
        top_k=1,
    )

    assert len(out) == 1
    assert isinstance(out[0], PgRetrievalResultRecord)
    assert out[0].element.id == "best"
    assert out[0].score == pytest.approx(1.0)


def test_keyword_search_compatibility_serializes_records_explicitly_double(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = _StaticCursor(
        [
            (
                json.dumps(
                    {
                        "id": "elem:keyword",
                        "type": "function",
                        "repo_name": "repoA",
                        "metadata": {"embedding_text_hash": "hash"},
                    }
                ),
                0.25,
            ),
        ]
    )
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(cursor))

    def _boom_element(_: PgRetrievalElementRecord) -> dict[str, Any]:
        raise AssertionError(
            "pg retrieval compatibility must not call element.to_dict()"
        )

    def _boom_result(_: PgRetrievalResultRecord) -> dict[str, Any]:
        raise AssertionError(
            "pg retrieval compatibility must not call result.to_dict()"
        )

    monkeypatch.setattr(PgRetrievalElementRecord, "to_dict", _boom_element)
    monkeypatch.setattr(PgRetrievalResultRecord, "to_dict", _boom_result)

    out = store.keyword_search(
        snapshot_id="snap:1",
        query="keyword",
        top_k=1,
    )

    assert out == [
        (
            {
                "id": "elem:keyword",
                "type": "function",
                "repo_name": "repoA",
                "metadata": {"embedding_text_hash": "hash"},
            },
            0.25,
        )
    ]


def test_upsert_elements_keeps_embedding_out_of_metadata_json_double():
    cursor = _RecordingCursor()
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(cursor))

    store.upsert_elements(
        "snap:1",
        [
            {
                "id": "elem:1",
                "type": "function",
                "name": "f",
                "relative_path": "pkg/a.py",
                "language": "python",
                "summary": "summary",
                "signature": "def f()",
                "docstring": None,
                "code": "return 1",
                "repo_name": "repo",
                "embedding": np.asarray([9.0, 9.0], dtype=np.float32),
                "metadata": {
                    "repo_name": "repo",
                    "embedding": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                    "embedding_text_hash": "hash",
                    "embedding_artifact_ref": "embedding_v2_ref",
                },
            }
        ],
    )

    assert len(cursor.calls) == 2
    vector_params = cursor.calls[0][1][0]
    search_params = cursor.calls[1][1][0]
    assert isinstance(vector_params[6], np.ndarray)
    assert vector_params[6].dtype == np.float32
    assert vector_params[6].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert vector_params[7] is None

    vector_payload = json.loads(vector_params[8])
    search_payload = json.loads(search_params[7])
    for payload in (vector_payload, search_payload):
        assert "embedding" not in payload
        assert "embedding" not in payload["metadata"]
        assert payload["metadata"]["embedding_text_hash"] == "hash"
        assert payload["metadata"]["embedding_artifact_ref"] == "embedding_v2_ref"


def test_upsert_elements_accepts_root_embedding_when_metadata_lacks_one_double():
    cursor = _RecordingCursor()
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(cursor))

    store.upsert_elements(
        "snap:1",
        [
            {
                "id": "elem:root-only",
                "type": "function",
                "name": "root_only",
                "relative_path": "pkg/root_only.py",
                "language": "python",
                "summary": "summary",
                "signature": "def root_only()",
                "docstring": None,
                "code": "return 1",
                "repo_name": "repo",
                "embedding": np.asarray([0.5, 0.25], dtype=np.float32),
                "metadata": {
                    "repo_name": "repo",
                    "embedding_text_hash": "hash-root-only",
                },
            }
        ],
    )

    vector_params = cursor.calls[0][1][0]
    vector_payload = json.loads(vector_params[8])
    assert isinstance(vector_params[6], np.ndarray)
    assert vector_params[6].tolist() == pytest.approx([0.5, 0.25])
    assert vector_params[7] is None
    assert "embedding" not in vector_payload
    assert vector_payload["metadata"]["embedding_text_hash"] == "hash-root-only"


def test_upsert_elements_carries_embedding_reference_metadata_double():
    cursor = _RecordingCursor()
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(cursor))

    fingerprint = {
        "provider": "sentence-transformers",
        "model": "model-a",
        "dimension": 2,
        "prepared_text_schema_version": 2,
    }

    store.upsert_elements(
        "snap:1",
        [
            {
                "id": "elem:fingerprint",
                "type": "function",
                "name": "fingerprint",
                "relative_path": "pkg/fingerprint.py",
                "language": "python",
                "repo_name": "repo",
                "embedding": np.asarray([0.5, 0.25], dtype=np.float32),
                "embedding_artifact_ref": "embedding_v2_ref",
                "embedding_fingerprint": fingerprint,
                "embedding_text_hash": "hash-from-root",
                "metadata": {
                    "repo_name": "repo",
                    "embedding_text_hash": "stale-hash",
                },
            }
        ],
    )

    vector_payload = json.loads(cursor.calls[0][1][0][8])
    search_payload = json.loads(cursor.calls[1][1][0][7])
    for payload in (vector_payload, search_payload):
        assert "embedding" not in payload
        assert payload["embedding_artifact_ref"] == "embedding_v2_ref"
        assert payload["embedding_fingerprint"] == fingerprint
        assert payload["metadata"]["embedding_artifact_ref"] == "embedding_v2_ref"
        assert payload["metadata"]["embedding_fingerprint"] == fingerprint
        assert payload["metadata"]["embedding_text_hash"] == "hash-from-root"


def test_upsert_elements_batches_vector_and_search_rows_double():
    cursor = _RecordingCursor()
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(cursor))

    store.upsert_elements(
        "snap:1",
        [
            {
                "id": "elem:a",
                "type": "function",
                "name": "a",
                "relative_path": "pkg/a.py",
                "language": "python",
                "repo_name": "repo",
                "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
                "metadata": {"repo_name": "repo"},
            },
            {
                "id": "elem:b",
                "type": "function",
                "name": "b",
                "relative_path": "pkg/b.py",
                "language": "python",
                "repo_name": "repo",
                "embedding": np.asarray([0.0, 1.0], dtype=np.float32),
                "metadata": {"repo_name": "repo"},
            },
        ],
    )

    assert len(cursor.calls) == 2
    assert "embedding_vectors" in str(cursor.calls[0][0])
    assert "search_documents" in str(cursor.calls[1][0])
    assert len(cursor.calls[0][1]) == 2
    assert len(cursor.calls[1][1]) == 2
    assert store.last_upsert_metrics == {
        "row_count": 2,
        "batch_count": 2,
        "vector_adapter_path": "pgvector_adapter",
    }


def test_vector_parameter_falls_back_to_literal_without_pgvector_adapter_double(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fastcode.store.infrastructure.runtime as db_runtime_module

    monkeypatch.setattr(db_runtime_module, "register_vector", None)

    param = PgRetrievalStore._vector_parameter(np.asarray([0.1, 0.2], dtype=np.float32))

    assert param == "[0.10000000,0.20000000]"


def test_upsert_elements_ignores_unknown_top_level_payloads_double():
    cursor = _RecordingCursor()
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(cursor))

    store.upsert_elements(
        "snap:1",
        [
            {
                "id": "elem:opaque",
                "type": "function",
                "name": "opaque",
                "relative_path": "pkg/opaque.py",
                "language": "python",
                "summary": "summary",
                "signature": "def opaque()",
                "docstring": None,
                "code": "return 1",
                "repo_name": "repo",
                "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
                "opaque_payload": object(),
                "metadata": {
                    "repo_name": "repo",
                    "opaque_metadata": _OpaqueValue(),
                },
            }
        ],
    )

    vector_payload = json.loads(cursor.calls[0][1][0][8])
    assert "opaque_payload" not in vector_payload
    assert vector_payload["metadata"]["opaque_metadata"] == "<opaque>"


def test_upsert_elements_rejects_array_metadata_json_double():
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(_RecordingCursor()))

    with pytest.raises(ValueError, match="NumPy arrays"):
        store.upsert_elements(
            "snap:1",
            [
                {
                    "id": "elem:bad-array",
                    "type": "function",
                    "name": "bad_array",
                    "relative_path": "pkg/bad_array.py",
                    "language": "python",
                    "repo_name": "repo",
                    "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
                    "metadata": {
                        "repo_name": "repo",
                        "unexpected_array": np.asarray([0.1, 0.2], dtype=np.float32),
                    },
                }
            ],
        )


def test_upsert_elements_rejects_embedding_like_numeric_metadata_json_double():
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(_FakeConn(_RecordingCursor()))

    with pytest.raises(ValueError, match="Embedding/vector arrays"):
        store.upsert_elements(
            "snap:1",
            [
                {
                    "id": "elem:bad-vector",
                    "type": "function",
                    "name": "bad_vector",
                    "relative_path": "pkg/bad_vector.py",
                    "language": "python",
                    "repo_name": "repo",
                    "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
                    "metadata": {
                        "repo_name": "repo",
                        "embedding_vector": [0.1, 0.2],
                    },
                }
            ],
        )
