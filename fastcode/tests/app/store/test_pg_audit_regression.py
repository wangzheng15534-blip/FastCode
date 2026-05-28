"""PG retrieval audit regression tests — infra hardening incident guards.

Reference: incident-register-2026-05-17.yaml (PGRE-001, PGRE-002)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from fastcode.app.store.vectors.pg_retrieval import PgRetrievalStore

pytestmark = [pytest.mark.test_double]


class _RecordingCursor:
    def __init__(self, *, fail_on_second: bool = False) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self._fail_on_second = fail_on_second
        self._call_count = 0

    def executemany(self, sql: Any, params_seq: Any) -> None:
        self._call_count += 1
        if self._fail_on_second and self._call_count == 2:
            raise RuntimeError("search_documents write failed")
        self.calls.append((sql, list(params_seq)))

    def execute(self, sql: Any, params: Any = ()) -> None:
        self.calls.append((sql, params))


class _FakeConn:
    def __init__(self, cursor: Any, *, track_commit: bool = False) -> None:
        self._cursor = cursor
        self.committed = False
        self._track_commit = track_commit

    def cursor(self) -> Any:
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> bool:
        return False


class _FakeDBRuntime:
    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self.backend = "postgres"

    def connect(self) -> Any:
        return self._conn

    def supports_pgvector_adapter(self) -> bool:
        return True


def _make_store(
    cursor: Any, *, track_commit: bool = False
) -> tuple[PgRetrievalStore, _FakeConn]:
    conn = _FakeConn(cursor, track_commit=track_commit)
    store = PgRetrievalStore.__new__(PgRetrievalStore)
    store.enabled = True
    store.logger = logging.getLogger(__name__)
    store.db_runtime = _FakeDBRuntime(conn)
    return store, conn


# ---------------------------------------------------------------------------
# PGRE-001: executemany atomicity — search_documents failure propagates
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestPgUpsertAtomicity:
    """PGRE-001: verify error propagates when second executemany fails."""

    @pytest.mark.audit_finding("PGRE-001")
    def test_search_documents_failure_propagates_and_does_not_commit(
        self,
    ) -> None:
        cursor = _RecordingCursor(fail_on_second=True)
        store, conn = _make_store(cursor, track_commit=True)

        with pytest.raises(RuntimeError, match="search_documents write failed"):
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

        assert len(cursor.calls) == 1, (
            "Only embedding_vectors executemany should have succeeded"
        )
        assert "embedding_vectors" in str(cursor.calls[0][0])
        assert not conn.committed, (
            "commit must not be called when search_documents fails"
        )


# ---------------------------------------------------------------------------
# PGRE-002: last_upsert_metrics on failure
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestPgUpsertMetrics:
    """PGRE-002: verify metrics state after executemany failure."""

    @pytest.mark.audit_finding("PGRE-002")
    def test_metrics_set_error_on_executemany_failure(self) -> None:
        cursor = _RecordingCursor(fail_on_second=True)
        store, _conn = _make_store(cursor)

        store.last_upsert_metrics = {
            "row_count": 99,
            "batch_count": 2,
            "vector_adapter_path": "stale",
        }

        with pytest.raises(RuntimeError):
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
                ],
            )

        assert store.last_upsert_metrics.get("error") is True, (
            "Metrics should record error state when executemany fails"
        )
        assert store.last_upsert_metrics["row_count"] == 1
