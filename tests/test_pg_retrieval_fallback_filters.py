import logging

from fastcode.pg_retrieval import PgRetrievalStore


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.execute_calls = 0

    def execute(self, sql, params=None):
        self.execute_calls += 1
        if self.execute_calls == 1:
            raise RuntimeError("force fallback path")

    def fetchall(self):
        return list(self.rows)


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeDBRuntime:
    def __init__(self, conn):
        self._conn = conn

    def connect(self):
        return self._conn


def test_semantic_fallback_rechecks_repo_and_element_type_filters():
    rows = [
        ({"id": "ok", "type": "design_document", "repo_name": "repoA"}, [1.0, 0.0]),
        ({"id": "wrong_type", "type": "function", "repo_name": "repoA"}, [1.0, 0.0]),
        ({"id": "wrong_repo", "type": "design_document", "repo_name": "repoB"}, [1.0, 0.0]),
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

