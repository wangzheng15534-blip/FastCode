from fastapi.testclient import TestClient

import api


class _FakeFastCode:
    def list_repo_refs(self, repo_name: str):
        return [{"branch": "main", "snapshot_id": "snap:repo:1"}]

    def find_symbol(self, snapshot_id: str, *, symbol_id=None, name=None, path=None):
        if name == "missing":
            return None
        return {"symbol_id": symbol_id or "sym:1", "display_name": name or "foo", "path": path or "a.py"}

    def get_graph_callees(self, snapshot_id: str, symbol_id: str, max_hops: int = 1):
        return [{"symbol_id": "sym:2", "distance": 1}]

    def get_graph_callers(self, snapshot_id: str, symbol_id: str, max_hops: int = 1):
        return [{"symbol_id": "sym:0", "distance": 1}]

    def get_graph_dependencies(self, snapshot_id: str, doc_id: str, max_hops: int = 1):
        return [{"doc_id": "doc:2", "distance": 1}]

    def process_redo_tasks(self, limit: int = 10):
        return {"processed": limit, "succeeded": limit, "failed": 0}


def test_graph_and_symbol_endpoints():
    original = api.fastcode_instance
    api.fastcode_instance = _FakeFastCode()
    try:
        client = TestClient(api.app)
        refs = client.get("/repos/repo/refs")
        assert refs.status_code == 200
        assert refs.json()["refs"][0]["branch"] == "main"

        symbol = client.get("/symbols/find", params={"snapshot_id": "snap:repo:1", "name": "foo"})
        assert symbol.status_code == 200
        assert symbol.json()["symbol"]["display_name"] == "foo"

        callees = client.get("/graph/callees/snap:repo:1/sym:1")
        callers = client.get("/graph/callers/snap:repo:1/sym:1")
        deps = client.get("/graph/dependencies/snap:repo:1/doc:1")
        assert callees.status_code == 200
        assert callers.status_code == 200
        assert deps.status_code == 200
    finally:
        api.fastcode_instance = original


def test_redo_process_endpoint():
    original = api.fastcode_instance
    api.fastcode_instance = _FakeFastCode()
    try:
        client = TestClient(api.app)
        resp = client.post("/redo/process", params={"limit": 3})
        assert resp.status_code == 200
        assert resp.json()["result"]["processed"] == 3
    finally:
        api.fastcode_instance = original
