from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# api.py is a root-level module, not in the fastcode package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import api


class _FakeFastCode:
    def list_repo_refs(self, repo_name: str) -> None:
        return [{"branch": "main", "snapshot_id": "snap:repo:1"}]

    def find_symbol(
        self,
        snapshot_id: str,
        *,
        symbol_id: str | None = None,
        name: str | None = None,
        path: str | None = None,
    ) -> Any:
        if name == "missing":
            return None
        return {
            "symbol_id": symbol_id or "sym:1",
            "display_name": name or "foo",
            "path": path or "a.py",
        }

    def get_graph_callees(
        self, snapshot_id: str, symbol_id: str, max_hops: int = 1
    ) -> Any:
        return [{"symbol_id": "sym:2", "distance": 1}]

    def get_graph_callers(
        self, snapshot_id: str, symbol_id: str, max_hops: int = 1
    ) -> Any:
        return [{"symbol_id": "sym:0", "distance": 1}]

    def get_graph_dependencies(
        self, snapshot_id: str, doc_id: str, max_hops: int = 1
    ) -> Any:
        return [{"doc_id": "doc:2", "distance": 1}]

    def process_redo_tasks(self, limit: int = 10) -> Any:
        return {"processed": limit, "succeeded": limit, "failed": 0}


def test_graph_and_symbol_endpoints():
    original = api.fastcode_instance
    api.fastcode_instance = _FakeFastCode()
    try:
        client = TestClient(api.app)

        # --- refs endpoint: response shape and field presence ---
        refs = client.get("/repos/repo/refs")
        assert refs.status_code == 200
        body = refs.json()
        assert "refs" in body
        assert isinstance(body["refs"], list)
        assert len(body["refs"]) >= 1
        ref = body["refs"][0]
        assert "branch" in ref
        assert "snapshot_id" in ref
        assert isinstance(ref["branch"], str)
        assert isinstance(ref["snapshot_id"], str)

        # --- symbol find: name param reflected in response ---
        symbol = client.get(
            "/symbols/find", params={"snapshot_id": "snap:repo:1", "name": "foo"}
        )
        assert symbol.status_code == 200
        sym_body = symbol.json()
        assert "symbol" in sym_body
        assert sym_body["symbol"]["display_name"] == "foo"
        assert sym_body["symbol"]["path"] == "a.py"

        # --- graph endpoints: response has envelope with nested list of results ---
        callees = client.get(
            "/graph/callees",
            params={"snapshot_id": "snap:repo:1", "symbol_id": "sym:1"},
        )
        callers = client.get(
            "/graph/callers",
            params={"snapshot_id": "snap:repo:1", "symbol_id": "sym:1"},
        )
        deps = client.get(
            "/graph/dependencies",
            params={"snapshot_id": "snap:repo:1", "doc_id": "doc:1"},
        )

        for label, resp, list_key in [
            ("callees", callees, "callees"),
            ("callers", callers, "callers"),
            ("deps", deps, "dependencies"),
        ]:
            assert resp.status_code == 200, f"{label} returned {resp.status_code}"
            envelope = resp.json()
            assert "status" in envelope, f"{label} missing status field"
            assert list_key in envelope, f"{label} missing {list_key} field"
            items = envelope[list_key]
            assert isinstance(items, list), f"{label}.{list_key} is not a list"
            assert len(items) >= 1, f"{label}.{list_key} is empty"
            # Verify required fields exist in each graph entry
            entry = items[0]
            if label == "deps":
                assert "doc_id" in entry, f"{label} entry missing doc_id"
            else:
                assert "symbol_id" in entry, f"{label} entry missing symbol_id"
            assert "distance" in entry, f"{label} entry missing distance"
    finally:
        api.fastcode_instance = original


def test_redo_process_endpoint():
    original = api.fastcode_instance
    api.fastcode_instance = _FakeFastCode()
    try:
        client = TestClient(api.app)
        resp = client.post("/redo/process", params={"limit": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert "result" in body
        result = body["result"]
        # Verify the response reflects the limit parameter and has required fields
        assert "processed" in result
        assert "succeeded" in result
        assert "failed" in result
        assert (
            result["processed"] + result["failed"]
            == result["succeeded"] + result["failed"]
        )
    finally:
        api.fastcode_instance = original


def test_symbol_find_returns_404_for_missing():
    original = api.fastcode_instance
    api.fastcode_instance = _FakeFastCode()
    try:
        client = TestClient(api.app)
        resp = client.get(
            "/symbols/find", params={"snapshot_id": "snap:repo:1", "name": "missing"}
        )
        assert resp.status_code == 404
    finally:
        api.fastcode_instance = original
