from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fastcode import web_app


class _FakeVectorStore:
    def invalidate_scan_cache(self) -> None:
        return None

    def scan_available_indexes(self, use_cache: bool = True) -> list[dict[str, Any]]:
        return []


class _FakeFastCode:
    def __init__(self) -> None:
        self.repo_loaded = True
        self.repo_indexed = True
        self.repo_info = {"name": "repo"}
        self.vector_store = _FakeVectorStore()
        self.cache_manager = SimpleNamespace(clear=lambda: True)
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def load_repository(self, source: str, is_url: bool | None) -> None:
        self.calls.append(("load_repository", (source, is_url), {}))

    def index_repository(self, force: bool = False) -> None:
        self.calls.append(("index_repository", (), {"force": force}))

    def load_multiple_repositories(self, sources: list[dict[str, Any]]) -> None:
        self.calls.append(("load_multiple_repositories", (sources,), {}))

    def _load_multi_repo_cache(self, *, repo_names: list[str]) -> bool:
        self.calls.append(("_load_multi_repo_cache", (), {"repo_names": repo_names}))
        return True

    def list_repositories(self) -> list[dict[str, Any]]:
        return [{"repo_name": "repo"}]

    def get_repository_stats(self) -> dict[str, Any]:
        return {"repo_count": 1}

    def get_repository_summary(self) -> str:
        return "summary"

    def remove_repository(
        self, repo_name: str, *, delete_source: bool = False
    ) -> dict[str, Any]:
        self.calls.append(
            ("remove_repository", (repo_name,), {"delete_source": delete_source})
        )
        return {"deleted_files": [], "freed_mb": 0.0}

    def query(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        *,
        repo_filter: list[str] | None = None,
        session_id: str | None = None,
        enable_multi_turn: bool | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "query",
                (question, filters),
                {
                    "repo_filter": repo_filter,
                    "session_id": session_id,
                    "enable_multi_turn": enable_multi_turn,
                },
            )
        )
        return {
            "answer": "ok",
            "query": question,
            "context_elements": 1,
            "sources": [],
        }


async def _run_inline(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    return func(*args, **kwargs)


def test_load_endpoint_offloads_blocking_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()
    monkeypatch.setattr(web_app, "fastcode_instance", fake)
    monkeypatch.setattr(web_app.asyncio, "to_thread", _run_inline)

    client = TestClient(web_app.app)
    response = client.post("/api/load", json={"source": "/tmp/repo", "is_url": False})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["repo_info"] == {"name": "repo"}
    assert fake.calls == [("load_repository", ("/tmp/repo", False), {})]


def test_load_and_index_endpoint_offloads_both_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()
    monkeypatch.setattr(web_app, "fastcode_instance", fake)
    monkeypatch.setattr(web_app.asyncio, "to_thread", _run_inline)

    client = TestClient(web_app.app)
    response = client.post(
        "/api/load-and-index?force=true",
        json={"source": "/tmp/repo", "is_url": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["summary"] == "summary"
    assert fake.calls == [
        ("load_repository", ("/tmp/repo", False), {}),
        ("index_repository", (), {"force": True}),
    ]


def test_query_endpoint_offloads_blocking_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()
    monkeypatch.setattr(web_app, "fastcode_instance", fake)
    monkeypatch.setattr(web_app.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(web_app.uuid, "uuid4", lambda: "abcd1234-uuid")

    client = TestClient(web_app.app)
    response = client.post(
        "/api/query",
        json={"question": "where is x?", "filters": {"snapshot_id": "snap:1"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "ok"
    assert body["query"] == "where is x?"
    assert body["context_elements"] == 1
    assert body["session_id"] == "abcd1234"
    assert fake.calls == [
        (
            "query",
            ("where is x?", {"snapshot_id": "snap:1"}),
            {
                "repo_filter": None,
                "session_id": "abcd1234",
                "enable_multi_turn": False,
            },
        )
    ]


def test_upload_zip_endpoint_offloads_blocking_zip_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()
    offloaded: list[Any] = []

    async def record_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        offloaded.append(func)
        return func(*args, **kwargs)

    def fake_upload_sync(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "success",
            "message": "uploaded",
            "repo_info": {"name": "repo"},
            "repo_path": "/tmp/repo",
        }

    monkeypatch.setattr(web_app, "fastcode_instance", fake)
    monkeypatch.setattr(web_app.asyncio, "to_thread", record_to_thread)
    monkeypatch.setattr(web_app, "_upload_repository_zip_sync", fake_upload_sync)

    client = TestClient(web_app.app)
    response = client.post(
        "/api/upload-zip",
        files={"file": ("repo.zip", b"zip-data", "application/zip")},
    )

    assert response.status_code == 200
    assert response.json()["repo_path"] == "/tmp/repo"
    assert offloaded == [fake_upload_sync]


def test_upload_and_index_endpoint_offloads_upload_and_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()
    offloaded: list[Any] = []

    async def record_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        offloaded.append(func)
        return func(*args, **kwargs)

    def fake_upload_sync(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "success",
            "message": "uploaded",
            "repo_info": {"name": "repo"},
            "repo_path": "/tmp/repo",
        }

    monkeypatch.setattr(web_app, "fastcode_instance", fake)
    monkeypatch.setattr(web_app.asyncio, "to_thread", record_to_thread)
    monkeypatch.setattr(web_app, "_upload_repository_zip_sync", fake_upload_sync)

    client = TestClient(web_app.app)
    response = client.post(
        "/api/upload-and-index?force=true",
        files={"file": ("repo.zip", b"zip-data", "application/zip")},
    )

    assert response.status_code == 200
    assert response.json()["summary"] == "summary"
    assert offloaded == [fake_upload_sync, fake.index_repository]


def test_mutating_maintenance_endpoints_offload_blocking_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()
    offloaded: list[Any] = []

    async def record_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        offloaded.append(func)
        return func(*args, **kwargs)

    monkeypatch.setattr(web_app, "fastcode_instance", fake)
    monkeypatch.setattr(web_app.asyncio, "to_thread", record_to_thread)

    client = TestClient(web_app.app)
    delete_response = client.post(
        "/api/delete-repos",
        json={"repo_names": ["repo"], "delete_source": False},
    )
    clear_response = client.post("/api/clear-cache")
    refresh_response = client.post("/api/refresh-index-cache")

    assert delete_response.status_code == 200
    assert clear_response.status_code == 200
    assert refresh_response.status_code == 200
    assert offloaded == [
        fake.remove_repository,
        fake.cache_manager.clear,
        fake.vector_store.invalidate_scan_cache,
        fake.vector_store.scan_available_indexes,
    ]
