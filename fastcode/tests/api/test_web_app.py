from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import fastcode.api.web as web_app
from fastcode.utils.archive import UnsafeArchiveError


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_ref:
        for name, payload in entries.items():
            zip_ref.writestr(name, payload)
    return buffer.getvalue()


class _NoDictSource:
    def __init__(self) -> None:
        self.repository = "repo"
        self.relative_path = "src/config.py"
        self.name = "load_config"
        self.type = "function"
        self.lines = "11-24"
        self.score = 0.75

    def to_dict(self) -> dict[str, Any]:
        raise AssertionError("web source serialization must not call to_dict()")


class _NoDictTurn:
    def __init__(self) -> None:
        self.session_id = "abcd1234"
        self.turn_number = 1
        self.timestamp = 111.25
        self.query = "Where is config loaded?"
        self.answer = "src/config.py"
        self.summary = "Config loader identified"
        self.retrieved_elements = [_NoDictSource()]
        self.metadata = {
            "intent": "where",
            "keywords": ("config", "loader"),
            "repo_filter": ("repo",),
            "multi_turn": True,
        }

    def to_dict(self) -> dict[str, Any]:
        raise AssertionError("web history serialization must not call to_dict()")


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
        self.cache_manager = SimpleNamespace(
            clear=lambda: True,
            get_session_index_record=lambda session_id: SimpleNamespace(
                multi_turn=True
            ),
        )
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

    def get_session_history(self, session_id: str) -> list[Any]:
        self.calls.append(("get_session_history", (session_id,), {}))
        return [_NoDictTurn()]

    def upload_repository_zip(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        self.calls.append(("upload_repository_zip", (file_bytes, filename), {}))
        return {
            "status": "success",
            "message": "uploaded",
            "repo_info": {"name": "repo"},
            "repo_path": "/tmp/repo",
        }

    def load_and_index(
        self, source: str, is_url: bool | None = None, *, force: bool = False
    ) -> dict[str, Any]:
        self.calls.append(("load_and_index", (source, is_url), {"force": force}))
        return {
            "status": "success",
            "message": "Repository loaded and indexed successfully",
            "summary": "summary",
        }

    def upload_and_index(
        self, file_bytes: bytes, filename: str, *, force: bool = False
    ) -> dict[str, Any]:
        self.calls.append(
            ("upload_and_index", (file_bytes, filename), {"force": force})
        )
        return {
            "status": "success",
            "message": "uploaded and indexed",
            "summary": "summary",
        }

    def clear_cache(self) -> bool:
        self.calls.append(("clear_cache", (), {}))
        return True

    def refresh_index_cache(self) -> list[dict[str, Any]]:
        self.calls.append(("refresh_index_cache", (), {}))
        return [{"repo_name": "repo"}]

    def get_status_info(self, *, full_scan: bool = False) -> dict[str, Any]:
        self.calls.append(("get_status_info", (), {"full_scan": full_scan}))
        return {
            "repo_loaded": self.repo_loaded,
            "repo_indexed": self.repo_indexed,
            "repo_info": self.repo_info,
            "multi_repo_mode": False,
            "storage_backend": "sqlite",
            "retrieval_backend": "local",
            "graph_expansion_backend": "graph_builder",
            "available_repositories": self.list_repositories(),
            "loaded_repositories": self.list_repositories(),
        }


async def _run_inline(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    return func(*args, **kwargs)


def test_load_endpoint_offloads_blocking_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()
    monkeypatch.setattr(web_app.app.state, "fastcode", fake, raising=False)
    monkeypatch.setattr(web_app.asyncio, "to_thread", _run_inline)

    client = TestClient(web_app.app)
    response = client.post("/api/load", json={"source": "/tmp/repo", "is_url": False})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["repo_info"] == {"name": "repo"}
    assert fake.calls == [("load_repository", ("/tmp/repo", False), {})]


def test_load_and_index_endpoint_delegates_to_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()
    offloaded: list[Any] = []

    async def record_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        offloaded.append(func)
        return func(*args, **kwargs)

    monkeypatch.setattr(web_app.app.state, "fastcode", fake, raising=False)
    monkeypatch.setattr(web_app.asyncio, "to_thread", record_to_thread)

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
        ("load_and_index", ("/tmp/repo", False), {"force": True}),
    ]
    assert offloaded == [fake.load_and_index]


def test_query_endpoint_offloads_blocking_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()
    monkeypatch.setattr(web_app.app.state, "fastcode", fake, raising=False)
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


def test_query_endpoint_serializes_sources_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()
    monkeypatch.setattr(web_app.app.state, "fastcode", fake, raising=False)
    monkeypatch.setattr(web_app.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(web_app.uuid, "uuid4", lambda: "abcd1234-uuid")

    def query_with_object_source(
        question: str,
        filters: dict[str, Any] | None = None,
        *,
        repo_filter: list[str] | None = None,
        session_id: str | None = None,
        enable_multi_turn: bool | None = None,
    ) -> dict[str, Any]:
        fake.calls.append(
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
            "sources": [_NoDictSource()],
        }

    fake.query = query_with_object_source

    client = TestClient(web_app.app)
    response = client.post(
        "/api/query",
        json={"question": "where is x?", "filters": {"snapshot_id": "snap:1"}},
    )

    assert response.status_code == 200
    assert response.json()["sources"] == [
        {
            "repository": "repo",
            "repo": "repo",
            "file": "src/config.py",
            "name": "load_config",
            "type": "function",
            "lines": "11-24",
            "start_line": 11,
            "end_line": 24,
            "score": 0.75,
        }
    ]


def test_session_endpoint_serializes_history_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()
    monkeypatch.setattr(web_app.app.state, "fastcode", fake, raising=False)

    client = TestClient(web_app.app)
    response = client.get("/api/session/abcd1234")

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "session_id": "abcd1234",
        "history": [
            {
                "session_id": "abcd1234",
                "turn_number": 1,
                "timestamp": 111.25,
                "query": "Where is config loaded?",
                "answer": "src/config.py",
                "summary": "Config loader identified",
                "retrieved_elements": [
                    {
                        "repository": "repo",
                        "repo": "repo",
                        "file": "src/config.py",
                        "name": "load_config",
                        "type": "function",
                        "lines": "11-24",
                        "start_line": 11,
                        "end_line": 24,
                        "score": 0.75,
                    }
                ],
                "metadata": {
                    "intent": "where",
                    "keywords": ["config", "loader"],
                    "repo_filter": ["repo"],
                    "multi_turn": True,
                },
            }
        ],
        "multi_turn": True,
    }


def test_upload_zip_endpoint_delegates_to_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()
    offloaded: list[Any] = []

    async def record_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        offloaded.append(func)
        return func(*args, **kwargs)

    monkeypatch.setattr(web_app.app.state, "fastcode", fake, raising=False)
    monkeypatch.setattr(web_app.asyncio, "to_thread", record_to_thread)

    client = TestClient(web_app.app)
    response = client.post(
        "/api/upload-zip",
        files={"file": ("repo.zip", b"zip-data", "application/zip")},
    )

    assert response.status_code == 200
    assert response.json()["repo_path"] == "/tmp/repo"
    assert offloaded == [fake.upload_repository_zip]
    assert fake.calls == [("upload_repository_zip", (b"zip-data", "repo.zip"), {})]


@pytest.mark.parametrize("endpoint", ["/api/upload-zip", "/api/upload-and-index"])
def test_upload_endpoints_reject_path_traversal_zip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    endpoint: str,
) -> None:
    fake = _FakeFastCode()

    # Simulate the UnsafeArchiveError that FastCode facade would raise
    # when it detects path traversal in the ZIP archive
    def raise_unsafe_archive(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise UnsafeArchiveError("unsafe path detected")

    if endpoint == "/api/upload-zip":
        fake.upload_repository_zip = raise_unsafe_archive  # type: ignore[assignment]
    else:
        fake.upload_and_index = raise_unsafe_archive  # type: ignore[assignment]

    monkeypatch.setattr(web_app.app.state, "fastcode", fake, raising=False)
    monkeypatch.setattr(web_app.asyncio, "to_thread", _run_inline)

    client = TestClient(web_app.app)
    response = client.post(
        endpoint,
        files={
            "file": (
                "repo.zip",
                _zip_bytes({"../escape.py": b"bad"}),
                "application/zip",
            )
        },
    )

    assert response.status_code == 400
    assert "unsafe path" in response.json()["detail"]
    assert fake.calls == []


def test_upload_and_index_endpoint_delegates_to_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()
    offloaded: list[Any] = []

    async def record_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        offloaded.append(func)
        return func(*args, **kwargs)

    monkeypatch.setattr(web_app.app.state, "fastcode", fake, raising=False)
    monkeypatch.setattr(web_app.asyncio, "to_thread", record_to_thread)

    client = TestClient(web_app.app)
    response = client.post(
        "/api/upload-and-index?force=true",
        files={"file": ("repo.zip", b"zip-data", "application/zip")},
    )

    assert response.status_code == 200
    assert response.json()["summary"] == "summary"
    assert offloaded == [fake.upload_and_index]
    assert fake.calls == [
        ("upload_and_index", (b"zip-data", "repo.zip"), {"force": True})
    ]


def test_mutating_maintenance_endpoints_offload_blocking_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()
    offloaded: list[Any] = []

    async def record_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        offloaded.append(func)
        return func(*args, **kwargs)

    monkeypatch.setattr(web_app.app.state, "fastcode", fake, raising=False)
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
        fake.clear_cache,
        fake.refresh_index_cache,
    ]
