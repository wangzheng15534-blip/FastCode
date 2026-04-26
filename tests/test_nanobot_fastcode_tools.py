"""
Tests for nanobot FastCode integration tools.

Each test verifies that the tool sends the correct HTTP request to the
FastCode backend API and formats the response properly.

Uses httpx.MockTransport for lightweight mocking — no running backend required.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

# Add nanobot directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nanobot"))

from nanobot.agent.tools.fastcode import (
    FastCodeBuildProjectionTool,
    FastCodeCallChainTool,
    FastCodeIndexRunTool,
    FastCodeListReposTool,
    FastCodeLoadRepoTool,
    FastCodeQueryTool,
    FastCodeSearchSymbolTool,
    FastCodeSessionTool,
    FastCodeStatusTool,
    FastCodeUploadRepoTool,
    create_all_tools,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockTransport(httpx.MockTransport):
    """Routes requests to a handler registry. Tracks which handlers were called."""

    def __init__(self, handlers: dict[str, Callable] | None = None):
        self._handlers: dict[str, Callable] = handlers or {}
        self.called_keys: set[str] = set()
        super().__init__(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        key = f"{request.method} {request.url.path}"
        handler = self._handlers.get(key)
        if handler is None:
            return httpx.Response(404, text=f"No handler for {key}")
        self.called_keys.add(key)
        return handler(request)


def _json_response(status_code: int = 200, body: Any = None) -> httpx.Response:
    return httpx.Response(status_code, json=body or {})


def _make_client(transport: _MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport, base_url="http://test:8001")


def _patch_client(client: httpx.AsyncClient):
    """Return a patch context that makes httpx.AsyncClient yield the given client."""
    mock_instance = AsyncMock()
    mock_instance.__aenter__ = AsyncMock(return_value=client)
    mock_instance.__aexit__ = AsyncMock(return_value=False)
    return patch(
        "nanobot.agent.tools.fastcode.httpx.AsyncClient", return_value=mock_instance
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def api_url() -> str:
    return "http://test:8001"


# ---------------------------------------------------------------------------
# Tool 1: FastCodeLoadRepoTool
# ---------------------------------------------------------------------------


class TestFastCodeLoadRepoTool:
    @pytest.mark.asyncio
    async def test_load_repo_posts_to_load_and_index(self, api_url):
        tool = FastCodeLoadRepoTool(api_url=api_url)
        payload = {"source": "https://github.com/user/repo", "is_url": True}

        request_bodies: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            request_bodies.append(body)
            return _json_response(
                body={
                    "status": "success",
                    "message": "Repository loaded and indexed",
                    "summary": {
                        "total_files": 42,
                        "total_elements": 350,
                        "languages": ["Python", "JavaScript"],
                    },
                }
            )

        transport = _MockTransport({"POST /load-and-index": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute(
                    source="https://github.com/user/repo", is_url=True
                )

        # Verify handler was called and received correct payload
        assert "POST /load-and-index" in transport.called_keys
        assert len(request_bodies) == 1
        assert request_bodies[0] == payload
        # Verify result contains expected content
        assert "Repository loaded and indexed" in result
        assert "Files: 42" in result
        assert "Code elements: 350" in result

    @pytest.mark.asyncio
    async def test_load_repo_handles_connection_error(self, api_url):
        tool = FastCodeLoadRepoTool(api_url=api_url)

        with patch("nanobot.agent.tools.fastcode.httpx.AsyncClient") as mock_cls:
            mock_cls.side_effect = httpx.ConnectError("connection refused")
            result = await tool.execute(source="https://github.com/user/repo")

        assert "Cannot connect" in result


# ---------------------------------------------------------------------------
# Tool 2: FastCodeQueryTool
# ---------------------------------------------------------------------------


class TestFastCodeQueryTool:
    @pytest.mark.asyncio
    async def test_query_posts_question(self, api_url):
        tool = FastCodeQueryTool(api_url=api_url)

        request_bodies: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            request_bodies.append(body)
            return _json_response(
                body={
                    "answer": "Auth uses JWT tokens.",
                    "query": "How does auth work?",
                    "context_elements": 5,
                    "sources": [
                        {
                            "name": "auth.py",
                            "relative_path": "src/auth.py",
                            "type": "file",
                        },
                    ],
                    "session_id": "abc123",
                    "total_tokens": 150,
                }
            )

        transport = _MockTransport({"POST /query": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute(question="How does auth work?")

        assert "POST /query" in transport.called_keys
        assert len(request_bodies) == 1
        assert request_bodies[0]["question"] == "How does auth work?"
        assert request_bodies[0]["multi_turn"] is True
        assert "JWT tokens" in result
        assert "auth.py" in result
        assert "[Session: abc123]" in result

    @pytest.mark.asyncio
    async def test_query_returns_no_repo_error_on_400(self, api_url):
        tool = FastCodeQueryTool(api_url=api_url)

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="No repository indexed")

        transport = _MockTransport({"POST /query": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute(question="test")

        assert "No repository indexed" in result


# ---------------------------------------------------------------------------
# Tool 3: FastCodeListReposTool
# ---------------------------------------------------------------------------


class TestFastCodeListReposTool:
    @pytest.mark.asyncio
    async def test_list_repos_formats_available_and_loaded(self, api_url):
        tool = FastCodeListReposTool(api_url=api_url)

        def handler(_request: httpx.Request) -> httpx.Response:
            return _json_response(
                body={
                    "available": [
                        {"name": "repo-a", "size_mb": 12.5},
                    ],
                    "loaded": [
                        {"name": "repo-b", "total_elements": 200},
                    ],
                }
            )

        transport = _MockTransport({"GET /repositories": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute()

        assert "Loaded Repositories (1)" in result
        assert "repo-b" in result
        assert "Available on Disk (1)" in result
        assert "repo-a" in result

    @pytest.mark.asyncio
    async def test_list_repos_shows_message_when_empty(self, api_url):
        tool = FastCodeListReposTool(api_url=api_url)

        def handler(_request: httpx.Request) -> httpx.Response:
            return _json_response(body={"available": [], "loaded": []})

        transport = _MockTransport({"GET /repositories": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute()

        assert "No repositories found" in result


# ---------------------------------------------------------------------------
# Tool 4: FastCodeStatusTool
# ---------------------------------------------------------------------------


class TestFastCodeStatusTool:
    @pytest.mark.asyncio
    async def test_status_formats_system_info(self, api_url):
        tool = FastCodeStatusTool(api_url=api_url)

        def handler(_request: httpx.Request) -> httpx.Response:
            return _json_response(
                body={
                    "status": "ready",
                    "repo_loaded": True,
                    "repo_indexed": True,
                    "repo_info": {"name": "fastcode"},
                    "available_repositories": [{"name": "fastcode"}],
                    "loaded_repositories": [{"name": "fastcode"}],
                }
            )

        transport = _MockTransport({"GET /status": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute()

        assert "ready" in result
        assert "Yes" in result
        assert "fastcode" in result


# ---------------------------------------------------------------------------
# Tool 5: FastCodeSessionTool
# ---------------------------------------------------------------------------


class TestFastCodeSessionTool:
    @pytest.mark.asyncio
    async def test_session_new(self, api_url):
        tool = FastCodeSessionTool(api_url=api_url)

        def handler(_request: httpx.Request) -> httpx.Response:
            return _json_response(body={"session_id": "new123"})

        transport = _MockTransport({"POST /new-session": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute(action="new")

        assert "new123" in result

    @pytest.mark.asyncio
    async def test_session_delete(self, api_url):
        tool = FastCodeSessionTool(api_url=api_url)

        def handler(_request: httpx.Request) -> httpx.Response:
            return _json_response(body={"status": "success"})

        transport = _MockTransport({"DELETE /session/s1": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute(action="delete", session_id="s1")

        assert "deleted" in result.lower()

    @pytest.mark.asyncio
    async def test_session_unknown_action(self, api_url):
        tool = FastCodeSessionTool(api_url=api_url)
        result = await tool.execute(action="invalid")
        assert "Unknown action" in result


# ---------------------------------------------------------------------------
# Tool 6: FastCodeSearchSymbolTool (NEW)
# ---------------------------------------------------------------------------


class TestFastCodeSearchSymbolTool:
    @pytest.mark.asyncio
    async def test_search_symbol_by_name(self, api_url):
        tool = FastCodeSearchSymbolTool(api_url=api_url)

        captured_params: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_params.append(dict(request.url.params))
            return _json_response(
                body={
                    "status": "success",
                    "symbol": {
                        "symbol_id": "sym_001",
                        "display_name": "FastCode",
                        "kind": "class",
                        "qualified_name": "fastcode.FastCode",
                        "doc_path": "fastcode/main.py",
                        "source_set": ["ast"],
                    },
                }
            )

        transport = _MockTransport({"GET /symbols/find": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute(
                    snapshot_id="snap:myrepo:abc123",
                    name="FastCode",
                )

        assert "GET /symbols/find" in transport.called_keys
        assert len(captured_params) == 1
        assert captured_params[0]["snapshot_id"] == "snap:myrepo:abc123"
        assert captured_params[0]["name"] == "FastCode"
        assert "FastCode" in result
        assert "class" in result
        assert "fastcode/main.py" in result

    @pytest.mark.asyncio
    async def test_search_symbol_by_path(self, api_url):
        tool = FastCodeSearchSymbolTool(api_url=api_url)

        def handler(request: httpx.Request) -> httpx.Response:
            params = dict(request.url.params)
            assert params["path"] == "fastcode/retriever.py"
            assert "name" not in params
            return _json_response(
                body={
                    "status": "success",
                    "symbol": {
                        "symbol_id": "sym_002",
                        "display_name": "HybridRetriever",
                        "kind": "class",
                        "qualified_name": "fastcode.retriever.HybridRetriever",
                        "doc_path": "fastcode/retriever.py",
                    },
                }
            )

        transport = _MockTransport({"GET /symbols/find": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute(
                    snapshot_id="snap:myrepo:abc123",
                    path="fastcode/retriever.py",
                )

        assert "HybridRetriever" in result

    @pytest.mark.asyncio
    async def test_search_symbol_not_found(self, api_url):
        tool = FastCodeSearchSymbolTool(api_url=api_url)

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "Symbol not found"})

        transport = _MockTransport({"GET /symbols/find": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute(
                    snapshot_id="snap:myrepo:abc123",
                    name="nonexistent",
                )

        assert "no symbol found" in result.lower()

    @pytest.mark.asyncio
    async def test_search_symbol_requires_snapshot_id(self, api_url):
        tool = FastCodeSearchSymbolTool(api_url=api_url)
        # snapshot_id is required in schema - verify validation
        errors = tool.validate_params({})
        assert any("snapshot_id" in e for e in errors)

    @pytest.mark.asyncio
    async def test_search_symbol_requires_at_least_one_filter(self, api_url):
        tool = FastCodeSearchSymbolTool(api_url=api_url)
        result = await tool.execute(snapshot_id="snap:myrepo:abc123")
        assert "name, symbol_id, or path" in result.lower()


# ---------------------------------------------------------------------------
# Tool 7: FastCodeCallChainTool (NEW)
# ---------------------------------------------------------------------------


class TestFastCodeCallChainTool:
    @pytest.mark.asyncio
    async def test_get_callees(self, api_url):
        tool = FastCodeCallChainTool(api_url=api_url)

        captured_params: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_params.append(dict(request.url.params))
            return _json_response(
                body={
                    "status": "success",
                    "snapshot_id": captured_params[-1]["snapshot_id"],
                    "symbol_id": captured_params[-1]["symbol_id"],
                    "callees": [
                        {
                            "symbol_id": "sym_010",
                            "display_name": "helper_func",
                            "kind": "function",
                        },
                        {
                            "symbol_id": "sym_011",
                            "display_name": "parse_input",
                            "kind": "function",
                        },
                    ],
                }
            )

        transport = _MockTransport({"GET /graph/callees": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute(
                    snapshot_id="snap:myrepo:abc123",
                    symbol_id="sym_001",
                    direction="callees",
                )

        assert "GET /graph/callees" in transport.called_keys
        assert len(captured_params) == 1
        assert captured_params[0]["snapshot_id"] == "snap:myrepo:abc123"
        assert captured_params[0]["symbol_id"] == "sym_001"
        assert "helper_func" in result
        assert "parse_input" in result

    @pytest.mark.asyncio
    async def test_get_both_directions(self, api_url):
        tool = FastCodeCallChainTool(api_url=api_url)

        def callees_handler(_request: httpx.Request) -> httpx.Response:
            return _json_response(
                body={
                    "status": "success",
                    "callees": [
                        {"symbol_id": "s1", "display_name": "f1", "kind": "function"}
                    ],
                }
            )

        def callers_handler(_request: httpx.Request) -> httpx.Response:
            return _json_response(
                body={
                    "status": "success",
                    "callers": [
                        {"symbol_id": "s2", "display_name": "f2", "kind": "function"}
                    ],
                }
            )

        transport = _MockTransport(
            {
                "GET /graph/callees": callees_handler,
                "GET /graph/callers": callers_handler,
            }
        )
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute(
                    snapshot_id="snap:myrepo:abc123",
                    symbol_id="sym_001",
                    direction="both",
                )

        assert "f1" in result
        assert "f2" in result
        assert "Callees" in result
        assert "Callers" in result

    @pytest.mark.asyncio
    async def test_requires_snapshot_id_and_symbol_id(self, api_url):
        tool = FastCodeCallChainTool(api_url=api_url)
        result = await tool.execute(snapshot_id="", symbol_id="")
        assert "required" in result.lower()


# ---------------------------------------------------------------------------
# Tool 8: FastCodeBuildProjectionTool (NEW)
# ---------------------------------------------------------------------------


class TestFastCodeBuildProjectionTool:
    @pytest.mark.asyncio
    async def test_build_snapshot_projection(self, api_url):
        tool = FastCodeBuildProjectionTool(api_url=api_url)

        request_bodies: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            request_bodies.append(json.loads(request.content))
            return _json_response(
                body={
                    "status": "success",
                    "result": {
                        "projection_id": "proj_001",
                        "scope_kind": "snapshot",
                        "snapshot_id": "snap:myrepo:abc123",
                        "layers_available": ["L0", "L1", "L2"],
                    },
                }
            )

        transport = _MockTransport({"POST /projection/build": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute(
                    action="build",
                    scope_kind="snapshot",
                    snapshot_id="snap:myrepo:abc123",
                )

        assert "POST /projection/build" in transport.called_keys
        assert len(request_bodies) == 1
        assert request_bodies[0]["scope_kind"] == "snapshot"
        assert request_bodies[0]["snapshot_id"] == "snap:myrepo:abc123"
        assert "proj_001" in result
        assert "L0" in result
        assert "L1" in result
        assert "L2" in result

    @pytest.mark.asyncio
    async def test_get_projection_layer(self, api_url):
        tool = FastCodeBuildProjectionTool(api_url=api_url)

        def handler(request: httpx.Request) -> httpx.Response:
            path_parts = request.url.path.split("/")
            projection_id = path_parts[2]
            layer = path_parts[3]
            assert projection_id == "proj_001"
            assert layer == "L0"
            return _json_response(
                body={
                    "status": "success",
                    "result": {
                        "layer": "L0",
                        "summary": "FastCode is a code understanding system.",
                        "languages": ["Python"],
                        "total_files": 42,
                    },
                }
            )

        transport = _MockTransport({"GET /projection/proj_001/L0": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute(
                    action="get_layer",
                    projection_id="proj_001",
                    layer="L0",
                )

        assert "code understanding" in result

    @pytest.mark.asyncio
    async def test_invalid_action(self, api_url):
        tool = FastCodeBuildProjectionTool(api_url=api_url)
        result = await tool.execute(action="invalid")
        assert "Unknown action" in result


# ---------------------------------------------------------------------------
# Tool 9: FastCodeIndexRunTool (NEW)
# ---------------------------------------------------------------------------


class TestFastCodeIndexRunTool:
    @pytest.mark.asyncio
    async def test_run_index_pipeline(self, api_url):
        tool = FastCodeIndexRunTool(api_url=api_url)

        request_bodies: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            request_bodies.append(json.loads(request.content))
            return _json_response(
                body={
                    "status": "success",
                    "result": {
                        "run_id": "run_001",
                        "snapshot_id": "snap:user-repo:abc123",
                        "repo_name": "user-repo",
                        "documents": 42,
                        "symbols": 350,
                        "edges": 1200,
                        "published": True,
                    },
                }
            )

        transport = _MockTransport({"POST /index/run": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute(
                    source="https://github.com/user/repo",
                    is_url=True,
                    ref="main",
                )

        assert "POST /index/run" in transport.called_keys
        assert len(request_bodies) == 1
        assert request_bodies[0]["source"] == "https://github.com/user/repo"
        assert request_bodies[0]["is_url"] is True
        assert request_bodies[0]["ref"] == "main"
        assert "run_001" in result
        assert "Documents: 42" in result
        assert "Symbols: 350" in result

    @pytest.mark.asyncio
    async def test_index_run_handles_error(self, api_url):
        tool = FastCodeIndexRunTool(api_url=api_url)

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Index pipeline failed: OOM")

        transport = _MockTransport({"POST /index/run": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                result = await tool.execute(source="https://github.com/user/repo")

        assert "500" in result
        assert "Index pipeline failed" in result


# ---------------------------------------------------------------------------
# Tool 10: FastCodeUploadRepoTool (NEW)
# ---------------------------------------------------------------------------


class TestFastCodeUploadRepoTool:
    @pytest.mark.asyncio
    async def test_upload_repo_posts_file(self, api_url):
        """Upload tool sends file to POST /upload-and-index."""
        tool = FastCodeUploadRepoTool(api_url=api_url)

        captured_content_types: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_content_types.append(request.headers.get("content-type", ""))
            return _json_response(
                body={
                    "status": "success",
                    "message": "Repository uploaded and indexed",
                    "summary": {
                        "total_files": 25,
                        "total_elements": 180,
                        "languages": ["Python"],
                    },
                }
            )

        transport = _MockTransport({"POST /upload-and-index": handler})
        async with _make_client(transport) as client:
            with _patch_client(client):
                # Use a real temp file for the upload
                import os
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
                    f.write(b"PK" + b"\x00" * 100)
                    tmp_path = f.name
                try:
                    result = await tool.execute(file_path=tmp_path)
                finally:
                    os.unlink(tmp_path)

        assert "POST /upload-and-index" in transport.called_keys
        assert len(captured_content_types) == 1
        assert "multipart" in captured_content_types[0]
        assert "uploaded and indexed" in result
        assert "25" in result

    @pytest.mark.asyncio
    async def test_upload_repo_file_not_found(self, api_url):
        tool = FastCodeUploadRepoTool(api_url=api_url)
        result = await tool.execute(file_path="/nonexistent/path.zip")
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# create_all_tools registration
# ---------------------------------------------------------------------------


class TestCreateAllTools:
    def test_includes_all_ten_tools(self):
        tools = create_all_tools(api_url="http://test:8001")
        names = {t.name for t in tools}
        expected = {
            "fastcode_load_repo",
            "fastcode_query",
            "fastcode_list_repos",
            "fastcode_status",
            "fastcode_session",
            "fastcode_search_symbol",
            "fastcode_call_chain",
            "fastcode_build_projection",
            "fastcode_index_run",
            "fastcode_upload_repo",
        }
        assert names == expected

    def test_all_tools_have_required_properties(self):
        tools = create_all_tools(api_url="http://test:8001")
        for tool in tools:
            assert isinstance(tool.name, str) and tool.name
            assert isinstance(tool.description, str) and tool.description
            assert isinstance(tool.parameters, dict)
            assert hasattr(tool, "execute")
