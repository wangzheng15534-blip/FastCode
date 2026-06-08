from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


class _FakeFastMCP:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def tool(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda func: func


def _server_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_mcp = types.ModuleType("mcp")
    fake_mcp_server = types.ModuleType("mcp.server")
    fake_fastmcp = types.ModuleType("mcp.server.fastmcp")
    fake_fastmcp.FastMCP = _FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_mcp_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp)
    sys.modules.pop("fastcode.mcp.server", None)
    return importlib.import_module("fastcode.mcp.server")


def test_explore_code_tool_delegates_through_facades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _server_module(monkeypatch)
    fake_facades = SimpleNamespace(
        ensure_repos_ready=MagicMock(return_value=["repo"]),
        ensure_loaded=MagicMock(return_value=True),
        query=SimpleNamespace(
            explore_code=MagicMock(
                return_value={
                    "query": "Where is auth?",
                    "snapshot_id": "snap:repo:1",
                    "freshness": {"state": "fresh"},
                    "completeness": {
                        "state": "complete",
                        "returned_snippets": 1,
                        "omitted_snippets": 0,
                    },
                    "groups": [
                        {
                            "ref_id": "g1",
                            "repo": "repo",
                            "file": "src/auth.py",
                            "snippets": [
                                {
                                    "ref_id": "e1",
                                    "type": "function",
                                    "name": "authenticate",
                                    "lines": "7-9",
                                    "score": 0.9,
                                }
                            ],
                        }
                    ],
                }
            )
        ),
    )
    monkeypatch.setattr(server, "_facades", fake_facades)

    result = server.explore_code(
        question="Where is auth?",
        repos=["/repo"],
        snapshot_id="snap:repo:1",
        detail_level="minimal",
        max_snippets=2,
    )

    fake_facades.ensure_repos_ready.assert_called_once_with(["/repo"])
    fake_facades.ensure_loaded.assert_called_once_with(["repo"])
    fake_facades.query.explore_code.assert_called_once_with(
        question="Where is auth?",
        snapshot_id="snap:repo:1",
        repo_name=None,
        ref_name=None,
        repo_filter=["repo"],
        detail_level="minimal",
        max_snippets=2,
    )
    assert "## g1 repo/src/auth.py" in result


def test_explore_code_tool_reports_readiness_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _server_module(monkeypatch)
    fake_facades = SimpleNamespace(
        ensure_repos_ready=MagicMock(return_value=[]),
        ensure_loaded=MagicMock(),
        query=SimpleNamespace(explore_code=MagicMock()),
    )
    monkeypatch.setattr(server, "_facades", fake_facades)

    result = server.explore_code(question="q", repos=["/missing"])

    assert "Error: None" in result
    fake_facades.ensure_loaded.assert_not_called()
    fake_facades.query.explore_code.assert_not_called()
