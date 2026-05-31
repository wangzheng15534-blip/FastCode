from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import fastcode.api.routes as api
from fastcode.api.outbound import DiagnosticBundleResponse, IndexRunResponse


def _mock_request() -> MagicMock:
    req = MagicMock()
    req.app = api.app
    return req


class _FakeFastCode:
    def __init__(self) -> None:
        self.vector_store = SimpleNamespace(invalidate_scan_cache=lambda: None)
        self.run_kwargs: dict[str, Any] | None = None

    def run_index_pipeline(self, **kwargs: Any) -> dict[str, Any]:
        self.run_kwargs = kwargs
        return {
            "status": "degraded",
            "run_id": "run_1",
            "repo_name": "repo",
            "snapshot_id": "snap:repo:1",
            "artifact_key": "art_1",
            "warnings": ["semantic_resolver_runs_without_graph_upgrade_signal"],
            "pipeline_layers": [
                {
                    "name": "plain_ast_embedding",
                    "source": "tree_sitter",
                    "status": "succeeded",
                    "metrics": {"files": 3},
                    "warnings": [],
                },
                {
                    "name": "language_specific_semantic_upgrade",
                    "source": "language_specific_ast_resolvers",
                    "status": "degraded",
                    "reason": "semantic_resolver_runs_without_graph_upgrade_signal",
                    "warnings": ["semantic warning"],
                    "metrics": {"resolver_runs": 1},
                },
            ],
            "pipeline_metrics": {
                "warning_count": 1,
                "layer_statuses": {
                    "language_specific_semantic_upgrade": "degraded",
                },
            },
        }

    def build_diagnostic_bundle(self) -> dict[str, Any]:
        return {
            "schema_version": "fastcode.diagnostic_bundle.v1",
            "config_summary": {"storage": {"backend": "sqlite"}},
            "storage": {"backend": "sqlite"},
            "dependencies": {"python": [], "external_tools": []},
            "latest_index_run": {"run_id": "run_1"},
        }


def test_index_run_response_defaults_are_independent() -> None:
    first = IndexRunResponse(status="success")
    second = IndexRunResponse(status="success")

    first.warnings.append("first")
    first.pipeline_metrics["warning_count"] = 1

    assert second.warnings == []
    assert second.pipeline_metrics == {}


def test_diagnostic_bundle_response_defaults_are_independent() -> None:
    first = DiagnosticBundleResponse(status="success")
    second = DiagnosticBundleResponse(status="success")

    first.bundle["schema_version"] = "fastcode.diagnostic_bundle.v1"

    assert second.bundle == {}


def test_index_run_promotes_pipeline_and_resolver_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()

    async def _run_inline(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(api.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(api, "_fc", lambda _request: fake)

    body = asyncio.run(
        api.run_index_pipeline(
            _mock_request(),
            api.IndexRunRequest(
                source="repo",
                is_url=None,
                ref="main",
                commit="abc123",
                force=True,
                publish=False,
                enable_scip=False,
                scip_artifact_path=None,
            )
        )
    )

    assert body.status == "success"
    assert body.result["status"] == "degraded"
    assert body.index_status == "degraded"
    assert body.run_id == "run_1"
    assert body.repo_name == "repo"
    assert body.snapshot_id == "snap:repo:1"
    assert body.artifact_key == "art_1"
    assert body.warnings == ["semantic_resolver_runs_without_graph_upgrade_signal"]
    assert body.pipeline_layers[1]["name"] == "language_specific_semantic_upgrade"
    assert body.pipeline_metrics["warning_count"] == 1
    assert body.pipeline_metrics["layer_statuses"] == {
        "language_specific_semantic_upgrade": "degraded"
    }
    assert [item.model_dump() for item in body.resolver_diagnostics] == [
        {
            "name": "language_specific_semantic_upgrade",
            "source": "language_specific_ast_resolvers",
            "status": "degraded",
            "reason": "semantic_resolver_runs_without_graph_upgrade_signal",
            "warnings": ["semantic warning"],
            "metrics": {"resolver_runs": 1},
        }
    ]
    assert fake.run_kwargs == {
        "source": "repo",
        "is_url": None,
        "ref": "main",
        "commit": "abc123",
        "force": True,
        "publish": False,
        "scip_artifact_path": None,
        "enable_scip": False,
    }


def test_diagnostics_endpoint_returns_support_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFastCode()

    async def _run_inline(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(api.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(api, "_fc", lambda _request: fake)

    body = asyncio.run(api.get_diagnostics(_mock_request()))

    assert body.status == "success"
    assert body.bundle["schema_version"] == "fastcode.diagnostic_bundle.v1"
    assert body.bundle["config_summary"]["storage"]["backend"] == "sqlite"
    assert body.bundle["storage"]["backend"] == "sqlite"
    assert body.bundle["latest_index_run"]["run_id"] == "run_1"
