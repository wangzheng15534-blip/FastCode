"""Tests for main module."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fastcode.main import FastCode


# ---------------------------------------------------------------------------
# Helpers (basic / doc pipeline tests)
# ---------------------------------------------------------------------------


def _make_fastcode(
    *,
    doc_ingester_enabled: bool = True,
    storage_backend: str = "sqlite",
    graph_enabled: bool = False,
    sync_result: bool = True,
) -> Any:
    fc = FastCode.__new__(FastCode)
    fc.doc_ingester = SimpleNamespace(enabled=doc_ingester_enabled)
    fc.snapshot_store = SimpleNamespace(
        db_runtime=SimpleNamespace(backend=storage_backend)
    )
    fc.graph_runtime = SimpleNamespace(
        enabled=graph_enabled, sync_docs=lambda **_: sync_result
    )
    return fc


# --- Doc pipeline tests ---


def test_should_ingest_docs_requires_active_sink():
    fc = _make_fastcode(
        doc_ingester_enabled=True, storage_backend="sqlite", graph_enabled=False
    )
    assert fc._should_ingest_docs() is False

    fc = _make_fastcode(
        doc_ingester_enabled=True, storage_backend="postgres", graph_enabled=False
    )
    assert fc._should_ingest_docs() is True

    fc = _make_fastcode(
        doc_ingester_enabled=True, storage_backend="sqlite", graph_enabled=True
    )
    assert fc._should_ingest_docs() is True


def test_should_ingest_docs_requires_feature_flag():
    fc = _make_fastcode(
        doc_ingester_enabled=False, storage_backend="postgres", graph_enabled=True
    )
    assert fc._should_ingest_docs() is False


def test_sync_doc_overlay_records_false_return_as_warning():
    fc = _make_fastcode(graph_enabled=True, sync_result=False)
    warnings = []

    fc._sync_doc_overlay(chunks=[{"chunk_id": "c1"}], mentions=[], warnings=warnings)

    assert warnings == ["ladybug_doc_sync_failed"]


def test_sync_doc_overlay_records_exceptions_as_warning():
    fc = FastCode.__new__(FastCode)
    fc.graph_runtime = SimpleNamespace(
        enabled=True,
        sync_docs=lambda **_: (_ for _ in ()).throw(RuntimeError("db offline")),
    )
    warnings = []

    fc._sync_doc_overlay(chunks=[{"chunk_id": "c1"}], mentions=[], warnings=warnings)

    assert warnings == ["ladybug_doc_sync_failed: db offline"]


# ---------------------------------------------------------------------------
# Helpers (session prefix tests)
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.test_double]


class _FakeProjectionStore:
    """Fake projection store that records calls and returns preset data."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._layers: dict[str, dict[str, dict[str, Any]]] = {}
        self._builds: dict[str, dict[str, Any]] = {}
        # When find_by_snapshot is called, return these projection_ids
        self._snapshot_to_projection: dict[str, str | None] = {}

    def set_layer(self, projection_id: str, layer: str, data: dict[str, Any]) -> None:
        self._layers.setdefault(projection_id, {})[layer.upper()] = data

    def get_layer(self, projection_id: str, layer: str) -> dict[str, Any] | None:
        return self._layers.get(projection_id, {}).get(layer.upper())

    def get_build(self, projection_id: str) -> dict[str, Any] | None:
        return self._builds.get(projection_id)

    def _connect(self) -> Any:
        return self


class _FakeCursor:
    """Fake DB cursor for query simulation."""

    def __init__(self, rows: list[tuple[Any, ...]] | None = None) -> None:
        self._rows = rows or []
        self.executed_sql = ""
        self.executed_params: tuple[Any, ...] = ()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self.executed_sql = sql
        self.executed_params = params

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def close(self) -> None:
        pass

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _FakeConnection:
    """Fake DB connection that returns a _FakeCursor."""

    def __init__(self, cursor: _FakeCursor | None = None) -> None:
        self._cursor = cursor or _FakeCursor()

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fc_with_prefix(
    snapshot_id: str = "snap:myrepo:abc123",
    projection_id: str | None = "proj_test123abc",
    l0_data: dict[str, Any] | None = None,
    l1_data: dict[str, Any] | None = None,
    store_enabled: bool = True,
) -> tuple[Any, _FakeProjectionStore, _FakeConnection]:
    """Build a minimal FastCode instance wired with a fake projection store."""
    from fastcode import FastCode

    fc = FastCode.__new__(FastCode)

    store = _FakeProjectionStore(enabled=store_enabled)
    if projection_id and l0_data is not None:
        store.set_layer(projection_id, "L0", l0_data)
    if projection_id and l1_data is not None:
        store.set_layer(projection_id, "L1", l1_data)

    fc.projection_store = store

    # Build a fake connection that returns the projection_id for the query
    cursor = _FakeCursor(rows=[(projection_id,)] if projection_id else [])
    conn = _FakeConnection(cursor)
    fc.projection_store._connect = MagicMock(return_value=conn)

    return fc, store, conn


# ---------------------------------------------------------------------------
# Unit tests for FastCode.get_session_prefix
# ---------------------------------------------------------------------------


def test_get_session_prefix_returns_l0_and_l1_double():
    fc, _store, _conn = _make_fc_with_prefix(
        snapshot_id="snap:repo:abc",
        projection_id="proj_abc123",
        l0_data={"layer": "L0", "summary": "Architectural overview"},
        l1_data={"layer": "L1", "summary": "Navigation structure"},
    )

    result = fc.get_session_prefix("snap:repo:abc")

    assert result["snapshot_id"] == "snap:repo:abc"
    assert result["projection_id"] == "proj_abc123"
    assert result["l0"]["layer"] == "L0"
    assert result["l1"]["layer"] == "L1"
    assert "error" not in result


def test_get_session_prefix_not_found_double():
    fc, _store, _conn = _make_fc_with_prefix(
        snapshot_id="snap:repo:missing",
        projection_id=None,
    )

    result = fc.get_session_prefix("snap:repo:missing")

    assert result["snapshot_id"] == "snap:repo:missing"
    assert result["l0"] is None
    assert result["l1"] is None
    assert "error" in result
    assert "no snapshot-scoped projection found" in result["error"]


def test_get_session_prefix_no_layers_double():
    """Projection exists in builds but has no L0/L1 data."""
    fc, _store, _conn = _make_fc_with_prefix(
        snapshot_id="snap:repo:empty",
        projection_id="proj_empty123",
        l0_data=None,
        l1_data=None,
    )

    result = fc.get_session_prefix("snap:repo:empty")

    assert result["snapshot_id"] == "snap:repo:empty"
    assert result["projection_id"] == "proj_empty123"
    assert result["l0"] is None
    assert result["l1"] is None
    assert "no L0 or L1 layers" in result["error"]


def test_get_session_prefix_store_disabled_double():
    fc, _store, _conn = _make_fc_with_prefix(store_enabled=False)

    with pytest.raises(RuntimeError, match="projection store is not configured"):
        fc.get_session_prefix("snap:repo:abc")


def test_get_session_prefix_l0_only_double():
    fc, _store, _conn = _make_fc_with_prefix(
        snapshot_id="snap:repo:partial",
        projection_id="proj_partial123",
        l0_data={"layer": "L0", "summary": "Overview only"},
        l1_data=None,
    )

    result = fc.get_session_prefix("snap:repo:partial")

    # L0 present, L1 missing -- still succeeds (not both None)
    assert result["l0"] is not None
    assert result["l1"] is None
    assert "error" not in result


def test_get_session_prefix_l1_only_double():
    fc, _store, _conn = _make_fc_with_prefix(
        snapshot_id="snap:repo:partial2",
        projection_id="proj_partial456",
        l0_data=None,
        l1_data={"layer": "L1", "summary": "Nav only"},
    )

    result = fc.get_session_prefix("snap:repo:partial2")

    assert result["l0"] is None
    assert result["l1"] is not None
    assert "error" not in result


def test_get_session_prefix_uses_latest_projection_double():
    """When multiple projections exist, it should pick the first (latest by updated_at)."""
    fc, _store, _conn = _make_fc_with_prefix(
        snapshot_id="snap:repo:multi",
        projection_id="proj_latest",
        l0_data={"layer": "L0", "summary": "Latest"},
        l1_data={"layer": "L1", "summary": "Latest nav"},
    )

    result = fc.get_session_prefix("snap:repo:multi")

    assert result["projection_id"] == "proj_latest"
    assert result["l0"]["summary"] == "Latest"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


def test_api_prefix_endpoint_success_double():
    """GET /projection/snapshot/{snapshot_id}/prefix returns 200 with L0+L1."""
    from fastapi.testclient import TestClient

    fc, _store, _conn = _make_fc_with_prefix(
        snapshot_id="snap:repo:api_test",
        projection_id="proj_api123",
        l0_data={"layer": "L0", "summary": "API overview"},
        l1_data={"layer": "L1", "summary": "API nav"},
    )

    from fastcode import api as api_mod

    with patch.object(api_mod, "_ensure_fastcode_initialized", return_value=fc):
        client = TestClient(api_mod.app)
        resp = client.get("/projection/snapshot/snap:repo:api_test/prefix")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["result"]["snapshot_id"] == "snap:repo:api_test"
    assert body["result"]["l0"]["summary"] == "API overview"
    assert body["result"]["l1"]["summary"] == "API nav"


def test_api_prefix_endpoint_not_found_double():
    """GET /projection/snapshot/{snapshot_id}/prefix returns 404 when no projection."""
    from fastapi.testclient import TestClient

    fc, _store, _conn = _make_fc_with_prefix(
        snapshot_id="snap:repo:notfound",
        projection_id=None,
    )

    from fastcode import api as api_mod

    with patch.object(api_mod, "_ensure_fastcode_initialized", return_value=fc):
        client = TestClient(api_mod.app)
        resp = client.get("/projection/snapshot/snap:repo:notfound/prefix")

    assert resp.status_code == 404
    body = resp.json()
    assert "no snapshot-scoped projection found" in body["detail"]


def test_api_prefix_endpoint_existing_layer_route_unaffected_double():
    """Existing /projection/{projection_id}/{layer} route still works."""
    from fastapi.testclient import TestClient

    fc, store, _conn = _make_fc_with_prefix(
        snapshot_id="snap:repo:existing",
        projection_id="proj_exist123",
        l0_data={"layer": "L0", "summary": "Existing L0"},
        l1_data={"layer": "L1", "summary": "Existing L1"},
    )

    store._builds["proj_exist123"] = {
        "projection_id": "proj_exist123",
        "snapshot_id": "snap:repo:existing",
        "status": "ready",
    }

    def fake_get_projection_layer(projection_id: str, layer: str) -> dict[str, Any]:
        return {
            "projection_id": projection_id,
            "layer": layer.upper(),
            "node": store.get_layer(projection_id, layer),
            "build": store.get_build(projection_id),
        }

    fc.get_projection_layer = fake_get_projection_layer

    from fastcode import api as api_mod

    with patch.object(api_mod, "_ensure_fastcode_initialized", return_value=fc):
        client = TestClient(api_mod.app)
        resp = client.get("/projection/proj_exist123/L0")

    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["layer"] == "L0"


# ---------------------------------------------------------------------------
# MCP tool tests
# ---------------------------------------------------------------------------


def test_mcp_get_session_prefix_success_double():
    """MCP get_session_prefix tool returns found=True with L0+L1."""
    from fastcode import mcp_server as mcp_mod

    fc, _store, _conn = _make_fc_with_prefix(
        snapshot_id="snap:repo:mcp_test",
        projection_id="proj_mcp123",
        l0_data={"layer": "L0", "summary": "MCP overview"},
        l1_data={"layer": "L1", "summary": "MCP nav"},
    )

    with patch.object(mcp_mod, "_get_fastcode", return_value=fc):
        result_str = mcp_mod.get_session_prefix("snap:repo:mcp_test")

    result = json.loads(result_str)
    assert result["found"] is True
    assert result["snapshot_id"] == "snap:repo:mcp_test"
    assert result["l0"]["summary"] == "MCP overview"
    assert result["l1"]["summary"] == "MCP nav"


def test_mcp_get_session_prefix_not_found_double():
    """MCP get_session_prefix tool returns found=False when no projection."""
    from fastcode import mcp_server as mcp_mod

    fc, _store, _conn = _make_fc_with_prefix(
        snapshot_id="snap:repo:mcp_missing",
        projection_id=None,
    )

    with patch.object(mcp_mod, "_get_fastcode", return_value=fc):
        result_str = mcp_mod.get_session_prefix("snap:repo:mcp_missing")

    result = json.loads(result_str)
    assert result["found"] is False
    assert "error" in result


def test_mcp_get_session_prefix_exception_double():
    """MCP get_session_prefix tool returns found=False on exception."""
    from fastcode import mcp_server as mcp_mod

    # The error comes from get_session_prefix raising, not from _get_fastcode.
    # Mock _get_fastcode to return a fake fc, then make get_session_prefix raise.
    fc, _store, _conn = _make_fc_with_prefix(store_enabled=False)
    with patch.object(mcp_mod, "_get_fastcode", return_value=fc):
        result_str = mcp_mod.get_session_prefix("snap:repo:err")

    result = json.loads(result_str)
    assert result["found"] is False
    assert "projection store is not configured" in result["error"]
