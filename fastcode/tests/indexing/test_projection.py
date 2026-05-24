"""Tests for projection service and session prefix."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fastcode.indexing.projection import ProjectionService
from fastcode.ir.projection import ProjectionBuildResult, ProjectionScope
from fastcode.ir.types import IRCodeUnit, IRSnapshot
from fastcode.main.fastcode import FastCode
from fastcode.store.projection_contracts import (
    ProjectionBuildRecord,
    ProjectionDirtyScopeRecord,
)

pytestmark = [pytest.mark.test_double]


class _FakeProjectionStore:
    """Fake projection store that records calls and returns preset data."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._layers: dict[str, dict[str, dict[str, Any]]] = {}
        self._builds: dict[str, ProjectionBuildRecord] = {}
        # When find_by_snapshot is called, return these projection_ids
        self._snapshot_to_projection: dict[str, str | None] = {}
        self.cached_id: str | None = None
        self.saved: list[tuple[ProjectionBuildResult, str]] = []
        self.dirty_scopes: set[tuple[str, str, str]] = set()
        self.cleared: list[tuple[str, str, str]] = []

    def set_layer(self, projection_id: str, layer: str, data: dict[str, Any]) -> None:
        self._layers.setdefault(projection_id, {})[layer.upper()] = data

    def get_layer(self, projection_id: str, layer: str) -> dict[str, Any] | None:
        return self._layers.get(projection_id, {}).get(layer.upper())

    def get_build_record(self, projection_id: str) -> ProjectionBuildRecord | None:
        return self._builds.get(projection_id)

    def find_cached_projection_id(self, _scope: Any, _params_hash: str) -> str | None:
        return self.cached_id

    def is_dirty(self, snapshot_id: str, scope_kind: str, scope_key: str) -> bool:
        return (snapshot_id, scope_kind, scope_key) in self.dirty_scopes

    def clear_dirty(self, snapshot_id: str, scope_kind: str, scope_key: str) -> None:
        self.cleared.append((snapshot_id, scope_kind, scope_key))
        self.dirty_scopes.discard((snapshot_id, scope_kind, scope_key))

    def list_dirty_scope_records(
        self, snapshot_id: str
    ) -> list[ProjectionDirtyScopeRecord]:
        return [
            ProjectionDirtyScopeRecord(
                snapshot_id=dirty_snapshot_id,
                scope_kind=scope_kind,
                scope_key=scope_key,
                dirty_paths=[],
                dirty_units=[],
                dirty_package_roots=[],
                dirty_reason="test",
                created_at="2026-05-05T00:00:00+00:00",
                updated_at="2026-05-05T00:00:00+00:00",
            )
            for dirty_snapshot_id, scope_kind, scope_key in sorted(self.dirty_scopes)
            if dirty_snapshot_id == snapshot_id
        ]

    def list_build_records_for_snapshot(
        self, snapshot_id: str
    ) -> list[ProjectionBuildRecord]:
        return [
            build
            for build in self._builds.values()
            if build.snapshot_id == snapshot_id
        ]

    def save(
        self,
        result: ProjectionBuildResult,
        params_hash: str,
        *,
        scope: Any,
        coverage_paths: list[str] | None = None,
    ) -> None:
        self.saved.append((result, params_hash))
        self._builds[result.projection_id] = ProjectionBuildRecord(
            projection_id=result.projection_id,
            snapshot_id=result.snapshot_id,
            scope_kind=result.scope_kind,
            scope_key=result.scope_key,
            params_hash=params_hash,
            status="ready",
            warnings=list(result.warnings),
            created_at=result.created_at,
            updated_at=result.created_at,
            query=getattr(scope, "query", None),
            target_id=getattr(scope, "target_id", None),
            filters=getattr(scope, "filters", {}) or {},
            coverage_paths=coverage_paths or [],
            coverage_nodes=[],
        )
        self.set_layer(result.projection_id, "L0", result.l0)
        self.set_layer(result.projection_id, "L1", result.l1)
        self.set_layer(result.projection_id, "L2", result.l2_index)

    def _connect(self) -> Any:
        return self


class _FakeProjectionTransformer:
    ALGO_VERSION = "test"

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def build(self, scope: Any, **_kwargs: Any) -> ProjectionBuildResult:
        self.calls.append(scope)
        return ProjectionBuildResult(
            projection_id=f"proj_built_{len(self.calls)}",
            snapshot_id=scope.snapshot_id,
            scope_kind=scope.scope_kind,
            scope_key=scope.scope_key,
            l0={
                "layer": "L0",
                "built": len(self.calls),
                "meta": {"covers_nodes": ["unit:a"]},
            },
            l1={
                "layer": "L1",
                "built": len(self.calls),
                "meta": {"covers_nodes": ["unit:a"]},
            },
            l2_index={
                "layer": "L2",
                "built": len(self.calls),
                "meta": {"covers_nodes": ["unit:a"]},
            },
            chunks=[],
        )


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


def _make_fc_with_prefix(
    snapshot_id: str = "snap:myrepo:abc123",
    projection_id: str | None = "proj_test123abc",
    l0_data: dict[str, Any] | None = None,
    l1_data: dict[str, Any] | None = None,
    store_enabled: bool = True,
) -> tuple[Any, _FakeProjectionStore, _FakeConnection]:
    """Build a minimal FastCode instance wired with a fake projection store."""
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

    # Wire projection_service so facade methods can delegate
    fc.projection_service = ProjectionService(
        config={},
        logger=logging.getLogger("test"),
        projection_store=store,
        projection_transformer=None,
        snapshot_store=None,
        manifest_store=None,
        load_artifacts_by_key=lambda _k: True,
    )

    return fc, store, conn


def _make_projection_service(
    tmp: str, store: _FakeProjectionStore
) -> tuple[ProjectionService, _FakeProjectionTransformer]:
    transformer = _FakeProjectionTransformer()
    snapshot = IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:projection",
        branch="main",
        commit_id="c1",
        tree_id="t1",
        units=[
            IRCodeUnit(
                unit_id="unit:a",
                kind="function",
                path="pkg/a.py",
                language="python",
                display_name="a",
            )
        ],
    )
    snapshot_store = SimpleNamespace(
        get_snapshot_record=lambda _snapshot_id: SimpleNamespace(artifact_key="ak"),
        load_snapshot=lambda _snapshot_id: snapshot,
        load_ir_graphs=lambda _snapshot_id: None,
        snapshot_dir=lambda snapshot_id: f"{tmp}/{snapshot_id}",
    )
    service = ProjectionService(
        config={},
        logger=logging.getLogger("test"),
        projection_store=store,
        projection_transformer=transformer,
        snapshot_store=snapshot_store,
        manifest_store=SimpleNamespace(),
        load_artifacts_by_key=lambda _key: True,
    )
    return service, transformer


# ---------------------------------------------------------------------------
# Unit tests for FastCode.get_session_prefix
# ---------------------------------------------------------------------------


def test_build_projection_reuses_clean_cached_scope_double():
    with tempfile.TemporaryDirectory(prefix="fc_projection_clean_") as tmp:
        store = _FakeProjectionStore()
        store.cached_id = "proj_cached"
        store.set_layer("proj_cached", "L0", {"layer": "L0"})
        store.set_layer("proj_cached", "L1", {"layer": "L1"})
        store.set_layer("proj_cached", "L2", {"layer": "L2"})
        service, transformer = _make_projection_service(tmp, store)

        result = service.build_projection("snapshot", snapshot_id="snap:projection")

        assert result["status"] == "reused"
        assert result["projection_id"] == "proj_cached"
        assert transformer.calls == []


def test_build_projection_rebuilds_dirty_cached_scope_double():
    with tempfile.TemporaryDirectory(prefix="fc_projection_dirty_") as tmp:
        store = _FakeProjectionStore()
        store.cached_id = "proj_cached"
        store.set_layer("proj_cached", "L0", {"layer": "L0"})
        store.set_layer("proj_cached", "L1", {"layer": "L1"})
        store.set_layer("proj_cached", "L2", {"layer": "L2"})
        scope_key = ProjectionService.projection_scope_key(
            "snapshot", "snap:projection", None, None, None
        )
        store.dirty_scopes.add(("snap:projection", "snapshot", scope_key))
        service, transformer = _make_projection_service(tmp, store)

        result = service.build_projection("snapshot", snapshot_id="snap:projection")

        assert result["status"] == "built"
        assert result["projection_id"] == "proj_built_1"
        assert len(transformer.calls) == 1
        assert store.cleared == [("snap:projection", "snapshot", scope_key)]


def test_build_projection_saves_coverage_paths_double():
    with tempfile.TemporaryDirectory(prefix="fc_projection_coverage_") as tmp:
        store = _FakeProjectionStore()
        service, _transformer = _make_projection_service(tmp, store)

        result = service.build_projection("snapshot", snapshot_id="snap:projection")

        assert result["status"] == "built"
        assert (
            store._builds[result["projection_id"]].coverage_paths == ["pkg/a.py"]
        )


def test_mirror_projection_artifacts_prunes_stale_chunk_files(
    tmp_path: Path,
) -> None:
    store = _FakeProjectionStore()
    service, _transformer = _make_projection_service(str(tmp_path), store)
    payload = {
        "projection_id": "proj_fixed",
        "l0": {"layer": "L0"},
        "l1": {"layer": "L1"},
        "l2_index": {"layer": "L2"},
        "chunks": [
            {"chunk_id": "chunk-a", "content": {"summary": "kept"}},
            {"chunk_id": "chunk-b", "content": {"summary": "stale"}},
        ],
    }

    root = Path(service._mirror_projection_artifacts("snap:projection", payload))
    assert (root / "chunks" / "chunk-b.json").exists()

    payload["chunks"] = [{"chunk_id": "chunk-a", "content": {"summary": "kept"}}]
    service._mirror_projection_artifacts("snap:projection", payload)

    assert (root / "chunks" / "chunk-a.json").exists()
    assert not (root / "chunks" / "chunk-b.json").exists()


def test_build_projection_uses_explicit_scope_and_build_serializers_double(
    monkeypatch: pytest.MonkeyPatch,
):
    with tempfile.TemporaryDirectory(prefix="fc_projection_typed_") as tmp:
        store = _FakeProjectionStore()
        service, _transformer = _make_projection_service(tmp, store)

        def _boom_scope(_: ProjectionScope) -> dict[str, Any]:
            raise AssertionError(
                "projection service must not call ProjectionScope.to_dict()"
            )

        def _boom_build(_: ProjectionBuildResult) -> dict[str, Any]:
            raise AssertionError(
                "projection service must not call ProjectionBuildResult.to_dict()"
            )

        monkeypatch.setattr(ProjectionScope, "to_dict", _boom_scope)
        monkeypatch.setattr(ProjectionBuildResult, "to_dict", _boom_build)

        result = service.build_projection("snapshot", snapshot_id="snap:projection")

        assert result["status"] == "built"
        assert result["projection_id"] == "proj_built_1"
        assert result["l0"]["layer"] == "L0"
        assert result["l2_index"]["layer"] == "L2"


def test_rebuild_dirty_projections_replays_recorded_scope_double():
    with tempfile.TemporaryDirectory(prefix="fc_projection_rebuild_dirty_") as tmp:
        store = _FakeProjectionStore()
        service, transformer = _make_projection_service(tmp, store)
        first = service.build_projection(
            "query",
            snapshot_id="snap:projection",
            query="find a",
            filters={"language": "python"},
        )
        store.dirty_scopes.add(("snap:projection", "query", first["scope_key"]))

        result = service.rebuild_dirty_projections("snap:projection")

        assert result["rebuilt"] == 1
        assert len(transformer.calls) == 2
        assert transformer.calls[-1].query == "find a"
        assert transformer.calls[-1].filters == {"language": "python"}


def test_rebuild_dirty_projections_clears_all_dirty_marker_double():
    with tempfile.TemporaryDirectory(prefix="fc_projection_rebuild_all_dirty_") as tmp:
        store = _FakeProjectionStore()
        service, transformer = _make_projection_service(tmp, store)
        service.build_projection("snapshot", snapshot_id="snap:projection")
        store.dirty_scopes.add(("snap:projection", "all", "*"))

        result = service.rebuild_dirty_projections("snap:projection")

        assert result["rebuilt"] == 1
        assert len(transformer.calls) == 2
        assert ("snap:projection", "all", "*") not in store.dirty_scopes


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

    import fastcode.api.routes as api_mod

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

    import fastcode.api.routes as api_mod

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

    store._builds["proj_exist123"] = ProjectionBuildRecord(
        projection_id="proj_exist123",
        snapshot_id="snap:repo:existing",
        scope_kind="snapshot",
        scope_key="snapshot:*",
        params_hash="hash",
        status="ready",
        warnings=[],
        created_at="2026-05-05T00:00:00+00:00",
        updated_at="2026-05-05T00:00:00+00:00",
        query=None,
        target_id=None,
        filters={},
        coverage_paths=[],
        coverage_nodes=[],
    )

    def fake_get_projection_layer(projection_id: str, layer: str) -> dict[str, Any]:
        return {
            "projection_id": projection_id,
            "layer": layer.upper(),
            "node": store.get_layer(projection_id, layer),
            "build": ProjectionService._projection_build_record_payload(
                store.get_build_record(projection_id)
            ),
        }

    fc.get_projection_layer = fake_get_projection_layer

    import fastcode.api.routes as api_mod

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
    import fastcode.mcp.server as mcp_mod

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
    import fastcode.mcp.server as mcp_mod

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
    import fastcode.mcp.server as mcp_mod

    # The error comes from get_session_prefix raising, not from _get_fastcode.
    # Mock _get_fastcode to return a fake fc, then make get_session_prefix raise.
    fc, _store, _conn = _make_fc_with_prefix(store_enabled=False)
    with patch.object(mcp_mod, "_get_fastcode", return_value=fc):
        result_str = mcp_mod.get_session_prefix("snap:repo:err")

    result = json.loads(result_str)
    assert result["found"] is False
    assert "projection store is not configured" in result["error"]
