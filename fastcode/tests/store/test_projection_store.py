from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest

from fastcode.ir.projection import ProjectionBuildResult, ProjectionScope
from fastcode.store.projection import ProjectionStore
from fastcode.store.records import ProjectionBuildRecord, ProjectionDirtyScopeRecord


class _FakeCursor:
    def __init__(
        self,
        responder: Callable[[str, tuple[Any, ...]], tuple[Any, list[Any]]],
        executed: list[tuple[str, tuple[Any, ...]]],
    ) -> None:
        self._responder = responder
        self._executed = executed
        self._row: Any = None
        self._rows: list[Any] = []

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self._executed.append((sql, params))
        self._row, self._rows = self._responder(sql, params)

    def fetchone(self) -> Any:
        return self._row

    def fetchall(self) -> list[Any]:
        return list(self._rows)

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _FakeConnection:
    def __init__(
        self,
        responder: Callable[[str, tuple[Any, ...]], tuple[Any, list[Any]]],
        executed: list[tuple[str, tuple[Any, ...]]],
    ) -> None:
        self._responder = responder
        self._executed = executed

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._responder, self._executed)

    def commit(self) -> None:
        return None

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _make_store(
    responder: Callable[[str, tuple[Any, ...]], tuple[Any, list[Any]]],
) -> tuple[ProjectionStore, list[tuple[str, tuple[Any, ...]]]]:
    executed: list[tuple[str, tuple[Any, ...]]] = []
    store = ProjectionStore.__new__(ProjectionStore)
    store.logger = logging.getLogger("test")
    store.enabled = True
    store.pool = None
    store.dsn = "postgresql://test"
    store._connect = lambda: _FakeConnection(responder, executed)  # type: ignore[method-assign]
    return store, executed


def test_dirty_scope_helpers_use_explicit_serializers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirty_row = (
        "snap:1",
        "query",
        "scope:1",
        '["pkg/a.py"]',
        ["unit:a"],
        None,
        "semantic_repair",
        "2026-05-05T00:00:00+00:00",
        "2026-05-05T00:00:05+00:00",
    )

    def responder(sql: str, _params: tuple[Any, ...]) -> tuple[Any, list[Any]]:
        if "FROM projection_dirty_scopes" not in sql:
            raise AssertionError(f"unexpected SQL: {sql}")
        if "ORDER BY updated_at DESC" in sql:
            return None, [dirty_row]
        return dirty_row, []

    store, _executed = _make_store(responder)

    def _boom(_: ProjectionDirtyScopeRecord) -> dict[str, Any]:
        raise AssertionError(
            "projection store must not call ProjectionDirtyScopeRecord.to_dict()"
        )

    monkeypatch.setattr(ProjectionDirtyScopeRecord, "to_dict", _boom)

    dirty = store.get_dirty_scope("snap:1", "query", "scope:1")
    listed = store.list_dirty_scopes("snap:1")

    assert dirty is not None
    assert dirty["dirty_paths"] == ["pkg/a.py"]
    assert dirty["dirty_units"] == ["unit:a"]
    assert dirty["dirty_package_roots"] == []
    assert listed == [dirty]


def test_mark_dirty_merges_existing_scope_via_typed_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_row = (
        "snap:1",
        "query",
        "scope:1",
        ["pkg/a.py"],
        '["unit:a"]',
        None,
        "old_reason",
        "2026-05-05T00:00:00+00:00",
        "2026-05-05T00:00:05+00:00",
    )

    def responder(sql: str, _params: tuple[Any, ...]) -> tuple[Any, list[Any]]:
        if "FROM projection_dirty_scopes" in sql:
            return existing_row, []
        if "INSERT INTO projection_dirty_scopes" in sql:
            return None, []
        raise AssertionError(f"unexpected SQL: {sql}")

    store, executed = _make_store(responder)

    def _boom(_: ProjectionDirtyScopeRecord) -> dict[str, Any]:
        raise AssertionError(
            "projection store must not call ProjectionDirtyScopeRecord.to_dict()"
        )

    monkeypatch.setattr(ProjectionDirtyScopeRecord, "to_dict", _boom)
    monkeypatch.setattr(
        "fastcode.store.projection.utc_now", lambda: "2026-05-05T00:00:10+00:00"
    )

    store.mark_dirty(
        snapshot_id="snap:1",
        scope_kind="query",
        scope_key="scope:1",
        dirty_paths=["pkg/b.py"],
        dirty_reason="new_reason",
        dirty_units=["unit:b"],
        dirty_package_roots=["pkg"],
    )

    insert_sql, insert_params = executed[-1]
    assert "INSERT INTO projection_dirty_scopes" in insert_sql
    assert insert_params[3] == '["pkg/a.py", "pkg/b.py"]'
    assert insert_params[4] == '["unit:a", "unit:b"]'
    assert insert_params[5] == '["pkg"]'
    assert insert_params[6] == "new_reason"


def test_build_helpers_use_explicit_serializers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_row = (
        "proj_1",
        "snap:1",
        "snapshot",
        "snapshot:*",
        "hash123",
        "ready",
        '["warn1"]',
        "2026-05-05T00:00:00+00:00",
        "2026-05-05T00:00:05+00:00",
        "where config",
        "unit:a",
        {"language": "python"},
        '["pkg/a.py"]',
        ["unit:a"],
    )

    def responder(sql: str, _params: tuple[Any, ...]) -> tuple[Any, list[Any]]:
        if "FROM projection_builds" not in sql:
            raise AssertionError(f"unexpected SQL: {sql}")
        if "WHERE projection_id=%s" in sql:
            return build_row, []
        if "WHERE snapshot_id=%s AND status='ready'" in sql:
            return None, [build_row]
        raise AssertionError(f"unexpected SQL: {sql}")

    store, _executed = _make_store(responder)

    def _boom(_: ProjectionBuildRecord) -> dict[str, Any]:
        raise AssertionError(
            "projection store must not call ProjectionBuildRecord.to_dict()"
        )

    monkeypatch.setattr(ProjectionBuildRecord, "to_dict", _boom)

    build = store.get_build("proj_1")
    builds = store.list_builds_for_snapshot("snap:1")

    assert build is not None
    assert build["warnings"] == ["warn1"]
    assert build["filters"] == {"language": "python"}
    assert build["coverage_paths"] == ["pkg/a.py"]
    assert build["coverage_nodes"] == ["unit:a"]
    assert builds == [build]


def test_save_prunes_stale_chunks_before_upserting_current_chunk_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def responder(_sql: str, _params: tuple[Any, ...]) -> tuple[Any, list[Any]]:
        return None, []

    store, executed = _make_store(responder)
    monkeypatch.setattr(
        "fastcode.store.projection.utc_now", lambda: "2026-05-05T00:00:10+00:00"
    )

    result = ProjectionBuildResult(
        projection_id="proj_1",
        snapshot_id="snap:1",
        scope_kind="snapshot",
        scope_key="snapshot:*",
        l0={"layer": "L0", "meta": {"covers_nodes": ["unit:a"]}},
        l1={"layer": "L1", "meta": {"covers_nodes": ["unit:a"]}},
        l2_index={"layer": "L2", "meta": {"covers_nodes": ["unit:a"]}},
        chunks=[{"chunk_id": "chunk-a", "content": {"summary": "kept"}}],
    )
    scope = ProjectionScope(
        scope_kind="snapshot",
        snapshot_id="snap:1",
        scope_key="snapshot:*",
    )

    store.save(result, "hash123", scope=scope, coverage_paths=["pkg/a.py"])

    delete_sql, delete_params = next(
        (sql, params)
        for sql, params in executed
        if "DELETE FROM projection_chunks" in sql
    )
    assert "DELETE FROM projection_chunks" in delete_sql
    assert delete_params[0] == "proj_1"
    assert delete_params[1] == ["chunk-a"]
