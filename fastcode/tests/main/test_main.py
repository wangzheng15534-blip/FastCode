"""Tests for main module."""

from __future__ import annotations

from dataclasses import is_dataclass
from types import SimpleNamespace
from typing import Any

from fastcode.main import FastCode
from fastcode.schemas.config import config_from_mapping

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


def test_apply_repository_runtime_overrides_refreshes_loader_and_runtime_config():
    fc = FastCode.__new__(FastCode)
    fc.runtime_config = config_from_mapping(
        {
            "repository": {
                "ignore_patterns": ["base"],
                "exclude_site_packages": False,
            }
        }
    )
    fc.config = fc.runtime_config.to_dict()
    fc.eval_config = fc.config.get("evaluation", {})
    fc.eval_mode = False
    fc.in_memory_index = False
    fc.loader = SimpleNamespace(
        repo_config=fc.config["repository"],
        ignore_patterns=fc.config["repository"]["ignore_patterns"],
    )

    fc.apply_repository_runtime_overrides(
        ignore_patterns=("base", ".venv"),
        exclude_site_packages=True,
    )

    assert is_dataclass(fc.runtime_config)
    assert fc.runtime_config.repository.ignore_patterns == ("base", ".venv")
    assert fc.runtime_config.repository.exclude_site_packages is True
    assert fc.loader.repo_config["ignore_patterns"] == ("base", ".venv")
    assert fc.loader.repo_config["exclude_site_packages"] is True
    assert fc.loader.ignore_patterns == ("base", ".venv")


def test_process_semantic_repair_frontier_replays_pipeline_with_payload():
    fc = FastCode.__new__(FastCode)
    fc.pipeline = SimpleNamespace(
        run_semantic_repair_frontier=lambda **kwargs: {
            "status": "repaired",
            "kwargs": kwargs,
            "repair_frontier": {
                "scope_kind": kwargs["scope_kind"],
                "scope_roots": kwargs["scope_roots"],
                "changed_paths": kwargs["changed_paths"],
                "target_paths": kwargs["changed_paths"],
            },
        }
    )

    result = fc.process_semantic_repair_frontier(
        {
            "snapshot_id": "snap:1",
            "repo_name": "repo",
            "changed_paths": ["a.py"],
            "reason": "api_or_edge_surface_changed",
            "scope_kind": "package",
            "scope_roots": ["pkg"],
        }
    )

    assert result["status"] == "repaired"
    assert result["kwargs"]["snapshot_id"] == "snap:1"
    assert result["repair_frontier"]["snapshot_id"] == "snap:1"
    assert result["repair_frontier"]["changed_paths"] == ["a.py"]
    assert result["repair_frontier"]["scope_kind"] == "package"
    assert result["repair_frontier"]["scope_roots"] == ["pkg"]


def test_process_semantic_repair_frontier_marks_existing_projections_dirty():
    fc = FastCode.__new__(FastCode)
    fc.pipeline = SimpleNamespace(
        run_semantic_repair_frontier=lambda **kwargs: {
            "status": "repaired",
            "warnings": [],
            "repair_frontier": {
                "scope_kind": kwargs["scope_kind"],
                "scope_roots": kwargs["scope_roots"],
                "changed_paths": kwargs["changed_paths"],
                "target_paths": ["pkg/a.py", "pkg/b.py"],
            },
        }
    )
    marked: list[dict[str, Any]] = []
    fc.projection_store = SimpleNamespace(
        enabled=True,
        list_builds_for_snapshot=lambda _snapshot_id: [
            {"scope_kind": "snapshot", "scope_key": "scope:snapshot"},
            {"scope_kind": "query", "scope_key": "scope:query"},
        ],
        mark_dirty=lambda **kwargs: marked.append(kwargs),
    )

    result = fc.process_semantic_repair_frontier(
        {
            "snapshot_id": "snap:1",
            "repo_name": "repo",
            "changed_paths": ["pkg/a.py"],
            "scope_kind": "package",
            "scope_roots": ["pkg"],
            "change_kinds": ["api_surface_hash"],
        }
    )

    assert result["projection_dirty"]["marked"] == 2
    assert result["projection_dirty"]["reason"] == "api"
    assert {entry["scope_key"] for entry in marked} == {
        "scope:snapshot",
        "scope:query",
    }
    assert marked[0]["dirty_paths"] == ["pkg/a.py", "pkg/b.py"]


def test_process_semantic_repair_frontier_skips_unrelated_projection_scopes():
    fc = FastCode.__new__(FastCode)
    fc.pipeline = SimpleNamespace(
        run_semantic_repair_frontier=lambda **kwargs: {
            "status": "repaired",
            "warnings": [],
            "repair_frontier": {
                "scope_kind": kwargs["scope_kind"],
                "scope_roots": kwargs["scope_roots"],
                "changed_paths": kwargs["changed_paths"],
                "target_paths": ["pkg/a.py"],
            },
        }
    )
    marked: list[dict[str, Any]] = []
    fc.projection_store = SimpleNamespace(
        enabled=True,
        list_builds_for_snapshot=lambda _snapshot_id: [
            {
                "scope_kind": "snapshot",
                "scope_key": "scope:snapshot",
                "coverage_paths": ["pkg/a.py"],
            },
            {
                "scope_kind": "query",
                "scope_key": "scope:query",
                "coverage_paths": ["other/c.py"],
            },
        ],
        mark_dirty=lambda **kwargs: marked.append(kwargs),
        mark_all_dirty=lambda *args, **kwargs: None,
    )

    result = fc.process_semantic_repair_frontier(
        {
            "snapshot_id": "snap:1",
            "repo_name": "repo",
            "changed_paths": ["pkg/a.py"],
            "scope_kind": "package",
            "scope_roots": ["pkg"],
            "change_kinds": ["embedding_text_hash"],
        }
    )

    assert result["projection_dirty"]["marked"] == 1
    assert {entry["scope_key"] for entry in marked} == {"scope:snapshot"}


def test_process_semantic_repair_frontier_widens_topology_dirty_scopes():
    fc = FastCode.__new__(FastCode)
    fc.pipeline = SimpleNamespace(
        run_semantic_repair_frontier=lambda **kwargs: {
            "status": "repaired",
            "warnings": [],
            "repair_frontier": {
                "scope_kind": kwargs["scope_kind"],
                "scope_roots": kwargs["scope_roots"],
                "changed_paths": kwargs["changed_paths"],
                "target_paths": ["pkg/a.py", "pkg/b.py", "pkg/c.py"],
            },
        }
    )
    all_dirty: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    fc.projection_store = SimpleNamespace(
        enabled=True,
        list_builds_for_snapshot=lambda _snapshot_id: [
            {
                "scope_kind": "snapshot",
                "scope_key": "scope:snapshot",
                "coverage_paths": ["pkg/a.py"],
            }
        ],
        mark_dirty=lambda **kwargs: None,
        mark_all_dirty=lambda *args, **kwargs: all_dirty.append((args, kwargs)),
    )

    result = fc.process_semantic_repair_frontier(
        {
            "snapshot_id": "snap:1",
            "repo_name": "repo",
            "changed_paths": ["pkg/a.py"],
            "scope_kind": "package",
            "scope_roots": ["pkg"],
            "change_kinds": ["api_surface_hash", "edge_surface_hash"],
        }
    )

    assert result["projection_dirty"]["reason"] == "graph_topology"
    assert result["projection_dirty"]["widened"] is True
    assert all_dirty
