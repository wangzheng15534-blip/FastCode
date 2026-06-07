"""Tests for PublishingFacade -- extracted publishing methods."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from fastcode.app.indexing.publishing_facade import PublishingFacade
from fastcode.ir.types import IRCodeUnit, IRSnapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FacadeHarness:
    """Test harness that holds the facade and its mock dependencies."""

    def __init__(
        self,
        *,
        projection_store: Any = None,
        snapshot_store: Any = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.publishing_service = MagicMock()
        self.pipeline = MagicMock()
        self.facade = PublishingFacade(
            publishing_service=self.publishing_service,
            pipeline=self.pipeline,
            projection_store=projection_store or SimpleNamespace(enabled=False),
            snapshot_store=snapshot_store or SimpleNamespace(),
            config=config or {},
        )


def _make_snapshot(units: list[IRCodeUnit]) -> IRSnapshot:
    return IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:1",
        units=units,
    )


# ---------------------------------------------------------------------------
# Simple delegation tests
# ---------------------------------------------------------------------------


class TestSimpleDelegations:
    def test_get_index_run_delegates(self) -> None:
        h = _FacadeHarness()
        h.publishing_service.get_index_run.return_value = {"run_id": "r1"}
        result = h.facade.get_index_run("r1")
        assert result == {"run_id": "r1"}
        h.publishing_service.get_index_run.assert_called_once_with("r1")

    def test_publish_index_run_delegates(self) -> None:
        h = _FacadeHarness()
        h.publishing_service.publish_index_run.return_value = {"status": "published"}
        result = h.facade.publish_index_run("r1", ref_name="main")
        assert result == {"status": "published"}
        h.publishing_service.publish_index_run.assert_called_once_with(
            "r1", ref_name="main"
        )

    def test_retry_pending_publishes_delegates(self) -> None:
        h = _FacadeHarness()
        h.publishing_service.retry_pending_publishes.return_value = {
            "processed": 3,
            "succeeded": 2,
            "failed": 1,
        }
        result = h.facade.retry_pending_publishes(5)
        assert result["processed"] == 3
        h.publishing_service.retry_pending_publishes.assert_called_once_with(5)

    def test_retry_index_run_recovery_delegates(self) -> None:
        h = _FacadeHarness()
        h.publishing_service.retry_index_run_recovery.return_value = {"status": "ok"}
        result = h.facade.retry_index_run_recovery("r1", payload={"source": "/tmp"})
        assert result == {"status": "ok"}
        h.publishing_service.retry_index_run_recovery.assert_called_once_with(
            "r1", payload={"source": "/tmp"}
        )

    def test_process_redo_tasks_delegates(self) -> None:
        h = _FacadeHarness()
        h.publishing_service.process_redo_tasks.return_value = {
            "processed": 1,
            "succeeded": 1,
            "failed": 0,
        }
        result = h.facade.process_redo_tasks(20)
        assert result["succeeded"] == 1
        h.publishing_service.process_redo_tasks.assert_called_once_with(20)


# ---------------------------------------------------------------------------
# process_semantic_repair_frontier tests
# ---------------------------------------------------------------------------


class TestProcessSemanticRepairFrontier:
    def test_replays_pipeline_with_payload(self) -> None:
        h = _FacadeHarness()
        h.pipeline.run_semantic_repair_frontier.return_value = {
            "status": "repaired",
            "repair_frontier": {
                "scope_kind": "package",
                "scope_roots": ["pkg"],
                "changed_paths": ["a.py"],
                "target_paths": ["a.py"],
            },
        }

        result = h.facade.process_semantic_repair_frontier(
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
        assert result["repair_frontier"]["snapshot_id"] == "snap:1"
        assert result["repair_frontier"]["repo_name"] == "repo"
        assert result["repair_frontier"]["reason"] == "api_or_edge_surface_changed"

    def test_raises_when_snapshot_id_missing(self) -> None:
        h = _FacadeHarness()
        with pytest.raises(RuntimeError, match="missing snapshot_id"):
            h.facade.process_semantic_repair_frontier({"scope_kind": "path"})

    def test_marks_existing_projections_dirty(self) -> None:
        marked: list[dict[str, Any]] = []
        h = _FacadeHarness(
            projection_store=SimpleNamespace(
                enabled=True,
                list_builds_for_snapshot=lambda _sid: [
                    {"scope_kind": "snapshot", "scope_key": "scope:snapshot"},
                    {"scope_kind": "query", "scope_key": "scope:query"},
                ],
                mark_dirty=lambda **kwargs: marked.append(kwargs),
            )
        )
        h.pipeline.run_semantic_repair_frontier.return_value = {
            "status": "repaired",
            "warnings": [],
            "repair_frontier": {
                "scope_kind": "package",
                "scope_roots": ["pkg"],
                "changed_paths": ["pkg/a.py"],
                "target_paths": ["pkg/a.py", "pkg/b.py"],
            },
        }

        result = h.facade.process_semantic_repair_frontier(
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

    def test_skips_unrelated_projection_scopes(self) -> None:
        marked: list[dict[str, Any]] = []
        h = _FacadeHarness(
            projection_store=SimpleNamespace(
                enabled=True,
                list_builds_for_snapshot=lambda _sid: [
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
        )
        h.pipeline.run_semantic_repair_frontier.return_value = {
            "status": "repaired",
            "warnings": [],
            "repair_frontier": {
                "scope_kind": "package",
                "scope_roots": ["pkg"],
                "changed_paths": ["pkg/a.py"],
                "target_paths": ["pkg/a.py"],
            },
        }

        result = h.facade.process_semantic_repair_frontier(
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

    def test_uses_coverage_nodes_when_paths_absent(self) -> None:
        snapshot = _make_snapshot(
            [
                IRCodeUnit(
                    unit_id="unit:a",
                    kind="function",
                    path="pkg/a.py",
                    language="python",
                    display_name="a",
                ),
                IRCodeUnit(
                    unit_id="unit:c",
                    kind="function",
                    path="other/c.py",
                    language="python",
                    display_name="c",
                ),
            ]
        )
        marked: list[dict[str, Any]] = []
        h = _FacadeHarness(
            projection_store=SimpleNamespace(
                enabled=True,
                list_builds_for_snapshot=lambda _sid: [
                    {
                        "scope_kind": "snapshot",
                        "scope_key": "scope:snapshot",
                        "coverage_paths": [],
                        "coverage_nodes": ["unit:a"],
                    },
                    {
                        "scope_kind": "query",
                        "scope_key": "scope:query",
                        "coverage_paths": [],
                        "coverage_nodes": ["unit:c"],
                    },
                ],
                mark_dirty=lambda **kwargs: marked.append(kwargs),
                mark_all_dirty=lambda *args, **kwargs: None,
            ),
            snapshot_store=SimpleNamespace(load_snapshot=lambda _sid: snapshot),
        )
        h.pipeline.run_semantic_repair_frontier.return_value = {
            "status": "repaired",
            "warnings": [],
            "repair_frontier": {
                "scope_kind": "package",
                "scope_roots": ["pkg"],
                "changed_paths": ["pkg/a.py"],
                "target_paths": ["pkg/a.py"],
            },
        }

        result = h.facade.process_semantic_repair_frontier(
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

    def test_widens_topology_dirty_scopes(self) -> None:
        all_dirty: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        h = _FacadeHarness(
            projection_store=SimpleNamespace(
                enabled=True,
                list_builds_for_snapshot=lambda _sid: [
                    {
                        "scope_kind": "snapshot",
                        "scope_key": "scope:snapshot",
                        "coverage_paths": ["pkg/a.py"],
                    }
                ],
                mark_dirty=lambda **kwargs: None,
                mark_all_dirty=lambda *args, **kwargs: all_dirty.append((args, kwargs)),
            )
        )
        h.pipeline.run_semantic_repair_frontier.return_value = {
            "status": "repaired",
            "warnings": [],
            "repair_frontier": {
                "scope_kind": "package",
                "scope_roots": ["pkg"],
                "changed_paths": ["pkg/a.py"],
                "target_paths": ["pkg/a.py", "pkg/b.py", "pkg/c.py"],
            },
        }

        result = h.facade.process_semantic_repair_frontier(
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

    def test_projection_dirty_failure_appends_warning(self) -> None:
        h = _FacadeHarness(
            projection_store=SimpleNamespace(
                enabled=True,
                list_builds_for_snapshot=lambda _sid: [
                    {"scope_kind": "snapshot", "scope_key": "scope:snapshot"},
                ],
                mark_dirty=lambda **kwargs: (_ for _ in ()).throw(
                    RuntimeError("store error")
                ),
            )
        )
        h.pipeline.run_semantic_repair_frontier.return_value = {
            "status": "repaired",
            "warnings": [],
            "repair_frontier": {
                "scope_kind": "package",
                "scope_roots": ["pkg"],
                "changed_paths": ["pkg/a.py"],
                "target_paths": ["pkg/a.py"],
            },
        }

        result = h.facade.process_semantic_repair_frontier(
            {
                "snapshot_id": "snap:1",
                "repo_name": "repo",
                "changed_paths": ["pkg/a.py"],
                "scope_kind": "package",
                "scope_roots": ["pkg"],
                "change_kinds": ["embedding_text_hash"],
            }
        )

        assert any(
            "projection_dirty_mark_failed" in w for w in result.get("warnings", [])
        )


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------


class TestProjectionDirtyReason:
    def test_edge_surface_returns_graph_topology(self) -> None:
        assert (
            PublishingFacade._projection_dirty_reason(["edge_surface_hash"])
            == "graph_topology"
        )

    def test_api_surface_returns_api(self) -> None:
        assert PublishingFacade._projection_dirty_reason(["api_surface_hash"]) == "api"

    def test_signature_hash_returns_api(self) -> None:
        assert PublishingFacade._projection_dirty_reason(["signature_hash"]) == "api"

    def test_embedding_text_returns_semantic(self) -> None:
        assert (
            PublishingFacade._projection_dirty_reason(["embedding_text_hash"])
            == "semantic"
        )

    def test_empty_returns_semantic(self) -> None:
        assert PublishingFacade._projection_dirty_reason([]) == "semantic"


class TestProjectionBuildIntersectsPaths:
    def test_intersects_when_coverage_overlaps(self) -> None:
        build = {"coverage_paths": ["pkg/a.py"]}
        assert PublishingFacade._projection_build_intersects_paths(build, ["pkg/a.py"])

    def test_no_intersect_when_disjoint(self) -> None:
        build = {"coverage_paths": ["other/c.py"]}
        assert not PublishingFacade._projection_build_intersects_paths(
            build, ["pkg/a.py"]
        )

    def test_empty_coverage_intersects(self) -> None:
        assert PublishingFacade._projection_build_intersects_paths({}, ["pkg/a.py"])


class TestProjectionBuildIntersectsFrontier:
    def test_path_overlap(self) -> None:
        build = {"coverage_paths": ["pkg/a.py"], "coverage_nodes": []}
        assert PublishingFacade._projection_build_intersects_frontier(
            build, dirty_paths=["pkg/a.py"], dirty_nodes=set()
        )

    def test_node_overlap(self) -> None:
        build = {"coverage_paths": [], "coverage_nodes": ["unit:a"]}
        assert PublishingFacade._projection_build_intersects_frontier(
            build, dirty_paths=["other.py"], dirty_nodes={"unit:a"}
        )

    def test_empty_coverage_and_nodes_returns_true(self) -> None:
        build = {"coverage_paths": [], "coverage_nodes": []}
        assert PublishingFacade._projection_build_intersects_frontier(
            build, dirty_paths=["pkg/a.py"], dirty_nodes=set()
        )

    def test_no_overlap(self) -> None:
        build = {"coverage_paths": ["other.py"], "coverage_nodes": ["unit:x"]}
        assert not PublishingFacade._projection_build_intersects_frontier(
            build, dirty_paths=["pkg/a.py"], dirty_nodes={"unit:b"}
        )
