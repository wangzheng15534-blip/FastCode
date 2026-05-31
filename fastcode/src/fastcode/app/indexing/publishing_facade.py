"""
PublishingFacade -- narrow API surface extracted from FastCode.

Wraps PublishingService with 6 public methods. Five are pure delegation;
process_semantic_repair_frontier has additional projection-dirty marking logic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from fastcode.utils.filesystem import normalize_path

if TYPE_CHECKING:
    from fastcode.app.indexing.pipeline.service import IndexPipeline
    from fastcode.app.indexing.publishing import PublishingService
    from fastcode.app.store.snapshots.projection import ProjectionStore
    from fastcode.app.store.snapshots.snapshot import SnapshotStore


class PublishingFacade:
    """Facade for publishing-related operations.

    Delegates simple calls to PublishingService and owns the
    projection-dirty-marking logic for semantic repair frontiers.
    """

    def __init__(
        self,
        publishing_service: PublishingService,
        pipeline: IndexPipeline,
        projection_store: ProjectionStore,
        snapshot_store: SnapshotStore,
        config: dict[str, Any],
    ) -> None:
        self._publishing_service = publishing_service
        self._pipeline = pipeline
        self._projection_store = projection_store
        self._snapshot_store = snapshot_store
        self._config = config

    # ------------------------------------------------------------------
    # Simple delegations
    # ------------------------------------------------------------------

    def get_index_run(self, run_id: str) -> dict[str, Any] | None:
        return self._publishing_service.get_index_run(run_id)

    def publish_index_run(
        self, run_id: str, ref_name: str | None = None
    ) -> dict[str, Any]:
        return self._publishing_service.publish_index_run(run_id, ref_name=ref_name)

    def retry_pending_publishes(self, limit: int = 10) -> dict[str, Any]:
        return self._publishing_service.retry_pending_publishes(limit)

    def retry_index_run_recovery(
        self, run_id: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._publishing_service.retry_index_run_recovery(
            run_id, payload=payload
        )

    def process_redo_tasks(self, limit: int = 10) -> dict[str, Any]:
        return self._publishing_service.process_redo_tasks(limit)

    # ------------------------------------------------------------------
    # Complex method: process_semantic_repair_frontier
    # ------------------------------------------------------------------

    def process_semantic_repair_frontier(
        self, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = payload or {}
        scope_kind = str(payload.get("scope_kind") or "path")
        scope_roots = list(payload.get("scope_roots") or [])
        snapshot_id = str(payload.get("snapshot_id") or "")
        if not snapshot_id:
            raise RuntimeError("semantic_repair_frontier payload missing snapshot_id")
        result = self._pipeline.run_semantic_repair_frontier(
            snapshot_id=snapshot_id,
            scope_kind=scope_kind,
            scope_roots=scope_roots,
            changed_paths=list(payload.get("changed_paths") or []),
            repo_name=(
                str(payload.get("repo_name")) if payload.get("repo_name") else None
            ),
            change_kinds=list(payload.get("change_kinds") or []),
        )
        result["repair_frontier"]["snapshot_id"] = snapshot_id
        result["repair_frontier"]["repo_name"] = payload.get("repo_name")
        result["repair_frontier"]["reason"] = (
            payload.get("reason") or "api_or_edge_surface_changed"
        )
        try:
            projection_dirty = self._mark_projection_dirty_for_repair(
                snapshot_id=snapshot_id,
                result=result,
                change_kinds=list(payload.get("change_kinds") or []),
            )
            if projection_dirty.get("marked"):
                result["projection_dirty"] = projection_dirty
        except Exception as exc:
            result.setdefault("warnings", []).append(
                f"projection_dirty_mark_failed: {exc}"
            )
        return result

    # ------------------------------------------------------------------
    # Projection-dirty helpers (moved from FastCode)
    # ------------------------------------------------------------------

    @staticmethod
    def _projection_dirty_reason(change_kinds: list[str]) -> str:
        kind_set = set(change_kinds)
        if "edge_surface_hash" in kind_set:
            return "graph_topology"
        if {"api_surface_hash", "signature_hash"} & kind_set:
            return "api"
        if "embedding_text_hash" in kind_set:
            return "semantic"
        return "semantic"

    def _projection_dirty_widened(
        self,
        *,
        change_kinds: list[str],
        dirty_paths: list[str],
    ) -> bool:
        kind_set = set(change_kinds)
        topology_heavy = bool(
            "edge_surface_hash" in kind_set
            and {"api_surface_hash", "signature_hash"} & kind_set
        )
        threshold = int(
            (self._config or {})
            .get("projection", {})
            .get("dirty_widen_path_threshold", 8)
        )
        return topology_heavy or len(dirty_paths) > threshold

    @staticmethod
    def _projection_build_intersects_paths(
        build: dict[str, Any], dirty_paths: list[str]
    ) -> bool:
        coverage_paths = {
            normalize_path(path) for path in build.get("coverage_paths") or [] if path
        }
        if not coverage_paths:
            return True
        dirty_path_set = {normalize_path(path) for path in dirty_paths if path}
        return bool(coverage_paths & dirty_path_set)

    def _projection_dirty_node_ids(
        self, snapshot_id: str, dirty_paths: list[str]
    ) -> set[str]:
        load_snapshot = getattr(self._snapshot_store, "load_snapshot", None)
        if not callable(load_snapshot):
            return set()
        snapshot = load_snapshot(snapshot_id)
        if snapshot is None:
            return set()
        dirty_path_set = {normalize_path(path) for path in dirty_paths if path}
        node_ids: set[str] = set()
        for unit in getattr(snapshot, "units", []) or []:
            unit_path = normalize_path(getattr(unit, "path", ""))
            unit_id = getattr(unit, "unit_id", "")
            if unit_path in dirty_path_set and unit_id:
                node_ids.add(str(unit_id))
        return node_ids

    @staticmethod
    def _projection_build_intersects_frontier(
        build: dict[str, Any],
        *,
        dirty_paths: list[str],
        dirty_nodes: set[str],
    ) -> bool:
        coverage_paths = {
            normalize_path(path) for path in build.get("coverage_paths") or [] if path
        }
        coverage_nodes = {
            str(node_id) for node_id in build.get("coverage_nodes") or [] if node_id
        }
        if not coverage_paths and not coverage_nodes:
            return True

        dirty_path_set = {normalize_path(path) for path in dirty_paths if path}
        if coverage_paths and coverage_paths & dirty_path_set:
            return True
        if coverage_nodes:
            if not dirty_nodes:
                return not coverage_paths
            return bool(coverage_nodes & dirty_nodes)
        return False

    def _mark_projection_dirty_for_repair(
        self,
        *,
        snapshot_id: str,
        result: dict[str, Any],
        change_kinds: list[str],
    ) -> dict[str, Any]:
        store = self._projection_store
        if not getattr(store, "enabled", False):
            return {"marked": 0, "reason": None}
        list_builds = getattr(store, "list_builds_for_snapshot", None)
        mark_dirty = getattr(store, "mark_dirty", None)
        mark_all_dirty = getattr(store, "mark_all_dirty", None)
        if not callable(list_builds) or not callable(mark_dirty):
            return {"marked": 0, "reason": "unsupported_store"}
        list_projection_builds = cast(
            Callable[[str], list[dict[str, Any]]], list_builds
        )
        mark_projection_dirty = cast(Callable[..., Any], mark_dirty)
        mark_all_projections_dirty = (
            cast(Callable[..., Any], mark_all_dirty)
            if callable(mark_all_dirty)
            else None
        )

        frontier = dict(result.get("repair_frontier", {}) or {})
        dirty_paths = sorted(
            {
                normalize_path(path)
                for path in (
                    list(frontier.get("changed_paths") or [])
                    + list(frontier.get("target_paths") or [])
                )
                if path
            }
        )
        if not dirty_paths:
            return {"marked": 0, "reason": None}

        dirty_reason = self._projection_dirty_reason(change_kinds)
        dirty_package_roots = list(frontier.get("scope_roots") or [])
        dirty_nodes = self._projection_dirty_node_ids(snapshot_id, dirty_paths)
        dirty_units = sorted(dirty_nodes)
        if (
            self._projection_dirty_widened(
                change_kinds=change_kinds,
                dirty_paths=dirty_paths,
            )
            and mark_all_projections_dirty is not None
        ):
            mark_all_projections_dirty(
                snapshot_id,
                dirty_reason,
                dirty_paths=dirty_paths,
                dirty_units=dirty_units,
                dirty_package_roots=dirty_package_roots,
            )
            return {
                "marked": 1,
                "reason": dirty_reason,
                "dirty_paths": dirty_paths,
                "dirty_units": dirty_units,
                "widened": True,
                "scope_kind": "all",
            }

        marked = 0
        skipped_clean = 0
        for build in list_projection_builds(snapshot_id):
            scope_kind = str(build.get("scope_kind") or "")
            scope_key = str(build.get("scope_key") or "")
            if not scope_kind or not scope_key:
                continue
            if not self._projection_build_intersects_frontier(
                build,
                dirty_paths=dirty_paths,
                dirty_nodes=dirty_nodes,
            ):
                skipped_clean += 1
                continue
            mark_projection_dirty(
                snapshot_id=snapshot_id,
                scope_kind=scope_kind,
                scope_key=scope_key,
                dirty_paths=dirty_paths,
                dirty_units=dirty_units,
                dirty_package_roots=dirty_package_roots,
                dirty_reason=dirty_reason,
            )
            marked += 1
        return {
            "marked": marked,
            "reason": dirty_reason,
            "dirty_paths": dirty_paths,
            "dirty_units": dirty_units,
            "skipped_clean": skipped_clean,
        }
