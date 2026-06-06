"""
ProjectionService — L0/L1/L2 projection generation, retrieval, and session prefix.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from fastcode.app.store.snapshots.projection_contracts import ProjectionBuildRecord

if TYPE_CHECKING:
    from fastcode.app.store.snapshots.manifest import ManifestStore
    from fastcode.app.store.snapshots.projection import ProjectionStore
    from fastcode.app.store.snapshots.snapshot import SnapshotStore
    from fastcode.ir.projection import ProjectionScope

    from .transform import ProjectionTransformer

from fastcode.utils.filesystem import ensure_dir
from fastcode.utils.hashing import projection_params_hash
from fastcode.utils.paths import projection_scope_key


class ProjectionService:
    """Handles projection building, retrieval, and session prefix generation."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        logger: logging.Logger,
        projection_store: ProjectionStore,
        projection_transformer: ProjectionTransformer,
        snapshot_store: SnapshotStore,
        manifest_store: ManifestStore,
        load_artifacts_by_key: Callable[[str], bool],
    ) -> None:
        self.config = config
        self.logger = logger
        self.projection_store = projection_store
        self.projection_transformer = projection_transformer
        self.snapshot_store = snapshot_store
        self.manifest_store = manifest_store
        self._load_artifacts_by_key = load_artifacts_by_key

    @staticmethod
    def projection_scope_key(
        scope_kind: str,
        snapshot_id: str,
        query: str | None,
        target_id: str | None,
        filters: dict[str, Any] | None,
    ) -> str:
        return projection_scope_key(scope_kind, snapshot_id, query, target_id, filters)

    @staticmethod
    def projection_params_hash(
        scope: ProjectionScope, projection_algo_version: str = "v1"
    ) -> str:
        return projection_params_hash(
            ProjectionService._scope_payload(scope), projection_algo_version
        )

    @staticmethod
    def _scope_payload(scope: ProjectionScope) -> dict[str, Any]:
        return {
            "scope_kind": scope.scope_kind,
            "snapshot_id": scope.snapshot_id,
            "scope_key": scope.scope_key,
            "query": scope.query,
            "target_id": scope.target_id,
            "filters": dict(scope.filters),
        }

    @staticmethod
    def _projection_build_payload(result: Any) -> dict[str, Any]:
        return {
            "projection_id": result.projection_id,
            "snapshot_id": result.snapshot_id,
            "scope_kind": result.scope_kind,
            "scope_key": result.scope_key,
            "l0": result.l0,
            "l1": result.l1,
            "l2_index": result.l2_index,
            "chunks": result.chunks,
            "warnings": list(result.warnings),
            "created_at": result.created_at,
        }

    @staticmethod
    def _projection_build_record_payload(
        record: ProjectionBuildRecord | None,
    ) -> dict[str, Any] | None:
        if record is None:
            return None
        return {
            "projection_id": record.projection_id,
            "snapshot_id": record.snapshot_id,
            "scope_kind": record.scope_kind,
            "scope_key": record.scope_key,
            "params_hash": record.params_hash,
            "status": record.status,
            "warnings": list(record.warnings),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "query": record.query,
            "target_id": record.target_id,
            "filters": dict(record.filters),
            "coverage_paths": list(record.coverage_paths),
            "coverage_nodes": list(record.coverage_nodes),
        }

    def _resolve_snapshot_id(
        self,
        snapshot_id: str | None,
        repo_name: str | None,
        ref_name: str | None,
    ) -> str:
        if snapshot_id:
            return snapshot_id
        if not repo_name or not ref_name:
            raise RuntimeError("projection requires snapshot_id or repo_name+ref_name")
        manifest = self.manifest_store.get_branch_manifest_record(repo_name, ref_name)
        if not manifest:
            raise RuntimeError(f"manifest not found for {repo_name}:{ref_name}")
        return manifest.snapshot_id

    def _mirror_projection_artifacts(
        self, snapshot_id: str, result: dict[str, Any]
    ) -> str:
        import os
        import shutil

        root = os.path.join(
            self.snapshot_store.snapshot_dir(snapshot_id),
            "projection",
            result["projection_id"],
        )
        ensure_dir(root)
        chunk_dir = os.path.join(root, "chunks")
        if os.path.isdir(chunk_dir):
            shutil.rmtree(chunk_dir, ignore_errors=True)
        ensure_dir(chunk_dir)
        with open(os.path.join(root, "node.l0.json"), "w", encoding="utf-8") as f:
            json.dump(result["l0"], f, ensure_ascii=False, indent=2)
        with open(os.path.join(root, "node.l1.json"), "w", encoding="utf-8") as f:
            json.dump(result["l1"], f, ensure_ascii=False, indent=2)
        with open(os.path.join(root, "node.l2.index.json"), "w", encoding="utf-8") as f:
            json.dump(result["l2_index"], f, ensure_ascii=False, indent=2)
        for chunk in result.get("chunks", []):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            with open(
                os.path.join(chunk_dir, f"{chunk_id}.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
        return root

    @staticmethod
    def _projection_coverage_paths(snapshot: Any, result: Any) -> list[str]:
        coverage_nodes: set[str] = set()
        for layer in (result.l0, result.l1, result.l2_index):
            meta = cast(dict[str, Any], layer.get("meta") or {})
            for node_id in cast(list[Any], meta.get("covers_nodes") or []):
                if node_id:
                    coverage_nodes.add(str(node_id))

        path_by_node: dict[str, str] = {}
        for symbol in getattr(snapshot, "symbols", []) or []:
            path = getattr(symbol, "path", "")
            if path:
                path_by_node[str(symbol.symbol_id)] = str(path)
        for document in getattr(snapshot, "documents", []) or []:
            path = getattr(document, "path", "")
            if path:
                path_by_node[str(document.doc_id)] = str(path)

        return sorted(
            {
                path_by_node[node_id]
                for node_id in coverage_nodes
                if node_id in path_by_node
            }
        )

    def build_projection(
        self,
        scope_kind: str,
        snapshot_id: str | None = None,
        repo_name: str | None = None,
        ref_name: str | None = None,
        query: str | None = None,
        target_id: str | None = None,
        filters: dict[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        from fastcode.ir.projection import ProjectionScope

        if scope_kind not in {"snapshot", "query", "entity"}:
            raise RuntimeError("scope_kind must be one of: snapshot, query, entity")
        if not self.projection_store.enabled:
            raise RuntimeError(
                "projection store is not configured (set projection.postgres_dsn)"
            )

        resolved_snapshot_id = self._resolve_snapshot_id(
            snapshot_id, repo_name, ref_name
        )
        snapshot_record = self.snapshot_store.get_snapshot_record(resolved_snapshot_id)
        if not snapshot_record:
            raise RuntimeError(f"snapshot not found: {resolved_snapshot_id}")

        if not self._load_artifacts_by_key(snapshot_record.artifact_key):
            raise RuntimeError(
                f"failed to load artifacts for snapshot: {resolved_snapshot_id}"
            )

        snapshot = self.snapshot_store.load_snapshot(resolved_snapshot_id)
        if not snapshot:
            raise RuntimeError(f"IR snapshot not found: {resolved_snapshot_id}")
        ir_graphs = self.snapshot_store.load_ir_graphs(resolved_snapshot_id)

        scope_key = self.projection_scope_key(
            scope_kind=scope_kind,
            snapshot_id=resolved_snapshot_id,
            query=query,
            target_id=target_id,
            filters=filters,
        )
        scope = ProjectionScope(
            scope_kind=scope_kind,
            snapshot_id=resolved_snapshot_id,
            scope_key=scope_key,
            query=query,
            target_id=target_id,
            filters=filters or {},
        )
        params_hash = self.projection_params_hash(
            scope,
            projection_algo_version=getattr(
                self.projection_transformer, "ALGO_VERSION", "v1"
            ),
        )

        scope_dirty = False
        is_dirty = getattr(self.projection_store, "is_dirty", None)
        if callable(is_dirty):
            scope_dirty = bool(is_dirty(resolved_snapshot_id, scope_kind, scope_key))

        if not force:
            cached_id = self.projection_store.find_cached_projection_id(
                scope, params_hash
            )
            if cached_id and not scope_dirty:
                l0 = self.projection_store.get_layer(cached_id, "L0")
                l1 = self.projection_store.get_layer(cached_id, "L1")
                l2 = self.projection_store.get_layer(cached_id, "L2")
                if l0 and l1 and l2:
                    return {
                        "status": "reused",
                        "projection_id": cached_id,
                        "snapshot_id": resolved_snapshot_id,
                        "scope_kind": scope_kind,
                        "scope_key": scope_key,
                        "l0": l0,
                        "l1": l1,
                        "l2_index": l2,
                        "warnings": [],
                    }

        doc_mentions: dict[str, Any] | None = None
        build = self.projection_transformer.build(
            scope,
            snapshot=snapshot,
            ir_graphs=ir_graphs,
            doc_mentions=doc_mentions or None,
        )
        coverage_paths = self._projection_coverage_paths(snapshot, build)
        self.projection_store.save(
            build,
            params_hash=params_hash,
            scope=scope,
            coverage_paths=coverage_paths,
        )
        clear_dirty = getattr(self.projection_store, "clear_dirty", None)
        if callable(clear_dirty):
            clear_dirty(resolved_snapshot_id, scope_kind, scope_key)
        payload = self._projection_build_payload(build)
        mirror_root = self._mirror_projection_artifacts(resolved_snapshot_id, payload)
        payload["status"] = "built"
        payload["mirror_path"] = mirror_root
        return payload

    def rebuild_dirty_projections(self, snapshot_id: str) -> dict[str, Any]:
        dirty_scopes = self.projection_store.list_dirty_scope_records(snapshot_id)
        if not dirty_scopes:
            return {"snapshot_id": snapshot_id, "rebuilt": 0, "skipped": 0}

        builds = self.projection_store.list_build_records_for_snapshot(snapshot_id)
        builds_by_scope = {
            (build.scope_kind, build.scope_key): build
            for build in builds
        }
        rebuilt: list[str] = []
        skipped = 0
        seen: set[tuple[str, str]] = set()
        rebuilt_all_marker = False
        for dirty in dirty_scopes:
            dirty_kind = dirty.scope_kind
            dirty_key = dirty.scope_key
            if dirty_kind == "all" and dirty_key == "*":
                targets = list(builds_by_scope.values())
                rebuilt_all_marker = bool(targets) or rebuilt_all_marker
            else:
                target = builds_by_scope.get((dirty_kind, dirty_key))
                targets = [target] if target else []
            if not targets:
                skipped += 1
                continue
            for build in targets:
                if not build:
                    skipped += 1
                    continue
                key = (build.scope_kind, build.scope_key)
                if key in seen:
                    continue
                seen.add(key)
                result = self.build_projection(
                    scope_kind=key[0],
                    snapshot_id=snapshot_id,
                    query=build.query,
                    target_id=build.target_id,
                    filters=build.filters or {},
                    force=True,
                )
                rebuilt.append(str(result.get("projection_id")))
        if rebuilt_all_marker:
            clear_dirty = getattr(self.projection_store, "clear_dirty", None)
            if callable(clear_dirty):
                clear_dirty(snapshot_id, "all", "*")
        return {
            "snapshot_id": snapshot_id,
            "rebuilt": len(rebuilt),
            "skipped": skipped,
            "projection_ids": rebuilt,
        }

    def get_projection_layer(self, projection_id: str, layer: str) -> dict[str, Any]:
        if not self.projection_store.enabled:
            raise RuntimeError(
                "projection store is not configured (set projection.postgres_dsn)"
            )
        layer_payload = self.projection_store.get_layer(projection_id, layer)
        if not layer_payload:
            raise RuntimeError(f"projection layer not found: {projection_id}:{layer}")
        build = self._projection_build_record_payload(
            self.projection_store.get_build_record(projection_id)
        )
        return {
            "projection_id": projection_id,
            "layer": layer.upper(),
            "node": layer_payload,
            "build": build,
        }

    def get_projection_chunk(self, projection_id: str, chunk_id: str) -> dict[str, Any]:
        if not self.projection_store.enabled:
            raise RuntimeError(
                "projection store is not configured (set projection.postgres_dsn)"
            )
        chunk_payload = self.projection_store.get_chunk(projection_id, chunk_id)
        if not chunk_payload:
            raise RuntimeError(
                f"projection chunk not found: {projection_id}:{chunk_id}"
            )
        build = self._projection_build_record_payload(
            self.projection_store.get_build_record(projection_id)
        )
        return {
            "projection_id": projection_id,
            "chunk_id": chunk_id,
            "chunk": chunk_payload,
            "build": build,
        }

    def get_session_prefix(self, snapshot_id: str) -> dict[str, Any]:
        """Return L0+L1 projection data for system prompt injection."""
        if not self.projection_store.enabled:
            raise RuntimeError(
                "projection store is not configured (set projection.postgres_dsn)"
            )

        with self.projection_store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT projection_id
                    FROM projection_builds
                    WHERE snapshot_id=%s
                      AND scope_kind='snapshot'
                      AND status='ready'
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                (snapshot_id,),
            )
            row = cur.fetchone()
            if not row:
                return {
                    "snapshot_id": snapshot_id,
                    "l0": None,
                    "l1": None,
                    "error": f"no snapshot-scoped projection found for {snapshot_id}",
                }
            projection_id = row[0]

        l0 = self.projection_store.get_layer(projection_id, "L0")
        l1 = self.projection_store.get_layer(projection_id, "L1")
        if not l0 and not l1:
            return {
                "snapshot_id": snapshot_id,
                "projection_id": projection_id,
                "l0": None,
                "l1": None,
                "error": f"projection {projection_id} has no L0 or L1 layers",
            }

        return {
            "snapshot_id": snapshot_id,
            "projection_id": projection_id,
            "l0": l0,
            "l1": l1,
        }
