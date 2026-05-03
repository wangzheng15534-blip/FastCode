"""
ProjectionService — L0/L1/L2 projection generation, retrieval, and session prefix.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .manifest_store import ManifestStore
    from .projection_models import ProjectionScope
    from .projection_store import ProjectionStore
    from .projection_transform import ProjectionTransformer
    from .snapshot_store import SnapshotStore

from .utils import ensure_dir, projection_params_hash, projection_scope_key


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
        return projection_params_hash(scope.to_dict(), projection_algo_version)

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

        root = os.path.join(
            self.snapshot_store.snapshot_dir(snapshot_id),
            "projection",
            result["projection_id"],
        )
        ensure_dir(root)
        chunk_dir = os.path.join(root, "chunks")
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
        from .projection_models import ProjectionScope

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

        if not force:
            cached_id = self.projection_store.find_cached_projection_id(
                scope, params_hash
            )
            if cached_id:
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
        self.projection_store.save(build, params_hash=params_hash)
        payload = build.to_dict()
        mirror_root = self._mirror_projection_artifacts(resolved_snapshot_id, payload)
        payload["status"] = "built"
        payload["mirror_path"] = mirror_root
        return payload

    def get_projection_layer(self, projection_id: str, layer: str) -> dict[str, Any]:
        if not self.projection_store.enabled:
            raise RuntimeError(
                "projection store is not configured (set projection.postgres_dsn)"
            )
        layer_payload = self.projection_store.get_layer(projection_id, layer)
        if not layer_payload:
            raise RuntimeError(f"projection layer not found: {projection_id}:{layer}")
        build = self.projection_store.get_build(projection_id)
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
        build = self.projection_store.get_build(projection_id)
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
