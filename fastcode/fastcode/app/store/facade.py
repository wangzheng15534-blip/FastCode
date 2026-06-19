"""StoreFacade — aggregates store subsystems behind a narrow API for entry frames.

Extracted from FastCode during Phase 4 of the FCIS split. Owns all read-only
store queries (status, snapshots, manifests, symbols, graphs, SCIP) and
delegates to VectorStore, SnapshotStore, ManifestStore, and SnapshotSymbolIndex.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from typing import Any, cast

from fastcode.app.store.snapshots.manifest import ManifestStore
from fastcode.app.store.snapshots.manifest_contracts import ManifestRecord
from fastcode.app.store.snapshots.snapshot import SnapshotStore
from fastcode.app.store.code_status_keys import default_code_status_keys
from fastcode.app.store.snapshots.snapshot_contracts import (
    SCIPArtifactRecord,
    SnapshotRefRecord,
)
from fastcode.app.store.vectors.vector import VectorStore
from fastcode.ir.code_status import build_code_status_pack
from fastcode.ir.types import IRSymbol
from fastcode.runtime_support.runtime_state import RuntimeState
from fastcode.semantic.symbol_index import SnapshotSymbolIndex
from fastcode.utils.json import safe_jsonable


class StoreFacade:
    """Aggregated read-only store API for entry frames and internal callers."""

    def __init__(
        self,
        vector_store: VectorStore,
        snapshot_store: SnapshotStore,
        manifest_store: ManifestStore,
        snapshot_symbol_index: SnapshotSymbolIndex,
        state: RuntimeState,
        config: dict[str, Any],
        *,
        projection_store: Any | None = None,
        projection_transformer: Any | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._snapshot_store = snapshot_store
        self._manifest_store = manifest_store
        self._snapshot_symbol_index = snapshot_symbol_index
        self._state = state
        self._config = config
        self._projection_store = projection_store
        self._projection_transformer = projection_transformer

    # ------------------------------------------------------------------
    # Status / info
    # ------------------------------------------------------------------

    def get_status_info(self, *, full_scan: bool = False) -> dict[str, Any]:
        """Return all status fields needed by entry frames."""
        available_repos = self._vector_store.scan_available_indexes(
            use_cache=not full_scan
        )
        loaded_repos = self.list_repositories()
        retrieval_cfg = self._config.get("retrieval", {}) or {}
        return {
            "repo_loaded": self._state.repo_loaded,
            "repo_indexed": self._state.repo_indexed,
            "repo_info": self._state.repo_info,
            "multi_repo_mode": self._state.multi_repo_mode,
            "storage_backend": self._snapshot_store.db_runtime.backend,
            "retrieval_backend": retrieval_cfg.get("retrieval_backend", "local"),
            "graph_expansion_backend": retrieval_cfg.get(
                "graph_expansion_backend", "graph_builder"
            ),
            "available_repositories": available_repos,
            "loaded_repositories": loaded_repos,
        }

    def get_repository_summary(self) -> str:
        """Get summary of the loaded repository."""
        if not self._state.repo_info:
            return "No repository loaded"

        summary_parts = [
            f"Repository: {self._state.repo_info.get('name', 'Unknown')}",
            f"Files: {self._state.repo_info.get('file_count', 0)}",
            f"Size: {self._state.repo_info.get('total_size_mb', 0):.2f} MB",
        ]

        if self._state.repo_indexed:
            summary_parts.append(f"Indexed elements: {self._vector_store.get_count()}")

        return "\n".join(summary_parts)

    def get_repository_stats(self) -> dict[str, Any]:
        """Get statistics about all indexed repositories."""
        repo_counts = self._vector_store.get_count_by_repository()
        repo_names = self._vector_store.get_repository_names()

        stats: dict[str, Any] = {
            "total_repositories": len(repo_names),
            "total_elements": self._vector_store.get_count(),
            "repositories": [],
        }

        for repo_name in repo_names:
            repo_info = self._state.loaded_repositories.get(repo_name, {})
            stats["repositories"].append(
                {
                    "name": repo_name,
                    "elements": repo_counts.get(repo_name, 0),
                    "files": repo_info.get("file_count", 0),
                    "size_mb": repo_info.get("total_size_mb", 0),
                }
            )

        return stats

    def list_repositories(self) -> list[dict[str, Any]]:
        """List all indexed repositories."""
        repo_names = self._vector_store.get_repository_names()
        repo_counts = self._vector_store.get_count_by_repository()

        repositories: list[dict[str, Any]] = []
        for repo_name in repo_names:
            repo_info = self._state.loaded_repositories.get(repo_name, {})
            repositories.append(
                {
                    "name": repo_name,
                    "element_count": repo_counts.get(repo_name, 0),
                    "file_count": repo_info.get("file_count", 0),
                    "size_mb": repo_info.get("total_size_mb", 0),
                    "url": repo_info.get("url", "N/A"),
                }
            )

        return repositories

    def list_available_repos(self) -> list[dict[str, Any]]:
        """List all indexed repositories with metadata."""
        return self._vector_store.scan_available_indexes(use_cache=False)

    def get_repo_overview(self, repo_name: str) -> dict[str, Any] | None:
        """Get overview for an indexed repository."""
        overviews = self._vector_store.load_repo_overviews(include_embeddings=False)
        return overviews.get(repo_name)

    def is_repo_indexed(self, repo_name: str) -> bool:
        """Check whether a persisted index exists for a repo."""
        has_saved_index = getattr(self._vector_store, "has_saved_index", None)
        if callable(has_saved_index):
            return bool(has_saved_index(repo_name))
        persist_dir = self._vector_store.persist_dir
        faiss_path = os.path.join(persist_dir, f"{repo_name}.faiss")
        meta_path = os.path.join(persist_dir, f"{repo_name}_metadata.pkl")
        return os.path.exists(faiss_path) and os.path.exists(meta_path)

    def repo_name_from_source(self, source: str, is_url: bool) -> str:
        """Derive a canonical repo name from a URL or local path."""
        from fastcode.utils.filesystem import get_repo_name_from_url

        if is_url:
            return get_repo_name_from_url(source)
        return os.path.basename(os.path.normpath(source))

    # ------------------------------------------------------------------
    # Snapshot / manifest
    # ------------------------------------------------------------------

    def list_repo_refs(self, repo_name: str) -> list[dict[str, Any]]:
        return [
            self._snapshot_ref_payload(record)
            for record in self._snapshot_store.list_repo_ref_records(repo_name)
        ]

    def get_snapshot_manifest(self, snapshot_id: str) -> dict[str, Any] | None:
        record = self._manifest_store.get_snapshot_manifest_record(snapshot_id)
        return self._manifest_payload(record) if record is not None else None

    def get_branch_manifest(
        self, repo_name: str, ref_name: str
    ) -> dict[str, Any] | None:
        record = self._manifest_store.get_branch_manifest_record(repo_name, ref_name)
        return self._manifest_payload(record) if record is not None else None

    def get_code_status_pack(
        self,
        snapshot_id: str,
        *,
        include_graph_facts: bool = True,
    ) -> dict[str, Any]:
        snapshot_record = self._snapshot_store.get_snapshot_record(snapshot_id)
        if snapshot_record is None:
            msg = f"snapshot not found: {snapshot_id}"
            raise RuntimeError(msg)
        snapshot = self._snapshot_store.load_snapshot(snapshot_id)
        if snapshot is None:
            msg = f"snapshot payload not found: {snapshot_id}"
            raise RuntimeError(msg)
        ir_graphs = (
            self._snapshot_store.load_ir_graphs(snapshot_id)
            if include_graph_facts
            else None
        )
        return build_code_status_pack(
            snapshot,
            artifact_key=snapshot_record.artifact_key,
            manifest=self.get_snapshot_manifest(snapshot_id),
            ir_graphs=ir_graphs,
            include_graph_facts=include_graph_facts,
            **default_code_status_keys(),
        )

    # ------------------------------------------------------------------
    # Symbol
    # ------------------------------------------------------------------

    def find_symbol(
        self,
        snapshot_id: str,
        *,
        symbol_id: str | None = None,
        name: str | None = None,
        path: str | None = None,
    ) -> dict[str, Any] | None:
        resolved = self.resolve_snapshot_symbol(
            snapshot_id, symbol_id=symbol_id, name=name, path=path
        )
        if not resolved:
            return None
        load_record = getattr(self._snapshot_store, "load_snapshot_symbol_record", None)
        if callable(load_record):
            record = load_record(snapshot_id, resolved)
            if isinstance(record, Mapping):
                return safe_jsonable({str(key): value for key, value in record.items()})

        snapshot = self._snapshot_store.load_snapshot(snapshot_id)
        if not snapshot:
            return None
        for symbol in snapshot.symbols:
            if symbol.symbol_id == resolved:
                return self._ir_symbol_payload(symbol)
        return None

    def resolve_snapshot_symbol(
        self,
        snapshot_id: str,
        *,
        symbol_id: str | None = None,
        name: str | None = None,
        path: str | None = None,
    ) -> str | None:
        self._ensure_snapshot_symbol_index(snapshot_id)
        return self._snapshot_symbol_index.resolve_symbol(
            snapshot_id,
            symbol_id=symbol_id,
            name=name,
            path=path,
        )

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def get_graph_callees(
        self, snapshot_id: str, symbol_id: str, max_hops: int = 1
    ) -> list[dict[str, Any]]:
        max_hops = max(1, min(max_hops, 20))
        ir_graphs = self._snapshot_store.load_ir_graphs(snapshot_id)
        if not ir_graphs:
            return []
        g = ir_graphs.call_graph
        dist = self._bounded_graph_distances(
            g,
            symbol_id,
            max_hops,
            direction="out",
        )
        return [{"symbol_id": node, "distance": d} for node, d in dist.items()]

    def get_graph_callers(
        self, snapshot_id: str, symbol_id: str, max_hops: int = 1
    ) -> list[dict[str, Any]]:
        max_hops = max(1, min(max_hops, 20))
        ir_graphs = self._snapshot_store.load_ir_graphs(snapshot_id)
        if not ir_graphs:
            return []
        dist = self._bounded_graph_distances(
            ir_graphs.call_graph,
            symbol_id,
            max_hops,
            direction="in",
        )
        return [{"symbol_id": node, "distance": d} for node, d in dist.items()]

    def get_graph_dependencies(
        self, snapshot_id: str, doc_id: str, max_hops: int = 1
    ) -> list[dict[str, Any]]:
        max_hops = max(1, min(max_hops, 20))
        ir_graphs = self._snapshot_store.load_ir_graphs(snapshot_id)
        if not ir_graphs:
            return []
        g = ir_graphs.dependency_graph
        dist = self._bounded_graph_distances(
            g,
            doc_id,
            max_hops,
            direction="out",
        )
        return [{"doc_id": node, "distance": d} for node, d in dist.items()]

    # ------------------------------------------------------------------
    # Graph analysis (delegates to fastcode.graph.analysis)
    # ------------------------------------------------------------------

    def compute_directed_path_for_snapshot(
        self,
        from_symbol: str,
        to_symbol: str,
        snapshot_id: str,
        max_hops: int = 5,
        graph_types: list[str] | None = None,
    ) -> dict[str, Any]:
        from fastcode.graph.analysis import compute_directed_path_for_snapshot as _impl

        return _impl(
            self._graph_analysis_ctx(),
            from_symbol,
            to_symbol,
            snapshot_id,
            max_hops,
            graph_types,
        )

    def compute_impact_analysis_for_snapshot(
        self,
        symbol: str,
        snapshot_id: str,
        max_hops: int = 3,
        graph_types: list[str] | None = None,
    ) -> dict[str, Any]:
        from fastcode.graph.analysis import (
            compute_impact_analysis_for_snapshot as _impl,
        )

        return _impl(
            self._graph_analysis_ctx(),
            symbol,
            snapshot_id,
            max_hops,
            graph_types,
        )

    def compute_leiden_clusters_for_snapshot(
        self,
        snapshot_id: str,
    ) -> dict[str, Any]:
        from fastcode.graph.analysis import (
            compute_leiden_clusters_for_snapshot as _impl,
        )

        return _impl(self._graph_analysis_ctx(), snapshot_id)

    def compute_steiner_path_for_snapshot(
        self,
        terminals: list[str],
        snapshot_id: str,
    ) -> dict[str, Any]:
        from fastcode.graph.analysis import compute_steiner_path_for_snapshot as _impl

        return _impl(self._graph_analysis_ctx(), terminals, snapshot_id)

    def compute_find_callers_for_snapshot(
        self,
        symbol: str,
        snapshot_id: str,
        max_hops: int = 2,
    ) -> dict[str, Any]:
        from fastcode.graph.analysis import compute_find_callers_for_snapshot as _impl

        return _impl(self._graph_analysis_ctx(), symbol, snapshot_id, max_hops)

    # ------------------------------------------------------------------
    # SCIP
    # ------------------------------------------------------------------

    def get_scip_artifact_ref(self, snapshot_id: str) -> dict[str, Any] | None:
        record = self._snapshot_store.get_scip_artifact_ref_record(snapshot_id)
        return self._scip_artifact_payload(record) if record is not None else None

    def list_scip_artifact_refs(self, snapshot_id: str) -> list[dict[str, Any]]:
        return [
            self._scip_artifact_payload(record)
            for record in self._snapshot_store.list_scip_artifact_ref_records(
                snapshot_id
            )
        ]

    # ------------------------------------------------------------------
    # Static / class helpers (moved from FastCode)
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_ref_payload(record: SnapshotRefRecord) -> dict[str, Any]:
        return {
            "ref_id": record.ref_id,
            "repo_name": record.repo_name,
            "branch": record.branch,
            "commit_id": record.commit_id,
            "tree_id": record.tree_id,
            "snapshot_id": record.snapshot_id,
            "created_at": record.created_at,
        }

    @staticmethod
    def _manifest_payload(record: ManifestRecord) -> dict[str, Any]:
        return {
            "manifest_id": record.manifest_id,
            "repo_name": record.repo_name,
            "ref_name": record.ref_name,
            "snapshot_id": record.snapshot_id,
            "index_run_id": record.index_run_id,
            "published_at": record.published_at,
            "previous_manifest_id": record.previous_manifest_id,
            "status": record.status,
        }

    @staticmethod
    def _ir_symbol_payload(symbol: IRSymbol) -> dict[str, Any]:
        return {
            "symbol_id": symbol.symbol_id,
            "external_symbol_id": symbol.external_symbol_id,
            "path": symbol.path,
            "display_name": symbol.display_name,
            "kind": symbol.kind,
            "language": symbol.language,
            "qualified_name": symbol.qualified_name,
            "signature": symbol.signature,
            "start_line": symbol.start_line,
            "start_col": symbol.start_col,
            "end_line": symbol.end_line,
            "end_col": symbol.end_col,
            "source_priority": symbol.source_priority,
            "source_set": sorted(value for value in symbol.source_set if value),
            "metadata": safe_jsonable(
                {str(key): item for key, item in symbol.metadata.items()}
            ),
        }

    @staticmethod
    def _scip_artifact_metadata_payload(raw_metadata: str | None) -> dict[str, Any]:
        if raw_metadata is None:
            return {}
        try:
            metadata = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError):
            return {}
        if not isinstance(metadata, Mapping):
            return {}
        return {str(key): item for key, item in metadata.items()}

    @classmethod
    def _scip_artifact_payload(cls, record: SCIPArtifactRecord) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "snapshot_id": record.snapshot_id,
            "indexer_name": record.indexer_name,
            "indexer_version": record.indexer_version,
            "artifact_path": record.artifact_path,
            "checksum": record.checksum,
            "created_at": record.created_at,
        }
        has_entry_fields = (
            record.artifact_id is not None
            or record.sequence_no is not None
            or record.role is not None
            or record.metadata_json is not None
        )
        if not has_entry_fields:
            return payload
        payload.update(
            {
                "artifact_id": record.artifact_id,
                "sequence_no": record.sequence_no,
                "role": record.role,
                "metadata": cls._scip_artifact_metadata_payload(record.metadata_json),
            }
        )
        return payload

    @staticmethod
    def _bounded_graph_distances(
        graph: Any,
        seed: str,
        max_hops: int,
        *,
        direction: str,
    ) -> dict[str, int]:
        try:
            if seed not in graph:
                return {}
        except TypeError:
            return {}

        native_distances = getattr(graph, "distances_within", None)
        if callable(native_distances):
            result = native_distances(seed, max_hops, mode=direction)
            if isinstance(result, Mapping):
                return {str(node): int(distance) for node, distance in result.items()}

        neighbor_fn = getattr(
            graph,
            "predecessors" if direction == "in" else "successors",
            None,
        )
        if not callable(neighbor_fn):
            neighbor_fn = getattr(graph, "neighbors", None)
        if not callable(neighbor_fn):
            return {}

        distances: dict[str, int] = {seed: 0}
        queue: list[str] = [seed]
        cursor = 0
        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            distance = distances[node]
            if distance >= max_hops:
                continue
            for neighbor in cast(Iterable[Any], neighbor_fn(node)):
                neighbor_id = str(neighbor)
                if neighbor_id in distances:
                    continue
                distances[neighbor_id] = distance + 1
                queue.append(neighbor_id)
        return {node: distance for node, distance in distances.items() if node != seed}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _graph_analysis_ctx(self) -> Any:
        """Build a duck-typed context for graph.analysis compute functions."""
        from types import SimpleNamespace

        return SimpleNamespace(
            snapshot_store=self._snapshot_store,
            projection_store=self._projection_store,
            projection_transformer=self._projection_transformer,
        )

    # ------------------------------------------------------------------
    # Snapshot symbol index helpers
    # ------------------------------------------------------------------

    def _ensure_snapshot_symbol_index(self, snapshot_id: str) -> None:
        if self._snapshot_symbol_index.has_snapshot(snapshot_id):
            return
        if self._register_snapshot_symbols_from_payload(snapshot_id):
            return
        snap = self._snapshot_store.load_snapshot(snapshot_id)
        if snap:
            self._snapshot_symbol_index.register_snapshot(snap)

    def _register_snapshot_symbols_from_payload(self, snapshot_id: str) -> bool:
        load_payload = getattr(
            self._snapshot_store,
            "load_snapshot_symbol_index_payload",
            None,
        )
        register_payload = getattr(
            self._snapshot_symbol_index,
            "register_snapshot_symbol_payload",
            None,
        )
        if not callable(load_payload) or not callable(register_payload):
            return False

        payload = load_payload(snapshot_id)
        if not isinstance(payload, Mapping):
            return False
        registered = register_payload(payload)
        if isinstance(registered, bool) and not registered:
            return False
        return self._snapshot_symbol_index.has_snapshot(snapshot_id)
