"""
QueryFacade -- narrow API surface extracted from FastCode.

Wraps QueryPipeline + VectorStore + CodeGraphBuilder with 6 public methods
and 2 internal helpers.  ``query``, ``query_snapshot``, and ``query_stream``
acquire the RuntimeState read lock; ``search_symbols``, ``get_file_structure``,
and ``walk_call_chain`` operate on loaded metadata without locking.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastcode.app.indexing.pipeline.service import IndexPipeline
    from fastcode.app.query.orchestration.handler import QueryPipeline
    from fastcode.app.query.selection.retriever import HybridRetriever
    from fastcode.app.store.snapshots.snapshot import SnapshotStore
    from fastcode.app.store.vectors.vector import VectorStore
    from fastcode.graph.build import CodeGraphBuilder
    from fastcode.ir.element import CodeElement
    from fastcode.ir.graph import IRGraphBuilder
    from fastcode.ir.types import IRSnapshot
    from fastcode.runtime_support.runtime_state import RuntimeState
    from fastcode.semantic.symbol_index import SnapshotSymbolIndex


class QueryFacade:
    """Facade for query, search, and graph-walk operations.

    Delegates query execution to ``QueryPipeline``, symbol/file/call-chain
    lookups to ``VectorStore`` and ``CodeGraphBuilder``.
    """

    def __init__(
        self,
        query_handler: QueryPipeline,
        vector_store: VectorStore,
        graph_builder: CodeGraphBuilder,
        snapshot_store: SnapshotStore,
        ir_graph_builder: IRGraphBuilder,
        snapshot_symbol_index: SnapshotSymbolIndex,
        pipeline: IndexPipeline,
        state: RuntimeState,
    ) -> None:
        self._query_handler = query_handler
        self._vector_store = vector_store
        self._graph_builder = graph_builder
        self._snapshot_store = snapshot_store
        self._ir_graph_builder = ir_graph_builder
        self._snapshot_symbol_index = snapshot_symbol_index
        self._pipeline = pipeline
        self._state = state

    # ------------------------------------------------------------------
    # Properties for external access (e.g. entry frames, callback wiring)
    # ------------------------------------------------------------------

    @property
    def query_handler(self) -> QueryPipeline:
        return self._query_handler

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def graph_builder(self) -> CodeGraphBuilder:
        return self._graph_builder

    # ------------------------------------------------------------------
    # Read-lock-guarded query delegations
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        repo_filter: list[str] | None = None,
        session_id: str | None = None,
        enable_multi_turn: bool | None = None,
        use_agency_mode: bool | None = None,
        prompt_builder: Callable[
            [str, str, dict[str, Any] | None, list[dict[str, Any]] | None], str
        ]
        | None = None,
    ) -> dict[str, Any]:
        if use_agency_mode:
            from fastcode.common.feature_lifecycle import CapabilityRegistry

            CapabilityRegistry.check("agency_mode")
        if enable_multi_turn:
            from fastcode.common.feature_lifecycle import CapabilityRegistry

            CapabilityRegistry.check("multi_turn_generation")
        with self._state.read_lock():
            return self._query_handler.query(
                question=question,
                filters=filters,
                repo_filter=repo_filter,
                session_id=session_id,
                enable_multi_turn=enable_multi_turn,
                use_agency_mode=use_agency_mode,
                prompt_builder=prompt_builder,
            )

    def query_snapshot(
        self,
        question: str,
        repo_name: str | None = None,
        ref_name: str | None = None,
        snapshot_id: str | None = None,
        filters: dict[str, Any] | None = None,
        session_id: str | None = None,
        enable_multi_turn: bool | None = None,
    ) -> dict[str, Any]:
        with self._state.read_lock():
            return self._query_handler.query_snapshot(
                question=question,
                repo_name=repo_name,
                ref_name=ref_name,
                snapshot_id=snapshot_id,
                filters=filters,
                session_id=session_id,
                enable_multi_turn=enable_multi_turn,
            )

    def query_stream(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        repo_filter: list[str] | None = None,
        session_id: str | None = None,
        enable_multi_turn: bool | None = None,
        use_agency_mode: bool | None = None,
        prompt_builder: Callable[
            [str, str, dict[str, Any] | None, list[dict[str, Any]] | None], str
        ]
        | None = None,
    ) -> Generator[tuple[str | None, dict[str, Any] | None], Any, None]:
        snapshot_value = (filters or {}).get("snapshot_id")
        snapshot_id = (
            snapshot_value
            if isinstance(snapshot_value, str) and snapshot_value
            else None
        )

        if snapshot_id:
            with self._state.read_lock():
                snapshot_record = self._snapshot_store.get_snapshot_record(snapshot_id)
                if not snapshot_record:
                    msg = f"snapshot not found: {snapshot_id}"
                    raise RuntimeError(msg)
                artifact_key = snapshot_record.artifact_key
                loaded_artifacts = self._pipeline.load_snapshot_artifacts_handle(
                    artifact_key,
                    snapshot_id=snapshot_id,
                )
                if loaded_artifacts is None:
                    msg = f"failed to load artifacts for snapshot: {snapshot_id}"
                    raise RuntimeError(msg)
                self._query_handler._ensure_snapshot_symbol_index(snapshot_id)

                merged_filters = dict(filters or {})
                merged_filters["snapshot_id"] = snapshot_id
                merged_filters["artifact_key"] = getattr(
                    loaded_artifacts,
                    "artifact_key",
                    artifact_key,
                )
                retriever = loaded_artifacts.retriever

            yield from self._query_handler.query_stream(
                question=question,
                filters=merged_filters,
                repo_filter=repo_filter,
                session_id=session_id,
                enable_multi_turn=enable_multi_turn,
                use_agency_mode=use_agency_mode,
                prompt_builder=prompt_builder,
                retriever=retriever,
            )
            return

        with self._state.read_lock():
            yield from self._query_handler.query_stream(
                question=question,
                filters=filters,
                repo_filter=repo_filter,
                session_id=session_id,
                enable_multi_turn=enable_multi_turn,
                use_agency_mode=use_agency_mode,
                prompt_builder=prompt_builder,
            )

    # ------------------------------------------------------------------
    # Symbol / file / call-chain lookups (no lock)
    # ------------------------------------------------------------------

    def search_symbols(
        self, name: str, *, symbol_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Search symbols by name with ranking: exact > prefix > contains."""
        query_lower = name.lower()
        exact: list[Any] = []
        prefix: list[Any] = []
        contains: list[Any] = []
        for meta in self._vector_store.metadata:
            elem_name = meta.get("name", "")
            elem_type = meta.get("type", "")
            if elem_type == "repository_overview":
                continue
            if symbol_type and elem_type != symbol_type:
                continue
            name_lower = elem_name.lower()
            if name_lower == query_lower:
                exact.append(meta)
            elif name_lower.startswith(query_lower):
                prefix.append(meta)
            elif query_lower in name_lower:
                contains.append(meta)
        return (exact + prefix + contains)[:20]

    def get_file_structure(self, file_path: str) -> dict[str, Any] | None:
        """Get structure summary of a file from loaded metadata."""
        matching: list[Any] = []
        for meta in self._vector_store.metadata:
            rel = meta.get("relative_path", "")
            if meta.get("type") == "repository_overview":
                continue
            if rel.endswith(file_path) or file_path in rel:
                matching.append(meta)
        if not matching:
            return None
        files = [m for m in matching if m.get("type") == "file"]
        classes = [m for m in matching if m.get("type") == "class"]
        functions = [m for m in matching if m.get("type") == "function"]
        return {
            "file": files[0] if files else matching[0],
            "files": files,
            "classes": classes,
            "functions": functions,
        }

    def walk_call_chain(
        self,
        symbol_name: str,
        *,
        direction: str = "both",
        max_hops: int = 2,
    ) -> dict[str, Any] | None:
        """Trace call chain for a symbol using the graph builder."""
        max_hops = min(max_hops, 5)
        gb = self._graph_builder
        name_lower = symbol_name.lower()
        target_id: str | None = None
        target_elem: Any = None

        # Exact match via element_by_name
        elem = gb.element_by_name.get(symbol_name)
        if elem:
            target_elem, target_id = elem, elem.id

        # Case-insensitive fallback
        if not target_id:
            for eid, e in gb.element_by_id.items():
                if e.name.lower() == name_lower:
                    target_elem, target_id = e, eid
                    break

        # Partial match fallback
        if not target_id:
            for eid, e in gb.element_by_id.items():
                if name_lower in e.name.lower():
                    target_elem, target_id = e, eid
                    break

        if not target_id or not target_elem:
            return None

        result: dict[str, Any] = {
            "name": target_elem.name,
            "type": target_elem.type,
            "path": getattr(target_elem, "relative_path", ""),
            "start_line": getattr(target_elem, "start_line", ""),
            "callers": [],
            "callees": [],
        }

        def _walk(
            element_id: str,
            walk_direction: str,
            hops_left: int,
            entries: list[dict[str, Any]],
            visited: set[str] | None = None,
            indent: int = 2,
        ) -> None:
            if visited is None:
                visited = {element_id}
            neighbors = (
                gb.get_callers(element_id)
                if walk_direction == "callers"
                else gb.get_callees(element_id)
            )
            if not neighbors:
                entries.append({"name": "(none)", "indent": indent})
                return
            for nid in neighbors:
                if nid in visited:
                    continue
                visited.add(nid)
                n_elem = gb.element_by_id.get(nid)
                if n_elem:
                    loc = (
                        f"{getattr(n_elem, 'relative_path', '')}:L{getattr(n_elem, 'start_line', '')}"
                        if getattr(n_elem, "relative_path", "")
                        else ""
                    )
                    entries.append(
                        {
                            "name": n_elem.name,
                            "loc": loc,
                            "indent": indent,
                        }
                    )
                    if hops_left > 1:
                        _walk(
                            nid,
                            walk_direction,
                            hops_left - 1,
                            entries,
                            visited,
                            indent + 1,
                        )

        if direction in ("callers", "both"):
            _walk(target_id, "callers", max_hops, result["callers"])

        if direction in ("callees", "both"):
            _walk(target_id, "callees", max_hops, result["callees"])

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _escalate_query_semantics(
        self,
        *,
        snapshot_id: str,
        retrieved: list[dict[str, Any]],
        processed_query: Any,
        budget: str,
        retriever: HybridRetriever | None = None,
        graph_builder: CodeGraphBuilder | None = None,
    ) -> dict[str, Any]:
        snapshot = self._snapshot_store.load_snapshot(snapshot_id)
        if snapshot is None:
            return {
                "status": "skipped",
                "reason": "snapshot_not_found",
                "budget": budget,
                "rerun_retrieval": False,
            }

        target_paths: set[str] = set()
        for item in retrieved[:10]:
            element = item.get("element") or {}
            path = element.get("relative_path") or element.get("file_path")
            if isinstance(path, str) and path:
                target_paths.add(path)

        query_filters = getattr(processed_query, "filters", {}) or {}
        file_path_filter = query_filters.get("file_path")
        if isinstance(file_path_filter, str) and file_path_filter:
            target_paths.add(file_path_filter)

        if not target_paths:
            return {
                "status": "skipped",
                "reason": "no_target_paths",
                "budget": budget,
                "rerun_retrieval": False,
            }

        active_retriever = retriever or self._query_handler.retriever
        active_graph_builder = graph_builder or self._graph_builder
        elements = list(active_graph_builder.element_by_id.values())
        warnings: list[str] = []
        upgraded_snapshot = self._apply_semantic_resolvers(
            snapshot=snapshot,
            elements=elements,
            graph_context=active_graph_builder,
            target_paths=target_paths,
            warnings=warnings,
            budget=budget,
        )
        self._snapshot_symbol_index.register_snapshot(upgraded_snapshot)
        upgraded_ir_graphs = self._ir_graph_builder.build_graphs(upgraded_snapshot)
        active_retriever.set_ir_graphs(upgraded_ir_graphs, snapshot_id=snapshot_id)

        semantic_runs = list(
            (upgraded_snapshot.metadata or {}).get("semantic_resolver_runs", [])
        )
        status = "degraded" if warnings else "applied"
        logger = logging.getLogger(__name__)
        logger.info(
            "Query semantic escalation completed",
            extra={
                "fc_event": "semantic_escalation",
                "snapshot_id": snapshot_id,
                "semantic_budget": budget,
                "semantic_status": status,
                "target_path_count": len(target_paths),
                "resolver_runs": len(semantic_runs),
                "warning_count": len(warnings),
            },
        )
        return {
            "status": status,
            "budget": budget,
            "target_path_count": len(target_paths),
            "target_paths": sorted(target_paths),
            "warnings": warnings,
            "resolver_runs": len(semantic_runs),
            "rerun_retrieval": True,
        }

    def _apply_semantic_resolvers(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        graph_context: CodeGraphBuilder | None,
        target_paths: set[str],
        warnings: list[str],
        budget: str = "changed_files",
    ) -> IRSnapshot:
        return self._pipeline._apply_semantic_resolvers(
            snapshot=snapshot,
            elements=elements,
            graph_context=graph_context,
            target_paths=target_paths,
            warnings=warnings,
            budget=budget,
        )
