"""CacheFacade — cache operations extracted from FastCode.

Owns clear_cache, get_cache_stats, invalidate_scan_cache, refresh_index_cache,
and load_cached_repos.  Wraps CacheManager + VectorStore + RuntimeState;
delegates multi-repo rehydration to app.store.cache.rehydration.
"""

from __future__ import annotations

from typing import Any

from fastcode.app.indexing.embedder import CodeEmbedder
from fastcode.app.query.selection.retriever import HybridRetriever
from fastcode.app.store.artifacts.graph import GraphArtifactStore
from fastcode.app.store.cache.rehydration import (
    load_multi_repo_cache as _load_multi_repo_cache_impl,
)
from fastcode.app.store.cache.service import CacheManager
from fastcode.app.store.vectors.vector import VectorStore
from fastcode.graph.build import CodeGraphBuilder
from fastcode.runtime_support.runtime_state import RuntimeState


class CacheFacade:
    """Aggregated cache API for entry frames."""

    def __init__(
        self,
        cache_manager: CacheManager,
        vector_store: VectorStore,
        embedder: CodeEmbedder,
        retriever: HybridRetriever,
        graph_builder: CodeGraphBuilder,
        graph_artifact_store: GraphArtifactStore,
        state: RuntimeState,
    ) -> None:
        self._cache_manager = cache_manager
        self._vector_store = vector_store
        self._embedder = embedder
        self._retriever = retriever
        self._graph_builder = graph_builder
        self._graph_artifact_store = graph_artifact_store
        self._state = state

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def clear_cache(self) -> bool:
        """Clear the query cache. Returns True if successful."""
        with self._state._lock:
            return bool(self._cache_manager.clear())

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache_manager.get_stats()

    def invalidate_scan_cache(self) -> None:
        """Invalidate the vector store scan cache."""
        self._vector_store.invalidate_scan_cache()

    def refresh_index_cache(self) -> list[dict[str, Any]]:
        """Invalidate and rescan available indexes."""
        with self._state._lock:
            self._vector_store.invalidate_scan_cache()
            return self._vector_store.scan_available_indexes(use_cache=False)

    def load_cached_repos(self, repo_names: list[str] | None = None) -> bool:
        """Load pre-indexed repos from cache into memory."""
        with self._state._lock:
            loaded = _load_multi_repo_cache_impl(
                repo_names=repo_names,
                vector_store=self._vector_store,
                embedder=self._embedder,
                retriever=self._retriever,
                graph_builder=self._graph_builder,
                graph_artifact_store=self._graph_artifact_store,
                loaded_repositories=self._state.loaded_repositories,
            )
            if loaded:
                self._state.repo_loaded = True
                self._state.repo_indexed = True
                self._state.multi_repo_mode = True
            return loaded
