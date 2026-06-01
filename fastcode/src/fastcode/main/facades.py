"""Facade container and factory for injection into entry frames.

Holds the seven facade instances plus orchestration methods that span
multiple facades.  Created by the composition root (main/) and injected
into entry frames (api/, mcp/) so they never import from main/ directly.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any, Callable, cast

from fastcode.app.indexing.facade import IndexingFacade
from fastcode.app.indexing.loader import RepositoryLoader
from fastcode.app.indexing.pipeline.service import IndexPipeline
from fastcode.app.indexing.projection.transform import ProjectionTransformer
from fastcode.app.indexing.projection_facade import ProjectionFacade
from fastcode.app.indexing.publishing_facade import PublishingFacade
from fastcode.app.query.facade import QueryFacade
from fastcode.app.query.selection.retriever import HybridRetriever
from fastcode.app.store.artifacts.graph import GraphArtifactStore
from fastcode.app.store.cache_facade import CacheFacade
from fastcode.app.store.context_facade import ContextFacade
from fastcode.app.store.facade import StoreFacade
from fastcode.app.store.snapshots.snapshot import SnapshotStore
from fastcode.app.store.snapshots.projection import ProjectionStore
from fastcode.app.store.vectors.vector import VectorStore
from fastcode.runtime_support.runtime_state import RuntimeState

from .diagnostics import DiagnosticBuilder


@dataclass
class FacadeContainer:
    """Narrow facade holder injected into entry frames.

    Entry frames access attributes via duck typing — no import of this
    type from main/ is needed.
    """

    indexing: IndexingFacade
    query: QueryFacade
    store: StoreFacade
    context: ContextFacade
    cache: CacheFacade
    projection: ProjectionFacade
    publishing: PublishingFacade

    # Internal components needed by orchestration methods
    state: RuntimeState
    config: dict[str, Any]
    logger: logging.Logger
    vector_store: VectorStore
    graph_artifact_store: GraphArtifactStore
    retriever: HybridRetriever
    loader: RepositoryLoader
    snapshot_store: SnapshotStore
    projection_store: ProjectionStore
    projection_transformer: ProjectionTransformer
    _diagnostic_builder: DiagnosticBuilder

    # Lifecycle
    shutdown_fn: Callable[[], None]
    _apply_runtime_overrides_fn: Callable[..., None]

    # Orchestration -------------------------------------------------------

    def ensure_repos_ready(
        self, repos: list[str], *, allow_incremental: bool = True
    ) -> list[str]:
        """Ensure all repos are cloned (if URL), loaded, and indexed."""
        self.apply_env_ignore_patterns()
        ready_names: list[str] = []

        for source in repos:
            resolved_is_url = IndexPipeline._infer_is_url(source)
            name = self.store.repo_name_from_source(source, resolved_is_url)

            if self.store.is_repo_indexed(name):
                if not resolved_is_url and allow_incremental:
                    abs_path = os.path.abspath(source)
                    if os.path.isdir(abs_path):
                        try:
                            result = self.indexing.run_index_pipeline(
                                source=abs_path,
                                is_url=False,
                                publish=True,
                                enable_scip=True,
                            )
                            if result and result.get("status") not in {"reused"}:
                                self.logger.info(
                                    "Graceful update for '%s': %s", name, result
                                )
                        except Exception as e:
                            self.logger.warning(
                                "Graceful update failed for '%s': %s", name, e
                            )
                self.logger.info("Repo '%s' ready.", name)
                ready_names.append(name)
                continue

            self.logger.info("Repo '%s' not indexed. Preparing …", name)

            if resolved_is_url:
                self.logger.info("Cloning %s …", source)
                self.indexing.load_repository(source, is_url=True)
            else:
                abs_path = os.path.abspath(source)
                if not os.path.isdir(abs_path):
                    self.logger.error("Local path does not exist: %s", abs_path)
                    continue
                self.indexing.load_repository(abs_path, is_url=False)

            self.logger.info("Indexing '%s' …", name)
            self.indexing.index_repository(force=False)
            self.logger.info("Indexing '%s' complete.", name)
            ready_names.append(name)

        return ready_names

    def ensure_loaded(self, repo_names: list[str]) -> bool:
        """Ensure repos are loaded into memory (vectors + BM25 + graphs)."""
        if not self.state.repo_indexed or set(repo_names) != set(
            self.state.loaded_repositories.keys()
        ):
            self.logger.info("Loading repos into memory: %s", repo_names)
            return self.cache.load_cached_repos(repo_names=repo_names)
        return True

    def remove_repository(
        self, repo_name: str, delete_source: bool = True
    ) -> dict[str, Any]:
        with self.state._lock:
            return self._remove_repository_unlocked(repo_name, delete_source)

    def _remove_repository_unlocked(
        self, repo_name: str, delete_source: bool = True
    ) -> dict[str, Any]:
        deleted_files: list[str] = []
        freed_bytes = 0

        for artifact_path in self._repository_artifact_paths(repo_name):
            size = _artifact_size_bytes(artifact_path)
            if os.path.isdir(artifact_path):
                shutil.rmtree(artifact_path)
            else:
                os.remove(artifact_path)
            deleted_files.append(os.path.basename(artifact_path))
            freed_bytes += size
            self.logger.info(f"Deleted {artifact_path} ({size / (1024 * 1024):.2f} MB)")

        if self.vector_store.delete_repo_overview(repo_name):
            deleted_files.append("repository overview storage (entry)")
            self.logger.info(f"Deleted overview entry for {repo_name}")

        if delete_source:
            repo_root = getattr(
                self.loader, "safe_repo_root", self.config.get("repo_root", "./repos")
            )
            repo_dir = os.path.join(repo_root, repo_name)
            if os.path.isdir(repo_dir):
                dir_size = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fns in os.walk(repo_dir)
                    for f in fns
                )
                shutil.rmtree(repo_dir)
                deleted_files.append(f"repos/{repo_name}/")
                freed_bytes += dir_size
                self.logger.info(f"Deleted source directory {repo_dir}")

        self.vector_store.invalidate_scan_cache()

        return {
            "repo_name": repo_name,
            "deleted_files": deleted_files,
            "freed_bytes": freed_bytes,
            "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        }

    def build_diagnostic_bundle(self) -> dict[str, Any]:
        return self._diagnostic_builder.build_diagnostic_bundle()

    def apply_env_ignore_patterns(self) -> None:
        """Force-ignore environment-related paths before indexing."""
        repo_cfg = self.config.get("repository", {})
        ignore_patterns = list(repo_cfg.get("ignore_patterns", []))

        forced_patterns = [
            ".venv", "venv", ".env", "env",
            "**/.venv/**", "**/venv/**", "**/.env/**", "**/env/**",
        ]

        if repo_cfg.get("exclude_site_packages", False):
            forced_patterns.extend(["site-packages", "**/site-packages/**"])

        added = []
        for pattern in forced_patterns:
            if pattern not in ignore_patterns:
                ignore_patterns.append(pattern)
                added.append(pattern)

        if added:
            self.apply_repository_runtime_overrides(
                ignore_patterns=tuple(ignore_patterns)
            )
            self.logger.info("Added forced ignore patterns: %s", added)

    def apply_repository_runtime_overrides(
        self,
        *,
        ignore_patterns: tuple[str, ...] | None = None,
        exclude_site_packages: bool | None = None,
    ) -> None:
        """Apply repository-scanning policy overrides and refresh loader state."""
        self._apply_runtime_overrides_fn(
            ignore_patterns=ignore_patterns,
            exclude_site_packages=exclude_site_packages,
        )

    def shutdown(self) -> None:
        self.shutdown_fn()

    # Private helpers -----------------------------------------------------

    def _repository_artifact_paths(self, repo_name: str) -> list[str]:
        persist_dir = self.vector_store.persist_dir
        artifact_paths = [
            os.path.join(persist_dir, f"{repo_name}_manifest.json")
        ]

        vector_artifact_paths = getattr(
            self.vector_store, "vector_artifact_paths", None
        )
        if callable(vector_artifact_paths):
            artifact_paths.extend(cast(list[str], vector_artifact_paths(repo_name)))
        else:
            artifact_paths.extend(
                [
                    os.path.join(persist_dir, f"{repo_name}.faiss"),
                    os.path.join(persist_dir, f"{repo_name}_vector_manifest.json"),
                    os.path.join(persist_dir, f"{repo_name}_vector_shards"),
                ]
            )

        graph_artifact_paths = getattr(
            self.graph_artifact_store, "graph_artifact_paths", None
        )
        if callable(graph_artifact_paths):
            artifact_paths.extend(cast(list[str], graph_artifact_paths(repo_name)))
        else:
            artifact_paths.extend(
                [
                    os.path.join(persist_dir, f"{repo_name}_graphs.pkl"),
                    os.path.join(persist_dir, f"{repo_name}_graph_manifest.json"),
                    os.path.join(persist_dir, f"{repo_name}_graph_shards"),
                ]
            )

        metadata_artifact_paths = getattr(
            self.vector_store, "metadata_artifact_paths", None
        )
        if callable(metadata_artifact_paths):
            artifact_paths.extend(cast(list[str], metadata_artifact_paths(repo_name)))
        else:
            artifact_paths.extend(
                [
                    os.path.join(persist_dir, f"{repo_name}_metadata.pkl"),
                    os.path.join(persist_dir, f"{repo_name}_metadata_manifest.json"),
                    os.path.join(persist_dir, f"{repo_name}_metadata_shards"),
                ]
            )

        bm25_artifact_paths = getattr(self.retriever, "bm25_artifact_paths", None)
        if callable(bm25_artifact_paths):
            artifact_paths.extend(cast(list[str], bm25_artifact_paths(repo_name)))
        else:
            artifact_paths.extend(
                [
                    os.path.join(self.retriever.persist_dir, f"{repo_name}_bm25.pkl"),
                    os.path.join(self.retriever.persist_dir, f"{repo_name}_bm25_manifest.json"),
                    os.path.join(self.retriever.persist_dir, f"{repo_name}_bm25_shards"),
                ]
            )

        seen: set[str] = set()
        existing: list[str] = []
        for path in artifact_paths:
            if path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            existing.append(path)
        return existing


def _artifact_size_bytes(path: str) -> int:
    if os.path.isdir(path):
        return sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(path)
            for filename in filenames
        )
    return os.path.getsize(path) if os.path.exists(path) else 0


def facade_container_from_fastcode(fc: Any) -> FacadeContainer:
    """Extract a FacadeContainer from an existing FastCode instance.

    This is the bridge function called by the composition root to create
    the narrow container that gets injected into entry frames.
    """
    return FacadeContainer(
        indexing=fc.indexing,
        query=fc.query,
        store=fc.store,
        context=fc.context,
        cache=fc.cache,
        projection=fc.projection,
        publishing=fc.publishing,
        state=fc.state,
        config=fc.config,
        logger=fc.logger,
        vector_store=fc.vector_store,
        graph_artifact_store=fc.graph_artifact_store,
        retriever=fc.retriever,
        loader=fc.loader,
        snapshot_store=fc.snapshot_store,
        projection_store=fc.projection_store,
        projection_transformer=fc.projection_transformer,
        _diagnostic_builder=fc._diagnostic_builder,
        shutdown_fn=fc.shutdown,
        _apply_runtime_overrides_fn=fc.apply_repository_runtime_overrides,
    )
