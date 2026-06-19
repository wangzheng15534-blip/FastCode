"""
Main FastCode Class - Orchestrate all components
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import os
from typing import Any, cast

from fastcode.app.app_runtime import build_app_runtime
from fastcode.app.indexing.graph_mapper import document_overlay_node_records
from fastcode.app.indexing.pipeline.manifest import (
    build_file_manifest,
    collect_unchanged_elements,
    detect_file_changes,
    load_existing_metadata,
    load_file_manifest,
    save_file_manifest,
)
from fastcode.app.indexing.pipeline.redo_worker import RedoDeps, RedoWorker
from fastcode.app.indexing.runtime_contracts import DocumentGraphRuntime
from fastcode.app.store.cache.rehydration import (
    reconstruct_elements_from_metadata,
)
from fastcode.common.config import FastCodeConfig
from fastcode.infrastructure.graph_runtime.ladybug import LadybugGraphRuntime
from fastcode.infrastructure.storage.runtime import DBRuntime
from fastcode.ir.element import CodeElement
from fastcode.runtime_support.runtime_state import RuntimeState
from fastcode.utils.filesystem import ensure_dir

from .config import (
    bootstrap_runtime_config,
    config_to_runtime_mapping,
    setup_logging,
)
from .diagnostics import DiagnosticBuilder


class FastCode:
    """Main FastCode system for repository-level code understanding"""

    def __init__(self, config_path: str | None = None):
        """
        Initialize FastCode system

        Args:
            config_path: Path to configuration file (default: config/config.yaml)
        """
        # Resolve FastCode project root from package location.
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )

        # Load configuration via the config run_kit entry point.
        if config_path is None:
            possible_paths = [
                "config/config.yaml",
                "../config/config.yaml",
                os.path.join(os.path.dirname(__file__), "../../../config/config.yaml"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break

        self.runtime_config = bootstrap_runtime_config(config_path, project_root)
        self.config = config_to_runtime_mapping(self.runtime_config)

        # Evaluation-specific overrides (keep core system decoupled)
        self.eval_config = self.config.get("evaluation", {})
        self.eval_mode = self.eval_config.get("enabled", False)
        self.in_memory_index = self.eval_config.get("in_memory_index", False)

        # Ensure in-memory mode disables disk-based caches/persistence
        if self.in_memory_index:
            self._set_runtime_config(
                self.runtime_config.with_runtime_overrides(
                    in_memory_index=True,
                    cache_enabled=False,
                )
            )

        # Allow explicit cache disable via evaluation config
        if self.eval_config.get("disable_cache", False):
            self._set_runtime_config(
                self.runtime_config.with_runtime_overrides(cache_enabled=False)
            )

        # Setup logging
        self.logger = setup_logging(self.config)
        self.logger.info("Initializing FastCode system")

        # Runtime state (shared mutable state for facades)
        self.state = RuntimeState()

        # Resolve repo_root and ensure it exists.
        config_repo_root: str = str(self.config.get("repo_root") or ".")
        config_repo_root = os.path.abspath(config_repo_root)
        ensure_dir(config_repo_root)
        self.logger.info(f"Configured repo_root: {config_repo_root}")

        # --- Effect facilities (built here; entry_frame -> effect_facility is legal) ---
        persist_dir = self.config.get("vector_store", {}).get("persist_directory") or ""
        storage_cfg: dict[str, Any] = self.config.get("storage", {}) or {}
        db_runtime = DBRuntime.from_storage_config(
            sqlite_path=os.path.join(os.path.abspath(persist_dir), "lineage.db"),
            storage_cfg=storage_cfg,
        )
        graph_runtime: DocumentGraphRuntime | None = None
        try:
            graph_runtime = LadybugGraphRuntime(self.config)
        except ImportError:
            self.logger.info(
                "LadybugGraphRuntime unavailable — graph persistence disabled"
            )

        # --- use_flow assembly (delegated; main imports no meaning_core/effect_tool) ---
        rt = build_app_runtime(
            config=self.config,
            logger=self.logger,
            db_runtime=db_runtime,
            graph_runtime=graph_runtime,
            state=self.state,
            eval_config=self.eval_config,
            set_repo_root_fn=self._set_repo_root,
            apply_env_ignore_patterns_fn=self.apply_env_ignore_patterns,
        )
        self._attach_runtime(rt)

        # --- Redo worker (postgres only); needs the assembled facades ---
        self._redo_worker: RedoWorker | None = None
        if db_runtime.backend == "postgres":
            poll_interval = int(storage_cfg.get("redo_poll_interval_seconds", 30))
            self._redo_worker = RedoWorker(
                RedoDeps(
                    snapshot_store=self.snapshot_store,
                    publishing=self.publishing,
                    terminus_publisher=self.terminus_publisher,
                    projection_service=self.projection_service,
                    config=self.config,
                ),
                poll_interval_seconds=poll_interval,
            )
            self._redo_worker.start()
        # Keep the publishing service's redo_worker handle in sync.
        self.publishing_service.redo_worker = self._redo_worker  # type: ignore[attr-defined]

        # --- DiagnosticBuilder ---
        self._diagnostic_builder = DiagnosticBuilder(
            config=self.config,
            vector_store=self.vector_store,
            snapshot_store=self.snapshot_store,
            manifest_store=self.manifest_store,
            state=self.state,
            logger=self.logger,
            eval_config=self.eval_config,
            index_run_store=self.index_run_store,
            projection_store=self.projection_store,
            cache_manager=self.cache_manager,
            loader=self.loader,
        )

        # Multi-repository state lives on self.state

    def _attach_runtime(self, rt: Any) -> None:
        """Copy the assembled use_flow graph onto self for legacy attribute access."""
        for name in (
            "loader", "parser", "embedder", "vector_store", "indexer",
            "graph_builder", "graph_artifact_store", "ir_graph_builder",
            "retriever", "query_processor", "answer_generator", "cache_manager",
            "snapshot_store", "manifest_store", "index_run_store",
            "unit_artifact_store", "file_artifact_store", "terminus_publisher",
            "projection_transformer", "projection_store", "snapshot_symbol_index",
            "store", "pg_retrieval_store", "doc_ingester",
            "semantic_helper_runtime", "scip_indexer_runtime",
            "semantic_resolver_registry", "pipeline", "direct_indexer",
            "multi_repo_direct_indexer", "indexing", "projection_service",
            "projection", "publishing_service", "publishing",
            "query_handler", "query", "context", "cache", "graph_runtime",
        ):
            setattr(self, name, getattr(rt, name))

    def _set_runtime_config(self, config: FastCodeConfig) -> None:
        """Replace canonical runtime config and refresh the shell mapping view."""
        self.runtime_config = config
        self.config = config_to_runtime_mapping(config)
        self.eval_config = self.config.get("evaluation", {})
        self.eval_mode = self.eval_config.get("enabled", False)
        self.in_memory_index = self.eval_config.get("in_memory_index", False)

    def _set_repo_root(self, repo_root: str) -> None:
        """Explicit runtime mutation point for repository-root binding."""
        self._set_runtime_config(
            self.runtime_config.with_runtime_overrides(repo_root=repo_root)
        )

    def apply_repository_runtime_overrides(
        self,
        *,
        ignore_patterns: tuple[str, ...] | None = None,
        exclude_site_packages: bool | None = None,
    ) -> None:
        """Apply repository-scanning policy overrides and refresh loader state."""
        self._set_runtime_config(
            self.runtime_config.with_runtime_overrides(
                repository_ignore_patterns=ignore_patterns,
                repository_exclude_site_packages=exclude_site_packages,
            )
        )
        repository_cfg = self.config.get("repository", {})
        self.loader.repo_config = repository_cfg
        self.loader.ignore_patterns = repository_cfg.get("ignore_patterns", [])

    @staticmethod
    def _infer_is_url(source: str) -> bool:
        return IndexPipeline._infer_is_url(source)

    def _state_lock(self) -> Any:
        return self.state._lock

    def _state_read_lock(self) -> Any:
        return self.state._lock.read_lock()

    def refresh_index_cache(self) -> list[dict[str, Any]]:
        """Invalidate and rescan available indexes."""
        return self.cache.refresh_index_cache()

    def clear_cache(self) -> bool:
        """Clear the query cache. Returns True if successful."""
        return self.cache.clear_cache()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_cache_stats()

    def invalidate_scan_cache(self) -> None:
        """Invalidate the vector store scan cache."""
        self.cache.invalidate_scan_cache()

    def load_cached_repos(self, repo_names: list[str] | None = None) -> bool:
        """Load pre-indexed repos from cache into memory."""
        return self.cache.load_cached_repos(repo_names=repo_names)

    def apply_env_ignore_patterns(self) -> None:
        """Force-ignore environment-related paths before indexing."""
        repo_cfg = self.config.get("repository", {})
        ignore_patterns = list(repo_cfg.get("ignore_patterns", []))

        forced_patterns = [
            ".venv",
            "venv",
            ".env",
            "env",
            "**/.venv/**",
            "**/venv/**",
            "**/.env/**",
            "**/env/**",
        ]

        if repo_cfg.get("exclude_site_packages", False):
            forced_patterns.extend(
                [
                    "site-packages",
                    "**/site-packages/**",
                ]
            )

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

    def ensure_repos_ready(
        self, repos: list[str], *, allow_incremental: bool = True
    ) -> list[str]:
        """Ensure all repos are cloned (if URL), loaded, and indexed.

        Returns the list of canonical repo names that are ready.
        """
        self.apply_env_ignore_patterns()
        ready_names: list[str] = []

        for source in repos:
            resolved_is_url = self._infer_is_url(source)
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
                            if result and result.get("status") != "reused":
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

    def _checkout_target_ref(
        self, ref: str | None = None, commit: str | None = None
    ) -> None:
        return self.pipeline._checkout_target_ref(ref=ref, commit=commit)

    def _resolve_snapshot_ref(
        self,
        repo_name: str,
        requested_ref: str | None = None,
        requested_commit: str | None = None,
    ) -> dict[str, Any]:
        return self.pipeline._resolve_snapshot_ref(
            repo_name,
            requested_ref=requested_ref,
            requested_commit=requested_commit,
        )

    def _build_git_meta(self, snapshot_ref: dict[str, Any]) -> dict[str, Any]:
        return self.pipeline._build_git_meta(snapshot_ref)

    def _previous_snapshot_symbol_versions(
        self,
        repo_name: str,
        ref_name: str,
        current_snapshot_id: str,
    ) -> dict[str, str] | None:
        return self.pipeline._previous_snapshot_symbol_versions(
            repo_name, ref_name, current_snapshot_id
        )

    def _load_artifacts_by_key(self, artifact_key: str) -> bool:
        return self.pipeline._load_artifacts_by_key(artifact_key)

    @staticmethod
    def _projection_scope_key(
        scope_kind: str,
        snapshot_id: str,
        query: str | None,
        target_id: str | None,
        filters: dict[str, Any] | None,
    ) -> str:
        return ProjectionService.projection_scope_key(
            scope_kind, snapshot_id, query, target_id, filters
        )

    @staticmethod
    def _projection_params_hash(scope: Any, projection_algo_version: str = "v1") -> str:
        return ProjectionService.projection_params_hash(scope, projection_algo_version)

    def _try_load_from_cache(self) -> bool:
        """Try to load indexed data from cache for single repository."""
        if not self._should_use_cache():
            self.logger.info("Cache loading disabled (ephemeral/evaluation mode)")
            return False
        return _try_load_from_cache_impl(
            cache_name=self._get_cache_name(),
            vector_store=self.vector_store,
            retriever=self.retriever,
            graph_builder=self.graph_builder,
            graph_artifact_store=self.graph_artifact_store,
            log_statistics_fn=self._log_statistics,
        )

    def _save_to_cache(self, cache_name: str | None = None):
        """Save indexed data to cache."""
        if not self._should_persist_indexes():
            self.logger.info("Cache save disabled (ephemeral/evaluation mode)")
            return
        _save_to_cache_impl(
            cache_name=cache_name or self._get_cache_name(),
            vector_store=self.vector_store,
        )

    def _get_cache_name(self) -> str:
        """Get cache name for current repository"""
        return self.state.repo_info.get("name", "default")

    def _get_repo_hash(self) -> str:
        """Get hash of repository for cache key"""
        return self.state.repo_info.get(
            "commit", self.state.repo_info.get("name", "default")
        )

    def _reconstruct_elements_from_metadata(self) -> list[CodeElement]:
        """Reconstruct CodeElement objects from vector store metadata."""
        return reconstruct_elements_from_metadata(self.vector_store)

    def _log_statistics(self):
        """Log indexing statistics"""
        stats = {
            "vector_count": self.vector_store.get_count(),
            "graph_stats": self.graph_builder.get_graph_stats(),
        }

        self.logger.info(f"Statistics: {stats}")

    def _is_ephemeral_mode(self) -> bool:
        """Return True when running in evaluation/in-memory mode."""
        return self.in_memory_index or getattr(self.vector_store, "in_memory", False)

    def _should_use_cache(self) -> bool:
        """Determine whether cache/index reuse is allowed."""
        if self.eval_config.get("disable_cache", False):
            return False
        return not self._is_ephemeral_mode()

    def _should_persist_indexes(self) -> bool:
        """Determine whether indexes should be persisted to disk."""
        if self.eval_config.get("disable_persistence", False):
            return False
        return not self._is_ephemeral_mode()

    def _has_active_doc_persistence(self) -> bool:
        """Return True when doc ingestion has at least one active sink."""
        return self.snapshot_store.db_runtime.backend == "postgres" or bool(
            getattr(self.graph_runtime, "enabled", False)
        )

    def _should_ingest_docs(self) -> bool:
        """Only ingest docs when the feature is enabled and results can be persisted."""
        return (
            bool(getattr(self.doc_ingester, "enabled", False))
            and self._has_active_doc_persistence()
        )

    def _sync_doc_overlay(
        self,
        *,
        chunks: list[dict[str, Any]],
        mentions: list[dict[str, Any]],
        warnings: list[str],
    ) -> None:
        """Best-effort Ladybug sync with explicit failure reporting."""
        if not chunks or not getattr(self.graph_runtime, "enabled", False):
            return
        try:
            if self.graph_runtime is None:
                msg = "Graph runtime not available"
                raise RuntimeError(msg)
            synced = self.graph_runtime.sync_nodes(
                nodes=document_overlay_node_records(chunks=chunks, mentions=mentions)
            )
        except Exception as e:
            warnings.append(f"ladybug_doc_sync_failed: {e}")
            return
        if not synced:
            warnings.append("ladybug_doc_sync_failed")

    def build_diagnostic_bundle(self) -> dict[str, Any]:
        """Build a support-safe runtime diagnostic bundle."""
        return self._diagnostic_builder.build_diagnostic_bundle()

    # ------------------------------------------------------------------
    # Incremental indexing
    # ------------------------------------------------------------------

    def _build_file_manifest(
        self, elements: list[CodeElement], repo_root: str
    ) -> dict[str, Any]:
        return build_file_manifest(
            elements, repo_root, repo_name=self.state.repo_info.get("name", "")
        )

    def _save_file_manifest(self, repo_name: str, manifest: dict[str, Any]) -> None:
        save_file_manifest(manifest, repo_name, self.vector_store.persist_dir)

    def _load_file_manifest(self, repo_name: str) -> dict[str, Any] | None:
        return load_file_manifest(repo_name, self.vector_store.persist_dir)

    def _load_existing_metadata(self, repo_name: str) -> list[dict[str, Any]]:
        return load_existing_metadata(
            repo_name, self.vector_store, self.vector_store.persist_dir
        )

    @staticmethod
    def _artifact_size_bytes(path: str) -> int:
        if os.path.isdir(path):
            return sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(path)
                for filename in filenames
            )
        return os.path.getsize(path) if os.path.exists(path) else 0

    def _repository_artifact_paths(self, repo_name: str) -> list[str]:
        artifact_paths = [
            os.path.join(self.vector_store.persist_dir, f"{repo_name}_manifest.json")
        ]

        vector_artifact_paths = getattr(
            self.vector_store, "vector_artifact_paths", None
        )
        if callable(vector_artifact_paths):
            artifact_paths.extend(cast(list[str], vector_artifact_paths(repo_name)))
        else:
            artifact_paths.extend(
                [
                    os.path.join(self.vector_store.persist_dir, f"{repo_name}.faiss"),
                    os.path.join(
                        self.vector_store.persist_dir,
                        f"{repo_name}_vector_manifest.json",
                    ),
                    os.path.join(
                        self.vector_store.persist_dir, f"{repo_name}_vector_shards"
                    ),
                ]
            )

        graph_artifact_paths = getattr(
            getattr(self, "graph_artifact_store", None), "graph_artifact_paths", None
        )
        if callable(graph_artifact_paths):
            artifact_paths.extend(cast(list[str], graph_artifact_paths(repo_name)))
        else:
            artifact_paths.extend(
                [
                    os.path.join(
                        self.vector_store.persist_dir, f"{repo_name}_graphs.pkl"
                    ),
                    os.path.join(
                        self.vector_store.persist_dir,
                        f"{repo_name}_graph_manifest.json",
                    ),
                    os.path.join(
                        self.vector_store.persist_dir, f"{repo_name}_graph_shards"
                    ),
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
                    os.path.join(
                        self.vector_store.persist_dir, f"{repo_name}_metadata.pkl"
                    ),
                    os.path.join(
                        self.vector_store.persist_dir,
                        f"{repo_name}_metadata_manifest.json",
                    ),
                    os.path.join(
                        self.vector_store.persist_dir, f"{repo_name}_metadata_shards"
                    ),
                ]
            )

        bm25_artifact_paths = getattr(self.retriever, "bm25_artifact_paths", None)
        if callable(bm25_artifact_paths):
            artifact_paths.extend(cast(list[str], bm25_artifact_paths(repo_name)))
        else:
            artifact_paths.extend(
                [
                    os.path.join(self.retriever.persist_dir, f"{repo_name}_bm25.pkl"),
                    os.path.join(
                        self.retriever.persist_dir, f"{repo_name}_bm25_manifest.json"
                    ),
                    os.path.join(
                        self.retriever.persist_dir, f"{repo_name}_bm25_shards"
                    ),
                ]
            )

        seen: set[str] = set()
        existing_paths: list[str] = []
        for path in artifact_paths:
            if path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            existing_paths.append(path)
        return existing_paths

    def _detect_file_changes(
        self, repo_name: str, current_files: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        return detect_file_changes(
            repo_name, current_files, self.vector_store.persist_dir
        )

    def _collect_unchanged_elements(
        self,
        manifest: dict[str, Any],
        unchanged_files: list[str],
        existing_metadata: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        return collect_unchanged_elements(manifest, unchanged_files, existing_metadata)

    def cleanup(self):
        """Cleanup resources"""
        with self._state_lock():
            self.shutdown()
            self.loader.cleanup()
            self.logger.info("Cleanup complete")

    def shutdown(self):
        """Stop background workers."""
        with self._state_lock():
            if self._redo_worker is not None:
                self._redo_worker.stop()
            graph_rt = getattr(self, "graph_runtime", None)
            if graph_rt is not None:
                graph_rt.close()

    def _get_full_dialogue_history(
        self, session_id: str | None, enable_multi_turn: bool = False
    ) -> list[dict[str, Any]] | None:
        return self.query_handler._get_full_dialogue_history(
            session_id, enable_multi_turn
        )

    def _get_next_turn_number(self, session_id: str) -> int:
        return self.query_handler._get_next_turn_number(session_id)

    def remove_repository(
        self, repo_name: str, delete_source: bool = True
    ) -> dict[str, Any]:
        with self._state_lock():
            return self._remove_repository_unlocked(repo_name, delete_source)

    def _remove_repository_unlocked(
        self, repo_name: str, delete_source: bool = True
    ) -> dict[str, Any]:
        """
        Fully remove a repository: vector index files, BM25, graphs,
        repo overview, and optionally the cloned source code.

        Args:
            repo_name: Name of the repository to remove
            delete_source: If True, also remove ./repos/<repo_name>

        Returns:
            Dict with deleted files and freed bytes
        """
        import shutil

        deleted_files: list[str] = []
        freed_bytes = 0

        for artifact_path in self._repository_artifact_paths(repo_name):
            size = self._artifact_size_bytes(artifact_path)
            if os.path.isdir(artifact_path):
                shutil.rmtree(artifact_path)
            else:
                os.remove(artifact_path)
            deleted_files.append(os.path.basename(artifact_path))
            freed_bytes += size
            self.logger.info(f"Deleted {artifact_path} ({size / (1024 * 1024):.2f} MB)")

        # Remove overview entry from repository overview storage
        if self.vector_store.delete_repo_overview(repo_name):
            deleted_files.append("repository overview storage (entry)")
            self.logger.info(f"Deleted overview entry for {repo_name}")

        # Remove cloned source code
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

        # Invalidate scan cache
        self.vector_store.invalidate_scan_cache()

        return {
            "repo_name": repo_name,
            "deleted_files": deleted_files,
            "freed_bytes": freed_bytes,
            "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        }
