"""
Main FastCode Class - Orchestrate all components
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import json
import logging
import os
import pickle
import threading
import time
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime
from typing import Any, cast

from rank_bm25 import BM25Okapi

from ..graph.build import CodeGraphBuilder
from ..graph_runtime import LadybugGraphRuntime
from ..indexing.doc_ingester import KeyDocIngester
from ..indexing.embedder import CodeEmbedder
from ..indexing.global_builder import GlobalIndexBuilder
from ..indexing.indexer import CodeIndexer
from ..indexing.loader import RepositoryLoader
from ..indexing.parser import CodeParser
from ..indexing.pipeline import IndexPipeline
from ..indexing.projection import ProjectionService
from ..indexing.projection_transform import ProjectionTransformer
from ..indexing.publishing import PublishingService
from ..indexing.redo_worker import RedoWorker
from ..indexing.terminus import TerminusPublisher
from ..ir.element import (
    CodeElement,
    CodeElementMeta,
    deserialize_code_element,
    serialize_code_element,
)
from ..ir.graph import IRGraphBuilder
from ..ir.types import IRSnapshot
from ..module_resolver import ModuleResolver
from ..query.answer import AnswerGenerator
from ..query.handler import QueryPipeline
from ..query.processor import QueryProcessor
from ..retrieval.core import snapshot as _snapshot
from ..retrieval.core.agent_context import (
    ContextBundle,
    HandoffArtifact,
    TurnJournal,
    WorkingMemoryArtifact,
)
from ..retrieval.core.context_compiler import (
    build_activation_record,
    build_handoff_from_working_memory,
    expand_bundle_source_ref,
)
from ..retrieval.core.context_compiler import (
    build_context_bundle as build_context_bundle_artifact,
)
from ..retrieval.core.context_compiler import (
    render_context_bundle as render_context_bundle_artifact,
)
from ..retrieval.hybrid import HybridRetriever
from ..schemas.config import FastCodeConfig, config_from_mapping
from ..scip.symbol_resolver import SymbolResolver
from ..semantic import build_default_semantic_resolver_registry
from ..semantic.symbol_index import SnapshotSymbolIndex
from ..store.cache import CacheManager
from ..store.index_run import IndexRunStore
from ..store.manifest import ManifestStore
from ..store.pg_retrieval import PgRetrievalStore
from ..store.projection import ProjectionStore
from ..store.records import (
    ContextActivationRecord,
    HandoffArtifactRecord,
    ManifestRecord,
    SCIPArtifactRecord,
    SnapshotRefRecord,
)
from ..store.snapshot import SnapshotStore
from ..store.unit_artifacts import UnitArtifactStore
from ..store.vector import VectorStore
from ..utils import (
    as_float32_matrix,
    config_to_legacy_dict,
    ensure_dir,
    load_runtime_config,
    normalize_path,
    prepare_runtime_config_mapping,
    setup_logging,
)

_STATE_LOCK_LOGGER = logging.getLogger(f"{__name__}.state_lock")


class _ReadWriteStateLock:
    """Reentrant writer lock with shared read sections for immutable queries."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._readers = 0
        self._reader_depths: dict[int, int] = {}
        self._writer: int | None = None
        self._write_depth = 0

    def __enter__(self) -> "_ReadWriteStateLock":
        self._acquire_write()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._release_write()

    @contextmanager
    def read_lock(self) -> Generator[None, None, None]:
        self._acquire_read()
        try:
            yield
        finally:
            self._release_read()

    @contextmanager
    def write_lock(self) -> Generator[None, None, None]:
        self._acquire_write()
        try:
            yield
        finally:
            self._release_write()

    def _acquire_read(self) -> None:
        ident = threading.get_ident()
        _STATE_LOCK_LOGGER.debug(
            "Acquiring service state read lock",
            extra={"fc_event": "service_lock_acquire", "lock_mode": "read"},
        )
        with self._condition:
            if self._writer == ident:
                self._readers += 1
                self._reader_depths[ident] = self._reader_depths.get(ident, 0) + 1
                _STATE_LOCK_LOGGER.debug(
                    "Acquired service state read lock",
                    extra={
                        "fc_event": "service_lock_acquired",
                        "lock_mode": "read",
                        "reader_count": self._readers,
                    },
                )
                return
            while self._writer is not None:
                self._condition.wait()
            self._readers += 1
            self._reader_depths[ident] = self._reader_depths.get(ident, 0) + 1
            _STATE_LOCK_LOGGER.debug(
                "Acquired service state read lock",
                extra={
                    "fc_event": "service_lock_acquired",
                    "lock_mode": "read",
                    "reader_count": self._readers,
                },
            )

    def _release_read(self) -> None:
        ident = threading.get_ident()
        with self._condition:
            depth = self._reader_depths.get(ident, 0)
            if depth <= 1:
                self._reader_depths.pop(ident, None)
            else:
                self._reader_depths[ident] = depth - 1
            self._readers -= 1
            _STATE_LOCK_LOGGER.debug(
                "Released service state read lock",
                extra={
                    "fc_event": "service_lock_released",
                    "lock_mode": "read",
                    "reader_count": self._readers,
                },
            )
            if self._readers == 0:
                self._condition.notify_all()

    def _acquire_write(self) -> None:
        ident = threading.get_ident()
        _STATE_LOCK_LOGGER.debug(
            "Acquiring service state write lock",
            extra={"fc_event": "service_lock_acquire", "lock_mode": "write"},
        )
        with self._condition:
            if self._writer == ident:
                self._write_depth += 1
                _STATE_LOCK_LOGGER.debug(
                    "Acquired service state write lock",
                    extra={
                        "fc_event": "service_lock_acquired",
                        "lock_mode": "write",
                        "write_depth": self._write_depth,
                    },
                )
                return
            own_read_depth = self._reader_depths.get(ident, 0)
            while self._writer is not None or self._readers > own_read_depth:
                self._condition.wait()
            self._writer = ident
            self._write_depth = 1
            _STATE_LOCK_LOGGER.debug(
                "Acquired service state write lock",
                extra={
                    "fc_event": "service_lock_acquired",
                    "lock_mode": "write",
                    "write_depth": self._write_depth,
                },
            )

    def _release_write(self) -> None:
        ident = threading.get_ident()
        with self._condition:
            if self._writer != ident:
                raise RuntimeError(
                    "cannot release state write lock not owned by thread"
                )
            self._write_depth -= 1
            _STATE_LOCK_LOGGER.debug(
                "Released service state write lock",
                extra={
                    "fc_event": "service_lock_released",
                    "lock_mode": "write",
                    "write_depth": self._write_depth,
                },
            )
            if self._write_depth == 0:
                self._writer = None
                self._condition.notify_all()


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

        # Load configuration
        if config_path is None:
            # Try to find config in standard locations
            possible_paths = [
                "config/config.yaml",
                "../config/config.yaml",
                os.path.join(os.path.dirname(__file__), "../../../config/config.yaml"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break

        if config_path and os.path.exists(config_path):
            self.runtime_config = load_runtime_config(config_path)
        else:
            # Use default configuration
            raw_default_config = self._get_default_config()
            resolved_default_config = prepare_runtime_config_mapping(
                raw_default_config,
                project_root=project_root,
            )
            self.runtime_config = config_from_mapping(resolved_default_config)

        self.config = config_to_legacy_dict(self.runtime_config)

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
        self._service_state_lock = _ReadWriteStateLock()

        # Initialize resolver attributes (will be set in index_repository)
        self.global_index_builder: GlobalIndexBuilder | None = None
        self.module_resolver: ModuleResolver | None = None
        self.symbol_resolver: SymbolResolver | None = None

        # Initialize components
        self.loader = RepositoryLoader(self.config)
        self.parser = CodeParser(self.config)
        self.embedder = CodeEmbedder(self.config)
        self.vector_store = VectorStore(self.config)
        self.indexer = CodeIndexer(
            self.config, self.loader, self.parser, self.embedder, self.vector_store
        )
        self.graph_builder = CodeGraphBuilder(self.config)
        self.ir_graph_builder = IRGraphBuilder()

        # Get repo_root from config if available
        config_repo_root: str = str(self.config.get("repo_root") or ".")
        config_repo_root = os.path.abspath(config_repo_root)
        ensure_dir(config_repo_root)
        self.logger.info(f"Configured repo_root: {config_repo_root}")

        self.retriever = HybridRetriever(
            self.config,
            self.vector_store,
            self.embedder,
            self.graph_builder,
            repo_root=config_repo_root,
        )
        self.query_processor = QueryProcessor(self.config)
        self.answer_generator = AnswerGenerator(self.config)
        self.cache_manager = CacheManager(self.config)

        persist_dir = self.vector_store.persist_dir
        storage_cfg: dict[str, Any] = self.config.get("storage", {})
        self.snapshot_store = SnapshotStore(persist_dir, storage_cfg=storage_cfg)
        self.manifest_store = ManifestStore(self.snapshot_store.db_runtime)
        self.index_run_store = IndexRunStore(self.snapshot_store.db_runtime)
        self.unit_artifact_store = UnitArtifactStore(self.snapshot_store.db_runtime)
        self.terminus_publisher = TerminusPublisher(self.config)
        self.projection_transformer = ProjectionTransformer(self.config)
        self.projection_store = ProjectionStore(self.config)
        self.snapshot_symbol_index = SnapshotSymbolIndex()
        self.pg_retrieval_store = PgRetrievalStore(
            self.snapshot_store.db_runtime, self.config
        )
        self.retriever.set_pg_retrieval_store(self.pg_retrieval_store)
        self.doc_ingester = KeyDocIngester(self.config, self.embedder)
        self.graph_runtime = None
        try:
            self.graph_runtime = LadybugGraphRuntime(self.config)
        except ImportError:
            self.logger.info(
                "LadybugGraphRuntime unavailable — graph persistence disabled"
            )

        self._redo_worker: RedoWorker | None = None
        if self.snapshot_store.db_runtime.backend == "postgres":
            storage_cfg: dict[str, Any] = self.config.get("storage", {}) or {}
            poll_interval = int(storage_cfg.get("redo_poll_interval_seconds", 30))
            self._redo_worker = RedoWorker(self, poll_interval_seconds=poll_interval)
            self._redo_worker.start()

        self.semantic_resolver_registry = build_default_semantic_resolver_registry()

        # State (must exist before IndexPipeline wiring)
        self.repo_loaded: bool = False
        self.repo_indexed: bool = False
        self.repo_info: dict[str, Any] = {}

        # --- IndexPipeline ---
        self.pipeline = IndexPipeline(
            config=self.config,
            logger=self.logger,
            loader=self.loader,
            snapshot_store=self.snapshot_store,
            manifest_store=self.manifest_store,
            index_run_store=self.index_run_store,
            unit_artifact_store=self.unit_artifact_store,
            snapshot_symbol_index=self.snapshot_symbol_index,
            vector_store=self.vector_store,
            embedder=self.embedder,
            indexer=self.indexer,
            retriever=self.retriever,
            graph_builder=self.graph_builder,
            ir_graph_builder=self.ir_graph_builder,
            pg_retrieval_store=self.pg_retrieval_store,
            terminus_publisher=self.terminus_publisher,
            doc_ingester=self.doc_ingester,
            semantic_resolver_registry=self.semantic_resolver_registry,
            set_repo_indexed=lambda v: setattr(self, "repo_indexed", v),
            set_repo_loaded=lambda v: setattr(self, "repo_loaded", v),
            set_repo_info=lambda v: setattr(self, "repo_info", v),
        )

        # --- Services ---
        self.projection_service = ProjectionService(
            config=self.config,
            logger=self.logger,
            projection_store=self.projection_store,
            projection_transformer=self.projection_transformer,
            snapshot_store=self.snapshot_store,
            manifest_store=self.manifest_store,
            load_artifacts_by_key=self.pipeline._load_artifacts_by_key,
        )
        self.publishing_service = PublishingService(
            config=self.config,
            logger=self.logger,
            index_run_store=self.index_run_store,
            manifest_store=self.manifest_store,
            snapshot_store=self.snapshot_store,
            terminus_publisher=self.terminus_publisher,
            redo_worker=self._redo_worker,
            build_git_meta=self.pipeline._build_git_meta,
            previous_snapshot_symbol_versions=self.pipeline._previous_snapshot_symbol_versions,
            run_index_pipeline_cb=self.run_index_pipeline,
        )

        self.query_handler = QueryPipeline(
            config=self.config,
            logger=self.logger,
            retriever=self.retriever,
            query_processor=self.query_processor,
            answer_generator=self.answer_generator,
            cache_manager=self.cache_manager,
            manifest_store=self.manifest_store,
            snapshot_store=self.snapshot_store,
            snapshot_symbol_index=self.snapshot_symbol_index,
            is_repo_indexed=lambda: self.repo_indexed,
            load_artifacts_by_key=self.pipeline._load_artifacts_by_key,
            load_snapshot_artifacts=self.pipeline.load_snapshot_artifacts_handle,
            get_session_prefix=self.get_session_prefix,
            semantic_escalation_cb=self._escalate_query_semantics,
        )

        # Multi-repository state
        self.multi_repo_mode: bool = False
        self.loaded_repositories: dict[
            str, dict[str, Any]
        ] = {}  # {repo_name: repo_info}

    def _set_runtime_config(self, config: FastCodeConfig) -> None:
        """Replace canonical runtime config and refresh dict compatibility view."""
        self.runtime_config = config
        self.config = config_to_legacy_dict(config)
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
        lock = getattr(self, "_service_state_lock", None)
        if lock is None:
            lock = _ReadWriteStateLock()
            self._service_state_lock = lock
        return lock

    def _state_read_lock(self) -> Any:
        lock = self._state_lock()
        read_lock = getattr(lock, "read_lock", None)
        if callable(read_lock):
            return read_lock()
        return lock

    def load_repository(
        self, source: str, is_url: bool | None = None, is_zip: bool = False
    ):
        with self._state_lock():
            return self._load_repository_unlocked(source, is_url, is_zip)

    def _load_repository_unlocked(
        self, source: str, is_url: bool | None = None, is_zip: bool = False
    ):
        """
        Load repository from URL, local path, or ZIP file

        Args:
            source: Repository URL, local path, or ZIP file path
            is_url: True if source is a URL, False if local path.
                    If None, FastCode auto-detects source type.
            is_zip: True if source is a ZIP file, False otherwise
        """
        self.logger.info(f"Loading repository: {source}")

        try:
            resolved_is_url = is_url
            if not is_zip and resolved_is_url is None:
                resolved_is_url = self._infer_is_url(source)
                source_type = "URL" if resolved_is_url else "local path"
                self.logger.info(
                    f"Auto-detected source type as {source_type}: {source}"
                )

            if is_zip:
                self.loader.load_from_zip(source)
            elif resolved_is_url:
                self.loader.load_from_url(source)
            else:
                self.loader.load_from_path(source)

            self.repo_loaded = True
            self.repo_info = self.loader.get_repository_info()

            # CRITICAL: Update config with the actual repo path.
            # This ensures path_utils can correctly normalize paths relative to the root.
            if self.loader.repo_path:
                self._set_repo_root(self.loader.repo_path)
                self.logger.info(f"Set repo_root to: {self.loader.repo_path}")

                # Initialize retriever agents if agency mode is enabled
                self.retriever.set_repo_root(self.loader.repo_path)

            self.logger.info(f"Loaded repository: {self.repo_info.get('name')}")
            self.logger.info(
                f"Files: {self.repo_info.get('file_count')}, "
                f"Size: {self.repo_info.get('total_size_mb', 0):.2f} MB"
            )

        except Exception as e:
            self.logger.error(f"Failed to load repository: {e}")
            raise

    def index_repository(self, force: bool = False):
        with self._state_lock():
            return self._index_repository_unlocked(force=force)

    def _legacy_direct_index_enabled(self) -> bool:
        indexing_config = self.config.get("indexing", {})
        if not isinstance(indexing_config, dict):
            return False
        return bool(indexing_config.get("allow_legacy_direct_index", False))

    def _index_repository_unlocked(self, force: bool = False):
        if self._legacy_direct_index_enabled():
            return self._index_repository_legacy_unlocked(force=force)
        force = force or self.eval_config.get("force_reindex", False)
        if not self.repo_loaded:
            raise RuntimeError("No repository loaded. Call load_repository() first.")
        if not self.loader.repo_path:
            raise RuntimeError("No repository path available for indexing.")
        return self.pipeline.run_index_pipeline(
            source=self.loader.repo_path,
            is_url=False,
            force=force,
            publish=True,
            enable_scip=True,
            load_repository_cb=None,
            get_loaded_repositories=lambda: self.loaded_repositories,
            graph_runtime=self.graph_runtime,
        )

    def _index_repository_legacy_unlocked(self, force: bool = False):
        """
        Legacy direct repository indexing path.

        The default public path uses the snapshot pipeline. This method remains
        as an explicit compatibility escape hatch for callers that set
        indexing.allow_legacy_direct_index=true.

        Args:
            force: Force re-indexing even if cache exists
        """
        # Evaluation can request forced re-indexing to respect commit checkouts
        force = force or self.eval_config.get("force_reindex", False)

        if not self.repo_loaded:
            raise RuntimeError("No repository loaded. Call load_repository() first.")

        self.logger.info("Indexing repository")

        repo_name = self.repo_info.get("name", "default")

        # Check cache
        if not force and self._should_use_cache():
            loaded = self._try_load_from_cache()
            if loaded:
                self.repo_indexed = True
                return

        try:
            # Get repository name for indexing
            repo_url = self.repo_info.get("url")

            # Index code elements with repository information
            elements = self.indexer.extract_elements(
                repo_name=repo_name, repo_url=repo_url
            )

            # Initialize vector store if not already done
            if self.vector_store.dimension is None:
                self.vector_store.initialize(self.embedder.embedding_dim)

            # Add embeddings to vector store
            vectors: list[Any] = []
            metadata: list[CodeElementMeta] = []

            for elem in elements:
                embedding = elem.metadata.get("embedding")
                if embedding is not None:
                    vectors.append(embedding)
                    metadata.append(serialize_code_element(elem))

            if vectors:
                vectors_array = as_float32_matrix(vectors, copy_policy="contiguous")
                self.vector_store.add_vectors(vectors_array, metadata)

            # Initialize resolvers for complete graph building
            # This fixes the "0 edges" issue by providing the necessary context for resolution
            try:
                self.logger.info("Initializing resolvers for precise graph building...")

                # Ensure repo_root is set
                repo_root = self.config.get("repo_root")
                if not repo_root and self.loader.repo_path:
                    repo_root = self.loader.repo_path
                    self._set_repo_root(repo_root)

                # 1. Create GlobalIndexBuilder
                self.global_index_builder = GlobalIndexBuilder(self.config)

                # 2. Build global maps
                self.logger.info(
                    f"Building global index maps (Repo Root: {repo_root})..."
                )
                self.global_index_builder.build_maps(elements, repo_root or "")
                self.logger.info(
                    f"  - Mapped {len(self.global_index_builder.file_map)} files"
                )
                self.logger.info(
                    f"  - Mapped {len(self.global_index_builder.module_map)} modules"
                )

                # 3. Create ModuleResolver
                self.module_resolver = ModuleResolver(self.global_index_builder)

                # 4. Create SymbolResolver
                self.symbol_resolver = SymbolResolver(
                    self.global_index_builder, self.module_resolver
                )

                self.logger.info("Resolvers initialized successfully")

            except Exception as e:
                self.logger.warning(f"Resolver initialization failed: {e}")
                self.logger.warning("Using fallback graph building (less accurate)")
                import traceback

                self.logger.error(traceback.format_exc())
                self.module_resolver = None
                self.symbol_resolver = None

            # Build code graphs with resolvers
            # This will now use the initialized resolvers to build precise graphs
            self.graph_builder.build_graphs(
                elements, self.module_resolver, self.symbol_resolver
            )

            # Index for BM25
            self.retriever.index_for_bm25(elements)

            # Build separate BM25 index for repository overviews
            self.retriever.build_repo_overview_bm25()

            # Save artifacts only when persistence is enabled
            if self._should_persist_indexes():
                # Save to cache with repository-specific name
                self._save_to_cache(cache_name=repo_name)

                # Save BM25 and graph data
                self.retriever.save_bm25(repo_name)
                self.graph_builder.save(repo_name)
                self._save_file_manifest(
                    repo_name,
                    self._build_file_manifest(elements, self.loader.repo_path or "."),
                )
            else:
                self.logger.info(
                    "Skipping on-disk persistence (ephemeral/evaluation mode)"
                )

            self.repo_indexed = True
            self.logger.info(f"Repository indexing complete for {repo_name}")

            # Log statistics
            self._log_statistics()

        except Exception as e:
            self.logger.error(f"Failed to index repository: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise

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

    def run_index_pipeline(
        self,
        source: str,
        is_url: bool | None = None,
        ref: str | None = None,
        commit: str | None = None,
        force: bool = False,
        publish: bool = True,
        scip_artifact_path: str | None = None,
        enable_scip: bool = True,
    ) -> dict[str, Any]:
        """Run snapshot-oriented indexing pipeline (delegates to IndexPipeline)."""
        with self._state_lock():
            return self.pipeline.run_index_pipeline(
                source=source,
                is_url=is_url,
                ref=ref,
                commit=commit,
                force=force,
                publish=publish,
                scip_artifact_path=scip_artifact_path,
                enable_scip=enable_scip,
                load_repository_cb=self.load_repository,
                get_loaded_repositories=lambda: self.loaded_repositories,
                graph_runtime=self.graph_runtime,
            )

    def _apply_semantic_resolvers(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        legacy_graph_builder: CodeGraphBuilder | None,
        target_paths: set[str],
        warnings: list[str],
        budget: str = "changed_files",
    ) -> IRSnapshot:
        return self.pipeline._apply_semantic_resolvers(
            snapshot=snapshot,
            elements=elements,
            legacy_graph_builder=legacy_graph_builder,
            target_paths=target_paths,
            warnings=warnings,
            budget=budget,
        )

    def get_index_run(self, run_id: str) -> dict[str, Any] | None:
        return self.publishing_service.get_index_run(run_id)

    def publish_index_run(
        self, run_id: str, ref_name: str | None = None
    ) -> dict[str, Any]:
        return self.publishing_service.publish_index_run(run_id, ref_name=ref_name)

    def retry_pending_publishes(self, limit: int = 10) -> dict[str, Any]:
        return self.publishing_service.retry_pending_publishes(limit)

    def retry_index_run_recovery(
        self, run_id: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.publishing_service.retry_index_run_recovery(run_id, payload=payload)

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
            (getattr(self, "config", {}) or {})
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
        snapshot_store = getattr(self, "snapshot_store", None)
        load_snapshot = getattr(snapshot_store, "load_snapshot", None)
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
        store = getattr(self, "projection_store", None)
        if store is None or not getattr(store, "enabled", False):
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

    def process_semantic_repair_frontier(
        self, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = payload or {}
        scope_kind = str(payload.get("scope_kind") or "path")
        scope_roots = list(payload.get("scope_roots") or [])
        snapshot_id = str(payload.get("snapshot_id") or "")
        if not snapshot_id:
            raise RuntimeError("semantic_repair_frontier payload missing snapshot_id")
        result = self.pipeline.run_semantic_repair_frontier(
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

    def process_redo_tasks(self, limit: int = 10) -> dict[str, Any]:
        return self.publishing_service.process_redo_tasks(limit)

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

    def list_repo_refs(self, repo_name: str) -> list[dict[str, Any]]:
        return [
            self._snapshot_ref_payload(record)
            for record in self.snapshot_store.list_repo_ref_records(repo_name)
        ]

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
        load_record = getattr(self.snapshot_store, "load_snapshot_symbol_record", None)
        if callable(load_record):
            record = load_record(snapshot_id, resolved)
            if isinstance(record, Mapping):
                return {str(key): value for key, value in record.items()}

        snapshot = self.snapshot_store.load_snapshot(snapshot_id)
        if not snapshot:
            return None
        for symbol in snapshot.symbols:
            if symbol.symbol_id == resolved:
                return symbol.to_dict()
        return None

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

    def get_graph_callees(
        self, snapshot_id: str, symbol_id: str, max_hops: int = 1
    ) -> list[dict[str, Any]]:
        max_hops = max(1, min(max_hops, 20))
        ir_graphs = self.snapshot_store.load_ir_graphs(snapshot_id)
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
        ir_graphs = self.snapshot_store.load_ir_graphs(snapshot_id)
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
        ir_graphs = self.snapshot_store.load_ir_graphs(snapshot_id)
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

    def get_branch_manifest(
        self, repo_name: str, ref_name: str
    ) -> dict[str, Any] | None:
        record = self.manifest_store.get_branch_manifest_record(repo_name, ref_name)
        return self._manifest_payload(record) if record is not None else None

    def get_snapshot_manifest(self, snapshot_id: str) -> dict[str, Any] | None:
        record = self.manifest_store.get_snapshot_manifest_record(snapshot_id)
        return self._manifest_payload(record) if record is not None else None

    def get_scip_artifact_ref(self, snapshot_id: str) -> dict[str, Any] | None:
        record = self.snapshot_store.get_scip_artifact_ref_record(snapshot_id)
        return self._scip_artifact_payload(record) if record is not None else None

    def list_scip_artifact_refs(self, snapshot_id: str) -> list[dict[str, Any]]:
        return [
            self._scip_artifact_payload(record)
            for record in self.snapshot_store.list_scip_artifact_ref_records(
                snapshot_id
            )
        ]

    def resolve_snapshot_symbol(
        self,
        snapshot_id: str,
        *,
        symbol_id: str | None = None,
        name: str | None = None,
        path: str | None = None,
    ) -> str | None:
        self._ensure_snapshot_symbol_index(snapshot_id)
        return self.snapshot_symbol_index.resolve_symbol(
            snapshot_id,
            symbol_id=symbol_id,
            name=name,
            path=path,
        )

    def _register_snapshot_symbols_from_payload(self, snapshot_id: str) -> bool:
        load_payload = getattr(
            self.snapshot_store,
            "load_snapshot_symbol_index_payload",
            None,
        )
        register_payload = getattr(
            self.snapshot_symbol_index,
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
        return self.snapshot_symbol_index.has_snapshot(snapshot_id)

    def _ensure_snapshot_symbol_index(self, snapshot_id: str) -> None:
        if self.snapshot_symbol_index.has_snapshot(snapshot_id):
            return
        if self._register_snapshot_symbols_from_payload(snapshot_id):
            return
        snap = self.snapshot_store.load_snapshot(snapshot_id)
        if snap:
            self.snapshot_symbol_index.register_snapshot(snap)

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
        with self._state_lock():
            return self.projection_service.build_projection(
                scope_kind=scope_kind,
                snapshot_id=snapshot_id,
                repo_name=repo_name,
                ref_name=ref_name,
                query=query,
                target_id=target_id,
                filters=filters,
                force=force,
            )

    def get_projection_layer(self, projection_id: str, layer: str) -> dict[str, Any]:
        return self.projection_service.get_projection_layer(projection_id, layer)

    def get_projection_chunk(self, projection_id: str, chunk_id: str) -> dict[str, Any]:
        return self.projection_service.get_projection_chunk(projection_id, chunk_id)

    def get_session_prefix(self, snapshot_id: str) -> dict[str, Any]:
        """Return L0+L1 projection data for system prompt injection."""
        return self.projection_service.get_session_prefix(snapshot_id)

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
        with self._state_read_lock():
            return self.query_handler.query_snapshot(
                question=question,
                repo_name=repo_name,
                ref_name=ref_name,
                snapshot_id=snapshot_id,
                filters=filters,
                session_id=session_id,
                enable_multi_turn=enable_multi_turn,
            )

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
        with self._state_read_lock():
            return self.query_handler.query(
                question=question,
                filters=filters,
                repo_filter=repo_filter,
                session_id=session_id,
                enable_multi_turn=enable_multi_turn,
                use_agency_mode=use_agency_mode,
                prompt_builder=prompt_builder,
            )

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
        snapshot = self.snapshot_store.load_snapshot(snapshot_id)
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

        active_retriever = retriever or self.retriever
        active_graph_builder = graph_builder or self.graph_builder
        elements = list(active_graph_builder.element_by_id.values())
        warnings: list[str] = []
        upgraded_snapshot = self._apply_semantic_resolvers(
            snapshot=snapshot,
            elements=elements,
            legacy_graph_builder=active_graph_builder,
            target_paths=target_paths,
            warnings=warnings,
            budget=budget,
        )
        self.snapshot_symbol_index.register_snapshot(upgraded_snapshot)
        upgraded_ir_graphs = self.ir_graph_builder.build_graphs(upgraded_snapshot)
        active_retriever.set_ir_graphs(upgraded_ir_graphs, snapshot_id=snapshot_id)

        semantic_runs = list(
            (upgraded_snapshot.metadata or {}).get("semantic_resolver_runs", [])
        )
        status = "degraded" if warnings else "applied"
        logger = getattr(self, "logger", logging.getLogger(__name__))
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
            with self._state_read_lock():
                snapshot_record = self.snapshot_store.get_snapshot_record(snapshot_id)
                if not snapshot_record:
                    raise RuntimeError(f"snapshot not found: {snapshot_id}")
                artifact_key = snapshot_record.artifact_key
                loaded_artifacts = self.pipeline.load_snapshot_artifacts_handle(
                    artifact_key,
                    snapshot_id=snapshot_id,
                )
                if loaded_artifacts is None:
                    raise RuntimeError(
                        f"failed to load artifacts for snapshot: {snapshot_id}"
                    )
                self.query_handler._ensure_snapshot_symbol_index(snapshot_id)

                merged_filters = dict(filters or {})
                merged_filters["snapshot_id"] = snapshot_id
                merged_filters["artifact_key"] = getattr(
                    loaded_artifacts,
                    "artifact_key",
                    artifact_key,
                )
                retriever = loaded_artifacts.retriever

            yield from self.query_handler.query_stream(
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

        with self._state_read_lock():
            yield from self.query_handler.query_stream(
                question=question,
                filters=filters,
                repo_filter=repo_filter,
                session_id=session_id,
                enable_multi_turn=enable_multi_turn,
                use_agency_mode=use_agency_mode,
                prompt_builder=prompt_builder,
            )

    def _extract_sources_from_elements(
        self, elements: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract source information from retrieved elements"""
        return _snapshot.extract_sources_from_elements(elements)

    def get_repository_summary(self) -> str:
        """Get summary of the loaded repository"""
        if not self.repo_info:
            return "No repository loaded"

        summary_parts = [
            f"Repository: {self.repo_info.get('name', 'Unknown')}",
            f"Files: {self.repo_info.get('file_count', 0)}",
            f"Size: {self.repo_info.get('total_size_mb', 0):.2f} MB",
        ]

        if self.repo_indexed:
            summary_parts.append(f"Indexed elements: {self.vector_store.get_count()}")

        return "\n".join(summary_parts)

    def _try_load_from_cache(self) -> bool:
        """Try to load indexed data from cache for single repository"""
        if not self._should_use_cache():
            self.logger.info("Cache loading disabled (ephemeral/evaluation mode)")
            return False

        try:
            cache_name = self._get_cache_name()

            # Try to load vector store
            if self.vector_store.load(cache_name):
                self.logger.info(f"Loaded vector store from cache for {cache_name}")

                # Load BM25 index
                bm25_loaded = self.retriever.load_bm25(cache_name)
                if not bm25_loaded:
                    self.logger.warning(
                        "Failed to load BM25 index, will need to rebuild"
                    )

                # Build separate repo overview BM25 index
                self.retriever.build_repo_overview_bm25()

                # Load graph data
                graph_loaded = self.graph_builder.load(cache_name)
                if not graph_loaded:
                    self.logger.warning(
                        "Failed to load graph data, will need to rebuild"
                    )

                # If BM25 or graph failed to load, reconstruct from metadata
                if not bm25_loaded or not graph_loaded:
                    self.logger.info(
                        "Reconstructing missing components from metadata..."
                    )
                    elements = self._reconstruct_elements_from_metadata()

                    if elements:
                        if not bm25_loaded:
                            self.retriever.index_for_bm25(elements)
                            self.logger.info(
                                f"Rebuilt BM25 index with {len(elements)} elements"
                            )

                        if not graph_loaded:
                            # Note: Rebuilding graph from metadata is a fallback.
                            # Precise linking might be limited if repo_root context is lost.
                            self.graph_builder.build_graphs(elements)
                            self.logger.info("Rebuilt code graph (fallback mode)")
                    else:
                        self.logger.warning("No elements reconstructed from metadata")

                self.logger.info("Cache loaded successfully")
                self._log_statistics()
                return True

            return False

        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {e}")
            return False

    def _save_to_cache(self, cache_name: str | None = None):
        """Save indexed data to cache"""
        if not self._should_persist_indexes():
            self.logger.info("Cache save disabled (ephemeral/evaluation mode)")
            return

        try:
            if cache_name is None:
                cache_name = self._get_cache_name()
            self.vector_store.save(cache_name)
            self.logger.info(f"Saved index to cache: {cache_name}")
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {e}")

    def _get_cache_name(self) -> str:
        """Get cache name for current repository"""
        return self.repo_info.get("name", "default")

    def _get_repo_hash(self) -> str:
        """Get hash of repository for cache key"""
        return self.repo_info.get("commit", self.repo_info.get("name", "default"))

    def _reconstruct_elements_from_metadata(self) -> list[CodeElement]:
        """
        Reconstruct CodeElement objects from vector store metadata
        Excludes repository_overview elements (they're in separate storage)

        Returns:
            List of CodeElement objects
        """
        elements: list[CodeElement] = []
        for meta in self.vector_store.metadata:
            try:
                # Skip repository_overview elements
                if meta.get("type") == "repository_overview":
                    continue

                # Reconstruct CodeElement from metadata dictionary
                element = CodeElement(
                    id=meta.get("id", ""),
                    type=meta.get("type", ""),
                    name=meta.get("name", ""),
                    file_path=meta.get("file_path", ""),
                    relative_path=meta.get("relative_path", ""),
                    language=meta.get("language", ""),
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    code=meta.get("code", ""),
                    signature=meta.get("signature"),
                    docstring=meta.get("docstring"),
                    summary=meta.get("summary"),
                    metadata=meta.get("metadata", {}),
                    repo_name=meta.get("repo_name"),
                    repo_url=meta.get("repo_url"),
                )
                elements.append(element)
            except Exception as e:
                self.logger.warning(f"Failed to reconstruct element: {e}")
                continue

        self.logger.info(
            f"Reconstructed {len(elements)} elements from metadata (excluding repository_overview)"
        )
        return elements

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
                raise RuntimeError("Graph runtime not available")
            synced = self.graph_runtime.sync_docs(chunks=chunks, mentions=mentions)
        except Exception as e:
            warnings.append(f"ladybug_doc_sync_failed: {e}")
            return
        if not synced:
            warnings.append("ladybug_doc_sync_failed")

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration.

        Categories:
          [ESSENTIAL] — deployment-specific; users change per environment
          [TUNABLE]  — power-user knobs with sensible defaults
          [INTERNAL] — algorithm internals; rarely need adjustment

        Total: 54 parameters across 13 sections.
        Config file (config/config.yaml) may contain additional sections
        (query, graph, agent, docs_integration) with their own defaults
        defined inline at consumption sites.
        """
        return {
            # ── storage ──────────────────────────────────────────────
            "storage": {
                "backend": "sqlite",  # [ESSENTIAL] "sqlite" or "postgres"
                "postgres_dsn": "",  # [ESSENTIAL] PostgreSQL connection string
                "pool_min": 1,  # [INTERNAL] connection pool minimum
                "pool_max": 8,  # [TUNABLE]  connection pool maximum
            },
            # ── repository ───────────────────────────────────────────
            "repository": {
                "clone_depth": 1,  # [INTERNAL] git shallow clone depth
                "max_file_size_mb": 5,  # [TUNABLE]  skip files larger than this
                "backup_directory": "./repo_backup",  # [TUNABLE] backup location
                "exclude_site_packages": False,  # [TUNABLE] ignore vendored site-packages
                "ignore_patterns": [
                    "*.pyc",
                    "__pycache__",
                    "node_modules",
                    ".git",
                ],  # [TUNABLE]
                "supported_extensions": [  # [TUNABLE]  file extensions to index
                    ".py",
                    ".js",
                    ".jsx",
                    ".ts",
                    ".tsx",
                    ".java",
                    ".go",
                    ".rs",
                    ".cs",
                    ".c",
                    ".h",
                    ".cc",
                    ".cpp",
                    ".cxx",
                    ".hh",
                    ".hpp",
                    ".hxx",
                    ".zig",
                    ".f",
                    ".for",
                    ".f77",
                    ".f90",
                    ".f95",
                    ".f03",
                    ".f08",
                    ".jl",
                ],
            },
            # ── parser ───────────────────────────────────────────────
            "parser": {
                "extract_docstrings": True,  # [INTERNAL]
                "extract_comments": True,  # [INTERNAL]
                "extract_imports": True,  # [INTERNAL]
            },
            # ── embedding ────────────────────────────────────────────
            "embedding": {
                "provider": "ollama",  # [ESSENTIAL] "ollama" or "sentence_transformers"
                "model": "bge-large-en-v1.5",  # [ESSENTIAL] embedding model name
                "ollama_url": "http://127.0.0.1:11434/api/embeddings",  # [ESSENTIAL]
                "device": "cpu",  # [TUNABLE]  "auto", "cuda", "mps", "cpu"
                "batch_size": 32,  # [INTERNAL]
            },
            # ── indexing ─────────────────────────────────────────────
            "indexing": {
                "levels": ["file", "class", "function", "documentation"],  # [INTERNAL]
                "allow_legacy_direct_index": False,  # [INTERNAL]
            },
            # ── vector_store ─────────────────────────────────────────
            "vector_store": {
                "persist_directory": "./data/vector_store",  # [TUNABLE]
                "distance_metric": "cosine",  # [INTERNAL] similarity metric
                "shard_storage": "compressed",  # [TUNABLE] "compressed" or "npy"
            },
            # ── retrieval ────────────────────────────────────────────
            "retrieval": {
                "semantic_weight": 0.6,  # [TUNABLE]  hybrid search semantic weight
                "keyword_weight": 0.3,  # [TUNABLE]  hybrid search keyword weight
                "graph_weight": 0.1,  # [TUNABLE]  hybrid search graph weight
                "max_results": 5,  # [TUNABLE]  max results per query
                "backend": "pg_hybrid",  # [ESSENTIAL] "pg_hybrid" or legacy
                "graph_backend": "ir",  # [ESSENTIAL] "ir" or "legacy"
                "allow_legacy_graph_fallback": True,  # [INTERNAL]
            },
            # ── generation ───────────────────────────────────────────
            "generation": {
                "provider": "openai",  # [ESSENTIAL] "openai", "anthropic", or "local"
                "model": "gpt-4-turbo-preview",  # [ESSENTIAL] LLM model name
                "temperature": 0.1,  # [TUNABLE]
                "max_tokens": 2000,  # [TUNABLE]
            },
            # ── evaluation ───────────────────────────────────────────
            "evaluation": {
                "enabled": False,  # [TUNABLE]  enable benchmark/eval mode
                "in_memory_index": False,  # [INTERNAL] keep index in RAM only
                "disable_cache": False,  # [INTERNAL] skip query/embedding cache
                "disable_persistence": False,  # [INTERNAL] skip writing artifacts
                "force_reindex": False,  # [INTERNAL] always rebuild index
            },
            # ── cache ────────────────────────────────────────────────
            "cache": {
                "enabled": True,  # [TUNABLE]
                "backend": "disk",  # [ESSENTIAL] "disk" or "redis"
                "cache_directory": "./data/cache",  # [TUNABLE]
                "cache_queries": False,  # [INTERNAL]
                "redis_host": "localhost",  # [ESSENTIAL] Redis cache host
                "redis_port": 6379,  # [ESSENTIAL] Redis cache port
            },
            # ── logging ──────────────────────────────────────────────
            "logging": {
                "level": "INFO",  # [TUNABLE]  DEBUG, INFO, WARNING, ERROR
                "console": True,  # [INTERNAL]
            },
            # ── terminus ─────────────────────────────────────────────
            "terminus": {
                "endpoint": "",  # [ESSENTIAL] TerminusDB publish endpoint
                "api_key": "",  # [ESSENTIAL] TerminusDB API key
                "timeout_seconds": 15,  # [INTERNAL]
            },
            # ── projection ───────────────────────────────────────────
            "projection": {
                "postgres_dsn": "",  # [ESSENTIAL] projection store DSN
                "enable_leiden": True,  # [TUNABLE]  enable Leiden clustering
                "llm_enabled": True,  # [TUNABLE]  enable LLM label generation
                "llm_timeout_seconds": 8,  # [INTERNAL]
                "llm_max_tokens": 180,  # [INTERNAL]
                "llm_temperature": 0.2,  # [INTERNAL]
                "max_entity_hops": 2,  # [INTERNAL]
                "max_query_hops": 2,  # [INTERNAL]
                "max_chunk_count": 64,  # [INTERNAL],
            },
        }

    def load_multiple_repositories(self, sources: list[dict[str, Any]]):
        with self._state_lock():
            return self._load_multiple_repositories_unlocked(sources)

    def _load_multiple_repositories_unlocked(self, sources: list[dict[str, Any]]):
        if not self._legacy_direct_index_enabled():
            return self._load_multiple_repositories_pipeline_unlocked(sources)
        return self._load_multiple_repositories_legacy_unlocked(sources)

    def _load_multiple_repositories_pipeline_unlocked(
        self, sources: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.logger.info(f"Loading {len(sources)} repositories")
        self.multi_repo_mode = True
        successfully_indexed: list[str] = []
        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        for i, source_info in enumerate(sources):
            source = str(source_info.get("source") or "")
            is_url: bool | None = source_info.get("is_url")
            is_zip = bool(source_info.get("is_zip", False))
            try:
                self.logger.info(
                    f"[{i + 1}/{len(sources)}] Loading repository: {source}"
                )
                resolved_is_url = (
                    self._infer_is_url(source)
                    if not is_zip and is_url is None
                    else is_url
                )
                pipeline_source = source
                pipeline_is_url = resolved_is_url
                load_repository_cb: Callable[..., None] | None
                if is_zip:
                    self._load_repository_unlocked(source, is_url=False, is_zip=True)
                    if not self.loader.repo_path:
                        raise RuntimeError("zip load did not produce a repository path")
                    pipeline_source = self.loader.repo_path
                    pipeline_is_url = False
                    load_repository_cb = None
                else:

                    def _load_repository_cb(
                        loaded_source: str, is_url: bool | None = None
                    ) -> None:
                        self._load_repository_unlocked(
                            loaded_source, is_url=is_url, is_zip=False
                        )

                    load_repository_cb = _load_repository_cb

                result = self.pipeline.run_index_pipeline(
                    source=pipeline_source,
                    is_url=pipeline_is_url,
                    force=bool(source_info.get("force", False)),
                    publish=True,
                    enable_scip=True,
                    load_repository_cb=load_repository_cb,
                    get_loaded_repositories=lambda: self.loaded_repositories,
                    graph_runtime=self.graph_runtime,
                )
                results.append(result)
                repo_name = str(result.get("repo_name") or "")
                if repo_name:
                    successfully_indexed.append(repo_name)
            except Exception as e:
                self.logger.error(f"Failed to index repository {source}: {e}")
                errors.append({"source": source, "error": str(e)})
                continue

        self.repo_indexed = len(successfully_indexed) > 0
        self.repo_loaded = len(successfully_indexed) > 0
        return {
            "status": "succeeded" if successfully_indexed else "failed",
            "repositories": successfully_indexed,
            "results": results,
            "errors": errors,
        }

    def _load_multiple_repositories_legacy_unlocked(
        self, sources: list[dict[str, Any]]
    ):
        """
        Legacy direct multi-repository indexing path.

        The default multi-repository path uses the snapshot pipeline. This
        direct vector/BM25/graph staging path is retained only for callers that
        set indexing.allow_legacy_direct_index=true.

        Args:
            sources: List of dictionaries with 'source', 'is_url', and optionally 'is_zip' keys
                    Example: [{'source': 'https://github.com/user/repo1', 'is_url': True},
                             {'source': '/path/to/repo2', 'is_url': False},
                             {'source': '/path/to/repo3.zip', 'is_url': False, 'is_zip': True}]
        """
        self.logger.info(f"Loading {len(sources)} repositories")
        self.multi_repo_mode = True

        successfully_indexed = []

        for i, source_info in enumerate(sources):
            source: str = str(source_info.get("source") or "")
            is_url: bool | None = source_info.get("is_url")
            is_zip: bool = bool(source_info.get("is_zip", False))

            try:
                self.logger.info(
                    f"[{i + 1}/{len(sources)}] Loading repository: {source}"
                )

                resolved_is_url = is_url
                if not is_zip and resolved_is_url is None:
                    resolved_is_url = self._infer_is_url(source)
                    source_type = "URL" if resolved_is_url else "local path"
                    self.logger.info(
                        f"[{i + 1}/{len(sources)}] Auto-detected source type as {source_type}"
                    )

                # Load repository
                if is_zip:
                    self.loader.load_from_zip(source)
                elif resolved_is_url:
                    self.loader.load_from_url(source)
                else:
                    self.loader.load_from_path(source)

                repo_info = self.loader.get_repository_info()
                repo_name: str = str(repo_info.get("name") or "")
                repo_url: str = str(repo_info.get("url") or source)

                # Update config with repo_root for each repo (Critical for graph building)
                if self.loader.repo_path:
                    self._set_repo_root(self.loader.repo_path)

                # Store repository info
                self.loaded_repositories[repo_name] = repo_info

                self.logger.info(f"Indexing repository: {repo_name}")

                # Create a fresh vector store for this repository
                temp_vector_store = VectorStore(self.config)
                temp_vector_store.initialize(self.embedder.embedding_dim)

                # Create a temporary indexer with the temp vector store for this repo
                temp_indexer = CodeIndexer(
                    self.config,
                    self.loader,
                    self.parser,
                    self.embedder,
                    temp_vector_store,
                )

                # Index with repository information
                elements = temp_indexer.extract_elements(
                    repo_name=repo_name, repo_url=repo_url
                )

                # Add to temporary vector store
                vectors: list[list[float]] = []
                metadata: list[CodeElementMeta] = []

                for elem in elements:
                    embedding = elem.metadata.get("embedding")
                    if embedding is not None:
                        vectors.append(embedding)
                        metadata.append(serialize_code_element(elem))

                if vectors:
                    vectors_array = as_float32_matrix(vectors, copy_policy="contiguous")
                    temp_vector_store.add_vectors(vectors_array, metadata)

                    # Save this repository's vector index separately
                    temp_vector_store.save(repo_name)

                    # Build and save BM25 index for this repository
                    temp_retriever = HybridRetriever(
                        self.config,
                        temp_vector_store,
                        self.embedder,
                        self.graph_builder,
                        repo_root=self.loader.repo_path,
                    )
                    temp_retriever.index_for_bm25(elements)
                    temp_retriever.save_bm25(repo_name)
                    self.logger.info(f"Saved BM25 index for {repo_name}")

                    # Build separate BM25 index for repository overviews
                    temp_retriever.build_repo_overview_bm25()
                    self.logger.info("Built repo overview BM25 index")

                    # Build and save graph for this repository (Using temporary graph builder)
                    # We need a fresh graph builder to avoid mixing graphs between repos during this loop
                    # unless we want to support cross-repo graphs immediately
                    temp_graph_builder = CodeGraphBuilder(self.config)

                    # Initialize resolvers for precise graph building
                    repo_root = self.loader.repo_path
                    temp_module_resolver = None
                    temp_symbol_resolver = None

                    try:
                        self.logger.info(f"Initializing resolvers for {repo_name}...")
                        temp_global_index = GlobalIndexBuilder(self.config)
                        temp_global_index.build_maps(elements, repo_root or ".")
                        temp_module_resolver = ModuleResolver(temp_global_index)
                        temp_symbol_resolver = SymbolResolver(
                            temp_global_index, temp_module_resolver
                        )
                        self.logger.info(f"Resolvers initialized for {repo_name}")
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to initialize resolvers for {repo_name}: {e}"
                        )
                        temp_module_resolver = None
                        temp_symbol_resolver = None

                    temp_graph_builder.build_graphs(
                        elements, temp_module_resolver, temp_symbol_resolver
                    )
                    temp_graph_builder.save(repo_name)
                    self.logger.info(f"Saved graph data for {repo_name}")

                    successfully_indexed.append(repo_name)

                    self.logger.info(
                        f"Successfully indexed and saved {repo_name}: {len(elements)} elements"
                    )
                else:
                    self.logger.warning(f"No vectors generated for {repo_name}")

            except Exception as e:
                self.logger.error(f"Failed to load repository {source}: {e}")
                import traceback

                self.logger.error(traceback.format_exc())
                # Continue with next repository
                continue

        if successfully_indexed:
            self.logger.info(
                f"Successfully indexed {len(successfully_indexed)} repositories:"
            )
            for repo_name in successfully_indexed:
                self.logger.info(f"  - {repo_name}")

            # Merge all indexed repositories into the main vector store for statistics
            self.logger.info(
                "Merging repositories into main vector store for statistics..."
            )
            if self.vector_store.dimension is None:
                self.vector_store.initialize(self.embedder.embedding_dim)

            for repo_name in successfully_indexed:
                if self.vector_store.merge_from_index(repo_name):
                    self.logger.info(f"Merged {repo_name} into main store")
                else:
                    self.logger.warning(f"Failed to merge {repo_name}")
        else:
            self.logger.error("No repositories were successfully indexed")

        self.repo_indexed = len(successfully_indexed) > 0
        self.repo_loaded = len(successfully_indexed) > 0

        self.logger.info("Indexing complete. Each repository saved separately.")

    def list_repositories(self) -> list[dict[str, Any]]:
        """
        List all indexed repositories

        Returns:
            List of repository information dictionaries
        """
        repo_names = self.vector_store.get_repository_names()
        repo_counts = self.vector_store.get_count_by_repository()

        repositories: list[dict[str, Any]] = []
        for repo_name in repo_names:
            repo_info = self.loaded_repositories.get(repo_name, {})
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

    def get_repository_stats(self) -> dict[str, Any]:
        """
        Get statistics about all indexed repositories

        Returns:
            Dictionary with repository statistics
        """
        repo_counts = self.vector_store.get_count_by_repository()
        repo_names = self.vector_store.get_repository_names()

        stats: dict[str, Any] = {
            "total_repositories": len(repo_names),
            "total_elements": self.vector_store.get_count(),
            "repositories": [],
        }

        for repo_name in repo_names:
            repo_info = self.loaded_repositories.get(repo_name, {})
            stats["repositories"].append(
                {
                    "name": repo_name,
                    "elements": repo_counts.get(repo_name, 0),
                    "files": repo_info.get("file_count", 0),
                    "size_mb": repo_info.get("total_size_mb", 0),
                }
            )

        return stats

    def _load_multi_repo_cache(self, repo_names: list[str] | None = None) -> bool:
        with self._state_lock():
            return self._load_multi_repo_cache_unlocked(repo_names=repo_names)

    def _load_multi_repo_cache_unlocked(
        self, repo_names: list[str] | None = None
    ) -> bool:
        """
        Load multi-repository index from cache by merging individual repository indices

        Args:
            repo_names: Optional list of specific repository names to load.
                       If None, loads all available repositories.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Discover available repository indexes
            available_repos: list[str] = []
            scan_available_indexes = getattr(
                self.vector_store, "scan_available_indexes", None
            )
            if callable(scan_available_indexes):
                for repo in cast(list[dict[str, Any]], scan_available_indexes(False)):
                    repo_name = str(repo.get("name") or repo.get("repo_name") or "")
                    if repo_name:
                        available_repos.append(repo_name)
            else:
                persist_dir = self.vector_store.persist_dir
                if os.path.exists(persist_dir):
                    for file in os.listdir(persist_dir):
                        if file.endswith(".faiss"):
                            repo_name = file.replace(".faiss", "")
                            metadata_file = os.path.join(
                                persist_dir, f"{repo_name}_metadata.pkl"
                            )
                            if os.path.exists(metadata_file):
                                available_repos.append(repo_name)

            if not available_repos:
                self.logger.error("No repository indexes found")
                return False

            # Filter repositories if specific ones are requested
            if repo_names:
                repos_to_load = [r for r in available_repos if r in repo_names]
                if not repos_to_load:
                    self.logger.error(
                        f"None of the requested repositories found: {repo_names}"
                    )
                    return False
            else:
                repos_to_load = available_repos

            self.logger.info(
                f"Found {len(repos_to_load)} repository indexes: {', '.join(repos_to_load)}"
            )

            # Always reinitialize for clean merge
            self.vector_store.initialize(self.embedder.embedding_dim)

            # Load each repository index and merge them
            for repo_name in repos_to_load:
                self.logger.info(f"Loading index for {repo_name}...")
                try:
                    # Merge this repository's index into the main vector store
                    if self.vector_store.merge_from_index(repo_name):
                        self.logger.info(f"Successfully merged {repo_name}")
                    else:
                        self.logger.warning(f"Failed to merge index for {repo_name}")

                except Exception as e:
                    self.logger.error(f"Error loading {repo_name}: {e}")
                    continue

            # Check if we successfully loaded any repositories
            if self.vector_store.get_count() == 0:
                self.logger.error("Failed to load any repository indexes")
                return False

            # Register loaded repositories
            # We know which repos were successfully loaded from repos_to_load
            for repo_name in repos_to_load:
                if repo_name not in self.loaded_repositories:
                    self.loaded_repositories[repo_name] = {
                        "name": repo_name,
                        "file_count": 0,  # Will be updated if needed
                        "total_size_mb": 0,
                    }

            # Try to load BM25 and graph data from saved files
            # For multi-repo, we merge BM25 data from all loaded repositories
            self.logger.info("Loading BM25 and graph data...")

            all_bm25_elements: list[CodeElement] = []
            all_bm25_corpus: list[list[str]] = []
            graphs_loaded = False

            load_bm25_payload = getattr(self.retriever, "load_bm25_payload", None)

            for repo_name in repos_to_load:
                # Try loading BM25 for each repo
                data: dict[str, Any] | None = None
                if callable(load_bm25_payload):
                    try:
                        payload = load_bm25_payload(repo_name)
                        if isinstance(payload, dict):
                            data = payload
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load BM25 data for {repo_name}: {e}"
                        )
                else:
                    bm25_path = os.path.join(
                        self.retriever.persist_dir, f"{repo_name}_bm25.pkl"
                    )
                    if os.path.exists(bm25_path):
                        try:
                            with open(bm25_path, "rb") as f:
                                payload = pickle.load(f)
                            if isinstance(payload, dict):
                                data = payload
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to load BM25 data for {repo_name}: {e}"
                            )

                if data is not None:
                    all_bm25_corpus.extend(
                        cast(list[list[str]], data.get("bm25_corpus", []))
                    )
                    for elem_payload in cast(
                        list[dict[str, Any]], data.get("bm25_elements", [])
                    ):
                        all_bm25_elements.append(deserialize_code_element(elem_payload))
                    self.logger.info(f"Loaded BM25 data for {repo_name}")

                # Load graph data (merge into main graph)
                if not graphs_loaded:
                    # Load the first repository's graph as base
                    if self.graph_builder.load(repo_name):
                        graphs_loaded = True
                        self.logger.info(f"Loaded graph data from {repo_name} as base")
                # Merge additional repository graphs
                elif self.graph_builder.merge_from_file(repo_name):
                    self.logger.info(f"Merged graph data from {repo_name}")
                else:
                    self.logger.warning(f"Failed to merge graph data from {repo_name}")
                    # TODO: Merge additional repository graphs if needed
            # Rebuild FULL BM25 index with merged data (for repository selection)
            if all_bm25_elements and all_bm25_corpus:
                self.retriever.full_bm25_elements = all_bm25_elements
                self.retriever.full_bm25_corpus = all_bm25_corpus
                self.retriever.full_bm25 = BM25Okapi(all_bm25_corpus)
                self.logger.info(
                    f"Rebuilt full BM25 index with {len(all_bm25_elements)} merged elements"
                )
            else:
                # Fallback: reconstruct from metadata
                self.logger.info("No BM25 data found, reconstructing from metadata...")
                elements = self._reconstruct_elements_from_metadata()

                if elements:
                    self.retriever.index_for_bm25(elements)
                    self.logger.info(
                        f"Rebuilt BM25 index with {len(elements)} elements"
                    )

                    if not graphs_loaded:
                        self.graph_builder.build_graphs(elements)
                        self.logger.info("Rebuilt code graph")
                else:
                    self.logger.warning("No elements reconstructed from metadata")

            # Build separate BM25 index for repository overviews
            self.retriever.build_repo_overview_bm25()
            self.logger.info("Built separate BM25 index for repository overviews")

            self.multi_repo_mode = True
            self.repo_indexed = True
            self.repo_loaded = True

            self.logger.info(
                f"Successfully loaded {len(repos_to_load)} repositories with {self.vector_store.get_count()} total vectors"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to load multi-repo cache: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # Incremental indexing
    # ------------------------------------------------------------------

    def _build_file_manifest(
        self, elements: list[CodeElement], repo_root: str
    ) -> dict[str, Any]:
        """Build a file manifest mapping files to their mtime/size and element IDs."""
        manifest: dict[str, Any] = {
            "repo_name": self.repo_info.get("name", ""),
            "created_at": datetime.now().isoformat(),
            "files": {},
        }

        for elem in elements:
            rel_path = elem.relative_path
            if rel_path not in manifest["files"]:
                abs_path = os.path.join(repo_root, rel_path)
                try:
                    stat = os.stat(abs_path)
                    manifest["files"][rel_path] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "element_ids": [],
                    }
                except OSError:
                    manifest["files"][rel_path] = {
                        "mtime": 0.0,
                        "size": 0,
                        "element_ids": [],
                    }
            manifest["files"][rel_path]["element_ids"].append(elem.id)

        return manifest

    def _save_file_manifest(self, repo_name: str, manifest: dict[str, Any]) -> None:
        """Save file manifest to disk as JSON."""
        manifest_path = os.path.join(
            self.vector_store.persist_dir, f"{repo_name}_manifest.json"
        )
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        self.logger.info(f"Saved file manifest: {manifest_path}")

    def _load_file_manifest(self, repo_name: str) -> dict[str, Any] | None:
        """Load file manifest from disk. Returns None if missing."""
        manifest_path = os.path.join(
            self.vector_store.persist_dir, f"{repo_name}_manifest.json"
        )
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path) as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load manifest for '{repo_name}': {e}")
            return None

    def _load_existing_metadata(self, repo_name: str) -> list[dict[str, Any]]:
        """Load existing vector store metadata for a repo directly from disk."""
        try:
            load_metadata_payload = getattr(
                self.vector_store, "load_metadata_payload", None
            )
            if callable(load_metadata_payload):
                data = load_metadata_payload(repo_name)
                metadata = data.get("metadata", []) if isinstance(data, dict) else []
                if isinstance(metadata, list):
                    return cast(list[dict[str, Any]], metadata)
        except Exception as e:
            self.logger.warning(f"Failed to load metadata for '{repo_name}': {e}")
        meta_path = os.path.join(
            self.vector_store.persist_dir, f"{repo_name}_metadata.pkl"
        )
        if not os.path.exists(meta_path):
            return []
        try:
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
            metadata = data.get("metadata", []) if isinstance(data, dict) else []
            return (
                cast(list[dict[str, Any]], metadata)
                if isinstance(metadata, list)
                else []
            )
        except Exception as e:
            self.logger.warning(f"Failed to load metadata for '{repo_name}': {e}")
        return []

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
            getattr(self, "graph_builder", None), "graph_artifact_paths", None
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
        """Compare current files against saved manifest to detect changes.

        Returns dict with added/modified/deleted/unchanged lists, or None
        if no manifest exists.
        """
        manifest = self._load_file_manifest(repo_name)
        if manifest is None:
            return None

        manifest_files = manifest.get("files", {})

        # Build lookup of current files with stat info
        current_lookup = {}
        for file_info in current_files:
            rel_path = file_info["relative_path"]
            abs_path = file_info["path"]
            try:
                stat = os.stat(abs_path)
                current_lookup[rel_path] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "file_info": file_info,
                }
            except OSError:
                continue

        added, modified, deleted, unchanged = [], [], [], []

        for rel_path, info in current_lookup.items():
            if rel_path not in manifest_files:
                added.append(rel_path)
            else:
                saved = manifest_files[rel_path]
                if info["mtime"] != saved["mtime"] or info["size"] != saved["size"]:
                    modified.append(rel_path)
                else:
                    unchanged.append(rel_path)

        for rel_path in manifest_files:
            if rel_path not in current_lookup:
                deleted.append(rel_path)

        return {
            "added": added,
            "modified": modified,
            "deleted": deleted,
            "unchanged": unchanged,
            "manifest": manifest,
            "current_lookup": current_lookup,
        }

    def _collect_unchanged_elements(
        self,
        manifest: dict[str, Any],
        unchanged_files: list[str],
        existing_metadata: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Collect element dicts and IDs for unchanged files from existing metadata."""
        unchanged_element_ids: set[str] = set()
        for rel_path in unchanged_files:
            file_entry: dict[str, Any] = manifest.get("files", {}).get(rel_path, {})
            for elem_id in file_entry.get("element_ids", []):
                unchanged_element_ids.add(elem_id)

        unchanged_elements = [
            meta
            for meta in existing_metadata
            if meta.get("id") in unchanged_element_ids
        ]

        return unchanged_elements, list(unchanged_element_ids)

    def incremental_reindex(
        self, repo_name: str, repo_path: str | None = None
    ) -> dict[str, Any]:
        if not repo_path or not os.path.isdir(repo_path):
            self.logger.warning(f"Invalid repo path for '{repo_name}': {repo_path}")
            return {"status": "path_not_found", "changes": 0}
        return self.run_index_pipeline(
            source=repo_path,
            is_url=False,
            publish=True,
            enable_scip=True,
        )

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

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """
        Get dialogue history for a session

        Args:
            session_id: Session ID

        Returns:
            List of dialogue turns
        """
        return self.cache_manager.get_dialogue_history(session_id)

    @staticmethod
    def _parse_working_memory_record_payload(
        record: Any,
    ) -> WorkingMemoryArtifact:
        payload = json.loads(str(record.payload_json or "{}"))
        if not isinstance(payload, dict):
            raise RuntimeError("working memory payload is invalid")
        return WorkingMemoryArtifact.from_dict(payload)

    @staticmethod
    def _parse_handoff_record_payload(record: Any) -> HandoffArtifact:
        payload = json.loads(str(record.payload_json or "{}"))
        if not isinstance(payload, dict):
            raise RuntimeError("handoff artifact payload is invalid")
        return HandoffArtifact.from_dict(payload)

    @staticmethod
    def _parse_turn_journal_record_payload(record: Any) -> TurnJournal:
        payload = json.loads(str(record.payload_json or "{}"))
        if not isinstance(payload, dict):
            raise RuntimeError("turn journal payload is invalid")
        return TurnJournal.from_dict(payload)

    @staticmethod
    def _parse_context_bundle_record_payload(record: Any) -> ContextBundle:
        payload = json.loads(str(record.payload_json or "{}"))
        if not isinstance(payload, dict):
            raise RuntimeError("context bundle payload is invalid")
        return ContextBundle.from_dict(payload)

    @staticmethod
    def _optional_string_tuple(
        value: Iterable[str] | str | None,
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in value)

    def _load_context_bundle_artifact(
        self,
        *,
        session_id: str | None = None,
        turn_number: int | None = None,
        bundle_id: str | None = None,
    ) -> ContextBundle:
        if bundle_id:
            get_by_id = getattr(
                self.cache_manager, "get_context_bundle_record_by_id", None
            )
            record = get_by_id(bundle_id) if callable(get_by_id) else None
            if record is None:
                raise RuntimeError(f"context bundle not found: {bundle_id}")
            return self._parse_context_bundle_record_payload(record)

        if not session_id:
            raise RuntimeError("session_id is required when bundle_id is omitted")

        record = None
        if turn_number is None:
            get_latest = getattr(
                self.cache_manager, "get_latest_context_bundle_record", None
            )
            record = get_latest(session_id) if callable(get_latest) else None
        else:
            get_record = getattr(self.cache_manager, "get_context_bundle_record", None)
            record = (
                get_record(session_id, turn_number) if callable(get_record) else None
            )
        if record is not None:
            return self._parse_context_bundle_record_payload(record)

        working_memory_record = (
            self.cache_manager.get_latest_working_memory_record(session_id)
            if turn_number is None
            else self.cache_manager.get_working_memory_record(session_id, turn_number)
        )
        if working_memory_record is None:
            raise RuntimeError(
                f"context bundle not found for session={session_id}, turn={turn_number}"
            )
        resolved_turn = int(working_memory_record.turn_number)
        journal_record = self.cache_manager.get_turn_journal_record(
            session_id, resolved_turn
        )
        if journal_record is None:
            raise RuntimeError(
                f"turn journal not found for session={session_id}, turn={resolved_turn}"
            )
        return build_context_bundle_artifact(
            working_memory=self._parse_working_memory_record_payload(
                working_memory_record
            ),
            turn_journal=self._parse_turn_journal_record_payload(journal_record),
        )

    @staticmethod
    def _context_bundle_response(
        bundle: ContextBundle,
        *,
        format: str,
        token_budget: int,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "bundle_id": bundle.bundle_id,
            "session_id": bundle.session_id,
            "turn_number": bundle.turn_number,
            "snapshot_id": bundle.snapshot_id,
            "artifact_key": bundle.artifact_key,
            "compiler_fingerprint": bundle.compiler_fingerprint,
            "format": format,
            "invalidation_key": bundle.distillation.invalidation_key,
            "activation_id": bundle.activation.activation_id,
            "distillation_id": bundle.distillation.distillation_id,
        }
        if format == "json":
            response["bundle"] = bundle.to_dict()
            return response
        if format == "rendered":
            response["rendered"] = render_context_bundle_artifact(
                bundle,
                token_budget=token_budget,
            )
            return response
        raise RuntimeError("format must be one of: json, rendered")

    def get_turn_context(
        self,
        session_id: str,
        turn_number: int | None = None,
        format: str = "fcx",
    ) -> dict[str, Any]:
        record = (
            self.cache_manager.get_latest_working_memory_record(session_id)
            if turn_number is None
            else self.cache_manager.get_working_memory_record(session_id, turn_number)
        )
        if record is None:
            raise RuntimeError(
                f"working memory not found for session={session_id}, turn={turn_number}"
            )
        artifact = self._parse_working_memory_record_payload(record)
        response: dict[str, Any] = {
            "session_id": record.session_id,
            "turn_number": record.turn_number,
            "snapshot_id": record.snapshot_id,
            "artifact_key": record.artifact_key,
            "compiler_fingerprint": record.compiler_fingerprint,
            "format": format,
        }
        if format == "fcx":
            response["stable_fcx"] = record.stable_fcx
            response["turn_fcx"] = record.turn_fcx
            response["obs_fcx"] = record.obs_fcx
            response["full_fcx"] = record.full_fcx
            return response
        if format == "json":
            response["artifact"] = artifact.to_dict()
            return response
        raise RuntimeError("format must be one of: fcx, json")

    def create_handoff(
        self,
        session_id: str,
        turn_number: int | None = None,
        mode: str = "delegate",
    ) -> dict[str, Any]:
        record = (
            self.cache_manager.get_latest_working_memory_record(session_id)
            if turn_number is None
            else self.cache_manager.get_working_memory_record(session_id, turn_number)
        )
        if record is None:
            raise RuntimeError(
                f"working memory not found for session={session_id}, turn={turn_number}"
            )
        artifact = build_handoff_from_working_memory(
            working_memory=self._parse_working_memory_record_payload(record),
            mode=mode,
        )
        payload_json = json.dumps(
            artifact.to_dict(),
            separators=(",", ":"),
            sort_keys=True,
        )
        self.cache_manager.save_handoff_artifact_record(
            HandoffArtifactRecord(
                artifact_id=artifact.artifact_id,
                session_id=artifact.session_id,
                turn_number=artifact.turn_number,
                snapshot_id=artifact.snapshot_id,
                compiler_fingerprint=artifact.compiler_fingerprint,
                mode=artifact.mode,
                payload_json=payload_json,
                full_fcx=artifact.full_fcx,
                created_at=artifact.created_at,
            )
        )
        return artifact.to_dict()

    def get_handoff_artifact(self, artifact_id: str) -> dict[str, Any]:
        record = self.cache_manager.get_handoff_artifact_record(artifact_id)
        if record is None:
            raise RuntimeError(f"handoff artifact not found: {artifact_id}")
        return self._parse_handoff_record_payload(record).to_dict()

    def get_context_bundle(
        self,
        session_id: str,
        turn_number: int | None = None,
        format: str = "json",
        token_budget: int = 2048,
    ) -> dict[str, Any]:
        bundle = self._load_context_bundle_artifact(
            session_id=session_id,
            turn_number=turn_number,
        )
        return self._context_bundle_response(
            bundle,
            format=format,
            token_budget=token_budget,
        )

    def get_context_bundle_by_id(
        self,
        bundle_id: str,
        format: str = "json",
        token_budget: int = 2048,
    ) -> dict[str, Any]:
        bundle = self._load_context_bundle_artifact(bundle_id=bundle_id)
        return self._context_bundle_response(
            bundle,
            format=format,
            token_budget=token_budget,
        )

    def expand_context_bundle_ref(
        self,
        ref_id: str,
        *,
        session_id: str | None = None,
        turn_number: int | None = None,
        bundle_id: str | None = None,
        depth: str = "L2",
    ) -> dict[str, Any]:
        bundle = self._load_context_bundle_artifact(
            session_id=session_id,
            turn_number=turn_number,
            bundle_id=bundle_id,
        )
        expanded = expand_bundle_source_ref(bundle, ref_id, depth=depth)
        if expanded is None:
            raise RuntimeError(f"context bundle ref not found: {ref_id}")
        return expanded

    def create_context_activation(
        self,
        session_id: str | None = None,
        turn_number: int | None = None,
        bundle_id: str | None = None,
        active_ref_ids: Iterable[str] | str | None = None,
        active_fact_ids: Iterable[str] | str | None = None,
        active_hypothesis_ids: Iterable[str] | str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        bundle = self._load_context_bundle_artifact(
            session_id=session_id,
            turn_number=turn_number,
            bundle_id=bundle_id,
        )
        activation = build_activation_record(
            bundle_id=bundle.bundle_id,
            working_memory=bundle.working_memory,
            active_ref_ids=self._optional_string_tuple(active_ref_ids),
            active_fact_ids=self._optional_string_tuple(active_fact_ids),
            active_hypothesis_ids=self._optional_string_tuple(active_hypothesis_ids),
            reason=reason,
            created_at=time.time(),
        )
        payload_json = json.dumps(
            activation.to_dict(),
            separators=(",", ":"),
            sort_keys=True,
        )
        self.cache_manager.save_context_activation_record(
            ContextActivationRecord(
                activation_id=activation.activation_id,
                bundle_id=activation.bundle_id,
                session_id=activation.session_id,
                turn_number=activation.turn_number,
                snapshot_id=activation.snapshot_id,
                compiler_fingerprint=activation.compiler_fingerprint,
                active_ref_ids=activation.active_ref_ids,
                active_fact_ids=activation.active_fact_ids,
                active_hypothesis_ids=activation.active_hypothesis_ids,
                reason=activation.reason,
                payload_json=payload_json,
                created_at=activation.created_at,
            )
        )
        return activation.to_dict()

    def expand_context_ref(
        self,
        session_id: str,
        turn_number: int,
        ref_id: str,
        depth: str = "L2",
    ) -> dict[str, Any]:
        record = self.cache_manager.get_working_memory_record(session_id, turn_number)
        if record is None:
            raise RuntimeError(
                f"working memory not found for session={session_id}, turn={turn_number}"
            )
        artifact = self._parse_working_memory_record_payload(record)
        for ref in artifact.evidence_refs:
            if ref.ref_id != ref_id:
                continue
            return {
                "session_id": session_id,
                "turn_number": turn_number,
                "depth": depth,
                "ref_id": ref.ref_id,
                "kind": ref.kind,
                "repo_name": ref.repo_name,
                "snapshot_id": ref.snapshot_id,
                "path": ref.path,
                "symbol_id": ref.symbol_id,
                "lines": ref.lines,
                "label": ref.label,
                "score": ref.score,
                "source": ref.source,
                "fresh": ref.fresh,
            }
        raise RuntimeError(f"context ref not found: {ref_id}")

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a dialogue session

        Args:
            session_id: Session ID

        Returns:
            True if successful
        """
        return self.cache_manager.delete_session(session_id)

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

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all dialogue sessions with enriched metadata

        Returns:
            List of session metadata with first query as title
        """
        list_session_records = getattr(self.cache_manager, "list_session_records", None)
        get_turn_record = getattr(self.cache_manager, "get_dialogue_turn_record", None)
        if callable(list_session_records) and callable(get_turn_record):
            enriched_sessions: list[dict[str, Any]] = []
            for session in cast(list[Any], list_session_records()):
                session_id = str(session.session_id)
                title = "Unknown Session"
                if session_id:
                    first_turn = cast(Any, get_turn_record(session_id, 1))
                    if first_turn is not None:
                        first_query = str(first_turn.query)
                        title = (
                            first_query[:77] + "..."
                            if len(first_query) > 80
                            else first_query
                        )
                    else:
                        title = f"Session {session_id}"
                enriched_sessions.append(
                    {
                        "session_id": session_id,
                        "created_at": float(session.created_at),
                        "total_turns": int(session.total_turns),
                        "last_updated": float(session.last_updated),
                        "multi_turn": bool(session.multi_turn),
                        "title": title,
                    }
                )
            return enriched_sessions

        sessions = self.cache_manager.list_sessions()

        # Enrich each session with the first query as a title
        enriched_sessions: list[dict[str, Any]] = []
        for session in sessions:
            session_id = session.get("session_id", "")
            if session_id:
                # Get the first turn to use its query as title
                first_turn = self.cache_manager.get_dialogue_turn(session_id, 1)
                if first_turn:
                    first_query = first_turn.get("query", "")
                    # Truncate long queries
                    if len(first_query) > 80:
                        title = first_query[:77] + "..."
                    else:
                        title = first_query
                    session["title"] = title
                else:
                    session["title"] = f"Session {session_id}"
            else:
                session["title"] = "Unknown Session"

            enriched_sessions.append(session)

        return enriched_sessions
