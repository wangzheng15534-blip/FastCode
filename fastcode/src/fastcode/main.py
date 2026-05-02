"""
Main FastCode Class - Orchestrate all components
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import json
import os
import pickle
from collections.abc import Callable, Generator
from datetime import datetime
from typing import Any

import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi

from .answer_generator import AnswerGenerator
from .cache import CacheManager
from .core import snapshot as _snapshot
from .doc_ingester import KeyDocIngester
from .embedder import CodeEmbedder
from .global_index_builder import GlobalIndexBuilder
from .graph_builder import CodeGraphBuilder
from .graph_runtime import LadybugGraphRuntime
from .index_run import IndexRunStore
from .indexer import CodeElement, CodeElementMeta, CodeIndexer
from .ir_graph_builder import IRGraphBuilder
from .loader import RepositoryLoader
from .manifest_store import ManifestStore
from .module_resolver import ModuleResolver
from .parser import CodeParser
from .pg_retrieval import PgRetrievalStore
from .pipeline import IndexPipeline
from .projection import ProjectionService
from .projection_store import ProjectionStore
from .projection_transform import ProjectionTransformer
from .publishing import PublishingService
from .query_handler import QueryPipeline
from .query_processor import QueryProcessor
from .redo_worker import RedoWorker
from .retriever import HybridRetriever
from .semantic_ir import IRSnapshot
from .semantic_resolvers import (
    build_default_semantic_resolver_registry,
)
from .snapshot_store import SnapshotStore
from .snapshot_symbol_index import SnapshotSymbolIndex
from .symbol_resolver import SymbolResolver
from .terminus_publisher import TerminusPublisher
from .utils import (
    ensure_dir,
    load_config,
    resolve_config_paths,
    setup_logging,
)
from .vector_store import VectorStore


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
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        # Load configuration
        if config_path is None:
            # Try to find config in standard locations
            possible_paths = [
                "config/config.yaml",
                "../config/config.yaml",
                os.path.join(os.path.dirname(__file__), "../../config/config.yaml"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break

        if config_path and os.path.exists(config_path):
            self.config = load_config(config_path)
        else:
            # Use default configuration
            self.config = self._get_default_config()
            self.config = resolve_config_paths(self.config, project_root)

        # Evaluation-specific overrides (keep core system decoupled)
        self.eval_config = self.config.get("evaluation", {})
        self.eval_mode = self.eval_config.get("enabled", False)
        self.in_memory_index = self.eval_config.get("in_memory_index", False)

        # Ensure in-memory mode disables disk-based caches/persistence
        if self.in_memory_index:
            self.config.setdefault("vector_store", {})["in_memory"] = True
            self.config.setdefault("cache", {})["enabled"] = False

        # Allow explicit cache disable via evaluation config
        if self.eval_config.get("disable_cache", False):
            self.config.setdefault("cache", {})["enabled"] = False

        # Setup logging
        self.logger = setup_logging(self.config)
        self.logger.info("Initializing FastCode system")

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
            semantic_escalation_cb=self._escalate_query_semantics,
        )

        # Multi-repository state
        self.multi_repo_mode: bool = False
        self.loaded_repositories: dict[
            str, dict[str, Any]
        ] = {}  # {repo_name: repo_info}

    @staticmethod
    def _infer_is_url(source: str) -> bool:
        return IndexPipeline._infer_is_url(source)

    def load_repository(
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
                self.config["repo_root"] = self.loader.repo_path
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
        """
        Index the loaded repository

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
                    metadata.append(elem.to_dict())

            if vectors:
                vectors_array: np.ndarray = np.array(vectors)
                self.vector_store.add_vectors(vectors_array, metadata)

            # Initialize resolvers for complete graph building
            # This fixes the "0 edges" issue by providing the necessary context for resolution
            try:
                self.logger.info("Initializing resolvers for precise graph building...")

                # Ensure repo_root is set
                repo_root = self.config.get("repo_root")
                if not repo_root and self.loader.repo_path:
                    repo_root = self.loader.repo_path
                    self.config["repo_root"] = repo_root

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

    def process_redo_tasks(self, limit: int = 10) -> dict[str, Any]:
        return self.publishing_service.process_redo_tasks(limit)

    def list_repo_refs(self, repo_name: str) -> list[dict[str, Any]]:
        with self.snapshot_store.db_runtime.connect() as conn:
            rows = self.snapshot_store.db_runtime.execute(
                conn,
                """
                SELECT branch, commit_id, tree_id, snapshot_id, created_at
                FROM snapshot_refs
                WHERE repo_name=?
                ORDER BY created_at DESC
                """,
                (repo_name,),
            ).fetchall()
        return [
            d
            for r in rows
            if r
            for d in [self.snapshot_store.db_runtime.row_to_dict(r)]
            if d is not None
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
        snapshot = self.snapshot_store.load_snapshot(snapshot_id)
        if not snapshot:
            return None
        for symbol in snapshot.symbols:
            if symbol.symbol_id == resolved:
                return symbol.to_dict()
        return None

    def get_graph_callees(
        self, snapshot_id: str, symbol_id: str, max_hops: int = 1
    ) -> list[dict[str, Any]]:
        max_hops = max(1, min(max_hops, 20))
        ir_graphs = self.snapshot_store.load_ir_graphs(snapshot_id)
        if not ir_graphs:
            return []
        g = ir_graphs.call_graph
        if symbol_id not in g:
            return []
        dist = nx.single_source_shortest_path_length(g, symbol_id, cutoff=max_hops)
        return [
            {"symbol_id": node, "distance": d}
            for node, d in dist.items()
            if node != symbol_id
        ]

    def get_graph_callers(
        self, snapshot_id: str, symbol_id: str, max_hops: int = 1
    ) -> list[dict[str, Any]]:
        max_hops = max(1, min(max_hops, 20))
        ir_graphs = self.snapshot_store.load_ir_graphs(snapshot_id)
        if not ir_graphs:
            return []
        g = ir_graphs.call_graph.reverse(copy=False)
        if symbol_id not in g:
            return []
        dist = nx.single_source_shortest_path_length(g, symbol_id, cutoff=max_hops)
        return [
            {"symbol_id": node, "distance": d}
            for node, d in dist.items()
            if node != symbol_id
        ]

    def get_graph_dependencies(
        self, snapshot_id: str, doc_id: str, max_hops: int = 1
    ) -> list[dict[str, Any]]:
        max_hops = max(1, min(max_hops, 20))
        ir_graphs = self.snapshot_store.load_ir_graphs(snapshot_id)
        if not ir_graphs:
            return []
        g = ir_graphs.dependency_graph
        if doc_id not in g:
            return []
        dist = nx.single_source_shortest_path_length(g, doc_id, cutoff=max_hops)
        return [
            {"doc_id": node, "distance": d}
            for node, d in dist.items()
            if node != doc_id
        ]

    def get_branch_manifest(
        self, repo_name: str, ref_name: str
    ) -> dict[str, Any] | None:
        return self.manifest_store.get_branch_manifest(repo_name, ref_name)

    def get_snapshot_manifest(self, snapshot_id: str) -> dict[str, Any] | None:
        return self.manifest_store.get_snapshot_manifest(snapshot_id)

    def get_scip_artifact_ref(self, snapshot_id: str) -> dict[str, Any] | None:
        return self.snapshot_store.get_scip_artifact_ref(snapshot_id)

    def list_scip_artifact_refs(self, snapshot_id: str) -> list[dict[str, Any]]:
        return self.snapshot_store.list_scip_artifact_refs(snapshot_id)

    def resolve_snapshot_symbol(
        self,
        snapshot_id: str,
        *,
        symbol_id: str | None = None,
        name: str | None = None,
        path: str | None = None,
    ) -> str | None:
        if not self.snapshot_symbol_index.has_snapshot(snapshot_id):
            snap = self.snapshot_store.load_snapshot(snapshot_id)
            if snap:
                self.snapshot_symbol_index.register_snapshot(snap)
        return self.snapshot_symbol_index.resolve_symbol(
            snapshot_id,
            symbol_id=symbol_id,
            name=name,
            path=path,
        )

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

        elements = list(self.graph_builder.element_by_id.values())
        warnings: list[str] = []
        upgraded_snapshot = self._apply_semantic_resolvers(
            snapshot=snapshot,
            elements=elements,
            legacy_graph_builder=self.graph_builder,
            target_paths=target_paths,
            warnings=warnings,
            budget=budget,
        )
        self.snapshot_symbol_index.register_snapshot(upgraded_snapshot)
        upgraded_ir_graphs = self.ir_graph_builder.build_graphs(upgraded_snapshot)
        self.retriever.set_ir_graphs(upgraded_ir_graphs, snapshot_id=snapshot_id)

        semantic_runs = list(
            (upgraded_snapshot.metadata or {}).get("semantic_resolver_runs", [])
        )
        return {
            "status": "degraded" if warnings else "applied",
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
            },
            # ── vector_store ─────────────────────────────────────────
            "vector_store": {
                "persist_directory": "./data/vector_store",  # [TUNABLE]
                "distance_metric": "cosine",  # [INTERNAL] similarity metric
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
        """
        Load and index multiple repositories (saves each repository separately)

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
                    self.config["repo_root"] = self.loader.repo_path

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
                        metadata.append(elem.to_dict())

                if vectors:
                    vectors_array: np.ndarray = np.array(vectors)
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
            persist_dir = self.vector_store.persist_dir
            available_repos: list[str] = []

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

            for repo_name in repos_to_load:
                # Try loading BM25 for each repo
                bm25_path = os.path.join(
                    self.retriever.persist_dir, f"{repo_name}_bm25.pkl"
                )
                if os.path.exists(bm25_path):
                    try:
                        with open(bm25_path, "rb") as f:
                            data = pickle.load(f)
                            all_bm25_corpus.extend(data["bm25_corpus"])

                            # Reconstruct CodeElement objects
                            for elem_dict in data["bm25_elements"]:
                                all_bm25_elements.append(CodeElement(**elem_dict))

                        self.logger.info(f"Loaded BM25 data for {repo_name}")
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load BM25 data for {repo_name}: {e}"
                        )

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
        meta_path = os.path.join(
            self.vector_store.persist_dir, f"{repo_name}_metadata.pkl"
        )
        if not os.path.exists(meta_path):
            return []
        try:
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
            return data.get("metadata", [])
        except Exception as e:
            self.logger.warning(f"Failed to load metadata for '{repo_name}': {e}")
            return []

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
        """Perform incremental reindexing: only re-embed changed files.

        Unchanged files reuse their existing embeddings. FAISS, BM25, and
        graphs are rebuilt from the combined element set (fast, since no
        model inference is needed for unchanged elements).

        Args:
            repo_name: Canonical repository name.
            repo_path: Local filesystem path to the repository.

        Returns:
            Dict with status and change summary.
        """
        self.logger.info(f"Starting incremental reindex for '{repo_name}'")

        # 1. Load manifest (skip if missing)
        manifest = self._load_file_manifest(repo_name)
        if manifest is None:
            self.logger.info(f"No manifest for '{repo_name}', skipping incremental")
            return {"status": "no_manifest", "changes": 0}

        # 2. Set up loader for this repo
        if not repo_path or not os.path.isdir(repo_path):
            self.logger.warning(f"Invalid repo path for '{repo_name}': {repo_path}")
            return {"status": "path_not_found", "changes": 0}

        self.loader.load_from_path(repo_path)
        self.config["repo_root"] = repo_path

        # 3. Scan current files and detect changes
        current_files = self.loader.scan_files()
        changes = self._detect_file_changes(repo_name, current_files)
        if changes is None:
            return {"status": "no_manifest", "changes": 0}

        added = changes["added"]
        modified = changes["modified"]
        deleted = changes["deleted"]
        unchanged = changes["unchanged"]
        total_changes = len(added) + len(modified) + len(deleted)

        self.logger.info(
            f"Changes: +{len(added)} ~{len(modified)} -{len(deleted)} ={len(unchanged)}"
        )

        if total_changes == 0:
            return {"status": "no_changes", "changes": 0}

        # 4. Load existing metadata from disk
        existing_metadata = self._load_existing_metadata(repo_name)
        if not existing_metadata:
            self.logger.warning(f"No existing metadata for '{repo_name}'")
            return {"status": "no_metadata", "changes": 0}

        # 5. Collect unchanged elements (with pre-computed embeddings)
        unchanged_elements, _ = self._collect_unchanged_elements(
            changes["manifest"], unchanged, existing_metadata
        )
        self.logger.info(
            f"Preserved {len(unchanged_elements)} elements from {len(unchanged)} unchanged files"
        )

        # 6. Parse & embed changed files
        changed_file_infos: list[dict[str, Any]] = []
        for rp in added + modified:
            lookup = changes["current_lookup"].get(rp)
            if lookup and lookup.get("file_info"):
                changed_file_infos.append(lookup["file_info"])

        new_elements: list[CodeElement] = []
        if changed_file_infos:
            repo_url = self.loaded_repositories.get(repo_name, {}).get("url")
            new_elements = self.indexer.index_files(
                changed_file_infos, repo_name, repo_url
            )
            self.logger.info(
                f"Indexed {len(new_elements)} elements from {len(changed_file_infos)} changed files"
            )

        # 7. Combine: convert unchanged metadata dicts → CodeElement objects
        all_elements: list[CodeElement] = []
        for meta in unchanged_elements:
            try:
                all_elements.append(
                    CodeElement(
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
                )
            except Exception as e:
                self.logger.warning(f"Failed to reconstruct element: {e}")
        all_elements.extend(new_elements)
        self.logger.info(f"Total elements after merge: {len(all_elements)}")

        # 8. Rebuild FAISS (temporary store — main instance untouched)
        temp_store = VectorStore(self.config)
        temp_store.initialize(self.embedder.embedding_dim)

        vectors: list[list[float]] = []
        metadata_list: list[CodeElementMeta] = []
        for elem in all_elements:
            embedding = elem.metadata.get("embedding")
            if embedding is not None:
                vectors.append(embedding)
                metadata_list.append(elem.to_dict())

        if vectors:
            temp_store.add_vectors(np.array(vectors), metadata_list)

        # 9. Rebuild BM25 (temporary retriever)
        temp_retriever = HybridRetriever(
            self.config,
            temp_store,
            self.embedder,
            CodeGraphBuilder(self.config),
            repo_root=repo_path,
        )
        temp_retriever.index_for_bm25(all_elements)

        # 10. Rebuild graphs (temporary builder)
        temp_graph = CodeGraphBuilder(self.config)
        module_resolver, symbol_resolver = None, None
        try:
            gib = GlobalIndexBuilder(self.config)
            gib.build_maps(all_elements, repo_path)
            module_resolver = ModuleResolver(gib)
            symbol_resolver = SymbolResolver(gib, module_resolver)
        except Exception as e:
            self.logger.warning(f"Resolver init failed during incremental reindex: {e}")

        temp_graph.build_graphs(all_elements, module_resolver, symbol_resolver)

        # 11. Save all artifacts
        if self._should_persist_indexes():
            temp_store.save(repo_name)
            temp_retriever.save_bm25(repo_name)
            temp_graph.save(repo_name)
            new_manifest = self._build_file_manifest(all_elements, repo_path)
            self._save_file_manifest(repo_name, new_manifest)
            self.logger.info(f"Saved all artifacts for '{repo_name}'")

        return {
            "status": "success",
            "changes": total_changes,
            "added_files": len(added),
            "modified_files": len(modified),
            "deleted_files": len(deleted),
            "unchanged_files": len(unchanged),
            "total_elements": len(all_elements),
            "new_elements": len(new_elements),
            "preserved_elements": len(unchanged_elements),
        }

    def cleanup(self):
        """Cleanup resources"""
        self.shutdown()
        self.loader.cleanup()
        self.logger.info("Cleanup complete")

    def shutdown(self):
        """Stop background workers."""
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

        persist_dir = self.vector_store.persist_dir
        deleted_files: list[str] = []
        freed_bytes = 0

        # Files to delete from vector_store directory
        file_patterns = [
            f"{repo_name}.faiss",
            f"{repo_name}_metadata.pkl",
            f"{repo_name}_bm25.pkl",
            f"{repo_name}_graphs.pkl",
        ]

        for fname in file_patterns:
            fpath = os.path.join(persist_dir, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                os.remove(fpath)
                deleted_files.append(fname)
                freed_bytes += size
                self.logger.info(f"Deleted {fpath} ({size / (1024 * 1024):.2f} MB)")

        # Remove overview entry from repo_overviews.pkl
        if self.vector_store.delete_repo_overview(repo_name):
            deleted_files.append("repo_overviews.pkl (entry)")
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
