"""
Main FastCode Class - Orchestrate all components
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import json
import logging
import os
import re
import shutil
import tempfile
import time
import zipfile
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

import fastcode.retrieval.context.snapshot as _snapshot
from fastcode.app.indexing.doc_ingester import KeyDocIngester
from fastcode.app.indexing.embedder import CodeEmbedder
from fastcode.app.indexing.extractors.parser import CodeParser
from fastcode.app.indexing.loader import RepositoryLoader
from fastcode.app.indexing.pipeline.direct_indexer import DirectIndexer
from fastcode.app.indexing.pipeline.indexer import CodeIndexer
from fastcode.app.indexing.pipeline.manifest import (
    build_file_manifest,
    collect_unchanged_elements,
    detect_file_changes,
    load_existing_metadata,
    load_file_manifest,
    save_file_manifest,
)
from fastcode.app.indexing.pipeline.multi_repo_direct import MultiRepoDirectIndexer
from fastcode.app.indexing.pipeline.redo_worker import RedoWorker
from fastcode.app.indexing.pipeline.service import IndexPipeline
from fastcode.app.indexing.projection.service import ProjectionService
from fastcode.app.indexing.projection.transform import ProjectionTransformer
from fastcode.app.indexing.publishing import PublishingService
from fastcode.app.indexing.publishing_facade import PublishingFacade
from fastcode.app.indexing.terminus import TerminusPublisher
from fastcode.app.query.context_payloads import (
    activation_payload,
    context_bundle_from_payload,
    context_bundle_payload,
    handoff_from_payload,
    handoff_payload,
    turn_journal_from_payload,
    working_memory_from_payload,
    working_memory_payload,
)
from fastcode.app.query.orchestration.answer import AnswerGenerator
from fastcode.app.query.orchestration.handler import QueryPipeline
from fastcode.app.query.orchestration.processor import QueryProcessor
from fastcode.app.query.selection.retriever import HybridRetriever
from fastcode.app.store.artifacts.file import FileArtifactStore
from fastcode.app.store.artifacts.graph import GraphArtifactStore
from fastcode.app.store.artifacts.unit import UnitArtifactStore
from fastcode.app.store.cache.contracts import (
    ContextActivationRecord,
    DialogueTurnRecord,
    HandoffArtifactRecord,
)
from fastcode.app.store.cache.rehydration import (
    load_multi_repo_cache as _load_multi_repo_cache_impl,
)
from fastcode.app.store.cache.rehydration import (
    reconstruct_elements_from_metadata,
)
from fastcode.app.store.cache.rehydration import (
    save_to_cache as _save_to_cache_impl,
)
from fastcode.app.store.cache.rehydration import (
    try_load_from_cache as _try_load_from_cache_impl,
)
from fastcode.app.store.cache.service import CacheManager
from fastcode.app.store.runs.index_run import IndexRunStore
from fastcode.app.store.runs.index_run_contracts import IndexRunRecord
from fastcode.app.store.snapshots.manifest import ManifestStore
from fastcode.app.store.snapshots.manifest_contracts import ManifestRecord
from fastcode.app.store.snapshots.projection import ProjectionStore
from fastcode.app.store.snapshots.snapshot import SnapshotStore
from fastcode.app.store.snapshots.snapshot_contracts import (
    SCIPArtifactRecord,
    SnapshotRefRecord,
)
from fastcode.app.store.vectors.pg_retrieval import PgRetrievalStore
from fastcode.app.store.vectors.vector import VectorStore
from fastcode.graph.build import CodeGraphBuilder
from fastcode.infrastructure.execution.scip_runner import SubprocessScipIndexerRuntime
from fastcode.infrastructure.execution.semantic_helper import (
    SubprocessSemanticHelperRuntime,
)
from fastcode.infrastructure.graph_runtime.contracts import DocumentGraphRuntime
from fastcode.infrastructure.graph_runtime.ladybug import LadybugGraphRuntime
from fastcode.infrastructure.storage.runtime import DBRuntime
from fastcode.ir.code_status import build_code_status_pack
from fastcode.ir.element import CodeElement
from fastcode.ir.graph import IRGraphBuilder
from fastcode.ir.types import IRSnapshot, IRSymbol
from fastcode.kernel.config import FastCodeConfig
from fastcode.main.config import config_from_mapping
from fastcode.retrieval.context.agent_context import (
    ContextBundle,
    HandoffArtifact,
    TurnJournal,
    WorkingMemoryArtifact,
)
from fastcode.retrieval.context.context_compiler import (
    build_activation_record,
    build_handoff_from_working_memory,
    expand_bundle_source_ref,
)
from fastcode.retrieval.context.context_compiler import (
    build_context_bundle as build_context_bundle_artifact,
)
from fastcode.retrieval.context.context_compiler import (
    render_context_bundle as render_context_bundle_artifact,
)
from fastcode.retrieval.contracts import Hit, SourceCitation
from fastcode.scip.global_builder import GlobalIndexBuilder
from fastcode.scip.module_resolver import ModuleResolver
from fastcode.scip.symbol_resolver import SymbolResolver
from fastcode.semantic.resolvers.engine.registry import (
    build_default_semantic_resolver_registry,
)
from fastcode.semantic.symbol_index import SnapshotSymbolIndex
from fastcode.utils.archive import safe_extract_zip, safe_repo_name_from_archive
from fastcode.utils.clock import utc_now
from fastcode.utils.filesystem import ensure_dir
from fastcode.utils.json import safe_jsonable

from .config import (
    config_to_runtime_mapping,
    load_runtime_config,
    prepare_runtime_config_mapping,
    setup_logging,
)
from .runtime_state import RuntimeState


class _VectorSearchStoreFactory:
    """Composition-root factory for query-scoped temporary vector stores."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    def create_vector_search_store(self) -> VectorStore:
        return VectorStore(self._config)


_DIAGNOSTIC_PYTHON_DEPENDENCIES: tuple[tuple[str, str, str], ...] = (
    ("core", "numpy", "numpy"),
    ("core", "pydantic", "pydantic"),
    ("core", "tree_sitter", "tree_sitter"),
    ("core", "gitpython", "git"),
    ("retrieval", "faiss", "faiss"),
    ("retrieval", "rank_bm25", "rank_bm25"),
    ("retrieval", "networkx", "networkx"),
    ("retrieval", "igraph", "igraph"),
    ("llm", "openai", "openai"),
    ("llm", "anthropic", "anthropic"),
    ("llm", "tiktoken", "tiktoken"),
    ("api", "fastapi", "fastapi"),
    ("api", "uvicorn", "uvicorn"),
    ("api", "flask", "flask"),
    ("postgres", "psycopg", "psycopg"),
    ("postgres", "psycopg_pool", "psycopg_pool"),
    ("postgres", "pgvector", "pgvector"),
    ("cache", "redis", "redis"),
    ("docs", "chonkie", "chonkie"),
    ("embeddings", "sentence_transformers", "sentence_transformers"),
    ("mcp", "mcp", "mcp"),
    ("scip", "protobuf", "google.protobuf"),
    ("ladybug", "real_ladybug", "real_ladybug"),
)

_DIAGNOSTIC_EXTERNAL_TOOLS: tuple[tuple[str, str], ...] = (
    ("git", "git"),
    ("scip", "scip"),
    ("node", "node"),
    ("go", "go"),
    ("cargo", "cargo"),
    ("javac", "javac"),
)

_DIAGNOSTIC_REDACTION = "[redacted]"
_DIAGNOSTIC_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(api[_-]?key|token|password|passwd|secret)"
    r"(\s*[:=]\s*)"
    r"[^,\s;]+",
    re.IGNORECASE,
)
_DIAGNOSTIC_AUTH_HEADER_RE = re.compile(
    r"\b(authorization)(\s*[:=]\s*)(?:(Bearer)\s+)?[^,\s;]+",
    re.IGNORECASE,
)
_DIAGNOSTIC_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_DIAGNOSTIC_URL_USERINFO_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9+.-]*://)([^/\s@]+@)")
_DIAGNOSTIC_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "auth_token",
    "authorization",
    "password",
    "passwd",
    "secret",
    "credential",
    "private_key",
    "postgres_dsn",
    "dsn",
    "cookie",
)


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

        # Runtime state (extracted for injection into facades)
        self.state = RuntimeState()

        # Initialize resolver attributes (will be set in index_repository)
        self.global_index_builder: GlobalIndexBuilder | None = None
        self.module_resolver: ModuleResolver | None = None
        self.symbol_resolver: SymbolResolver | None = None

        # Initialize components
        self.loader = RepositoryLoader(self.config)
        self.parser = CodeParser(self.config)
        self.embedder = CodeEmbedder(self.config)
        self.vector_store = VectorStore(self.config)
        self.vector_store_factory = _VectorSearchStoreFactory(self.config)
        self.indexer = CodeIndexer(
            self.config, self.loader, self.parser, self.embedder, self.vector_store
        )
        self.graph_builder = CodeGraphBuilder(self.config)
        self.graph_artifact_store = GraphArtifactStore(self.config)
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
            vector_store_factory=self.vector_store_factory,
        )
        self.query_processor = QueryProcessor(self.config)
        self.answer_generator = AnswerGenerator(self.config)
        self.cache_manager = CacheManager(self.config)

        persist_dir = self.vector_store.persist_dir
        storage_cfg: dict[str, Any] = self.config.get("storage", {}) or {}
        db_runtime = DBRuntime.from_storage_config(
            sqlite_path=os.path.join(os.path.abspath(persist_dir), "lineage.db"),
            storage_cfg=storage_cfg,
        )
        self.snapshot_store = SnapshotStore(persist_dir, db_runtime=db_runtime)
        self.manifest_store = ManifestStore(db_runtime)
        self.index_run_store = IndexRunStore(db_runtime)
        self.unit_artifact_store = UnitArtifactStore(db_runtime)
        self.file_artifact_store = FileArtifactStore(db_runtime)
        self.terminus_publisher = TerminusPublisher(self.config)
        self.projection_transformer = ProjectionTransformer(self.config)
        self.projection_store = ProjectionStore(self.config)
        self.snapshot_symbol_index = SnapshotSymbolIndex()
        self.pg_retrieval_store = PgRetrievalStore(db_runtime, self.config)
        self.retriever.set_pg_retrieval_store(self.pg_retrieval_store)
        self.doc_ingester = KeyDocIngester(self.config, self.embedder)
        self.graph_runtime: DocumentGraphRuntime | None = None
        try:
            self.graph_runtime = LadybugGraphRuntime(self.config)
        except ImportError:
            self.logger.info(
                "LadybugGraphRuntime unavailable — graph persistence disabled"
            )

        self._redo_worker: RedoWorker | None = None
        if db_runtime.backend == "postgres":
            poll_interval = int(storage_cfg.get("redo_poll_interval_seconds", 30))
            self._redo_worker = RedoWorker(self, poll_interval_seconds=poll_interval)
            self._redo_worker.start()

        self.semantic_helper_runtime = SubprocessSemanticHelperRuntime()
        self.scip_indexer_runtime = SubprocessScipIndexerRuntime()
        self.semantic_resolver_registry = build_default_semantic_resolver_registry(
            semantic_helper_runtime=self.semantic_helper_runtime
        )

        # State lives on self.state (RuntimeState) — no direct fields

        # --- IndexPipeline ---
        self.pipeline = IndexPipeline(
            config=self.config,
            logger=self.logger,
            loader=self.loader,
            snapshot_store=self.snapshot_store,
            manifest_store=self.manifest_store,
            index_run_store=self.index_run_store,
            unit_artifact_store=self.unit_artifact_store,
            file_artifact_store=self.file_artifact_store,
            snapshot_symbol_index=self.snapshot_symbol_index,
            vector_store=self.vector_store,
            embedder=self.embedder,
            indexer=self.indexer,
            retriever=self.retriever,
            graph_builder=self.graph_builder,
            ir_graph_builder=self.ir_graph_builder,
            graph_artifact_store=self.graph_artifact_store,
            pg_retrieval_store=self.pg_retrieval_store,
            terminus_publisher=self.terminus_publisher,
            doc_ingester=self.doc_ingester,
            semantic_resolver_registry=self.semantic_resolver_registry,
            set_repo_indexed=lambda v: setattr(self.state, "repo_indexed", v),
            set_repo_loaded=lambda v: setattr(self.state, "repo_loaded", v),
            set_repo_info=lambda v: setattr(self.state, "repo_info", v),
            semantic_helper_runtime=self.semantic_helper_runtime,
            scip_indexer_runtime=self.scip_indexer_runtime,
        )

        # --- Direct indexing (use_flow delegate) ---
        self._direct_indexer = DirectIndexer(
            config=self.config,
            loader=self.loader,
            indexer=self.indexer,
            embedder=self.embedder,
            vector_store=self.vector_store,
            graph_builder=self.graph_builder,
            retriever=self.retriever,
            graph_artifact_store=self.graph_artifact_store,
            should_use_cache_fn=self._should_use_cache,
            try_load_from_cache_fn=self._try_load_from_cache,
            should_persist_fn=self._should_persist_indexes,
            save_to_cache_fn=self._save_to_cache,
            set_repo_root_fn=self._set_repo_root,
            log_statistics_fn=self._log_statistics,
        )
        self._multi_repo_direct_indexer = MultiRepoDirectIndexer(
            config=self.config,
            loader=self.loader,
            parser=self.parser,
            embedder=self.embedder,
            vector_store=self.vector_store,
            graph_builder=self.graph_builder,
            graph_artifact_store=self.graph_artifact_store,
            set_repo_root_fn=self._set_repo_root,
            infer_is_url_fn=self._infer_is_url,
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
        self.publishing = PublishingFacade(
            publishing_service=self.publishing_service,
            pipeline=self.pipeline,
            projection_store=self.projection_store,
            snapshot_store=self.snapshot_store,
            config=self.config,
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
            is_repo_indexed=lambda: self.state.repo_indexed,
            load_artifacts_by_key=self.pipeline._load_artifacts_by_key,
            load_snapshot_artifacts=self.pipeline.load_snapshot_artifacts_handle,
            get_session_prefix=self.get_session_prefix,
            semantic_escalation_cb=self._escalate_query_semantics,
        )

        # Multi-repository state lives on self.state

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

            self.state.repo_loaded = True
            self.state.repo_info = self.loader.get_repository_info()

            # CRITICAL: Update config with the actual repo path.
            # This ensures path_utils can correctly normalize paths relative to the root.
            if self.loader.repo_path:
                self._set_repo_root(self.loader.repo_path)
                self.logger.info(f"Set repo_root to: {self.loader.repo_path}")

                # Initialize retriever agents if agency mode is enabled
                self.retriever.set_repo_root(self.loader.repo_path)

            self.logger.info(f"Loaded repository: {self.state.repo_info.get('name')}")
            self.logger.info(
                f"Files: {self.state.repo_info.get('file_count')}, "
                f"Size: {self.state.repo_info.get('total_size_mb', 0):.2f} MB"
            )

        except Exception as e:
            self.logger.error(f"Failed to load repository: {e}")
            raise

    def upload_repository_zip(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        """Upload a ZIP archive, extract, and load the repository.

        Receives raw bytes so the caller (entry frame) owns the protocol-
        specific file reading. Returns a result dict with status, repo_info,
        and repo_path.
        """
        with self._state_lock():
            return self._upload_repository_zip_unlocked(file_bytes, filename)

    def _upload_repository_zip_unlocked(
        self, file_bytes: bytes, filename: str
    ) -> dict[str, Any]:
        max_size = 100 * 1024 * 1024  # 100MB
        if len(file_bytes) > max_size:
            raise ValueError(
                f"File too large. Maximum size is {max_size / (1024 * 1024)}MB"
            )

        archive_filename = Path(filename).name
        repo_name = safe_repo_name_from_archive(archive_filename)

        repo_workspace = getattr(self.loader, "safe_repo_root", "./repos")
        repos_dir = Path(repo_workspace)
        repos_dir.mkdir(parents=True, exist_ok=True)
        repo_path = repos_dir / repo_name

        if repo_path.exists():
            self.loader._backup_existing_repo(str(repo_path))

        temp_dir = tempfile.mkdtemp(prefix="fastcode_upload_")
        try:
            zip_path = Path(temp_dir) / archive_filename

            self.logger.info(
                "Saving uploaded ZIP file: %s (%d bytes)",
                archive_filename,
                len(file_bytes),
            )
            zip_path.write_bytes(file_bytes)

            extract_dir = Path(temp_dir) / "extracted"
            extract_dir.mkdir(exist_ok=True)

            self.logger.info(
                "Extracting ZIP file to temporary directory: %s", extract_dir
            )
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                safe_extract_zip(zip_ref, extract_dir)

            extracted_items = list(extract_dir.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                source_repo_path = extracted_items[0]
            else:
                source_repo_path = extract_dir

            self.logger.info("Moving repository to: %s", repo_path)
            shutil.move(str(source_repo_path), str(repo_path))

            self.logger.info("Loading repository from: %s", repo_path)
            self._load_repository_unlocked(str(repo_path), False)

            return {
                "status": "success",
                "message": f"ZIP file '{archive_filename}' uploaded and extracted to repos/{repo_name}",
                "repo_info": self.state.repo_info,
                "repo_path": str(repo_path),
            }
        finally:
            try:
                shutil.rmtree(temp_dir)
                self.logger.info("Cleaned up temporary directory: %s", temp_dir)
            except Exception as cleanup_error:
                self.logger.warning(
                    "Failed to clean up temp directory: %s", cleanup_error
                )

    def index_repository(self, force: bool = False):
        with self._state_lock():
            result = self._index_repository_unlocked(force=force)
            self.vector_store.invalidate_scan_cache()
        return result

    def load_and_index(
        self, source: str, is_url: bool | None = None, *, force: bool = False
    ) -> dict[str, Any]:
        """Load and index a repository in one call. Handles locking and
        cache invalidation."""
        with self._state_lock():
            self._load_repository_unlocked(source, is_url)
            self._index_repository_unlocked(force=force)
            self.vector_store.invalidate_scan_cache()
            return {
                "status": "success",
                "message": "Repository loaded and indexed successfully",
                "summary": self.get_repository_summary(),
            }

    def upload_and_index(
        self, file_bytes: bytes, filename: str, *, force: bool = False
    ) -> dict[str, Any]:
        """Upload a ZIP archive, load, and index in one call."""
        with self._state_lock():
            upload_result = self._upload_repository_zip_unlocked(file_bytes, filename)
            if upload_result.get("status") != "success":
                return upload_result
            self._index_repository_unlocked(force=force)
            self.vector_store.invalidate_scan_cache()
            return {
                "status": "success",
                "message": "Repository uploaded and indexed successfully",
                "summary": self.get_repository_summary(),
            }

    def refresh_index_cache(self) -> list[dict[str, Any]]:
        """Invalidate and rescan available indexes."""
        with self._state_lock():
            self.vector_store.invalidate_scan_cache()
            return self.vector_store.scan_available_indexes(use_cache=False)

    # -- Facade methods: narrow API for entry frames --

    def get_status_info(self, *, full_scan: bool = False) -> dict[str, Any]:
        """Return all status fields needed by entry frames."""
        available_repos = self.vector_store.scan_available_indexes(
            use_cache=not full_scan
        )
        loaded_repos = self.list_repositories()
        retrieval_cfg = self.config.get("retrieval", {}) or {}
        return {
            "repo_loaded": self.state.repo_loaded,
            "repo_indexed": self.state.repo_indexed,
            "repo_info": self.state.repo_info,
            "multi_repo_mode": self.state.multi_repo_mode,
            "storage_backend": self.snapshot_store.db_runtime.backend,
            "retrieval_backend": retrieval_cfg.get("retrieval_backend", "local"),
            "graph_expansion_backend": retrieval_cfg.get(
                "graph_expansion_backend", "graph_builder"
            ),
            "available_repositories": available_repos,
            "loaded_repositories": loaded_repos,
        }

    def search_symbols(
        self, name: str, *, symbol_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Search symbols by name with ranking: exact > prefix > contains."""
        query_lower = name.lower()
        exact: list[Any] = []
        prefix: list[Any] = []
        contains: list[Any] = []
        for meta in self.vector_store.metadata:
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
        for meta in self.vector_store.metadata:
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
        """Trace call chain for a symbol using the graph builder.

        Returns a structured dict with target info and caller/callee lists,
        or None if the symbol is not found.
        """
        max_hops = min(max_hops, 5)
        gb = self.graph_builder
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

    def reindex_repository(self, source: str) -> str:
        """Force re-index a repository. Returns a result message."""
        self.apply_env_ignore_patterns()
        resolved_is_url = self._infer_is_url(source)
        name = self.repo_name_from_source(source, resolved_is_url)
        self.logger.info("Force re-indexing '%s' from %s", name, source)

        if resolved_is_url:
            self.load_repository(source, is_url=True)
        else:
            abs_path = os.path.abspath(source)
            if not os.path.isdir(abs_path):
                return f"Error: Local path does not exist: {abs_path}"
            self.load_repository(abs_path, is_url=False)

        self.index_repository(force=True)
        count = self.vector_store.get_count()

        # Reset in-memory state for clean reload
        self.state.repo_indexed = False
        self.state.loaded_repositories.clear()

        return f"Successfully re-indexed '{name}': {count} elements indexed."

    def list_available_repos(self) -> list[dict[str, Any]]:
        """List all indexed repositories with metadata."""
        return self.vector_store.scan_available_indexes(use_cache=False)

    def get_repo_overview(self, repo_name: str) -> dict[str, Any] | None:
        """Get overview for an indexed repository."""
        overviews = self.vector_store.load_repo_overviews(include_embeddings=False)
        return overviews.get(repo_name)

    def clear_cache(self) -> bool:
        """Clear the query cache. Returns True if successful."""
        with self._state_lock():
            return bool(self.cache_manager.clear())

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.cache_manager.get_stats()

    def invalidate_scan_cache(self) -> None:
        """Invalidate the vector store scan cache."""
        self.vector_store.invalidate_scan_cache()

    def load_cached_repos(self, repo_names: list[str] | None = None) -> bool:
        """Load pre-indexed repos from cache into memory."""
        return self._load_multi_repo_cache(repo_names=repo_names)

    def get_session_multi_turn(self, session_id: str) -> bool:
        """Return whether a session is multi-turn."""
        record = self.cache_manager.get_session_index_record(session_id)
        return bool(record.multi_turn) if record is not None else False

    def is_repo_indexed(self, repo_name: str) -> bool:
        """Check whether a persisted index exists for a repo."""
        has_saved_index = getattr(self.vector_store, "has_saved_index", None)
        if callable(has_saved_index):
            return bool(has_saved_index(repo_name))
        persist_dir = self.vector_store.persist_dir
        faiss_path = os.path.join(persist_dir, f"{repo_name}.faiss")
        meta_path = os.path.join(persist_dir, f"{repo_name}_metadata.pkl")
        return os.path.exists(faiss_path) and os.path.exists(meta_path)

    def repo_name_from_source(self, source: str, is_url: bool) -> str:
        """Derive a canonical repo name from a URL or local path."""
        from fastcode.utils.filesystem import get_repo_name_from_url

        if is_url:
            return get_repo_name_from_url(source)
        return os.path.basename(os.path.normpath(source))

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
            name = self.repo_name_from_source(source, resolved_is_url)

            if self.is_repo_indexed(name):
                if not resolved_is_url and allow_incremental:
                    abs_path = os.path.abspath(source)
                    if os.path.isdir(abs_path):
                        try:
                            result = self.run_index_pipeline(
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
                self.load_repository(source, is_url=True)
            else:
                abs_path = os.path.abspath(source)
                if not os.path.isdir(abs_path):
                    self.logger.error("Local path does not exist: %s", abs_path)
                    continue
                self.load_repository(abs_path, is_url=False)

            self.logger.info("Indexing '%s' …", name)
            self.index_repository(force=False)
            self.logger.info("Indexing '%s' complete.", name)
            ready_names.append(name)

        return ready_names

    def ensure_loaded(self, repo_names: list[str]) -> bool:
        """Ensure repos are loaded into memory (vectors + BM25 + graphs)."""
        if not self.state.repo_indexed or set(repo_names) != set(
            self.state.loaded_repositories.keys()
        ):
            self.logger.info("Loading repos into memory: %s", repo_names)
            return self._load_multi_repo_cache(repo_names=repo_names)
        return True

    def _direct_index_enabled(self) -> bool:
        indexing_config = self.config.get("indexing", {})
        if not isinstance(indexing_config, dict):
            return False
        return bool(indexing_config.get("allow_direct_index", False))

    def _index_repository_unlocked(self, force: bool = False):
        if self._direct_index_enabled():
            return self._index_repository_direct_unlocked(force=force)
        force = force or self.eval_config.get("force_reindex", False)
        if not self.state.repo_loaded:
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
            get_loaded_repositories=lambda: self.state.loaded_repositories,
            graph_runtime=self.graph_runtime,
        )

    def _index_repository_direct_unlocked(self, force: bool = False):
        """Direct repository indexing path — delegates to DirectIndexer (use_flow)."""
        indexed, gib, mr, sr = self._direct_indexer.run(
            repo_loaded=self.state.repo_loaded,
            repo_info=self.state.repo_info,
            eval_config=self.eval_config,
            force=force,
        )
        if gib is not None:
            self.global_index_builder = gib
        if mr is not None:
            self.module_resolver = mr
        if sr is not None:
            self.symbol_resolver = sr
        self.state.repo_indexed = indexed

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
                get_loaded_repositories=lambda: self.state.loaded_repositories,
                graph_runtime=self.graph_runtime,
            )

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
        return self.pipeline._apply_semantic_resolvers(
            snapshot=snapshot,
            elements=elements,
            graph_context=graph_context,
            target_paths=target_paths,
            warnings=warnings,
            budget=budget,
        )

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
                return safe_jsonable({str(key): value for key, value in record.items()})

        snapshot = self.snapshot_store.load_snapshot(snapshot_id)
        if not snapshot:
            return None
        for symbol in snapshot.symbols:
            if symbol.symbol_id == resolved:
                return self._ir_symbol_payload(symbol)
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

    def get_code_status_pack(
        self,
        snapshot_id: str,
        *,
        include_graph_facts: bool = True,
    ) -> dict[str, Any]:
        snapshot_record = self.snapshot_store.get_snapshot_record(snapshot_id)
        if snapshot_record is None:
            raise RuntimeError(f"snapshot not found: {snapshot_id}")
        snapshot = self.snapshot_store.load_snapshot(snapshot_id)
        if snapshot is None:
            raise RuntimeError(f"snapshot payload not found: {snapshot_id}")
        ir_graphs = (
            self.snapshot_store.load_ir_graphs(snapshot_id)
            if include_graph_facts
            else None
        )
        return build_code_status_pack(
            snapshot,
            artifact_key=snapshot_record.artifact_key,
            manifest=self.get_snapshot_manifest(snapshot_id),
            ir_graphs=ir_graphs,
            include_graph_facts=include_graph_facts,
        )

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
            graph_context=active_graph_builder,
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
        self, elements: Sequence[Hit]
    ) -> list[dict[str, Any]]:
        """Extract source information from retrieved elements"""
        sources = _snapshot.extract_sources_from_elements(elements)
        return [self._source_citation_payload(source) for source in sources]

    @staticmethod
    def _source_citation_payload(source: SourceCitation) -> dict[str, Any]:
        return {
            "repository": source.repository,
            "file": source.file,
            "name": source.name,
            "element_type": source.element_type,
            "lines": source.lines,
            "score": source.score,
        }

    def get_repository_summary(self) -> str:
        """Get summary of the loaded repository"""
        if not self.state.repo_info:
            return "No repository loaded"

        summary_parts = [
            f"Repository: {self.state.repo_info.get('name', 'Unknown')}",
            f"Files: {self.state.repo_info.get('file_count', 0)}",
            f"Size: {self.state.repo_info.get('total_size_mb', 0):.2f} MB",
        ]

        if self.state.repo_indexed:
            summary_parts.append(f"Indexed elements: {self.vector_store.get_count()}")

        return "\n".join(summary_parts)

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
                "allow_direct_index": False,  # [INTERNAL]
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
                "retrieval_backend": "pg_hybrid",  # [ESSENTIAL] "pg_hybrid" or "local"
                "graph_expansion_backend": "ir",  # [ESSENTIAL] "ir" or "graph_builder"
                "allow_graph_builder_fallback": True,  # [INTERNAL]
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
        if not self._direct_index_enabled():
            return self._load_multiple_repositories_pipeline_unlocked(sources)
        return self._load_multiple_repositories_direct_unlocked(sources)

    def _load_multiple_repositories_pipeline_unlocked(
        self, sources: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.logger.info(f"Loading {len(sources)} repositories")
        self.state.multi_repo_mode = True
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
                    get_loaded_repositories=lambda: self.state.loaded_repositories,
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

        self.state.repo_indexed = len(successfully_indexed) > 0
        self.state.repo_loaded = len(successfully_indexed) > 0
        return {
            "status": "succeeded" if successfully_indexed else "failed",
            "repositories": successfully_indexed,
            "results": results,
            "errors": errors,
        }

    def _load_multiple_repositories_direct_unlocked(
        self, sources: list[dict[str, Any]]
    ):
        """Direct multi-repo indexing — delegates to MultiRepoDirectIndexer (use_flow)."""
        self.state.multi_repo_mode = True
        result = self._multi_repo_direct_indexer.run(
            sources,
            loaded_repositories=self.state.loaded_repositories,
        )
        self.state.repo_indexed = result["has_success"]
        self.state.repo_loaded = result["has_success"]

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
            repo_info = self.state.loaded_repositories.get(repo_name, {})
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
            repo_info = self.state.loaded_repositories.get(repo_name, {})
            stats["repositories"].append(
                {
                    "name": repo_name,
                    "elements": repo_counts.get(repo_name, 0),
                    "files": repo_info.get("file_count", 0),
                    "size_mb": repo_info.get("total_size_mb", 0),
                }
            )

        return stats

    @staticmethod
    def _diagnostic_mapping(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        return {str(key): item for key, item in value.items()}

    @staticmethod
    def _diagnostic_list(value: Any) -> list[Any]:
        if not isinstance(value, (list, tuple, set, frozenset)):
            return []
        return list(value)

    @staticmethod
    def _diagnostic_key_is_sensitive(key: str) -> bool:
        normalized = key.lower().replace("-", "_")
        return any(marker in normalized for marker in _DIAGNOSTIC_SECRET_KEY_MARKERS)

    @staticmethod
    def _redact_diagnostic_string(value: str) -> str:
        redacted = value
        try:
            parts = urlsplit(value)
        except ValueError:
            parts = None
        if parts is not None and parts.scheme and parts.netloc:
            host = parts.hostname or ""
            if host:
                netloc = host
                try:
                    port = parts.port
                except ValueError:
                    port = None
                if port is not None:
                    netloc = f"{netloc}:{port}"
                if parts.username or parts.password:
                    netloc = f"{_DIAGNOSTIC_REDACTION}@{netloc}"
                redacted = urlunsplit(
                    (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
                )
        redacted = _DIAGNOSTIC_URL_USERINFO_RE.sub(
            rf"\1{_DIAGNOSTIC_REDACTION}@", redacted
        )

        def _redact_auth_header(match: re.Match[str]) -> str:
            scheme = f"{match.group(3)} " if match.group(3) else ""
            return f"{match.group(1)}{match.group(2)}{scheme}{_DIAGNOSTIC_REDACTION}"

        redacted = _DIAGNOSTIC_AUTH_HEADER_RE.sub(_redact_auth_header, redacted)
        redacted = _DIAGNOSTIC_BEARER_RE.sub(
            f"Bearer {_DIAGNOSTIC_REDACTION}", redacted
        )
        return _DIAGNOSTIC_SECRET_ASSIGNMENT_RE.sub(
            rf"\1\2{_DIAGNOSTIC_REDACTION}", redacted
        )

    @classmethod
    def _redact_diagnostic_value(cls, key: str, value: Any) -> Any:
        if cls._diagnostic_key_is_sensitive(key):
            return bool(value) if key.endswith("_configured") else _DIAGNOSTIC_REDACTION
        if isinstance(value, str):
            return cls._redact_diagnostic_string(value)
        if isinstance(value, Mapping):
            return {
                str(item_key): cls._redact_diagnostic_value(str(item_key), item_value)
                for item_key, item_value in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [cls._redact_diagnostic_value(key, item) for item in value]
        return value

    @staticmethod
    def _configured(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return bool(value)

    @classmethod
    def _diagnostic_json_object(cls, raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {}
        return cls._diagnostic_mapping(parsed)

    @staticmethod
    def _diagnostic_json_string_list(raw: str | None) -> list[str]:
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item) for item in parsed if item is not None]

    @classmethod
    def _diagnostic_config_summary(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        storage = cls._diagnostic_mapping(config.get("storage"))
        repository = cls._diagnostic_mapping(config.get("repository"))
        embedding = cls._diagnostic_mapping(config.get("embedding"))
        indexing = cls._diagnostic_mapping(config.get("indexing"))
        vector_store = cls._diagnostic_mapping(config.get("vector_store"))
        retrieval = cls._diagnostic_mapping(config.get("retrieval"))
        generation = cls._diagnostic_mapping(config.get("generation"))
        evaluation = cls._diagnostic_mapping(config.get("evaluation"))
        cache = cls._diagnostic_mapping(config.get("cache"))
        terminus = cls._diagnostic_mapping(config.get("terminus"))
        projection = cls._diagnostic_mapping(config.get("projection"))
        supported_extensions = cls._diagnostic_list(
            repository.get("supported_extensions")
        )
        ignore_patterns = cls._diagnostic_list(repository.get("ignore_patterns"))
        return {
            "storage": {
                "backend": str(storage.get("backend") or "sqlite"),
                "postgres_dsn_configured": cls._configured(storage.get("postgres_dsn")),
                "pool_min": storage.get("pool_min"),
                "pool_max": storage.get("pool_max"),
            },
            "repository": {
                "repo_root": cls._redact_diagnostic_value(
                    "repo_root", config.get("repo_root")
                ),
                "max_file_size_mb": repository.get("max_file_size_mb"),
                "exclude_site_packages": bool(
                    repository.get("exclude_site_packages", False)
                ),
                "ignore_pattern_count": len(ignore_patterns),
                "supported_extension_count": len(supported_extensions),
            },
            "embedding": {
                "provider": embedding.get("provider"),
                "model": embedding.get("model"),
                "device": embedding.get("device"),
                "batch_size": embedding.get("batch_size"),
                "ollama_url_configured": cls._configured(embedding.get("ollama_url")),
            },
            "indexing": {
                "levels": [
                    str(item) for item in cls._diagnostic_list(indexing.get("levels"))
                ],
                "allow_direct_index": bool(indexing.get("allow_direct_index", False)),
            },
            "vector_store": {
                "persist_directory": cls._redact_diagnostic_value(
                    "persist_directory", vector_store.get("persist_directory")
                ),
                "distance_metric": vector_store.get("distance_metric"),
                "shard_storage": vector_store.get("shard_storage"),
            },
            "retrieval": {
                "retrieval_backend": retrieval.get("retrieval_backend"),
                "graph_expansion_backend": retrieval.get("graph_expansion_backend"),
                "max_results": retrieval.get("max_results"),
                "semantic_weight": retrieval.get("semantic_weight"),
                "keyword_weight": retrieval.get("keyword_weight"),
                "graph_weight": retrieval.get("graph_weight"),
            },
            "generation": {
                "provider": generation.get("provider"),
                "model": generation.get("model"),
                "temperature": generation.get("temperature"),
                "max_tokens": generation.get("max_tokens"),
            },
            "evaluation": {
                "enabled": bool(evaluation.get("enabled", False)),
                "in_memory_index": bool(evaluation.get("in_memory_index", False)),
                "disable_cache": bool(evaluation.get("disable_cache", False)),
                "disable_persistence": bool(
                    evaluation.get("disable_persistence", False)
                ),
                "force_reindex": bool(evaluation.get("force_reindex", False)),
            },
            "cache": {
                "enabled": bool(cache.get("enabled", True)),
                "backend": cache.get("backend"),
                "cache_directory": cls._redact_diagnostic_value(
                    "cache_directory", cache.get("cache_directory")
                ),
                "cache_queries": bool(cache.get("cache_queries", False)),
            },
            "terminus": {
                "endpoint_configured": cls._configured(terminus.get("endpoint")),
                "api_key_configured": cls._configured(terminus.get("api_key")),
            },
            "projection": {
                "postgres_dsn_configured": cls._configured(
                    projection.get("postgres_dsn")
                ),
                "enable_leiden": bool(projection.get("enable_leiden", False)),
                "llm_enabled": bool(projection.get("llm_enabled", False)),
            },
        }

    def _diagnostic_storage_summary(self) -> dict[str, Any]:
        snapshot_store = getattr(self, "snapshot_store", None)
        db_runtime = getattr(snapshot_store, "db_runtime", None)
        vector_store = getattr(self, "vector_store", None)
        projection_store = getattr(self, "projection_store", None)
        cache_manager = getattr(self, "cache_manager", None)
        config = self._diagnostic_mapping(getattr(self, "config", {}))
        cache_config = self._diagnostic_mapping(config.get("cache"))
        return {
            "backend": str(getattr(db_runtime, "backend", "unknown")),
            "sqlite_path": self._redact_diagnostic_value(
                "sqlite_path", getattr(db_runtime, "sqlite_path", None)
            ),
            "postgres_dsn_configured": self._configured(
                getattr(db_runtime, "postgres_dsn", None)
            ),
            "pool_min": getattr(db_runtime, "pool_min", None),
            "pool_max": getattr(db_runtime, "pool_max", None),
            "pool_configured": getattr(db_runtime, "pool", None) is not None,
            "vector_persist_dir": self._redact_diagnostic_value(
                "vector_persist_dir", getattr(vector_store, "persist_dir", None)
            ),
            "vector_in_memory": bool(getattr(vector_store, "in_memory", False)),
            "cache_backend": cache_config.get("backend"),
            "cache_enabled": bool(cache_config.get("enabled", True)),
            "cache_manager_kind": type(cache_manager).__name__
            if cache_manager is not None
            else None,
            "projection_store_enabled": bool(
                getattr(projection_store, "enabled", False)
            ),
        }

    @staticmethod
    def _dependency_available(import_name: str) -> bool:
        try:
            return importlib_util.find_spec(import_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    @classmethod
    def _diagnostic_dependency_summary(cls) -> dict[str, Any]:
        python_dependencies = [
            {
                "group": group,
                "name": name,
                "import_name": import_name,
                "available": cls._dependency_available(import_name),
            }
            for group, name, import_name in _DIAGNOSTIC_PYTHON_DEPENDENCIES
        ]
        external_tools = []
        for name, executable in _DIAGNOSTIC_EXTERNAL_TOOLS:
            resolved = shutil.which(executable)
            external_tools.append(
                {
                    "name": name,
                    "executable": executable,
                    "available": resolved is not None,
                    "path": cls._redact_diagnostic_value("path", resolved),
                }
            )
        return {
            "python": python_dependencies,
            "external_tools": external_tools,
        }

    def _latest_index_run_diagnostic_payload(self) -> dict[str, Any] | None:
        latest_run_record: IndexRunRecord | None = None
        index_run_store = getattr(self, "index_run_store", None)
        get_latest_run_record = getattr(index_run_store, "get_latest_run_record", None)
        if callable(get_latest_run_record):
            latest_run_record = cast(IndexRunRecord | None, get_latest_run_record())
        if latest_run_record is None:
            return None

        payload: dict[str, Any] = {
            "run_id": latest_run_record.run_id,
            "repo_name": latest_run_record.repo_name,
            "snapshot_id": latest_run_record.snapshot_id,
            "branch": latest_run_record.branch,
            "commit_id": latest_run_record.commit_id,
            "status": latest_run_record.status,
            "error_message": self._redact_diagnostic_value(
                "error_message", latest_run_record.error_message
            ),
            "warnings": self._redact_diagnostic_value(
                "warnings",
                self._diagnostic_json_string_list(latest_run_record.warnings_json),
            ),
            "created_at": latest_run_record.created_at,
            "started_at": latest_run_record.started_at,
            "completed_at": latest_run_record.completed_at,
        }
        snapshot_record = None
        snapshot_store = getattr(self, "snapshot_store", None)
        get_snapshot_record = getattr(snapshot_store, "get_snapshot_record", None)
        if callable(get_snapshot_record):
            snapshot_record = get_snapshot_record(latest_run_record.snapshot_id)
        if snapshot_record is None:
            return payload

        metadata = self._diagnostic_json_object(
            getattr(snapshot_record, "metadata_json", None)
        )
        payload["snapshot"] = {
            "snapshot_id": getattr(snapshot_record, "snapshot_id", None),
            "repo_name": getattr(snapshot_record, "repo_name", None),
            "branch": getattr(snapshot_record, "branch", None),
            "commit_id": getattr(snapshot_record, "commit_id", None),
            "tree_id": getattr(snapshot_record, "tree_id", None),
            "artifact_key": getattr(snapshot_record, "artifact_key", None),
            "created_at": getattr(snapshot_record, "created_at", None),
            "warnings": self._redact_diagnostic_value(
                "warnings",
                [
                    str(item)
                    for item in self._diagnostic_list(metadata.get("warnings"))
                    if item is not None
                ],
            ),
            "pipeline_layers": self._redact_diagnostic_value(
                "pipeline_layers",
                [
                    self._diagnostic_mapping(item)
                    for item in self._diagnostic_list(metadata.get("pipeline_layers"))
                    if isinstance(item, Mapping)
                ],
            ),
            "pipeline_metrics": self._redact_diagnostic_value(
                "pipeline_metrics",
                self._diagnostic_mapping(metadata.get("pipeline_metrics")),
            ),
        }
        return payload

    def build_diagnostic_bundle(self) -> dict[str, Any]:
        """Build a support-safe runtime diagnostic bundle."""
        config_payload = self._diagnostic_mapping(getattr(self, "config", {}))
        repo_info = self._diagnostic_mapping(getattr(self.state, "repo_info", {}))
        return {
            "schema_version": "fastcode.diagnostic_bundle.v1",
            "generated_at": utc_now(),
            "runtime": {
                "repo_loaded": bool(getattr(self.state, "repo_loaded", False)),
                "repo_indexed": bool(getattr(self.state, "repo_indexed", False)),
                "multi_repo_mode": bool(getattr(self.state, "multi_repo_mode", False)),
                "repo_info": {
                    "name": repo_info.get("name"),
                    "url": self._redact_diagnostic_value("url", repo_info.get("url")),
                    "file_count": repo_info.get("file_count"),
                    "total_size_mb": repo_info.get("total_size_mb"),
                    "branch": repo_info.get("branch"),
                    "commit": repo_info.get("commit"),
                },
                "loaded_repository_count": len(
                    getattr(self.state, "loaded_repositories", {}) or {}
                ),
                "loader_repo_path": self._redact_diagnostic_value(
                    "loader_repo_path",
                    getattr(
                        getattr(self, "loader", None),
                        "repo_path",
                        None,
                    ),
                ),
            },
            "config_summary": self._diagnostic_config_summary(config_payload),
            "storage": self._diagnostic_storage_summary(),
            "dependencies": self._diagnostic_dependency_summary(),
            "latest_index_run": self._latest_index_run_diagnostic_payload(),
        }

    def _load_multi_repo_cache(self, repo_names: list[str] | None = None) -> bool:
        with self._state_lock():
            return self._load_multi_repo_cache_unlocked(repo_names=repo_names)

    def _load_multi_repo_cache_unlocked(
        self, repo_names: list[str] | None = None
    ) -> bool:
        """Load multi-repo cache — delegates to rehydration (use_flow)."""
        return _load_multi_repo_cache_impl(
            repo_names=repo_names,
            vector_store=self.vector_store,
            embedder=self.embedder,
            retriever=self.retriever,
            graph_builder=self.graph_builder,
            graph_artifact_store=self.graph_artifact_store,
            loaded_repositories=self.state.loaded_repositories,
        )

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

    def get_session_history(self, session_id: str) -> list[DialogueTurnRecord]:
        """
        Get dialogue history for a session

        Args:
            session_id: Session ID

        Returns:
            List of dialogue turns
        """
        return self.cache_manager.get_dialogue_history_records(session_id)

    @staticmethod
    def _parse_working_memory_record_payload(
        record: Any,
    ) -> WorkingMemoryArtifact:
        payload = json.loads(str(record.payload_json or "{}"))
        if not isinstance(payload, dict):
            raise RuntimeError("working memory payload is invalid")
        return working_memory_from_payload(payload)

    @staticmethod
    def _parse_handoff_record_payload(record: Any) -> HandoffArtifact:
        payload = json.loads(str(record.payload_json or "{}"))
        if not isinstance(payload, dict):
            raise RuntimeError("handoff artifact payload is invalid")
        return handoff_from_payload(payload)

    @staticmethod
    def _parse_turn_journal_record_payload(record: Any) -> TurnJournal:
        payload = json.loads(str(record.payload_json or "{}"))
        if not isinstance(payload, dict):
            raise RuntimeError("turn journal payload is invalid")
        return turn_journal_from_payload(payload)

    @staticmethod
    def _parse_context_bundle_record_payload(record: Any) -> ContextBundle:
        payload = json.loads(str(record.payload_json or "{}"))
        if not isinstance(payload, dict):
            raise RuntimeError("context bundle payload is invalid")
        return context_bundle_from_payload(payload)

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
            response["bundle"] = context_bundle_payload(bundle)
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
            response["artifact"] = working_memory_payload(artifact)
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
            handoff_payload(artifact),
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
        return handoff_payload(artifact)

    def get_handoff_artifact(self, artifact_id: str) -> dict[str, Any]:
        record = self.cache_manager.get_handoff_artifact_record(artifact_id)
        if record is None:
            raise RuntimeError(f"handoff artifact not found: {artifact_id}")
        return handoff_payload(self._parse_handoff_record_payload(record))

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
            activation_payload(activation),
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
        return activation_payload(activation)

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
        enriched_sessions: list[dict[str, Any]] = []
        for session in self.cache_manager.list_session_records():
            session_id = str(session.session_id)
            title = "Unknown Session"
            if session_id:
                first_turn = self.cache_manager.get_dialogue_turn_record(session_id, 1)
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
