"""Direct single-repository indexing path.

Moved from main/fastcode.py (assembly_root) to use_flow (app/indexing)
because this is indexing workflow orchestration, not composition root wiring.

This path is enabled only when indexing.allow_direct_index=true.
The default snapshot-oriented path lives in service.py IndexPipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastcode.app.indexing.embedder import CodeEmbedder
from fastcode.app.indexing.loader import RepositoryLoader
from fastcode.app.query.selection.retriever import HybridRetriever
from fastcode.app.store.artifacts.graph import GraphArtifactStore
from fastcode.app.store.vectors.vector import VectorStore
from fastcode.app.store.vectors.vector_math import as_float32_matrix
from fastcode.graph.build import CodeGraphBuilder
from fastcode.ir.element import CodeElementMeta, serialize_code_element
from fastcode.scip.global_builder import GlobalIndexBuilder
from fastcode.scip.module_resolver import ModuleResolver
from fastcode.scip.symbol_resolver import SymbolResolver

from .indexer import CodeIndexer
from .manifest import build_file_manifest, save_file_manifest

logger = logging.getLogger(__name__)


class DirectIndexer:
    """Direct (non-snapshot) single-repository indexing workflow.

    Receives all dependencies via constructor injection. Owns no mutable
    state beyond what is injected; reports results via return values and
    callbacks.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        loader: RepositoryLoader,
        indexer: CodeIndexer,
        embedder: CodeEmbedder,
        vector_store: VectorStore,
        graph_builder: CodeGraphBuilder,
        retriever: HybridRetriever,
        graph_artifact_store: GraphArtifactStore,
        should_use_cache_fn: Callable[[], bool],
        try_load_from_cache_fn: Callable[[], bool],
        should_persist_fn: Callable[[], bool],
        save_to_cache_fn: Callable[[str], None],
        set_repo_root_fn: Callable[[str], None],
        log_statistics_fn: Callable[[], None],
    ) -> None:
        self.config = config
        self.loader = loader
        self.indexer = indexer
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_builder = graph_builder
        self.retriever = retriever
        self.graph_artifact_store = graph_artifact_store
        self._should_use_cache = should_use_cache_fn
        self._try_load_from_cache = try_load_from_cache_fn
        self._should_persist = should_persist_fn
        self._save_to_cache = save_to_cache_fn
        self._set_repo_root = set_repo_root_fn
        self._log_statistics = log_statistics_fn

    def run(
        self,
        *,
        repo_loaded: bool,
        repo_info: dict[str, Any],
        eval_config: dict[str, Any],
        force: bool = False,
    ) -> tuple[
        bool, GlobalIndexBuilder | None, ModuleResolver | None, SymbolResolver | None
    ]:
        """Run direct indexing for a single repository.

        Returns:
            (indexed, global_index_builder, module_resolver, symbol_resolver)
            indexed is True if indexing succeeded.
            The resolver objects are returned so the caller can cache them.
        """
        force = force or eval_config.get("force_reindex", False)

        if not repo_loaded:
            msg = "No repository loaded. Call load_repository() first."
            raise RuntimeError(msg)

        logger.info("Indexing repository")

        repo_name = repo_info.get("name", "default")

        # Check cache
        if not force and self._should_use_cache():
            loaded = self._try_load_from_cache()
            if loaded:
                return True, None, None, None

        global_index_builder: GlobalIndexBuilder | None = None
        module_resolver: ModuleResolver | None = None
        symbol_resolver: SymbolResolver | None = None

        try:
            repo_url = repo_info.get("url")

            elements = self.indexer.extract_elements(
                repo_name=repo_name, repo_url=repo_url
            )

            if self.vector_store.dimension is None:
                self.vector_store.initialize(self.embedder.embedding_dim)

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

            try:
                logger.info("Initializing resolvers for precise graph building...")

                repo_root = self.config.get("repo_root")
                if not repo_root and self.loader.repo_path:
                    repo_root = self.loader.repo_path
                    self._set_repo_root(repo_root)

                global_index_builder = GlobalIndexBuilder(self.config)

                logger.info(
                    "Building global index maps (Repo Root: %s)...",
                    repo_root,
                )
                global_index_builder.build_maps(elements, repo_root or "")
                logger.info(
                    "  - Mapped %d files",
                    len(global_index_builder.file_map),
                )
                logger.info(
                    "  - Mapped %d modules",
                    len(global_index_builder.module_map),
                )

                module_resolver = ModuleResolver(global_index_builder)
                symbol_resolver = SymbolResolver(global_index_builder, module_resolver)

                logger.info("Resolvers initialized successfully")

            except Exception as e:
                logger.warning("Resolver initialization failed: %s", e)
                logger.warning("Using fallback graph building (less accurate)")
                import traceback

                logger.error(traceback.format_exc())
                module_resolver = None
                symbol_resolver = None

            self.graph_builder.build_graphs(elements, module_resolver, symbol_resolver)

            self.retriever.index_for_bm25(elements)
            self.retriever.build_repo_overview_bm25()

            if self._should_persist():
                self._save_to_cache(repo_name)
                self.retriever.save_bm25(repo_name)
                self.graph_artifact_store.save(self.graph_builder, repo_name)
                manifest = build_file_manifest(
                    elements,
                    self.loader.repo_path or ".",
                    repo_name=repo_name,
                )
                save_file_manifest(
                    manifest,
                    repo_name,
                    self.vector_store.persist_dir,
                )
            else:
                logger.info("Skipping on-disk persistence (ephemeral/evaluation mode)")

            logger.info("Repository indexing complete for %s", repo_name)
            self._log_statistics()

            return True, global_index_builder, module_resolver, symbol_resolver

        except Exception as e:
            logger.error("Failed to index repository: %s", e)
            import traceback

            logger.error(traceback.format_exc())
            raise
