"""Direct multi-repository indexing path.

Moved from main/fastcode.py (assembly_root) to use_flow (app/indexing)
because multi-repo indexing orchestration is workflow logic, not composition
root wiring.

This path is enabled only when indexing.allow_direct_index=true.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastcode.app.indexing.embedder import CodeEmbedder
from fastcode.app.indexing.extractors.parser import CodeParser
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

logger = logging.getLogger(__name__)


class MultiRepoDirectIndexer:
    """Direct (non-snapshot) multi-repository indexing workflow.

    Receives all dependencies via constructor injection. Each repo gets its
    own temp vector store, retriever, and graph builder to avoid cross-contamination.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        loader: RepositoryLoader,
        parser: CodeParser,
        embedder: CodeEmbedder,
        vector_store: VectorStore,
        graph_builder: CodeGraphBuilder,
        graph_artifact_store: GraphArtifactStore,
        set_repo_root_fn: Callable[[str], None],
        infer_is_url_fn: Callable[[str], bool],
    ) -> None:
        self.config = config
        self.loader = loader
        self.parser = parser
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_builder = graph_builder
        self.graph_artifact_store = graph_artifact_store
        self._set_repo_root = set_repo_root_fn
        self._infer_is_url = infer_is_url_fn

    def run(
        self,
        sources: list[dict[str, Any]],
        *,
        loaded_repositories: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Index multiple repositories directly.

        Args:
            sources: List of dicts with 'source', 'is_url', optionally 'is_zip'.
            loaded_repositories: Mutable dict to register loaded repos into.

        Returns:
            Dict with 'successfully_indexed' list and per-repo results.
        """
        logger.info("Loading %d repositories", len(sources))

        successfully_indexed: list[str] = []
        errors: list[dict[str, str]] = []

        for i, source_info in enumerate(sources):
            source: str = str(source_info.get("source") or "")
            is_url: bool | None = source_info.get("is_url")
            is_zip: bool = bool(source_info.get("is_zip", False))

            try:
                logger.info(
                    "[%d/%d] Loading repository: %s",
                    i + 1,
                    len(sources),
                    source,
                )

                resolved_is_url = is_url
                if not is_zip and resolved_is_url is None:
                    resolved_is_url = self._infer_is_url(source)
                    source_type = "URL" if resolved_is_url else "local path"
                    logger.info(
                        "[%d/%d] Auto-detected source type as %s",
                        i + 1,
                        len(sources),
                        source_type,
                    )

                if is_zip:
                    self.loader.load_from_zip(source)
                elif resolved_is_url:
                    self.loader.load_from_url(source)
                else:
                    self.loader.load_from_path(source)

                repo_info = self.loader.get_repository_info()
                repo_name: str = str(repo_info.get("name") or "")
                repo_url: str = str(repo_info.get("url") or source)

                if self.loader.repo_path:
                    self._set_repo_root(self.loader.repo_path)

                loaded_repositories[repo_name] = repo_info

                logger.info("Indexing repository: %s", repo_name)

                temp_vector_store = VectorStore(self.config)
                temp_vector_store.initialize(self.embedder.embedding_dim)

                temp_indexer = CodeIndexer(
                    self.config,
                    self.loader,
                    self.parser,
                    self.embedder,
                    temp_vector_store,
                )

                elements = temp_indexer.extract_elements(
                    repo_name=repo_name, repo_url=repo_url
                )

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
                    temp_vector_store.save(repo_name)

                    temp_retriever = HybridRetriever(
                        self.config,
                        temp_vector_store,
                        self.embedder,
                        self.graph_builder,
                        repo_root=self.loader.repo_path,
                    )
                    temp_retriever.index_for_bm25(elements)
                    temp_retriever.save_bm25(repo_name)
                    logger.info("Saved BM25 index for %s", repo_name)

                    temp_retriever.build_repo_overview_bm25()
                    logger.info("Built repo overview BM25 index")

                    temp_graph_builder = CodeGraphBuilder(self.config)

                    repo_root = self.loader.repo_path
                    temp_module_resolver = None
                    temp_symbol_resolver = None

                    try:
                        logger.info("Initializing resolvers for %s...", repo_name)
                        temp_global_index = GlobalIndexBuilder(self.config)
                        temp_global_index.build_maps(elements, repo_root or ".")
                        temp_module_resolver = ModuleResolver(temp_global_index)
                        temp_symbol_resolver = SymbolResolver(
                            temp_global_index, temp_module_resolver
                        )
                        logger.info("Resolvers initialized for %s", repo_name)
                    except Exception as e:
                        logger.warning(
                            "Failed to initialize resolvers for %s: %s",
                            repo_name,
                            e,
                        )
                        temp_module_resolver = None
                        temp_symbol_resolver = None

                    temp_graph_builder.build_graphs(
                        elements, temp_module_resolver, temp_symbol_resolver
                    )
                    self.graph_artifact_store.save(temp_graph_builder, repo_name)
                    logger.info("Saved graph data for %s", repo_name)

                    successfully_indexed.append(repo_name)

                    logger.info(
                        "Successfully indexed and saved %s: %d elements",
                        repo_name,
                        len(elements),
                    )
                else:
                    logger.warning("No vectors generated for %s", repo_name)

            except Exception as e:
                logger.error("Failed to load repository %s: %s", source, e)
                import traceback

                logger.error(traceback.format_exc())
                errors.append({"source": source, "error": str(e)})
                continue

        if successfully_indexed:
            logger.info(
                "Successfully indexed %d repositories:",
                len(successfully_indexed),
            )
            for repo_name in successfully_indexed:
                logger.info("  - %s", repo_name)

            logger.info("Merging repositories into main vector store for statistics...")
            if self.vector_store.dimension is None:
                self.vector_store.initialize(self.embedder.embedding_dim)

            for repo_name in successfully_indexed:
                if self.vector_store.merge_from_index(repo_name):
                    logger.info("Merged %s into main store", repo_name)
                else:
                    logger.warning("Failed to merge %s", repo_name)
        else:
            logger.error("No repositories were successfully indexed")

        logger.info("Indexing complete. Each repository saved separately.")

        return {
            "successfully_indexed": successfully_indexed,
            "errors": errors,
            "has_success": len(successfully_indexed) > 0,
        }
