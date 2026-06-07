"""Cache rehydration: loading, saving, and reconstructing indexed artifacts.

Moved from main/fastcode.py (assembly_root) to use_flow (app/store)
because cache/persistence orchestration is store workflow logic,
not composition root wiring.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, cast

from fastcode.app.indexing.embedder import CodeEmbedder
from fastcode.app.query.selection.retriever import HybridRetriever
from fastcode.app.store.artifacts.graph import GraphArtifactStore
from fastcode.app.store.vectors.vector import VectorStore
from fastcode.graph.build import CodeGraphBuilder
from fastcode.ir.element import CodeElement

logger = logging.getLogger(__name__)


def reconstruct_elements_from_metadata(
    vector_store: VectorStore,
) -> list[CodeElement]:
    """Reconstruct CodeElement objects from vector store metadata.

    Excludes repository_overview elements (they're in separate storage).
    """
    elements: list[CodeElement] = []
    for meta in vector_store.metadata:
        try:
            if meta.get("type") == "repository_overview":
                continue

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
            logger.warning("Failed to reconstruct element: %s", e)
            continue

    logger.info(
        "Reconstructed %d elements from metadata (excluding repository_overview)",
        len(elements),
    )
    return elements


def try_load_from_cache(
    *,
    cache_name: str,
    vector_store: VectorStore,
    retriever: HybridRetriever,
    graph_builder: CodeGraphBuilder,
    graph_artifact_store: GraphArtifactStore,
    log_statistics_fn: Callable[[], None],
) -> bool:
    """Try to load indexed data from cache for a single repository.

    Returns True if cache was loaded successfully.
    """
    try:
        if vector_store.load(cache_name):
            logger.info("Loaded vector store from cache for %s", cache_name)

            bm25_loaded = retriever.load_bm25(cache_name)
            if not bm25_loaded:
                logger.warning("Failed to load BM25 index, will need to rebuild")

            retriever.build_repo_overview_bm25()

            graph_loaded = graph_artifact_store.load(graph_builder, cache_name)
            if not graph_loaded:
                logger.warning("Failed to load graph data, will need to rebuild")

            if not bm25_loaded or not graph_loaded:
                logger.info("Reconstructing missing components from metadata...")
                elements = reconstruct_elements_from_metadata(vector_store)

                if elements:
                    if not bm25_loaded:
                        retriever.index_for_bm25(elements)
                        logger.info(
                            "Rebuilt BM25 index with %d elements", len(elements)
                        )

                    if not graph_loaded:
                        graph_builder.build_graphs(elements)
                        logger.info("Rebuilt code graph (fallback mode)")
                else:
                    logger.warning("No elements reconstructed from metadata")

            logger.info("Cache loaded successfully")
            log_statistics_fn()
            return True

        return False

    except Exception as e:
        logger.warning("Failed to load from cache: %s", e)
        return False


def save_to_cache(
    *,
    cache_name: str,
    vector_store: VectorStore,
) -> None:
    """Save indexed data to cache."""
    try:
        vector_store.save(cache_name)
        logger.info("Saved index to cache: %s", cache_name)
    except Exception as e:
        logger.warning("Failed to save to cache: %s", e)


def load_multi_repo_cache(
    *,
    repo_names: list[str] | None,
    vector_store: VectorStore,
    embedder: CodeEmbedder,
    retriever: HybridRetriever,
    graph_builder: CodeGraphBuilder,
    graph_artifact_store: GraphArtifactStore,
    loaded_repositories: dict[str, dict[str, Any]],
) -> bool:
    """Load multi-repository index from cache by merging individual repo indices.

    Returns True if successful.
    """
    try:
        available_repos: list[str] = []
        scan_available_indexes = getattr(vector_store, "scan_available_indexes", None)
        if callable(scan_available_indexes):
            for repo in cast(list[dict[str, Any]], scan_available_indexes(False)):
                repo_name = str(repo.get("name") or repo.get("repo_name") or "")
                if repo_name:
                    available_repos.append(repo_name)
        else:
            persist_dir = vector_store.persist_dir
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
            logger.error("No repository indexes found")
            return False

        if repo_names:
            repos_to_load = [r for r in available_repos if r in repo_names]
            if not repos_to_load:
                logger.error("None of the requested repositories found: %s", repo_names)
                return False
        else:
            repos_to_load = available_repos

        logger.info(
            "Found %d repository indexes: %s",
            len(repos_to_load),
            ", ".join(repos_to_load),
        )

        vector_store.initialize(embedder.embedding_dim)

        for repo_name in repos_to_load:
            logger.info("Loading index for %s...", repo_name)
            try:
                if vector_store.merge_from_index(repo_name):
                    logger.info("Successfully merged %s", repo_name)
                else:
                    logger.warning("Failed to merge index for %s", repo_name)
            except Exception as e:
                logger.error("Error loading %s: %s", repo_name, e)
                continue

        if vector_store.get_count() == 0:
            logger.error("Failed to load any repository indexes")
            return False

        for repo_name in repos_to_load:
            if repo_name not in loaded_repositories:
                loaded_repositories[repo_name] = {
                    "name": repo_name,
                    "file_count": 0,
                    "total_size_mb": 0,
                }

        logger.info("Loading BM25 and graph data...")

        graphs_loaded = False

        for repo_name in repos_to_load:
            if not graphs_loaded:
                if graph_artifact_store.load(graph_builder, repo_name):
                    graphs_loaded = True
                    logger.info("Loaded graph data from %s as base", repo_name)
            elif graph_artifact_store.merge(graph_builder, repo_name):
                logger.info("Merged graph data from %s", repo_name)
            else:
                logger.warning("Failed to merge graph data from %s", repo_name)

        load_bm25_sources = getattr(retriever, "load_bm25_sources", None)
        load_bm25_legacy_sources = getattr(retriever, "load_bm25_legacy_sources", None)
        if callable(load_bm25_sources) and load_bm25_sources(
            repos_to_load, filtered=False
        ):
            logger.info(
                "Loaded shard-native multi-repo BM25 for %d repositories",
                len(repos_to_load),
            )
        elif callable(load_bm25_legacy_sources) and load_bm25_legacy_sources(
            repos_to_load, filtered=False
        ):
            logger.info(
                "Loaded legacy multi-repo BM25 through shard-runtime scorer "
                "for %d repositories",
                len(repos_to_load),
            )
        else:
            logger.info("No BM25 data found, reconstructing from metadata...")
            elements = reconstruct_elements_from_metadata(vector_store)

            if elements:
                retriever.index_for_bm25(elements)
                logger.info("Rebuilt BM25 index with %d elements", len(elements))

                if not graphs_loaded:
                    graph_builder.build_graphs(elements)
                    logger.info("Rebuilt code graph")
            else:
                logger.warning("No elements reconstructed from metadata")

        retriever.build_repo_overview_bm25()
        logger.info("Built separate BM25 index for repository overviews")

        logger.info(
            "Successfully loaded %d repositories with %d total vectors",
            len(repos_to_load),
            vector_store.get_count(),
        )
        return True

    except Exception as e:
        logger.error("Failed to load multi-repo cache: %s", e)
        import traceback

        logger.error(traceback.format_exc())
        return False
