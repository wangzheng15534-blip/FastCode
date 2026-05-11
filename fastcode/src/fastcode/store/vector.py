"""
Vector Store - Store and retrieve code embeddings
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
from collections.abc import Mapping
from typing import Any, TypedDict, cast

import faiss
import numpy as np

from ..ir.element import CodeElementMeta
from ..utils import as_float32_matrix, as_float32_vector, ensure_dir
from ..utils.materialization import (
    BOUNDARY_JSON_DECODE,
    BOUNDARY_JSON_ENCODE,
    BOUNDARY_PICKLE_DUMP,
    BOUNDARY_PICKLE_LOAD,
    BOUNDARY_VECTOR_LIST_CONVERSION,
    increment_materialization_boundary,
)

_METADATA_SHARD_STORAGE_VERSION = 1
_VECTOR_SHARD_STORAGE_VERSION = 1
_REPO_OVERVIEW_STORAGE_VERSION = 1
_REPO_OVERVIEW_MANIFEST_FILENAME = "repo_overviews.json"
_REPO_OVERVIEW_EMBEDDINGS_FILENAME = "repo_overviews_embeddings.npz"
_LEGACY_REPO_OVERVIEW_FILENAME = "repo_overviews.pkl"


class _RepoOverviewManifestEntry(TypedDict):
    content: str
    metadata_json: str


class _RepoOverviewStoredEntry(TypedDict, total=False):
    repo_name: str
    content: str
    embedding: np.ndarray
    metadata: dict[str, Any]
    metadata_json: str
    raw_overview: dict[str, Any]


class _MetadataShardManifestEntry(TypedDict):
    path_key: str
    shard_file: str
    digest: str
    count: int


class _MetadataShardManifest(TypedDict):
    version: int
    dimension: int | None
    distance_metric: str
    index_type: str
    vector_count: int
    shards: list[_MetadataShardManifestEntry]


class _VectorShardManifestEntry(TypedDict):
    path_key: str
    shard_file: str
    digest: str
    count: int


class _VectorShardManifest(TypedDict):
    version: int
    dimension: int | None
    distance_metric: str
    index_type: str
    vector_count: int
    shards: list[_VectorShardManifestEntry]


class _IncrementalShardPlan(TypedDict):
    sequences_by_path: dict[str, list[int]]
    reusable_vector_entries: dict[str, _VectorShardManifestEntry]
    max_previous_sequence_no: int


class VectorStore:
    """Vector database for code embeddings using FAISS"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.vector_config: dict[str, Any] = config.get("vector_store", {})
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Evaluation mode can request a purely in-memory index that never touches disk.
        self.in_memory = self.vector_config.get(
            "in_memory",
            config.get("evaluation", {}).get("in_memory_index", False),
        )
        # Keep repo overviews in-memory when persistence is disabled.
        self._in_memory_repo_overviews: dict[str, dict[str, Any]] = {}

        self.dimension: int | None = None
        self.index: Any = None  # faiss index types are untyped
        self.metadata: list[CodeElementMeta] = []  # Store metadata for each vector
        self._vector_rows: np.ndarray | None = None
        self._vector_row_count = 0

        self.persist_dir: str = self.vector_config.get(
            "persist_directory", "./data/vector_store"
        )
        self.distance_metric: str = self.vector_config.get("distance_metric", "cosine")
        self.index_type: str = self.vector_config.get("index_type", "HNSW")
        self.persist_faiss_binary: bool = bool(
            self.vector_config.get("persist_faiss_binary", False)
        )
        raw_threshold = self.vector_config.get("vector_rows_search_threshold", 10000)
        try:
            self.vector_rows_search_threshold = max(0, int(raw_threshold))
        except (TypeError, ValueError):
            self.vector_rows_search_threshold = 10000

        # HNSW parameters
        self.m: int = self.vector_config.get("m", 16)
        self.ef_construction: int = self.vector_config.get("ef_construction", 200)
        self.ef_search: int = self.vector_config.get("ef_search", 50)

        # Cache for scan_available_indexes to avoid repeated file I/O
        self._index_scan_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._index_scan_cache_ttl: float = self.vector_config.get(
            "index_scan_cache_ttl", 30.0
        )
        self._index_scan_sample_size: int = self.vector_config.get(
            "index_scan_sample_size", 100
        )

        if not self.in_memory:
            ensure_dir(self.persist_dir)
        else:
            self.logger.info(
                "VectorStore running in in-memory mode; persistence disabled."
            )

    def initialize(self, dimension: int) -> None:
        """
        Initialize the vector store

        Args:
            dimension: Dimension of embedding vectors
        """
        self.dimension = dimension
        self.logger.info(f"Initializing vector store with dimension {dimension}")
        self.index = self._create_index(dimension)
        self.metadata: list[CodeElementMeta] = []
        self._vector_rows = np.empty((0, dimension), dtype=np.float32)
        self._vector_row_count = 0
        self.logger.info(
            f"Initialized {self.index_type} index with {self.distance_metric} distance"
        )

    def _create_index(self, dimension: int) -> Any:
        """Build a FAISS index for the configured metric/index type."""

        if self.index_type == "HNSW":
            # HNSW index for fast approximate search
            if self.distance_metric == "cosine":
                # Use inner product for cosine with normalized vectors
                index = faiss.IndexHNSWFlat(
                    dimension, self.m, faiss.METRIC_INNER_PRODUCT
                )
            else:
                # L2 distance
                index = faiss.IndexHNSWFlat(dimension, self.m, faiss.METRIC_L2)

            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            return index

        # Flat index for exact search (slower but more accurate)
        if self.distance_metric == "cosine":
            return faiss.IndexFlatIP(dimension)  # Inner product
        return faiss.IndexFlatL2(dimension)  # L2 distance

    def add_vectors(self, vectors: np.ndarray, metadata: list[CodeElementMeta]) -> None:
        """
        Add vectors to the store

        Args:
            vectors: Array of embedding vectors (N x dimension)
            metadata: List of metadata dictionaries for each vector
        """
        if self.dimension is None:
            raise RuntimeError("Vector store not initialized")
        if self.index is None and not self._ensure_faiss_index():
            raise RuntimeError("Vector store not initialized")

        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")

        # FAISS cosine normalization mutates input buffers, so this backend
        # boundary explicitly owns a mutable float32 matrix.
        vectors = as_float32_matrix(vectors, copy_policy="mutable")
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("vectors must be a non-empty 2D float32 matrix")

        # Normalize if using cosine similarity
        if self.distance_metric == "cosine":
            faiss.normalize_L2(vectors)

        # Add to index
        self.index.add(vectors)
        self.metadata.extend(metadata)
        self._append_vector_rows(vectors)

        self.logger.info(
            f"Added {len(vectors)} vectors to store (total: {len(self.metadata)})"
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        min_score: float | None = None,
        repo_filter: list[str] | None = None,
        element_type_filter: str | None = None,
    ) -> list[tuple[CodeElementMeta, float]]:
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            min_score: Minimum similarity score (optional)
            repo_filter: Optional list of repository names to filter by
            element_type_filter: Optional element type to filter by (e.g., "repository_overview")

        Returns:
            List of (metadata, score) tuples
        """
        if len(self.metadata) == 0:
            return []

        vector_rows = self._vector_rows_for_search()
        if vector_rows is not None:
            return self._search_with_vector_rows(
                vector_rows=vector_rows,
                query_vector=query_vector,
                k=k,
                min_score=min_score,
                repo_filter=repo_filter,
                element_type_filter=element_type_filter,
            )
        if self.index is None and not self._ensure_faiss_index():
            return []

        # Ensure query is float32 and 2D
        query_vector = as_float32_matrix(query_vector, copy_policy="mutable")
        if (
            query_vector.ndim != 2
            or query_vector.shape[0] != 1
            or query_vector.shape[1] == 0
        ):
            return []

        # Normalize if using cosine similarity
        if self.distance_metric == "cosine":
            faiss.normalize_L2(query_vector)

        # Search with larger k only for element_type_filter (not repo_filter)
        # Note: repo_filter now uses reloaded indexes, so no need to multiply k
        search_k = k * 5 if element_type_filter else k
        search_k = min(search_k, len(self.metadata))
        distances, indices = self.index.search(query_vector, search_k)

        # Prepare results
        results: list[tuple[CodeElementMeta, float]] = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            # Apply repository filter
            if repo_filter:
                repo_name = self.metadata[idx].get("repo_name")
                if repo_name not in repo_filter:
                    continue

            # Apply element type filter
            if element_type_filter:
                elem_type = self.metadata[idx].get("type")
                if elem_type != element_type_filter:
                    continue

            # Convert distance to similarity score
            if self.distance_metric == "cosine":
                score = float(dist)  # Inner product (already similarity)
            else:
                # Convert L2 distance to similarity
                score = 1.0 / (1.0 + float(dist))

            # Filter by minimum score
            if min_score is not None and score < min_score:
                continue

            results.append((cast(CodeElementMeta, self.metadata[idx]), score))

            # Stop if we have enough results
            if len(results) >= k:
                break

        return results

    def save_repo_overview(
        self,
        repo_name: str,
        overview_content: str,
        embedding: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        """
        Save a single repository overview to a separate file

        Args:
            repo_name: Name of the repository
            overview_content: Text content of the overview
            embedding: Embedding vector for the overview
            metadata: Additional metadata (repo_url, summary, structure, etc.)
        """
        normalized_embedding = self._normalize_repo_overview_embedding(embedding)
        if normalized_embedding is None:
            raise ValueError("Repository overview embedding must be a numeric vector")
        normalized_metadata = metadata

        if self.in_memory:
            # Keep entirely in memory during evaluation.
            self._in_memory_repo_overviews[repo_name] = {
                "repo_name": repo_name,
                "content": overview_content,
                "embedding": normalized_embedding,
                "metadata": normalized_metadata,
            }
            self.logger.info(f"Stored repository overview for {repo_name} (in-memory)")
            return

        overviews = self._load_repo_overview_entries(
            include_embeddings=True,
            decode_metadata=False,
        )
        overviews[repo_name] = {
            "repo_name": repo_name,
            "content": overview_content,
            "embedding": normalized_embedding,
            "metadata_json": self._serialize_repo_overview_metadata(
                normalized_metadata
            ),
        }

        try:
            self._write_repo_overview_entries(overviews)
            self.logger.info(f"Saved repository overview for {repo_name}")
        except Exception as e:
            self.logger.error(f"Failed to save repository overview: {e}")

    def delete_repo_overview(self, repo_name: str) -> bool:
        """
        Delete a repository overview from storage

        Args:
            repo_name: Name of the repository to remove

        Returns:
            True if the overview was found and removed
        """
        if self.in_memory:
            if repo_name in self._in_memory_repo_overviews:
                del self._in_memory_repo_overviews[repo_name]
                self.logger.info(f"Deleted in-memory overview for {repo_name}")
                return True
            return False

        manifest_path = self._repo_overview_manifest_path()
        legacy_path = self._legacy_repo_overview_path()
        if not os.path.exists(manifest_path) and not os.path.exists(legacy_path):
            return False

        try:
            overviews = self._load_repo_overview_entries(
                include_embeddings=True,
                decode_metadata=False,
            )
            if repo_name not in overviews:
                return False

            del overviews[repo_name]
            self._write_repo_overview_entries(overviews)
            self.logger.info(f"Deleted repository overview for {repo_name}")
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to delete repository overview for {repo_name}: {e}"
            )
            return False

    def load_repo_overviews(
        self, include_embeddings: bool = True
    ) -> dict[str, dict[str, Any]]:
        """
        Load all repository overviews from storage

        Args:
            include_embeddings: When False, avoid loading the embedding archive
                                and return only text/metadata fields.

        Returns:
            Dictionary mapping repo_name to overview data
        """
        return cast(
            dict[str, dict[str, Any]],
            self._load_repo_overview_entries(
                include_embeddings=include_embeddings,
                decode_metadata=True,
            ),
        )

    def search_repository_overviews(
        self, query_vector: np.ndarray, k: int = 5, min_score: float | None = None
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Search specifically for repository overview elements using separate storage

        Args:
            query_vector: Query embedding vector
            k: Number of repositories to return
            min_score: Minimum similarity score

        Returns:
            List of (metadata, score) tuples for repository overviews only
        """
        overviews = self._load_repo_overview_entries(
            include_embeddings=True,
            decode_metadata=False,
        )

        if not overviews:
            self.logger.warning("No repository overviews available for search")
            return []
        if k <= 0:
            return []

        query = as_float32_matrix(query_vector, copy_policy="mutable")
        if query.ndim != 2 or query.shape[0] != 1 or query.shape[1] == 0:
            return []

        repo_names: list[str] = []
        overview_payloads: list[_RepoOverviewStoredEntry] = []
        embedding_rows: list[np.ndarray] = []
        for repo_name, overview_data in overviews.items():
            embedding = as_float32_vector(
                overview_data.get("embedding"), copy_policy="view"
            )
            if embedding is None:
                continue
            if embedding.size != query.shape[1]:
                continue
            repo_names.append(repo_name)
            overview_payloads.append(overview_data)
            embedding_rows.append(embedding)

        if not embedding_rows:
            return []

        matrix = as_float32_matrix(embedding_rows, copy_policy="mutable")
        if matrix.shape[1] != query.shape[1]:
            return []
        if self.distance_metric == "cosine":
            faiss.normalize_L2(query)
            faiss.normalize_L2(matrix)
            scores = matrix @ query.reshape(-1)
        else:
            distances = np.linalg.norm(matrix - query, axis=1)
            scores = 1.0 / (1.0 + distances)

        ranked_indexes = np.argsort(scores)[::-1]
        results: list[tuple[dict[str, Any], float]] = []
        for raw_index in ranked_indexes:
            index = int(raw_index)
            score = float(scores[index])
            if min_score is not None and score < min_score:
                continue
            metadata = self._result_metadata_from_overview(overview_payloads[index])
            result_metadata = {
                "repo_name": repo_names[index],
                "type": "repository_overview",
                **metadata,
            }
            results.append((result_metadata, score))
            if len(results) >= k:
                break
        return results

    def search_batch(
        self, query_vectors: np.ndarray, k: int = 10, min_score: float | None = None
    ) -> list[list[tuple[CodeElementMeta, float]]]:
        """
        Search for multiple queries at once

        Args:
            query_vectors: Array of query vectors (N x dimension)
            k: Number of results per query
            min_score: Minimum similarity score

        Returns:
            List of result lists (one per query)
        """
        if len(self.metadata) == 0:
            return [[] for _ in range(len(query_vectors))]

        vector_rows = self._vector_rows_for_search()
        if vector_rows is not None:
            return self._search_batch_with_vector_rows(
                vector_rows=vector_rows,
                query_vectors=query_vectors,
                k=k,
                min_score=min_score,
            )
        if self.index is None and not self._ensure_faiss_index():
            return [[] for _ in range(len(query_vectors))]

        try:
            query_count = len(query_vectors)
        except TypeError:
            query_count = 0

        # Ensure float32
        query_vectors = as_float32_matrix(query_vectors, copy_policy="mutable")
        if query_vectors.ndim != 2 or query_vectors.shape[0] == 0:
            return [[] for _ in range(query_count)]

        # Normalize if using cosine
        if self.distance_metric == "cosine":
            faiss.normalize_L2(query_vectors)

        # Search
        k = min(k, len(self.metadata))
        distances, indices = self.index.search(query_vectors, k)

        # Prepare results for each query
        all_results: list[list[tuple[CodeElementMeta, float]]] = []
        for query_distances, query_indices in zip(distances, indices, strict=True):
            results: list[tuple[CodeElementMeta, float]] = []
            for dist, idx in zip(query_distances, query_indices, strict=True):
                if idx == -1:
                    continue

                # Convert distance to score
                if self.distance_metric == "cosine":
                    score = float(dist)
                else:
                    score = 1.0 / (1.0 + float(dist))

                if min_score is not None and score < min_score:
                    continue

                results.append((cast(CodeElementMeta, self.metadata[idx]), score))

            all_results.append(results)

        return all_results

    def get_count(self) -> int:
        """Get number of vectors in store"""
        return len(self.metadata)

    def get_repository_names(self) -> list[str]:
        """Get list of unique repository names in the store"""
        repo_names = set()
        for meta in self.metadata:
            repo_name = meta.get("repo_name")
            if repo_name:
                repo_names.add(repo_name)
        return sorted(repo_names)

    def get_count_by_repository(self) -> dict[str, int]:
        """Get count of vectors per repository"""
        repo_counts = {}
        for meta in self.metadata:
            repo_name = meta.get("repo_name", "unknown")
            repo_counts[repo_name] = repo_counts.get(repo_name, 0) + 1
        return repo_counts

    def filter_by_repositories(self, repo_names: list[str]) -> list[int]:
        """
        Get indices of vectors belonging to specific repositories

        Args:
            repo_names: List of repository names to filter by

        Returns:
            List of indices
        """
        indices: list[int] = []
        for i, meta in enumerate(self.metadata):
            if meta.get("repo_name") in repo_names:
                indices.append(i)
        return indices

    def save(self, name: str = "index") -> None:
        """
        Save index and metadata to disk

        Args:
            name: Name for the saved files
        """
        if self.in_memory:
            self.logger.info("Skipping vector store save (in-memory mode enabled)")
            return

        if len(self.metadata) == 0 or self.dimension is None:
            self.logger.warning("No vectors to save")
            return

        self._write_vector_bundle(name)
        self._write_metadata_bundle(name)
        self._persist_optional_faiss_binary(name)

        # Invalidate cache since we just modified the indexes
        self.invalidate_scan_cache()

        self.logger.info(f"Saved vector store to {self.persist_dir}")

    def save_incremental(
        self,
        name: str,
        *,
        previous_name: str,
        reusable_path_keys: set[str],
    ) -> dict[str, int]:
        """Save a new artifact while reusing unchanged vector path shards.

        Metadata shards are rewritten because they carry snapshot-scoped fields such
        as ``snapshot_id``. Vector shards can be reused safely when the path is
        unchanged and the row count still matches the previous shard.
        """
        if self.in_memory:
            self.logger.info(
                "Skipping vector store incremental save (in-memory mode enabled)"
            )
            return {
                "vector_shards_reused": 0,
                "vector_shards_written": 0,
                "metadata_shards_written": 0,
            }

        if len(self.metadata) == 0 or self.dimension is None:
            self.logger.warning("No vectors to save")
            return {
                "vector_shards_reused": 0,
                "vector_shards_written": 0,
                "metadata_shards_written": 0,
            }

        vectors = self._vector_matrix_for_persist()
        if len(vectors) != len(self.metadata):
            raise RuntimeError("Vector/metadata count mismatch during persistence")

        grouped_counts = self._path_row_counts(self.metadata)
        plan = self._build_incremental_shard_plan(
            previous_name=previous_name,
            reusable_path_keys=reusable_path_keys,
            grouped_counts=grouped_counts,
        )
        vector_stats = self._write_vector_bundle_with_sequences(
            name,
            vectors=vectors,
            sequences_by_path=plan["sequences_by_path"],
            previous_name=previous_name,
            reusable_entries=plan["reusable_vector_entries"],
        )
        metadata_stats = self._write_metadata_bundle_with_sequences(
            name,
            sequences_by_path=plan["sequences_by_path"],
        )
        ordered_vectors = self._vector_matrix_ordered_by_sequences(
            vectors, plan["sequences_by_path"]
        )
        self._persist_optional_faiss_binary(name, vectors=ordered_vectors)
        self.invalidate_scan_cache()
        self.logger.info(
            "Saved vector store to %s with %d reused vector shards",
            self.persist_dir,
            vector_stats["vector_shards_reused"],
        )
        return {**vector_stats, **metadata_stats}

    def load(self, name: str = "index") -> bool:
        """
        Load index and metadata from disk

        Args:
            name: Name of the saved files

        Returns:
            True if successful, False otherwise
        """
        if self.in_memory:
            self.logger.info("Skipping vector store load (in-memory mode enabled)")
            return False

        try:
            data = self.load_metadata_payload(name)
            if data is None:
                self.logger.warning(f"Index files not found in {self.persist_dir}")
                return False
            vector_payload = self.load_vector_payload(name)
            if vector_payload is not None:
                self.distance_metric = str(
                    data.get("distance_metric")
                    or vector_payload.get("distance_metric")
                    or "cosine"
                )
                self.index_type = str(
                    data.get("index_type") or vector_payload.get("index_type") or "HNSW"
                )
                vectors = cast(np.ndarray, vector_payload["vectors"])
                dimension = int(
                    data.get("dimension")
                    or vector_payload.get("dimension")
                    or (vectors.shape[1] if vectors.ndim == 2 else 0)
                )
                metadata = cast(list[CodeElementMeta], data["metadata"])
                if len(metadata) != len(vectors):
                    self.logger.error(
                        "Vector/metadata count mismatch for %s: %d vs %d",
                        name,
                        len(vectors),
                        len(metadata),
                    )
                    return False
                self.dimension = dimension
                self.metadata = metadata
                self._vector_rows = vectors.astype(np.float32, copy=False)
                self._vector_row_count = len(vectors)
                self.index = None
            else:
                index_path = self._legacy_index_path(name)
                if not os.path.exists(index_path):
                    self.logger.warning(f"Index files not found in {self.persist_dir}")
                    return False

                # Load FAISS index
                self.index = faiss.read_index(index_path)
                self.metadata = cast(list[CodeElementMeta], data["metadata"])
                self.dimension = data["dimension"]
                self.distance_metric = data.get("distance_metric", "cosine")
                self.index_type = data.get("index_type", "HNSW")
                self._vector_rows = None
                self._vector_row_count = 0

                # Set search parameters for HNSW
                if self.index_type == "HNSW" and hasattr(self.index, "hnsw"):
                    self.index.hnsw.efSearch = self.ef_search

            self.logger.info(
                f"Loaded vector store with {len(self.metadata)} vectors "
                f"from {self.persist_dir}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to load vector store: {e}")
            return False

    def clear(self) -> None:
        """Clear all vectors and metadata"""
        if self.dimension:
            self.initialize(self.dimension)
        else:
            self.index = None
            self.metadata: list[CodeElementMeta] = []
            self._vector_rows = None
            self._vector_row_count = 0
        self.logger.info("Cleared vector store")

    def merge_from_index(self, index_name: str) -> bool:
        """
        Merge vectors from another saved index into this store

        Args:
            index_name: Name of the index to merge from

        Returns:
            True if successful, False otherwise
        """
        if self.in_memory:
            self.logger.info("Skipping merge_from_index (in-memory mode enabled)")
            return False

        try:
            data = self.load_metadata_payload(index_name)
            if data is None:
                self.logger.warning(f"Index files not found for {index_name}")
                return False
            other_metadata = cast(list[CodeElementMeta], data["metadata"])
            other_dimension = data["dimension"]

            # Verify dimensions match
            if self.dimension and self.dimension != other_dimension:
                self.logger.error(
                    f"Dimension mismatch: {self.dimension} vs {other_dimension}"
                )
                return False

            if self.dimension is None:
                self.dimension = other_dimension

            vector_payload = self.load_vector_payload(index_name)
            if vector_payload is not None:
                vectors = cast(np.ndarray, vector_payload["vectors"])
                if len(vectors) == 0:
                    self.logger.warning(f"No vectors in {index_name}")
                    return False
                self.add_vectors(vectors, other_metadata)
                self.logger.info(f"Merged {len(vectors)} vectors from {index_name}")
                return True

            index_path = self._legacy_index_path(index_name)
            if not os.path.exists(index_path):
                self.logger.warning(f"Index files not found for {index_name}")
                return False

            # Load the other index
            other_index = faiss.read_index(index_path)
            n_vectors = other_index.ntotal
            if n_vectors == 0:
                self.logger.warning(f"No vectors in {index_name}")
                return False

            # Try to reconstruct all vectors efficiently
            try:
                # Reconstruct vectors - do it in batches for better performance
                vectors = np.zeros((n_vectors, other_dimension), dtype=np.float32)

                # Reconstruct all vectors at once
                for i in range(n_vectors):
                    other_index.reconstruct(int(i), vectors[i])

                # Add to our index in one batch operation
                self.add_vectors(vectors, other_metadata)
                self.logger.info(f"Merged {n_vectors} vectors from {index_name}")
                return True

            except Exception as e:
                self.logger.error(
                    f"Failed to reconstruct vectors from {index_name}: {e}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Failed to merge from {index_name}: {e}")
            return False

    def delete_by_filter(self, filter_func: Any) -> int:
        """
        Delete vectors matching a filter function

        Args:
            filter_func: Function that takes metadata and returns True to delete

        Returns:
            Number of vectors deleted
        """
        # FAISS doesn't support direct deletion, need to rebuild
        indices_to_keep: list[int] = []
        metadata_to_keep: list[CodeElementMeta] = []

        for i, meta in enumerate(self.metadata):
            if not filter_func(meta):
                indices_to_keep.append(i)
                metadata_to_keep.append(meta)

        num_deleted = len(self.metadata) - len(metadata_to_keep)

        if num_deleted > 0:
            # Rebuild index
            self.logger.info(f"Rebuilding index after deleting {num_deleted} vectors")

            # Get vectors to keep (this is expensive)
            # For now, we'll just track metadata
            # In production, consider storing vectors separately
            self.metadata = metadata_to_keep
            rows = self._valid_vector_rows(
                expected_count=len(indices_to_keep) + num_deleted
            )
            if rows is not None:
                self._vector_rows = rows[indices_to_keep].copy()
                self._vector_row_count = len(indices_to_keep)
            self.index = None

            self.logger.warning(
                "Note: FAISS doesn't support efficient deletion. "
                "Consider rebuilding the entire index for best performance."
            )

        return num_deleted

    def scan_available_indexes(self, use_cache: bool = True) -> list[dict[str, Any]]:
        """
        Scan persist directory for available index files (with caching)

        Args:
            use_cache: Use cached results if available (default: True)

        Returns:
            List of dictionaries with repository information
        """
        import time

        available_repos: list[dict[str, Any]] = []

        if self.in_memory:
            self.logger.info("Skipping index scan (in-memory mode enabled)")
            return available_repos

        if not os.path.exists(self.persist_dir):
            return available_repos

        # Check cache
        if use_cache and self._index_scan_cache is not None:
            cache_time, cached_results = self._index_scan_cache
            if time.time() - cache_time < self._index_scan_cache_ttl:
                self.logger.debug("Using cached index scan results")
                return cached_results

        # Perform actual scan
        self.logger.info("Scanning available indexes...")

        for repo_name in self._discover_saved_repo_names():
            if not self.has_saved_index(repo_name):
                continue
            try:
                total_size = self._vector_storage_size(
                    repo_name
                ) + self._metadata_storage_size(repo_name)
                total_size_mb = total_size / (1024 * 1024)

                element_count, file_count, repo_url = self._metadata_scan_stats(
                    repo_name
                )

                available_repos.append(
                    {
                        "name": repo_name,
                        "element_count": element_count,
                        "file_count": file_count,
                        "size_mb": round(total_size_mb, 2),
                        "url": repo_url,
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to read metadata for {repo_name}: {e}")
                available_repos.append(
                    {
                        "name": repo_name,
                        "element_count": 0,
                        "file_count": 0,
                        "size_mb": 0,
                        "url": "N/A",
                    }
                )

        results = sorted(available_repos, key=lambda x: x["name"])

        # Update cache
        self._index_scan_cache = (time.time(), results)
        self.logger.info(f"Index scan complete: found {len(results)} repositories")

        return results

    def invalidate_scan_cache(self) -> None:
        """Invalidate the scan cache (call this when indexes change)"""
        self._index_scan_cache = None
        self.logger.debug("Invalidated index scan cache")

    def has_saved_vectors(self, name: str) -> bool:
        return os.path.exists(self._vector_manifest_path(name)) or os.path.exists(
            self._legacy_index_path(name)
        )

    def has_saved_index(self, name: str) -> bool:
        return self.has_saved_metadata(name) and self.has_saved_vectors(name)

    def has_saved_metadata(self, name: str) -> bool:
        return os.path.exists(self._metadata_manifest_path(name)) or os.path.exists(
            self._legacy_metadata_path(name)
        )

    def vector_artifact_paths(self, name: str) -> list[str]:
        paths = [
            self._legacy_index_path(name),
            self._vector_manifest_path(name),
            self._vector_shards_dir(name),
        ]
        return [path for path in paths if os.path.exists(path)]

    def metadata_artifact_paths(self, name: str) -> list[str]:
        paths = [
            self._legacy_metadata_path(name),
            self._metadata_manifest_path(name),
            self._metadata_shards_dir(name),
        ]
        return [path for path in paths if os.path.exists(path)]

    def _legacy_index_path(self, name: str) -> str:
        return os.path.join(self.persist_dir, f"{name}.faiss")

    def _legacy_metadata_path(self, name: str) -> str:
        return os.path.join(self.persist_dir, f"{name}_metadata.pkl")

    def _vector_manifest_path(self, name: str) -> str:
        return os.path.join(self.persist_dir, f"{name}_vector_manifest.json")

    def _vector_shards_dir(self, name: str) -> str:
        return os.path.join(self.persist_dir, f"{name}_vector_shards")

    def _discover_saved_repo_names(self) -> list[str]:
        suffixes = (
            ".faiss",
            "_vector_manifest.json",
            "_vector_shards",
            "_metadata_manifest.json",
            "_metadata.pkl",
        )
        repo_names: set[str] = set()
        for entry in os.listdir(self.persist_dir):
            for suffix in suffixes:
                if entry.endswith(suffix):
                    repo_names.add(entry.removesuffix(suffix))
                    break
        return sorted(repo_names)

    def _append_vector_rows(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            return
        if vectors.shape[0] == 0:
            return
        if self.dimension is None:
            return

        required_count = self._vector_row_count + vectors.shape[0]
        rows = self._vector_rows
        if rows is None or rows.ndim != 2 or rows.shape[1] != vectors.shape[1]:
            rows = np.empty(
                (max(required_count, 1), vectors.shape[1]), dtype=np.float32
            )
            self._vector_rows = rows
            self._vector_row_count = 0
            required_count = vectors.shape[0]

        if rows.shape[0] < required_count:
            new_capacity = max(required_count, 1, rows.shape[0] * 2)
            grown = np.empty((new_capacity, rows.shape[1]), dtype=np.float32)
            if self._vector_row_count:
                grown[: self._vector_row_count] = rows[: self._vector_row_count]
            self._vector_rows = grown
            rows = grown

        start = self._vector_row_count
        rows[start : start + vectors.shape[0]] = vectors.astype(np.float32, copy=False)
        self._vector_row_count += vectors.shape[0]

    def _valid_vector_rows(
        self, *, expected_count: int | None = None
    ) -> np.ndarray | None:
        count = int(
            expected_count if expected_count is not None else len(self.metadata)
        )
        if self._vector_rows is None or self._vector_row_count != count:
            return None
        if self._vector_rows.ndim != 2 or self._vector_rows.shape[0] < count:
            return None
        return self._vector_rows[:count].astype(np.float32, copy=False)

    def _vector_rows_for_search(self) -> np.ndarray | None:
        rows = self._valid_vector_rows()
        if rows is None:
            return None
        if (
            self.vector_rows_search_threshold > 0
            and len(rows) > self.vector_rows_search_threshold
        ):
            return None
        return rows

    def _search_with_vector_rows(
        self,
        *,
        vector_rows: np.ndarray,
        query_vector: np.ndarray,
        k: int,
        min_score: float | None,
        repo_filter: list[str] | None,
        element_type_filter: str | None,
    ) -> list[tuple[CodeElementMeta, float]]:
        if k <= 0 or vector_rows.ndim != 2 or vector_rows.shape[0] == 0:
            return []

        query = as_float32_matrix(query_vector, copy_policy="mutable")
        if (
            query.ndim != 2
            or query.shape[0] != 1
            or query.shape[1] != vector_rows.shape[1]
        ):
            return []

        if self.distance_metric == "cosine":
            faiss.normalize_L2(query)
            scores = vector_rows @ query.reshape(-1)
        else:
            distances = np.linalg.norm(vector_rows - query, axis=1)
            scores = 1.0 / (1.0 + distances)

        candidate_indexes = np.arange(len(self.metadata), dtype=np.int64)
        if repo_filter:
            allowed_repos = set(repo_filter)
            candidate_indexes = candidate_indexes[
                [
                    self.metadata[int(idx)].get("repo_name") in allowed_repos
                    for idx in candidate_indexes
                ]
            ]
        if element_type_filter:
            candidate_indexes = candidate_indexes[
                [
                    self.metadata[int(idx)].get("type") == element_type_filter
                    for idx in candidate_indexes
                ]
            ]
        if candidate_indexes.size == 0:
            return []

        ranked_indexes = candidate_indexes[np.argsort(scores[candidate_indexes])[::-1]]
        results: list[tuple[CodeElementMeta, float]] = []
        for raw_index in ranked_indexes:
            index = int(raw_index)
            score = float(scores[index])
            if min_score is not None and score < min_score:
                continue
            results.append((self.metadata[index], score))
            if len(results) >= k:
                break
        return results

    def _search_batch_with_vector_rows(
        self,
        *,
        vector_rows: np.ndarray,
        query_vectors: np.ndarray,
        k: int,
        min_score: float | None,
    ) -> list[list[tuple[CodeElementMeta, float]]]:
        try:
            query_count = len(query_vectors)
        except TypeError:
            query_count = 0
        if k <= 0 or vector_rows.ndim != 2 or vector_rows.shape[0] == 0:
            return [[] for _ in range(query_count)]

        queries = as_float32_matrix(query_vectors, copy_policy="mutable")
        if queries.ndim != 2 or queries.shape[0] == 0:
            return [[] for _ in range(query_count)]
        if queries.shape[1] != vector_rows.shape[1]:
            return [[] for _ in range(len(queries))]

        if self.distance_metric == "cosine":
            faiss.normalize_L2(queries)
            score_matrix = queries @ vector_rows.T
        else:
            diff = vector_rows[None, :, :] - queries[:, None, :]
            distances = np.linalg.norm(diff, axis=2)
            score_matrix = 1.0 / (1.0 + distances)

        all_results: list[list[tuple[CodeElementMeta, float]]] = []
        for row_scores in score_matrix:
            ranked_indexes = np.argsort(row_scores)[::-1]
            results: list[tuple[CodeElementMeta, float]] = []
            for raw_index in ranked_indexes:
                index = int(raw_index)
                score = float(row_scores[index])
                if min_score is not None and score < min_score:
                    continue
                results.append((self.metadata[index], score))
                if len(results) >= k:
                    break
            all_results.append(results)
        return all_results

    def _ensure_faiss_index(self) -> bool:
        if self.index is not None:
            return True
        if self.dimension is None:
            return False
        self.index = self._create_index(self.dimension)
        rows = self._valid_vector_rows()
        if rows is not None and rows.size > 0:
            self.index.add(rows)
        return True

    def _vector_matrix_for_persist(self) -> np.ndarray:
        rows = self._valid_vector_rows()
        if rows is not None:
            return rows

        if self.index is None or self.dimension is None:
            raise RuntimeError("Vector rows unavailable for persistence")

        vectors = np.zeros((len(self.metadata), self.dimension), dtype=np.float32)
        for i in range(len(self.metadata)):
            self.index.reconstruct(int(i), vectors[i])
        self._vector_rows = vectors
        self._vector_row_count = len(vectors)
        return vectors

    def _persist_optional_faiss_binary(
        self, name: str, *, vectors: np.ndarray | None = None
    ) -> None:
        index_path = self._legacy_index_path(name)
        if self.persist_faiss_binary:
            if vectors is not None:
                if self.dimension is None:
                    return
                matrix = as_float32_matrix(vectors, copy_policy="contiguous")
                if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
                    return
                index = self._create_index(self.dimension)
                index.add(matrix)
                faiss.write_index(index, index_path)
                return
            if self.index is None and not self._ensure_faiss_index():
                return
            faiss.write_index(self.index, index_path)
            return
        if os.path.exists(index_path):
            os.remove(index_path)

    @staticmethod
    def _vector_shard_filename(path_key: str) -> str:
        digest = hashlib.sha256(path_key.encode("utf-8")).hexdigest()[:20]
        return f"{digest}.npz"

    @staticmethod
    def _vector_shard_bytes(
        *,
        sequence_nos: np.ndarray,
        vectors: np.ndarray,
    ) -> bytes:
        import io

        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            sequence_nos=sequence_nos.astype(np.int64, copy=False),
            vectors=vectors.astype(np.float32, copy=False),
        )
        return buffer.getvalue()

    @staticmethod
    def _copy_or_link_file(source_path: str, target_path: str) -> None:
        if os.path.abspath(source_path) == os.path.abspath(target_path):
            return
        if os.path.exists(target_path):
            os.remove(target_path)
        try:
            os.link(source_path, target_path)
        except OSError:
            shutil.copy2(source_path, target_path)

    @staticmethod
    def _path_row_counts(metadata: list[CodeElementMeta]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sequence_no, meta in enumerate(metadata):
            path_key = VectorStore._metadata_path_key(meta, sequence_no)
            counts[path_key] = counts.get(path_key, 0) + 1
        return counts

    def _load_vector_sequences_by_path(
        self,
        name: str,
        *,
        path_keys: set[str] | None = None,
    ) -> tuple[dict[str, list[int]], int]:
        manifest = self._load_vector_manifest(name)
        if manifest is None:
            return {}, -1
        shard_dir = self._vector_shards_dir(name)
        sequences_by_path: dict[str, list[int]] = {}
        max_sequence_no = -1
        for entry in manifest.get("shards", []):
            path_key = str(entry.get("path_key") or "")
            if not path_key or (path_keys is not None and path_key not in path_keys):
                continue
            shard_file = entry.get("shard_file")
            if not shard_file:
                continue
            shard_path = os.path.join(shard_dir, str(shard_file))
            try:
                with np.load(shard_path, allow_pickle=False) as archive:
                    sequence_nos = archive["sequence_nos"]
            except Exception as exc:
                self.logger.warning(
                    "Failed to read previous vector shard %s: %s",
                    shard_path,
                    exc,
                )
                continue
            if sequence_nos.ndim != 1:
                continue
            values = [int(value) for value in sequence_nos.astype(np.int64, copy=False)]
            if values:
                max_sequence_no = max(max_sequence_no, *values)
            sequences_by_path[path_key] = values
        return sequences_by_path, max_sequence_no

    def _build_incremental_shard_plan(
        self,
        *,
        previous_name: str,
        reusable_path_keys: set[str],
        grouped_counts: dict[str, int],
    ) -> _IncrementalShardPlan:
        previous_manifest = self._load_vector_manifest(previous_name)
        if not self._vector_manifest_supports_reuse(previous_manifest):
            return self._fresh_incremental_shard_plan(grouped_counts)
        previous_entries = {
            str(entry.get("path_key")): entry
            for entry in (
                previous_manifest.get("shards", []) if previous_manifest else []
            )
            if entry.get("path_key")
        }
        previous_sequences, max_previous_sequence_no = (
            self._load_vector_sequences_by_path(previous_name)
        )
        sequences_by_path: dict[str, list[int]] = {}
        reusable_entries: dict[str, _VectorShardManifestEntry] = {}
        used_sequences: set[int] = set()

        for path_key, count in grouped_counts.items():
            if path_key.startswith("__pathless__"):
                continue
            if path_key not in reusable_path_keys:
                continue
            previous_entry = previous_entries.get(path_key)
            previous_sequence_nos = previous_sequences.get(path_key)
            if not previous_entry or previous_sequence_nos is None:
                continue
            if len(previous_sequence_nos) != count:
                continue
            if any(
                sequence_no in used_sequences for sequence_no in previous_sequence_nos
            ):
                continue
            sequences_by_path[path_key] = list(previous_sequence_nos)
            reusable_entries[path_key] = previous_entry
            used_sequences.update(previous_sequence_nos)

        next_sequence_no = max(max_previous_sequence_no + 1, 0)
        for path_key, count in grouped_counts.items():
            if path_key in sequences_by_path:
                continue
            while next_sequence_no in used_sequences:
                next_sequence_no += 1
            assigned = list(range(next_sequence_no, next_sequence_no + count))
            sequences_by_path[path_key] = assigned
            used_sequences.update(assigned)
            next_sequence_no += count

        return {
            "sequences_by_path": sequences_by_path,
            "reusable_vector_entries": reusable_entries,
            "max_previous_sequence_no": max_previous_sequence_no,
        }

    def _vector_manifest_supports_reuse(
        self, manifest: _VectorShardManifest | None
    ) -> bool:
        if manifest is None:
            return False
        return (
            int(manifest.get("version") or 0) == _VECTOR_SHARD_STORAGE_VERSION
            and int(manifest.get("dimension") or 0) == int(self.dimension or 0)
            and str(manifest.get("distance_metric") or "cosine") == self.distance_metric
            and str(manifest.get("index_type") or "HNSW") == self.index_type
        )

    @staticmethod
    def _fresh_incremental_shard_plan(
        grouped_counts: dict[str, int],
    ) -> _IncrementalShardPlan:
        sequences_by_path: dict[str, list[int]] = {}
        next_sequence_no = 0
        for path_key, count in grouped_counts.items():
            sequences_by_path[path_key] = list(
                range(next_sequence_no, next_sequence_no + count)
            )
            next_sequence_no += count
        return {
            "sequences_by_path": sequences_by_path,
            "reusable_vector_entries": {},
            "max_previous_sequence_no": -1,
        }

    def _vector_matrix_ordered_by_sequences(
        self,
        vectors: np.ndarray,
        sequences_by_path: dict[str, list[int]],
    ) -> np.ndarray:
        ordered_rows: list[tuple[int, np.ndarray]] = []
        grouped_seen: dict[str, int] = {}
        for row_index, meta in enumerate(self.metadata):
            path_key = self._metadata_path_key(meta, row_index)
            seen_count = grouped_seen.get(path_key, 0)
            sequence_nos = sequences_by_path.get(path_key)
            if sequence_nos is None or len(sequence_nos) <= seen_count:
                raise RuntimeError(
                    f"Missing incremental vector sequence for path: {path_key}"
                )
            grouped_seen[path_key] = seen_count + 1
            ordered_rows.append((sequence_nos[seen_count], vectors[row_index]))
        ordered_rows.sort(key=lambda item: item[0])
        if not ordered_rows:
            return np.empty((0, int(self.dimension or 0)), dtype=np.float32)
        dimension = int(self.dimension or vectors.shape[1])
        ordered = np.empty((len(ordered_rows), dimension), dtype=np.float32)
        for dest_index, (_sequence_no, row) in enumerate(ordered_rows):
            ordered[dest_index] = row
        return ordered

    def _load_vector_manifest(self, name: str) -> _VectorShardManifest | None:
        manifest_path = self._vector_manifest_path(name)
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                increment_materialization_boundary(BOUNDARY_JSON_DECODE)
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        shards = data.get("shards")
        if not isinstance(shards, list):
            return None
        return cast(_VectorShardManifest, data)

    def _write_vector_bundle(self, name: str) -> None:
        vectors = self._vector_matrix_for_persist()
        if len(vectors) != len(self.metadata):
            raise RuntimeError("Vector/metadata count mismatch during persistence")

        shard_dir = self._vector_shards_dir(name)
        ensure_dir(shard_dir)
        existing_manifest = self._load_vector_manifest(name)
        existing_by_path = {
            str(entry.get("path_key")): entry
            for entry in (
                existing_manifest.get("shards", []) if existing_manifest else []
            )
            if entry.get("path_key")
        }

        grouped_rows: dict[str, list[tuple[int, np.ndarray]]] = {}
        for sequence_no, meta in enumerate(self.metadata):
            path_key = self._metadata_path_key(meta, sequence_no)
            grouped_rows.setdefault(path_key, []).append(
                (sequence_no, vectors[sequence_no])
            )

        manifest: _VectorShardManifest = {
            "version": _VECTOR_SHARD_STORAGE_VERSION,
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "index_type": self.index_type,
            "vector_count": len(self.metadata),
            "shards": [],
        }
        active_files: set[str] = set()
        for path_key, rows in grouped_rows.items():
            sequence_nos = np.asarray(
                [sequence_no for sequence_no, _ in rows], dtype=np.int64
            )
            row_indexes = np.asarray([sequence_no for sequence_no, _ in rows])
            shard_vectors = vectors[row_indexes].astype(np.float32, copy=False)
            shard_bytes = self._vector_shard_bytes(
                sequence_nos=sequence_nos,
                vectors=shard_vectors,
            )
            digest = hashlib.sha256(shard_bytes).hexdigest()
            existing = existing_by_path.get(path_key)
            shard_file = (
                str(existing.get("shard_file"))
                if existing and existing.get("shard_file")
                else self._vector_shard_filename(path_key)
            )
            shard_path = os.path.join(shard_dir, shard_file)
            active_files.add(shard_file)
            if not (
                existing
                and existing.get("digest") == digest
                and os.path.exists(shard_path)
            ):
                tmp_path = f"{shard_path}.tmp"
                with open(tmp_path, "wb") as handle:
                    handle.write(shard_bytes)
                os.replace(tmp_path, shard_path)
            manifest["shards"].append(
                {
                    "path_key": path_key,
                    "shard_file": shard_file,
                    "digest": digest,
                    "count": len(rows),
                }
            )

        if existing_manifest is not None:
            for entry in existing_manifest.get("shards", []):
                shard_file = entry.get("shard_file")
                if not shard_file or shard_file in active_files:
                    continue
                stale_path = os.path.join(shard_dir, str(shard_file))
                if os.path.exists(stale_path):
                    os.remove(stale_path)

        manifest_path = self._vector_manifest_path(name)
        tmp_manifest = f"{manifest_path}.tmp"
        with open(tmp_manifest, "w", encoding="utf-8") as handle:
            increment_materialization_boundary(
                BOUNDARY_JSON_ENCODE,
                items=len(manifest["shards"]),
            )
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        os.replace(tmp_manifest, manifest_path)

    def _write_vector_bundle_with_sequences(
        self,
        name: str,
        *,
        vectors: np.ndarray,
        sequences_by_path: dict[str, list[int]],
        previous_name: str,
        reusable_entries: dict[str, _VectorShardManifestEntry],
    ) -> dict[str, int]:
        shard_dir = self._vector_shards_dir(name)
        ensure_dir(shard_dir)
        previous_shard_dir = self._vector_shards_dir(previous_name)

        grouped_rows: dict[str, list[tuple[int, int, np.ndarray]]] = {}
        for row_index, meta in enumerate(self.metadata):
            path_key = self._metadata_path_key(meta, row_index)
            sequence_nos = sequences_by_path.get(path_key)
            if sequence_nos is None or len(sequence_nos) <= len(
                grouped_rows.get(path_key, [])
            ):
                raise RuntimeError(
                    f"Missing incremental vector sequence for path: {path_key}"
                )
            sequence_no = sequence_nos[len(grouped_rows.get(path_key, []))]
            grouped_rows.setdefault(path_key, []).append(
                (sequence_no, row_index, vectors[row_index])
            )

        manifest: _VectorShardManifest = {
            "version": _VECTOR_SHARD_STORAGE_VERSION,
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "index_type": self.index_type,
            "vector_count": len(self.metadata),
            "shards": [],
        }
        active_files: set[str] = set()
        reused = 0
        written = 0
        for path_key, rows in grouped_rows.items():
            reusable = reusable_entries.get(path_key)
            if reusable is not None:
                shard_file = str(reusable.get("shard_file") or "")
                source_path = os.path.join(previous_shard_dir, shard_file)
                target_path = os.path.join(shard_dir, shard_file)
                if shard_file and os.path.exists(source_path):
                    self._copy_or_link_file(source_path, target_path)
                    active_files.add(shard_file)
                    manifest["shards"].append(
                        {
                            "path_key": path_key,
                            "shard_file": shard_file,
                            "digest": str(reusable.get("digest") or ""),
                            "count": int(reusable.get("count") or len(rows)),
                        }
                    )
                    reused += 1
                    continue

            sequence_nos = np.asarray(
                [sequence_no for sequence_no, _, _ in rows], dtype=np.int64
            )
            row_indexes = np.asarray(
                [row_index for _, row_index, _ in rows], dtype=np.int64
            )
            shard_vectors = vectors[row_indexes].astype(np.float32, copy=False)
            shard_bytes = self._vector_shard_bytes(
                sequence_nos=sequence_nos,
                vectors=shard_vectors,
            )
            digest = hashlib.sha256(shard_bytes).hexdigest()
            shard_file = self._vector_shard_filename(path_key)
            shard_path = os.path.join(shard_dir, shard_file)
            active_files.add(shard_file)
            tmp_path = f"{shard_path}.tmp"
            with open(tmp_path, "wb") as handle:
                handle.write(shard_bytes)
            os.replace(tmp_path, shard_path)
            manifest["shards"].append(
                {
                    "path_key": path_key,
                    "shard_file": shard_file,
                    "digest": digest,
                    "count": len(rows),
                }
            )
            written += 1

        for entry_name in os.listdir(shard_dir):
            if not entry_name.endswith(".npz") or entry_name in active_files:
                continue
            os.remove(os.path.join(shard_dir, entry_name))

        manifest_path = self._vector_manifest_path(name)
        tmp_manifest = f"{manifest_path}.tmp"
        with open(tmp_manifest, "w", encoding="utf-8") as handle:
            increment_materialization_boundary(
                BOUNDARY_JSON_ENCODE,
                items=len(manifest["shards"]),
            )
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        os.replace(tmp_manifest, manifest_path)
        return {
            "vector_shards_reused": reused,
            "vector_shards_written": written,
        }

    def load_vector_payload(self, name: str) -> dict[str, Any] | None:
        manifest = self._load_vector_manifest(name)
        if manifest is None:
            return None
        try:
            shard_dir = self._vector_shards_dir(name)
            vector_count = int(manifest.get("vector_count") or 0)
            dimension = int(manifest.get("dimension") or 0)
            vectors = (
                np.empty((vector_count, dimension), dtype=np.float32)
                if vector_count > 0 and dimension > 0
                else np.empty((0, dimension), dtype=np.float32)
            )
            filled = np.zeros((vector_count,), dtype=bool) if vector_count > 0 else None
            fallback_rows: list[tuple[int, np.ndarray]] = []
            for entry in manifest.get("shards", []):
                shard_file = entry.get("shard_file")
                if not shard_file:
                    continue
                shard_path = os.path.join(shard_dir, str(shard_file))
                with np.load(shard_path, allow_pickle=False) as archive:
                    sequence_nos = archive["sequence_nos"]
                    shard_vectors = archive["vectors"]
                if sequence_nos.ndim != 1 or shard_vectors.ndim != 2:
                    continue
                seq = sequence_nos.astype(np.int64, copy=False)
                shard_matrix = shard_vectors.astype(np.float32, copy=False)
                if (
                    filled is not None
                    and shard_matrix.shape[0] == seq.shape[0]
                    and shard_matrix.shape[1] == dimension
                ):
                    valid = (seq >= 0) & (seq < vector_count)
                    if bool(valid.all()):
                        vectors[seq] = shard_matrix
                        filled[seq] = True
                        continue
                for sequence_no, vector in zip(seq, shard_matrix, strict=True):
                    fallback_rows.append((int(sequence_no), vector))
            if filled is not None and bool(filled.all()):
                loaded_vectors = vectors
            else:
                ordered_rows: list[tuple[int, np.ndarray]] = []
                if filled is not None:
                    for sequence_no in np.flatnonzero(filled):
                        ordered_rows.append((int(sequence_no), vectors[sequence_no]))
                ordered_rows.extend(fallback_rows)
                ordered_rows.sort(key=lambda item: item[0])
                loaded_vectors = np.empty(
                    (len(ordered_rows), dimension), dtype=np.float32
                )
                for row_index, (_sequence_no, row) in enumerate(ordered_rows):
                    loaded_vectors[row_index] = row
                if not ordered_rows and vector_count == 0:
                    loaded_vectors = np.empty(
                        (0, int(manifest.get("dimension") or 0)), dtype=np.float32
                    )
                elif not ordered_rows and vector_count > 0:
                    loaded_vectors = vectors[filled] if filled is not None else vectors
            return {
                "vectors": loaded_vectors,
                "dimension": manifest.get("dimension"),
                "distance_metric": manifest.get("distance_metric", "cosine"),
                "index_type": manifest.get("index_type", "HNSW"),
            }
        except Exception as e:
            self.logger.warning(f"Failed to load sharded vectors for {name}: {e}")
            return None

    def _vector_storage_size(self, name: str) -> int:
        total = 0
        manifest = self._load_vector_manifest(name)
        if manifest is not None:
            manifest_path = self._vector_manifest_path(name)
            if os.path.exists(manifest_path):
                total += os.path.getsize(manifest_path)
            shard_dir = self._vector_shards_dir(name)
            for entry in manifest.get("shards", []):
                shard_file = entry.get("shard_file")
                if not shard_file:
                    continue
                shard_path = os.path.join(shard_dir, str(shard_file))
                if os.path.exists(shard_path):
                    total += os.path.getsize(shard_path)
        legacy_index_path = self._legacy_index_path(name)
        if os.path.exists(legacy_index_path):
            total += os.path.getsize(legacy_index_path)
        return total

    def _metadata_manifest_path(self, name: str) -> str:
        return os.path.join(self.persist_dir, f"{name}_metadata_manifest.json")

    def _metadata_shards_dir(self, name: str) -> str:
        return os.path.join(self.persist_dir, f"{name}_metadata_shards")

    @staticmethod
    def _metadata_path_key(meta: Mapping[str, Any], sequence_no: int) -> str:
        relative_path = meta.get("relative_path")
        file_path = meta.get("file_path")
        if relative_path:
            return str(relative_path)
        if file_path:
            return str(file_path)
        fallback_id = meta.get("id")
        return f"__pathless__:{fallback_id or sequence_no}"

    @staticmethod
    def _metadata_shard_filename(path_key: str) -> str:
        digest = hashlib.sha256(path_key.encode("utf-8")).hexdigest()[:20]
        return f"{digest}.pkl"

    @staticmethod
    def _metadata_shard_bytes(entries: list[dict[str, Any]]) -> bytes:
        increment_materialization_boundary(BOUNDARY_PICKLE_DUMP, items=len(entries))
        return pickle.dumps({"entries": entries}, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_metadata_manifest(self, name: str) -> _MetadataShardManifest | None:
        manifest_path = self._metadata_manifest_path(name)
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                increment_materialization_boundary(BOUNDARY_JSON_DECODE)
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        shards = data.get("shards")
        if not isinstance(shards, list):
            return None
        return cast(_MetadataShardManifest, data)

    def _write_metadata_bundle(self, name: str) -> None:
        shard_dir = self._metadata_shards_dir(name)
        ensure_dir(shard_dir)
        existing_manifest = self._load_metadata_manifest(name)
        existing_by_path = {
            str(entry.get("path_key")): entry
            for entry in (
                existing_manifest.get("shards", []) if existing_manifest else []
            )
            if entry.get("path_key")
        }

        grouped_entries: dict[str, list[dict[str, Any]]] = {}
        for sequence_no, meta in enumerate(self.metadata):
            path_key = self._metadata_path_key(meta, sequence_no)
            grouped_entries.setdefault(path_key, []).append(
                {"sequence_no": sequence_no, "payload": meta}
            )

        manifest: _MetadataShardManifest = {
            "version": _METADATA_SHARD_STORAGE_VERSION,
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "index_type": self.index_type,
            "vector_count": len(self.metadata),
            "shards": [],
        }
        active_files: set[str] = set()
        for path_key, entries in grouped_entries.items():
            shard_bytes = self._metadata_shard_bytes(entries)
            digest = hashlib.sha256(shard_bytes).hexdigest()
            existing = existing_by_path.get(path_key)
            shard_file = (
                str(existing.get("shard_file"))
                if existing and existing.get("shard_file")
                else self._metadata_shard_filename(path_key)
            )
            shard_path = os.path.join(shard_dir, shard_file)
            active_files.add(shard_file)
            if not (
                existing
                and existing.get("digest") == digest
                and os.path.exists(shard_path)
            ):
                tmp_path = f"{shard_path}.tmp"
                with open(tmp_path, "wb") as handle:
                    handle.write(shard_bytes)
                os.replace(tmp_path, shard_path)
            manifest["shards"].append(
                {
                    "path_key": path_key,
                    "shard_file": shard_file,
                    "digest": digest,
                    "count": len(entries),
                }
            )

        if existing_manifest is not None:
            for entry in existing_manifest.get("shards", []):
                shard_file = entry.get("shard_file")
                if not shard_file or shard_file in active_files:
                    continue
                stale_path = os.path.join(shard_dir, str(shard_file))
                if os.path.exists(stale_path):
                    os.remove(stale_path)

        manifest_path = self._metadata_manifest_path(name)
        tmp_manifest = f"{manifest_path}.tmp"
        with open(tmp_manifest, "w", encoding="utf-8") as handle:
            increment_materialization_boundary(
                BOUNDARY_JSON_ENCODE,
                items=len(manifest["shards"]),
            )
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        os.replace(tmp_manifest, manifest_path)

        legacy_metadata_path = self._legacy_metadata_path(name)
        if os.path.exists(legacy_metadata_path):
            os.remove(legacy_metadata_path)

    def _write_metadata_bundle_with_sequences(
        self,
        name: str,
        *,
        sequences_by_path: dict[str, list[int]],
    ) -> dict[str, int]:
        shard_dir = self._metadata_shards_dir(name)
        ensure_dir(shard_dir)

        grouped_entries: dict[str, list[dict[str, Any]]] = {}
        for row_index, meta in enumerate(self.metadata):
            path_key = self._metadata_path_key(meta, row_index)
            sequence_nos = sequences_by_path.get(path_key)
            current_entries = grouped_entries.setdefault(path_key, [])
            if sequence_nos is None or len(sequence_nos) <= len(current_entries):
                raise RuntimeError(
                    f"Missing incremental metadata sequence for path: {path_key}"
                )
            current_entries.append(
                {"sequence_no": sequence_nos[len(current_entries)], "payload": meta}
            )

        manifest: _MetadataShardManifest = {
            "version": _METADATA_SHARD_STORAGE_VERSION,
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "index_type": self.index_type,
            "vector_count": len(self.metadata),
            "shards": [],
        }
        active_files: set[str] = set()
        written = 0
        for path_key, entries in grouped_entries.items():
            shard_bytes = self._metadata_shard_bytes(entries)
            digest = hashlib.sha256(shard_bytes).hexdigest()
            shard_file = self._metadata_shard_filename(path_key)
            shard_path = os.path.join(shard_dir, shard_file)
            active_files.add(shard_file)
            tmp_path = f"{shard_path}.tmp"
            with open(tmp_path, "wb") as handle:
                handle.write(shard_bytes)
            os.replace(tmp_path, shard_path)
            manifest["shards"].append(
                {
                    "path_key": path_key,
                    "shard_file": shard_file,
                    "digest": digest,
                    "count": len(entries),
                }
            )
            written += 1

        for entry_name in os.listdir(shard_dir):
            if not entry_name.endswith(".pkl") or entry_name in active_files:
                continue
            os.remove(os.path.join(shard_dir, entry_name))

        manifest_path = self._metadata_manifest_path(name)
        tmp_manifest = f"{manifest_path}.tmp"
        with open(tmp_manifest, "w", encoding="utf-8") as handle:
            increment_materialization_boundary(
                BOUNDARY_JSON_ENCODE,
                items=len(manifest["shards"]),
            )
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        os.replace(tmp_manifest, manifest_path)

        legacy_metadata_path = self._legacy_metadata_path(name)
        if os.path.exists(legacy_metadata_path):
            os.remove(legacy_metadata_path)
        return {"metadata_shards_written": written}

    def load_metadata_payload(self, name: str) -> dict[str, Any] | None:
        manifest = self._load_metadata_manifest(name)
        if manifest is not None:
            try:
                shard_dir = self._metadata_shards_dir(name)
                ordered_rows: list[tuple[int, CodeElementMeta]] = []
                for entry in manifest.get("shards", []):
                    shard_file = entry.get("shard_file")
                    if not shard_file:
                        continue
                    shard_path = os.path.join(shard_dir, str(shard_file))
                    with open(shard_path, "rb") as handle:
                        increment_materialization_boundary(BOUNDARY_PICKLE_LOAD)
                        payload = pickle.load(handle)
                    entries = (
                        payload.get("entries", []) if isinstance(payload, dict) else []
                    )
                    if not isinstance(entries, list):
                        continue
                    for item in entries:
                        if not isinstance(item, dict):
                            continue
                        sequence_no = item.get("sequence_no")
                        row = item.get("payload")
                        if not isinstance(sequence_no, int) or not isinstance(
                            row, Mapping
                        ):
                            continue
                        ordered_rows.append(
                            (
                                sequence_no,
                                cast(
                                    CodeElementMeta, dict(cast(Mapping[str, Any], row))
                                ),
                            )
                        )
                ordered_rows.sort(key=lambda item: item[0])
                return {
                    "metadata": [row for _, row in ordered_rows],
                    "dimension": manifest.get("dimension"),
                    "distance_metric": manifest.get("distance_metric", "cosine"),
                    "index_type": manifest.get("index_type", "HNSW"),
                }
            except Exception as e:
                self.logger.warning(f"Failed to load sharded metadata for {name}: {e}")

        metadata_path = self._legacy_metadata_path(name)
        if not os.path.exists(metadata_path):
            return None
        try:
            with open(metadata_path, "rb") as handle:
                increment_materialization_boundary(BOUNDARY_PICKLE_LOAD)
                data = pickle.load(handle)
            return cast(dict[str, Any], data) if isinstance(data, dict) else None
        except Exception as e:
            self.logger.warning(f"Failed to load metadata bundle for {name}: {e}")
            return None

    def _metadata_storage_size(self, name: str) -> int:
        manifest = self._load_metadata_manifest(name)
        if manifest is not None:
            total = 0
            manifest_path = self._metadata_manifest_path(name)
            if os.path.exists(manifest_path):
                total += os.path.getsize(manifest_path)
            shard_dir = self._metadata_shards_dir(name)
            for entry in manifest.get("shards", []):
                shard_file = entry.get("shard_file")
                if not shard_file:
                    continue
                shard_path = os.path.join(shard_dir, str(shard_file))
                if os.path.exists(shard_path):
                    total += os.path.getsize(shard_path)
            return total

        metadata_path = self._legacy_metadata_path(name)
        return os.path.getsize(metadata_path) if os.path.exists(metadata_path) else 0

    def _metadata_scan_stats(self, name: str) -> tuple[int, int, str]:
        manifest = self._load_metadata_manifest(name)
        if manifest is not None:
            element_count = int(manifest.get("vector_count", 0) or 0)
            file_count = len(
                [
                    entry
                    for entry in manifest.get("shards", [])
                    if str(entry.get("path_key") or "").startswith("__pathless__")
                    is False
                ]
            )
            repo_url = "N/A"
            shards = manifest.get("shards", [])
            if shards:
                first = shards[0]
                if first.get("shard_file"):
                    shard_path = os.path.join(
                        self._metadata_shards_dir(name), str(first["shard_file"])
                    )
                    try:
                        with open(shard_path, "rb") as handle:
                            increment_materialization_boundary(BOUNDARY_PICKLE_LOAD)
                            payload = pickle.load(handle)
                        entries = (
                            payload.get("entries", [])
                            if isinstance(payload, dict)
                            else []
                        )
                        if entries:
                            first_row = (
                                entries[0].get("payload")
                                if isinstance(entries[0], dict)
                                else None
                            )
                            if isinstance(first_row, Mapping):
                                repo_url = str(first_row.get("repo_url") or "N/A")
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to sample metadata shard for {name}: {e}"
                        )
            return element_count, file_count, repo_url

        data = self.load_metadata_payload(name)
        if data is None:
            return 0, 0, "N/A"
        metadata_list = data.get("metadata", [])
        if not isinstance(metadata_list, list):
            return 0, 0, "N/A"
        repo_url = "N/A"
        seen_files = set()
        for meta in metadata_list[: self._index_scan_sample_size]:
            if not isinstance(meta, Mapping):
                continue
            file_path = meta.get("file_path")
            if file_path:
                seen_files.add(str(file_path))
            if repo_url == "N/A" and meta.get("repo_url"):
                repo_url = str(meta.get("repo_url"))
        return len(metadata_list), len(seen_files), repo_url

    def _repo_overview_manifest_path(self) -> str:
        return os.path.join(self.persist_dir, _REPO_OVERVIEW_MANIFEST_FILENAME)

    def _repo_overview_embeddings_path(self) -> str:
        return os.path.join(self.persist_dir, _REPO_OVERVIEW_EMBEDDINGS_FILENAME)

    def _legacy_repo_overview_path(self) -> str:
        return os.path.join(self.persist_dir, _LEGACY_REPO_OVERVIEW_FILENAME)

    @staticmethod
    def _normalize_repo_overview_embedding(embedding: Any) -> np.ndarray | None:
        return as_float32_vector(embedding, copy_policy="contiguous")

    @staticmethod
    def _serialize_repo_overview_metadata(metadata: Any) -> str:
        safe_metadata = metadata if isinstance(metadata, dict) else {}
        increment_materialization_boundary(BOUNDARY_JSON_ENCODE)
        return json.dumps(
            safe_metadata,
            sort_keys=True,
            ensure_ascii=False,
            default=repr,
        )

    @staticmethod
    def _deserialize_repo_overview_metadata(raw_metadata: Any) -> dict[str, Any]:
        if not isinstance(raw_metadata, str):
            return {}
        try:
            increment_materialization_boundary(BOUNDARY_JSON_DECODE)
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            return {}
        return metadata if isinstance(metadata, dict) else {}

    def _load_repo_overview_entries(
        self,
        *,
        include_embeddings: bool,
        decode_metadata: bool,
    ) -> dict[str, _RepoOverviewStoredEntry]:
        if self.in_memory:
            return self._load_in_memory_repo_overview_entries(
                include_embeddings=include_embeddings,
                decode_metadata=decode_metadata,
            )

        manifest_path = self._repo_overview_manifest_path()
        legacy_path = self._legacy_repo_overview_path()

        if os.path.exists(manifest_path):
            try:
                overviews = self._load_repo_overviews_from_bundle(
                    include_embeddings=include_embeddings,
                    decode_metadata=decode_metadata,
                )
                self.logger.info(f"Loaded {len(overviews)} repository overviews")
                return overviews
            except Exception as e:
                self.logger.error(f"Failed to load repository overviews: {e}")
                return {}

        if os.path.exists(legacy_path):
            try:
                overviews = self._load_repo_overviews_from_legacy_pickle(
                    include_embeddings=include_embeddings,
                    decode_metadata=decode_metadata,
                )
                self.logger.info(f"Loaded {len(overviews)} legacy repository overviews")
                return overviews
            except Exception as e:
                self.logger.error(f"Failed to load repository overviews: {e}")
                return {}

        self.logger.info("No repository overviews found")
        return {}

    def _load_in_memory_repo_overview_entries(
        self,
        *,
        include_embeddings: bool,
        decode_metadata: bool,
    ) -> dict[str, _RepoOverviewStoredEntry]:
        overviews: dict[str, _RepoOverviewStoredEntry] = {}
        for repo_name, overview in self._in_memory_repo_overviews.items():
            entry: _RepoOverviewStoredEntry = {
                "repo_name": repo_name,
                "content": str(overview.get("content", "")),
            }
            if include_embeddings:
                embedding = self._normalize_repo_overview_embedding(
                    overview.get("embedding")
                )
                if embedding is None:
                    continue
                entry["embedding"] = embedding
            if decode_metadata:
                metadata = overview.get("metadata", {})
                entry["metadata"] = metadata if isinstance(metadata, dict) else {}
            else:
                entry["raw_overview"] = overview
            overviews[repo_name] = entry
        return overviews

    def _load_repo_overviews_from_bundle(
        self,
        *,
        include_embeddings: bool,
        decode_metadata: bool,
    ) -> dict[str, _RepoOverviewStoredEntry]:
        with open(self._repo_overview_manifest_path(), encoding="utf-8") as f:
            increment_materialization_boundary(BOUNDARY_JSON_DECODE)
            raw_manifest = json.load(f)

        repos = raw_manifest.get("repos")
        if not isinstance(repos, dict):
            return {}

        embeddings_by_repo: dict[str, np.ndarray] = {}
        if include_embeddings:
            embeddings_by_repo = self._load_repo_overview_embeddings()

        overviews: dict[str, _RepoOverviewStoredEntry] = {}
        for raw_repo_name, raw_entry in repos.items():
            if not isinstance(raw_repo_name, str) or not isinstance(raw_entry, dict):
                continue

            entry: _RepoOverviewStoredEntry = {
                "repo_name": raw_repo_name,
                "content": str(raw_entry.get("content", "")),
            }

            if include_embeddings:
                embedding = embeddings_by_repo.get(raw_repo_name)
                if embedding is None:
                    continue
                entry["embedding"] = embedding

            raw_metadata = raw_entry.get("metadata_json", "{}")
            metadata_json = raw_metadata if isinstance(raw_metadata, str) else "{}"
            if decode_metadata:
                entry["metadata"] = self._deserialize_repo_overview_metadata(
                    metadata_json
                )
            else:
                entry["metadata_json"] = metadata_json

            overviews[raw_repo_name] = entry

        return overviews

    def _load_repo_overviews_from_legacy_pickle(
        self,
        *,
        include_embeddings: bool,
        decode_metadata: bool,
    ) -> dict[str, _RepoOverviewStoredEntry]:
        with open(self._legacy_repo_overview_path(), "rb") as f:
            increment_materialization_boundary(BOUNDARY_PICKLE_LOAD)
            raw_overviews = pickle.load(f)
        if not isinstance(raw_overviews, dict):
            return {}

        overviews: dict[str, _RepoOverviewStoredEntry] = {}
        for raw_repo_name, raw_entry in raw_overviews.items():
            if not isinstance(raw_repo_name, str) or not isinstance(raw_entry, dict):
                continue

            entry: _RepoOverviewStoredEntry = {
                "repo_name": raw_repo_name,
                "content": str(raw_entry.get("content", "")),
            }

            if include_embeddings:
                embedding = self._normalize_repo_overview_embedding(
                    raw_entry.get("embedding")
                )
                if embedding is None:
                    continue
                entry["embedding"] = embedding

            if decode_metadata:
                metadata = raw_entry.get("metadata", {})
                entry["metadata"] = metadata if isinstance(metadata, dict) else {}
            else:
                entry["metadata_json"] = self._serialize_repo_overview_metadata(
                    raw_entry.get("metadata", {})
                )

            overviews[raw_repo_name] = entry

        return overviews

    def _load_repo_overview_embeddings(self) -> dict[str, np.ndarray]:
        embeddings_path = self._repo_overview_embeddings_path()
        if not os.path.exists(embeddings_path):
            return {}

        embeddings_by_repo: dict[str, np.ndarray] = {}
        with np.load(embeddings_path, allow_pickle=False) as archive:
            increment_materialization_boundary(BOUNDARY_VECTOR_LIST_CONVERSION)
            repo_names = np.asarray(archive["repo_names"]).tolist()
            if not isinstance(repo_names, list):
                return {}

            for index, raw_repo_name in enumerate(repo_names):
                repo_name = str(raw_repo_name)
                embedding_key = f"e{index}"
                if embedding_key not in archive.files:
                    continue
                embedding = self._normalize_repo_overview_embedding(
                    archive[embedding_key]
                )
                if embedding is None:
                    continue
                embeddings_by_repo[repo_name] = embedding

        return embeddings_by_repo

    def _result_metadata_from_overview(
        self, overview: _RepoOverviewStoredEntry
    ) -> dict[str, Any]:
        metadata_json = overview.get("metadata_json")
        if isinstance(metadata_json, str):
            return self._deserialize_repo_overview_metadata(metadata_json)
        metadata = overview.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        raw_overview = overview.get("raw_overview")
        if isinstance(raw_overview, dict):
            raw_metadata = raw_overview.get("metadata", {})
            return raw_metadata if isinstance(raw_metadata, dict) else {}
        return {}

    def _write_repo_overview_entries(
        self, overviews: dict[str, _RepoOverviewStoredEntry]
    ) -> None:
        manifest_path = self._repo_overview_manifest_path()
        embeddings_path = self._repo_overview_embeddings_path()
        legacy_path = self._legacy_repo_overview_path()

        if not overviews:
            for path in (manifest_path, embeddings_path, legacy_path):
                if os.path.exists(path):
                    os.remove(path)
            return

        manifest_repos: dict[str, _RepoOverviewManifestEntry] = {}
        repo_names: list[str] = []
        embeddings: list[np.ndarray] = []

        for repo_name in sorted(overviews):
            entry = overviews[repo_name]
            embedding = self._normalize_repo_overview_embedding(entry.get("embedding"))
            if embedding is None:
                continue

            raw_metadata_json = entry.get("metadata_json")
            metadata_json = (
                raw_metadata_json
                if isinstance(raw_metadata_json, str)
                else self._serialize_repo_overview_metadata(entry.get("metadata", {}))
            )

            manifest_repos[repo_name] = {
                "content": str(entry.get("content", "")),
                "metadata_json": metadata_json,
            }
            repo_names.append(repo_name)
            embeddings.append(embedding)

        if not manifest_repos:
            for path in (manifest_path, embeddings_path, legacy_path):
                if os.path.exists(path):
                    os.remove(path)
            return

        manifest_payload = {
            "version": _REPO_OVERVIEW_STORAGE_VERSION,
            "repos": manifest_repos,
        }
        manifest_tmp = f"{manifest_path}.tmp"
        embeddings_tmp = f"{embeddings_path}.tmp.npz"

        with open(manifest_tmp, "w", encoding="utf-8") as f:
            increment_materialization_boundary(
                BOUNDARY_JSON_ENCODE,
                items=len(manifest_repos),
            )
            json.dump(
                manifest_payload,
                f,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        os.replace(manifest_tmp, manifest_path)

        archive_payload: dict[str, Any] = {
            "repo_names": np.asarray(repo_names, dtype=np.str_)
        }
        for index, embedding in enumerate(embeddings):
            archive_payload[f"e{index}"] = embedding
        np.savez_compressed(embeddings_tmp, **archive_payload)
        os.replace(embeddings_tmp, embeddings_path)

        if os.path.exists(legacy_path):
            os.remove(legacy_path)
