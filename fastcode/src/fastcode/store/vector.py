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
from collections.abc import Mapping, Sequence
from typing import Any, Literal, NotRequired, TypedDict, cast

import faiss
import numpy as np

from ..ir.element import CodeElementMeta
from ..utils.filesystem import ensure_dir
from ..utils.materialization import (
    BOUNDARY_JSON_DECODE,
    BOUNDARY_JSON_ENCODE,
    BOUNDARY_PICKLE_DUMP,
    BOUNDARY_PICKLE_LOAD,
    BOUNDARY_VECTOR_LIST_CONVERSION,
    increment_materialization_boundary,
)
from .records import RepositoryOverviewRecord, VectorSearchResultRecord
from .vector_math import as_float32_matrix, as_float32_vector

_METADATA_SHARD_STORAGE_VERSION = 1
_VECTOR_SHARD_STORAGE_VERSION = 1
_VECTOR_SHARD_FORMAT_COMPRESSED = "compressed"
_VECTOR_SHARD_FORMAT_NPY = "npy"
_REPO_OVERVIEW_STORAGE_VERSION = 1
_REPO_OVERVIEW_MANIFEST_FILENAME = "repo_overviews.json"
_REPO_OVERVIEW_EMBEDDINGS_FILENAME = "repo_overviews_embeddings.npz"
_LEGACY_REPO_OVERVIEW_FILENAME = "repo_overviews.pkl"


class _RepoOverviewManifestEntry(TypedDict):
    content: str
    metadata_json: str
    embedding_fingerprint: NotRequired[dict[str, Any]]


class _RepoOverviewStoredEntry(TypedDict, total=False):
    repo_name: str
    content: str
    embedding: np.ndarray
    metadata: dict[str, Any]
    metadata_json: str
    embedding_fingerprint: dict[str, Any]
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
    embedding_fingerprint: dict[str, Any] | None
    shards: list[_MetadataShardManifestEntry]


class _VectorShardManifestEntry(TypedDict, total=False):
    path_key: str
    shard_file: str
    sequence_file: str
    vector_file: str
    storage_format: str
    digest: str
    count: int


class _VectorShardManifest(TypedDict):
    version: int
    dimension: int | None
    distance_metric: str
    index_type: str
    vector_count: int
    embedding_fingerprint: dict[str, Any] | None
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
        self._vector_shard_handles: tuple[_VectorShardManifestEntry, ...] | None = None
        self._vector_shard_dir: str | None = None

        self.persist_dir: str = self.vector_config.get(
            "persist_directory", "./data/vector_store"
        )
        self.distance_metric: str = self.vector_config.get("distance_metric", "cosine")
        self.index_type: str = self.vector_config.get("index_type", "HNSW")
        self.persist_faiss_binary: bool = bool(
            self.vector_config.get("persist_faiss_binary", False)
        )
        configured_shard_storage = str(
            self.vector_config.get("shard_storage") or _VECTOR_SHARD_FORMAT_COMPRESSED
        ).lower()
        if configured_shard_storage in {"mmap", "memory_mapped"}:
            configured_shard_storage = _VECTOR_SHARD_FORMAT_NPY
        if configured_shard_storage not in {
            _VECTOR_SHARD_FORMAT_COMPRESSED,
            _VECTOR_SHARD_FORMAT_NPY,
        }:
            configured_shard_storage = _VECTOR_SHARD_FORMAT_COMPRESSED
        self.vector_shard_storage = configured_shard_storage
        raw_threshold = self.vector_config.get("vector_rows_search_threshold", 10000)
        try:
            self.vector_rows_search_threshold = max(0, int(raw_threshold))
        except (TypeError, ValueError):
            self.vector_rows_search_threshold = 10000
        self.lazy_shard_search: bool = bool(
            self.vector_config.get("lazy_shard_search", False)
        )

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
        self._vector_shard_handles = None
        self._vector_shard_dir = None
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

    @staticmethod
    def _vector_search_payloads_from_records(
        records: Sequence[VectorSearchResultRecord],
    ) -> list[tuple[CodeElementMeta, float]]:
        return [(record.metadata, record.score) for record in records]

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        min_score: float | None = None,
        repo_filter: list[str] | None = None,
        element_type_filter: str | None = None,
        query_embedding_fingerprint: Mapping[str, Any] | None = None,
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
        return self._vector_search_payloads_from_records(
            self.search_records(
                query_vector=query_vector,
                k=k,
                min_score=min_score,
                repo_filter=repo_filter,
                element_type_filter=element_type_filter,
                query_embedding_fingerprint=query_embedding_fingerprint,
            )
        )

    def search_records(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        min_score: float | None = None,
        repo_filter: list[str] | None = None,
        element_type_filter: str | None = None,
        query_embedding_fingerprint: Mapping[str, Any] | None = None,
    ) -> list[VectorSearchResultRecord]:
        """Search for similar vectors and return typed result records."""
        if len(self.metadata) == 0:
            return []

        vector_rows = self._vector_rows_for_search()
        if vector_rows is not None:
            return self._search_records_with_vector_rows(
                vector_rows=vector_rows,
                query_vector=query_vector,
                k=k,
                min_score=min_score,
                repo_filter=repo_filter,
                element_type_filter=element_type_filter,
                query_embedding_fingerprint=query_embedding_fingerprint,
            )
        if self._vector_shard_handles is not None:
            return self._search_records_with_vector_shard_handles(
                query_vector=query_vector,
                k=k,
                min_score=min_score,
                repo_filter=repo_filter,
                element_type_filter=element_type_filter,
                query_embedding_fingerprint=query_embedding_fingerprint,
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
        search_k = (
            len(self.metadata)
            if query_embedding_fingerprint is not None
            else k * 5
            if element_type_filter
            else k
        )
        search_k = min(search_k, len(self.metadata))
        distances, indices = self.index.search(query_vector, search_k)

        # Prepare results
        results: list[VectorSearchResultRecord] = []
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
            if not self._metadata_embedding_fingerprint_matches(
                self.metadata[idx], query_embedding_fingerprint
            ):
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

            index = int(idx)
            results.append(
                VectorSearchResultRecord(
                    metadata=self.metadata[index],
                    score=score,
                    index=index,
                )
            )

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
        embedding_fingerprint = self._repo_overview_embedding_fingerprint(
            {"metadata": normalized_metadata},
        )

        if self.in_memory:
            # Keep entirely in memory during evaluation.
            overview_entry: dict[str, Any] = {
                "repo_name": repo_name,
                "content": overview_content,
                "embedding": normalized_embedding,
                "metadata": normalized_metadata,
            }
            if embedding_fingerprint is not None:
                overview_entry["embedding_fingerprint"] = embedding_fingerprint
            self._in_memory_repo_overviews[repo_name] = overview_entry
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
        return {
            repo_name: self._repo_overview_payload_from_record(record)
            for repo_name, record in self.load_repo_overview_records(
                include_embeddings=include_embeddings
            ).items()
        }

    def load_repo_overview_records(
        self, include_embeddings: bool = True
    ) -> dict[str, RepositoryOverviewRecord]:
        """Load repository overview storage rows as typed records."""
        entries = self._load_repo_overview_entries(
            include_embeddings=include_embeddings,
            decode_metadata=False,
        )
        records: dict[str, RepositoryOverviewRecord] = {}
        for repo_name, entry in entries.items():
            record = self._repo_overview_record_from_entry(
                repo_name,
                entry,
                materialize_metadata=True,
            )
            if record is not None:
                records[repo_name] = record
        return records

    def search_repository_overviews(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        min_score: float | None = None,
        query_embedding_fingerprint: Mapping[str, Any] | None = None,
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
        return [
            (self._repo_overview_result_metadata_from_record(record), score)
            for record, score in self.search_repository_overview_records(
                query_vector=query_vector,
                k=k,
                min_score=min_score,
                query_embedding_fingerprint=query_embedding_fingerprint,
            )
        ]

    def search_repository_overview_records(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        min_score: float | None = None,
        query_embedding_fingerprint: Mapping[str, Any] | None = None,
    ) -> list[tuple[RepositoryOverviewRecord, float]]:
        """Search repository overviews and return typed records."""
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
            if not self._repo_overview_embedding_fingerprint_matches(
                overview_data,
                query_embedding_fingerprint,
                metadata_json=overview_data.get("metadata_json"),
            ):
                continue
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
        results: list[tuple[RepositoryOverviewRecord, float]] = []
        for raw_index in ranked_indexes:
            index = int(raw_index)
            score = float(scores[index])
            if min_score is not None and score < min_score:
                continue
            record = self._repo_overview_record_from_entry(
                repo_names[index],
                overview_payloads[index],
                materialize_metadata=True,
            )
            if record is None:
                continue
            results.append((record, score))
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
        return [
            self._vector_search_payloads_from_records(records)
            for records in self.search_batch_records(
                query_vectors=query_vectors,
                k=k,
                min_score=min_score,
            )
        ]

    def search_batch_records(
        self, query_vectors: np.ndarray, k: int = 10, min_score: float | None = None
    ) -> list[list[VectorSearchResultRecord]]:
        """Search for multiple queries and return typed result records."""
        if len(self.metadata) == 0:
            return [[] for _ in range(len(query_vectors))]

        vector_rows = self._vector_rows_for_search()
        if vector_rows is not None:
            return self._search_batch_records_with_vector_rows(
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
        all_results: list[list[VectorSearchResultRecord]] = []
        for query_distances, query_indices in zip(distances, indices, strict=True):
            results: list[VectorSearchResultRecord] = []
            for dist, idx in zip(query_distances, query_indices, strict=True):
                if idx == -1:
                    continue
                index = int(idx)

                # Convert distance to score
                if self.distance_metric == "cosine":
                    score = float(dist)
                else:
                    score = 1.0 / (1.0 + float(dist))

                if min_score is not None and score < min_score:
                    continue

                results.append(
                    VectorSearchResultRecord(
                        metadata=self.metadata[index],
                        score=score,
                        index=index,
                    )
                )

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

    def publish_delta(
        self,
        name: str,
        *,
        previous_name: str,
        reusable_path_keys: set[str],
        snapshot_id: str | None = None,
    ) -> dict[str, int | str | None]:
        """Publish changed rows plus previous shard handles as a new artifact.

        Unlike ``save_incremental()``, ``self.metadata`` and ``self._vector_rows`` are
        expected to contain only changed/new rows. Unchanged rows are copied or
        hard-linked from the previous artifact by path shard handle.
        """
        if self.in_memory:
            self.logger.info("Skipping vector delta publish (in-memory mode enabled)")
            return {
                "vector_shards_reused": 0,
                "vector_shards_written": 0,
                "metadata_shards_reused": 0,
                "metadata_shards_written": 0,
                "vector_rows_reused": 0,
                "vector_rows_written": 0,
                "fallback_reason": "in_memory",
            }
        if self.dimension is None:
            self.logger.warning("No vector dimension configured for delta publish")
            return {
                "vector_shards_reused": 0,
                "vector_shards_written": 0,
                "metadata_shards_reused": 0,
                "metadata_shards_written": 0,
                "vector_rows_reused": 0,
                "vector_rows_written": 0,
                "fallback_reason": "missing_dimension",
            }

        previous_manifest = self._load_vector_manifest(previous_name)
        if not self._vector_manifest_supports_reuse(previous_manifest):
            if len(self.metadata) == 0:
                return {
                    "vector_shards_reused": 0,
                    "vector_shards_written": 0,
                    "metadata_shards_reused": 0,
                    "metadata_shards_written": 0,
                    "vector_rows_reused": 0,
                    "vector_rows_written": 0,
                    "fallback_reason": "previous_manifest_incompatible",
                }
            return {
                **self.save_incremental(
                    name,
                    previous_name=previous_name,
                    reusable_path_keys=set(),
                ),
                "metadata_shards_reused": 0,
                "vector_rows_reused": 0,
                "vector_rows_written": len(self.metadata),
                "fallback_reason": "previous_manifest_incompatible",
            }
        previous_vector_count = int(
            cast(Mapping[str, Any], previous_manifest).get("vector_count") or 0
        )

        vectors = (
            self._vector_matrix_for_persist()
            if len(self.metadata) > 0
            else np.empty((0, int(self.dimension)), dtype=np.float32)
        )
        if len(vectors) != len(self.metadata):
            raise RuntimeError("Vector/metadata count mismatch during delta publish")

        plan = self._build_delta_shard_plan(
            previous_name=previous_name,
            reusable_path_keys=reusable_path_keys,
        )
        vector_stats = self._write_vector_bundle_delta(
            name,
            previous_name=previous_name,
            vectors=vectors,
            sequences_by_path=plan["sequences_by_path"],
            reusable_entries=plan["reusable_vector_entries"],
            previous_vector_count=previous_vector_count,
        )
        metadata_stats = self._write_metadata_bundle_delta(
            name,
            previous_name=previous_name,
            sequences_by_path=plan["sequences_by_path"],
            reusable_path_keys=set(plan["reusable_vector_entries"]),
            snapshot_id=snapshot_id,
            previous_vector_count=previous_vector_count,
        )
        index_path = self._legacy_index_path(name)
        if os.path.exists(index_path):
            os.remove(index_path)
        self.invalidate_scan_cache()
        return {
            **vector_stats,
            **metadata_stats,
            "fallback_reason": None,
        }

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
            metadata = cast(list[CodeElementMeta], data["metadata"])
            lazy_manifest = self._load_vector_manifest(name)
            if self.lazy_shard_search and lazy_manifest is not None:
                vector_count = int(lazy_manifest.get("vector_count") or 0)
                if len(metadata) != vector_count:
                    self.logger.error(
                        "Vector/metadata count mismatch for %s: %d vs %d",
                        name,
                        vector_count,
                        len(metadata),
                    )
                    return False
                self.distance_metric = str(
                    data.get("distance_metric")
                    or lazy_manifest.get("distance_metric")
                    or "cosine"
                )
                self.index_type = str(
                    data.get("index_type") or lazy_manifest.get("index_type") or "HNSW"
                )
                self.dimension = int(
                    data.get("dimension") or lazy_manifest.get("dimension") or 0
                )
                self.metadata = metadata
                self._vector_rows = None
                self._vector_row_count = vector_count
                self._vector_shard_handles = tuple(lazy_manifest.get("shards", []))
                self._vector_shard_dir = self._vector_shards_dir(name)
                self.index = None
                self.logger.info(
                    "Loaded vector store metadata and %d lazy shard handles from %s",
                    len(self._vector_shard_handles),
                    self.persist_dir,
                )
                return True

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
                self._vector_shard_handles = None
                self._vector_shard_dir = None
                self.index = None
            else:
                index_path = self._legacy_index_path(name)
                if not os.path.exists(index_path):
                    self.logger.warning(f"Index files not found in {self.persist_dir}")
                    return False

                # Load FAISS index
                self.index = faiss.read_index(index_path)
                self.metadata = metadata
                self.dimension = data["dimension"]
                self.distance_metric = data.get("distance_metric", "cosine")
                self.index_type = data.get("index_type", "HNSW")
                self._vector_rows = None
                self._vector_row_count = 0
                self._vector_shard_handles = None
                self._vector_shard_dir = None

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
            self._vector_shard_handles = None
            self._vector_shard_dir = None
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
        query_embedding_fingerprint: Mapping[str, Any] | None,
    ) -> list[tuple[CodeElementMeta, float]]:
        return self._vector_search_payloads_from_records(
            self._search_records_with_vector_rows(
                vector_rows=vector_rows,
                query_vector=query_vector,
                k=k,
                min_score=min_score,
                repo_filter=repo_filter,
                element_type_filter=element_type_filter,
                query_embedding_fingerprint=query_embedding_fingerprint,
            )
        )

    def _search_records_with_vector_rows(
        self,
        *,
        vector_rows: np.ndarray,
        query_vector: np.ndarray,
        k: int,
        min_score: float | None,
        repo_filter: list[str] | None,
        element_type_filter: str | None,
        query_embedding_fingerprint: Mapping[str, Any] | None,
    ) -> list[VectorSearchResultRecord]:
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
        if query_embedding_fingerprint is not None:
            candidate_indexes = candidate_indexes[
                [
                    self._metadata_embedding_fingerprint_matches(
                        self.metadata[int(idx)], query_embedding_fingerprint
                    )
                    for idx in candidate_indexes
                ]
            ]
        if candidate_indexes.size == 0:
            return []

        ranked_indexes = candidate_indexes[np.argsort(scores[candidate_indexes])[::-1]]
        results: list[VectorSearchResultRecord] = []
        for raw_index in ranked_indexes:
            index = int(raw_index)
            score = float(scores[index])
            if min_score is not None and score < min_score:
                continue
            results.append(
                VectorSearchResultRecord(
                    metadata=self.metadata[index],
                    score=score,
                    index=index,
                )
            )
            if len(results) >= k:
                break
        return results

    def _search_with_vector_shard_handles(
        self,
        *,
        query_vector: np.ndarray,
        k: int,
        min_score: float | None,
        repo_filter: list[str] | None,
        element_type_filter: str | None,
        query_embedding_fingerprint: Mapping[str, Any] | None,
    ) -> list[tuple[CodeElementMeta, float]]:
        return self._vector_search_payloads_from_records(
            self._search_records_with_vector_shard_handles(
                query_vector=query_vector,
                k=k,
                min_score=min_score,
                repo_filter=repo_filter,
                element_type_filter=element_type_filter,
                query_embedding_fingerprint=query_embedding_fingerprint,
            )
        )

    def _search_records_with_vector_shard_handles(
        self,
        *,
        query_vector: np.ndarray,
        k: int,
        min_score: float | None,
        repo_filter: list[str] | None,
        element_type_filter: str | None,
        query_embedding_fingerprint: Mapping[str, Any] | None,
    ) -> list[VectorSearchResultRecord]:
        if k <= 0 or self.dimension is None:
            return []
        if not self._vector_shard_handles or not self._vector_shard_dir:
            return []

        query = as_float32_matrix(query_vector, copy_policy="mutable")
        if query.ndim != 2 or query.shape[0] != 1 or query.shape[1] != self.dimension:
            return []
        if self.distance_metric == "cosine":
            faiss.normalize_L2(query)
        query_row = query.reshape(-1)
        allowed_repos = set(repo_filter or [])
        results: list[tuple[int, float]] = []
        for entry in self._vector_shard_handles:
            arrays = self._load_vector_shard_arrays(
                shard_dir=self._vector_shard_dir,
                entry=entry,
                mmap_mode="r",
            )
            if arrays is None:
                continue
            sequence_nos, shard_vectors = arrays
            if sequence_nos.ndim != 1 or shard_vectors.ndim != 2:
                continue
            if shard_vectors.shape[1] != self.dimension:
                continue
            seq = sequence_nos.astype(np.int64, copy=False)
            matrix = shard_vectors.astype(np.float32, copy=False)
            if self.distance_metric == "cosine":
                scores = matrix @ query_row
            else:
                distances = np.linalg.norm(matrix - query_row, axis=1)
                scores = 1.0 / (1.0 + distances)
            for local_index, sequence_no in enumerate(seq):
                metadata_index = int(sequence_no)
                if metadata_index < 0 or metadata_index >= len(self.metadata):
                    continue
                metadata = self.metadata[metadata_index]
                if allowed_repos and metadata.get("repo_name") not in allowed_repos:
                    continue
                if element_type_filter and metadata.get("type") != element_type_filter:
                    continue
                if not self._metadata_embedding_fingerprint_matches(
                    metadata, query_embedding_fingerprint
                ):
                    continue
                score = float(scores[local_index])
                if min_score is not None and score < min_score:
                    continue
                results.append((metadata_index, score))
        results.sort(key=lambda item: (-item[1], item[0]))
        return [
            VectorSearchResultRecord(
                metadata=self.metadata[index],
                score=score,
                index=index,
            )
            for index, score in results[:k]
        ]

    def _search_batch_with_vector_rows(
        self,
        *,
        vector_rows: np.ndarray,
        query_vectors: np.ndarray,
        k: int,
        min_score: float | None,
    ) -> list[list[tuple[CodeElementMeta, float]]]:
        return [
            self._vector_search_payloads_from_records(records)
            for records in self._search_batch_records_with_vector_rows(
                vector_rows=vector_rows,
                query_vectors=query_vectors,
                k=k,
                min_score=min_score,
            )
        ]

    def _search_batch_records_with_vector_rows(
        self,
        *,
        vector_rows: np.ndarray,
        query_vectors: np.ndarray,
        k: int,
        min_score: float | None,
    ) -> list[list[VectorSearchResultRecord]]:
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

        all_results: list[list[VectorSearchResultRecord]] = []
        for row_scores in score_matrix:
            ranked_indexes = np.argsort(row_scores)[::-1]
            results: list[VectorSearchResultRecord] = []
            for raw_index in ranked_indexes:
                index = int(raw_index)
                score = float(row_scores[index])
                if min_score is not None and score < min_score:
                    continue
                results.append(
                    VectorSearchResultRecord(
                        metadata=self.metadata[index],
                        score=score,
                        index=index,
                    )
                )
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
    def _metadata_embedding_fingerprint_value(meta: Mapping[str, Any]) -> Any:
        value = meta.get("embedding_fingerprint")
        if value is not None:
            return value
        nested = meta.get("metadata")
        if isinstance(nested, Mapping):
            return nested.get("embedding_fingerprint")
        return None

    @staticmethod
    def _embedding_fingerprint_matches(
        stored: Any,
        expected: Mapping[str, Any] | None,
    ) -> bool:
        if expected is None:
            return True
        if not isinstance(stored, Mapping):
            return False
        stored_payload = cast(Mapping[str, Any], stored)
        for field_name, expected_value in expected.items():
            if stored_payload.get(field_name) != expected_value:
                return False
        return True

    @classmethod
    def _metadata_embedding_fingerprint_matches(
        cls,
        meta: Mapping[str, Any],
        expected: Mapping[str, Any] | None,
    ) -> bool:
        return cls._embedding_fingerprint_matches(
            cls._metadata_embedding_fingerprint_value(meta),
            expected,
        )

    @classmethod
    def _embedding_fingerprint_from_metadata(
        cls, metadata: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any] | None:
        for meta in metadata:
            value = cls._metadata_embedding_fingerprint_value(meta)
            if isinstance(value, Mapping):
                return dict(cast(Mapping[str, Any], value))
        return None

    @staticmethod
    def _vector_shard_file_stem(path_key: str) -> str:
        return hashlib.sha256(path_key.encode("utf-8")).hexdigest()[:20]

    @classmethod
    def _vector_shard_filename(cls, path_key: str) -> str:
        return f"{cls._vector_shard_file_stem(path_key)}.npz"

    @classmethod
    def _vector_shard_sequence_filename(cls, path_key: str) -> str:
        return f"{cls._vector_shard_file_stem(path_key)}.seq.npy"

    @classmethod
    def _vector_shard_vector_filename(cls, path_key: str) -> str:
        return f"{cls._vector_shard_file_stem(path_key)}.vec.npy"

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
    def _vector_shard_array_digest(
        *,
        sequence_nos: np.ndarray,
        vectors: np.ndarray,
    ) -> str:
        seq = np.ascontiguousarray(sequence_nos.astype(np.int64, copy=False))
        matrix = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
        digest = hashlib.sha256()
        for array in (seq, matrix):
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(array.dtype.str.encode("utf-8"))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    @staticmethod
    def _write_npy_atomic(path: str, array: np.ndarray) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(tmp_path, path)

    @staticmethod
    def _vector_entry_files(entry: Mapping[str, Any]) -> list[str]:
        files: list[str] = []
        sequence_file = entry.get("sequence_file")
        vector_file = entry.get("vector_file")
        if sequence_file and vector_file:
            files.extend([str(sequence_file), str(vector_file)])
            return files
        shard_file = entry.get("shard_file")
        return [str(shard_file)] if shard_file else []

    def _write_vector_shard_files(
        self,
        *,
        path_key: str,
        shard_dir: str,
        sequence_nos: np.ndarray,
        vectors: np.ndarray,
        existing: Mapping[str, Any] | None = None,
    ) -> tuple[_VectorShardManifestEntry, set[str], bool]:
        seq = sequence_nos.astype(np.int64, copy=False)
        matrix = vectors.astype(np.float32, copy=False)
        if self.vector_shard_storage == _VECTOR_SHARD_FORMAT_NPY:
            sequence_file = (
                str(existing.get("sequence_file"))
                if existing
                and existing.get("storage_format") == _VECTOR_SHARD_FORMAT_NPY
                and existing.get("sequence_file")
                else self._vector_shard_sequence_filename(path_key)
            )
            vector_file = (
                str(existing.get("vector_file"))
                if existing
                and existing.get("storage_format") == _VECTOR_SHARD_FORMAT_NPY
                and existing.get("vector_file")
                else self._vector_shard_vector_filename(path_key)
            )
            sequence_path = os.path.join(shard_dir, sequence_file)
            vector_path = os.path.join(shard_dir, vector_file)
            digest = self._vector_shard_array_digest(
                sequence_nos=seq,
                vectors=matrix,
            )
            unchanged = (
                existing is not None
                and existing.get("storage_format") == _VECTOR_SHARD_FORMAT_NPY
                and existing.get("digest") == digest
                and os.path.exists(sequence_path)
                and os.path.exists(vector_path)
            )
            if not unchanged:
                self._write_npy_atomic(sequence_path, seq)
                self._write_npy_atomic(vector_path, matrix)
            return (
                {
                    "path_key": path_key,
                    "sequence_file": sequence_file,
                    "vector_file": vector_file,
                    "storage_format": _VECTOR_SHARD_FORMAT_NPY,
                    "digest": digest,
                    "count": int(seq.shape[0]),
                },
                {sequence_file, vector_file},
                not unchanged,
            )

        shard_bytes = self._vector_shard_bytes(sequence_nos=seq, vectors=matrix)
        digest = hashlib.sha256(shard_bytes).hexdigest()
        shard_file = (
            str(existing.get("shard_file"))
            if existing
            and existing.get("storage_format", _VECTOR_SHARD_FORMAT_COMPRESSED)
            == _VECTOR_SHARD_FORMAT_COMPRESSED
            and existing.get("shard_file")
            else self._vector_shard_filename(path_key)
        )
        shard_path = os.path.join(shard_dir, shard_file)
        unchanged = (
            existing is not None
            and existing.get("digest") == digest
            and os.path.exists(shard_path)
        )
        if not unchanged:
            tmp_path = f"{shard_path}.tmp"
            with open(tmp_path, "wb") as handle:
                handle.write(shard_bytes)
            os.replace(tmp_path, shard_path)
        return (
            {
                "path_key": path_key,
                "shard_file": shard_file,
                "storage_format": _VECTOR_SHARD_FORMAT_COMPRESSED,
                "digest": digest,
                "count": int(seq.shape[0]),
            },
            {shard_file},
            not unchanged,
        )

    def _copy_or_link_vector_entry_files(
        self,
        *,
        source_dir: str,
        target_dir: str,
        entry: Mapping[str, Any],
    ) -> set[str]:
        active_files: set[str] = set()
        for file_name in self._vector_entry_files(entry):
            source_path = os.path.join(source_dir, file_name)
            target_path = os.path.join(target_dir, file_name)
            if not os.path.exists(source_path):
                continue
            self._copy_or_link_file(source_path, target_path)
            active_files.add(file_name)
        return active_files

    @staticmethod
    def _reused_vector_manifest_entry(
        *,
        path_key: str,
        reusable: Mapping[str, Any],
        count: int,
    ) -> _VectorShardManifestEntry:
        entry: _VectorShardManifestEntry = {
            "path_key": path_key,
            "digest": str(reusable.get("digest") or ""),
            "count": int(reusable.get("count") or count),
        }
        for field_name in (
            "storage_format",
            "shard_file",
            "sequence_file",
            "vector_file",
        ):
            value = reusable.get(field_name)
            if value:
                entry[field_name] = str(value)
        return entry

    def _load_vector_shard_arrays(
        self,
        *,
        shard_dir: str,
        entry: Mapping[str, Any],
        mmap_mode: Literal["r+", "r", "w+", "c"] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        sequence_file = entry.get("sequence_file")
        vector_file = entry.get("vector_file")
        if sequence_file and vector_file:
            sequence_path = os.path.join(shard_dir, str(sequence_file))
            vector_path = os.path.join(shard_dir, str(vector_file))
            return (
                np.load(
                    sequence_path,
                    allow_pickle=False,
                    mmap_mode=mmap_mode,
                ),
                np.load(
                    vector_path,
                    allow_pickle=False,
                    mmap_mode=mmap_mode,
                ),
            )

        shard_file = entry.get("shard_file")
        if not shard_file:
            return None
        shard_path = os.path.join(shard_dir, str(shard_file))
        with np.load(shard_path, allow_pickle=False) as archive:
            return archive["sequence_nos"], archive["vectors"]

    def _load_vector_shard_sequences(
        self,
        *,
        shard_dir: str,
        entry: Mapping[str, Any],
        mmap_mode: Literal["r+", "r", "w+", "c"] | None = None,
    ) -> np.ndarray | None:
        sequence_file = entry.get("sequence_file")
        if sequence_file:
            return np.load(
                os.path.join(shard_dir, str(sequence_file)),
                allow_pickle=False,
                mmap_mode=mmap_mode,
            )
        arrays = self._load_vector_shard_arrays(
            shard_dir=shard_dir,
            entry=entry,
            mmap_mode=mmap_mode,
        )
        return None if arrays is None else arrays[0]

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
            try:
                sequence_nos = self._load_vector_shard_sequences(
                    shard_dir=shard_dir,
                    entry=entry,
                    mmap_mode="r",
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to read previous vector shard %s: %s",
                    ", ".join(self._vector_entry_files(entry)),
                    exc,
                )
                continue
            if sequence_nos is None:
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

    def _build_delta_shard_plan(
        self,
        *,
        previous_name: str,
        reusable_path_keys: set[str],
    ) -> _IncrementalShardPlan:
        previous_manifest = self._load_vector_manifest(previous_name)
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
        for path_key in sorted(reusable_path_keys):
            if path_key.startswith("__pathless__"):
                continue
            previous_entry = previous_entries.get(path_key)
            sequence_nos = previous_sequences.get(path_key)
            if previous_entry is None or sequence_nos is None:
                continue
            if any(sequence_no in used_sequences for sequence_no in sequence_nos):
                continue
            sequences_by_path[path_key] = list(sequence_nos)
            reusable_entries[path_key] = previous_entry
            used_sequences.update(sequence_nos)

        next_sequence_no = max(max_previous_sequence_no + 1, 0)
        grouped_counts = self._path_row_counts(self.metadata)
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
        current_fingerprint = self._embedding_fingerprint_from_metadata(self.metadata)
        previous_fingerprint = manifest.get("embedding_fingerprint")
        if (
            current_fingerprint is not None
            and previous_fingerprint != current_fingerprint
        ):
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
            "embedding_fingerprint": self._embedding_fingerprint_from_metadata(
                self.metadata
            ),
            "shards": [],
        }
        active_files: set[str] = set()
        for path_key, rows in grouped_rows.items():
            sequence_nos = np.asarray(
                [sequence_no for sequence_no, _ in rows], dtype=np.int64
            )
            row_indexes = np.asarray([sequence_no for sequence_no, _ in rows])
            shard_vectors = vectors[row_indexes].astype(np.float32, copy=False)
            entry, entry_files, _written = self._write_vector_shard_files(
                path_key=path_key,
                shard_dir=shard_dir,
                sequence_nos=sequence_nos,
                vectors=shard_vectors,
                existing=existing_by_path.get(path_key),
            )
            active_files.update(entry_files)
            manifest["shards"].append(entry)

        if existing_manifest is not None:
            for entry in existing_manifest.get("shards", []):
                for file_name in self._vector_entry_files(entry):
                    if file_name in active_files:
                        continue
                    stale_path = os.path.join(shard_dir, file_name)
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
            "embedding_fingerprint": self._embedding_fingerprint_from_metadata(
                self.metadata
            ),
            "shards": [],
        }
        active_files: set[str] = set()
        reused = 0
        written = 0
        for path_key, rows in grouped_rows.items():
            reusable = reusable_entries.get(path_key)
            if reusable is not None:
                entry_files = self._copy_or_link_vector_entry_files(
                    source_dir=previous_shard_dir,
                    target_dir=shard_dir,
                    entry=reusable,
                )
                if entry_files == set(self._vector_entry_files(reusable)):
                    active_files.update(entry_files)
                    manifest["shards"].append(
                        self._reused_vector_manifest_entry(
                            path_key=path_key,
                            reusable=reusable,
                            count=len(rows),
                        )
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
            entry, entry_files, _did_write = self._write_vector_shard_files(
                path_key=path_key,
                shard_dir=shard_dir,
                sequence_nos=sequence_nos,
                vectors=shard_vectors,
            )
            active_files.update(entry_files)
            manifest["shards"].append(entry)
            written += 1

        for entry_name in os.listdir(shard_dir):
            if not entry_name.endswith((".npz", ".npy")) or entry_name in active_files:
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

    def _write_vector_bundle_delta(
        self,
        name: str,
        *,
        previous_name: str,
        vectors: np.ndarray,
        sequences_by_path: dict[str, list[int]],
        reusable_entries: dict[str, _VectorShardManifestEntry],
        previous_vector_count: int,
    ) -> dict[str, int]:
        shard_dir = self._vector_shards_dir(name)
        ensure_dir(shard_dir)
        previous_shard_dir = self._vector_shards_dir(previous_name)
        previous_manifest = self._load_vector_manifest(previous_name)

        grouped_rows: dict[str, list[tuple[int, int, np.ndarray]]] = {}
        grouped_seen: dict[str, int] = {}
        for row_index, meta in enumerate(self.metadata):
            path_key = self._metadata_path_key(meta, row_index)
            sequence_nos = sequences_by_path.get(path_key)
            seen_count = grouped_seen.get(path_key, 0)
            if sequence_nos is None or len(sequence_nos) <= seen_count:
                raise RuntimeError(
                    f"Missing delta vector sequence for path: {path_key}"
                )
            grouped_seen[path_key] = seen_count + 1
            grouped_rows.setdefault(path_key, []).append(
                (sequence_nos[seen_count], row_index, vectors[row_index])
            )

        manifest: _VectorShardManifest = {
            "version": _VECTOR_SHARD_STORAGE_VERSION,
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "index_type": self.index_type,
            "vector_count": sum(len(rows) for rows in sequences_by_path.values()),
            "embedding_fingerprint": (
                self._embedding_fingerprint_from_metadata(self.metadata)
                or (
                    previous_manifest.get("embedding_fingerprint")
                    if previous_manifest is not None
                    else None
                )
            ),
            "shards": [],
        }
        active_files: set[str] = set()
        reused = 0
        written = 0
        rows_reused = 0
        for path_key, reusable in reusable_entries.items():
            entry_files = self._copy_or_link_vector_entry_files(
                source_dir=previous_shard_dir,
                target_dir=shard_dir,
                entry=reusable,
            )
            if entry_files != set(self._vector_entry_files(reusable)):
                continue
            active_files.update(entry_files)
            count = int(reusable.get("count") or len(sequences_by_path[path_key]))
            manifest["shards"].append(
                self._reused_vector_manifest_entry(
                    path_key=path_key,
                    reusable=reusable,
                    count=count,
                )
            )
            reused += 1
            rows_reused += count

        for path_key, rows in grouped_rows.items():
            sequence_nos = np.asarray(
                [sequence_no for sequence_no, _, _ in rows], dtype=np.int64
            )
            row_indexes = np.asarray(
                [row_index for _, row_index, _ in rows], dtype=np.int64
            )
            shard_vectors = vectors[row_indexes].astype(np.float32, copy=False)
            entry, entry_files, _did_write = self._write_vector_shard_files(
                path_key=path_key,
                shard_dir=shard_dir,
                sequence_nos=sequence_nos,
                vectors=shard_vectors,
            )
            active_files.update(entry_files)
            manifest["shards"].append(entry)
            written += 1

        for entry_name in os.listdir(shard_dir):
            if not entry_name.endswith((".npz", ".npy")) or entry_name in active_files:
                continue
            os.remove(os.path.join(shard_dir, entry_name))

        manifest["shards"].sort(key=lambda entry: str(entry.get("path_key") or ""))
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
            "vector_rows_reused": rows_reused,
            "vector_rows_written": len(self.metadata),
            "vector_rows_previous": previous_vector_count,
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
                arrays = self._load_vector_shard_arrays(
                    shard_dir=shard_dir,
                    entry=entry,
                    mmap_mode="r",
                )
                if arrays is None:
                    continue
                sequence_nos, shard_vectors = arrays
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
                for file_name in self._vector_entry_files(entry):
                    shard_path = os.path.join(shard_dir, file_name)
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
            "embedding_fingerprint": self._embedding_fingerprint_from_metadata(
                self.metadata
            ),
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
            "embedding_fingerprint": self._embedding_fingerprint_from_metadata(
                self.metadata
            ),
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

    def _load_metadata_entries_by_path(
        self,
        name: str,
        *,
        path_keys: set[str],
    ) -> dict[str, list[dict[str, Any]]]:
        manifest = self._load_metadata_manifest(name)
        if manifest is None:
            return {}
        shard_dir = self._metadata_shards_dir(name)
        entries_by_path: dict[str, list[dict[str, Any]]] = {}
        for entry in manifest.get("shards", []):
            path_key = str(entry.get("path_key") or "")
            shard_file = entry.get("shard_file")
            if not path_key or path_key not in path_keys or not shard_file:
                continue
            shard_path = os.path.join(shard_dir, str(shard_file))
            try:
                with open(shard_path, "rb") as handle:
                    increment_materialization_boundary(BOUNDARY_PICKLE_LOAD)
                    payload = pickle.load(handle)
            except Exception as exc:
                self.logger.warning(
                    "Failed to read previous metadata shard %s: %s",
                    shard_path,
                    exc,
                )
                continue
            rows = payload.get("entries", []) if isinstance(payload, dict) else []
            if not isinstance(rows, list):
                continue
            entries_by_path[path_key] = [
                dict(cast(dict[str, Any], row))
                for row in rows
                if isinstance(row, dict)
                and isinstance(row.get("sequence_no"), int)
                and isinstance(row.get("payload"), Mapping)
            ]
        return entries_by_path

    @staticmethod
    def _metadata_with_snapshot_id(
        metadata: Mapping[str, Any],
        snapshot_id: str | None,
    ) -> CodeElementMeta:
        row: dict[str, Any] = dict(metadata)
        if snapshot_id is None:
            return cast(CodeElementMeta, row)
        row["snapshot_id"] = snapshot_id
        nested = row.get("metadata")
        if isinstance(nested, Mapping):
            nested_row = dict(cast(Mapping[str, Any], nested))
            nested_row["snapshot_id"] = snapshot_id
            row["metadata"] = nested_row
        return cast(CodeElementMeta, row)

    def _write_metadata_bundle_delta(
        self,
        name: str,
        *,
        previous_name: str,
        sequences_by_path: dict[str, list[int]],
        reusable_path_keys: set[str],
        snapshot_id: str | None,
        previous_vector_count: int,
    ) -> dict[str, int]:
        shard_dir = self._metadata_shards_dir(name)
        ensure_dir(shard_dir)

        previous_entries = self._load_metadata_entries_by_path(
            previous_name,
            path_keys=reusable_path_keys,
        )
        grouped_entries: dict[str, list[dict[str, Any]]] = {}
        rows_reused = 0
        for path_key, rows in previous_entries.items():
            entries: list[dict[str, Any]] = []
            for row in rows:
                sequence_no = row.get("sequence_no")
                payload = row.get("payload")
                if not isinstance(sequence_no, int) or not isinstance(payload, Mapping):
                    continue
                entries.append(
                    {
                        "sequence_no": sequence_no,
                        "payload": self._metadata_with_snapshot_id(
                            payload, snapshot_id
                        ),
                    }
                )
            if entries:
                rows_reused += len(entries)
                grouped_entries[path_key] = entries

        grouped_seen: dict[str, int] = {}
        for row_index, meta in enumerate(self.metadata):
            path_key = self._metadata_path_key(meta, row_index)
            sequence_nos = sequences_by_path.get(path_key)
            seen_count = grouped_seen.get(path_key, 0)
            if sequence_nos is None or len(sequence_nos) <= seen_count:
                raise RuntimeError(
                    f"Missing delta metadata sequence for path: {path_key}"
                )
            grouped_seen[path_key] = seen_count + 1
            grouped_entries.setdefault(path_key, []).append(
                {
                    "sequence_no": sequence_nos[seen_count],
                    "payload": self._metadata_with_snapshot_id(meta, snapshot_id),
                }
            )

        previous_manifest = self._load_metadata_manifest(previous_name)
        manifest: _MetadataShardManifest = {
            "version": _METADATA_SHARD_STORAGE_VERSION,
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "index_type": self.index_type,
            "vector_count": sum(len(entries) for entries in grouped_entries.values()),
            "embedding_fingerprint": (
                self._embedding_fingerprint_from_metadata(self.metadata)
                or (
                    previous_manifest.get("embedding_fingerprint")
                    if previous_manifest is not None
                    else None
                )
            ),
            "shards": [],
        }
        active_files: set[str] = set()
        reused = len(previous_entries)
        written = 0
        for path_key, entries in grouped_entries.items():
            entries.sort(key=lambda row: int(row.get("sequence_no") or 0))
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
            if path_key not in previous_entries:
                written += 1

        for entry_name in os.listdir(shard_dir):
            if not entry_name.endswith(".pkl") or entry_name in active_files:
                continue
            os.remove(os.path.join(shard_dir, entry_name))

        manifest["shards"].sort(key=lambda entry: str(entry.get("path_key") or ""))
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
        return {
            "metadata_shards_reused": reused,
            "metadata_shards_written": written,
            "metadata_rows_reused": rows_reused,
            "metadata_rows_written": len(self.metadata),
            "metadata_rows_previous": previous_vector_count,
        }

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

    def _repo_overview_embedding_fingerprint(
        self,
        entry: Mapping[str, Any],
        *,
        metadata_json: str | None = None,
        include_metadata: bool = True,
    ) -> dict[str, Any] | None:
        direct = entry.get("embedding_fingerprint")
        if isinstance(direct, Mapping):
            return dict(cast(Mapping[str, Any], direct))
        if include_metadata:
            metadata = entry.get("metadata")
            if isinstance(metadata, Mapping):
                value = metadata.get("embedding_fingerprint")
                if isinstance(value, Mapping):
                    return dict(cast(Mapping[str, Any], value))
        if include_metadata and metadata_json:
            decoded = self._deserialize_repo_overview_metadata(metadata_json)
            value = decoded.get("embedding_fingerprint")
            if isinstance(value, Mapping):
                return dict(cast(Mapping[str, Any], value))
        return None

    def _repo_overview_embedding_fingerprint_matches(
        self,
        entry: Mapping[str, Any],
        expected: Mapping[str, Any] | None,
        *,
        metadata_json: str | None = None,
    ) -> bool:
        return self._embedding_fingerprint_matches(
            self._repo_overview_embedding_fingerprint(
                entry,
                metadata_json=metadata_json,
                include_metadata=expected is not None,
            ),
            expected,
        )

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
            fingerprint = self._repo_overview_embedding_fingerprint(
                overview,
                include_metadata=decode_metadata,
            )
            if fingerprint is not None:
                entry["embedding_fingerprint"] = fingerprint
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
                if fingerprint is not None:
                    entry["metadata"].setdefault("embedding_fingerprint", fingerprint)
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
            fingerprint = raw_entry.get("embedding_fingerprint")
            if isinstance(fingerprint, Mapping):
                fingerprint_payload = dict(cast(Mapping[str, Any], fingerprint))
                if fingerprint_payload:
                    entry["embedding_fingerprint"] = fingerprint_payload

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
                embedding_fingerprint = entry.get("embedding_fingerprint")
                if embedding_fingerprint:
                    entry["metadata"].setdefault(
                        "embedding_fingerprint", embedding_fingerprint
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

    def _repo_overview_record_from_entry(
        self,
        repo_name: str,
        entry: _RepoOverviewStoredEntry,
        *,
        materialize_metadata: bool,
    ) -> RepositoryOverviewRecord | None:
        embedding: np.ndarray | None = None
        if "embedding" in entry:
            embedding = self._normalize_repo_overview_embedding(entry.get("embedding"))
            if embedding is None:
                return None

        metadata_json = entry.get("metadata_json")
        if not isinstance(metadata_json, str):
            if materialize_metadata:
                metadata_json = self._serialize_repo_overview_metadata(
                    self._result_metadata_from_overview(entry)
                )
            else:
                metadata_json = "{}"

        fingerprint = self._repo_overview_embedding_fingerprint(
            entry,
            metadata_json=metadata_json,
            include_metadata=materialize_metadata,
        )
        return RepositoryOverviewRecord(
            repo_name=repo_name,
            content=str(entry.get("content", "")),
            metadata_json=metadata_json,
            embedding=embedding,
            embedding_fingerprint=fingerprint,
        )

    def _repo_overview_payload_from_record(
        self,
        record: RepositoryOverviewRecord,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "repo_name": record.repo_name,
            "content": record.content,
            "metadata": self._deserialize_repo_overview_metadata(record.metadata_json),
        }
        if record.embedding is not None:
            payload["embedding"] = record.embedding
        if record.embedding_fingerprint is not None:
            payload["embedding_fingerprint"] = dict(record.embedding_fingerprint)
            payload["metadata"].setdefault(
                "embedding_fingerprint",
                record.embedding_fingerprint,
            )
        return payload

    def _repo_overview_result_metadata_from_record(
        self,
        record: RepositoryOverviewRecord,
    ) -> dict[str, Any]:
        return {
            "repo_name": record.repo_name,
            "type": "repository_overview",
            **self._deserialize_repo_overview_metadata(record.metadata_json),
        }

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
            embedding_fingerprint = self._repo_overview_embedding_fingerprint(
                entry,
                metadata_json=metadata_json,
            )

            manifest_repos[repo_name] = {
                "content": str(entry.get("content", "")),
                "metadata_json": metadata_json,
            }
            if embedding_fingerprint:
                manifest_repos[repo_name]["embedding_fingerprint"] = (
                    embedding_fingerprint
                )
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
