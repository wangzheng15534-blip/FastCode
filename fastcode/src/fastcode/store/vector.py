"""
Vector Store - Store and retrieve code embeddings
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import TYPE_CHECKING, Any, TypedDict, cast

import faiss
import numpy as np

if TYPE_CHECKING:
    from ..ir.element import CodeElementMeta

from ..utils import ensure_dir

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

        self.persist_dir: str = self.vector_config.get(
            "persist_directory", "./data/vector_store"
        )
        self.distance_metric: str = self.vector_config.get("distance_metric", "cosine")
        self.index_type: str = self.vector_config.get("index_type", "HNSW")

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
            self.index = index

        # Flat index for exact search (slower but more accurate)
        elif self.distance_metric == "cosine":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product
        else:
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance

        self.metadata: list[CodeElementMeta] = []
        self.logger.info(
            f"Initialized {self.index_type} index with {self.distance_metric} distance"
        )

    def add_vectors(self, vectors: np.ndarray, metadata: list[CodeElementMeta]) -> None:
        """
        Add vectors to the store

        Args:
            vectors: Array of embedding vectors (N x dimension)
            metadata: List of metadata dictionaries for each vector
        """
        if self.index is None:
            raise RuntimeError("Vector store not initialized")

        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")

        # Ensure vectors are float32
        vectors = vectors.astype(np.float32)

        # Normalize if using cosine similarity
        if self.distance_metric == "cosine":
            faiss.normalize_L2(vectors)

        # Add to index
        self.index.add(vectors)
        self.metadata.extend(metadata)

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
        if self.index is None or len(self.metadata) == 0:
            return []

        # Ensure query is float32 and 2D
        query_vector = query_vector.astype(np.float32).reshape(1, -1)

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

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] == 0:
            return []

        repo_names: list[str] = []
        overview_payloads: list[_RepoOverviewStoredEntry] = []
        embedding_rows: list[np.ndarray] = []
        for repo_name, overview_data in overviews.items():
            raw_embedding = overview_data.get("embedding")
            if raw_embedding is None:
                continue
            try:
                embedding = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                continue
            if embedding.size != query.shape[1]:
                continue
            if not np.isfinite(embedding).all():
                embedding = np.nan_to_num(
                    embedding, copy=False, nan=0.0, posinf=0.0, neginf=0.0
                )
            repo_names.append(repo_name)
            overview_payloads.append(overview_data)
            embedding_rows.append(embedding)

        if not embedding_rows:
            return []

        matrix = np.vstack(embedding_rows).astype(np.float32, copy=False)
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
        if self.index is None or len(self.metadata) == 0:
            return [[] for _ in range(len(query_vectors))]

        # Ensure float32
        query_vectors = query_vectors.astype(np.float32)

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

        if self.index is None:
            self.logger.warning("No index to save")
            return

        index_path = os.path.join(self.persist_dir, f"{name}.faiss")
        metadata_path = os.path.join(self.persist_dir, f"{name}_metadata.pkl")

        # Save FAISS index
        faiss.write_index(self.index, index_path)

        # Save metadata
        with open(metadata_path, "wb") as f:
            pickle.dump(
                {
                    "metadata": self.metadata,
                    "dimension": self.dimension,
                    "distance_metric": self.distance_metric,
                    "index_type": self.index_type,
                },
                f,
            )

        # Invalidate cache since we just modified the indexes
        self.invalidate_scan_cache()

        self.logger.info(f"Saved vector store to {self.persist_dir}")

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

        index_path = os.path.join(self.persist_dir, f"{name}.faiss")
        metadata_path = os.path.join(self.persist_dir, f"{name}_metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            self.logger.warning(f"Index files not found in {self.persist_dir}")
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)

            # Load metadata
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.metadata = cast(list[CodeElementMeta], data["metadata"])
                self.dimension = data["dimension"]
                self.distance_metric = data.get("distance_metric", "cosine")
                self.index_type = data.get("index_type", "HNSW")

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

        index_path = os.path.join(self.persist_dir, f"{index_name}.faiss")
        metadata_path = os.path.join(self.persist_dir, f"{index_name}_metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            self.logger.warning(f"Index files not found for {index_name}")
            return False

        try:
            # Load the other index
            other_index = faiss.read_index(index_path)

            # Load metadata
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                other_metadata = cast(list[CodeElementMeta], data["metadata"])
                other_dimension = data["dimension"]

            # Verify dimensions match
            if self.dimension and self.dimension != other_dimension:
                self.logger.error(
                    f"Dimension mismatch: {self.dimension} vs {other_dimension}"
                )
                return False

            # Initialize if needed
            if self.index is None:
                self.initialize(other_dimension)

            # Reconstruct vectors from the FAISS index
            # For flat indices, we can access vectors directly
            # For HNSW, we need to reconstruct
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

        for file in os.listdir(self.persist_dir):
            if file.endswith(".faiss"):
                repo_name = file.replace(".faiss", "")
                metadata_file = os.path.join(
                    self.persist_dir, f"{repo_name}_metadata.pkl"
                )

                if os.path.exists(metadata_file):
                    try:
                        # Get file sizes (fast operation)
                        index_path = os.path.join(self.persist_dir, file)
                        file_size = os.path.getsize(index_path)
                        metadata_size = os.path.getsize(metadata_file)
                        total_size_mb = (file_size + metadata_size) / (1024 * 1024)

                        # Optimized: Only read first chunk of metadata for basic info
                        # This avoids loading potentially huge metadata files
                        element_count = 0
                        file_count = 0
                        repo_url = "N/A"

                        with open(metadata_file, "rb") as f:
                            try:
                                data = pickle.load(f)
                                metadata_list = data.get("metadata", [])
                                element_count = len(metadata_list)

                                # Sample first few entries to get URL and estimate file count
                                # (much faster than iterating through all)
                                sample_size = min(
                                    self._index_scan_sample_size, len(metadata_list)
                                )
                                seen_files = set()

                                for i in range(sample_size):
                                    meta = metadata_list[i]
                                    file_path = meta.get("file_path")
                                    if file_path:
                                        seen_files.add(file_path)
                                    if not repo_url or repo_url == "N/A":
                                        repo_url = meta.get("repo_url", "N/A")

                                # Estimate total file count based on sample
                                if sample_size > 0 and sample_size < len(metadata_list):
                                    file_count = int(
                                        len(seen_files)
                                        * (len(metadata_list) / sample_size)
                                    )
                                else:
                                    file_count = len(seen_files)

                            except Exception as load_error:
                                self.logger.warning(
                                    f"Failed to parse metadata for {repo_name}: {load_error}"
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
                        self.logger.warning(
                            f"Failed to read metadata for {repo_name}: {e}"
                        )
                        # Still add it with minimal info
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

    def _repo_overview_manifest_path(self) -> str:
        return os.path.join(self.persist_dir, _REPO_OVERVIEW_MANIFEST_FILENAME)

    def _repo_overview_embeddings_path(self) -> str:
        return os.path.join(self.persist_dir, _REPO_OVERVIEW_EMBEDDINGS_FILENAME)

    def _legacy_repo_overview_path(self) -> str:
        return os.path.join(self.persist_dir, _LEGACY_REPO_OVERVIEW_FILENAME)

    @staticmethod
    def _normalize_repo_overview_embedding(embedding: Any) -> np.ndarray | None:
        try:
            normalized = np.asarray(embedding, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if normalized.size == 0:
            return None
        if not np.isfinite(normalized).all():
            normalized = np.nan_to_num(
                normalized, copy=False, nan=0.0, posinf=0.0, neginf=0.0
            )
        return normalized.astype(np.float32, copy=False)

    @staticmethod
    def _serialize_repo_overview_metadata(metadata: Any) -> str:
        safe_metadata = metadata if isinstance(metadata, dict) else {}
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
