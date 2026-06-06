"""Retrieval storage capability ports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from fastcode.common.types import CodeElementMeta


class VectorSearchStore(Protocol):
    """Vector-backed retrieval capability consumed by query workflows."""

    def initialize(self, dimension: int) -> None: ...

    def search(
        self,
        query_vector: Any,
        k: int = 10,
        min_score: float | None = None,
        repo_filter: list[str] | None = None,
        element_type_filter: str | None = None,
        query_embedding_fingerprint: Mapping[str, Any] | None = None,
    ) -> list[tuple[CodeElementMeta, float]]: ...

    def load_repo_overviews(
        self, include_embeddings: bool = True
    ) -> dict[str, dict[str, Any]]: ...

    def search_repository_overviews(
        self,
        query_vector: Any,
        k: int = 5,
        min_score: float | None = None,
        query_embedding_fingerprint: Mapping[str, Any] | None = None,
    ) -> list[tuple[dict[str, Any], float]]: ...

    def get_count(self) -> int: ...

    def get_repository_names(self) -> list[str]: ...

    def clear(self) -> None: ...

    def merge_from_index(self, index_name: str) -> bool: ...


class VectorSearchStoreFactory(Protocol):
    """Factory supplied by composition roots for temporary vector stores."""

    def create_vector_search_store(self) -> VectorSearchStore: ...


class HybridRetrievalStore(Protocol):
    """External hybrid retrieval capability used as a query fast path."""

    def is_active(self) -> bool: ...

    def semantic_search(
        self,
        snapshot_id: str,
        query_embedding: Any,
        *,
        repo_filter: list[str] | None = None,
        element_types: list[str] | None = None,
        top_k: int = 20,
        query_embedding_fingerprint: Mapping[str, Any] | None = None,
    ) -> list[tuple[dict[str, Any], float]]: ...

    def keyword_search(
        self,
        snapshot_id: str,
        query: str,
        *,
        repo_filter: list[str] | None = None,
        element_types: list[str] | None = None,
        top_k: int = 20,
    ) -> list[tuple[dict[str, Any], float]]: ...
