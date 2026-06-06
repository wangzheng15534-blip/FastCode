"""Inbound DTOs for persistence, cache, vector-store, and projection config."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from ._config_schema_base import _ConfigDTO


def _default_projection_edge_weights() -> dict[str, float]:
    return {
        "contain": 4.0,
        "defines": 4.0,
        "owns": 4.0,
        "call": 2.0,
        "import": 2.0,
        "inherit": 2.0,
        "ref": 2.0,
        "reference": 2.0,
    }


class StorageBackendDTO(StrEnum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"


class VectorShardStorageDTO(StrEnum):
    COMPRESSED = "compressed"
    NPY = "npy"


class StorageConfigDTO(_ConfigDTO):
    backend: StorageBackendDTO = StorageBackendDTO.SQLITE
    postgres_dsn: str = ""
    pool_min: int = Field(default=1, ge=1)
    pool_max: int = Field(default=8, ge=1)


class VectorStoreConfigDTO(_ConfigDTO):
    type: str = "faiss"
    distance_metric: str = "cosine"
    index_type: str = "HNSW"
    shard_storage: VectorShardStorageDTO = VectorShardStorageDTO.COMPRESSED
    ef_construction: int = Field(default=200, ge=1)
    ef_search: int = Field(default=50, ge=1)
    m: int = Field(default=16, ge=1)
    persist_directory: str = "./data/vector_store"
    in_memory: bool = False
    index_scan_cache_ttl: float = Field(default=30.0, ge=0.0)
    index_scan_sample_size: int = Field(default=100, ge=1)


class CacheConfigDTO(_ConfigDTO):
    enabled: bool = True
    backend: str = "disk"
    ttl: int = Field(default=3600, ge=0)
    dialogue_ttl: int = Field(default=2592000, ge=0)
    max_size_mb: int = Field(default=1000, ge=1)
    cache_embeddings: bool = True
    cache_queries: bool = False
    cache_directory: str = "./data/cache"
    redis_host: str = "localhost"
    redis_port: int = Field(default=6379, ge=1)


class ProjectionConfigDTO(_ConfigDTO):
    postgres_dsn: str = ""
    enable_leiden: bool = True
    hierarchical_leiden_enabled: bool = False
    leiden_resolutions: list[float] = Field(default_factory=lambda: [1.0])
    hierarchy_max_levels: int = Field(default=4, ge=1)
    hierarchy_max_nodes: int = Field(default=12000, ge=1)
    steiner_prune: bool = True
    aggregation_top_members: int = Field(default=8, ge=1)
    max_supporting_docs_per_cluster: int = Field(default=5, ge=1)
    llm_enabled: bool = True
    llm_timeout_seconds: int = Field(default=8, ge=1)
    llm_max_tokens: int = Field(default=180, ge=1)
    llm_temperature: float = Field(default=0.2, ge=0.0)
    max_entity_hops: int = Field(default=2, ge=0)
    max_query_hops: int = Field(default=2, ge=0)
    max_chunk_count: int = Field(default=64, ge=1)
    dirty_widen_path_threshold: int = Field(default=8, ge=1)
    edge_weights: dict[str, float] = Field(
        default_factory=_default_projection_edge_weights
    )
