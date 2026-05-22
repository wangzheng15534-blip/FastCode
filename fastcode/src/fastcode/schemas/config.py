"""Inbound configuration schemas and DTOs.

Pydantic is used only at the external configuration boundary. Inbound mappers
translate these validated DTOs into frozen runtime contracts.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _dict_any() -> dict[str, Any]:
    return {}


def _default_indexing_levels() -> list[IndexingLevelDTO]:
    return [
        IndexingLevelDTO.FILE,
        IndexingLevelDTO.CLASS,
        IndexingLevelDTO.FUNCTION,
        IndexingLevelDTO.DOCUMENTATION,
    ]


def _default_curated_doc_paths() -> list[str]:
    return [
        "README*",
        "docs/design/**",
        "docs/research/**",
        "docs/adr/**",
        "docs/rfc/**",
    ]


def _default_leiden_resolutions() -> list[float]:
    return [1.0]


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


class _ConfigDTO(BaseModel):
    model_config = ConfigDict(extra="allow")


class StorageBackendDTO(StrEnum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"


class LocalSourceModeDTO(StrEnum):
    IN_PLACE = "in_place"
    COPY = "copy"
    HARDLINK = "hardlink"


class VectorShardStorageDTO(StrEnum):
    COMPRESSED = "compressed"
    NPY = "npy"


class RetrievalBackendDTO(StrEnum):
    PG_HYBRID = "pg_hybrid"
    LOCAL = "local"


class GraphExpansionBackendDTO(StrEnum):
    IR = "ir"
    GRAPH_BUILDER = "graph_builder"


class IndexingLevelDTO(StrEnum):
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    DOCUMENTATION = "documentation"


class StorageConfigDTO(_ConfigDTO):
    backend: StorageBackendDTO = StorageBackendDTO.SQLITE
    postgres_dsn: str = ""
    pool_min: int = Field(default=1, ge=1)
    pool_max: int = Field(default=8, ge=1)


class RepositoryConfigDTO(_ConfigDTO):
    clone_depth: int = Field(default=1, ge=1)
    max_file_size_mb: int = Field(default=5, ge=1)
    backup_directory: str = "./repo_backup"
    local_source_mode: LocalSourceModeDTO = LocalSourceModeDTO.IN_PLACE
    exclude_site_packages: bool = False
    ignore_patterns: list[str] = Field(default_factory=list)
    supported_extensions: list[str] = Field(default_factory=list)


class ParserConfigDTO(_ConfigDTO):
    extract_docstrings: bool = True
    extract_comments: bool = True
    extract_imports: bool = True
    compute_complexity: bool = True
    max_function_lines: int = Field(default=1000, ge=1)


class EmbeddingConfigDTO(_ConfigDTO):
    provider: str = "ollama"
    model: str = "bge-large-en-v1.5"
    ollama_url: str = "http://127.0.0.1:11434/api/embeddings"
    device: str = "cpu"
    batch_size: int = Field(default=32, ge=1)
    max_seq_length: int = Field(default=512, ge=1)
    normalize_embeddings: bool = True


class IndexingConfigDTO(_ConfigDTO):
    levels: list[IndexingLevelDTO] = Field(default_factory=_default_indexing_levels)
    include_imports: bool = True
    include_class_context: bool = True
    generate_repo_overview: bool = True
    allow_direct_index: bool = False


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


class RetrievalConfigDTO(_ConfigDTO):
    semantic_weight: float = 0.6
    keyword_weight: float = 0.3
    graph_weight: float = 0.1
    retrieval_backend: RetrievalBackendDTO = RetrievalBackendDTO.PG_HYBRID
    graph_expansion_backend: GraphExpansionBackendDTO = GraphExpansionBackendDTO.IR
    allow_graph_builder_fallback: bool = True
    min_similarity: float = 0.3
    max_results: int = Field(default=5, ge=1)
    diversity_penalty: float = 0.1
    enable_two_stage_retrieval: bool = True
    select_repos_by_overview: bool = True
    repo_selection_method: str = "llm"
    top_repos_to_search: int = Field(default=5, ge=1)
    min_repo_similarity: float = 0.1
    max_files_to_search: int = Field(default=15, ge=1)
    enable_agency_mode: bool = False
    adaptive_fusion: dict[str, Any] = Field(default_factory=_dict_any)


class GenerationConfigDTO(_ConfigDTO):
    provider: str = "openai"
    model: str = "gpt-4-turbo-preview"
    base_url: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    temperature: float = 0.1
    max_tokens: int = Field(default=2000, ge=1)
    max_context_tokens: int = Field(default=200000, ge=1)
    reserve_tokens_for_response: int = Field(default=10000, ge=0)
    include_file_paths: bool = True
    include_line_numbers: bool = True
    include_related_code: bool = True
    enable_multi_turn: bool = False
    context_rounds: int = Field(default=10, ge=0)


class QueryConfigDTO(_ConfigDTO):
    expand_query: bool = True
    decompose_complex: bool = True
    max_subqueries: int = Field(default=3, ge=1)
    extract_keywords: bool = True
    detect_intent: bool = True
    use_llm_enhancement: bool = True
    llm_enhancement_mode: str = "adaptive"
    history_summary_rounds: int = Field(default=10, ge=0)
    max_summary_words: int = Field(default=250, ge=1)


class LadybugGraphConfigDTO(_ConfigDTO):
    enabled: bool = False
    db_path: str = "./data/ladybug/fastcode.lb"
    postgres_attach_dsn: str = ""


class GraphConfigDTO(_ConfigDTO):
    build_call_graph: bool = True
    build_dependency_graph: bool = True
    build_inheritance_graph: bool = True
    max_depth: int = Field(default=5, ge=1)
    ladybug: LadybugGraphConfigDTO = Field(default_factory=LadybugGraphConfigDTO)


class DocsIntegrationConfigDTO(_ConfigDTO):
    enabled: bool = False
    curated_paths: list[str] = Field(default_factory=_default_curated_doc_paths)
    allow_paths: list[str] = Field(default_factory=list)
    deny_paths: list[str] = Field(default_factory=list)
    chunk_token_size: int = Field(default=512, ge=1)
    similarity_threshold: float = Field(default=0.5, ge=0.0)
    chunk_size: int = Field(default=420, ge=1)
    chunk_overlap: int = Field(default=80, ge=0)
    max_chunk_chars: int = Field(default=2400, ge=1)


class IterativeAgentConfigDTO(_ConfigDTO):
    max_iterations: int = Field(default=4, ge=1)
    confidence_threshold: int = Field(default=95, ge=0)
    min_confidence_gain: int = Field(default=5, ge=0)
    max_total_lines: int = Field(default=12000, ge=1)
    temperature_agent: float = 0.2
    max_tokens_agent: int = Field(default=8000, ge=1)
    max_elements: int = Field(default=100, ge=1)
    max_candidates_display: int = Field(default=100, ge=1)


class AgentConfigDTO(_ConfigDTO):
    iterative: IterativeAgentConfigDTO = Field(default_factory=IterativeAgentConfigDTO)


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


class EvaluationConfigDTO(_ConfigDTO):
    enabled: bool = False
    in_memory_index: bool = False
    disable_cache: bool = False
    disable_persistence: bool = False
    force_reindex: bool = False


class LoggingConfigDTO(_ConfigDTO):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/fastcode.log"
    console: bool = True


class TerminusConfigDTO(_ConfigDTO):
    endpoint: str = ""
    api_key: str = ""
    timeout_seconds: int = Field(default=15, ge=1)


class ProjectionConfigDTO(_ConfigDTO):
    postgres_dsn: str = ""
    enable_leiden: bool = True
    hierarchical_leiden_enabled: bool = False
    leiden_resolutions: list[float] = Field(default_factory=_default_leiden_resolutions)
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


class FastCodeConfigDTO(_ConfigDTO):
    repo_root: str = "./repos"
    storage: StorageConfigDTO = Field(default_factory=StorageConfigDTO)
    repository: RepositoryConfigDTO = Field(default_factory=RepositoryConfigDTO)
    parser: ParserConfigDTO = Field(default_factory=ParserConfigDTO)
    embedding: EmbeddingConfigDTO = Field(default_factory=EmbeddingConfigDTO)
    indexing: IndexingConfigDTO = Field(default_factory=IndexingConfigDTO)
    vector_store: VectorStoreConfigDTO = Field(default_factory=VectorStoreConfigDTO)
    retrieval: RetrievalConfigDTO = Field(default_factory=RetrievalConfigDTO)
    generation: GenerationConfigDTO = Field(default_factory=GenerationConfigDTO)
    query: QueryConfigDTO = Field(default_factory=QueryConfigDTO)
    graph: GraphConfigDTO = Field(default_factory=GraphConfigDTO)
    docs_integration: DocsIntegrationConfigDTO = Field(
        default_factory=DocsIntegrationConfigDTO
    )
    agent: AgentConfigDTO = Field(default_factory=AgentConfigDTO)
    cache: CacheConfigDTO = Field(default_factory=CacheConfigDTO)
    evaluation: EvaluationConfigDTO = Field(default_factory=EvaluationConfigDTO)
    logging: LoggingConfigDTO = Field(default_factory=LoggingConfigDTO)
    terminus: TerminusConfigDTO = Field(default_factory=TerminusConfigDTO)
    projection: ProjectionConfigDTO = Field(default_factory=ProjectionConfigDTO)
