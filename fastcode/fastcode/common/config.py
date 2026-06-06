"""Frozen runtime configuration contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any

ProjectionEdgeWeights = tuple[tuple[str, float], ...]


def _dict_any() -> dict[str, Any]:
    return {}


class StorageBackend(StrEnum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"


class LocalSourceMode(StrEnum):
    IN_PLACE = "in_place"
    COPY = "copy"
    HARDLINK = "hardlink"


class VectorShardStorage(StrEnum):
    COMPRESSED = "compressed"
    NPY = "npy"


class RetrievalBackend(StrEnum):
    PG_HYBRID = "pg_hybrid"
    LOCAL = "local"


class GraphExpansionBackend(StrEnum):
    IR = "ir"
    GRAPH_BUILDER = "graph_builder"


class IndexingLevel(StrEnum):
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    DOCUMENTATION = "documentation"


def _default_indexing_levels() -> tuple[IndexingLevel, ...]:
    return (
        IndexingLevel.FILE,
        IndexingLevel.CLASS,
        IndexingLevel.FUNCTION,
        IndexingLevel.DOCUMENTATION,
    )


def _default_curated_doc_paths() -> tuple[str, ...]:
    return (
        "README*",
        "docs/design/**",
        "docs/research/**",
        "docs/adr/**",
        "docs/rfc/**",
    )


def _default_leiden_resolutions() -> tuple[float, ...]:
    return (1.0,)


def _default_projection_edge_weights() -> ProjectionEdgeWeights:
    return (
        ("contain", 4.0),
        ("defines", 4.0),
        ("owns", 4.0),
        ("call", 2.0),
        ("import", 2.0),
        ("inherit", 2.0),
        ("ref", 2.0),
        ("reference", 2.0),
    )


def _require_minimum(field_name: str, value: int | float, minimum: int | float) -> None:
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")


def _require_maximum(field_name: str, value: int | float, maximum: int | float) -> None:
    if value > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")


def _string_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _float_tuple(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _float_items(values: ProjectionEdgeWeights) -> ProjectionEdgeWeights:
    return tuple((str(key), float(value)) for key, value in values)


def _storage_backend(value: StorageBackend | str) -> StorageBackend:
    if isinstance(value, StorageBackend):
        return value
    return StorageBackend(str(value))


def _local_source_mode(value: LocalSourceMode | str) -> LocalSourceMode:
    if isinstance(value, LocalSourceMode):
        return value
    return LocalSourceMode(str(value))


def _vector_shard_storage(value: VectorShardStorage | str) -> VectorShardStorage:
    if isinstance(value, VectorShardStorage):
        return value
    return VectorShardStorage(str(value))


def _retrieval_backend(value: RetrievalBackend | str) -> RetrievalBackend:
    if isinstance(value, RetrievalBackend):
        return value
    return RetrievalBackend(str(value))


def _graph_expansion_backend(
    value: GraphExpansionBackend | str,
) -> GraphExpansionBackend:
    if isinstance(value, GraphExpansionBackend):
        return value
    return GraphExpansionBackend(str(value))


def _indexing_level(value: IndexingLevel | str) -> IndexingLevel:
    if isinstance(value, IndexingLevel):
        return value
    return IndexingLevel(str(value))


@dataclass(frozen=True)
class StorageConfig:
    backend: StorageBackend = StorageBackend.SQLITE
    postgres_dsn: str = ""
    pool_min: int = 1
    pool_max: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _storage_backend(self.backend))
        _require_minimum("storage.pool_min", self.pool_min, 1)
        _require_minimum("storage.pool_max", self.pool_max, 1)
        if self.pool_max < self.pool_min:
            raise ValueError("storage.pool_max must be >= storage.pool_min")


@dataclass(frozen=True)
class RepositoryConfig:
    clone_depth: int = 1
    max_file_size_mb: int = 5
    backup_directory: str = "./repo_backup"
    local_source_mode: LocalSourceMode = LocalSourceMode.IN_PLACE
    exclude_site_packages: bool = False
    ignore_patterns: tuple[str, ...] = ()
    supported_extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "local_source_mode", _local_source_mode(self.local_source_mode)
        )
        _require_minimum("repository.clone_depth", self.clone_depth, 1)
        _require_minimum("repository.max_file_size_mb", self.max_file_size_mb, 1)
        object.__setattr__(
            self,
            "ignore_patterns",
            tuple(str(pattern) for pattern in self.ignore_patterns),
        )
        object.__setattr__(
            self,
            "supported_extensions",
            tuple(str(extension) for extension in self.supported_extensions),
        )


@dataclass(frozen=True)
class ParserConfig:
    extract_docstrings: bool = True
    extract_comments: bool = True
    extract_imports: bool = True
    compute_complexity: bool = True
    max_function_lines: int = 1000

    def __post_init__(self) -> None:
        _require_minimum("parser.max_function_lines", self.max_function_lines, 1)


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "ollama"
    model: str = "bge-large-en-v1.5"
    ollama_url: str = "http://127.0.0.1:11434/api/embeddings"
    device: str = "cpu"
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        _require_minimum("embedding.batch_size", self.batch_size, 1)
        _require_minimum("embedding.max_seq_length", self.max_seq_length, 1)


@dataclass(frozen=True)
class IndexingConfig:
    levels: tuple[IndexingLevel, ...] = field(default_factory=_default_indexing_levels)
    include_imports: bool = True
    include_class_context: bool = True
    generate_repo_overview: bool = True
    allow_direct_index: bool = False

    def __post_init__(self) -> None:
        levels = tuple(_indexing_level(level) for level in self.levels)
        if not levels:
            raise ValueError("indexing.levels must not be empty")
        object.__setattr__(self, "levels", levels)


@dataclass(frozen=True)
class VectorStoreConfig:
    type: str = "faiss"
    distance_metric: str = "cosine"
    index_type: str = "HNSW"
    shard_storage: VectorShardStorage = VectorShardStorage.COMPRESSED
    ef_construction: int = 200
    ef_search: int = 50
    m: int = 16
    persist_directory: str = "./data/vector_store"
    in_memory: bool = False
    index_scan_cache_ttl: float = 30.0
    index_scan_sample_size: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "shard_storage", _vector_shard_storage(self.shard_storage)
        )
        _require_minimum("vector_store.ef_construction", self.ef_construction, 1)
        _require_minimum("vector_store.ef_search", self.ef_search, 1)
        _require_minimum("vector_store.m", self.m, 1)
        _require_minimum(
            "vector_store.index_scan_cache_ttl", self.index_scan_cache_ttl, 0
        )
        _require_minimum(
            "vector_store.index_scan_sample_size", self.index_scan_sample_size, 1
        )


@dataclass(frozen=True)
class RetrievalConfig:
    semantic_weight: float = 0.6
    keyword_weight: float = 0.3
    graph_weight: float = 0.1
    retrieval_backend: RetrievalBackend = RetrievalBackend.PG_HYBRID
    graph_expansion_backend: GraphExpansionBackend = GraphExpansionBackend.IR
    allow_graph_builder_fallback: bool = True
    min_similarity: float = 0.3
    max_results: int = 5
    diversity_penalty: float = 0.1
    enable_two_stage_retrieval: bool = True
    select_repos_by_overview: bool = True
    repo_selection_method: str = "llm"
    top_repos_to_search: int = 5
    min_repo_similarity: float = 0.1
    max_files_to_search: int = 15
    enable_agency_mode: bool = False
    adaptive_fusion: dict[str, Any] = field(default_factory=_dict_any)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "retrieval_backend", _retrieval_backend(self.retrieval_backend)
        )
        object.__setattr__(
            self,
            "graph_expansion_backend",
            _graph_expansion_backend(self.graph_expansion_backend),
        )
        _require_minimum("retrieval.max_results", self.max_results, 1)
        _require_minimum("retrieval.top_repos_to_search", self.top_repos_to_search, 1)
        _require_minimum("retrieval.max_files_to_search", self.max_files_to_search, 1)


@dataclass(frozen=True)
class GenerationConfig:
    provider: str = "openai"
    model: str = "gpt-4-turbo-preview"
    base_url: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    temperature: float = 0.1
    max_tokens: int = 2000
    max_context_tokens: int = 200000
    reserve_tokens_for_response: int = 10000
    include_file_paths: bool = True
    include_line_numbers: bool = True
    include_related_code: bool = True
    enable_multi_turn: bool = False
    context_rounds: int = 10

    def __post_init__(self) -> None:
        _require_minimum("generation.max_tokens", self.max_tokens, 1)
        _require_minimum("generation.max_context_tokens", self.max_context_tokens, 1)
        _require_minimum(
            "generation.reserve_tokens_for_response",
            self.reserve_tokens_for_response,
            0,
        )
        _require_minimum("generation.context_rounds", self.context_rounds, 0)


@dataclass(frozen=True)
class QueryConfig:
    expand_query: bool = True
    decompose_complex: bool = True
    max_subqueries: int = 3
    extract_keywords: bool = True
    detect_intent: bool = True
    use_llm_enhancement: bool = True
    llm_enhancement_mode: str = "adaptive"
    history_summary_rounds: int = 10
    max_summary_words: int = 250

    def __post_init__(self) -> None:
        _require_minimum("query.max_subqueries", self.max_subqueries, 1)
        _require_minimum("query.history_summary_rounds", self.history_summary_rounds, 0)
        _require_minimum("query.max_summary_words", self.max_summary_words, 1)


@dataclass(frozen=True)
class LadybugGraphConfig:
    enabled: bool = False
    db_path: str = "./data/ladybug/fastcode.lb"
    postgres_attach_dsn: str = ""


@dataclass(frozen=True)
class GraphConfig:
    build_call_graph: bool = True
    build_dependency_graph: bool = True
    build_inheritance_graph: bool = True
    max_depth: int = 5
    ladybug: LadybugGraphConfig = field(default_factory=LadybugGraphConfig)

    def __post_init__(self) -> None:
        _require_minimum("graph.max_depth", self.max_depth, 1)


@dataclass(frozen=True)
class DocsIntegrationConfig:
    enabled: bool = False
    curated_paths: tuple[str, ...] = field(default_factory=_default_curated_doc_paths)
    allow_paths: tuple[str, ...] = ()
    deny_paths: tuple[str, ...] = ()
    chunk_token_size: int = 512
    similarity_threshold: float = 0.5
    chunk_size: int = 420
    chunk_overlap: int = 80
    max_chunk_chars: int = 2400

    def __post_init__(self) -> None:
        object.__setattr__(self, "curated_paths", _string_tuple(self.curated_paths))
        object.__setattr__(self, "allow_paths", _string_tuple(self.allow_paths))
        object.__setattr__(self, "deny_paths", _string_tuple(self.deny_paths))
        _require_minimum("docs_integration.chunk_token_size", self.chunk_token_size, 1)
        _require_minimum(
            "docs_integration.similarity_threshold", self.similarity_threshold, 0
        )
        _require_maximum(
            "docs_integration.similarity_threshold", self.similarity_threshold, 1
        )
        _require_minimum("docs_integration.chunk_size", self.chunk_size, 1)
        _require_minimum("docs_integration.chunk_overlap", self.chunk_overlap, 0)
        _require_minimum("docs_integration.max_chunk_chars", self.max_chunk_chars, 1)
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("docs_integration.chunk_overlap must be < chunk_size")


@dataclass(frozen=True)
class IterativeAgentConfig:
    max_iterations: int = 4
    confidence_threshold: int = 95
    min_confidence_gain: int = 5
    max_total_lines: int = 12000
    temperature_agent: float = 0.2
    max_tokens_agent: int = 8000
    max_elements: int = 100
    max_candidates_display: int = 100

    def __post_init__(self) -> None:
        _require_minimum("agent.iterative.max_iterations", self.max_iterations, 1)
        _require_minimum(
            "agent.iterative.confidence_threshold", self.confidence_threshold, 0
        )
        _require_maximum(
            "agent.iterative.confidence_threshold", self.confidence_threshold, 100
        )
        _require_minimum(
            "agent.iterative.min_confidence_gain", self.min_confidence_gain, 0
        )
        _require_maximum(
            "agent.iterative.min_confidence_gain", self.min_confidence_gain, 100
        )
        _require_minimum("agent.iterative.max_total_lines", self.max_total_lines, 1)
        _require_minimum("agent.iterative.temperature_agent", self.temperature_agent, 0)
        _require_minimum("agent.iterative.max_tokens_agent", self.max_tokens_agent, 1)
        _require_minimum("agent.iterative.max_elements", self.max_elements, 1)
        _require_minimum(
            "agent.iterative.max_candidates_display", self.max_candidates_display, 1
        )


@dataclass(frozen=True)
class AgentConfig:
    iterative: IterativeAgentConfig = field(default_factory=IterativeAgentConfig)


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    backend: str = "disk"
    ttl: int = 3600
    dialogue_ttl: int = 2592000
    max_size_mb: int = 1000
    cache_embeddings: bool = True
    cache_queries: bool = False
    cache_directory: str = "./data/cache"
    redis_host: str = "localhost"
    redis_port: int = 6379

    def __post_init__(self) -> None:
        _require_minimum("cache.ttl", self.ttl, 0)
        _require_minimum("cache.dialogue_ttl", self.dialogue_ttl, 0)
        _require_minimum("cache.max_size_mb", self.max_size_mb, 1)
        _require_minimum("cache.redis_port", self.redis_port, 1)


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool = False
    in_memory_index: bool = False
    disable_cache: bool = False
    disable_persistence: bool = False
    force_reindex: bool = False


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/fastcode.log"
    console: bool = True


@dataclass(frozen=True)
class TerminusConfig:
    endpoint: str = ""
    api_key: str = ""
    timeout_seconds: int = 15

    def __post_init__(self) -> None:
        _require_minimum("terminus.timeout_seconds", self.timeout_seconds, 1)


@dataclass(frozen=True)
class ProjectionConfig:
    postgres_dsn: str = ""
    enable_leiden: bool = True
    hierarchical_leiden_enabled: bool = False
    leiden_resolutions: tuple[float, ...] = field(
        default_factory=_default_leiden_resolutions
    )
    hierarchy_max_levels: int = 4
    hierarchy_max_nodes: int = 12000
    steiner_prune: bool = True
    aggregation_top_members: int = 8
    max_supporting_docs_per_cluster: int = 5
    llm_enabled: bool = True
    llm_timeout_seconds: int = 8
    llm_max_tokens: int = 180
    llm_temperature: float = 0.2
    max_entity_hops: int = 2
    max_query_hops: int = 2
    max_chunk_count: int = 64
    dirty_widen_path_threshold: int = 8
    edge_weights: ProjectionEdgeWeights = field(
        default_factory=_default_projection_edge_weights
    )

    def __post_init__(self) -> None:
        resolutions = _float_tuple(self.leiden_resolutions)
        if not resolutions:
            raise ValueError("projection.leiden_resolutions must not be empty")
        for resolution in resolutions:
            _require_minimum("projection.leiden_resolutions", resolution, 0)
            if resolution == 0:
                raise ValueError("projection.leiden_resolutions must be > 0")
        object.__setattr__(self, "leiden_resolutions", resolutions)
        edge_weights = _float_items(self.edge_weights)
        for _name, weight in edge_weights:
            _require_minimum("projection.edge_weights", weight, 0)
        object.__setattr__(self, "edge_weights", edge_weights)
        _require_minimum(
            "projection.hierarchy_max_levels", self.hierarchy_max_levels, 1
        )
        _require_minimum("projection.hierarchy_max_nodes", self.hierarchy_max_nodes, 1)
        _require_minimum(
            "projection.aggregation_top_members", self.aggregation_top_members, 1
        )
        _require_minimum(
            "projection.max_supporting_docs_per_cluster",
            self.max_supporting_docs_per_cluster,
            1,
        )
        _require_minimum("projection.llm_timeout_seconds", self.llm_timeout_seconds, 1)
        _require_minimum("projection.llm_max_tokens", self.llm_max_tokens, 1)
        _require_minimum("projection.llm_temperature", self.llm_temperature, 0)
        _require_minimum("projection.max_entity_hops", self.max_entity_hops, 0)
        _require_minimum("projection.max_query_hops", self.max_query_hops, 0)
        _require_minimum("projection.max_chunk_count", self.max_chunk_count, 1)
        _require_minimum(
            "projection.dirty_widen_path_threshold",
            self.dirty_widen_path_threshold,
            1,
        )


@dataclass(frozen=True)
class FastCodeConfig:
    repo_root: str = "./repos"
    storage: StorageConfig = field(default_factory=StorageConfig)
    repository: RepositoryConfig = field(default_factory=RepositoryConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    docs_integration: DocsIntegrationConfig = field(
        default_factory=DocsIntegrationConfig
    )
    agent: AgentConfig = field(default_factory=AgentConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    terminus: TerminusConfig = field(default_factory=TerminusConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)

    def with_runtime_overrides(
        self,
        *,
        repo_root: str | None = None,
        in_memory_index: bool | None = None,
        cache_enabled: bool | None = None,
        repository_ignore_patterns: tuple[str, ...] | None = None,
        repository_exclude_site_packages: bool | None = None,
    ) -> FastCodeConfig:
        next_config = self
        if repo_root is not None:
            next_config = replace(next_config, repo_root=repo_root)
        if in_memory_index is not None:
            next_config = replace(
                next_config,
                evaluation=replace(
                    next_config.evaluation, in_memory_index=in_memory_index
                ),
                vector_store=replace(
                    next_config.vector_store, in_memory=in_memory_index
                ),
            )
        if cache_enabled is not None:
            next_config = replace(
                next_config,
                cache=replace(next_config.cache, enabled=cache_enabled),
            )
        if repository_ignore_patterns is not None:
            next_config = replace(
                next_config,
                repository=replace(
                    next_config.repository,
                    ignore_patterns=tuple(repository_ignore_patterns),
                ),
            )
        if repository_exclude_site_packages is not None:
            next_config = replace(
                next_config,
                repository=replace(
                    next_config.repository,
                    exclude_site_packages=repository_exclude_site_packages,
                ),
            )
        return next_config

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        projection_obj = data.get("projection")
        if isinstance(projection_obj, dict):
            projection_obj["edge_weights"] = {
                str(name): float(weight)
                for name, weight in self.projection.edge_weights
            }
        return data
