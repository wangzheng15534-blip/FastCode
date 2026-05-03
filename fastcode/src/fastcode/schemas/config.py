"""Configuration boundary models and frozen runtime settings.

Pydantic is used only at the configuration boundary. Runtime code should prefer
``FastCodeConfig`` and its frozen dataclass sections. ``to_dict()`` remains a
compatibility adapter for components that have not yet migrated off dict config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _string_list(values: list[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


class _BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class StorageSettings(_BoundaryModel):
    backend: Literal["sqlite", "postgres"] = "sqlite"
    postgres_dsn: str = ""
    pool_min: int = Field(default=1, ge=1)
    pool_max: int = Field(default=8, ge=1)


@dataclass(frozen=True)
class StorageConfig:
    backend: Literal["sqlite", "postgres"] = "sqlite"
    postgres_dsn: str = ""
    pool_min: int = 1
    pool_max: int = 8

    @classmethod
    def from_settings(cls, settings: StorageSettings) -> StorageConfig:
        pool_min = int(settings.pool_min)
        pool_max = max(int(settings.pool_max), pool_min)
        return cls(
            backend=settings.backend,
            postgres_dsn=settings.postgres_dsn,
            pool_min=pool_min,
            pool_max=pool_max,
        )


class RepositorySettings(_BoundaryModel):
    clone_depth: int = Field(default=1, ge=1)
    max_file_size_mb: int = Field(default=5, ge=1)
    backup_directory: str = "./repo_backup"
    exclude_site_packages: bool = False
    ignore_patterns: list[str] = Field(default_factory=list)
    supported_extensions: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class RepositoryConfig:
    clone_depth: int = 1
    max_file_size_mb: int = 5
    backup_directory: str = "./repo_backup"
    exclude_site_packages: bool = False
    ignore_patterns: tuple[str, ...] = ()
    supported_extensions: tuple[str, ...] = ()

    @classmethod
    def from_settings(cls, settings: RepositorySettings) -> RepositoryConfig:
        return cls(
            clone_depth=int(settings.clone_depth),
            max_file_size_mb=int(settings.max_file_size_mb),
            backup_directory=settings.backup_directory,
            exclude_site_packages=bool(settings.exclude_site_packages),
            ignore_patterns=_string_list(settings.ignore_patterns),
            supported_extensions=_string_list(settings.supported_extensions),
        )


class ParserSettings(_BoundaryModel):
    extract_docstrings: bool = True
    extract_comments: bool = True
    extract_imports: bool = True
    compute_complexity: bool = True
    max_function_lines: int = Field(default=1000, ge=1)


@dataclass(frozen=True)
class ParserConfig:
    extract_docstrings: bool = True
    extract_comments: bool = True
    extract_imports: bool = True
    compute_complexity: bool = True
    max_function_lines: int = 1000

    @classmethod
    def from_settings(cls, settings: ParserSettings) -> ParserConfig:
        return cls(
            extract_docstrings=bool(settings.extract_docstrings),
            extract_comments=bool(settings.extract_comments),
            extract_imports=bool(settings.extract_imports),
            compute_complexity=bool(settings.compute_complexity),
            max_function_lines=int(settings.max_function_lines),
        )


class EmbeddingSettings(_BoundaryModel):
    provider: str = "ollama"
    model: str = "bge-large-en-v1.5"
    ollama_url: str = "http://127.0.0.1:11434/api/embeddings"
    device: str = "cpu"
    batch_size: int = Field(default=32, ge=1)
    max_seq_length: int = Field(default=512, ge=1)
    normalize_embeddings: bool = True


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "ollama"
    model: str = "bge-large-en-v1.5"
    ollama_url: str = "http://127.0.0.1:11434/api/embeddings"
    device: str = "cpu"
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True

    @classmethod
    def from_settings(cls, settings: EmbeddingSettings) -> EmbeddingConfig:
        return cls(
            provider=settings.provider,
            model=settings.model,
            ollama_url=settings.ollama_url,
            device=settings.device,
            batch_size=int(settings.batch_size),
            max_seq_length=int(settings.max_seq_length),
            normalize_embeddings=bool(settings.normalize_embeddings),
        )


class VectorStoreSettings(_BoundaryModel):
    type: str = "faiss"
    distance_metric: str = "cosine"
    index_type: str = "HNSW"
    ef_construction: int = Field(default=200, ge=1)
    ef_search: int = Field(default=50, ge=1)
    m: int = Field(default=16, ge=1)
    persist_directory: str = "./data/vector_store"
    in_memory: bool = False
    index_scan_cache_ttl: float = Field(default=30.0, ge=0.0)
    index_scan_sample_size: int = Field(default=100, ge=1)


@dataclass(frozen=True)
class VectorStoreConfig:
    type: str = "faiss"
    distance_metric: str = "cosine"
    index_type: str = "HNSW"
    ef_construction: int = 200
    ef_search: int = 50
    m: int = 16
    persist_directory: str = "./data/vector_store"
    in_memory: bool = False
    index_scan_cache_ttl: float = 30.0
    index_scan_sample_size: int = 100

    @classmethod
    def from_settings(cls, settings: VectorStoreSettings) -> VectorStoreConfig:
        return cls(
            type=settings.type,
            distance_metric=settings.distance_metric,
            index_type=settings.index_type,
            ef_construction=int(settings.ef_construction),
            ef_search=int(settings.ef_search),
            m=int(settings.m),
            persist_directory=settings.persist_directory,
            in_memory=bool(settings.in_memory),
            index_scan_cache_ttl=float(settings.index_scan_cache_ttl),
            index_scan_sample_size=int(settings.index_scan_sample_size),
        )


class RetrievalSettings(_BoundaryModel):
    semantic_weight: float = 0.6
    keyword_weight: float = 0.3
    graph_weight: float = 0.1
    backend: str = "pg_hybrid"
    graph_backend: str = "ir"
    allow_legacy_graph_fallback: bool = True
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
    adaptive_fusion: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalConfig:
    semantic_weight: float = 0.6
    keyword_weight: float = 0.3
    graph_weight: float = 0.1
    backend: str = "pg_hybrid"
    graph_backend: str = "ir"
    allow_legacy_graph_fallback: bool = True
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
    adaptive_fusion: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: RetrievalSettings) -> RetrievalConfig:
        return cls(
            semantic_weight=float(settings.semantic_weight),
            keyword_weight=float(settings.keyword_weight),
            graph_weight=float(settings.graph_weight),
            backend=settings.backend,
            graph_backend=settings.graph_backend,
            allow_legacy_graph_fallback=bool(settings.allow_legacy_graph_fallback),
            min_similarity=float(settings.min_similarity),
            max_results=int(settings.max_results),
            diversity_penalty=float(settings.diversity_penalty),
            enable_two_stage_retrieval=bool(settings.enable_two_stage_retrieval),
            select_repos_by_overview=bool(settings.select_repos_by_overview),
            repo_selection_method=settings.repo_selection_method,
            top_repos_to_search=int(settings.top_repos_to_search),
            min_repo_similarity=float(settings.min_repo_similarity),
            max_files_to_search=int(settings.max_files_to_search),
            enable_agency_mode=bool(settings.enable_agency_mode),
            adaptive_fusion=dict(settings.adaptive_fusion),
        )


class GenerationSettings(_BoundaryModel):
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

    @classmethod
    def from_settings(cls, settings: GenerationSettings) -> GenerationConfig:
        return cls(
            provider=settings.provider,
            model=settings.model,
            base_url=settings.base_url,
            openai_api_key=settings.openai_api_key,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=float(settings.temperature),
            max_tokens=int(settings.max_tokens),
            max_context_tokens=int(settings.max_context_tokens),
            reserve_tokens_for_response=int(settings.reserve_tokens_for_response),
            include_file_paths=bool(settings.include_file_paths),
            include_line_numbers=bool(settings.include_line_numbers),
            include_related_code=bool(settings.include_related_code),
            enable_multi_turn=bool(settings.enable_multi_turn),
            context_rounds=int(settings.context_rounds),
        )


class QuerySettings(_BoundaryModel):
    expand_query: bool = True
    decompose_complex: bool = True
    max_subqueries: int = Field(default=3, ge=1)
    extract_keywords: bool = True
    detect_intent: bool = True
    use_llm_enhancement: bool = True
    llm_enhancement_mode: str = "adaptive"
    history_summary_rounds: int = Field(default=10, ge=0)
    max_summary_words: int = Field(default=250, ge=1)


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

    @classmethod
    def from_settings(cls, settings: QuerySettings) -> QueryConfig:
        return cls(
            expand_query=bool(settings.expand_query),
            decompose_complex=bool(settings.decompose_complex),
            max_subqueries=int(settings.max_subqueries),
            extract_keywords=bool(settings.extract_keywords),
            detect_intent=bool(settings.detect_intent),
            use_llm_enhancement=bool(settings.use_llm_enhancement),
            llm_enhancement_mode=settings.llm_enhancement_mode,
            history_summary_rounds=int(settings.history_summary_rounds),
            max_summary_words=int(settings.max_summary_words),
        )


class CacheSettings(_BoundaryModel):
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

    @classmethod
    def from_settings(cls, settings: CacheSettings) -> CacheConfig:
        return cls(
            enabled=bool(settings.enabled),
            backend=settings.backend,
            ttl=int(settings.ttl),
            dialogue_ttl=int(settings.dialogue_ttl),
            max_size_mb=int(settings.max_size_mb),
            cache_embeddings=bool(settings.cache_embeddings),
            cache_queries=bool(settings.cache_queries),
            cache_directory=settings.cache_directory,
            redis_host=settings.redis_host,
            redis_port=int(settings.redis_port),
        )


class EvaluationSettings(_BoundaryModel):
    enabled: bool = False
    in_memory_index: bool = False
    disable_cache: bool = False
    disable_persistence: bool = False
    force_reindex: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool = False
    in_memory_index: bool = False
    disable_cache: bool = False
    disable_persistence: bool = False
    force_reindex: bool = False

    @classmethod
    def from_settings(cls, settings: EvaluationSettings) -> EvaluationConfig:
        return cls(
            enabled=bool(settings.enabled),
            in_memory_index=bool(settings.in_memory_index),
            disable_cache=bool(settings.disable_cache),
            disable_persistence=bool(settings.disable_persistence),
            force_reindex=bool(settings.force_reindex),
        )


class LoggingSettings(_BoundaryModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/fastcode.log"
    console: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/fastcode.log"
    console: bool = True

    @classmethod
    def from_settings(cls, settings: LoggingSettings) -> LoggingConfig:
        return cls(
            level=settings.level,
            format=settings.format,
            file=settings.file,
            console=bool(settings.console),
        )


class FastCodeSettings(_BoundaryModel):
    repo_root: str = "./repos"
    storage: StorageSettings = Field(default_factory=StorageSettings)
    repository: RepositorySettings = Field(default_factory=RepositorySettings)
    parser: ParserSettings = Field(default_factory=ParserSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    indexing: dict[str, Any] = Field(default_factory=dict)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    query: QuerySettings = Field(default_factory=QuerySettings)
    graph: dict[str, Any] = Field(default_factory=dict)
    docs_integration: dict[str, Any] = Field(default_factory=dict)
    agent: dict[str, Any] = Field(default_factory=dict)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    terminus: dict[str, Any] = Field(default_factory=dict)
    projection: StorageSettings = Field(default_factory=StorageSettings)


@dataclass(frozen=True)
class FastCodeConfig:
    repo_root: str = "./repos"
    storage: StorageConfig = field(default_factory=StorageConfig)
    repository: RepositoryConfig = field(default_factory=RepositoryConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    indexing: dict[str, Any] = field(default_factory=dict)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    graph: dict[str, Any] = field(default_factory=dict)
    docs_integration: dict[str, Any] = field(default_factory=dict)
    agent: dict[str, Any] = field(default_factory=dict)
    cache: CacheConfig = field(default_factory=CacheConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    terminus: dict[str, Any] = field(default_factory=dict)
    projection: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> FastCodeConfig:
        settings = FastCodeSettings.model_validate(raw or {})
        return cls(
            repo_root=settings.repo_root,
            storage=StorageConfig.from_settings(settings.storage),
            repository=RepositoryConfig.from_settings(settings.repository),
            parser=ParserConfig.from_settings(settings.parser),
            embedding=EmbeddingConfig.from_settings(settings.embedding),
            indexing=dict(settings.indexing),
            vector_store=VectorStoreConfig.from_settings(settings.vector_store),
            retrieval=RetrievalConfig.from_settings(settings.retrieval),
            generation=GenerationConfig.from_settings(settings.generation),
            query=QueryConfig.from_settings(settings.query),
            graph=dict(settings.graph),
            docs_integration=dict(settings.docs_integration),
            agent=dict(settings.agent),
            cache=CacheConfig.from_settings(settings.cache),
            evaluation=EvaluationConfig.from_settings(settings.evaluation),
            logging=LoggingConfig.from_settings(settings.logging),
            terminus=dict(settings.terminus),
            projection=StorageConfig.from_settings(settings.projection),
        )

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
        return asdict(self)


def config_from_mapping(raw: dict[str, Any] | None) -> FastCodeConfig:
    """Validate raw settings and return frozen runtime config."""
    return FastCodeConfig.from_mapping(raw)


def config_to_dict(config: FastCodeConfig) -> dict[str, Any]:
    """Explicit compatibility adapter for legacy dict-based consumers."""
    return config.to_dict()
