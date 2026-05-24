"""Map inbound config DTOs into frozen runtime config contracts."""

from __future__ import annotations

from typing import Any

import fastcode.runtime.config as runtime_config
from fastcode.inbound.config_schema import (
    AgentConfigDTO,
    CacheConfigDTO,
    DocsIntegrationConfigDTO,
    EmbeddingConfigDTO,
    EvaluationConfigDTO,
    FastCodeConfigDTO,
    GenerationConfigDTO,
    GraphConfigDTO,
    IndexingConfigDTO,
    LoggingConfigDTO,
    ParserConfigDTO,
    ProjectionConfigDTO,
    QueryConfigDTO,
    RepositoryConfigDTO,
    RetrievalConfigDTO,
    StorageConfigDTO,
    TerminusConfigDTO,
    VectorShardStorageDTO,
    VectorStoreConfigDTO,
)


def _string_list(values: list[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _float_list(values: list[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _float_items(values: dict[str, float]) -> runtime_config.ProjectionEdgeWeights:
    return tuple((str(key), float(value)) for key, value in values.items())


def _storage_config(dto: StorageConfigDTO) -> runtime_config.StorageConfig:
    pool_min = int(dto.pool_min)
    pool_max = max(int(dto.pool_max), pool_min)
    return runtime_config.StorageConfig(
        backend=runtime_config.StorageBackend(dto.backend.value),
        postgres_dsn=dto.postgres_dsn,
        pool_min=pool_min,
        pool_max=pool_max,
    )


def _repository_config(
    dto: RepositoryConfigDTO,
) -> runtime_config.RepositoryConfig:
    return runtime_config.RepositoryConfig(
        clone_depth=int(dto.clone_depth),
        max_file_size_mb=int(dto.max_file_size_mb),
        backup_directory=dto.backup_directory,
        local_source_mode=runtime_config.LocalSourceMode(dto.local_source_mode.value),
        exclude_site_packages=bool(dto.exclude_site_packages),
        ignore_patterns=_string_list(dto.ignore_patterns),
        supported_extensions=_string_list(dto.supported_extensions),
    )


def _parser_config(dto: ParserConfigDTO) -> runtime_config.ParserConfig:
    return runtime_config.ParserConfig(
        extract_docstrings=bool(dto.extract_docstrings),
        extract_comments=bool(dto.extract_comments),
        extract_imports=bool(dto.extract_imports),
        compute_complexity=bool(dto.compute_complexity),
        max_function_lines=int(dto.max_function_lines),
    )


def _embedding_config(dto: EmbeddingConfigDTO) -> runtime_config.EmbeddingConfig:
    return runtime_config.EmbeddingConfig(
        provider=dto.provider,
        model=dto.model,
        ollama_url=dto.ollama_url,
        device=dto.device,
        batch_size=int(dto.batch_size),
        max_seq_length=int(dto.max_seq_length),
        normalize_embeddings=bool(dto.normalize_embeddings),
    )


def _indexing_config(dto: IndexingConfigDTO) -> runtime_config.IndexingConfig:
    return runtime_config.IndexingConfig(
        levels=tuple(runtime_config.IndexingLevel(level.value) for level in dto.levels),
        include_imports=bool(dto.include_imports),
        include_class_context=bool(dto.include_class_context),
        generate_repo_overview=bool(dto.generate_repo_overview),
        allow_direct_index=bool(dto.allow_direct_index),
    )


def _vector_store_config(
    dto: VectorStoreConfigDTO,
) -> runtime_config.VectorStoreConfig:
    return runtime_config.VectorStoreConfig(
        type=dto.type,
        distance_metric=dto.distance_metric,
        index_type=dto.index_type,
        shard_storage=_vector_shard_storage(dto.shard_storage),
        ef_construction=int(dto.ef_construction),
        ef_search=int(dto.ef_search),
        m=int(dto.m),
        persist_directory=dto.persist_directory,
        in_memory=bool(dto.in_memory),
        index_scan_cache_ttl=float(dto.index_scan_cache_ttl),
        index_scan_sample_size=int(dto.index_scan_sample_size),
    )


def _vector_shard_storage(
    shard_storage: VectorShardStorageDTO,
) -> runtime_config.VectorShardStorage:
    return runtime_config.VectorShardStorage(shard_storage.value)


def _retrieval_config(dto: RetrievalConfigDTO) -> runtime_config.RetrievalConfig:
    return runtime_config.RetrievalConfig(
        semantic_weight=float(dto.semantic_weight),
        keyword_weight=float(dto.keyword_weight),
        graph_weight=float(dto.graph_weight),
        retrieval_backend=runtime_config.RetrievalBackend(dto.retrieval_backend.value),
        graph_expansion_backend=runtime_config.GraphExpansionBackend(
            dto.graph_expansion_backend.value
        ),
        allow_graph_builder_fallback=bool(dto.allow_graph_builder_fallback),
        min_similarity=float(dto.min_similarity),
        max_results=int(dto.max_results),
        diversity_penalty=float(dto.diversity_penalty),
        enable_two_stage_retrieval=bool(dto.enable_two_stage_retrieval),
        select_repos_by_overview=bool(dto.select_repos_by_overview),
        repo_selection_method=dto.repo_selection_method,
        top_repos_to_search=int(dto.top_repos_to_search),
        min_repo_similarity=float(dto.min_repo_similarity),
        max_files_to_search=int(dto.max_files_to_search),
        enable_agency_mode=bool(dto.enable_agency_mode),
        adaptive_fusion=dict(dto.adaptive_fusion),
    )


def _generation_config(dto: GenerationConfigDTO) -> runtime_config.GenerationConfig:
    return runtime_config.GenerationConfig(
        provider=dto.provider,
        model=dto.model,
        base_url=dto.base_url,
        openai_api_key=dto.openai_api_key,
        anthropic_api_key=dto.anthropic_api_key,
        temperature=float(dto.temperature),
        max_tokens=int(dto.max_tokens),
        max_context_tokens=int(dto.max_context_tokens),
        reserve_tokens_for_response=int(dto.reserve_tokens_for_response),
        include_file_paths=bool(dto.include_file_paths),
        include_line_numbers=bool(dto.include_line_numbers),
        include_related_code=bool(dto.include_related_code),
        enable_multi_turn=bool(dto.enable_multi_turn),
        context_rounds=int(dto.context_rounds),
    )


def _query_config(dto: QueryConfigDTO) -> runtime_config.QueryConfig:
    return runtime_config.QueryConfig(
        expand_query=bool(dto.expand_query),
        decompose_complex=bool(dto.decompose_complex),
        max_subqueries=int(dto.max_subqueries),
        extract_keywords=bool(dto.extract_keywords),
        detect_intent=bool(dto.detect_intent),
        use_llm_enhancement=bool(dto.use_llm_enhancement),
        llm_enhancement_mode=dto.llm_enhancement_mode,
        history_summary_rounds=int(dto.history_summary_rounds),
        max_summary_words=int(dto.max_summary_words),
    )


def _graph_config(dto: GraphConfigDTO) -> runtime_config.GraphConfig:
    return runtime_config.GraphConfig(
        build_call_graph=bool(dto.build_call_graph),
        build_dependency_graph=bool(dto.build_dependency_graph),
        build_inheritance_graph=bool(dto.build_inheritance_graph),
        max_depth=int(dto.max_depth),
        ladybug=runtime_config.LadybugGraphConfig(
            enabled=bool(dto.ladybug.enabled),
            db_path=dto.ladybug.db_path,
            postgres_attach_dsn=dto.ladybug.postgres_attach_dsn,
        ),
    )


def _docs_integration_config(
    dto: DocsIntegrationConfigDTO,
) -> runtime_config.DocsIntegrationConfig:
    return runtime_config.DocsIntegrationConfig(
        enabled=bool(dto.enabled),
        curated_paths=_string_list(dto.curated_paths),
        allow_paths=_string_list(dto.allow_paths),
        deny_paths=_string_list(dto.deny_paths),
        chunk_token_size=int(dto.chunk_token_size),
        similarity_threshold=float(dto.similarity_threshold),
        chunk_size=int(dto.chunk_size),
        chunk_overlap=int(dto.chunk_overlap),
        max_chunk_chars=int(dto.max_chunk_chars),
    )


def _agent_config(dto: AgentConfigDTO) -> runtime_config.AgentConfig:
    return runtime_config.AgentConfig(
        iterative=runtime_config.IterativeAgentConfig(
            max_iterations=int(dto.iterative.max_iterations),
            confidence_threshold=int(dto.iterative.confidence_threshold),
            min_confidence_gain=int(dto.iterative.min_confidence_gain),
            max_total_lines=int(dto.iterative.max_total_lines),
            temperature_agent=float(dto.iterative.temperature_agent),
            max_tokens_agent=int(dto.iterative.max_tokens_agent),
            max_elements=int(dto.iterative.max_elements),
            max_candidates_display=int(dto.iterative.max_candidates_display),
        )
    )


def _cache_config(dto: CacheConfigDTO) -> runtime_config.CacheConfig:
    return runtime_config.CacheConfig(
        enabled=bool(dto.enabled),
        backend=dto.backend,
        ttl=int(dto.ttl),
        dialogue_ttl=int(dto.dialogue_ttl),
        max_size_mb=int(dto.max_size_mb),
        cache_embeddings=bool(dto.cache_embeddings),
        cache_queries=bool(dto.cache_queries),
        cache_directory=dto.cache_directory,
        redis_host=dto.redis_host,
        redis_port=int(dto.redis_port),
    )


def _evaluation_config(dto: EvaluationConfigDTO) -> runtime_config.EvaluationConfig:
    return runtime_config.EvaluationConfig(
        enabled=bool(dto.enabled),
        in_memory_index=bool(dto.in_memory_index),
        disable_cache=bool(dto.disable_cache),
        disable_persistence=bool(dto.disable_persistence),
        force_reindex=bool(dto.force_reindex),
    )


def _logging_config(dto: LoggingConfigDTO) -> runtime_config.LoggingConfig:
    return runtime_config.LoggingConfig(
        level=dto.level,
        format=dto.format,
        file=dto.file,
        console=bool(dto.console),
    )


def _terminus_config(dto: TerminusConfigDTO) -> runtime_config.TerminusConfig:
    return runtime_config.TerminusConfig(
        endpoint=dto.endpoint,
        api_key=dto.api_key,
        timeout_seconds=int(dto.timeout_seconds),
    )


def _projection_config(dto: ProjectionConfigDTO) -> runtime_config.ProjectionConfig:
    return runtime_config.ProjectionConfig(
        postgres_dsn=dto.postgres_dsn,
        enable_leiden=bool(dto.enable_leiden),
        hierarchical_leiden_enabled=bool(dto.hierarchical_leiden_enabled),
        leiden_resolutions=_float_list(dto.leiden_resolutions),
        hierarchy_max_levels=int(dto.hierarchy_max_levels),
        hierarchy_max_nodes=int(dto.hierarchy_max_nodes),
        steiner_prune=bool(dto.steiner_prune),
        aggregation_top_members=int(dto.aggregation_top_members),
        max_supporting_docs_per_cluster=int(dto.max_supporting_docs_per_cluster),
        llm_enabled=bool(dto.llm_enabled),
        llm_timeout_seconds=int(dto.llm_timeout_seconds),
        llm_max_tokens=int(dto.llm_max_tokens),
        llm_temperature=float(dto.llm_temperature),
        max_entity_hops=int(dto.max_entity_hops),
        max_query_hops=int(dto.max_query_hops),
        max_chunk_count=int(dto.max_chunk_count),
        dirty_widen_path_threshold=int(dto.dirty_widen_path_threshold),
        edge_weights=_float_items(dto.edge_weights),
    )


def config_from_dto(
    dto: FastCodeConfigDTO,
) -> runtime_config.FastCodeConfig:
    """Translate a validated inbound config DTO into frozen runtime config."""
    return runtime_config.FastCodeConfig(
        repo_root=dto.repo_root,
        storage=_storage_config(dto.storage),
        repository=_repository_config(dto.repository),
        parser=_parser_config(dto.parser),
        embedding=_embedding_config(dto.embedding),
        indexing=_indexing_config(dto.indexing),
        vector_store=_vector_store_config(dto.vector_store),
        retrieval=_retrieval_config(dto.retrieval),
        generation=_generation_config(dto.generation),
        query=_query_config(dto.query),
        graph=_graph_config(dto.graph),
        docs_integration=_docs_integration_config(dto.docs_integration),
        agent=_agent_config(dto.agent),
        cache=_cache_config(dto.cache),
        evaluation=_evaluation_config(dto.evaluation),
        logging=_logging_config(dto.logging),
        terminus=_terminus_config(dto.terminus),
        projection=_projection_config(dto.projection),
    )


def config_from_mapping(raw: dict[str, Any] | None) -> runtime_config.FastCodeConfig:
    """Validate raw inbound config shape and map it into frozen runtime config."""
    return config_from_dto(FastCodeConfigDTO.model_validate(raw or {}))


def config_to_dict(config: runtime_config.FastCodeConfig) -> dict[str, Any]:
    """Explicit runtime mapping adapter for dict-based shell consumers."""
    return {
        "repo_root": config.repo_root,
        "storage": {
            "backend": config.storage.backend.value,
            "postgres_dsn": config.storage.postgres_dsn,
            "pool_min": config.storage.pool_min,
            "pool_max": config.storage.pool_max,
        },
        "repository": {
            "clone_depth": config.repository.clone_depth,
            "max_file_size_mb": config.repository.max_file_size_mb,
            "backup_directory": config.repository.backup_directory,
            "local_source_mode": config.repository.local_source_mode.value,
            "exclude_site_packages": config.repository.exclude_site_packages,
            "ignore_patterns": config.repository.ignore_patterns,
            "supported_extensions": config.repository.supported_extensions,
        },
        "parser": {
            "extract_docstrings": config.parser.extract_docstrings,
            "extract_comments": config.parser.extract_comments,
            "extract_imports": config.parser.extract_imports,
            "compute_complexity": config.parser.compute_complexity,
            "max_function_lines": config.parser.max_function_lines,
        },
        "embedding": {
            "provider": config.embedding.provider,
            "model": config.embedding.model,
            "ollama_url": config.embedding.ollama_url,
            "device": config.embedding.device,
            "batch_size": config.embedding.batch_size,
            "max_seq_length": config.embedding.max_seq_length,
            "normalize_embeddings": config.embedding.normalize_embeddings,
        },
        "indexing": {
            "levels": config.indexing.levels,
            "include_imports": config.indexing.include_imports,
            "include_class_context": config.indexing.include_class_context,
            "generate_repo_overview": config.indexing.generate_repo_overview,
            "allow_direct_index": config.indexing.allow_direct_index,
        },
        "vector_store": {
            "type": config.vector_store.type,
            "distance_metric": config.vector_store.distance_metric,
            "index_type": config.vector_store.index_type,
            "shard_storage": config.vector_store.shard_storage.value,
            "ef_construction": config.vector_store.ef_construction,
            "ef_search": config.vector_store.ef_search,
            "m": config.vector_store.m,
            "persist_directory": config.vector_store.persist_directory,
            "in_memory": config.vector_store.in_memory,
            "index_scan_cache_ttl": config.vector_store.index_scan_cache_ttl,
            "index_scan_sample_size": config.vector_store.index_scan_sample_size,
        },
        "retrieval": {
            "semantic_weight": config.retrieval.semantic_weight,
            "keyword_weight": config.retrieval.keyword_weight,
            "graph_weight": config.retrieval.graph_weight,
            "retrieval_backend": config.retrieval.retrieval_backend.value,
            "graph_expansion_backend": (config.retrieval.graph_expansion_backend.value),
            "allow_graph_builder_fallback": (
                config.retrieval.allow_graph_builder_fallback
            ),
            "min_similarity": config.retrieval.min_similarity,
            "max_results": config.retrieval.max_results,
            "diversity_penalty": config.retrieval.diversity_penalty,
            "enable_two_stage_retrieval": (config.retrieval.enable_two_stage_retrieval),
            "select_repos_by_overview": config.retrieval.select_repos_by_overview,
            "repo_selection_method": config.retrieval.repo_selection_method,
            "top_repos_to_search": config.retrieval.top_repos_to_search,
            "min_repo_similarity": config.retrieval.min_repo_similarity,
            "max_files_to_search": config.retrieval.max_files_to_search,
            "enable_agency_mode": config.retrieval.enable_agency_mode,
            "adaptive_fusion": dict(config.retrieval.adaptive_fusion),
        },
        "generation": {
            "provider": config.generation.provider,
            "model": config.generation.model,
            "base_url": config.generation.base_url,
            "openai_api_key": config.generation.openai_api_key,
            "anthropic_api_key": config.generation.anthropic_api_key,
            "temperature": config.generation.temperature,
            "max_tokens": config.generation.max_tokens,
            "max_context_tokens": config.generation.max_context_tokens,
            "reserve_tokens_for_response": (
                config.generation.reserve_tokens_for_response
            ),
            "include_file_paths": config.generation.include_file_paths,
            "include_line_numbers": config.generation.include_line_numbers,
            "include_related_code": config.generation.include_related_code,
            "enable_multi_turn": config.generation.enable_multi_turn,
            "context_rounds": config.generation.context_rounds,
        },
        "query": {
            "expand_query": config.query.expand_query,
            "decompose_complex": config.query.decompose_complex,
            "max_subqueries": config.query.max_subqueries,
            "extract_keywords": config.query.extract_keywords,
            "detect_intent": config.query.detect_intent,
            "use_llm_enhancement": config.query.use_llm_enhancement,
            "llm_enhancement_mode": config.query.llm_enhancement_mode,
            "history_summary_rounds": config.query.history_summary_rounds,
            "max_summary_words": config.query.max_summary_words,
        },
        "graph": {
            "build_call_graph": config.graph.build_call_graph,
            "build_dependency_graph": config.graph.build_dependency_graph,
            "build_inheritance_graph": config.graph.build_inheritance_graph,
            "max_depth": config.graph.max_depth,
            "ladybug": {
                "enabled": config.graph.ladybug.enabled,
                "db_path": config.graph.ladybug.db_path,
                "postgres_attach_dsn": config.graph.ladybug.postgres_attach_dsn,
            },
        },
        "docs_integration": {
            "enabled": config.docs_integration.enabled,
            "curated_paths": config.docs_integration.curated_paths,
            "allow_paths": config.docs_integration.allow_paths,
            "deny_paths": config.docs_integration.deny_paths,
            "chunk_token_size": config.docs_integration.chunk_token_size,
            "similarity_threshold": config.docs_integration.similarity_threshold,
            "chunk_size": config.docs_integration.chunk_size,
            "chunk_overlap": config.docs_integration.chunk_overlap,
            "max_chunk_chars": config.docs_integration.max_chunk_chars,
        },
        "agent": {
            "iterative": {
                "max_iterations": config.agent.iterative.max_iterations,
                "confidence_threshold": (config.agent.iterative.confidence_threshold),
                "min_confidence_gain": config.agent.iterative.min_confidence_gain,
                "max_total_lines": config.agent.iterative.max_total_lines,
                "temperature_agent": config.agent.iterative.temperature_agent,
                "max_tokens_agent": config.agent.iterative.max_tokens_agent,
                "max_elements": config.agent.iterative.max_elements,
                "max_candidates_display": (
                    config.agent.iterative.max_candidates_display
                ),
            }
        },
        "cache": {
            "enabled": config.cache.enabled,
            "backend": config.cache.backend,
            "ttl": config.cache.ttl,
            "dialogue_ttl": config.cache.dialogue_ttl,
            "max_size_mb": config.cache.max_size_mb,
            "cache_embeddings": config.cache.cache_embeddings,
            "cache_queries": config.cache.cache_queries,
            "cache_directory": config.cache.cache_directory,
            "redis_host": config.cache.redis_host,
            "redis_port": config.cache.redis_port,
        },
        "evaluation": {
            "enabled": config.evaluation.enabled,
            "in_memory_index": config.evaluation.in_memory_index,
            "disable_cache": config.evaluation.disable_cache,
            "disable_persistence": config.evaluation.disable_persistence,
            "force_reindex": config.evaluation.force_reindex,
        },
        "logging": {
            "level": config.logging.level,
            "format": config.logging.format,
            "file": config.logging.file,
            "console": config.logging.console,
        },
        "terminus": {
            "endpoint": config.terminus.endpoint,
            "api_key": config.terminus.api_key,
            "timeout_seconds": config.terminus.timeout_seconds,
        },
        "projection": {
            "postgres_dsn": config.projection.postgres_dsn,
            "enable_leiden": config.projection.enable_leiden,
            "hierarchical_leiden_enabled": (
                config.projection.hierarchical_leiden_enabled
            ),
            "leiden_resolutions": config.projection.leiden_resolutions,
            "hierarchy_max_levels": config.projection.hierarchy_max_levels,
            "hierarchy_max_nodes": config.projection.hierarchy_max_nodes,
            "steiner_prune": config.projection.steiner_prune,
            "aggregation_top_members": config.projection.aggregation_top_members,
            "max_supporting_docs_per_cluster": (
                config.projection.max_supporting_docs_per_cluster
            ),
            "llm_enabled": config.projection.llm_enabled,
            "llm_timeout_seconds": config.projection.llm_timeout_seconds,
            "llm_max_tokens": config.projection.llm_max_tokens,
            "llm_temperature": config.projection.llm_temperature,
            "max_entity_hops": config.projection.max_entity_hops,
            "max_query_hops": config.projection.max_query_hops,
            "max_chunk_count": config.projection.max_chunk_count,
            "dirty_widen_path_threshold": (
                config.projection.dirty_widen_path_threshold
            ),
            "edge_weights": {
                str(name): float(weight)
                for name, weight in config.projection.edge_weights
            },
        },
    }
