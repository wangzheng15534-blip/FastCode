"""App-global runtime assembly for the FastCode use_flow layer.

This module owns the construction of the use_flow object graph: repository
loader, embedder, vector store, stores, IndexPipeline, QueryPipeline, and the
facades (indexing/query/store/context/cache/projection/publishing). It is the
FastCode analogue of aw-checker's `checker-app::app_runtime` and zotero-rs's
`zotero-app::app_runtime` — the app-global assembly that the assembly root
(``FastCode``) calls but does not own the semantics of.

Effect-facility instances (``DBRuntime``, ``LadybugGraphRuntime``) are built by
the assembly root / entry frame (legal ``entry_frame -> effect_facility`` edge)
and passed in here, because ``use_flow -> effect_facility`` is forbidden by the
FCIS role graph. Effect-tool runtimes (SCIP/semantic subprocess helpers) and
meaning_core types (retrieval/semantic) are imported here directly — those
edges are legal from ``use_flow``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastcode.app.indexing.doc_ingester import KeyDocIngester
from fastcode.app.indexing.embedder import CodeEmbedder
from fastcode.app.indexing.extractors.parser import CodeParser
from fastcode.app.indexing.facade import IndexingFacade
from fastcode.app.indexing.loader import RepositoryLoader
from fastcode.app.indexing.pipeline.direct_indexer import DirectIndexer
from fastcode.app.indexing.pipeline.indexer import CodeIndexer
from fastcode.app.indexing.pipeline.multi_repo_direct import MultiRepoDirectIndexer
from fastcode.app.indexing.pipeline.redo_worker import RedoWorker
from fastcode.app.indexing.pipeline.service import IndexPipeline
from fastcode.app.indexing.projection.service import ProjectionService
from fastcode.app.indexing.projection.transform import ProjectionTransformer
from fastcode.app.indexing.projection_facade import ProjectionFacade
from fastcode.app.indexing.publishing import PublishingService
from fastcode.app.indexing.publishing_facade import PublishingFacade
from fastcode.app.indexing.terminus import TerminusPublisher
from fastcode.app.query.facade import QueryFacade
from fastcode.app.query.orchestration.answer import AnswerGenerator
from fastcode.app.query.orchestration.handler import QueryPipeline
from fastcode.app.query.orchestration.processor import QueryProcessor
from fastcode.app.query.selection.retriever import HybridRetriever
from fastcode.app.store.artifacts.file import FileArtifactStore
from fastcode.app.store.artifacts.graph import GraphArtifactStore
from fastcode.app.store.artifacts.unit import UnitArtifactStore
from fastcode.app.store.cache.service import CacheManager
from fastcode.app.store.cache_facade import CacheFacade
from fastcode.app.store.context_facade import ContextFacade
from fastcode.app.store.facade import StoreFacade
from fastcode.app.store.runs.index_run import IndexRunStore
from fastcode.app.store.snapshots.manifest import ManifestStore
from fastcode.app.store.snapshots.projection import ProjectionStore
from fastcode.app.store.snapshots.snapshot import SnapshotStore
from fastcode.app.store.vectors.pg_retrieval import PgRetrievalStore
from fastcode.app.store.vectors.vector import VectorStore
from fastcode.graph.build import CodeGraphBuilder
from fastcode.infrastructure.execution.scip_runner import SubprocessScipIndexerRuntime
from fastcode.infrastructure.execution.semantic_helper import (
    SubprocessSemanticHelperRuntime,
)
from fastcode.infrastructure.execution.helper_operations import (
    SemanticHelperOperations,
)
from fastcode.ir.graph import IRGraphBuilder
from fastcode.semantic.resolvers.engine.registry import (
    build_default_semantic_resolver_registry,
)
from fastcode.semantic.symbol_index import SnapshotSymbolIndex

if TYPE_CHECKING:
    from fastcode.app.indexing.runtime_contracts import DocumentGraphRuntime
    from fastcode.app.store.runtime_contracts import StoreDatabaseRuntime


@dataclass
class AppRuntime:
    """Assembled use_flow object graph held by the FastCode assembly root."""

    loader: RepositoryLoader
    parser: CodeParser
    embedder: CodeEmbedder
    vector_store: VectorStore
    indexer: CodeIndexer
    graph_builder: CodeGraphBuilder
    graph_artifact_store: GraphArtifactStore
    ir_graph_builder: IRGraphBuilder
    retriever: HybridRetriever
    query_processor: QueryProcessor
    answer_generator: AnswerGenerator
    cache_manager: CacheManager
    snapshot_store: SnapshotStore
    manifest_store: ManifestStore
    index_run_store: IndexRunStore
    unit_artifact_store: UnitArtifactStore
    file_artifact_store: FileArtifactStore
    terminus_publisher: TerminusPublisher
    projection_transformer: ProjectionTransformer
    projection_store: ProjectionStore
    snapshot_symbol_index: SnapshotSymbolIndex
    store: StoreFacade
    pg_retrieval_store: PgRetrievalStore
    doc_ingester: KeyDocIngester
    semantic_helper_runtime: SubprocessSemanticHelperRuntime
    scip_indexer_runtime: SubprocessScipIndexerRuntime
    semantic_resolver_registry: Any
    pipeline: IndexPipeline
    direct_indexer: DirectIndexer
    multi_repo_direct_indexer: MultiRepoDirectIndexer
    indexing: IndexingFacade
    projection_service: ProjectionService
    projection: ProjectionFacade
    publishing_service: PublishingService
    publishing: PublishingFacade
    query_handler: QueryPipeline
    query: QueryFacade
    context: ContextFacade
    cache: CacheFacade
    db_runtime: StoreDatabaseRuntime
    graph_runtime: DocumentGraphRuntime | None
    redo_worker: RedoWorker | None


def build_app_runtime(
    *,
    config: dict[str, Any],
    logger: logging.Logger,
    db_runtime: StoreDatabaseRuntime,
    graph_runtime: DocumentGraphRuntime | None,
    state: Any,
    eval_config: dict[str, Any],
    set_repo_root_fn: Any,
    apply_env_ignore_patterns_fn: Any,
) -> AppRuntime:
    """Build the full use_flow object graph from a typed config + effect handles.

    ``db_runtime`` and ``graph_runtime`` are effect_facility instances built by
    the assembly root; everything else is constructed here in use_flow.
    """
    loader = RepositoryLoader(config)
    parser = CodeParser(config)
    embedder = CodeEmbedder(config)
    vector_store = VectorStore(config)
    indexer = CodeIndexer(config, loader, parser, embedder, vector_store)
    graph_builder = CodeGraphBuilder(config)
    graph_artifact_store = GraphArtifactStore(config)
    ir_graph_builder = IRGraphBuilder()

    persist_dir = vector_store.persist_dir
    snapshot_store = SnapshotStore(persist_dir, db_runtime=db_runtime)
    manifest_store = ManifestStore(db_runtime)
    index_run_store = IndexRunStore(db_runtime)
    unit_artifact_store = UnitArtifactStore(db_runtime)
    file_artifact_store = FileArtifactStore(db_runtime)
    terminus_publisher = TerminusPublisher(config)
    projection_transformer = ProjectionTransformer(config)
    projection_store = ProjectionStore(config)
    snapshot_symbol_index = SnapshotSymbolIndex()
    store = StoreFacade(
        vector_store=vector_store,
        snapshot_store=snapshot_store,
        manifest_store=manifest_store,
        snapshot_symbol_index=snapshot_symbol_index,
        state=state,
        config=config,
        projection_store=projection_store,
        projection_transformer=projection_transformer,
    )
    pg_retrieval_store = PgRetrievalStore(db_runtime, config)

    # repo_root resolution happens in the assembly root before calling here;
    # retriever only needs a repo_root string for path-relative scanning.
    config_repo_root = str(config.get("repo_root") or ".")
    retriever = HybridRetriever(
        config,
        vector_store,
        embedder,
        graph_builder,
        repo_root=config_repo_root,
        vector_store_factory=_VectorSearchStoreFactory(config),
    )
    retriever.set_pg_retrieval_store(pg_retrieval_store)
    query_processor = QueryProcessor(config)
    answer_generator = AnswerGenerator(config)
    cache_manager = CacheManager(config)
    doc_ingester = KeyDocIngester(config, embedder)

    semantic_helper_runtime = SubprocessSemanticHelperRuntime()
    scip_indexer_runtime = SubprocessScipIndexerRuntime()
    semantic_resolver_registry = build_default_semantic_resolver_registry(
        semantic_helper_runtime=semantic_helper_runtime,
        helper_ops_factory=lambda runtime: SemanticHelperOperations(runtime),
    )

    pipeline = IndexPipeline(
        config=config,
        logger=logger,
        loader=loader,
        snapshot_store=snapshot_store,
        manifest_store=manifest_store,
        index_run_store=index_run_store,
        unit_artifact_store=unit_artifact_store,
        file_artifact_store=file_artifact_store,
        snapshot_symbol_index=snapshot_symbol_index,
        vector_store=vector_store,
        embedder=embedder,
        indexer=indexer,
        retriever=retriever,
        graph_builder=graph_builder,
        ir_graph_builder=ir_graph_builder,
        graph_artifact_store=graph_artifact_store,
        pg_retrieval_store=pg_retrieval_store,
        terminus_publisher=terminus_publisher,
        doc_ingester=doc_ingester,
        semantic_resolver_registry=semantic_resolver_registry,
        set_repo_indexed=lambda v: setattr(state, "repo_indexed", v),
        set_repo_loaded=lambda v: setattr(state, "repo_loaded", v),
        set_repo_info=lambda v: setattr(state, "repo_info", v),
        semantic_helper_runtime=semantic_helper_runtime,
        scip_indexer_runtime=scip_indexer_runtime,
    )

    direct_indexer = DirectIndexer(
        config=config,
        loader=loader,
        indexer=indexer,
        embedder=embedder,
        vector_store=vector_store,
        graph_builder=graph_builder,
        retriever=retriever,
        graph_artifact_store=graph_artifact_store,
        should_use_cache_fn=lambda: _should_use_cache(eval_config, vector_store),
        try_load_from_cache_fn=lambda: _try_load_from_cache(
            eval_config, logger, state, vector_store, retriever,
            graph_builder, graph_artifact_store,
        ),
        should_persist_fn=lambda: _should_persist_indexes(eval_config, vector_store),
        save_to_cache_fn=lambda cache_name=None: _save_to_cache(
            eval_config, logger, state, vector_store, cache_name
        ),
        set_repo_root_fn=set_repo_root_fn,
        log_statistics_fn=lambda: _log_statistics(logger, vector_store, graph_builder),
    )
    multi_repo_direct_indexer = MultiRepoDirectIndexer(
        config=config,
        loader=loader,
        parser=parser,
        embedder=embedder,
        vector_store=vector_store,
        graph_builder=graph_builder,
        graph_artifact_store=graph_artifact_store,
        set_repo_root_fn=set_repo_root_fn,
        infer_is_url_fn=IndexPipeline._infer_is_url,
    )

    indexing = IndexingFacade(
        loader=loader,
        pipeline=pipeline,
        state=state,
        vector_store=vector_store,
        store=store,
        direct_indexer=direct_indexer,
        multi_repo_direct_indexer=multi_repo_direct_indexer,
        graph_runtime=graph_runtime,
        retriever=retriever,
        config=config,
        eval_config=eval_config,
        logger=logger,
        set_repo_root_fn=set_repo_root_fn,
        apply_env_ignore_patterns_fn=apply_env_ignore_patterns_fn,
    )

    projection_service = ProjectionService(
        config=config,
        logger=logger,
        projection_store=projection_store,
        projection_transformer=projection_transformer,
        snapshot_store=snapshot_store,
        manifest_store=manifest_store,
        load_artifacts_by_key=pipeline._load_artifacts_by_key,
    )
    projection = ProjectionFacade(service=projection_service, state=state)
    publishing_service = PublishingService(
        config=config,
        logger=logger,
        index_run_store=index_run_store,
        manifest_store=manifest_store,
        snapshot_store=snapshot_store,
        terminus_publisher=terminus_publisher,
        redo_worker=None,
        build_git_meta=pipeline._build_git_meta,
        previous_snapshot_symbol_versions=pipeline._previous_snapshot_symbol_versions,
        run_index_pipeline_cb=indexing.run_index_pipeline,
    )
    publishing = PublishingFacade(
        publishing_service=publishing_service,
        pipeline=pipeline,
        projection_store=projection_store,
        snapshot_store=snapshot_store,
        config=config,
    )

    query_handler = QueryPipeline(
        config=config,
        logger=logger,
        retriever=retriever,
        query_processor=query_processor,
        answer_generator=answer_generator,
        cache_manager=cache_manager,
        manifest_store=manifest_store,
        snapshot_store=snapshot_store,
        snapshot_symbol_index=snapshot_symbol_index,
        is_repo_indexed=lambda: state.repo_indexed,
        load_artifacts_by_key=pipeline._load_artifacts_by_key,
        load_snapshot_artifacts=pipeline.load_snapshot_artifacts_handle,
        get_session_prefix=projection.get_session_prefix,
        semantic_escalation_cb=None,
    )
    query = QueryFacade(
        query_handler=query_handler,
        vector_store=vector_store,
        graph_builder=graph_builder,
        snapshot_store=snapshot_store,
        ir_graph_builder=ir_graph_builder,
        snapshot_symbol_index=snapshot_symbol_index,
        pipeline=pipeline,
        state=state,
    )
    query_handler.semantic_escalation_cb = query._escalate_query_semantics

    context = ContextFacade(cache_manager)
    cache = CacheFacade(
        cache_manager=cache_manager,
        vector_store=vector_store,
        embedder=embedder,
        retriever=retriever,
        graph_builder=graph_builder,
        graph_artifact_store=graph_artifact_store,
        state=state,
    )

    return AppRuntime(
        loader=loader,
        parser=parser,
        embedder=embedder,
        vector_store=vector_store,
        indexer=indexer,
        graph_builder=graph_builder,
        graph_artifact_store=graph_artifact_store,
        ir_graph_builder=ir_graph_builder,
        retriever=retriever,
        query_processor=query_processor,
        answer_generator=answer_generator,
        cache_manager=cache_manager,
        snapshot_store=snapshot_store,
        manifest_store=manifest_store,
        index_run_store=index_run_store,
        unit_artifact_store=unit_artifact_store,
        file_artifact_store=file_artifact_store,
        terminus_publisher=terminus_publisher,
        projection_transformer=projection_transformer,
        projection_store=projection_store,
        snapshot_symbol_index=snapshot_symbol_index,
        store=store,
        pg_retrieval_store=pg_retrieval_store,
        doc_ingester=doc_ingester,
        semantic_helper_runtime=semantic_helper_runtime,
        scip_indexer_runtime=scip_indexer_runtime,
        semantic_resolver_registry=semantic_resolver_registry,
        pipeline=pipeline,
        direct_indexer=direct_indexer,
        multi_repo_direct_indexer=multi_repo_direct_indexer,
        indexing=indexing,
        projection_service=projection_service,
        projection=projection,
        publishing_service=publishing_service,
        publishing=publishing,
        query_handler=query_handler,
        query=query,
        context=context,
        cache=cache,
        db_runtime=db_runtime,
        graph_runtime=graph_runtime,
        redo_worker=None,
    )


class _VectorSearchStoreFactory:
    """Factory for query-scoped temporary vector stores."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    def create_vector_search_store(self) -> VectorStore:
        return VectorStore(self._config)


def _is_ephemeral_mode(eval_config: dict[str, Any], vector_store: VectorStore) -> bool:
    return bool(
        eval_config.get("in_memory_index", False)
        or getattr(vector_store, "in_memory", False)
    )


def _should_use_cache(eval_config: dict[str, Any], vector_store: VectorStore) -> bool:
    if eval_config.get("disable_cache", False):
        return False
    return not _is_ephemeral_mode(eval_config, vector_store)


def _should_persist_indexes(
    eval_config: dict[str, Any], vector_store: VectorStore
) -> bool:
    if eval_config.get("disable_persistence", False):
        return False
    return not _is_ephemeral_mode(eval_config, vector_store)


def _log_statistics(
    logger: logging.Logger,
    vector_store: VectorStore,
    graph_builder: CodeGraphBuilder,
) -> None:
    stats = {
        "vector_count": vector_store.get_count(),
        "graph_stats": graph_builder.get_graph_stats(),
    }
    logger.info("Statistics: %s", stats)


def _try_load_from_cache(
    eval_config: dict[str, Any],
    logger: logging.Logger,
    state: Any,
    vector_store: VectorStore,
    retriever: HybridRetriever,
    graph_builder: CodeGraphBuilder,
    graph_artifact_store: GraphArtifactStore,
) -> bool:
    if not _should_use_cache(eval_config, vector_store):
        logger.info("Cache loading disabled (ephemeral/evaluation mode)")
        return False
    # Late import keeps the cache rehydration kit out of the hot import path.
    from fastcode.app.store.cache.rehydration import try_load_from_cache as _impl

    return _impl(
        cache_name=state.repo_info.get("name", "default"),
        vector_store=vector_store,
        retriever=retriever,
        graph_builder=graph_builder,
        graph_artifact_store=graph_artifact_store,
        log_statistics_fn=lambda: _log_statistics(
            logger, vector_store, graph_builder
        ),
    )


def _save_to_cache(
    eval_config: dict[str, Any],
    logger: logging.Logger,
    state: Any,
    vector_store: VectorStore,
    cache_name: str | None,
) -> None:
    if not _should_persist_indexes(eval_config, vector_store):
        logger.info("Cache save disabled (ephemeral/evaluation mode)")
        return
    from fastcode.app.store.cache.rehydration import save_to_cache as _impl

    _impl(cache_name=cache_name or state.repo_info.get("name", "default"),
          vector_store=vector_store)
