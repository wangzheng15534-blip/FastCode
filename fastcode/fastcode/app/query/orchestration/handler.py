"""
QueryPipeline — query, query_stream, and query_snapshot extracted from FastCode.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import traceback
from collections.abc import Callable, Generator, Mapping
from typing import TYPE_CHECKING, Any, cast

import fastcode.retrieval.context.snapshot as _snapshot
from fastcode.app.query.boundary import processed_query_payload
from fastcode.app.query.context_payloads import (
    activation_payload,
    context_bundle_payload,
    distillation_from_payload,
    distillation_payload,
    turn_journal_payload,
    working_memory_payload,
)
from fastcode.app.query.selection.retriever import HybridRetriever
from fastcode.app.store.cache.contracts import (
    ContextActivationRecord,
    ContextBundleRecord,
    ContextDistillationRecord,
    TurnJournalRecord,
    WorkingMemoryRecord,
)
from fastcode.app.store.cache.service import CacheManager
from fastcode.app.store.snapshots.manifest import ManifestStore
from fastcode.app.store.snapshots.snapshot import SnapshotStore
from fastcode.retrieval.context.agent_context import (
    DistillationRecord,
    ToolObservation,
    TurnIntent,
    build_acceptance_contract,
    compute_risk_state,
    promote_observations,
)
from fastcode.retrieval.context.context_compiler import (
    COMPILER_FINGERPRINT,
    DEFAULT_BUDGET_FINGERPRINT,
    DEFAULT_DISTILLATION_PROMPT_FINGERPRINT,
    DEFAULT_EMBEDDING_FINGERPRINT,
    DEFAULT_PROJECTION_FINGERPRINT,
    build_context_bundle,
    build_context_invalidation_key,
    build_evidence_refs_from_sources,
    build_tool_observation,
    build_turn_journal,
    build_turn_plan,
    compile_working_memory,
)
from fastcode.retrieval.contracts import Hit, SourceCitation
from fastcode.semantic.symbol_index import SnapshotSymbolIndex

from .answer import AnswerGenerator
from .processor import QueryProcessor

if TYPE_CHECKING:
    from fastcode.app.query.orchestration.processor import ProcessedQuery


SemanticEscalationCallback = Callable[..., dict[str, Any] | None]


class QueryPipeline:
    """Handles query, query_stream, and query_snapshot for FastCode."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        logger: logging.Logger,
        retriever: HybridRetriever,
        query_processor: QueryProcessor,
        answer_generator: AnswerGenerator,
        cache_manager: CacheManager,
        manifest_store: ManifestStore,
        snapshot_store: SnapshotStore,
        snapshot_symbol_index: SnapshotSymbolIndex,
        is_repo_indexed: Callable[[], bool],
        load_artifacts_by_key: Callable[[str], bool],
        load_snapshot_artifacts: Callable[..., Any | None] | None = None,
        get_session_prefix: Callable[[str], dict[str, Any]] | None = None,
        semantic_escalation_cb: SemanticEscalationCallback | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.retriever = retriever
        self.query_processor = query_processor
        self.answer_generator = answer_generator
        self.cache_manager = cache_manager
        self.manifest_store = manifest_store
        self.snapshot_store = snapshot_store
        self.snapshot_symbol_index = snapshot_symbol_index
        self.is_repo_indexed = is_repo_indexed
        self.load_artifacts_by_key = load_artifacts_by_key
        self.load_snapshot_artifacts = load_snapshot_artifacts
        self.get_session_prefix = get_session_prefix
        self.semantic_escalation_cb = semantic_escalation_cb
        self._snapshot_query_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def _register_snapshot_symbols_from_payload(self, snapshot_id: str) -> bool:
        load_payload = getattr(
            self.snapshot_store,
            "load_snapshot_symbol_index_payload",
            None,
        )
        register_payload = getattr(
            self.snapshot_symbol_index,
            "register_snapshot_symbol_payload",
            None,
        )
        if not callable(load_payload) or not callable(register_payload):
            return False

        payload = load_payload(snapshot_id)
        if not isinstance(payload, Mapping):
            return False
        registered = register_payload(payload)
        if isinstance(registered, bool) and not registered:
            return False
        return self.snapshot_symbol_index.has_snapshot(snapshot_id)

    def _ensure_snapshot_symbol_index(self, snapshot_id: str) -> None:
        if self.snapshot_symbol_index.has_snapshot(snapshot_id):
            return
        if self._register_snapshot_symbols_from_payload(snapshot_id):
            return
        loaded_snapshot = self.snapshot_store.load_snapshot(snapshot_id)
        if loaded_snapshot:
            self.snapshot_symbol_index.register_snapshot(loaded_snapshot)

    def _semantic_escalation_budget(
        self,
        *,
        question: str,
        processed_query: ProcessedQuery,
        filters: dict[str, Any] | None,
    ) -> str:
        snapshot_id = (filters or {}).get("snapshot_id")
        if not snapshot_id or self.semantic_escalation_cb is None:
            return "none"

        normalized = question.lower()
        path_critical_markers = (
            "call chain",
            "caller",
            "callee",
            "reach",
            "path",
            "impact",
            "break",
            "dependency",
            "depends",
            "inherit",
            "override",
            "implementation",
            "used",
            "usage",
            "flow",
        )
        if any(marker in normalized for marker in path_critical_markers):
            return "path-critical"
        if processed_query.intent in {"debug", "find", "where"}:
            return "local"
        return "none"

    def _maybe_escalate_query_semantics(
        self,
        *,
        question: str,
        processed_query: ProcessedQuery,
        filters: dict[str, Any] | None,
        retrieved: list[dict[str, Any]],
        retriever: HybridRetriever | None = None,
        graph_builder: Any | None = None,
    ) -> dict[str, Any] | None:
        budget = self._semantic_escalation_budget(
            question=question,
            processed_query=processed_query,
            filters=filters,
        )
        if budget == "none" or self.semantic_escalation_cb is None:
            return None
        snapshot_id = (filters or {}).get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            return None
        return self.semantic_escalation_cb(
            snapshot_id=snapshot_id,
            retrieved=retrieved,
            processed_query=processed_query,
            budget=budget,
            retriever=retriever,
            graph_builder=graph_builder,
        )

    def query_snapshot(
        self,
        question: str,
        repo_name: str | None = None,
        ref_name: str | None = None,
        snapshot_id: str | None = None,
        filters: dict[str, Any] | None = None,
        session_id: str | None = None,
        enable_multi_turn: bool | None = None,
    ) -> dict[str, Any]:
        if not snapshot_id:
            if not repo_name or not ref_name:
                msg = "query_snapshot requires snapshot_id or repo_name+ref_name"
                raise RuntimeError(msg)
            manifest = self.manifest_store.get_branch_manifest_record(
                repo_name, ref_name
            )
            if not manifest:
                msg = f"manifest not found for {repo_name}:{ref_name}"
                raise RuntimeError(msg)
            snapshot_id = manifest.snapshot_id

        snapshot_record = self.snapshot_store.get_snapshot_record(snapshot_id)
        if not snapshot_record:
            msg = f"snapshot not found: {snapshot_id}"
            raise RuntimeError(msg)

        artifact_key = snapshot_record.artifact_key
        if self.load_snapshot_artifacts is not None:
            loaded_artifacts = self.load_snapshot_artifacts(
                artifact_key,
                snapshot_id=snapshot_id,
            )
            if loaded_artifacts is None:
                msg = f"failed to load artifacts for snapshot: {snapshot_id}"
                raise RuntimeError(msg)
            self._ensure_snapshot_symbol_index(snapshot_id)

            merged_filters = dict(filters or {})
            merged_filters["snapshot_id"] = snapshot_id
            merged_filters["artifact_key"] = getattr(
                loaded_artifacts,
                "artifact_key",
                artifact_key,
            )

            result = self.query(
                question=question,
                filters=merged_filters,
                repo_filter=None,
                session_id=session_id,
                enable_multi_turn=enable_multi_turn,
                retriever=loaded_artifacts.retriever,
                graph_builder=loaded_artifacts.graph_builder,
            )
            result["snapshot_id"] = snapshot_id
            result["artifact_key"] = getattr(
                loaded_artifacts,
                "artifact_key",
                artifact_key,
            )
            return result

        with self._snapshot_query_lock:
            if not self.load_artifacts_by_key(artifact_key):
                msg = f"failed to load artifacts for snapshot: {snapshot_id}"
                raise RuntimeError(msg)
            self._ensure_snapshot_symbol_index(snapshot_id)

            merged_filters = dict(filters or {})
            merged_filters["snapshot_id"] = snapshot_id
            merged_filters["artifact_key"] = artifact_key

            result = self.query(
                question=question,
                filters=merged_filters,
                repo_filter=None,
                session_id=session_id,
                enable_multi_turn=enable_multi_turn,
            )
        result["snapshot_id"] = snapshot_id
        result["artifact_key"] = artifact_key
        return result

    def query(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        repo_filter: list[str] | None = None,
        session_id: str | None = None,
        enable_multi_turn: bool | None = None,
        use_agency_mode: bool | None = None,
        prompt_builder: Callable[
            [str, str, dict[str, Any] | None, list[dict[str, Any]] | None], str
        ]
        | None = None,
        retriever: HybridRetriever | None = None,
        graph_builder: Any | None = None,
    ) -> dict[str, Any]:
        """
        Query the repository (or multiple repositories)

        Args:
            question: User question
            filters: Optional filters for retrieval
            repo_filter: Optional list of repository names to search in
            session_id: Optional session ID for multi-turn dialogue
            enable_multi_turn: Override config setting for multi-turn mode
            prompt_builder: Optional callable to build a custom LLM prompt using
                (question, prepared_context, query_info, dialogue_history)

        Returns:
            Dictionary with answer and metadata (including summary if multi-turn)
        """
        active_retriever = retriever or self.retriever

        if not self.is_repo_indexed():
            msg = "Repository not indexed. Call index_repository() first."
            raise RuntimeError(msg)

        # Determine if multi-turn mode is enabled
        if enable_multi_turn is None:
            enable_multi_turn = self.config.get("generation", {}).get(
                "enable_multi_turn", False
            )

        if repo_filter:
            self.logger.info(
                f"Processing query: {question} in repositories: {repo_filter}"
            )
        else:
            self.logger.info(f"Processing query: {question}")

        # Get dialogue history if in multi-turn mode
        dialogue_history: list[dict[str, Any]] = []
        if enable_multi_turn and session_id:
            # Get recent summaries from cache (last 10 turns for iterative agent)
            history_summary_rounds = self.config.get("query", {}).get(
                "history_summary_rounds", 10
            )
            dialogue_history = self.cache_manager.get_recent_summaries(
                session_id, history_summary_rounds
            )

            if dialogue_history:
                self.logger.info(
                    f"Retrieved {len(dialogue_history)} previous dialogue summaries"
                )

        # NOTE: Query result caching is disabled to ensure full iterative_agent flow
        # Original cache logic (disabled):
        # use_cache = (not enable_multi_turn or not session_id) and self._should_use_cache()
        # cached_result = None
        # cache_key = None
        # repo_hash = None
        # if use_cache:
        #     repo_hash = self._get_repo_hash()
        #     cache_key = f"{question}_{','.join(sorted(repo_filter)) if repo_filter else 'all'}"
        #     cached_result = self.cache_manager.get_query_result(cache_key, repo_hash)
        #     if cached_result:
        #         self.logger.info("Returning cached result")
        # result = cached_result

        result = None  # Always process through full flow
        processed_query = None
        retrieved: list[dict[str, Any]] = []
        turn_bundle: dict[str, Any] | None = None
        prior_compiled_context = self._load_prior_compiled_context(session_id)

        try:
            if result is None:
                # Determine if iterative enhancement should be used
                use_iterative_enhancement = (
                    active_retriever.enable_agency_mode
                    and active_retriever.iterative_agent is not None
                )

                # Process query: skip query_processor entirely in iterative mode
                if use_iterative_enhancement:
                    # Iterative agent will handle all query enhancement
                    # Create minimal ProcessedQuery object
                    from fastcode.app.query.orchestration.processor import (
                        ProcessedQuery,
                    )

                    intent = "unknown"
                    detect_intent = getattr(
                        self.query_processor, "_detect_intent", None
                    )
                    if callable(detect_intent):
                        intent = detect_intent(question)
                    processed_query = ProcessedQuery(
                        original=question,
                        expanded=question,
                        keywords=[],
                        intent=str(intent or "unknown"),
                        subqueries=[],
                        filters=filters or {},
                        rewritten_query=None,
                        pseudocode_hints=None,
                        search_strategy=None,
                    )
                    self.logger.info(
                        "Iterative mode: skipping query_processor, all enhancements handled by iterative_agent"
                    )
                else:
                    # Standard mode: use full query processing
                    processed_query = self.query_processor.process(
                        question, dialogue_history, use_llm_enhancement=True
                    )
                    self.logger.info(f"Query intent: {processed_query.intent}")
                    self.logger.info(f"Keywords: {processed_query.keywords}")

                # Retrieve relevant code (with repository filter and agency mode)
                # Pass ProcessedQuery object for enhanced retrieval
                # Pass dialogue_history for multi-turn context in iterative mode
                retrieved = active_retriever.retrieve(
                    processed_query,  # Pass full ProcessedQuery object for multi-repo support
                    filters=filters,
                    repo_filter=repo_filter,
                    use_agency_mode=use_agency_mode,
                    dialogue_history=dialogue_history if enable_multi_turn else None,
                    compiled_context=prior_compiled_context,
                )
                semantic_escalation = self._maybe_escalate_query_semantics(
                    question=question,
                    processed_query=processed_query,
                    filters=filters,
                    retrieved=retrieved,
                    retriever=active_retriever,
                    graph_builder=graph_builder,
                )
                if semantic_escalation and semantic_escalation.get("rerun_retrieval"):
                    retrieved = active_retriever.retrieve(
                        processed_query,
                        filters=filters,
                        repo_filter=repo_filter,
                        use_agency_mode=use_agency_mode,
                        dialogue_history=(
                            dialogue_history if enable_multi_turn else None
                        ),
                        compiled_context=prior_compiled_context,
                    )

                turn_bundle = self._build_turn_artifact_bundle(
                    question=question,
                    processed_query=processed_query,
                    retrieved=retrieved,
                    session_id=session_id,
                    filters=filters,
                    repo_filter=repo_filter,
                    retriever=active_retriever,
                )

                # Generate answer (with dialogue history for multi-turn)
                result = self.answer_generator.generate(
                    question,
                    retrieved,
                    query_info=processed_query_payload(processed_query),
                    dialogue_history=self._get_full_dialogue_history(
                        session_id, enable_multi_turn or False
                    ),
                    compiled_context=cast(str, turn_bundle["compiled_context"]),
                    prompt_builder=prompt_builder,
                )
                if semantic_escalation is not None:
                    result["semantic_escalation"] = semantic_escalation
                result["turn_number"] = cast(int, turn_bundle["turn_number"])
                if session_id:
                    result["session_id"] = session_id

            # Add repository information to result
            if repo_filter:
                result["searched_repositories"] = repo_filter

            if processed_query is not None and turn_bundle is not None:
                self._persist_turn_artifacts(
                    session_id=session_id,
                    question=question,
                    answer=str(result.get("answer", "")),
                    summary=(
                        str(result["summary"])
                        if result.get("summary") is not None
                        else None
                    ),
                    sources=self._source_citations_from_payloads(result.get("sources"))
                    or cast(tuple[SourceCitation, ...], turn_bundle["sources"]),
                    processed_query=processed_query,
                    repo_filter=repo_filter,
                    enable_multi_turn=bool(enable_multi_turn),
                    bundle=turn_bundle,
                )

            # Cache result for stateless flows (including single-turn sessions)
            # NOTE: Query result caching is disabled to ensure full iterative_agent flow
            # if use_cache and result is not None and cache_key and repo_hash:
            #     self.cache_manager.set_query_result(cache_key, repo_hash, result)

            return result

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return {
                "answer": f"Error processing query: {e!s}",
                "query": question,
                "error": str(e),
            }

    def query_stream(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        repo_filter: list[str] | None = None,
        session_id: str | None = None,
        enable_multi_turn: bool | None = None,
        use_agency_mode: bool | None = None,
        prompt_builder: Callable[
            [str, str, dict[str, Any] | None, list[dict[str, Any]] | None], str
        ]
        | None = None,
        retriever: HybridRetriever | None = None,
    ) -> Generator[tuple[str | None, dict[str, Any] | None], Any, None]:
        """
        Query the repository with streaming response (yields answer chunks)

        Args:
            question: User question
            filters: Optional filters for retrieval
            repo_filter: Optional list of repository names to search in
            session_id: Optional session ID for multi-turn dialogue
            enable_multi_turn: Override config setting for multi-turn mode
            use_agency_mode: Override config setting for agency mode
            prompt_builder: Optional callable to build custom LLM prompt

        Yields:
            Tuples of (chunk_text or None, metadata_dict or None)
            - First yield: (None, {"status": "retrieving"})
            - After retrieval: (None, {"status": "generating", "sources": [...], ...})
            - During generation: (text_chunk, None)
            - Final yield: (None, {"status": "complete", "summary": ..., ...})
        """
        active_retriever = retriever or self.retriever
        if retriever is None and not self.is_repo_indexed():
            yield (
                None,
                {"error": "Repository not indexed. Call index_repository() first."},
            )
            return

        # Determine if multi-turn mode is enabled
        if enable_multi_turn is None:
            enable_multi_turn = self.config.get("generation", {}).get(
                "enable_multi_turn", False
            )

        if repo_filter:
            self.logger.info(
                f"Processing streaming query: {question} in repositories: {repo_filter}"
            )
        else:
            self.logger.info(f"Processing streaming query: {question}")

        # Get dialogue history if in multi-turn mode
        dialogue_history: list[dict[str, Any]] = []
        if enable_multi_turn and session_id:
            history_summary_rounds = self.config.get("query", {}).get(
                "history_summary_rounds", 10
            )
            dialogue_history = self.cache_manager.get_recent_summaries(
                session_id, history_summary_rounds
            )
            if dialogue_history:
                self.logger.info(
                    f"Retrieved {len(dialogue_history)} previous dialogue summaries"
                )

        try:
            # Notify start of retrieval
            yield None, {"status": "retrieving", "query": question}
            prior_compiled_context = self._load_prior_compiled_context(session_id)

            # Retrieval phase (same as query method)
            use_iterative_enhancement = (
                active_retriever.enable_agency_mode
                and active_retriever.iterative_agent is not None
            )

            if use_iterative_enhancement:
                from fastcode.app.query.orchestration.processor import ProcessedQuery

                intent = "unknown"
                detect_intent = getattr(self.query_processor, "_detect_intent", None)
                if callable(detect_intent):
                    intent = detect_intent(question)
                processed_query = ProcessedQuery(
                    original=question,
                    expanded=question,
                    keywords=[],
                    intent=str(intent or "unknown"),
                    subqueries=[],
                    filters=filters or {},
                    rewritten_query=None,
                    pseudocode_hints=None,
                    search_strategy=None,
                )
                self.logger.info(
                    "Iterative mode: skipping query_processor, all enhancements handled by iterative_agent"
                )
            else:
                processed_query = self.query_processor.process(
                    question, dialogue_history, use_llm_enhancement=True
                )
                self.logger.info(f"Query intent: {processed_query.intent}")
                self.logger.info(f"Keywords: {processed_query.keywords}")

            # Retrieve relevant code
            retrieved = active_retriever.retrieve(
                processed_query,
                filters=filters,
                repo_filter=repo_filter,
                use_agency_mode=use_agency_mode,
                dialogue_history=dialogue_history if enable_multi_turn else None,
                compiled_context=prior_compiled_context,
            )
            turn_bundle = self._build_turn_artifact_bundle(
                question=question,
                processed_query=processed_query,
                retrieved=retrieved,
                session_id=session_id,
                filters=filters,
                repo_filter=repo_filter,
                retriever=active_retriever,
            )

            # Notify start of generation
            yield (
                None,
                {
                    "status": "generating",
                    "retrieved_count": len(retrieved),
                    "turn_number": turn_bundle["turn_number"],
                },
            )

            # Stream answer generation
            full_answer_parts: list[str] = []
            answer_metadata: dict[str, Any] = {}

            for chunk, metadata in self.answer_generator.generate_stream(
                question,
                retrieved,
                query_info=processed_query_payload(processed_query),
                dialogue_history=self._get_full_dialogue_history(
                    session_id, enable_multi_turn or False
                ),
                compiled_context=cast(str, turn_bundle["compiled_context"]),
                prompt_builder=prompt_builder,
            ):
                if chunk:
                    full_answer_parts.append(chunk)
                    yield chunk, None
                if metadata:
                    answer_metadata.update(metadata)

            # Build complete result
            full_answer = "".join(full_answer_parts)
            summary = answer_metadata.get("summary")

            result: dict[str, Any] = {
                "status": "complete",
                "answer": full_answer,
                "query": question,
                "context_elements": len(retrieved),
                "turn_number": turn_bundle["turn_number"],
                "sources": answer_metadata.get(
                    "sources", self._extract_sources_from_elements(retrieved)
                ),
            }

            if summary:
                result["summary"] = summary

            if repo_filter:
                result["searched_repositories"] = repo_filter

            if session_id:
                result["session_id"] = session_id

            self._persist_turn_artifacts(
                session_id=session_id,
                question=question,
                answer=full_answer,
                summary=str(summary) if summary is not None else None,
                sources=self._source_citations_from_payloads(result.get("sources"))
                or cast(tuple[SourceCitation, ...], turn_bundle["sources"]),
                processed_query=processed_query,
                repo_filter=repo_filter,
                enable_multi_turn=bool(enable_multi_turn),
                bundle=turn_bundle,
            )

            # Final yield with complete result
            yield None, result

        except Exception as e:
            self.logger.error(f"Streaming query failed: {e}")
            error_trace = traceback.format_exc()
            self.logger.error(f"Full error traceback:\n{error_trace}")
            yield (
                None,
                {
                    "status": "error",
                    "error": str(e),
                    "query": question,
                },
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_payload_json(payload: dict[str, Any]) -> str:
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _fingerprint_payload(prefix: str, payload: Any) -> str:
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}:{digest}"

    @staticmethod
    def _mapping_or_empty(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        return {
            str(key): item
            for key, item in cast(Mapping[Any, Any], value).items()
            if isinstance(key, str | int | float | bool)
        }

    @staticmethod
    def _decode_distillation_record(
        record: ContextDistillationRecord | None,
    ) -> DistillationRecord | None:
        if record is None:
            return None
        try:
            payload = json.loads(str(record.payload_json or "{}"))
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        return distillation_from_payload(cast(dict[str, Any], payload))

    @staticmethod
    def _resolve_snapshot_scope(
        filters: dict[str, Any] | None,
    ) -> tuple[str | None, str | None]:
        snapshot_value = (filters or {}).get("snapshot_id")
        artifact_value = (filters or {}).get("artifact_key")
        snapshot_id = (
            str(snapshot_value)
            if isinstance(snapshot_value, str) and snapshot_value
            else None
        )
        artifact_key = (
            str(artifact_value)
            if isinstance(artifact_value, str) and artifact_value
            else None
        )
        return snapshot_id, artifact_key

    def _safe_get_session_prefix(
        self,
        snapshot_id: str | None,
    ) -> dict[str, Any] | None:
        if not snapshot_id or self.get_session_prefix is None:
            return None
        try:
            payload = self.get_session_prefix(snapshot_id)
        except Exception as exc:
            self.logger.info(
                f"Skipping session prefix for snapshot {snapshot_id}: {exc}"
            )
            return None
        if payload.get("error"):
            return None
        return payload

    def _context_bundle_fingerprints(
        self,
        *,
        session_prefix: dict[str, Any] | None,
        retrieved: list[dict[str, Any]],
        filters: dict[str, Any] | None,
        repo_filter: list[str] | None,
        retriever: HybridRetriever,
    ) -> dict[str, str]:
        projection_payload: dict[str, Any] = {
            "snapshot_id": (filters or {}).get("snapshot_id"),
            "artifact_key": (filters or {}).get("artifact_key"),
            "projection_id": None,
        }
        if isinstance(session_prefix, dict):
            projection_payload["projection_id"] = session_prefix.get("projection_id")

        embedding_refs: list[Any] = []
        for item in retrieved:
            raw_element = item.get("element")
            if not isinstance(raw_element, dict):
                continue
            element = cast(dict[str, Any], raw_element)
            raw_metadata = element.get("metadata")
            metadata_payload = (
                cast(dict[str, Any], raw_metadata)
                if isinstance(raw_metadata, dict)
                else {}
            )
            fingerprint = element.get("embedding_fingerprint") or metadata_payload.get(
                "embedding_fingerprint"
            )
            artifact_ref = element.get(
                "embedding_artifact_ref"
            ) or metadata_payload.get("embedding_artifact_ref")
            if fingerprint is not None or artifact_ref is not None:
                embedding_refs.append(
                    {
                        "path": element.get("relative_path") or element.get("path"),
                        "fingerprint": fingerprint,
                        "artifact_ref": artifact_ref,
                    }
                )

        retrieval_config = self._mapping_or_empty(
            getattr(retriever, "retrieval_config", {})
        )
        iteration_value = getattr(retriever, "last_iteration_metadata", None)
        iteration_metadata = (
            self._mapping_or_empty(iteration_value)
            if isinstance(iteration_value, Mapping)
            else None
        )
        retrieval_policy_payload = {
            "repo_filter": list(repo_filter or []),
            "config": retrieval_config,
            "iteration_metadata": iteration_metadata,
        }
        generation_config = self.config.get("generation", {})
        token_budget = generation_config.get("context_token_budget")
        if token_budget is None:
            token_budget = generation_config.get("max_context_tokens")

        return {
            "projection_fingerprint": (
                self._fingerprint_payload("projection", projection_payload)
                if any(value is not None for value in projection_payload.values())
                else DEFAULT_PROJECTION_FINGERPRINT
            ),
            "embedding_fingerprint": (
                self._fingerprint_payload("embedding", embedding_refs)
                if embedding_refs
                else DEFAULT_EMBEDDING_FINGERPRINT
            ),
            "retrieval_policy_fingerprint": self._fingerprint_payload(
                "retrieval", retrieval_policy_payload
            ),
            "distillation_prompt_fingerprint": DEFAULT_DISTILLATION_PROMPT_FINGERPRINT,
            "budget_fingerprint": (
                self._fingerprint_payload("budget", {"token_budget": token_budget})
                if token_budget is not None
                else DEFAULT_BUDGET_FINGERPRINT
            ),
        }

    def _load_prior_compiled_context(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        prior_record = self.cache_manager.get_latest_working_memory_record(session_id)
        if prior_record is None or not prior_record.full_fcx:
            return None
        return prior_record.full_fcx

    @staticmethod
    def _answer_summary(summary: str | None, answer: str) -> str:
        if summary:
            return str(summary)
        normalized = " ".join(str(answer or "").split())
        if len(normalized) <= 280:
            return normalized
        return normalized[:277] + "..."

    @staticmethod
    def _source_citation_payloads(
        sources: tuple[SourceCitation, ...],
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for source in sources:
            payloads.append(
                {
                    "repository": source.repository,
                    "repo": source.repository,
                    "file": source.file,
                    "name": source.name,
                    "type": source.element_type,
                    "lines": source.lines,
                    "score": source.score,
                }
            )
        return payloads

    @staticmethod
    def _source_citation_from_payload(payload: Any) -> SourceCitation | None:
        if not isinstance(payload, Mapping):
            return None
        mapping = cast(Mapping[str, Any], payload)
        repository = str(mapping.get("repository") or mapping.get("repo") or "")
        file_path = str(mapping.get("file") or mapping.get("path") or "")
        name = str(mapping.get("name") or "")
        element_type = str(mapping.get("type") or mapping.get("element_type") or "")
        lines_value = mapping.get("lines")
        start_line = mapping.get("start_line")
        end_line = mapping.get("end_line")
        if (
            not lines_value
            and isinstance(start_line, int)
            and isinstance(end_line, int)
        ):
            lines_value = f"{start_line}-{end_line}"
        lines = str(lines_value or "")
        score_value = mapping.get("score")
        if score_value is None:
            score_value = mapping.get("total_score")
        score = float(score_value) if isinstance(score_value, (int, float)) else 0.0
        return SourceCitation(
            repository=repository,
            file=file_path,
            name=name,
            element_type=element_type,
            lines=lines,
            score=score,
        )

    @classmethod
    def _source_citations_from_payloads(
        cls,
        payloads: Any,
    ) -> tuple[SourceCitation, ...]:
        if not isinstance(payloads, list | tuple):
            return ()
        records: list[SourceCitation] = []
        for payload in cast(tuple[Any, ...] | list[Any], payloads):
            if isinstance(payload, SourceCitation):
                records.append(payload)
                continue
            record = cls._source_citation_from_payload(payload)
            if record is not None:
                records.append(record)
        return tuple(records)

    def _build_tool_observations(
        self,
        *,
        retriever: HybridRetriever,
        evidence_refs: tuple[Any, ...],
        retrieved: list[dict[str, Any]],
        repo_filter: list[str] | None,
    ) -> tuple[ToolObservation, ...]:
        observations: list[ToolObservation] = []
        ref_ids = tuple(item.ref_id for item in evidence_refs)
        iteration_metadata = self._mapping_or_empty(
            getattr(retriever, "last_iteration_metadata", None)
        )
        retrieval_mode = (
            "iterative"
            if getattr(retriever, "last_iteration_metadata", None) is not None
            else "standard"
        )
        retrieval_summary = f"Retrieved {len(retrieved)} repository elements via {retrieval_mode} retrieval."
        if iteration_metadata:
            rounds = int(iteration_metadata.get("rounds") or 0)
            retrieval_summary = (
                f"Retrieved {len(retrieved)} repository elements via iterative "
                f"retrieval across {rounds} rounds."
            )
        observations.append(
            build_tool_observation(
                observation_id="o_retrieve",
                tool="retrieve",
                ok=bool(retrieved),
                parameters={
                    "mode": retrieval_mode,
                    "repo_filter": list(repo_filter or []),
                    "result_count": len(retrieved),
                    "rounds": int(iteration_metadata.get("rounds") or 0),
                },
                ref_ids=ref_ids[: min(len(ref_ids), 6)],
                summary=retrieval_summary,
                round_number=0,
            )
        )

        path_to_ref_ids: dict[str, list[str]] = {}
        for item in evidence_refs:
            if item.path:
                path_to_ref_ids.setdefault(item.path, []).append(item.ref_id)

        raw_tool_value = getattr(retriever, "last_tool_observations", [])
        raw_tool_observations = (
            cast(list[dict[str, Any]], raw_tool_value)
            if isinstance(raw_tool_value, list)
            else []
        )
        for index, raw in enumerate(raw_tool_observations, 1):
            parameters_value = raw.get("parameters")
            parameters = (
                {
                    str(key): value
                    for key, value in cast(dict[Any, Any], parameters_value).items()
                }
                if isinstance(parameters_value, dict)
                else {}
            )
            sample_paths = [
                str(path)
                for path in raw.get("sample_paths", [])
                if isinstance(path, str) and path
            ]
            matched_ref_ids: list[str] = []
            for path in sample_paths:
                matched_ref_ids.extend(path_to_ref_ids.get(path, []))
            if not matched_ref_ids and ref_ids:
                matched_ref_ids.extend(ref_ids[: min(len(ref_ids), 4)])
            candidate_count = int(raw.get("candidate_count") or 0)
            tool_name = str(raw.get("tool") or f"tool_{index}")
            summary = f"{tool_name} produced {candidate_count} candidate paths."
            if sample_paths:
                summary = (
                    f"{tool_name} produced {candidate_count} candidate paths, "
                    f"including {', '.join(sample_paths[:2])}."
                )
            observations.append(
                build_tool_observation(
                    observation_id=f"o_tool_{index}",
                    tool=tool_name,
                    ok=bool(raw.get("ok", False)),
                    parameters=dict(parameters),
                    ref_ids=tuple(matched_ref_ids),
                    summary=summary,
                    round_number=int(raw.get("round_number") or 0),
                )
            )
        return tuple(observations)

    def _build_turn_artifact_bundle(
        self,
        *,
        question: str,
        processed_query: ProcessedQuery,
        retrieved: list[dict[str, Any]],
        session_id: str | None,
        filters: dict[str, Any] | None,
        repo_filter: list[str] | None,
        retriever: HybridRetriever,
    ) -> dict[str, Any]:
        turn_number = self._get_next_turn_number(session_id) if session_id else 1
        snapshot_id, artifact_key = self._resolve_snapshot_scope(filters)
        prior_compiled_context = self._load_prior_compiled_context(session_id)

        sources = self._extract_sources_from_elements(retrieved)
        evidence_refs = build_evidence_refs_from_sources(
            sources,
            snapshot_id=snapshot_id,
        )
        observations = self._build_tool_observations(
            retriever=retriever,
            evidence_refs=evidence_refs,
            retrieved=retrieved,
            repo_filter=repo_filter,
        )
        accepted_facts, hypotheses, rejected_hypotheses = promote_observations(
            question=question,
            evidence_refs=evidence_refs,
            observations=observations,
            snapshot_id=snapshot_id,
        )
        risk_state = compute_risk_state(
            question=question,
            snapshot_id=snapshot_id,
            evidence_refs=evidence_refs,
            hypotheses=hypotheses,
            accepted_facts=accepted_facts,
            rejected_hypotheses=rejected_hypotheses,
        )
        acceptance_contract = build_acceptance_contract(requested_outcome="answer")
        plan = build_turn_plan(
            risk_state=risk_state,
            contract=acceptance_contract,
        )
        unresolved_questions: tuple[str, ...] = ()
        if risk_state.action_bias != "answer":
            unresolved_questions = (
                f"Need additional repository evidence for: {question}",
            )
        turn_intent = TurnIntent(
            session_id=session_id or "stateless",
            turn_number=turn_number,
            question=question,
            kind=processed_query.intent,
            requested_outcome="answer",
            snapshot_id=snapshot_id,
            artifact_key=artifact_key,
            repo_filter=tuple(repo_filter or ()),
        )
        session_prefix = self._safe_get_session_prefix(snapshot_id)
        working_memory = compile_working_memory(
            intent=turn_intent,
            contract=acceptance_contract,
            risk_state=risk_state,
            plan=plan,
            evidence_refs=evidence_refs,
            observations=observations,
            accepted_facts=accepted_facts,
            hypotheses=hypotheses,
            rejected_hypotheses=rejected_hypotheses,
            unresolved_questions=unresolved_questions,
            session_prefix=session_prefix,
        )
        context_fingerprints = self._context_bundle_fingerprints(
            session_prefix=session_prefix,
            retrieved=retrieved,
            filters=filters,
            repo_filter=repo_filter,
            retriever=retriever,
        )
        compiled_context = "\n\n".join(
            part for part in (prior_compiled_context, working_memory.full_fcx) if part
        )
        return {
            "turn_number": turn_number,
            "sources": sources,
            "compiled_context": compiled_context or working_memory.full_fcx,
            "intent": turn_intent,
            "working_memory": working_memory,
            "plan": plan,
            "observations": observations,
            "evidence_refs": evidence_refs,
            "risk_state": risk_state,
            "acceptance_contract": acceptance_contract,
            "hypotheses": hypotheses,
            "rejected_hypotheses": rejected_hypotheses,
            "accepted_facts": accepted_facts,
            "context_fingerprints": context_fingerprints,
        }

    def _persist_turn_artifacts(
        self,
        *,
        session_id: str | None,
        question: str,
        answer: str,
        summary: str | None,
        sources: tuple[SourceCitation, ...],
        processed_query: ProcessedQuery,
        repo_filter: list[str] | None,
        enable_multi_turn: bool,
        bundle: dict[str, Any],
    ) -> None:
        if not session_id:
            return

        working_memory = bundle["working_memory"]
        journal = build_turn_journal(
            intent=bundle["intent"],
            plan=bundle["plan"],
            observations=bundle["observations"],
            evidence_refs=bundle["evidence_refs"],
            risk_state=bundle["risk_state"],
            acceptance_contract=bundle["acceptance_contract"],
            hypotheses=bundle["hypotheses"],
            rejected_hypotheses=bundle["rejected_hypotheses"],
            accepted_facts=bundle["accepted_facts"],
            working_set=working_memory.working_set,
            answer_summary=self._answer_summary(summary, answer),
            created_at=working_memory.created_at,
        )

        working_memory_record = WorkingMemoryRecord(
            session_id=working_memory.session_id,
            turn_number=working_memory.turn_number,
            snapshot_id=working_memory.snapshot_id,
            artifact_key=working_memory.artifact_key,
            compiler_fingerprint=working_memory.compiler_fingerprint,
            payload_json=self._encode_payload_json(
                working_memory_payload(working_memory)
            ),
            stable_fcx=working_memory.stable_fcx,
            turn_fcx=working_memory.turn_fcx,
            obs_fcx=working_memory.obs_fcx,
            full_fcx=working_memory.full_fcx,
            created_at=working_memory.created_at,
        )
        journal_record = TurnJournalRecord(
            session_id=journal.session_id,
            turn_number=journal.turn_number,
            snapshot_id=journal.snapshot_id,
            artifact_key=journal.artifact_key,
            compiler_fingerprint=COMPILER_FINGERPRINT,
            payload_json=self._encode_payload_json(turn_journal_payload(journal)),
            created_at=journal.created_at,
        )
        invalidation_key = build_context_invalidation_key(
            session_id=working_memory.session_id,
            snapshot_id=working_memory.snapshot_id,
            artifact_key=working_memory.artifact_key,
            evidence_refs=working_memory.evidence_refs,
            compiler_fingerprint=working_memory.compiler_fingerprint,
            **bundle["context_fingerprints"],
        )
        prior_distillation = self._decode_distillation_record(
            self.cache_manager.find_reusable_context_distillation_record(
                working_memory.session_id,
                invalidation_key=invalidation_key,
                compiler_fingerprint=working_memory.compiler_fingerprint,
            )
        )
        context_bundle = build_context_bundle(
            working_memory=working_memory,
            turn_journal=journal,
            previous_distillation=prior_distillation,
            **bundle["context_fingerprints"],
        )
        context_bundle_record = ContextBundleRecord(
            bundle_id=context_bundle.bundle_id,
            session_id=context_bundle.session_id,
            turn_number=context_bundle.turn_number,
            snapshot_id=context_bundle.snapshot_id,
            artifact_key=context_bundle.artifact_key,
            compiler_fingerprint=context_bundle.compiler_fingerprint,
            payload_json=self._encode_payload_json(
                context_bundle_payload(context_bundle)
            ),
            invalidation_key=context_bundle.distillation.invalidation_key,
            created_at=context_bundle.created_at,
            projection_fingerprint=context_bundle.projection_fingerprint,
            embedding_fingerprint=context_bundle.embedding_fingerprint,
            retrieval_policy_fingerprint=context_bundle.retrieval_policy_fingerprint,
            distillation_prompt_fingerprint=(
                context_bundle.distillation_prompt_fingerprint
            ),
            budget_fingerprint=context_bundle.budget_fingerprint,
        )
        context_distillation_record = ContextDistillationRecord(
            distillation_id=context_bundle.distillation.distillation_id,
            session_id=context_bundle.distillation.session_id,
            turn_number=context_bundle.distillation.turn_number,
            snapshot_id=context_bundle.distillation.snapshot_id,
            compiler_fingerprint=context_bundle.distillation.compiler_fingerprint,
            summary=context_bundle.distillation.summary,
            payload_json=self._encode_payload_json(
                distillation_payload(context_bundle.distillation)
            ),
            invalidation_key=context_bundle.distillation.invalidation_key,
            source_ref_ids=tuple(
                ref.ref_id for ref in context_bundle.distillation.source_refs
            ),
            reused_from_distillation_id=(
                context_bundle.distillation.reused_from_distillation_id
            ),
            created_at=context_bundle.distillation.created_at,
            projection_fingerprint=context_bundle.distillation.projection_fingerprint,
            embedding_fingerprint=context_bundle.distillation.embedding_fingerprint,
            retrieval_policy_fingerprint=(
                context_bundle.distillation.retrieval_policy_fingerprint
            ),
            distillation_prompt_fingerprint=(
                context_bundle.distillation.distillation_prompt_fingerprint
            ),
            budget_fingerprint=context_bundle.distillation.budget_fingerprint,
        )
        context_activation_record = ContextActivationRecord(
            activation_id=context_bundle.activation.activation_id,
            bundle_id=context_bundle.activation.bundle_id,
            session_id=context_bundle.activation.session_id,
            turn_number=context_bundle.activation.turn_number,
            snapshot_id=context_bundle.activation.snapshot_id,
            compiler_fingerprint=context_bundle.activation.compiler_fingerprint,
            active_ref_ids=context_bundle.activation.active_ref_ids,
            active_fact_ids=context_bundle.activation.active_fact_ids,
            active_hypothesis_ids=context_bundle.activation.active_hypothesis_ids,
            reason=context_bundle.activation.reason,
            payload_json=self._encode_payload_json(
                activation_payload(context_bundle.activation)
            ),
            created_at=context_bundle.activation.created_at,
        )
        self.cache_manager.save_working_memory_record(working_memory_record)
        self.cache_manager.save_turn_journal_record(journal_record)
        self.cache_manager.save_context_bundle_record(context_bundle_record)
        self.cache_manager.save_context_distillation_record(context_distillation_record)
        self.cache_manager.save_context_activation_record(context_activation_record)

        dialogue_summary = self._answer_summary(summary, answer)
        metadata = {
            "intent": getattr(processed_query, "intent", None),
            "keywords": getattr(processed_query, "keywords", None),
            "repo_filter": repo_filter,
            "multi_turn": enable_multi_turn,
        }
        self.cache_manager.save_dialogue_turn(
            session_id=session_id,
            turn_number=bundle["turn_number"],
            query=question,
            answer=answer,
            summary=dialogue_summary,
            retrieved_elements=self._source_citation_payloads(sources),
            metadata=metadata,
        )
        self.logger.info(
            f"Saved typed working memory and dialogue turn {bundle['turn_number']} for session {session_id}"
        )

    def _get_full_dialogue_history(
        self, session_id: str | None, enable_multi_turn: bool = False
    ) -> list[dict[str, Any]] | None:
        """
        Get full dialogue history for answer generation

        Args:
            session_id: Session ID
            enable_multi_turn: Whether multi-turn is enabled

        Returns:
            List of dialogue turns or None
        """
        if not enable_multi_turn or not session_id:
            return None

        context_rounds = self.config.get("generation", {}).get("context_rounds", 10)
        history = [
            self._dialogue_turn_payload(record)
            for record in self.cache_manager.get_dialogue_history_records(
                session_id, max_turns=context_rounds
            )
        ]

        return history or None

    @staticmethod
    def _dialogue_turn_payload(record: Any) -> dict[str, Any]:
        return {
            "session_id": str(record.session_id),
            "turn_number": int(record.turn_number),
            "timestamp": float(record.timestamp),
            "query": str(record.query),
            "answer": str(record.answer),
            "summary": str(record.summary),
            "retrieved_elements": list(record.retrieved_elements),
            "metadata": dict(record.metadata),
        }

    def _get_next_turn_number(self, session_id: str) -> int:
        """
        Get the next turn number for a session

        Args:
            session_id: Session ID

        Returns:
            Next turn number (1-indexed)
        """
        session_record = cast(
            Any, self.cache_manager.get_session_index_record(session_id)
        )
        if session_record is not None:
            return int(session_record.total_turns) + 1
        return 1

    def _extract_sources_from_elements(
        self, elements: list[dict[str, Any]]
    ) -> tuple[SourceCitation, ...]:
        """Extract source information from retrieved elements"""
        hits = tuple(Hit.from_retrieval_row(item) for item in elements)
        return _snapshot.extract_sources_from_elements(hits)
