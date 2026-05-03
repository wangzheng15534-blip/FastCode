"""
QueryPipeline — query, query_stream, and query_snapshot extracted from FastCode.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any

from .answer_generator import AnswerGenerator
from .cache import CacheManager
from .core import snapshot as _snapshot
from .manifest_store import ManifestStore
from .query_processor import QueryProcessor
from .retriever import HybridRetriever
from .snapshot_store import SnapshotStore
from .snapshot_symbol_index import SnapshotSymbolIndex
from .utils import safe_jsonable

if TYPE_CHECKING:
    from .query_processor import ProcessedQuery


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
        self.semantic_escalation_cb = semantic_escalation_cb

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

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
                raise RuntimeError(
                    "query_snapshot requires snapshot_id or repo_name+ref_name"
                )
            manifest = self.manifest_store.get_branch_manifest_record(
                repo_name, ref_name
            )
            if not manifest:
                raise RuntimeError(f"manifest not found for {repo_name}:{ref_name}")
            snapshot_id = manifest.snapshot_id

        snapshot_record = self.snapshot_store.get_snapshot_record(snapshot_id)
        if not snapshot_record:
            raise RuntimeError(f"snapshot not found: {snapshot_id}")

        if not self.load_artifacts_by_key(snapshot_record["artifact_key"]):
            raise RuntimeError(f"failed to load artifacts for snapshot: {snapshot_id}")
        if not self.snapshot_symbol_index.has_snapshot(snapshot_id):
            loaded_snapshot = self.snapshot_store.load_snapshot(snapshot_id)
            if loaded_snapshot:
                self.snapshot_symbol_index.register_snapshot(loaded_snapshot)

        merged_filters = dict(filters or {})
        merged_filters["snapshot_id"] = snapshot_id

        result = self.query(
            question=question,
            filters=merged_filters,
            repo_filter=None,
            session_id=session_id,
            enable_multi_turn=enable_multi_turn,
        )
        result["snapshot_id"] = snapshot_id
        result["artifact_key"] = snapshot_record["artifact_key"]
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
        if not self.is_repo_indexed():
            raise RuntimeError("Repository not indexed. Call index_repository() first.")

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

        try:
            if result is None:
                # Determine if iterative enhancement should be used
                use_iterative_enhancement = (
                    self.retriever.enable_agency_mode
                    and self.retriever.iterative_agent is not None
                )

                # Process query: skip query_processor entirely in iterative mode
                if use_iterative_enhancement:
                    # Iterative agent will handle all query enhancement
                    # Create minimal ProcessedQuery object
                    from .query_processor import ProcessedQuery

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
                    # Standard mode: use full query processing
                    processed_query = self.query_processor.process(
                        question, dialogue_history, use_llm_enhancement=True
                    )
                    self.logger.info(f"Query intent: {processed_query.intent}")
                    self.logger.info(f"Keywords: {processed_query.keywords}")

                # Retrieve relevant code (with repository filter and agency mode)
                # Pass ProcessedQuery object for enhanced retrieval
                # Pass dialogue_history for multi-turn context in iterative mode
                retrieved = self.retriever.retrieve(
                    processed_query,  # Pass full ProcessedQuery object for multi-repo support
                    filters=filters,
                    repo_filter=repo_filter,
                    use_agency_mode=use_agency_mode,
                    dialogue_history=dialogue_history if enable_multi_turn else None,
                )
                semantic_escalation = self._maybe_escalate_query_semantics(
                    question=question,
                    processed_query=processed_query,
                    filters=filters,
                    retrieved=retrieved,
                )
                if semantic_escalation and semantic_escalation.get("rerun_retrieval"):
                    retrieved = self.retriever.retrieve(
                        processed_query,
                        filters=filters,
                        repo_filter=repo_filter,
                        use_agency_mode=use_agency_mode,
                        dialogue_history=(
                            dialogue_history if enable_multi_turn else None
                        ),
                    )

                # Generate answer (with dialogue history for multi-turn)
                result = self.answer_generator.generate(
                    question,
                    retrieved,
                    query_info=processed_query.to_dict(),
                    dialogue_history=self._get_full_dialogue_history(
                        session_id, enable_multi_turn or False
                    ),
                    prompt_builder=prompt_builder,
                )
                if semantic_escalation is not None:
                    result["semantic_escalation"] = semantic_escalation

            # Add repository information to result
            if repo_filter:
                result["searched_repositories"] = repo_filter

            # Persist dialogue for any session (even single-turn) so users keep history
            if session_id:
                turn_number = self._get_next_turn_number(session_id)
                summary = result.get("summary", "")
                # Use formatted sources from result instead of raw retrieved elements
                # This ensures proper display format when loading history
                sources = result.get("sources", [])
                # Ensure sources are fully JSON-serializable
                serializable_sources = safe_jsonable(sources)

                # Ensure metadata is JSON-serializable
                metadata = {
                    "intent": getattr(processed_query, "intent", None),
                    "keywords": getattr(processed_query, "keywords", None),
                    "repo_filter": repo_filter,
                    "multi_turn": enable_multi_turn,
                }
                serializable_metadata = safe_jsonable(metadata)

                self.cache_manager.save_dialogue_turn(
                    session_id=session_id,
                    turn_number=turn_number,
                    query=question,
                    answer=result.get("answer", ""),
                    summary=summary,
                    retrieved_elements=serializable_sources,
                    metadata=serializable_metadata,
                )

                self.logger.info(
                    f"Saved dialogue turn {turn_number} for session {session_id}"
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
        if not self.is_repo_indexed():
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

            # Retrieval phase (same as query method)
            use_iterative_enhancement = (
                self.retriever.enable_agency_mode
                and self.retriever.iterative_agent is not None
            )

            if use_iterative_enhancement:
                from .query_processor import ProcessedQuery

                processed_query = ProcessedQuery(
                    original=question,
                    expanded=question,
                    keywords=[],
                    intent="unknown",
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
            retrieved = self.retriever.retrieve(
                processed_query,
                filters=filters,
                repo_filter=repo_filter,
                use_agency_mode=use_agency_mode,
                dialogue_history=dialogue_history if enable_multi_turn else None,
            )

            # Notify start of generation
            yield None, {"status": "generating", "retrieved_count": len(retrieved)}

            # Stream answer generation
            full_answer_parts: list[str] = []
            answer_metadata: dict[str, Any] = {}

            for chunk, metadata in self.answer_generator.generate_stream(
                question,
                retrieved,
                query_info=processed_query.to_dict(),
                dialogue_history=self._get_full_dialogue_history(
                    session_id, enable_multi_turn or False
                ),
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
                "sources": answer_metadata.get(
                    "sources", self._extract_sources_from_elements(retrieved)
                ),
            }

            if summary:
                result["summary"] = summary

            if repo_filter:
                result["searched_repositories"] = repo_filter

            # Save dialogue turn if session_id provided
            if session_id:
                turn_number = self._get_next_turn_number(session_id)
                serializable_sources = safe_jsonable(result.get("sources", []))
                serializable_metadata = safe_jsonable(
                    {
                        "intent": getattr(processed_query, "intent", None),
                        "keywords": getattr(processed_query, "keywords", None),
                        "repo_filter": repo_filter,
                        "multi_turn": enable_multi_turn,
                    }
                )

                self.cache_manager.save_dialogue_turn(
                    session_id=session_id,
                    turn_number=turn_number,
                    query=question,
                    answer=full_answer,
                    summary=summary or "",
                    retrieved_elements=serializable_sources,
                    metadata=serializable_metadata,
                )

                self.logger.info(
                    f"Saved dialogue turn {turn_number} for session {session_id}"
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
        history = self.cache_manager.get_dialogue_history(
            session_id, max_turns=context_rounds
        )

        return history if history else None

    def _get_next_turn_number(self, session_id: str) -> int:
        """
        Get the next turn number for a session

        Args:
            session_id: Session ID

        Returns:
            Next turn number (1-indexed)
        """
        session_index = self.cache_manager._get_session_index(session_id)
        if session_index:
            return session_index.get("total_turns", 0) + 1
        return 1

    def _extract_sources_from_elements(
        self, elements: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract source information from retrieved elements"""
        return _snapshot.extract_sources_from_elements(elements)
