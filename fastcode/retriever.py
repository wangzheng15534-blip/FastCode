"""
Hybrid Retriever - Multi-stage retrieval with semantic, keyword, and graph-based search
Enhanced with LLM-processed query support
"""

import os
import pickle
import logging
import math
import re
from collections import Counter
from typing import List, Dict, Any, Set, Tuple, Optional, Union
import numpy as np
import networkx as nx
from rank_bm25 import BM25Okapi

from .vector_store import VectorStore
from .embedder import CodeEmbedder
from .graph_builder import CodeGraphBuilder
from .indexer import CodeElement
from .ir_graph_builder import IRGraphs
from .query_processor import ProcessedQuery
from .repo_selector import RepositorySelector
from .utils import ensure_dir
from .iterative_agent import IterativeAgent
from .pg_retrieval import PgRetrievalStore


class HybridRetriever:
    """Hybrid retrieval combining semantic search, keyword search, and graph traversal"""
    
    def __init__(self, config: Dict[str, Any], vector_store: VectorStore,
                 embedder: CodeEmbedder, graph_builder: CodeGraphBuilder,
                 repo_root: Optional[str] = None):
        self.config = config
        self.retrieval_config = config.get("retrieval", {})
        self.logger = logging.getLogger(__name__)
        
        self.vector_store = vector_store
        self.embedder = embedder
        self.graph_builder = graph_builder
        
        # Weights for hybrid search
        self.semantic_weight = self.retrieval_config.get("semantic_weight", 0.6)
        self.keyword_weight = self.retrieval_config.get("keyword_weight", 0.3)
        self.graph_weight = self.retrieval_config.get("graph_weight", 0.1)
        self.graph_backend = self.retrieval_config.get("graph_backend", "ir")
        self.allow_legacy_graph_fallback = self.retrieval_config.get("allow_legacy_graph_fallback", True)
        self.retrieval_backend = self.retrieval_config.get("backend", "pg_hybrid")
        self.adaptive_fusion_cfg = self.retrieval_config.get("adaptive_fusion", {}) or {}
        self.adaptive_fusion_enabled = bool(self.adaptive_fusion_cfg.get("enabled", False))
        
        # Retrieval parameters
        self.min_similarity = self.retrieval_config.get("min_similarity", 0.3)
        self.max_results = self.retrieval_config.get("max_results", 5)
        self.diversity_penalty = self.retrieval_config.get("diversity_penalty", 0.1)
        
        # Multi-repository parameters
        self.enable_two_stage_retrieval = self.retrieval_config.get("enable_two_stage_retrieval", True)
        self.select_repos_by_overview = self.retrieval_config.get("select_repos_by_overview", True)
        self.repo_selection_method = self.retrieval_config.get("repo_selection_method", "llm")  # "llm" or "embedding"
        self.top_repos_to_search = self.retrieval_config.get("top_repos_to_search", 5)
        self.min_repo_similarity = self.retrieval_config.get("min_repo_similarity", 0.3)
        self.max_files_to_search = self.retrieval_config.get("max_files_to_search", 5)
        
        # Agency mode parameters
        self.enable_agency_mode = self.retrieval_config.get("enable_agency_mode", True)
        
        # Full indexes (for repository selection - never cleared)
        self.full_bm25 = None
        self.full_bm25_corpus = []
        self.full_bm25_elements = []
        
        # Separate BM25 index for repository overviews
        self.repo_overview_bm25 = None
        self.repo_overview_bm25_corpus = []
        self.repo_overview_names = []  # List of repo names corresponding to corpus
        
        # Filtered indexes (for actual retrieval after repo selection)
        self.filtered_bm25 = None
        self.filtered_bm25_corpus = []
        self.filtered_bm25_elements = []
        
        # Filtered vector store for selected repositories
        self.filtered_vector_store = None
        
        # Repository selector for LLM-based file selection
        self.repo_selector = RepositorySelector(config)
        
        # Initialize agents for agency mode (will be initialized later when repo_root is known)
        self.iterative_agent = None
        self.repo_root = repo_root
        
        # Try to initialize agents if repo_root is provided
        if self.enable_agency_mode and repo_root:
            self.logger.info(f"Initializing agency mode agents with repo_root: {repo_root}")
            self._initialize_agents(repo_root)
        elif self.enable_agency_mode and not repo_root:
            self.logger.info("Agency mode enabled but repo_root not set yet. Agents will initialize when repository is loaded.")
        
        # Persistence
        self.persist_dir = config.get("vector_store", {}).get("persist_directory", "./data/vector_store")
        ensure_dir(self.persist_dir)
        
        # Track currently loaded repositories for filtering
        self.current_loaded_repos = None  # None means all repos loaded, List means specific repos
        self.ir_graphs: Optional[IRGraphs] = None
        self.ir_snapshot_id: Optional[str] = None
        self.pg_retrieval_store: Optional[PgRetrievalStore] = None
        self._active_snapshot_id: Optional[str] = None
        self._last_fusion_debug: Optional[Dict[str, Any]] = None
    
    def index_for_bm25(self, elements: List[CodeElement]):
        """
        Build full BM25 index for keyword search (excludes repository overviews)
        
        Args:
            elements: List of code elements (without repository_overview type)
        """
        self.logger.info("Building full BM25 index for keyword search")
        
        self.full_bm25_elements = elements
        self.full_bm25_corpus = []
        
        for elem in elements:
            # Skip repository_overview elements if any (they should be in separate storage)
            if elem.type == "repository_overview":
                continue
            
            # Combine different text fields for indexing
            text_parts = [
                elem.name,
                elem.type,
                elem.language,
                elem.relative_path,
            ]
            
            if elem.docstring:
                text_parts.append(elem.docstring)
            
            if elem.signature:
                text_parts.append(elem.signature)
            
            if elem.summary:
                text_parts.append(elem.summary)
            
            # Add some code content
            if elem.code:
                text_parts.append(elem.code[:1000])  # First 1000 chars
            
            text = " ".join(text_parts)
            # Tokenize (simple whitespace tokenization)
            tokens = text.lower().split()
            self.full_bm25_corpus.append(tokens)
        
        self.full_bm25 = BM25Okapi(self.full_bm25_corpus)
        self.logger.info(f"Built full BM25 index with {len(self.full_bm25_corpus)} documents")

    def set_ir_graphs(self, ir_graphs: Optional[IRGraphs], snapshot_id: Optional[str] = None) -> None:
        self.ir_graphs = ir_graphs
        self.ir_snapshot_id = snapshot_id
        if ir_graphs is not None:
            self.logger.info(f"IR graph backend active for snapshot {snapshot_id or 'unknown'}")
        else:
            self.logger.info("IR graph backend cleared")

    def set_pg_retrieval_store(self, store: Optional[PgRetrievalStore]) -> None:
        self.pg_retrieval_store = store
        if store and store.is_active():
            self.logger.info("PG hybrid retrieval backend enabled")
        else:
            self.logger.info("PG hybrid retrieval backend disabled")
    
    def build_repo_overview_bm25(self):
        """
        Build separate BM25 index for repository overviews
        Uses the separate repo overview storage from vector_store
        """
        self.logger.info("Building BM25 index for repository overviews")
        
        # Load repo overviews from separate storage
        repo_overviews = self.vector_store.load_repo_overviews()
        
        if not repo_overviews:
            self.logger.warning("No repository overviews found for BM25 indexing")
            return
        
        self.repo_overview_bm25_corpus = []
        self.repo_overview_names = []
        
        for repo_name, overview_data in repo_overviews.items():
            # Get text content for BM25
            content = overview_data.get("content", "")
            metadata = overview_data.get("metadata", {})
            
            # Combine all text
            readme = metadata.get("readme_content")
            text_parts = [
                repo_name,
                content,
                metadata.get("summary", ""),
                metadata.get("structure_text", ""),
                (readme if readme else "")[:1000]  # 确保是字符串
            ]
            
            text = " ".join(text_parts)
            tokens = text.lower().split()
            
            self.repo_overview_bm25_corpus.append(tokens)
            self.repo_overview_names.append(repo_name)
        
        self.repo_overview_bm25 = BM25Okapi(self.repo_overview_bm25_corpus)
        self.logger.info(f"Built repo overview BM25 index with {len(self.repo_overview_bm25_corpus)} repositories")
    
    def retrieve(self, query: Union[str, ProcessedQuery], filters: Optional[Dict[str, Any]] = None,
                 repo_filter: Optional[List[str]] = None, enable_file_selection: bool = True,
                 use_agency_mode: Optional[bool] = None,
                 dialogue_history: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant code elements using hybrid approach with LLM-enhanced queries
        Supports two-stage retrieval for multi-repository scenarios
        Enhanced with agency mode for accurate and comprehensive retrieval

        Args:
            query: User query (string) or ProcessedQuery object with enhancements
            filters: Optional filters (file_type, language, etc.)
            repo_filter: Optional list of repository names to search in
            enable_file_selection: Whether to use LLM for file selection (multi-repo only)
            use_agency_mode: Whether to use agency mode (None = auto-decide based on intent)
            dialogue_history: Previous dialogue summaries for multi-turn context

        Returns:
            List of retrieved elements with metadata
        """
        # Handle both string and ProcessedQuery inputs
        self._active_snapshot_id = (filters or {}).get("snapshot_id") or self.ir_snapshot_id
        if isinstance(query, ProcessedQuery):
            processed_query = query
            query_str = processed_query.original
            query_info = {
                "intent": processed_query.intent if hasattr(processed_query, 'intent') else "unknown",
                "keywords": processed_query.keywords,
                "filters": processed_query.filters,
                "expanded": processed_query.expanded,
                "rewritten_query": processed_query.rewritten_query,
                "pseudocode_hints": processed_query.pseudocode_hints
            }
            
            # Use enhanced query information if available
            if processed_query.rewritten_query:
                search_text4repo_selection = processed_query.rewritten_query
                search_text4semantic = processed_query.rewritten_query
            else:
                search_text4repo_selection = processed_query.original
                search_text4semantic = processed_query.original
                
            keywords = processed_query.keywords
            pseudocode = processed_query.pseudocode_hints
            
            # Merge filters
            if filters is None:
                filters = processed_query.filters
            else:
                filters = {**processed_query.filters, **filters}
            
            self.logger.info(f"Retrieving with ProcessedQuery: {query_str[:100]}...")
            if pseudocode:
                self.logger.debug("Using pseudocode hints for additional matching")
        else:
            query_str = query
            query_info = {"intent": "unknown", "keywords": [], "filters": {}}
            # search_text = query
            search_text4repo_selection = query
            search_text4semantic = query
            keywords = None
            pseudocode = None
            self.logger.info(f"Retrieving for query: {query_str}")
        
        # Determine if agency mode should be used
        should_use_agency = self.enable_agency_mode

        # ========================================
        # STEP 1: Repository Selection (Independent of agency mode)
        # Select relevant repositories based on overview if configured
        # ========================================

        if self.select_repos_by_overview:
            # Check if we have multiple repositories
            available_repos = self.vector_store.get_repository_names()

            # Determine the effective repo list: user's selection (repo_filter) or all available repos
            # If user explicitly selected specific repos via repo_filter, use that count
            effective_repos = repo_filter if repo_filter else available_repos

            if len(effective_repos) > 1:
                self.logger.info(f"Multi-repository scenario detected ({len(effective_repos)} repos)")

                if self.repo_selection_method == "llm":
                    # LLM-based repo selection with robust fuzzy matching
                    selected_repos = self._select_relevant_repositories_by_llm(
                        search_text4repo_selection, self.top_repos_to_search,
                        scope_repos=repo_filter
                    )
                else:
                    # Legacy embedding+BM25-based repo selection
                    selected_repos = self._select_relevant_repositories(
                        search_text4repo_selection, keywords, self.top_repos_to_search
                    )

                if selected_repos:
                    repo_filter = selected_repos
                    self.logger.info(f"Selected repositories by overview: {repo_filter}")
                else:
                    self.logger.warning("No repositories selected, searching all")
            else:
                # Single repo mode - no need to call LLM for repo selection
                self.logger.info(f"Single repository mode ({effective_repos}), skipping LLM repo selection")

        # ========================================
        # STEP 2: Choose Retrieval Mode (Agency or Standard)
        # ========================================

        # If using iterative agency mode, use agency-based retrieval
        if should_use_agency and self.iterative_agent:
            self.logger.info("Using iterative agency mode for retrieval")

            # Ensure repos are loaded if needed
            if repo_filter and self.current_loaded_repos != repo_filter:
                self.logger.info("Reloading specific repository indexes for iterative mode")
                if self.reload_specific_repositories(repo_filter):
                    self.current_loaded_repos = repo_filter

            # Go directly to iterative agency mode with dialogue_history
            final_results = self._apply_agency_mode(query_str, [], query_info, repo_filter, dialogue_history)

            # Final safety check
            if repo_filter:
                final_results = self._final_repo_filter(final_results, repo_filter)

            return final_results

        # ========================================
        # STEP 3: Standard Retrieval Mode
        # ========================================

        # Reload specific repository indexes for accurate retrieval
        if repo_filter:
            self.logger.info(f"Filtering by repositories: {repo_filter}")
            
            # Always reload when the requested repositories change to avoid stale indexes
            if self.current_loaded_repos != repo_filter:
                self.logger.info("Reloading specific repository indexes for accurate retrieval")
                if self.reload_specific_repositories(repo_filter):
                    self.current_loaded_repos = repo_filter
                else:
                    self.logger.warning("Failed to reload specific repositories, using filtered search")
                    self.current_loaded_repos = None
        
        # CRITICAL: Always pass repo_filter for safety, even when using filtered indexes
        # The search methods will apply appropriate filtering based on which index they use
        
        # Stage 1: retrieval channels
        keyword_query = " ".join(keywords) if keywords else query_str
        doc_integration_enabled = bool(self.config.get("docs_integration", {}).get("enabled", False))

        code_types = ["file", "class", "function", "documentation"]
        doc_types = ["design_document"]

        semantic_results = self._semantic_search(
            search_text4semantic,
            top_k=20,
            repo_filter=repo_filter,
            element_types=code_types if doc_integration_enabled else None,
        )
        pseudocode_results = []
        if pseudocode:
            pseudocode_results = self._semantic_search(
                pseudocode,
                top_k=10,
                repo_filter=repo_filter,
                element_types=code_types if doc_integration_enabled else None,
            )
            self.logger.info(f"Pseudocode search found {len(pseudocode_results)} additional results")
        keyword_results = self._keyword_search(
            keyword_query,
            top_k=10,
            repo_filter=repo_filter,
            element_types=code_types if doc_integration_enabled else None,
        )

        code_combined = self._combine_results(semantic_results, keyword_results, pseudocode_results)
        if self.graph_weight > 0:
            code_combined = self._expand_with_graph(code_combined, max_hops=2)
        code_final = self._rerank(query_str, code_combined)

        doc_final: List[Dict[str, Any]] = []
        if doc_integration_enabled:
            doc_semantic = self._semantic_search(
                search_text4semantic,
                top_k=16,
                repo_filter=repo_filter,
                element_types=doc_types,
            )
            doc_keyword = self._keyword_search(
                keyword_query,
                top_k=16,
                repo_filter=repo_filter,
                element_types=doc_types,
            )
            doc_final = self._rerank(query_str, self._combine_results(doc_semantic, doc_keyword, []))

        if doc_integration_enabled and self.adaptive_fusion_enabled and doc_final:
            final_results = self._adaptive_fuse_channels(
                query=query_str,
                query_info=query_info,
                code_results=code_final,
                doc_results=doc_final,
            )
        else:
            final_results = code_final

        # Stage 6: Apply filters
        if filters:
            final_results = self._apply_filters(final_results, filters)

        # Stage 7: Diversification
        final_results = self._diversify(final_results)

        # Limit results
        final_results = final_results[:self.max_results]
        
        self.logger.info(f"Retrieved {len(final_results)} elements")
        
        # Optional: LLM-based file selection (single or multi-repo) - only if not using agency mode
        if enable_file_selection and not should_use_agency:
            self.logger.info("Using file selection for accurate and comprehensive retrieval")
            self.logger.info(f"enable_file_selection: {enable_file_selection}")
            self.logger.info(f"should_use_agency: {should_use_agency}")
            target_repos = repo_filter or self.vector_store.get_repository_names()
            if target_repos:
                final_results = self._enhance_with_file_selection(
                    query_str, final_results, target_repos
                )
        
        # Agency mode: accurate search + association lookup
        if should_use_agency:
            self.logger.info("Using agency mode for accurate and comprehensive retrieval")
            final_results = self._apply_agency_mode(query_str, final_results, query_info, repo_filter)
        
        # Final safety check: ensure only results from selected repos are returned
        # This is the last line of defense to catch any leaks from earlier stages
        if repo_filter:
            self.logger.debug(f"Applying final repo filter: {repo_filter}")
            final_results = self._final_repo_filter(final_results, repo_filter)
        
        return final_results
    
    def _select_relevant_repositories(self, query: Union[str, List[str]], keywords: Optional[List[str]], top_k: int = 5) -> List[str]:
        """
        Select top N most relevant repositories based on overview matching
        Combines both semantic vector search and BM25 keyword search
        Uses separate repository overview storage
        
        Args:
            query: User query
            top_k: Number of top repositories to select
        
        Returns:
            List of selected repository names
        """
        self.logger.info("Performing repository selection based on overviews (semantic + BM25)")
        
        # Prepare text for semantic search
        if isinstance(query, list):
            semantic_query_text = " ".join(query)
        elif keywords:
            semantic_query_text = " ".join(keywords)
        else:
            semantic_query_text = query

        # Stage 1: Semantic search on repository overviews (use separate storage)
        query_embedding = self.embedder.embed_text(semantic_query_text)
        semantic_results = self.vector_store.search_repository_overviews(
            query_embedding,
            k=top_k * 2,  # Get more candidates for combining
            min_score=self.min_repo_similarity
        )
        
        # Stage 2: BM25 search on repository overviews (use separate BM25 index)
        bm25_results = []
        if self.repo_overview_bm25 is not None and self.repo_overview_names:
            # Tokenize using provided keywords when available; fall back to query text
            query_tokens: List[str] = []
            if keywords:
                for kw in keywords:
                    query_tokens.extend(kw.lower().split())
            elif isinstance(query, list):
                for part in query:
                    query_tokens.extend(part.lower().split())
            else:
                query_tokens = query.lower().split()
            scores = self.repo_overview_bm25.get_scores(query_tokens)
            # Get repository overview results from separate index
            for idx, score in enumerate(scores):
                if idx < len(self.repo_overview_names) and score > 0:
                    repo_name = self.repo_overview_names[idx]
                    bm25_results.append((repo_name, float(score)))
            
            # Sort by BM25 score and take top candidates
            bm25_results.sort(key=lambda x: x[1], reverse=True)
            bm25_results = bm25_results[:top_k*2]
        
        # Stage 3: Combine scores from both methods
        repo_scores = {}
        
        # Add semantic scores
        for metadata, score in semantic_results:
            repo_name = metadata.get("repo_name")
            if repo_name:
                repo_scores[repo_name] = {
                    "semantic_score": score,
                    "bm25_score": 0.0,
                    "total_score": score * 0.7  # 70% weight for semantic
                }
        
        # Add BM25 scores (normalize them first)
        if bm25_results:
            max_bm25_score = max(score for _, score in bm25_results)
            if max_bm25_score > 0:
                for repo_name, score in bm25_results:
                    normalized_bm25 = score / max_bm25_score
                    if repo_name in repo_scores:
                        repo_scores[repo_name]["bm25_score"] = normalized_bm25
                        repo_scores[repo_name]["total_score"] += normalized_bm25 * 0.3  # 30% weight for BM25
                    else:
                        repo_scores[repo_name] = {
                            "semantic_score": 0.0,
                            "bm25_score": normalized_bm25,
                            "total_score": normalized_bm25 * 0.3
                        }
        
        
        MIN_SCORE_THRESHOLD = 0.15

        for repo_name, scores in repo_scores.items():
            self.logger.info(
                f"repo: {repo_name} "
                f"(semantic: {scores['semantic_score']:.3f}, "
                f"bm25: {scores['bm25_score']:.3f}, "
                f"total: {scores['total_score']:.3f})"
            )

        # 1. filter and sort
        qualified_repos = [
            (name, scores) 
            for name, scores in repo_scores.items() 
            if scores["total_score"] > MIN_SCORE_THRESHOLD or scores["semantic_score"] > 0.4 or scores["bm25_score"] > 0.95
        ]
        
        sorted_repos = sorted(
            qualified_repos, 
            key=lambda x: x[1]["total_score"], 
            reverse=True
        )

        selected_repos = []
        
        # 2. get top k
        for repo_name, scores in sorted_repos[:top_k]:
            selected_repos.append(repo_name)
            print(
                f"Selected repo: {repo_name} "
                f"(semantic: {scores['semantic_score']:.3f}, "
                f"bm25: {scores['bm25_score']:.3f}, "
                f"total: {scores['total_score']:.3f})"
            )
            
        # 3. no repo selected
        if not selected_repos:
            print(f"No repositories met the minimum score threshold of {MIN_SCORE_THRESHOLD}")
        
        return selected_repos

    def _select_relevant_repositories_by_llm(
        self, query: str, top_k: int = 5,
        scope_repos: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select relevant repositories using LLM.

        The LLM receives repository overviews and returns the names of the
        most relevant ones.  Robust fuzzy matching is applied so that minor
        differences in the names returned by the LLM (casing, extra punctuation,
        partial names, etc.) do not cause mismatches.

        Args:
            query: User query (or rewritten query)
            top_k: Maximum number of repositories to select
            scope_repos: If provided, only consider these repositories
                         (e.g. from request.repo_names). When None, all
                         indexed repositories are considered.

        Returns:
            List of matched repository names (empty on failure, falls back to
            embedding-based selection)
        """
        self.logger.info("Performing LLM-based repository selection")

        # Load overviews from storage
        all_overviews = self.vector_store.load_repo_overviews()
        if not all_overviews:
            self.logger.warning("No repository overviews available for LLM selection")
            return []

        # Narrow down to only the repos the caller cares about
        if scope_repos:
            repo_overviews = {k: v for k, v in all_overviews.items() if k in scope_repos}
            self.logger.info(
                f"Scoped LLM repo selection to {len(repo_overviews)}/{len(all_overviews)} repos "
                f"(scope_repos={scope_repos})"
            )
        else:
            repo_overviews = all_overviews
            self.logger.info(f"LLM repo selection considering all {len(repo_overviews)} repos")

        if not repo_overviews:
            self.logger.warning("No overviews remain after scoping, searching all scoped repos")
            return scope_repos or []

        try:
            selected = self.repo_selector.select_relevant_repos(
                query, repo_overviews, max_repos=top_k
            )
            if selected:
                self.logger.info(f"LLM selected repos: {selected}")
                return selected

            self.logger.warning("LLM returned no repos, falling back to embedding-based selection")
        except Exception as e:
            self.logger.error(f"LLM repo selection failed: {e}, falling back to embedding-based selection")

        # Fallback 1: use the original embedding+BM25 method
        try:
            embedding_selected = self._select_relevant_repositories(query, None, top_k)
            if embedding_selected:
                self.logger.info(f"Embedding-based selection returned: {embedding_selected}")
                return embedding_selected
            self.logger.warning("Embedding-based selection also returned empty")
        except Exception as e:
            self.logger.error(f"Embedding-based repo selection also failed: {e}")

        # Fallback 2: return user's original scope_repos selection
        if scope_repos:
            self.logger.warning(f"All repo selection methods failed, falling back to user's original selection: {scope_repos}")
            return scope_repos

        # Final fallback: return empty list (will search all repos)
        self.logger.warning("All repo selection methods failed and no scope_repos provided, will search all repos")
        return []

    def _enhance_with_file_selection(self, query: str, results: List[Dict[str, Any]],
                                     repo_names: List[str]) -> List[Dict[str, Any]]:
        """
        Use LLM to select specific files and enhance results
        
        Args:
            query: User query
            results: Current retrieval results
            repo_names: List of repository names being searched
        
        Returns:
            Enhanced results with LLM-selected files
        """
        # Get repository overviews for selected repos
        repo_overviews = self._get_repository_overviews(repo_names)
        
        if not repo_overviews:
            self.logger.warning("No repository overviews found in retrieval phase")
            return results
        
        # Use LLM to select relevant files
        scenario_mode = "single" if len(repo_names) == 1 else "multi"
        selected_files = self.repo_selector.select_relevant_files(
            query,
            repo_overviews,
            max_files=self.max_files_to_search,
            scenario_mode=scenario_mode,
        )
        
        if not selected_files:
            self.logger.warning("No files selected by LLM")
            return results
        
        self.logger.info(f"LLM selected {len(selected_files)} specific files")
        
        # Retrieve elements from selected files
        file_elements = self._retrieve_elements_from_files(selected_files)
        
        # Merge with existing results (prioritize LLM-selected files)
        # Add selected file elements with boosted scores
        enhanced_results = []
        seen_ids = set()
        
        # First add LLM-selected file elements with boosted score
        for elem_data in file_elements:
            elem_id = elem_data["element"].get("id")
            if elem_id and elem_id not in seen_ids:
                # Boost LLM-selected elements - apply to all score components
                boost_factor = 1.3
                elem_data["total_score"] *= boost_factor
                elem_data["semantic_score"] *= boost_factor
                elem_data["keyword_score"] *= boost_factor
                elem_data["pseudocode_score"] *= boost_factor
                elem_data["graph_score"] *= boost_factor
                elem_data["llm_selected"] = True
                enhanced_results.append(elem_data)
                seen_ids.add(elem_id)
        
        # Then add original results
        for elem_data in results:
            elem_id = elem_data["element"].get("id")
            if elem_id and elem_id not in seen_ids:
                enhanced_results.append(elem_data)
                seen_ids.add(elem_id)
        
        # Re-sort by score
        enhanced_results.sort(key=lambda x: x["total_score"], reverse=True)
        
        return enhanced_results
    
    def _get_repository_overviews(self, repo_names: List[str]) -> List[Dict[str, Any]]:
        """Get repository overview information for given repo names from separate storage"""
        overviews = []
        
        # Load from separate storage
        all_overviews = self.vector_store.load_repo_overviews()
        
        for repo_name in repo_names:
            if repo_name in all_overviews:
                overview_data = all_overviews[repo_name]
                metadata = overview_data.get("metadata", {})
                
                overview = {
                    "repo_name": repo_name,
                    "summary": metadata.get("summary", ""),
                    "structure_text": metadata.get("structure_text", ""),
                    "file_structure": metadata.get("file_structure", {}),
                }
                overviews.append(overview)
        
        return overviews
    
    def _retrieve_elements_from_files(self, selected_files: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Retrieve code elements from specific files selected by LLM
        
        Args:
            selected_files: List of dicts with repo_name and file_path
        
        Returns:
            List of file-level code elements from those files
        """
        results = []
        
        # Use filtered elements if available, otherwise use full elements
        elements_to_search = (self.filtered_bm25_elements 
                             if self.filtered_bm25_elements 
                             else self.full_bm25_elements)
        
        for file_info in selected_files:
            repo_name = file_info["repo_name"]
            file_path = file_info["file_path"]
            
            # Find file-level elements only from this file
            # Note: No need to check for repository_overview as they're in separate storage
            for elem in elements_to_search:
                if (elem.repo_name == repo_name and 
                    file_path in elem.relative_path and
                    elem.type == "file"):  # Only select file-level elements
                    
                    results.append({
                        "element": elem.to_dict(),
                        "semantic_score": 0.8,  # Give good base score for LLM selection
                        "keyword_score": 0.0,
                        "pseudocode_score": 0.0,
                        "graph_score": 0.0,
                        "total_score": 0.8,
                        "llm_file_selected": True,
                        "file_selection_reason": file_info.get("reason", ""),
                    })
        
        return results

    def _adaptive_fuse_channels(
        self,
        *,
        query: str,
        query_info: Dict[str, Any],
        code_results: List[Dict[str, Any]],
        doc_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        alpha, k_code, k_doc = self._compute_adaptive_fusion_params(
            query=query,
            query_info=query_info,
            code_results=code_results,
            doc_results=doc_results,
        )
        fused: Dict[str, Dict[str, Any]] = {}
        for rank, row in enumerate(code_results, start=1):
            elem = row.get("element", {})
            elem_id = elem.get("id")
            if not elem_id:
                continue
            rrf = 1.0 / (float(k_code) + float(rank))
            entry = self._ensure_fused_entry(fused, elem_id, elem)
            entry["semantic_score"] = max(entry.get("semantic_score", 0.0), row.get("semantic_score", 0.0))
            entry["keyword_score"] = max(entry.get("keyword_score", 0.0), row.get("keyword_score", 0.0))
            entry["pseudocode_score"] = max(entry.get("pseudocode_score", 0.0), row.get("pseudocode_score", 0.0))
            entry["graph_score"] = max(entry.get("graph_score", 0.0), row.get("graph_score", 0.0))
            entry["total_score"] += alpha * rrf
            entry["fusion"]["code_rrf"] = rrf
            entry["fusion"]["alpha"] = alpha
            entry["fusion"]["k_code"] = k_code
            entry["fusion"]["k_doc"] = k_doc

        for rank, row in enumerate(doc_results, start=1):
            elem = row.get("element", {})
            elem_id = elem.get("id")
            if not elem_id:
                continue
            rrf = 1.0 / (float(k_doc) + float(rank))
            entry = self._ensure_fused_entry(fused, elem_id, elem)
            entry["semantic_score"] = max(entry.get("semantic_score", 0.0), row.get("semantic_score", 0.0))
            entry["keyword_score"] = max(entry.get("keyword_score", 0.0), row.get("keyword_score", 0.0))
            entry["total_score"] += (1.0 - alpha) * rrf
            entry["fusion"]["doc_rrf"] = rrf
            entry["fusion"]["alpha"] = alpha
            entry["fusion"]["k_code"] = k_code
            entry["fusion"]["k_doc"] = k_doc

        out = list(fused.values())
        out.sort(key=lambda x: x.get("total_score", 0.0), reverse=True)
        debug = dict(self._last_fusion_debug or {})
        debug.update(
            {
                "alpha": alpha,
                "k_code": k_code,
                "k_doc": k_doc,
                "code_candidates": len(code_results),
                "doc_candidates": len(doc_results),
            }
        )
        self._last_fusion_debug = debug
        return out

    @staticmethod
    def _new_fused_entry(element: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "element": element,
            "semantic_score": 0.0,
            "keyword_score": 0.0,
            "pseudocode_score": 0.0,
            "graph_score": 0.0,
            "total_score": 0.0,
            "fusion": {},
        }

    def _ensure_fused_entry(
        self,
        fused: Dict[str, Dict[str, Any]],
        elem_id: str,
        element: Dict[str, Any],
    ) -> Dict[str, Any]:
        entry = fused.get(elem_id)
        if entry is None:
            entry = self._new_fused_entry(element)
            fused[elem_id] = entry
        return entry

    @staticmethod
    def _sigmoid(x: float) -> float:
        # Clamp input for numerical stability under extreme values.
        x = max(-30.0, min(30.0, float(x)))
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _tokenize_signal(text: str) -> List[str]:
        return re.findall(r"[a-z0-9_]+", (text or "").lower())

    @staticmethod
    def _normalized_query_entropy(tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        total = float(len(tokens))
        if total <= 1:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(max(p, 1e-12))
        max_entropy = math.log2(max(2, len(counts)))
        if max_entropy <= 0:
            return 0.0
        return max(0.0, min(1.0, entropy / max_entropy))

    @staticmethod
    def _weighted_keyword_affinity(
        tokens: List[str],
        weights: Dict[str, float],
    ) -> float:
        if not tokens or not weights:
            return 0.0
        token_set = set(tokens)
        total = float(sum(max(0.0, w) for w in weights.values()))
        if total <= 0:
            return 0.0
        matched = 0.0
        for term, weight in weights.items():
            if term in token_set:
                matched += max(0.0, float(weight))
        return max(0.0, min(1.0, matched / total))

    def _compute_adaptive_fusion_params(
        self,
        *,
        query: str,
        query_info: Dict[str, Any],
        code_results: List[Dict[str, Any]],
        doc_results: List[Dict[str, Any]],
    ) -> Tuple[float, float, float]:
        cfg = self.adaptive_fusion_cfg or {}
        alpha_base = float(cfg.get("alpha_base", 0.80))
        alpha_min = float(cfg.get("alpha_min", 0.25))
        alpha_max = float(cfg.get("alpha_max", 0.90))
        k_base = float(cfg.get("rrf_k_base", 60))
        k_min = float(cfg.get("rrf_k_min", 20))
        k_max = float(cfg.get("rrf_k_max", 100))

        q = query or ""
        query_info = query_info or {}
        intent = str(query_info.get("intent") or "")
        keywords = query_info.get("keywords")
        if isinstance(keywords, list):
            keyword_text = " ".join(str(k) for k in keywords)
        else:
            keyword_text = ""

        signal_text = " ".join([q, intent, keyword_text])
        tokens = self._tokenize_signal(signal_text)
        query_entropy = self._normalized_query_entropy(tokens)

        doc_term_weights = {
            "design": 1.0,
            "architecture": 1.0,
            "adr": 1.2,
            "rfc": 1.2,
            "decision": 1.0,
            "tradeoff": 1.1,
            "rationale": 1.1,
            "approach": 0.8,
            "spec": 0.9,
            "why": 0.7,
            "intent": 0.7,
        }
        code_term_weights = {
            "function": 1.0,
            "class": 1.0,
            "method": 1.0,
            "line": 0.7,
            "call": 0.9,
            "bug": 1.0,
            "fix": 1.0,
            "trace": 0.9,
            "implementation": 1.0,
            "stack": 0.7,
            "runtime": 0.8,
        }
        doc_affinity = self._weighted_keyword_affinity(tokens, doc_term_weights)
        code_affinity = self._weighted_keyword_affinity(tokens, code_term_weights)

        code_top = float(code_results[0].get("total_score", 0.0)) if code_results else 0.0
        doc_top = float(doc_results[0].get("total_score", 0.0)) if doc_results else 0.0

        alpha = alpha_base
        # Continuous domain affinity (replaces binary doc_hit/code_hit).
        alpha -= 0.30 * doc_affinity
        alpha += 0.18 * code_affinity

        # Continuous confidence skew from retrieval channel strengths.
        strength_delta = math.tanh((code_top - doc_top) * 2.2)
        alpha += 0.12 * strength_delta

        # High-entropy queries are more ambiguous, so pull alpha toward balanced blending.
        entropy_pull = 0.22 * query_entropy
        alpha = (1.0 - entropy_pull) * alpha + entropy_pull * 0.5
        alpha = min(alpha_max, max(alpha_min, alpha))

        code_conf = min(1.0, max(0.0, code_top))
        doc_conf = min(1.0, max(0.0, doc_top))

        # Smooth sigmoid k to avoid cliff effects from piecewise/linear shifts.
        code_k_z = ((0.55 - code_conf) * 4.2) + ((query_entropy - 0.5) * 1.4) + ((doc_affinity - code_affinity) * 1.0)
        doc_k_z = ((0.55 - doc_conf) * 4.2) + ((query_entropy - 0.5) * 1.4) + ((code_affinity - doc_affinity) * 1.0)
        k_code_sig = self._sigmoid(code_k_z)
        k_doc_sig = self._sigmoid(doc_k_z)
        k_code = k_min + (k_max - k_min) * k_code_sig
        k_doc = k_min + (k_max - k_min) * k_doc_sig

        # Gentle pull toward configured base to keep behavior stable across repos.
        k_code = 0.8 * k_code + 0.2 * k_base
        k_doc = 0.8 * k_doc + 0.2 * k_base

        k_code = min(k_max, max(k_min, k_code))
        k_doc = min(k_max, max(k_min, k_doc))
        self._last_fusion_debug = {
            "alpha": alpha,
            "k_code": k_code,
            "k_doc": k_doc,
            "query_entropy": query_entropy,
            "doc_affinity": doc_affinity,
            "code_affinity": code_affinity,
            "code_top": code_top,
            "doc_top": doc_top,
        }
        return alpha, k_code, k_doc
    
    def _semantic_search(
        self,
        query: str,
        top_k: int = 20,
        repo_filter: Optional[List[str]] = None,
        element_types: Optional[List[str]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Semantic search using embeddings
        Uses filtered_vector_store if available, otherwise uses full vector_store
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query)

        if (
            self.retrieval_backend == "pg_hybrid"
            and self.pg_retrieval_store is not None
            and self.pg_retrieval_store.is_active()
            and self._active_snapshot_id
        ):
            pg_results = self.pg_retrieval_store.semantic_search(
                snapshot_id=self._active_snapshot_id,
                query_embedding=query_embedding,
                repo_filter=repo_filter,
                element_types=element_types,
                top_k=top_k,
            )
            if pg_results:
                return pg_results
        
        # Choose which vector store to use
        if self.filtered_vector_store is not None and self.filtered_vector_store.get_count() > 0:
            # Use filtered vector store
            # IMPORTANT: Still pass repo_filter for double-checking, in case filtered store has stale data
            results = self.filtered_vector_store.search(
                query_embedding,
                k=top_k * 2,  # Get more candidates for filtering
                min_score=self.min_similarity,
                repo_filter=repo_filter  # Apply filter for safety
            )
            self.logger.debug(f"Semantic search (filtered) found {len(results)} results")
        else:
            # Use full vector store with repo_filter
            results = self.vector_store.search(
                query_embedding,
                k=top_k,
                min_score=self.min_similarity,
                repo_filter=repo_filter
            )
            self.logger.debug(f"Semantic search (full) found {len(results)} results")
        
        # Additional safety check: manually filter results by repo
        if repo_filter:
            filtered_results = []
            for metadata, score in results:
                repo_name = metadata.get("repo_name", "")
                if repo_name in repo_filter:
                    filtered_results.append((metadata, score))
                else:
                    self.logger.warning(
                        f"Semantic search returned element from unexpected repo: {repo_name} "
                        f"(expected: {repo_filter}). Element: {metadata.get('name', 'unknown')}"
                    )
            results = filtered_results[:top_k]  # Limit to top_k after filtering

        if element_types:
            allowed = set(element_types)
            results = [(m, s) for m, s in results if (m.get("type") in allowed)]

        return results
    
    def _keyword_search(
        self,
        query: str,
        top_k: int = 10,
        repo_filter: Optional[List[str]] = None,
        element_types: Optional[List[str]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Keyword search using BM25
        Uses filtered_bm25 if available, otherwise uses full_bm25
        """
        if (
            self.retrieval_backend == "pg_hybrid"
            and self.pg_retrieval_store is not None
            and self.pg_retrieval_store.is_active()
            and self._active_snapshot_id
        ):
            pg_results = self.pg_retrieval_store.keyword_search(
                snapshot_id=self._active_snapshot_id,
                query=query,
                repo_filter=repo_filter,
                element_types=element_types,
                top_k=top_k,
            )
            if pg_results:
                return pg_results

        # Choose which BM25 index to use
        if self.filtered_bm25 is not None and len(self.filtered_bm25_elements) > 0:
            # Use filtered BM25
            bm25_index = self.filtered_bm25
            bm25_elements = self.filtered_bm25_elements
            # CRITICAL FIX: Always apply repo_filter check for safety, even with filtered index
            use_filter = bool(repo_filter)
            self.logger.debug("Using filtered BM25 index")
        elif self.full_bm25 is not None:
            # Use full BM25
            bm25_index = self.full_bm25
            bm25_elements = self.full_bm25_elements
            use_filter = bool(repo_filter)
            self.logger.debug("Using full BM25 index")
        else:
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = bm25_index.get_scores(query_tokens)
        
        # Get top-k results with more candidates for filtering
        search_limit = top_k * 3 if use_filter else top_k
        top_indices = np.argsort(scores)[::-1][:min(search_limit, len(scores))]
        
        results = []
        filtered_count = 0
        allowed_types = set(element_types) if element_types else None
        
        for idx in top_indices:
            score = scores[idx]
            if score > 0:  # Only include non-zero scores
                elem = bm25_elements[idx]
                if allowed_types and elem.type not in allowed_types:
                    continue
                
                # CRITICAL: Always apply repository filter when repo_filter is provided
                if use_filter and elem.repo_name not in repo_filter:
                    filtered_count += 1
                    if filtered_count <= 3:  # Log first few for debugging
                        self.logger.warning(
                            f"BM25 search filtered out element from unexpected repo: {elem.repo_name} "
                            f"(expected: {repo_filter}). Element: {elem.name}"
                        )
                    continue
                
                metadata = elem.to_dict()
                results.append((metadata, float(score)))
                
                # Stop if we have enough results
                if len(results) >= top_k:
                    break
        
        if filtered_count > 0:
            self.logger.info(f"BM25 search filtered out {filtered_count} elements from unexpected repos")
        
        self.logger.debug(f"Keyword search found {len(results)} results")
        return results
    
    def _combine_results(self, semantic_results: List[Tuple[Dict[str, Any], float]],
                         keyword_results: List[Tuple[Dict[str, Any], float]],
                         pseudocode_results: Optional[List[Tuple[Dict[str, Any], float]]] = None) -> List[Dict[str, Any]]:
        """Combine semantic, keyword, and pseudocode search results"""
        # Create a dictionary to merge results by element ID
        combined = {}
        
        # Pseudocode weight (slightly lower than semantic)
        pseudocode_weight = 0.4 if pseudocode_results else 0.0
        
        # Add semantic results
        for metadata, score in semantic_results:
            elem_id = metadata.get("id")
            if elem_id:
                combined[elem_id] = {
                    "element": metadata,
                    "semantic_score": score * self.semantic_weight,
                    "keyword_score": 0.0,
                    "pseudocode_score": 0.0,
                    "graph_score": 0.0,
                    "total_score": score * self.semantic_weight,
                }
        
        # Add pseudocode results (for implementation queries)
        if pseudocode_results:
            for metadata, score in pseudocode_results:
                elem_id = metadata.get("id")
                if elem_id:
                    pseudocode_contrib = score * pseudocode_weight
                    
                    if elem_id in combined:
                        combined[elem_id]["pseudocode_score"] = pseudocode_contrib
                        combined[elem_id]["total_score"] += pseudocode_contrib
                    else:
                        combined[elem_id] = {
                            "element": metadata,
                            "semantic_score": 0.0,
                            "keyword_score": 0.0,
                            "pseudocode_score": pseudocode_contrib,
                            "graph_score": 0.0,
                            "total_score": pseudocode_contrib,
                        }
        
        # Add keyword results
        # Normalize BM25 scores to 0-1 range
        if keyword_results:
            max_bm25 = max(score for _, score in keyword_results)
            if max_bm25 > 0:
                for metadata, score in keyword_results:
                    elem_id = metadata.get("id")
                    if elem_id:
                        normalized_score = (score / max_bm25) * self.keyword_weight
                        
                        if elem_id in combined:
                            combined[elem_id]["keyword_score"] = normalized_score
                            combined[elem_id]["total_score"] += normalized_score
                        else:
                            combined[elem_id] = {
                                "element": metadata,
                                "semantic_score": 0.0,
                                "keyword_score": normalized_score,
                                "pseudocode_score": 0.0,
                                "graph_score": 0.0,
                                "total_score": normalized_score,
                            }
        
        # Convert to list and sort by total score
        results = list(combined.values())

        # Source-aware boost: prefer precise/SCIP-derived facts.
        for result in results:
            elem = result.get("element", {})
            meta = elem.get("metadata", {}) if isinstance(elem, dict) else {}
            source_priority = meta.get("source_priority", 0)
            try:
                source_priority = float(source_priority)
            except Exception:
                source_priority = 0.0
            boost = 1.0 + min(max(source_priority, 0.0), 100.0) / 200.0
            result["total_score"] *= boost

        results.sort(key=lambda x: x["total_score"], reverse=True)
        
        return results
    
    def _expand_with_graph(self, results: List[Dict[str, Any]], max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Expand results using code graph relationships

        IMPORTANT: This function preserves ALL original elements, even if they are not in the graph.
        It only adds graph-expanded elements for those that exist in the graph.
        """
        if not results:
            return results

        # Step 1: Keep all original results (even those not in graph)
        expanded = {}

        # Add all original results first, using a generated key for those without elem_id
        for idx, result in enumerate(results):
            elem_id = result["element"].get("id")
            if elem_id:
                expanded[elem_id] = result
            else:
                # For elements without ID (not in graph), use a unique key to preserve them
                unique_key = f"_no_graph_id_{idx}"
                expanded[unique_key] = result

        # Step 2: Expand only the top 10 elements that exist in the active graph backend
        for result in results[:10]:
            elem_id = result["element"].get("id")
            elem_name = result["element"].get("name")

            if not elem_id or elem_id not in expanded:
                continue
            related_ids = self._get_related_ids(elem_id, result["element"], max_hops=max_hops)

            # Add related elements with reduced score
            for related_id in related_ids:
                if related_id in expanded:
                    continue
                elem = self.graph_builder.element_by_id.get(related_id)
                if elem is None:
                    continue
                graph_score = result["total_score"] * 0.5 * self.graph_weight
                expanded[related_id] = {
                    "element": elem.to_dict(),
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "graph_score": graph_score,
                    "total_score": graph_score,
                    "related_to": elem_name,
                }

        # Convert back to list and sort
        results = list(expanded.values())
        results.sort(key=lambda x: x["total_score"], reverse=True)

        return results

    def _get_related_ids(self, element_id: str, element_meta: Dict[str, Any], max_hops: int = 2) -> Set[str]:
        use_ir = self.graph_backend == "ir" and self.ir_graphs is not None
        if use_ir:
            ids = self._get_related_ids_from_ir(element_id, element_meta, max_hops=max_hops)
            if ids:
                return ids
            if not self.allow_legacy_graph_fallback:
                return set()
            self.logger.debug("IR graph expansion yielded no results; falling back to legacy graph")
        return self.graph_builder.get_related_elements(element_id, max_hops)

    def _get_related_ids_from_ir(self, element_id: str, element_meta: Dict[str, Any], max_hops: int = 2) -> Set[str]:
        if self.ir_graphs is None:
            return set()
        seed = (
            (element_meta.get("metadata", {}) or {}).get("ir_symbol_id")
            or (element_meta.get("metadata", {}) or {}).get("ir_node_id")
        )
        if not seed:
            seed = self._heuristic_ir_seed(element_meta)
        if not seed:
            return set()

        g = nx.Graph()
        for graph in [
            self.ir_graphs.dependency_graph,
            self.ir_graphs.call_graph,
            self.ir_graphs.inheritance_graph,
            self.ir_graphs.reference_graph,
            self.ir_graphs.containment_graph,
        ]:
            if graph is None:
                continue
            g.add_nodes_from(graph.nodes())
            g.add_edges_from(graph.edges())

        if seed not in g:
            return set()
        related_ir_nodes = set(nx.single_source_shortest_path_length(g, seed, cutoff=max_hops).keys())
        related_ir_nodes.discard(seed)
        if not related_ir_nodes:
            return set()

        # Map IR nodes back to legacy element ids through lightweight heuristics.
        mapped: Set[str] = set()
        for elem in self.graph_builder.element_by_id.values():
            meta = elem.metadata or {}
            elem_ir = meta.get("ir_symbol_id") or meta.get("ir_node_id")
            if elem_ir and elem_ir in related_ir_nodes:
                mapped.add(elem.id)
                continue
            # fallback match by path and symbol name
            rel_path = getattr(elem, "relative_path", None) or getattr(elem, "file_path", None)
            name = getattr(elem, "name", None)
            for ir_id in related_ir_nodes:
                if rel_path and rel_path in ir_id:
                    mapped.add(elem.id)
                    break
                if name and name in ir_id:
                    mapped.add(elem.id)
                    break
        return mapped

    def _heuristic_ir_seed(self, element_meta: Dict[str, Any]) -> Optional[str]:
        rel_path = element_meta.get("relative_path") or element_meta.get("file_path")
        name = element_meta.get("name")
        if not rel_path and not name:
            return None

        candidates = []
        for graph in [
            self.ir_graphs.containment_graph if self.ir_graphs else None,
            self.ir_graphs.call_graph if self.ir_graphs else None,
        ]:
            if graph is None:
                continue
            for n in graph.nodes():
                if rel_path and rel_path in str(n):
                    candidates.append(str(n))
                elif name and name in str(n):
                    candidates.append(str(n))
        return candidates[0] if candidates else None
    
    def _rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank results based on additional factors"""
        # Simple re-ranking based on element type preferences
        type_weights = {
            "function": 1.2,  # Prefer functions
            "class": 1.1,     # Then classes
            "file": 0.9,      # Then files
            "documentation": 0.8,  # Then docs
            "design_document": 0.95,
        }
        
        for result in results:
            elem_type = result["element"].get("type", "")
            weight = type_weights.get(elem_type, 1.0)
            # Apply weight to all score components to maintain consistency
            result["total_score"] *= weight
            result["semantic_score"] *= weight
            result["keyword_score"] *= weight
            result["pseudocode_score"] *= weight
            result["graph_score"] *= weight
        
        # Sort by updated scores
        results.sort(key=lambda x: x["total_score"], reverse=True)
        
        return results
    
    def _apply_filters(self, results: List[Dict[str, Any]], 
                       filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to results"""
        filtered = []
        
        for result in results:
            elem = result["element"]
            
            # Check language filter
            if "language" in filters:
                if elem.get("language") != filters["language"]:
                    continue
            
            # Check type filter
            if "type" in filters:
                if elem.get("type") != filters["type"]:
                    continue
            
            # Check file path filter
            if "file_path" in filters:
                if filters["file_path"] not in elem.get("relative_path", ""):
                    continue

            # Check snapshot filter
            if "snapshot_id" in filters:
                elem_snapshot = elem.get("snapshot_id") or (elem.get("metadata", {}) or {}).get("snapshot_id")
                if elem_snapshot != filters["snapshot_id"]:
                    continue
            
            filtered.append(result)
        
        return filtered
    
    def _diversify(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Diversify results to avoid too many similar elements"""
        if not results or self.diversity_penalty == 0:
            return results
        
        diversified = []
        seen_files = set()
        
        for result in results:
            file_path = result["element"].get("file_path", "")
            
            # Penalize if we've seen this file too many times
            if file_path in seen_files:
                penalty_factor = (1 - self.diversity_penalty)
                result["total_score"] *= penalty_factor
                result["semantic_score"] *= penalty_factor
                result["keyword_score"] *= penalty_factor
                result["pseudocode_score"] *= penalty_factor
                result["graph_score"] *= penalty_factor
            else:
                seen_files.add(file_path)
            
            diversified.append(result)
        
        # Re-sort after diversification
        diversified.sort(key=lambda x: x["total_score"], reverse=True)
        
        return diversified
    
    def _final_repo_filter(self, results: List[Dict[str, Any]], repo_filter: List[str]) -> List[Dict[str, Any]]:
        """
        Final safety filter to ensure only results from selected repositories are returned
        This is a critical safety check to prevent leakage of results from other repositories
        
        Args:
            results: List of retrieval results
            repo_filter: List of allowed repository names
        
        Returns:
            Filtered results containing only elements from allowed repositories
        """
        if not repo_filter:
            return results
        
        filtered_results = []
        filtered_count = 0
        
        for result in results:
            elem = result["element"]
            repo_name = elem.get("repo_name", "")
            
            if repo_name in repo_filter:
                filtered_results.append(result)
            else:
                filtered_count += 1
                self.logger.warning(
                    f"Filtered out element from unexpected repo: {repo_name} "
                    f"(expected one of: {repo_filter}). Element: {elem.get('name', 'unknown')}"
                )
        
        if filtered_count > 0:
            self.logger.warning(
                f"Final repo filter removed {filtered_count} elements from unexpected repositories. "
                f"This indicates a potential issue in the retrieval pipeline."
            )
        
        return filtered_results
    
    def retrieve_by_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Retrieve all elements from a specific file"""
        results = []
        
        # Use filtered elements if available, otherwise use full elements
        elements_to_search = (self.filtered_bm25_elements 
                             if self.filtered_bm25_elements 
                             else self.full_bm25_elements)
        
        for elem in elements_to_search:
            if elem.file_path == file_path or elem.relative_path == file_path:
                results.append({
                    "element": elem.to_dict(),
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "graph_score": 0.0,
                    "total_score": 1.0,
                })
        
        return results
    
    def retrieve_by_type(self, element_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve elements by type"""
        results = []
        
        # Use filtered elements if available, otherwise use full elements
        elements_to_search = (self.filtered_bm25_elements 
                             if self.filtered_bm25_elements 
                             else self.full_bm25_elements)
        
        for elem in elements_to_search:
            if elem.type == element_type:
                results.append({
                    "element": elem.to_dict(),
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "graph_score": 0.0,
                    "total_score": 1.0,
                })
                
                if len(results) >= limit:
                    break
        
        return results
    
    def reload_specific_repositories(self, repo_names: List[str]) -> bool:
        """
        Reload specific repository indexes for accurate retrieval
        Populates FILTERED indexes while keeping FULL indexes intact
        
        Args:
            repo_names: List of repository names to reload
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Reloading specific repositories into filtered indexes: {repo_names}")
        
        try:
            # Create/clear filtered vector store (separate from main vector store)
            if self.filtered_vector_store is None:
                self.filtered_vector_store = VectorStore(self.config)
                self.filtered_vector_store.initialize(self.embedder.embedding_dim)
            else:
                self.filtered_vector_store.clear()
            
            # Load each repository's vector index and merge into FILTERED store
            loaded_count = 0
            for repo_name in repo_names:
                self.logger.info(f"Loading vector index for {repo_name}...")
                if self.filtered_vector_store.merge_from_index(repo_name):
                    self.logger.info(f"Successfully loaded {repo_name} vector index")
                    loaded_count += 1
                else:
                    self.logger.warning(f"Failed to load vector index for {repo_name}")
            
            if loaded_count == 0:
                self.logger.error("Failed to load any repository vector indexes")
                return False
            
            # Reload BM25 indexes for specific repositories into FILTERED indexes
            all_bm25_elements = []
            all_bm25_corpus = []
            
            for repo_name in repo_names:
                bm25_path = os.path.join(self.persist_dir, f"{repo_name}_bm25.pkl")
                if os.path.exists(bm25_path):
                    try:
                        with open(bm25_path, 'rb') as f:
                            data = pickle.load(f)
                            all_bm25_corpus.extend(data["bm25_corpus"])
                            
                            # Reconstruct CodeElement objects
                            for elem_dict in data["bm25_elements"]:
                                all_bm25_elements.append(CodeElement(**elem_dict))
                        
                        self.logger.info(f"Loaded BM25 index for {repo_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load BM25 index for {repo_name}: {e}")
                else:
                    self.logger.warning(f"BM25 index not found for {repo_name}")
            
            # Rebuild FILTERED BM25 with selected repositories
            if all_bm25_elements and all_bm25_corpus:
                self.filtered_bm25_elements = all_bm25_elements
                self.filtered_bm25_corpus = all_bm25_corpus
                self.filtered_bm25 = BM25Okapi(all_bm25_corpus)
                self.logger.info(f"Rebuilt filtered BM25 index with {len(all_bm25_elements)} elements")
            else:
                self.logger.warning("No BM25 data found for the specified repositories")
            
            # Optionally reload graph data (if needed)
            # Note: Graph reloading is commented out since it might not be necessary for all use cases
            # Uncomment if graph data needs to be reloaded as well
            # for i, repo_name in enumerate(repo_names):
            #     if i == 0:
            #         self.graph_builder.load(repo_name)
            #     else:
            #         self.graph_builder.merge_from_file(repo_name)
            
            # Update iterative_agent's bm25_elements reference to use filtered elements
            if self.iterative_agent is not None:
                self.iterative_agent.bm25_elements = self.filtered_bm25_elements
                self.logger.info("Updated iterative_agent with filtered BM25 elements")
            
            # Update iterative_agent's repo stats if needed
            if self.iterative_agent is not None:
                repo_stats = self._calculate_repo_stats()
                if repo_stats:
                    self.iterative_agent.set_repo_stats(repo_stats)
                    self.logger.info("Updated iterative_agent with repo stats")
            
            self.logger.info(
                f"Successfully reloaded {loaded_count} repositories with "
                f"{self.filtered_vector_store.get_count()} vectors and "
                f"{len(self.filtered_bm25_elements)} BM25 elements"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reload specific repositories: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def save_bm25(self, name: str = "index"):
        """
        Save FULL BM25 index and elements to disk
        
        Args:
            name: Name for the saved files
        """
        bm25_path = os.path.join(self.persist_dir, f"{name}_bm25.pkl")
        
        try:
            with open(bm25_path, 'wb') as f:
                pickle.dump({
                    "bm25_corpus": self.full_bm25_corpus,
                    "bm25_elements": [elem.to_dict() for elem in self.full_bm25_elements],
                }, f)
            
            self.logger.info(f"Saved full BM25 data to {bm25_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save BM25 data: {e}")
            return False
    
    def load_bm25(self, name: str = "index") -> bool:
        """
        Load FULL BM25 index and elements from disk
        
        Args:
            name: Name of the saved files
        
        Returns:
            True if successful, False otherwise
        """
        bm25_path = os.path.join(self.persist_dir, f"{name}_bm25.pkl")
        
        if not os.path.exists(bm25_path):
            self.logger.warning(f"BM25 data not found: {bm25_path}")
            return False
        
        try:
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
                self.full_bm25_corpus = data["bm25_corpus"]
                
                # Reconstruct CodeElement objects
                self.full_bm25_elements = []
                for elem_dict in data["bm25_elements"]:
                    self.full_bm25_elements.append(CodeElement(**elem_dict))
            
            # Rebuild FULL BM25 index from corpus
            if self.full_bm25_corpus:
                self.full_bm25 = BM25Okapi(self.full_bm25_corpus)
                self.logger.info(f"Loaded full BM25 data with {len(self.full_bm25_elements)} elements")
            else:
                self.logger.warning("BM25 corpus is empty")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load BM25 data: {e}")
            return False
    
    def _initialize_agents(self, repo_root: str) -> bool:
        """
        Initialize agents for agency mode
        
        Args:
            repo_root: Root directory of the repository
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.repo_root = repo_root
            # Pass BM25 elements reference to iterative agent for retrieving indexed details
            bm25_elements = self.filtered_bm25_elements if self.filtered_bm25_elements else self.full_bm25_elements
            self.iterative_agent = IterativeAgent(self.config, self, repo_root, bm25_elements=bm25_elements)

            # Set repo stats for iterative agent
            repo_stats = self._calculate_repo_stats()
            if repo_stats:
                self.iterative_agent.set_repo_stats(repo_stats)

            self.logger.info("Agency mode enabled with iterative agent")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to initialize agents: {e}")
            import traceback
            self.logger.warning(traceback.format_exc())
            return False
    
    def set_repo_root(self, repo_root: str):
        """
        Set repository root and initialize agents if agency mode is enabled
        
        Args:
            repo_root: Root directory of the repository
        """
        self.repo_root = repo_root
        if self.enable_agency_mode and not self.iterative_agent:
            self._initialize_agents(repo_root)
    
    def _calculate_repo_stats(self) -> Optional[Dict[str, Any]]:
        """
        Calculate repository statistics for cost estimation
        
        Returns:
            Dict with repo statistics or None if unavailable
        """
        try:
            elements = self.filtered_bm25_elements if self.filtered_bm25_elements else self.full_bm25_elements
            
            if not elements:
                return None
            
            total_files = 0
            total_classes = 0
            total_functions = 0
            total_lines = 0
            max_depth = 0
            
            seen_files = set()
            
            for elem in elements:
                # Count unique files
                if elem.relative_path not in seen_files:
                    seen_files.add(elem.relative_path)
                    total_files += 1
                    
                    # Calculate directory depth
                    depth = elem.relative_path.count('/') + elem.relative_path.count('\\')
                    max_depth = max(max_depth, depth)
                
                # Count by type
                if elem.type == "class":
                    total_classes += 1
                elif elem.type == "function":
                    total_functions += 1
                
                # Accumulate lines
                if elem.end_line > elem.start_line:
                    total_lines += (elem.end_line - elem.start_line + 1)
            
            avg_file_lines = total_lines / total_files if total_files > 0 else 0
            
            return {
                "total_files": total_files,
                "total_classes": total_classes,
                "total_functions": total_functions,
                "avg_file_lines": avg_file_lines,
                "max_depth": max_depth
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate repo stats: {e}")
            return None
    

    
    def _apply_agency_mode(self, query: str, results: List[Dict[str, Any]],
                          query_info: Dict[str, Any],
                          repo_filter: Optional[List[str]] = None,
                          dialogue_history: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Apply agency mode: iterative retrieval with confidence and cost control

        Args:
            query: User query
            results: Initial retrieval results (not used in iterative mode)
            query_info: Query information
            repo_filter: Optional list of repository names to filter by
            dialogue_history: Previous dialogue summaries for multi-turn context

        Returns:
            Enhanced results from iterative agency mode
        """
        self.logger.info("Applying iterative agency mode")

        try:
            # Use iterative agent if available
            if self.iterative_agent:
                # Get processed query (should be in query_info or we need to pass it)
                from .query_processor import ProcessedQuery

                # Create a ProcessedQuery object from query_info
                processed_query = ProcessedQuery(
                    original=query,
                    expanded=query_info.get("expanded", query),
                    keywords=query_info.get("keywords", []),
                    intent=query_info.get("intent", "unknown"),
                    subqueries=[],
                    filters=query_info.get("filters", {}),
                    rewritten_query=query_info.get("rewritten_query"),
                    pseudocode_hints=query_info.get("pseudocode_hints")
                )

                final_results, iteration_metadata = self.iterative_agent.retrieve_with_iteration(
                    query, processed_query, query_info, repo_filter, dialogue_history
                )

                self.logger.info(f"Iterative agent completed: {iteration_metadata}")

                return final_results

            # Fallback: iterative agent not available
            else:
                self.logger.warning("Iterative agent not available, returning original results")
                return results

        except Exception as e:
            self.logger.error(f"Error in agency mode: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Fallback to original results
            return results
