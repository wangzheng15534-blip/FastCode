"""Inbound DTOs for retrieval, generation, query, and agent config."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from ._config_schema_base import _ConfigDTO


def _dict_any() -> dict[str, Any]:
    return {}


class RetrievalBackendDTO(StrEnum):
    PG_HYBRID = "pg_hybrid"
    LOCAL = "local"


class GraphExpansionBackendDTO(StrEnum):
    IR = "ir"
    GRAPH_BUILDER = "graph_builder"


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
