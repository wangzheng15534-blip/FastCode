"""Frozen retrieval contracts owned by the retrieval domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast


def _empty_payload() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class Hit:
    """A retrieval result carrying per-channel scores and provenance flags."""

    element_id: str
    element_type: str
    element_name: str
    score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    pseudocode_score: float = 0.0
    graph_score: float = 0.0
    total_score: float = 0.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=_empty_payload)
    projected_only: bool = False
    llm_selected: bool = False
    agent_found: bool = False

    @classmethod
    def from_retrieval_row(cls, row: dict[str, Any]) -> Hit:
        raw_elem = row.get("element")
        elem = cast(dict[str, Any], raw_elem) if isinstance(raw_elem, dict) else {}
        raw_metadata = elem.get("metadata")
        metadata = (
            dict(cast(dict[str, Any], raw_metadata))
            if isinstance(raw_metadata, dict)
            else {}
        )
        return cls(
            element_id=str(elem.get("id") or ""),
            element_type=str(elem.get("type") or ""),
            element_name=str(elem.get("name") or ""),
            score=float(row.get("score") or 0.0),
            semantic_score=float(row.get("semantic_score") or 0.0),
            keyword_score=float(row.get("keyword_score") or 0.0),
            pseudocode_score=float(row.get("pseudocode_score") or 0.0),
            graph_score=float(row.get("graph_score") or 0.0),
            total_score=float(row.get("total_score") or 0.0),
            source=str(row.get("source") or ""),
            metadata=metadata,
            projected_only=bool(row.get("projected_only")),
            llm_selected=bool(row.get("llm_file_selected")),
            agent_found=bool(row.get("agent_found")),
        )

    def to_retrieval_row(self) -> dict[str, Any]:
        return {
            "element": {
                "id": self.element_id,
                "type": self.element_type,
                "name": self.element_name,
                "metadata": dict(self.metadata),
            },
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "total_score": self.total_score,
            "score": self.score,
            "pseudocode_score": self.pseudocode_score,
            "graph_score": self.graph_score,
            "source": self.source,
            "projected_only": self.projected_only,
            "llm_file_selected": self.llm_selected,
            "agent_found": self.agent_found,
        }


@dataclass(frozen=True)
class FusionConfig:
    """Adaptive fusion hyper-parameters."""

    alpha_base: float = 0.8
    alpha_min: float = 0.25
    alpha_max: float = 0.9
    rrf_k_base: int = 60
    rrf_k_min: int = 20
    rrf_k_max: int = 100

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FusionConfig:
        return cls(
            alpha_base=float(d["alpha_base"]) if "alpha_base" in d else cls.alpha_base,
            alpha_min=float(d["alpha_min"]) if "alpha_min" in d else cls.alpha_min,
            alpha_max=float(d["alpha_max"]) if "alpha_max" in d else cls.alpha_max,
            rrf_k_base=int(d["rrf_k_base"]) if "rrf_k_base" in d else cls.rrf_k_base,
            rrf_k_min=int(d["rrf_k_min"]) if "rrf_k_min" in d else cls.rrf_k_min,
            rrf_k_max=int(d["rrf_k_max"]) if "rrf_k_max" in d else cls.rrf_k_max,
        )


@dataclass(frozen=True)
class FusionWeights:
    """Weights for cross-collection fusion."""

    code_weight: float = 0.7
    doc_weight: float = 0.3
    alpha: float = 0.8
    beta: float = 0.35
    rrf_k_code: float = 60.0
    rrf_k_doc: float = 60.0


@dataclass(frozen=True)
class RetrievalChannelOutput:
    """Output from a single retrieval channel."""

    collection: str
    semantic_results: tuple[dict[str, Any], ...]
    keyword_results: tuple[dict[str, Any], ...]
    pseudocode_results: tuple[dict[str, Any], ...] = ()
    ranked_results: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ElementFilter:
    """Filter criteria for retrieval results."""

    language: str | None = None
    element_type: str | None = None
    file_path: str | None = None
    snapshot_id: str | None = None


@dataclass(frozen=True)
class IterationConfig:
    """Parameters governing the adaptive iteration loop."""

    base_max_iterations: int = 4
    base_confidence_threshold: int = 95
    min_confidence_gain: float = 0.5
    max_total_lines: int = 12000
    max_iterations: int = 4
    confidence_threshold: int = 95
    adaptive_line_budget: int = 12000
    max_elements: int = 30
    max_candidates_display: int = 40
    temperature: float = 0.3
    max_tokens: int = 4096


@dataclass(frozen=True)
class IterationHistoryEntry:
    """Metrics captured after a single iteration round."""

    round: int
    confidence: int
    query_complexity: int
    elements_count: int
    total_lines: int
    confidence_gain: float
    lines_added: int
    roi: float
    budget_usage_pct: float


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the LLM."""

    tool: str
    parameters: dict[str, Any] = field(default_factory=_empty_payload)


@dataclass(frozen=True)
class RoundResult:
    """Structured result from parsing an LLM iteration round."""

    confidence: int
    tool_calls: tuple[ToolCall, ...]
    keep_files: tuple[str, ...]
    reasoning: str
    query_complexity: int | None = None
    should_answer_directly: bool = False


@dataclass(frozen=True)
class IterationMetrics:
    """Aggregate metrics after the iteration loop completes."""

    rounds: int
    answered_directly: bool
    query_complexity: int
    initial_confidence: int
    final_confidence: int
    confidence_gain: int
    total_elements: int
    total_lines: int
    budget_used_pct: float
    iterations_used_pct: float
    overall_roi: float
    round_efficiencies: tuple[dict[str, Any], ...]
    adaptive_params: dict[str, Any]
    stopping_reason: str
    efficiency_rating: str


@dataclass(frozen=True)
class IterationState:
    """Immutable state carried across iteration rounds."""

    round_num: int
    elements: tuple[Hit, ...]
    history: tuple[IterationHistoryEntry, ...]
    tool_call_history: tuple[ToolCall, ...]
    retained_elements: tuple[Hit, ...] = ()
    pending_elements: tuple[Hit, ...] = ()
    confidence: int = 0
    dialogue_history: tuple[dict[str, Any], ...] = ()

    def with_elements(self, new_elements: tuple[Hit, ...]) -> IterationState:
        return IterationState(
            round_num=self.round_num,
            elements=new_elements,
            history=self.history,
            tool_call_history=self.tool_call_history,
            retained_elements=self.retained_elements,
            pending_elements=self.pending_elements,
            confidence=self.confidence,
            dialogue_history=self.dialogue_history,
        )

    def with_history_entry(self, entry: IterationHistoryEntry) -> IterationState:
        return IterationState(
            round_num=self.round_num,
            elements=self.elements,
            history=(*self.history, entry),
            tool_call_history=self.tool_call_history,
            retained_elements=self.retained_elements,
            pending_elements=self.pending_elements,
            confidence=self.confidence,
            dialogue_history=self.dialogue_history,
        )

    def with_tool_calls(self, calls: tuple[ToolCall, ...]) -> IterationState:
        return IterationState(
            round_num=self.round_num,
            elements=self.elements,
            history=self.history,
            tool_call_history=(*self.tool_call_history, *calls),
            retained_elements=self.retained_elements,
            pending_elements=self.pending_elements,
            confidence=self.confidence,
            dialogue_history=self.dialogue_history,
        )

    def next_round(self) -> IterationState:
        return IterationState(
            round_num=self.round_num + 1,
            elements=self.elements,
            history=self.history,
            tool_call_history=self.tool_call_history,
            retained_elements=self.retained_elements,
            pending_elements=self.pending_elements,
            confidence=self.confidence,
            dialogue_history=self.dialogue_history,
        )


@dataclass(frozen=True)
class GenerationInput:
    """Structured input for answer generation."""

    query: str
    context: str
    prompt_tokens: int
    max_tokens: int
    dialogue_history: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SourceRef:
    """A source reference cited in a generated answer."""

    path: str
    name: str
    line: int = 0
    element_type: str = ""
    repo_name: str = ""


@dataclass(frozen=True)
class GenerationResult:
    """Structured output from answer generation."""

    answer: str
    sources: tuple[SourceRef, ...]
    prompt_tokens: int
    summary: str | None = None
    error: str | None = None
