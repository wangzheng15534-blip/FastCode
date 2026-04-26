"""Core frozen dataclasses for the FP refactoring.

Three Golden Rules:
1. Pydantic Stops at the Door -- no pydantic imports in core/
2. Database Trusts Dataclasses -- all return types are frozen dataclasses
3. Explicit Translation -- explicit field mapping, no **kwargs unpacking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# ScipKind / ScipRole -- string constants
# ---------------------------------------------------------------------------


class ScipKind:
    """SCIP symbol kind constants."""

    FUNCTION = "Function"
    METHOD = "Method"
    CLASS = "Class"
    MODULE = "Module"
    INTERFACE = "Interface"
    ENUM = "Enum"
    VARIABLE = "Variable"
    CONSTANT = "Constant"
    PROPERTY = "Property"
    TYPE = "Type"
    UNKNOWN = "Unknown"


class ScipRole:
    """SCIP symbol occurrence role constants."""

    DEFINITION = "Definition"
    REFERENCE = "Reference"
    IMPORT = "Import"
    WRITE_ACCESS = "WriteAccess"
    FORWARD_DEFINITION = "ForwardDefinition"


# ---------------------------------------------------------------------------
# Hit -- retrieval result with scores and provenance
# ---------------------------------------------------------------------------


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
    metadata: dict[str, Any] = field(default_factory=dict)
    projected_only: bool = False
    llm_selected: bool = False
    agent_found: bool = False

    # -- factory / converter -------------------------------------------------

    @classmethod
    def from_retrieval_row(cls, row: dict[str, Any]) -> Hit:
        """Construct from the Dict[str, Any] format used by ``retriever.py``.

        Expected row structure::

            {
                "element": {"id": ..., "type": ..., "name": ..., "metadata": {...}},
                "semantic_score": ...,
                "keyword_score": ...,
                "total_score": ...,
                "projected_only": ...,
                "llm_file_selected": ...,
                "agent_found": ...,
            }
        """
        elem = (
            (row.get("element") or {}) if isinstance(row.get("element"), dict) else {}
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
            metadata=dict(elem.get("metadata") or {}),
            projected_only=bool(row.get("projected_only")),
            llm_selected=bool(row.get("llm_file_selected")),
            agent_found=bool(row.get("agent_found")),
        )

    def to_retrieval_row(self) -> dict[str, Any]:
        """Convert back to the dict format consumed by retriever.py."""
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


# ---------------------------------------------------------------------------
# FusionConfig -- adaptive fusion parameters
# ---------------------------------------------------------------------------


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
        """Construct from a dict with explicit field mapping."""
        return cls(
            alpha_base=float(d["alpha_base"]) if "alpha_base" in d else cls.alpha_base,
            alpha_min=float(d["alpha_min"]) if "alpha_min" in d else cls.alpha_min,
            alpha_max=float(d["alpha_max"]) if "alpha_max" in d else cls.alpha_max,
            rrf_k_base=int(d["rrf_k_base"]) if "rrf_k_base" in d else cls.rrf_k_base,
            rrf_k_min=int(d["rrf_k_min"]) if "rrf_k_min" in d else cls.rrf_k_min,
            rrf_k_max=int(d["rrf_k_max"]) if "rrf_k_max" in d else cls.rrf_k_max,
        )


# ---------------------------------------------------------------------------
# FusionWeights -- cross-collection weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionWeights:
    """Weights for cross-collection (code + doc) fusion."""

    code_weight: float = 0.7
    doc_weight: float = 0.3
    alpha: float = 0.8
    beta: float = 0.35
    rrf_k_code: float = 60.0
    rrf_k_doc: float = 60.0


# ---------------------------------------------------------------------------
# RetrievalChannelOutput -- output from one retrieval channel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalChannelOutput:
    """Output from a single retrieval channel (code or docs)."""

    collection: str
    semantic_results: tuple
    keyword_results: tuple
    pseudocode_results: tuple = ()
    ranked_results: tuple = ()


# ---------------------------------------------------------------------------
# ElementFilter -- filter criteria
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElementFilter:
    """Filter criteria for retrieval results."""

    language: str | None = None
    element_type: str | None = None
    file_path: str | None = None
    snapshot_id: str | None = None


# ---------------------------------------------------------------------------
# IterationConfig -- adaptive iteration parameters
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# IterationHistoryEntry -- one round of iteration history
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ToolCall -- tool call from LLM
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the LLM."""

    tool: str
    parameters: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RoundResult -- parsed LLM round result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundResult:
    """Structured result from parsing an LLM iteration round."""

    confidence: int
    tool_calls: tuple[ToolCall, ...]
    keep_files: tuple[str, ...]
    reasoning: str
    query_complexity: int | None = None
    should_answer_directly: bool = False


# ---------------------------------------------------------------------------
# IterationMetrics -- final metrics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# IterationState -- immutable state threaded through rounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IterationState:
    """Immutable state carried across iteration rounds.

    Every mutator returns a *new* ``IterationState`` instance.
    """

    round_num: int
    elements: tuple[Hit, ...]
    history: tuple[IterationHistoryEntry, ...]
    tool_call_history: tuple[ToolCall, ...]
    retained_elements: tuple[Hit, ...] = ()
    pending_elements: tuple[Hit, ...] = ()
    confidence: int = 0
    dialogue_history: tuple[dict[str, Any], ...] = ()

    def with_elements(self, new_elements: tuple[Hit, ...]) -> IterationState:
        """Return a new state with ``elements`` replaced."""
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
        """Return a new state with ``entry`` appended to ``history``."""
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
        """Return a new state with ``calls`` appended to ``tool_call_history``."""
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
        """Return a new state with ``round_num`` incremented by 1."""
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


# ---------------------------------------------------------------------------
# GenerationInput / GenerationResult
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# FileAnalysis / RepoStructure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileAnalysis:
    """Repository file analysis."""

    total_files: int
    languages: dict[str, int] = field(default_factory=dict)
    file_types: dict[str, int] = field(default_factory=dict)
    key_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class SnapshotRecord:
    """A snapshot metadata row from the database."""

    snapshot_id: str
    repo_name: str
    branch: str | None = None
    commit_id: str | None = None
    tree_id: str | None = None


@dataclass(frozen=True)
class RepoStructure:
    """Complete repository overview."""

    repo_name: str
    summary: str
    analysis: FileAnalysis
    has_readme: bool = False
    readme_content: str | None = None
    structure_text: str | None = None
