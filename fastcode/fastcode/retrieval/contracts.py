"""Frozen retrieval contracts owned by the retrieval domain."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, cast


def _empty_payload() -> dict[str, Any]:
    return {}


class RetrievalSource(StrEnum):
    """String-backed retrieval provenance used at API and prompt boundaries."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    PSEUDOCODE = "pseudocode"
    GRAPH = "graph"
    TOOL = "tool"
    LLM_SELECTION = "llm_selection"
    RETRIEVAL = "retrieval"
    SCIP = "scip"
    UNKNOWN = "unknown"


def _string_value(value: Any) -> str:
    return str(value) if value is not None else ""


def _optional_string_value(value: Any) -> str | None:
    return str(value) if value is not None else None


def _int_value(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mapping_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    mapping = cast(Mapping[Any, Any], value)
    return {str(key): item for key, item in mapping.items()}


def _source_value(value: Any) -> RetrievalSource:
    if isinstance(value, RetrievalSource):
        return value
    raw = str(value or "").strip()
    if not raw:
        return RetrievalSource.UNKNOWN
    try:
        return RetrievalSource(raw)
    except ValueError:
        return RetrievalSource.UNKNOWN


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
    source: RetrievalSource | str = RetrievalSource.UNKNOWN
    language: str = ""
    file_path: str = ""
    relative_path: str = ""
    repo_name: str = ""
    repo_url: str = ""
    snapshot_id: str = ""
    start_line: int = 0
    end_line: int = 0
    signature: str | None = None
    docstring: str | None = None
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=_empty_payload)
    element_extra: dict[str, Any] = field(default_factory=_empty_payload)
    extra: dict[str, Any] = field(default_factory=_empty_payload)
    retrieval_score: float = 0.0
    projection_score: float = 0.0
    seed_score: float = 0.0
    traceability: tuple[ProjectionEvidence, ...] = ()
    projected_only: bool = False
    llm_selected: bool = False
    agent_found: bool = False
    related_to: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _source_value(self.source))

    @property
    def source_kind(self) -> RetrievalSource:
        return _source_value(self.source)

    @classmethod
    def from_retrieval_row(cls, row: dict[str, Any]) -> Hit:
        raw_elem = row.get("element")
        elem = (
            cast(Mapping[str, Any], raw_elem) if isinstance(raw_elem, Mapping) else {}
        )
        metadata = _mapping_payload(elem.get("metadata"))
        element_known = {
            "id",
            "type",
            "element_type",
            "name",
            "language",
            "file_path",
            "relative_path",
            "repo_name",
            "repo_url",
            "snapshot_id",
            "start_line",
            "end_line",
            "signature",
            "docstring",
            "summary",
            "metadata",
        }
        top_level_known = {
            "element",
            "score",
            "semantic_score",
            "keyword_score",
            "pseudocode_score",
            "graph_score",
            "total_score",
            "source",
            "retrieval_score",
            "projection_score",
            "seed_score",
            "traceability",
            "projected_only",
            "llm_file_selected",
            "agent_found",
            "related_to",
        }
        raw_type = elem.get("type")
        if raw_type is None:
            raw_type = elem.get("element_type")
        total_score = _float_value(row.get("total_score"))
        score = _float_value(row.get("score"))
        if score == 0.0 and total_score != 0.0:
            score = total_score
        return cls(
            element_id=_string_value(elem.get("id")),
            element_type=_string_value(raw_type),
            element_name=_string_value(elem.get("name")),
            score=score,
            semantic_score=_float_value(row.get("semantic_score")),
            keyword_score=_float_value(row.get("keyword_score")),
            pseudocode_score=_float_value(row.get("pseudocode_score")),
            graph_score=_float_value(row.get("graph_score")),
            total_score=total_score,
            source=_source_value(row.get("source")),
            language=_string_value(elem.get("language")),
            file_path=_string_value(elem.get("file_path")),
            relative_path=_string_value(elem.get("relative_path")),
            repo_name=_string_value(elem.get("repo_name")),
            repo_url=_string_value(elem.get("repo_url")),
            snapshot_id=_string_value(elem.get("snapshot_id")),
            start_line=_int_value(elem.get("start_line")),
            end_line=_int_value(elem.get("end_line")),
            signature=_optional_string_value(elem.get("signature")),
            docstring=_optional_string_value(elem.get("docstring")),
            summary=_optional_string_value(elem.get("summary")),
            metadata=metadata,
            element_extra={
                str(key): value
                for key, value in elem.items()
                if str(key) not in element_known
            },
            extra={
                str(key): value
                for key, value in row.items()
                if str(key) not in top_level_known
            },
            retrieval_score=_float_value(row.get("retrieval_score")),
            projection_score=_float_value(row.get("projection_score")),
            seed_score=_float_value(row.get("seed_score")),
            traceability=tuple(
                cast(tuple[ProjectionEvidence, ...], row.get("traceability") or ())
            ),
            projected_only=bool(row.get("projected_only")),
            llm_selected=bool(row.get("llm_file_selected")),
            agent_found=bool(row.get("agent_found")),
            related_to=_string_value(row.get("related_to")),
        )

    @classmethod
    def from_element(
        cls,
        element: Mapping[str, Any],
        *,
        score: float,
        source: RetrievalSource,
        semantic_score: float = 0.0,
        keyword_score: float = 0.0,
        pseudocode_score: float = 0.0,
        graph_score: float = 0.0,
        total_score: float | None = None,
        related_to: str = "",
    ) -> Hit:
        row = {
            "element": dict(element),
            "score": score,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "pseudocode_score": pseudocode_score,
            "graph_score": graph_score,
            "total_score": score if total_score is None else total_score,
            "source": source.value,
            "related_to": related_to,
        }
        return cls.from_retrieval_row(row)

    def to_retrieval_row(self) -> dict[str, Any]:
        element: dict[str, Any] = dict(self.element_extra)
        element.update(
            {
                "id": self.element_id,
                "type": self.element_type,
                "name": self.element_name,
                "language": self.language,
                "file_path": self.file_path,
                "relative_path": self.relative_path,
                "repo_name": self.repo_name,
                "repo_url": self.repo_url,
                "snapshot_id": self.snapshot_id,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "signature": self.signature,
                "docstring": self.docstring,
                "summary": self.summary,
                "metadata": dict(self.metadata),
            }
        )
        row = dict(self.extra)
        row.update(
            {
                "element": element,
                "semantic_score": self.semantic_score,
                "keyword_score": self.keyword_score,
                "total_score": self.total_score,
                "score": self.score,
                "pseudocode_score": self.pseudocode_score,
                "graph_score": self.graph_score,
                "source": self.source_kind.value,
                "retrieval_score": self.retrieval_score,
                "projection_score": self.projection_score,
                "seed_score": self.seed_score,
                "traceability": list(self.traceability),
                "projected_only": self.projected_only,
                "llm_file_selected": self.llm_selected,
                "agent_found": self.agent_found,
            }
        )
        if self.related_to:
            row["related_to"] = self.related_to
        return row

    def with_scores(
        self,
        *,
        score: float | None = None,
        semantic_score: float | None = None,
        keyword_score: float | None = None,
        pseudocode_score: float | None = None,
        graph_score: float | None = None,
        total_score: float | None = None,
        retrieval_score: float | None = None,
        projection_score: float | None = None,
        seed_score: float | None = None,
        traceability: tuple[ProjectionEvidence, ...] | None = None,
        projected_only: bool | None = None,
    ) -> Hit:
        return replace(
            self,
            score=self.score if score is None else score,
            semantic_score=(
                self.semantic_score if semantic_score is None else semantic_score
            ),
            keyword_score=self.keyword_score
            if keyword_score is None
            else keyword_score,
            pseudocode_score=(
                self.pseudocode_score if pseudocode_score is None else pseudocode_score
            ),
            graph_score=self.graph_score if graph_score is None else graph_score,
            total_score=self.total_score if total_score is None else total_score,
            retrieval_score=(
                self.retrieval_score if retrieval_score is None else retrieval_score
            ),
            projection_score=(
                self.projection_score if projection_score is None else projection_score
            ),
            seed_score=self.seed_score if seed_score is None else seed_score,
            traceability=self.traceability if traceability is None else traceability,
            projected_only=self.projected_only
            if projected_only is None
            else projected_only,
        )

    def scaled_scores(self, factor: float) -> Hit:
        return self.with_scores(
            score=self.score * factor,
            semantic_score=self.semantic_score * factor,
            keyword_score=self.keyword_score * factor,
            pseudocode_score=self.pseudocode_score * factor,
            graph_score=self.graph_score * factor,
            total_score=self.total_score * factor,
        )


@dataclass(frozen=True)
class FusionConfig:
    """Adaptive fusion hyper-parameters."""

    alpha_base: float = 0.8
    alpha_min: float = 0.25
    alpha_max: float = 0.9
    rrf_k_base: int = 60
    rrf_k_min: int = 20
    rrf_k_max: int = 100


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
class TraceLink:
    """Grounded doc-to-code trace link extracted from retrieval metadata."""

    unit_id: str
    weight: float
    evidence_type: str
    chunk_id: str
    symbol_name: str | None = None
    confidence: str | None = None


@dataclass(frozen=True)
class ProjectionEvidence:
    """Evidence contribution from a document hit to a projected code unit."""

    link: TraceLink
    doc_id: str
    doc_score: float
    contribution: float


@dataclass(frozen=True)
class DocProjectionPriors:
    """Doc-derived projection priors keyed by IR unit id."""

    p_doc: float
    beta: float
    priors: dict[str, float] = field(default_factory=_empty_payload)
    evidence: dict[str, tuple[ProjectionEvidence, ...]] = field(
        default_factory=_empty_payload
    )


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
class ToolHistoryEntry:
    """A tool invocation with the round that produced it."""

    round: int
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
class SourceCitation:
    """Source citation used by answer formatting and API serialization."""

    repository: str
    file: str
    name: str
    element_type: str
    lines: str
    score: float


@dataclass(frozen=True)
class AnswerDisplayResult:
    """Typed answer payload for user-facing formatting."""

    answer: str
    sources: tuple[SourceCitation, ...] = ()
    prompt_tokens: int | None = None
    context_elements: int = 0


@dataclass(frozen=True)
class GenerationResult:
    """Structured output from answer generation."""

    answer: str
    sources: tuple[SourceRef, ...]
    prompt_tokens: int
    summary: str | None = None
    error: str | None = None
