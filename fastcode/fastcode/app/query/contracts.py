"""Frozen query-shell records.

These records are the internal shape used after API or LLM boundary
validation.  Dict payload helpers exist only for legacy adapter seams that
still expect JSON-like values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


def _empty_payload() -> dict[str, Any]:
    return {}


class QueryIntent(StrEnum):
    GENERAL = "general"
    HOW = "how"
    WHAT = "what"
    WHERE = "where"
    DEBUG = "debug"
    EXPLAIN = "explain"
    FIND = "find"
    IMPLEMENT = "implement"
    CODE_QA = "code_qa"
    DOCUMENT_QA = "document_qa"
    API_USAGE = "api_usage"
    BUG_FIXING = "bug_fixing"
    FEATURE_ADDITION = "feature_addition"
    ARCHITECTURE = "architecture"
    CROSS_REPO = "cross_repo"
    UNKNOWN = "unknown"


class ToolStatus(StrEnum):
    REQUESTED = "requested"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StreamEventType(StrEnum):
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass(frozen=True)
class QueryEnhancement:
    """Validated query-enhancement payload produced by an LLM boundary."""

    refined_intent: str | None = None
    rewritten_query: str | None = None
    selected_keywords: tuple[str, ...] = ()
    pseudocode_hints: str | None = None
    search_strategy: str | None = None
    needed: bool = False


@dataclass(frozen=True)
class ToolCallRecord:
    """Validated LLM-requested tool invocation."""

    tool: str
    parameters: dict[str, Any] = field(default_factory=_empty_payload)


@dataclass(frozen=True)
class AgentRoundResult:
    """Validated result of one iterative-agent LLM round."""

    confidence: int
    reasoning: str
    tool_calls: tuple[ToolCallRecord, ...] = ()
    keep_files: tuple[str, ...] = ()
    query_complexity: int | None = None
    query_enhancement: QueryEnhancement | None = None
    should_answer_directly: bool = False


@dataclass(frozen=True)
class ElementSelectionRecord:
    """Validated LLM-selected element candidate."""

    file_path: str
    element_type: str = "file"
    name: str = ""
    repo_name: str = ""


def empty_query_enhancement() -> QueryEnhancement:
    return QueryEnhancement()


def query_enhancement_payload(record: QueryEnhancement) -> dict[str, Any]:
    """Materialize a JSON-like payload field-by-field."""
    payload: dict[str, Any] = {
        "selected_keywords": list(record.selected_keywords),
        "needed": record.needed,
    }
    if record.refined_intent is not None:
        payload["refined_intent"] = record.refined_intent
    if record.rewritten_query is not None:
        payload["rewritten_query"] = record.rewritten_query
    if record.pseudocode_hints is not None:
        payload["pseudocode_hints"] = record.pseudocode_hints
    if record.search_strategy is not None:
        payload["search_strategy"] = record.search_strategy
    return payload


def tool_call_payload(record: ToolCallRecord) -> dict[str, Any]:
    return {
        "tool": record.tool,
        "parameters": dict(record.parameters),
    }


def agent_round_result_payload(record: AgentRoundResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "confidence": record.confidence,
        "reasoning": record.reasoning,
        "tool_calls": [tool_call_payload(call) for call in record.tool_calls],
        "should_answer_directly": record.should_answer_directly,
    }
    if record.query_complexity is not None:
        payload["query_complexity"] = record.query_complexity
    if record.keep_files:
        payload["keep_files"] = list(record.keep_files)
    if record.query_enhancement is not None:
        payload["query_enhancement"] = query_enhancement_payload(
            record.query_enhancement
        )
    return payload


def element_selection_payload(record: ElementSelectionRecord) -> dict[str, Any]:
    return {
        "file_path": record.file_path,
        "type": record.element_type,
        "name": record.name,
        "repo_name": record.repo_name,
    }
