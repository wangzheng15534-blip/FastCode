"""Query-shell boundary validators and explicit mappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fastcode.app.query.contracts import (
    AgentRoundResult,
    ElementSelectionRecord,
    QueryEnhancement,
    ToolCallRecord,
)

if TYPE_CHECKING:
    from fastcode.app.query.orchestration.processor import ProcessedQuery


def _empty_str_list() -> list[str]:
    return []


def _empty_tool_call_dtos() -> list[ToolCallDTO]:
    return []


def _empty_element_selection_dtos() -> list[ElementSelectionDTO]:
    return []


class QueryEnhancementDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")

    refined_intent: str | None = None
    rewritten_query: str | None = None
    selected_keywords: list[str] = Field(default_factory=_empty_str_list)
    pseudocode_hints: str | None = None
    search_strategy: str | None = None
    needed: bool | None = None

    @field_validator("selected_keywords", mode="before")
    @classmethod
    def _keywords_from_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


class ToolCallDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tool: str = Field(..., min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class RoundOneResponseDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")

    confidence: int = Field(..., ge=0, le=100)
    reasoning: str = ""
    query_complexity: int | None = Field(None, ge=0, le=100)
    query_enhancement: QueryEnhancementDTO | str | None = None
    tool_calls: list[ToolCallDTO] = Field(default_factory=_empty_tool_call_dtos)


class RoundNResponseDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")

    keep_files: list[str] = Field(default_factory=_empty_str_list)
    confidence: int = Field(..., ge=0, le=100)
    reasoning: str = ""
    tool_calls: list[ToolCallDTO] = Field(default_factory=_empty_tool_call_dtos)


class ElementSelectionDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file_path: str = Field(..., min_length=1)
    type: str = "file"
    name: str = ""
    repo_name: str = ""


class ElementSelectionResponseDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")

    selected_elements: list[ElementSelectionDTO] = Field(
        default_factory=_empty_element_selection_dtos
    )


def processed_query_payload(processed_query: ProcessedQuery) -> dict[str, Any]:
    """Materialize ProcessedQuery for legacy prompt/retrieval adapters."""
    return {
        "original": processed_query.original,
        "expanded": processed_query.expanded,
        "keywords": list(processed_query.keywords),
        "intent": processed_query.intent,
        "subqueries": list(processed_query.subqueries),
        "filters": dict(processed_query.filters),
        "rewritten_query": processed_query.rewritten_query,
        "pseudocode_hints": processed_query.pseudocode_hints,
        "search_strategy": processed_query.search_strategy,
    }


def query_enhancement_from_dto(dto: QueryEnhancementDTO) -> QueryEnhancement:
    selected_keywords = tuple(
        item.strip() for item in dto.selected_keywords if item.strip()
    )
    needed = bool(dto.needed)
    if dto.needed is None:
        needed = any(
            (
                bool(dto.refined_intent),
                bool(dto.rewritten_query),
                bool(selected_keywords),
                bool(dto.pseudocode_hints),
                bool(dto.search_strategy),
            )
        )
    return QueryEnhancement(
        refined_intent=dto.refined_intent,
        rewritten_query=dto.rewritten_query,
        selected_keywords=selected_keywords,
        pseudocode_hints=dto.pseudocode_hints,
        search_strategy=dto.search_strategy,
        needed=needed,
    )


def tool_call_from_dto(dto: ToolCallDTO) -> ToolCallRecord:
    return ToolCallRecord(tool=dto.tool, parameters=dict(dto.parameters))


def round_one_from_dto(
    dto: RoundOneResponseDTO,
    *,
    confidence_threshold: int,
    fallback_enhancement: QueryEnhancement | None = None,
) -> AgentRoundResult:
    enhancement = None
    if isinstance(dto.query_enhancement, QueryEnhancementDTO):
        enhancement = query_enhancement_from_dto(dto.query_enhancement)
    elif fallback_enhancement is not None:
        enhancement = fallback_enhancement

    return AgentRoundResult(
        confidence=dto.confidence,
        reasoning=dto.reasoning,
        query_complexity=dto.query_complexity,
        query_enhancement=enhancement,
        tool_calls=tuple(tool_call_from_dto(call) for call in dto.tool_calls),
        should_answer_directly=dto.confidence >= confidence_threshold,
    )


def round_n_from_dto(dto: RoundNResponseDTO) -> AgentRoundResult:
    return AgentRoundResult(
        confidence=dto.confidence,
        reasoning=dto.reasoning,
        keep_files=tuple(item for item in dto.keep_files if item),
        tool_calls=tuple(tool_call_from_dto(call) for call in dto.tool_calls),
    )


def element_selection_from_dto(dto: ElementSelectionDTO) -> ElementSelectionRecord:
    return ElementSelectionRecord(
        file_path=dto.file_path,
        element_type=dto.type,
        name=dto.name,
        repo_name=dto.repo_name,
    )
