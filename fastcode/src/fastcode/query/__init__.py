"""Query processing, answer generation, and repository selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .processor import ProcessedQuery, QueryProcessor
from .selector import RepositorySelector

if TYPE_CHECKING:
    from .answer import AnswerGenerator
    from .handler import QueryPipeline

__all__ = [
    "AnswerGenerator",
    "ProcessedQuery",
    "QueryPipeline",
    "QueryProcessor",
    "RepositorySelector",
]


def __getattr__(name: str) -> Any:
    if name == "AnswerGenerator":
        from .answer import AnswerGenerator

        return AnswerGenerator
    if name == "QueryPipeline":
        from .handler import QueryPipeline

        return QueryPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
