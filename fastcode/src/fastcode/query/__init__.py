"""Query processing, answer generation, and repository selection exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .answer import AnswerGenerator
    from .handler import QueryPipeline
    from .processor import ProcessedQuery, QueryProcessor
    from .selector import RepositorySelector

_EXPORTS = {
    "AnswerGenerator": ("fastcode.query.answer", "AnswerGenerator"),
    "ProcessedQuery": ("fastcode.query.processor", "ProcessedQuery"),
    "QueryPipeline": ("fastcode.query.handler", "QueryPipeline"),
    "QueryProcessor": ("fastcode.query.processor", "QueryProcessor"),
    "RepositorySelector": ("fastcode.query.selector", "RepositorySelector"),
}

__all__ = [
    "AnswerGenerator",
    "ProcessedQuery",
    "QueryPipeline",
    "QueryProcessor",
    "RepositorySelector",
]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
