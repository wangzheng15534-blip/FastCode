"""
FastCode 2.0 - Repository-Level Code Understanding System
With Multi-Repository Support
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "2.0.0"

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentTools": ("fastcode.retrieval.agent_tools", "AgentTools"),
    "AnswerGenerator": ("fastcode.query.answer", "AnswerGenerator"),
    "CodeIndexer": ("fastcode.indexing.indexer", "CodeIndexer"),
    "CodeParser": ("fastcode.indexing.parser", "CodeParser"),
    "FastCode": ("fastcode.main.fastcode", "FastCode"),
    "HybridRetriever": ("fastcode.retrieval.hybrid", "HybridRetriever"),
    "IRCodeUnit": ("fastcode.ir.types", "IRCodeUnit"),
    "IRDocument": ("fastcode.ir.types", "IRDocument"),
    "IREdge": ("fastcode.ir.types", "IREdge"),
    "IROccurrence": ("fastcode.ir.types", "IROccurrence"),
    "IRRelation": ("fastcode.ir.types", "IRRelation"),
    "IRSnapshot": ("fastcode.ir.types", "IRSnapshot"),
    "IRSymbol": ("fastcode.ir.types", "IRSymbol"),
    "IRUnitEmbedding": ("fastcode.ir.types", "IRUnitEmbedding"),
    "IRUnitSupport": ("fastcode.ir.types", "IRUnitSupport"),
    "IterativeAgent": ("fastcode.retrieval.iterative", "IterativeAgent"),
    "RepositoryLoader": ("fastcode.indexing.loader", "RepositoryLoader"),
    "RepositoryOverviewGenerator": (
        "fastcode.indexing.overview",
        "RepositoryOverviewGenerator",
    ),
    "RepositorySelector": ("fastcode.query.selector", "RepositorySelector"),
}

if TYPE_CHECKING:
    from .indexing.indexer import CodeIndexer
    from .indexing.loader import RepositoryLoader
    from .indexing.overview import RepositoryOverviewGenerator
    from .indexing.parser import CodeParser
    from .ir.types import (
        IRCodeUnit,
        IRDocument,
        IREdge,
        IROccurrence,
        IRRelation,
        IRSnapshot,
        IRSymbol,
        IRUnitEmbedding,
        IRUnitSupport,
    )
    from .main.fastcode import FastCode
    from .query.answer import AnswerGenerator
    from .query.selector import RepositorySelector
    from .retrieval.agent_tools import AgentTools
    from .retrieval.hybrid import HybridRetriever
    from .retrieval.iterative import IterativeAgent


def __getattr__(name: str) -> Any:
    """Load compatibility re-exports only when callers request them."""
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentTools",
    "AnswerGenerator",
    "CodeIndexer",
    "CodeParser",
    "FastCode",
    "HybridRetriever",
    "IRCodeUnit",
    "IRDocument",
    "IREdge",
    "IROccurrence",
    "IRRelation",
    "IRSnapshot",
    "IRSymbol",
    "IRUnitEmbedding",
    "IRUnitSupport",
    "IterativeAgent",
    "RepositoryLoader",
    "RepositoryOverviewGenerator",
    "RepositorySelector",
]
