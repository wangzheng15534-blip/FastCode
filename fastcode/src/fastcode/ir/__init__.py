"""
Canonical IR types — the deepest truth layer.

All extraction produces IR types defined here. Everything else in the
codebase depends on this package; it depends on nothing below it.
"""

from .element import CodeElement, CodeElementMeta
from .graph import IRGraphBuilder, IRGraphs
from .merge import merge_ir
from .projection import ProjectionBuildResult, ProjectionScope
from .types import (
    IRAttachment,
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
from .validate import validate_snapshot

__all__ = [
    "CodeElement",
    "CodeElementMeta",
    "IRAttachment",
    "IRCodeUnit",
    "IRDocument",
    "IREdge",
    "IRGraphBuilder",
    "IRGraphs",
    "IROccurrence",
    "IRRelation",
    "IRSnapshot",
    "IRSymbol",
    "IRUnitEmbedding",
    "IRUnitSupport",
    "ProjectionBuildResult",
    "ProjectionScope",
    "merge_ir",
    "validate_snapshot",
]
