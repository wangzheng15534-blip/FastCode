"""Canonical IR public exports.

The IR package is the lowest behavioral layer, so its package initializer stays
lazy. Importing ``fastcode.ir.types`` should not import graph builders or other
orchestration helpers.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_EXPORTS = {
    "CodeElement": ("fastcode.ir.element", "CodeElement"),
    "CodeElementMeta": ("fastcode.ir.element", "CodeElementMeta"),
    "IRAttachment": ("fastcode.ir.types", "IRAttachment"),
    "IRCodeUnit": ("fastcode.ir.types", "IRCodeUnit"),
    "IRDocument": ("fastcode.ir.types", "IRDocument"),
    "IREdge": ("fastcode.ir.types", "IREdge"),
    "IRGraphBuilder": ("fastcode.ir.graph", "IRGraphBuilder"),
    "IRGraphs": ("fastcode.ir.graph", "IRGraphs"),
    "IROccurrence": ("fastcode.ir.types", "IROccurrence"),
    "IRRelation": ("fastcode.ir.types", "IRRelation"),
    "IRSnapshot": ("fastcode.ir.types", "IRSnapshot"),
    "IRSymbol": ("fastcode.ir.types", "IRSymbol"),
    "IRUnitEmbedding": ("fastcode.ir.types", "IRUnitEmbedding"),
    "IRUnitSupport": ("fastcode.ir.types", "IRUnitSupport"),
    "ProjectionBuildResult": ("fastcode.ir.projection", "ProjectionBuildResult"),
    "ProjectionScope": ("fastcode.ir.projection", "ProjectionScope"),
    "merge_ir": ("fastcode.ir.merge", "merge_ir"),
    "validate_snapshot": ("fastcode.ir.validate", "validate_snapshot"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
