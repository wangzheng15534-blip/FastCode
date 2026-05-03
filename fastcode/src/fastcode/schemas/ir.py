"""Compatibility exports for canonical IR types.

Canonical IR dataclasses live in ``fastcode.ir.types``. This module remains for
older imports and must not define a second set of IR classes.
"""

from fastcode.ir.types import (
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
    resolution_rank,
    resolution_to_confidence,
)

__all__ = [
    "IRAttachment",
    "IRCodeUnit",
    "IRDocument",
    "IREdge",
    "IROccurrence",
    "IRRelation",
    "IRSnapshot",
    "IRSymbol",
    "IRUnitEmbedding",
    "IRUnitSupport",
    "resolution_rank",
    "resolution_to_confidence",
]
