"""Semantic analysis exports.

Keep package import light so callers can import semantic entrypoints without
eagerly importing all resolver implementations and helper backends.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .resolvers import (
        apply_resolution_patch,
        build_default_semantic_resolver_registry,
    )
    from .symbol_index import SnapshotSymbolIndex, SnapshotSymbolMaps

_EXPORTS = {
    "SnapshotSymbolIndex": ("fastcode.semantic.symbol_index", "SnapshotSymbolIndex"),
    "SnapshotSymbolMaps": ("fastcode.semantic.symbol_index", "SnapshotSymbolMaps"),
    "apply_resolution_patch": (
        "fastcode.semantic.resolvers",
        "apply_resolution_patch",
    ),
    "build_default_semantic_resolver_registry": (
        "fastcode.semantic.resolvers",
        "build_default_semantic_resolver_registry",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SnapshotSymbolIndex",
    "SnapshotSymbolMaps",
    "apply_resolution_patch",
    "build_default_semantic_resolver_registry",
]
