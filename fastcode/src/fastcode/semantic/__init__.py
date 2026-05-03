"""Semantic analysis: symbol indexing and language-specific resolvers."""

from .resolvers import (
    apply_resolution_patch,
    build_default_semantic_resolver_registry,
)
from .symbol_index import SnapshotSymbolIndex, SnapshotSymbolMaps

__all__ = [
    "SnapshotSymbolIndex",
    "SnapshotSymbolMaps",
    "apply_resolution_patch",
    "build_default_semantic_resolver_registry",
]
