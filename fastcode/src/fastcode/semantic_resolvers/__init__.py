"""Semantic resolver plugins for canonical IR upgrades."""

from .base import ResolutionPatch, SemanticResolver
from .c_family import CSemanticResolver, CppSemanticResolver
from .graph_backed import GraphBackedSemanticResolver
from .patching import apply_resolution_patch
from .python import (
    PYTHON_RESOLVER_EXTRACTOR,
    PYTHON_RESOLVER_SOURCE,
    PythonSemanticResolver,
)
from .registry import (
    SemanticResolverRegistry,
    build_default_semantic_resolver_registry,
)

__all__ = [
    "CSemanticResolver",
    "CppSemanticResolver",
    "GraphBackedSemanticResolver",
    "PYTHON_RESOLVER_EXTRACTOR",
    "PYTHON_RESOLVER_SOURCE",
    "PythonSemanticResolver",
    "ResolutionPatch",
    "SemanticResolver",
    "SemanticResolverRegistry",
    "apply_resolution_patch",
    "build_default_semantic_resolver_registry",
]
