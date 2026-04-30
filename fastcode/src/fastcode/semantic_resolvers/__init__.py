"""Semantic resolver plugins for canonical IR upgrades."""

from .base import (
    ResolutionPatch,
    ResolutionTier,
    ResolverSpec,
    SemanticCapability,
    SemanticResolutionRequest,
    SemanticResolver,
    ToolDiagnostic,
)
from .c_family import CppSemanticResolver, CSemanticResolver
from .graph_backed import GraphBackedSemanticResolver
from .language_graph import (
    CSharpSemanticResolver,
    FortranSemanticResolver,
    GoSemanticResolver,
    JavaScriptSemanticResolver,
    JavaSemanticResolver,
    JuliaSemanticResolver,
    RustSemanticResolver,
    TypeScriptSemanticResolver,
    ZigSemanticResolver,
)
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
    "PYTHON_RESOLVER_EXTRACTOR",
    "PYTHON_RESOLVER_SOURCE",
    "CSemanticResolver",
    "CSharpSemanticResolver",
    "CppSemanticResolver",
    "FortranSemanticResolver",
    "GoSemanticResolver",
    "GraphBackedSemanticResolver",
    "JavaScriptSemanticResolver",
    "JavaSemanticResolver",
    "JuliaSemanticResolver",
    "PythonSemanticResolver",
    "ResolutionPatch",
    "ResolutionTier",
    "ResolverSpec",
    "RustSemanticResolver",
    "SemanticCapability",
    "SemanticResolutionRequest",
    "SemanticResolver",
    "SemanticResolverRegistry",
    "ToolDiagnostic",
    "TypeScriptSemanticResolver",
    "ZigSemanticResolver",
    "apply_resolution_patch",
    "build_default_semantic_resolver_registry",
]
