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
from .csharp import CSharpCompilerResolver
from .fortran import FortranCompilerResolver
from .go import GoCompilerResolver
from .graph_backed import GraphBackedSemanticResolver
from .helper_backed import HelperBackedSemanticResolver
from .java import JavaCompilerResolver
from .js_ts import JavaScriptCompilerResolver, TypeScriptCompilerResolver
from .julia import JuliaCompilerResolver
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
from .rust import RustCompilerResolver
from .zig import ZigCompilerResolver

__all__ = [
    "PYTHON_RESOLVER_EXTRACTOR",
    "PYTHON_RESOLVER_SOURCE",
    "CSemanticResolver",
    "CSharpCompilerResolver",
    "CSharpSemanticResolver",
    "CppSemanticResolver",
    "FortranCompilerResolver",
    "FortranSemanticResolver",
    "GoCompilerResolver",
    "GoSemanticResolver",
    "GraphBackedSemanticResolver",
    "HelperBackedSemanticResolver",
    "JavaCompilerResolver",
    "JavaScriptCompilerResolver",
    "JavaScriptSemanticResolver",
    "JavaSemanticResolver",
    "JuliaCompilerResolver",
    "JuliaSemanticResolver",
    "PythonSemanticResolver",
    "ResolutionPatch",
    "ResolutionTier",
    "ResolverSpec",
    "RustCompilerResolver",
    "RustSemanticResolver",
    "SemanticCapability",
    "SemanticResolutionRequest",
    "SemanticResolver",
    "SemanticResolverRegistry",
    "ToolDiagnostic",
    "TypeScriptCompilerResolver",
    "TypeScriptSemanticResolver",
    "ZigCompilerResolver",
    "ZigSemanticResolver",
    "apply_resolution_patch",
    "build_default_semantic_resolver_registry",
]
