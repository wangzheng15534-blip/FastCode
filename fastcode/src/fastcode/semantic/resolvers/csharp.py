"""C# semantic resolver via helper-backed source analysis."""

from __future__ import annotations

from .base import ResolverSpec, SemanticCapability
from .graph_backed import GraphBackedSemanticResolver
from .helper_backed import HelperBackedSemanticResolver

_CSHARP_SPEC = ResolverSpec(
    language="csharp",
    capabilities=frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORTS,
            SemanticCapability.RESOLVE_INHERITANCE,
            SemanticCapability.RESOLVE_TYPES,
        }
    ),
    cost_class="medium",
    source_name="csharp_resolver",
    extractor_name="csharp_roslyn",
    frontend_kind="roslyn_diagnostics",
    required_tools=("dotnet",),
)


class CSharpCompilerResolver(HelperBackedSemanticResolver):
    language = _CSHARP_SPEC.language
    capabilities = _CSHARP_SPEC.capabilities
    cost_class = _CSHARP_SPEC.cost_class
    source_name = _CSHARP_SPEC.source_name
    extractor_name = _CSHARP_SPEC.extractor_name
    frontend_kind = _CSHARP_SPEC.frontend_kind
    required_tools = _CSHARP_SPEC.required_tools
    helper_filename = "csharp_semantic_helper.py"
    helper_runtime = "python"
    helper_timeout_seconds = 120
    file_extensions = (".cs",)

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        super().__init__(fallback)
