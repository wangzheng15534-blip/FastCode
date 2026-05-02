"""Julia semantic resolver via helper-backed source analysis."""

from __future__ import annotations

from .base import ResolverSpec, SemanticCapability
from .graph_backed import GraphBackedSemanticResolver
from .helper_backed import HelperBackedSemanticResolver

_JULIA_SPEC = ResolverSpec(
    language="julia",
    capabilities=frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORTS,
            SemanticCapability.RESOLVE_TYPES,
        }
    ),
    cost_class="high",
    source_name="julia_resolver",
    extractor_name="julia_languageserver",
    frontend_kind="julia_languageserver",
    required_tools=("julia",),
)


class JuliaCompilerResolver(HelperBackedSemanticResolver):
    language = _JULIA_SPEC.language
    capabilities = _JULIA_SPEC.capabilities
    cost_class = _JULIA_SPEC.cost_class
    source_name = _JULIA_SPEC.source_name
    extractor_name = _JULIA_SPEC.extractor_name
    frontend_kind = _JULIA_SPEC.frontend_kind
    required_tools = _JULIA_SPEC.required_tools
    helper_filename = "julia_semantic_helper.py"
    helper_runtime = "python"
    helper_timeout_seconds = 90
    file_extensions = (".jl",)

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        super().__init__(fallback)
