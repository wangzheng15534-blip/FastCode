"""Go semantic resolver via helper-backed `go/types` extraction."""

from __future__ import annotations

import shutil

from .base import ResolverSpec, SemanticCapability
from .graph_backed import GraphBackedSemanticResolver
from .helper_backed import HelperBackedSemanticResolver

_GO_SPEC = ResolverSpec(
    language="go",
    capabilities=frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORTS,
            SemanticCapability.RESOLVE_INHERITANCE,
            SemanticCapability.RESOLVE_TYPES,
        }
    ),
    cost_class="medium",
    source_name="go_resolver",
    extractor_name="go_packages",
    frontend_kind="go_packages",
    required_tools=("go",),
)


class GoCompilerResolver(HelperBackedSemanticResolver):
    language = _GO_SPEC.language
    capabilities = _GO_SPEC.capabilities
    cost_class = _GO_SPEC.cost_class
    source_name = _GO_SPEC.source_name
    extractor_name = _GO_SPEC.extractor_name
    frontend_kind = _GO_SPEC.frontend_kind
    required_tools = _GO_SPEC.required_tools
    helper_filename = "go_semantic_helper.go"
    helper_runtime = "go"
    helper_timeout_seconds = 120
    file_extensions = (".go",)

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        super().__init__(fallback)

    def _has_tools(self) -> bool:
        return all(shutil.which(tool) is not None for tool in self.required_tools)
