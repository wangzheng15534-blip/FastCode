"""Java semantic resolver via helper-backed source analysis."""

from __future__ import annotations

import shutil

from .base import ResolverSpec, SemanticCapability
from .graph_backed import GraphBackedSemanticResolver
from .helper_backed import HelperBackedSemanticResolver

_JAVA_SPEC = ResolverSpec(
    language="java",
    capabilities=frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORTS,
            SemanticCapability.RESOLVE_INHERITANCE,
            SemanticCapability.RESOLVE_TYPES,
        }
    ),
    cost_class="medium",
    source_name="java_resolver",
    extractor_name="java_javac",
    frontend_kind="javac_diagnostics",
    required_tools=("javac",),
)


class JavaCompilerResolver(HelperBackedSemanticResolver):
    language = _JAVA_SPEC.language
    capabilities = _JAVA_SPEC.capabilities
    cost_class = _JAVA_SPEC.cost_class
    source_name = _JAVA_SPEC.source_name
    extractor_name = _JAVA_SPEC.extractor_name
    frontend_kind = _JAVA_SPEC.frontend_kind
    required_tools = _JAVA_SPEC.required_tools
    helper_filename = "java_semantic_helper.py"
    helper_runtime = "python"
    helper_timeout_seconds = 120
    file_extensions = (".java",)

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        super().__init__(fallback)

    def _has_tools(self) -> bool:
        return all(shutil.which(tool) is not None for tool in self.required_tools)
