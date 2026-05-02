"""Zig semantic resolver via helper-backed source analysis."""

from __future__ import annotations

import shutil

from .base import ResolverSpec, SemanticCapability
from .graph_backed import GraphBackedSemanticResolver
from .helper_backed import HelperBackedSemanticResolver

_ZIG_SPEC = ResolverSpec(
    language="zig",
    capabilities=frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORTS,
            SemanticCapability.RESOLVE_TYPES,
        }
    ),
    cost_class="medium",
    source_name="zig_resolver",
    extractor_name="zig_zls",
    frontend_kind="zls_semantic",
    required_tools=("zig", "zls"),
)


class ZigCompilerResolver(HelperBackedSemanticResolver):
    language = _ZIG_SPEC.language
    capabilities = _ZIG_SPEC.capabilities
    cost_class = _ZIG_SPEC.cost_class
    source_name = _ZIG_SPEC.source_name
    extractor_name = _ZIG_SPEC.extractor_name
    frontend_kind = _ZIG_SPEC.frontend_kind
    required_tools = _ZIG_SPEC.required_tools
    helper_filename = "zig_semantic_helper.py"
    helper_runtime = "python"
    helper_timeout_seconds = 90
    file_extensions = (".zig",)

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        super().__init__(fallback)

    def _has_tools(self) -> bool:
        return all(shutil.which(tool) is not None for tool in self.required_tools)
