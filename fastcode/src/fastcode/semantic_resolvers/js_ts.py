"""JavaScript / TypeScript semantic resolvers via TypeScript Compiler API."""

from __future__ import annotations

import shutil

from .base import ResolverSpec, SemanticCapability
from .graph_backed import GraphBackedSemanticResolver
from .helper_backed import HelperBackedSemanticResolver

_COMMON_CAPABILITIES = frozenset(
    {
        SemanticCapability.RESOLVE_CALLS,
        SemanticCapability.RESOLVE_IMPORTS,
        SemanticCapability.RESOLVE_INHERITANCE,
        SemanticCapability.RESOLVE_TYPES,
    }
)

_JS_SPEC = ResolverSpec(
    language="javascript",
    capabilities=_COMMON_CAPABILITIES,
    cost_class="medium",
    source_name="javascript_resolver",
    extractor_name="javascript_tsc",
    frontend_kind="typescript_compiler_api",
    required_tools=("node", "tsc"),
)

_TS_SPEC = ResolverSpec(
    language="typescript",
    capabilities=_COMMON_CAPABILITIES,
    cost_class="medium",
    source_name="typescript_resolver",
    extractor_name="typescript_tsc",
    frontend_kind="typescript_compiler_api",
    required_tools=("node", "tsc"),
)


class _JsTsResolverBase(HelperBackedSemanticResolver):
    helper_filename = "ts_semantic_helper.js"
    helper_runtime = "node"
    helper_timeout_seconds = 90
    file_extensions = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".mts", ".cjs", ".cts")

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        super().__init__(fallback)

    def _has_tools(self) -> bool:
        return all(shutil.which(tool) is not None for tool in self.required_tools)


class JavaScriptCompilerResolver(_JsTsResolverBase):
    language = _JS_SPEC.language
    capabilities = _JS_SPEC.capabilities
    cost_class = _JS_SPEC.cost_class
    source_name = _JS_SPEC.source_name
    extractor_name = _JS_SPEC.extractor_name
    frontend_kind = _JS_SPEC.frontend_kind
    required_tools = _JS_SPEC.required_tools


class TypeScriptCompilerResolver(_JsTsResolverBase):
    language = _TS_SPEC.language
    capabilities = _TS_SPEC.capabilities
    cost_class = _TS_SPEC.cost_class
    source_name = _TS_SPEC.source_name
    extractor_name = _TS_SPEC.extractor_name
    frontend_kind = _TS_SPEC.frontend_kind
    required_tools = _TS_SPEC.required_tools
