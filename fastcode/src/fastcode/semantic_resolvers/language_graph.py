"""Language-specific graph-backed semantic resolvers.

These resolvers expose the compiler/LSP-backed resolver contract for languages
whose deep frontend integration is staged behind external tools. They use the
existing compatibility graphs as structural evidence today and record missing
tool diagnostics so callers can distinguish fallback facts from true compiler
facts.
"""

from __future__ import annotations

from .base import SemanticCapability
from .graph_backed import GraphBackedSemanticResolver


class JavaScriptSemanticResolver(GraphBackedSemanticResolver):
    language = "javascript"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_BINDINGS,
        }
    )
    cost_class = "medium"
    source_name = "javascript_resolver"
    extractor_name = "fastcode.semantic_resolvers.javascript"
    frontend_kind = "typescript_compiler_api_fallback"
    required_tools = ("node",)


class TypeScriptSemanticResolver(GraphBackedSemanticResolver):
    language = "typescript"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_TYPES,
            SemanticCapability.RESOLVE_BINDINGS,
        }
    )
    cost_class = "medium"
    source_name = "typescript_resolver"
    extractor_name = "fastcode.semantic_resolvers.typescript"
    frontend_kind = "typescript_compiler_api_fallback"
    required_tools = ("node", "tsc")


class JavaSemanticResolver(GraphBackedSemanticResolver):
    language = "java"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_INHERITANCE,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_TYPES,
        }
    )
    cost_class = "high"
    source_name = "java_resolver"
    extractor_name = "fastcode.semantic_resolvers.java"
    frontend_kind = "jdt_fallback"
    required_tools = ("java",)


class GoSemanticResolver(GraphBackedSemanticResolver):
    language = "go"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_TYPES,
            SemanticCapability.RESOLVE_BINDINGS,
        }
    )
    cost_class = "medium"
    source_name = "go_resolver"
    extractor_name = "fastcode.semantic_resolvers.go"
    frontend_kind = "go_packages_fallback"
    required_tools = ("go",)


class RustSemanticResolver(GraphBackedSemanticResolver):
    language = "rust"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_INHERITANCE,
            SemanticCapability.RESOLVE_TYPES,
            SemanticCapability.RESOLVE_BINDINGS,
            SemanticCapability.EXPAND_MACROS,
        }
    )
    cost_class = "high"
    source_name = "rust_resolver"
    extractor_name = "fastcode.semantic_resolvers.rust"
    frontend_kind = "rust_analyzer_fallback"
    required_tools = ("rust-analyzer",)


class CSharpSemanticResolver(GraphBackedSemanticResolver):
    language = "csharp"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_INHERITANCE,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_TYPES,
        }
    )
    cost_class = "high"
    source_name = "csharp_resolver"
    extractor_name = "fastcode.semantic_resolvers.csharp"
    frontend_kind = "roslyn_fallback"
    required_tools = ("dotnet",)


class ZigSemanticResolver(GraphBackedSemanticResolver):
    language = "zig"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_TYPES,
            SemanticCapability.RESOLVE_BINDINGS,
        }
    )
    cost_class = "high"
    source_name = "zig_resolver"
    extractor_name = "fastcode.semantic_resolvers.zig"
    frontend_kind = "zls_fallback"
    required_tools = ("zig", "zls")


class FortranSemanticResolver(GraphBackedSemanticResolver):
    language = "fortran"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_TYPES,
            SemanticCapability.RESOLVE_INHERITANCE,
        }
    )
    cost_class = "high"
    source_name = "fortran_resolver"
    extractor_name = "fastcode.semantic_resolvers.fortran"
    frontend_kind = "fortls_fallback"
    required_tools = ("fortls",)


class JuliaSemanticResolver(GraphBackedSemanticResolver):
    language = "julia"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_TYPES,
            SemanticCapability.RESOLVE_BINDINGS,
        }
    )
    cost_class = "high"
    source_name = "julia_resolver"
    extractor_name = "fastcode.semantic_resolvers.julia"
    frontend_kind = "julia_language_server_fallback"
    required_tools = ("julia",)
