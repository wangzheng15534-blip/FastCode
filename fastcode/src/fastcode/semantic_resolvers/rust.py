"""Rust semantic resolver via helper-backed source analysis."""

from __future__ import annotations

from .base import ResolverSpec, SemanticCapability
from .graph_backed import GraphBackedSemanticResolver
from .helper_backed import HelperBackedSemanticResolver

_RUST_SPEC = ResolverSpec(
    language="rust",
    capabilities=frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORTS,
            SemanticCapability.RESOLVE_INHERITANCE,
            SemanticCapability.RESOLVE_TYPES,
            SemanticCapability.EXPAND_MACROS,
        }
    ),
    cost_class="high",
    source_name="rust_resolver",
    extractor_name="rust_analyzer",
    frontend_kind="rust_analyzer_scip",
    required_tools=("rust-analyzer", "cargo"),
)


class RustCompilerResolver(HelperBackedSemanticResolver):
    language = _RUST_SPEC.language
    capabilities = _RUST_SPEC.capabilities
    cost_class = _RUST_SPEC.cost_class
    source_name = _RUST_SPEC.source_name
    extractor_name = _RUST_SPEC.extractor_name
    frontend_kind = _RUST_SPEC.frontend_kind
    required_tools = _RUST_SPEC.required_tools
    helper_filename = "rust_semantic_helper.py"
    helper_runtime = "python"
    helper_timeout_seconds = 120
    file_extensions = (".rs",)

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        super().__init__(fallback)
