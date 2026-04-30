"""Python semantic resolver backed by compatibility graph outputs."""

from __future__ import annotations

from .graph_backed import GraphBackedSemanticResolver

PYTHON_RESOLVER_SOURCE = "python_resolver"
PYTHON_RESOLVER_EXTRACTOR = "fastcode.semantic_resolvers.python"


class PythonSemanticResolver(GraphBackedSemanticResolver):
    language = "python"
    capabilities = frozenset(
        {
            "resolve_calls",
            "resolve_inheritance",
            "resolve_import_aliases",
            "resolve_bindings",
        }
    )
    cost_class = "medium"
    source_name = PYTHON_RESOLVER_SOURCE
    extractor_name = PYTHON_RESOLVER_EXTRACTOR
