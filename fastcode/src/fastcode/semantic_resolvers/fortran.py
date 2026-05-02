"""Fortran semantic resolver via helper-backed source analysis."""

from __future__ import annotations

import shutil

from .base import ResolverSpec, SemanticCapability
from .graph_backed import GraphBackedSemanticResolver
from .helper_backed import HelperBackedSemanticResolver

_FORTRAN_SPEC = ResolverSpec(
    language="fortran",
    capabilities=frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORTS,
            SemanticCapability.RESOLVE_INHERITANCE,
            SemanticCapability.RESOLVE_TYPES,
        }
    ),
    cost_class="medium",
    source_name="fortran_resolver",
    extractor_name="fortran_fortls",
    frontend_kind="fortls_semantic",
    required_tools=("fortls",),
)


class FortranCompilerResolver(HelperBackedSemanticResolver):
    language = _FORTRAN_SPEC.language
    capabilities = _FORTRAN_SPEC.capabilities
    cost_class = _FORTRAN_SPEC.cost_class
    source_name = _FORTRAN_SPEC.source_name
    extractor_name = _FORTRAN_SPEC.extractor_name
    frontend_kind = _FORTRAN_SPEC.frontend_kind
    required_tools = _FORTRAN_SPEC.required_tools
    helper_filename = "fortran_semantic_helper.py"
    helper_runtime = "python"
    helper_timeout_seconds = 90
    file_extensions = (".f", ".for", ".f77", ".f90", ".f95", ".f03", ".f08")

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        super().__init__(fallback)

    def _has_tools(self) -> bool:
        return all(shutil.which(tool) is not None for tool in self.required_tools)
