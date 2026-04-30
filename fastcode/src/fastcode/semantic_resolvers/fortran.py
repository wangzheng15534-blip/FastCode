"""Fortran semantic resolver via fparser / fortls.

Uses ``fortls`` or fparser subprocess for modules, USE associations,
procedures, derived types, and type extension.  Falls back to
``GraphBackedSemanticResolver`` when ``fortls`` is not installed.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

from ..indexer import CodeElement
from ..semantic_ir import IRSnapshot
from .base import (
    ResolutionPatch,
    ResolutionTier,
    ResolverSpec,
    SemanticCapability,
    SemanticResolver,
    ToolDiagnostic,
)
from .graph_backed import GraphBackedSemanticResolver

logger = logging.getLogger(__name__)

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


class FortranCompilerResolver(SemanticResolver):
    """Fortran resolver backed by fortls / fparser."""

    language = _FORTRAN_SPEC.language
    capabilities = _FORTRAN_SPEC.capabilities
    cost_class = _FORTRAN_SPEC.cost_class
    source_name = _FORTRAN_SPEC.source_name
    frontend_kind = _FORTRAN_SPEC.frontend_kind
    required_tools = _FORTRAN_SPEC.required_tools

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        self._fallback = fallback

    def applicable(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
    ) -> bool:
        _FORTRAN_EXTS = (".f", ".for", ".f77", ".f90", ".f95", ".f03", ".f08")
        return any(
            elem.language == "fortran"
            and (elem.relative_path or elem.file_path) in target_paths
            for elem in elements
        )

    def resolve(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
        legacy_graph_builder: Any,
    ) -> ResolutionPatch:
        if self._has_tools():
            return self._resolve_via_compiler(snapshot, elements, target_paths)

        if self._fallback is not None:
            patch = self._fallback.resolve(
                snapshot=snapshot,
                elements=elements,
                target_paths=target_paths,
                legacy_graph_builder=legacy_graph_builder,
            )
        else:
            patch = ResolutionPatch(
                metadata_updates={
                    "semantic_resolver_runs": [
                        {
                            "language": self.language,
                            "source": self.source_name,
                            "frontend_kind": self.frontend_kind,
                            "fallback": True,
                        }
                    ]
                },
                resolution_tier=ResolutionTier.STRUCTURAL_FALLBACK,
            )
        for tool in self.required_tools:
            if shutil.which(tool) is None:
                patch.diagnostics.append(
                    ToolDiagnostic(
                        language=self.language,
                        tool=tool,
                        code="required_tool_missing",
                        message=f"'{tool}' not found in PATH; Fortran resolution is structural-only",
                    )
                )
        return patch

    def _has_tools(self) -> bool:
        return all(shutil.which(t) is not None for t in self.required_tools)

    def _resolve_via_compiler(
        self,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
    ) -> ResolutionPatch:
        """Invoke ``fortls`` for Fortran semantic analysis."""
        patch = ResolutionPatch(
            metadata_updates={
                "semantic_resolver_runs": [
                    {
                        "language": self.language,
                        "source": self.source_name,
                        "frontend_kind": self.frontend_kind,
                        "compiler_backed": True,
                    }
                ]
            },
            resolution_tier=ResolutionTier.COMPILER_CONFIRMED,
        )

        try:
            fortls_path = shutil.which("fortls") or "fortls"
            result = subprocess.run(
                [fortls_path, "--debug_diagnostics"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            patch.stats["fortls_exit_code"] = result.returncode
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            patch.warnings.append(f"fortls_invocation_failed: {exc}")
            patch.diagnostics.append(
                ToolDiagnostic(
                    language=self.language,
                    tool="fortls",
                    code="tool_invocation_failed",
                    message=str(exc),
                )
            )

        return patch
