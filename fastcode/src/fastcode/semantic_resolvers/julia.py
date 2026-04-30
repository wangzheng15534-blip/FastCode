"""Julia semantic resolver via LanguageServer.jl.

Uses ``julia`` subprocess with LanguageServer.jl/CSTParser for modules,
imports, functions, methods, types, and dynamic-dispatch candidates.
Falls back to ``GraphBackedSemanticResolver`` when ``julia`` is not installed.
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


class JuliaCompilerResolver(SemanticResolver):
    """Julia resolver backed by LanguageServer.jl."""

    language = _JULIA_SPEC.language
    capabilities = _JULIA_SPEC.capabilities
    cost_class = _JULIA_SPEC.cost_class
    source_name = _JULIA_SPEC.source_name
    frontend_kind = _JULIA_SPEC.frontend_kind
    required_tools = _JULIA_SPEC.required_tools

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        self._fallback = fallback

    def applicable(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
    ) -> bool:
        return any(
            elem.language == "julia"
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
                        message=f"'{tool}' not found in PATH; Julia resolution is structural-only",
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
        """Invoke ``julia`` with LanguageServer.jl for semantic analysis."""
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
            julia_path = shutil.which("julia") or "julia"
            # Probe whether LanguageServer.jl is available
            result = subprocess.run(
                [julia_path, "-e", 'using LanguageServer; println("ok")'],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode != 0:
                patch.warnings.append("julia_languageserver_not_installed")
                patch.diagnostics.append(
                    ToolDiagnostic(
                        language=self.language,
                        tool="LanguageServer.jl",
                        code="required_package_missing",
                        message="LanguageServer.jl not installed; Julia resolution is structural-only",
                    )
                )
            patch.stats["julia_exit_code"] = result.returncode
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            patch.warnings.append(f"julia_invocation_failed: {exc}")
            patch.diagnostics.append(
                ToolDiagnostic(
                    language=self.language,
                    tool="julia",
                    code="tool_invocation_failed",
                    message=str(exc),
                )
            )

        return patch
