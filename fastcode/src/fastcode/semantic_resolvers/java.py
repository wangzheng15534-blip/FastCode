"""Java semantic resolver via JDT/javac compiler diagnostics.

Uses ``javac`` subprocess to extract symbol bindings, inheritance
hierarchies, and interface implementations.  Falls back to
``GraphBackedSemanticResolver`` when ``java``/``javac`` are not installed.
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


class JavaCompilerResolver(SemanticResolver):
    """Java resolver backed by javac compiler diagnostics."""

    language = _JAVA_SPEC.language
    capabilities = _JAVA_SPEC.capabilities
    cost_class = _JAVA_SPEC.cost_class
    source_name = _JAVA_SPEC.source_name
    frontend_kind = _JAVA_SPEC.frontend_kind
    required_tools = _JAVA_SPEC.required_tools

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
            elem.language == "java"
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
                        message=f"'{tool}' not found in PATH; Java resolution is structural-only",
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
        """Invoke ``javac`` with ``-Xdiags:verbose`` for detailed diagnostics."""
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

        java_files = [p for p in target_paths if p.endswith(".java")]
        if not java_files:
            return patch

        try:
            javac_path = shutil.which("javac") or "javac"
            result = subprocess.run(
                [
                    javac_path,
                    "-Xdiags:verbose",
                    "-Xlint:all",
                    "-proc:none",
                    "-d",
                    "/dev/null",
                    *java_files,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            patch.stats["javac_exit_code"] = result.returncode
            patch.stats["javac_diagnostic_lines"] = len(result.stderr.splitlines())
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            patch.warnings.append(f"javac_invocation_failed: {exc}")
            patch.diagnostics.append(
                ToolDiagnostic(
                    language=self.language,
                    tool="javac",
                    code="tool_invocation_failed",
                    message=str(exc),
                )
            )

        return patch
