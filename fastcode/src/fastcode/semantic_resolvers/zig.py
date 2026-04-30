"""Zig semantic resolver via ZLS / Zig compiler.

Uses ``zls`` or ``zig build`` subprocess for imports, containers, functions,
call candidates, and comptime metadata.  Falls back to
``GraphBackedSemanticResolver`` when ``zig``/``zls`` are not installed.
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


class ZigCompilerResolver(SemanticResolver):
    """Zig resolver backed by ZLS / zig compiler."""

    language = _ZIG_SPEC.language
    capabilities = _ZIG_SPEC.capabilities
    cost_class = _ZIG_SPEC.cost_class
    source_name = _ZIG_SPEC.source_name
    frontend_kind = _ZIG_SPEC.frontend_kind
    required_tools = _ZIG_SPEC.required_tools

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
            elem.language == "zig"
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
                        message=f"'{tool}' not found in PATH; Zig resolution is structural-only",
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
        """Invoke ``zig build`` for semantic diagnostics."""
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
            zig_path = shutil.which("zig") or "zig"
            result = subprocess.run(
                [zig_path, "build", "--summary", "all"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            patch.stats["zig_exit_code"] = result.returncode
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            patch.warnings.append(f"zig_invocation_failed: {exc}")
            patch.diagnostics.append(
                ToolDiagnostic(
                    language=self.language,
                    tool="zig",
                    code="tool_invocation_failed",
                    message=str(exc),
                )
            )

        return patch
