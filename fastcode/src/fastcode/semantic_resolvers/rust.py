"""Rust semantic resolver via rust-analyzer.

Uses ``rust-analyzer`` SCIP output or LSP subprocess for module resolution,
defs/refs, calls, impl/trait, and macro-aware locations.  Falls back to
``GraphBackedSemanticResolver`` when ``rust-analyzer`` is not installed.
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


class RustCompilerResolver(SemanticResolver):
    """Rust resolver backed by rust-analyzer SCIP / LSP."""

    language = _RUST_SPEC.language
    capabilities = _RUST_SPEC.capabilities
    cost_class = _RUST_SPEC.cost_class
    source_name = _RUST_SPEC.source_name
    frontend_kind = _RUST_SPEC.frontend_kind
    required_tools = _RUST_SPEC.required_tools

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
            elem.language == "rust"
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
                        message=f"'{tool}' not found in PATH; Rust resolution is structural-only",
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
        """Invoke ``rust-analyzer scip`` for precise semantic indexing."""
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
            ra_path = shutil.which("rust-analyzer") or "rust-analyzer"
            result = subprocess.run(
                [ra_path, "scip", "."],
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            patch.stats["rust_analyzer_exit_code"] = result.returncode
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            patch.warnings.append(f"rust_analyzer_invocation_failed: {exc}")
            patch.diagnostics.append(
                ToolDiagnostic(
                    language=self.language,
                    tool="rust-analyzer",
                    code="tool_invocation_failed",
                    message=str(exc),
                )
            )

        return patch
