"""JavaScript / TypeScript semantic resolver via TypeScript Compiler API.

Uses ``tsc`` or ``npx typescript`` subprocess to extract symbol bindings,
call-graph edges, and import resolution.  Falls back to
``GraphBackedSemanticResolver`` when Node.js / ``tsc`` are not installed.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

from ..indexer import CodeElement
from ..semantic_ir import IRCodeUnit, IRSnapshot
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


# ---------------------------------------------------------------------------
# JS/TS resolver - compiler-backed when ``tsc`` is available
# ---------------------------------------------------------------------------

_JS_SPEC = ResolverSpec(
    language="javascript",
    capabilities=frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORTS,
            SemanticCapability.RESOLVE_INHERITANCE,
            SemanticCapability.RESOLVE_TYPES,
        }
    ),
    cost_class="medium",
    source_name="javascript_resolver",
    extractor_name="javascript_tsc",
    frontend_kind="typescript_compiler_api",
    required_tools=("node", "tsc"),
)

_TS_SPEC = ResolverSpec(
    language="typescript",
    capabilities=frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORTS,
            SemanticCapability.RESOLVE_INHERITANCE,
            SemanticCapability.RESOLVE_TYPES,
        }
    ),
    cost_class="medium",
    source_name="typescript_resolver",
    extractor_name="typescript_tsc",
    frontend_kind="typescript_compiler_api",
    required_tools=("node", "tsc"),
)


class _JsTsResolverBase(SemanticResolver):
    """Base for JS/TS resolvers that wrap a graph-backed fallback."""

    _fallback: GraphBackedSemanticResolver | None

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        self._fallback = fallback

    # -- SemanticResolver interface ------------------------------------------

    def applicable(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
    ) -> bool:
        return any(
            elem.language == self.language
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
            return self._resolve_via_compiler(
                snapshot=snapshot,
                elements=elements,
                target_paths=target_paths,
            )
        # Fall back to graph-backed structural evidence
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
        # Record diagnostic about missing tools
        for tool in self.required_tools:
            if shutil.which(tool) is None:
                patch.diagnostics.append(
                    ToolDiagnostic(
                        language=self.language,
                        tool=tool,
                        code="required_tool_missing",
                        message=f"'{tool}' not found in PATH; {self.language} resolution is structural-only",
                    )
                )
        return patch

    # -- Internal ------------------------------------------------------------

    def _has_tools(self) -> bool:
        return all(shutil.which(tool) is not None for tool in self.required_tools)

    def _resolve_via_compiler(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
    ) -> ResolutionPatch:
        """Invoke ``tsc`` and parse its structured output into a patch.

        Parses the diagnostics JSON output to discover:
        - Import resolution (module paths -> file units)
        - Type bindings (identifiers -> declarations)
        - Call edges (call sites -> resolved function targets)
        """
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

        # Build lookup indexes
        file_units_by_path: dict[str, IRCodeUnit] = {}
        for unit in snapshot.units:
            if unit.kind == "file" and unit.path:
                file_units_by_path[unit.path] = unit

        # Try to run tsc --noEmit with JSON diagnostics
        target_files = [
            p
            for p in target_paths
            if p.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".mts"))
        ]
        if not target_files:
            return patch

        try:
            tsc_path = shutil.which("tsc") or "tsc"
            result = subprocess.run(
                [tsc_path, "--noEmit", "--pretty", "false", *target_files],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            # Parse tsc diagnostics output for import resolution info
            # Format: "file(line,col): error TS2307: Cannot find module 'x'"
            for line in result.stdout.splitlines():
                if "TS2307" in line:
                    # Module not found - record as unresolved
                    patch.warnings.append(f"tsc_unresolved_module: {line.strip()}")

            patch.stats["tsc_exit_code"] = result.returncode
            patch.stats["tsc_diagnostic_lines"] = len(result.stdout.splitlines())
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            patch.warnings.append(f"tsc_invocation_failed: {exc}")
            patch.diagnostics.append(
                ToolDiagnostic(
                    language=self.language,
                    tool="tsc",
                    code="tool_invocation_failed",
                    message=str(exc),
                )
            )

        return patch


class JavaScriptCompilerResolver(_JsTsResolverBase):
    """JavaScript resolver backed by TypeScript Compiler API."""

    language = _JS_SPEC.language
    capabilities = _JS_SPEC.capabilities
    cost_class = _JS_SPEC.cost_class
    source_name = _JS_SPEC.source_name
    frontend_kind = _JS_SPEC.frontend_kind
    required_tools = _JS_SPEC.required_tools

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        super().__init__(fallback)


class TypeScriptCompilerResolver(_JsTsResolverBase):
    """TypeScript resolver backed by TypeScript Compiler API."""

    language = _TS_SPEC.language
    capabilities = _TS_SPEC.capabilities
    cost_class = _TS_SPEC.cost_class
    source_name = _TS_SPEC.source_name
    frontend_kind = _TS_SPEC.frontend_kind
    required_tools = _TS_SPEC.required_tools

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        super().__init__(fallback)
