"""Semantic resolver domain types and patch application contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, cast

from fastcode.ir.element import CodeElement
from fastcode.ir.types import IRRelation, IRSnapshot, IRUnitSupport
from fastcode.semantic.contracts import SemanticGraphContext


def _empty_tool_context() -> dict[str, Any]:
    return {}


def _empty_unit_metadata_updates() -> dict[str, dict[str, Any]]:
    return {}


def _empty_metadata_updates() -> dict[str, Any]:
    return {}


def _empty_supports() -> list[IRUnitSupport]:
    return []


def _empty_relations() -> list[IRRelation]:
    return []


def _empty_warnings() -> list[str]:
    return []


def _empty_diagnostics() -> list[ToolDiagnostic]:
    return []


def _empty_stats() -> dict[str, Any]:
    return {}


class SemanticCapability:
    """Well-known capability identifiers for resolver advertisements."""

    RESOLVE_CALLS = "resolve_calls"
    RESOLVE_IMPORTS = "resolve_imports"
    RESOLVE_IMPORT_ALIASES = "resolve_import_aliases"
    RESOLVE_TYPES = "resolve_types"
    RESOLVE_INHERITANCE = "resolve_inheritance"
    RESOLVE_BINDINGS = "resolve_bindings"
    EXPAND_MACROS = "expand_macros"
    RECOVER_QUALIFIED_NAMES = "recover_qualified_names"
    RESOLVE_INCLUDES = "resolve_includes"


class ResolutionTier:
    """Resolution evidence quality tier."""

    STRUCTURAL_FALLBACK = "structural_fallback"
    COMPILER_CONFIRMED = "compiler_confirmed"
    ANCHORED = "anchored"


@dataclass(frozen=True)
class SemanticResolutionRequest:
    """Immutable input for a semantic resolver invocation."""

    snapshot_id: str
    target_paths: frozenset[str]
    budget: str = "changed_files"
    repo_root: str = ""
    tool_context: dict[str, Any] = field(default_factory=_empty_tool_context)


@dataclass
class ToolDiagnostic:
    """Diagnostic emitted when a language frontend cannot provide full semantics."""

    language: str
    tool: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "language": self.language,
            "tool": self.tool,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class ResolverSpec:
    """Public capability metadata for a semantic resolver plugin."""

    language: str
    capabilities: frozenset[str]
    cost_class: str
    source_name: str
    extractor_name: str
    frontend_kind: str
    required_tools: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "capabilities": sorted(self.capabilities),
            "cost_class": self.cost_class,
            "source_name": self.source_name,
            "extractor_name": self.extractor_name,
            "frontend_kind": self.frontend_kind,
            "required_tools": list(self.required_tools),
        }


@dataclass
class ResolutionPatch:
    """Patch emitted by a semantic resolver.

    The patch is applied onto an existing canonical snapshot. Resolvers do not
    create or mutate snapshots directly.
    """

    unit_metadata_updates: dict[str, dict[str, Any]] = field(
        default_factory=_empty_unit_metadata_updates
    )
    metadata_updates: dict[str, Any] = field(default_factory=_empty_metadata_updates)
    supports: list[IRUnitSupport] = field(default_factory=_empty_supports)
    relations: list[IRRelation] = field(default_factory=_empty_relations)
    warnings: list[str] = field(default_factory=_empty_warnings)
    diagnostics: list[ToolDiagnostic] = field(default_factory=_empty_diagnostics)
    stats: dict[str, Any] = field(default_factory=_empty_stats)
    resolution_tier: str = ResolutionTier.STRUCTURAL_FALLBACK


class SemanticResolver(ABC):
    """Domain polymorphism interface for semantic resolver implementations."""

    language: str
    capabilities: frozenset[str]
    cost_class: str
    source_name: str = ""
    extractor_name: str = ""
    frontend_kind: str = "structural"
    required_tools: tuple[str, ...] = ()

    def set_tool_runtime(self, tool_runtime: Any | None) -> None:
        """Inject shell-side tool lookup capability."""
        self._tool_runtime = tool_runtime

    def find_executable(self, executable: str) -> str | None:
        """Resolve a required tool through the injected runtime capability."""
        tool_runtime = getattr(self, "_tool_runtime", None)
        find_executable = getattr(tool_runtime, "find_executable", None)
        if not callable(find_executable):
            return None
        return cast(str | None, find_executable(executable))

    @property
    def spec(self) -> ResolverSpec:
        """Return stable metadata used by capability routing and diagnostics."""
        return ResolverSpec(
            language=self.language,
            capabilities=self.capabilities,
            cost_class=self.cost_class,
            source_name=self.source_name or f"{self.language}_resolver",
            extractor_name=self.extractor_name or self.__class__.__module__,
            frontend_kind=self.frontend_kind,
            required_tools=self.required_tools,
        )

    def applicable(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
    ) -> bool:
        """Return True when the resolver should run for this batch."""
        del snapshot
        return any(
            elem.language == self.language
            and (elem.relative_path or elem.file_path) in target_paths
            for elem in elements
        )

    @abstractmethod
    def resolve(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
        graph_context: SemanticGraphContext | None,
    ) -> ResolutionPatch:
        """Emit a patch against an existing canonical snapshot."""
