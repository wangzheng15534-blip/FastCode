"""Base types for semantic resolver plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..indexer import CodeElement
from ..semantic_ir import IRRelation, IRSnapshot, IRUnitSupport


# ---------------------------------------------------------------------------
# Semantic capability constants
# ---------------------------------------------------------------------------


class SemanticCapability:
    """Well-known capability identifiers for resolver advertisements.

    Resolvers declare which of these they can satisfy.  Relations carry
    ``pending_capabilities`` when none of the active resolvers could fulfil
    a given slot, keeping the system honest about uncertainty.
    """

    RESOLVE_CALLS = "resolve_calls"
    RESOLVE_IMPORTS = "resolve_imports"
    RESOLVE_IMPORT_ALIASES = "resolve_import_aliases"
    RESOLVE_TYPES = "resolve_types"
    RESOLVE_INHERITANCE = "resolve_inheritance"
    RESOLVE_BINDINGS = "resolve_bindings"
    EXPAND_MACROS = "expand_macros"
    RECOVER_QUALIFIED_NAMES = "recover_qualified_names"
    RESOLVE_INCLUDES = "resolve_includes"


# ---------------------------------------------------------------------------
# Resolution tier - distinguishes structural fallback from compiler facts
# ---------------------------------------------------------------------------


class ResolutionTier:
    """Resolution evidence quality tier.

    Attached to relation metadata so downstream consumers can distinguish
    graph-backed structural evidence from compiler-confirmed facts.
    """

    STRUCTURAL_FALLBACK = "structural_fallback"
    COMPILER_CONFIRMED = "compiler_confirmed"
    ANCHORED = "anchored"


# ---------------------------------------------------------------------------
# Typed resolver request (immutable input for resolvers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticResolutionRequest:
    """Immutable input for a semantic resolver invocation.

    Encapsulates everything a resolver needs so it never reaches into global
    state, stores, or the graph builder directly.
    """

    snapshot_id: str
    target_paths: frozenset[str]
    budget: str = "changed_files"
    repo_root: str = ""
    tool_context: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Resolver spec (frozen metadata)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Resolution patch (mutable output from resolvers)
# ---------------------------------------------------------------------------


@dataclass
class ResolutionPatch:
    """Patch emitted by a semantic resolver.

    The patch is applied onto an existing canonical snapshot. Resolvers do not
    create or mutate snapshots directly.
    """

    unit_metadata_updates: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    metadata_updates: dict[str, Any] = field(default_factory=dict)
    supports: list[IRUnitSupport] = field(default_factory=list)
    relations: list[IRRelation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: list[ToolDiagnostic] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    resolution_tier: str = ResolutionTier.STRUCTURAL_FALLBACK


class SemanticResolver(ABC):
    """Abstract semantic resolver interface."""

    language: str
    capabilities: frozenset[str]
    cost_class: str
    source_name: str = ""
    extractor_name: str = ""
    frontend_kind: str = "structural"
    required_tools: tuple[str, ...] = ()

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
        legacy_graph_builder: Any | None,
    ) -> ResolutionPatch:
        """Emit a patch against an existing canonical snapshot."""
