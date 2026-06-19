"""Pure execution-contract for semantic helper operations.

This module lives in meaning_core ``semantic`` and carries ONLY pure types:
the execution/cache-identity dataclasses plus the ``SemanticHelperOps``
Protocol that the concrete effect_tool implementation satisfies
structurally. It must not import effect_tool and must not perform any I/O.

The concrete mechanism (subprocess invocation, filesystem JSON cache) lives in
``fastcode.infrastructure.execution.helper_operations`` (effect_tool) and is
injected at the composition root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from fastcode.semantic.resolution import ToolDiagnostic


# Dependency-inversion hook: the concrete ``SemanticHelperOperations``
# (effect_tool) cannot be imported by this meaning_core module. The composition
# root (entry_frame / app assembly) registers a factory here; the resolver uses
# it to build a default adapter when one is not explicitly injected. This keeps
# meaning_core free of any effect_tool import edge while still allowing
# stand-alone resolver construction in tests and the registry builder.
_default_helper_ops_factory: Callable[[Any | None], SemanticHelperOps] | None = None


def set_default_helper_ops_factory(
    factory: Callable[[Any | None], SemanticHelperOps] | None,
) -> None:
    """Register the concrete effect_tool helper-ops factory (composition root)."""
    global _default_helper_ops_factory
    _default_helper_ops_factory = factory


def get_default_helper_ops_factory() -> (
    Callable[[Any | None], SemanticHelperOps] | None
):
    return _default_helper_ops_factory


def _empty_payload() -> dict[str, Any]:
    return {}


def _empty_stats() -> dict[str, Any]:
    return {}


def _empty_warnings() -> list[str]:
    return []


def _empty_diagnostics() -> list[ToolDiagnostic]:
    return []


@dataclass(frozen=True)
class SemanticHelperSpec:
    """Execution/cache identity for one helper-backed resolver."""

    cache_version: str
    language: str
    source_name: str
    frontend_kind: str
    extractor_name: str
    required_tools: tuple[str, ...]
    helper_filename: str
    helper_runtime: str
    helper_timeout_seconds: int
    file_extensions: tuple[str, ...]


@dataclass(frozen=True)
class SemanticHelperCacheEntry:
    """Concrete cache entry for a helper invocation identity."""

    key: str
    path: str
    identity: dict[str, Any]


@dataclass(frozen=True)
class SemanticHelperInvocation:
    """Result of invoking or decoding a helper command."""

    payload: dict[str, Any] = field(default_factory=_empty_payload)
    stats: dict[str, Any] = field(default_factory=_empty_stats)
    warnings: list[str] = field(default_factory=_empty_warnings)
    diagnostics: list[ToolDiagnostic] = field(default_factory=_empty_diagnostics)


class SemanticHelperOps(Protocol):
    """Pure meaning_core surface for helper execution.

    Implemented structurally by the effect_tool ``SemanticHelperOperations``
    class; wired into the resolver at the composition root.
    """

    def resolve_repo_root(self, repo_root: str | None) -> str: ...

    def has_tools(self, spec: SemanticHelperSpec) -> bool: ...

    def missing_tool_diagnostics(
        self, spec: SemanticHelperSpec
    ) -> list[ToolDiagnostic]: ...

    def target_files(
        self,
        target_paths: set[str],
        *,
        repo_root: str,
        spec: SemanticHelperSpec,
    ) -> list[str]: ...

    def helper_path(self, spec: SemanticHelperSpec) -> Path: ...

    def helper_command(
        self, spec: SemanticHelperSpec, helper_files: list[str]
    ) -> list[str]: ...

    def run_helper(
        self,
        spec: SemanticHelperSpec,
        helper_files: list[str],
        *,
        repo_root: str,
    ) -> Any:
        """Run a helper subprocess.

        Returns the effect_tool's raw result object (its diagnostics are raw
        dicts, not ``ToolDiagnostic`` — this meaning_core layer cannot name the
        effect_tool type without an import edge). The resolver wraps it into a
        ``SemanticHelperInvocation`` via ``_invocation_from_raw``.
        """
        ...

    def cache_entry(
        self,
        spec: SemanticHelperSpec,
        helper_files: list[str],
        *,
        repo_root: str,
    ) -> SemanticHelperCacheEntry: ...

    def load_cache(
        self, entry: SemanticHelperCacheEntry
    ) -> dict[str, Any] | None: ...

    def save_cache(
        self, entry: SemanticHelperCacheEntry, helper_payload: dict[str, Any]
    ) -> None: ...
