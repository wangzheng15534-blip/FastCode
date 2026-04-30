"""Base types for semantic resolver plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..indexer import CodeElement
from ..semantic_ir import IRRelation, IRSnapshot, IRUnitSupport


def _empty_dict() -> dict[str, Any]:
    return {}


def _empty_list() -> list[Any]:
    return []


@dataclass(frozen=True)
class ResolutionPatch:
    """Patch emitted by a semantic resolver.

    The patch is applied onto an existing canonical snapshot. Resolvers do not
    create or mutate snapshots directly.
    """

    unit_metadata_updates: dict[str, dict[str, Any]] = field(default_factory=_empty_dict)
    metadata_updates: dict[str, Any] = field(default_factory=_empty_dict)
    supports: list[IRUnitSupport] = field(default_factory=_empty_list)
    relations: list[IRRelation] = field(default_factory=_empty_list)
    warnings: list[str] = field(default_factory=_empty_list)
    stats: dict[str, Any] = field(default_factory=_empty_dict)


class SemanticResolver(ABC):
    """Abstract semantic resolver interface."""

    language: str
    capabilities: frozenset[str]
    cost_class: str

    @abstractmethod
    def applicable(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
    ) -> bool:
        """Return True when the resolver should run for this batch."""

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
