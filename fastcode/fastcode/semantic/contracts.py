"""Frozen semantic-domain contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, overload


class SemanticGraph(Protocol):
    """Graph view consumed by structural semantic resolvers."""

    @overload
    def edges(self, data: Literal[False] = False) -> Iterable[tuple[Any, Any]]: ...

    @overload
    def edges(
        self, data: Literal[True]
    ) -> Iterable[tuple[Any, Any, Mapping[str, Any]]]: ...

    def edges(
        self,
        data: bool = False,
    ) -> Iterable[tuple[Any, Any] | tuple[Any, Any, Mapping[str, Any]]]: ...


class SemanticGraphContext(Protocol):
    """Graph context available to graph-backed semantic resolvers."""

    dependency_graph: SemanticGraph
    inheritance_graph: SemanticGraph
    call_graph: SemanticGraph


@dataclass(frozen=True)
class SemanticPatchSummary:
    """Summary of a semantic resolver patch application."""

    resolver_name: str
    updated_symbols: int = 0
    updated_relations: int = 0
    diagnostics: tuple[dict[str, Any], ...] = field(default_factory=tuple)
