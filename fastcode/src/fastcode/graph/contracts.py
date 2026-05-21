"""Frozen graph-domain contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _empty_metadata() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class GraphBuildOptions:
    """Options for graph construction flows."""

    build_call_graph: bool = True
    build_dependency_graph: bool = True
    build_inheritance_graph: bool = True
    max_depth: int = 5
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)
