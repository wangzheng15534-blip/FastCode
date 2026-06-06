"""Frozen graph-domain contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


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


class ModuleImportResolver(Protocol):
    """Graph-domain contract for resolving an import to an internal file id."""

    def resolve_import(
        self,
        current_module_path: str,
        import_name: str,
        level: int,
        is_package: bool = False,
    ) -> str | None: ...


class SymbolNameResolver(Protocol):
    """Graph-domain contract for resolving a symbol to an internal element id."""

    def resolve_symbol(
        self,
        symbol_name: str,
        current_file_id: str,
        imports: list[dict[str, Any]],
    ) -> str | None: ...
