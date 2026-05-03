"""Code relationship graph exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .build import CodeGraphBuilder


def __getattr__(name: str) -> Any:
    if name == "CodeGraphBuilder":
        value = getattr(import_module("fastcode.graph.build"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CodeGraphBuilder",
]
