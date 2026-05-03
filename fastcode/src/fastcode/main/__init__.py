"""main/ sub-package — composition root (FastCode class + CLI)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .fastcode import FastCode


def __getattr__(name: str) -> Any:
    """Load composition-root exports only when callers request them."""
    if name == "FastCode":
        value = getattr(importlib.import_module("fastcode.main.fastcode"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FastCode"]
