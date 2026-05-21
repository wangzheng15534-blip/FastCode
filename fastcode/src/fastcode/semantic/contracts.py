"""Frozen semantic-domain contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SemanticPatchSummary:
    """Summary of a semantic resolver patch application."""

    resolver_name: str
    updated_symbols: int = 0
    updated_relations: int = 0
    diagnostics: tuple[dict[str, Any], ...] = field(default_factory=tuple)
