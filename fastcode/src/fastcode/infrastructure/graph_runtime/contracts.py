"""Document graph runtime capability contracts.

Generic facility trait owned by the graph_runtime effect_facility unit.
Moved from axis_surface (ports) because this is a generic infrastructure
capability, not a semantic business trait.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol


class DocumentGraphRuntime(Protocol):
    """Optional document graph overlay capability used by indexing flows."""

    enabled: bool

    def sync_docs(
        self,
        *,
        chunks: Iterable[dict[str, Any]],
        mentions: Iterable[dict[str, Any]],
    ) -> bool:
        """Persist document chunks and mentions into the graph overlay."""
        ...
