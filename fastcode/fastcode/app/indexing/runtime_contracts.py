"""Indexing-side capability contract for the document graph runtime.

Following the FCIS consumer-owns-the-port rule (mirrors zotero's
``zotero-app/src/catalog/port.rs``): the capability trait lives with the
use_flow consumer that needs it. The concrete effect_facility adapter
(``fastcode.infrastructure.graph_runtime.ladybug.LadybugGraphRuntime``) satisfies
this structurally and is injected at assembly time; it does not import this module.
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

    def sync_nodes(
        self,
        *,
        nodes: Iterable[Any],
    ) -> bool:
        """Persist pre-built graph node records into the overlay."""
        ...
