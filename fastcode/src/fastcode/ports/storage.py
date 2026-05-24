"""Low-level storage runtime capability ports."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import Any, Protocol


class StoreDatabaseRuntime(Protocol):
    """SQL runtime capability shared by store code and concrete DB adapters."""

    backend: str

    def connect(self) -> AbstractContextManager[Any]:
        """Open a backend connection context."""
        ...

    def execute(self, conn: Any, sql: str, params: tuple[Any, ...] = ()) -> Any:
        """Execute a single SQL statement through the backend adapter."""
        ...

    def executemany(
        self, conn: Any, sql: str, params_seq: list[tuple[Any, ...]]
    ) -> Any:
        """Execute a batch SQL statement through the backend adapter."""
        ...

    def begin_write(self, conn: Any) -> None:
        """Start a write transaction when the backend requires an explicit one."""
        ...

    def supports_pgvector_adapter(self) -> bool:
        """Return True when native pgvector binding is available."""
        ...


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
