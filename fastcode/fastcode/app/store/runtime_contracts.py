"""Store-side capability contract for the SQL runtime.

Following the FCIS consumer-owns-the-port rule (mirrors zotero's
``zotero-app/src/catalog/port.rs``): the capability trait lives with the
use_flow consumer that needs it. The concrete effect_facility adapter
(``fastcode.infrastructure.storage.runtime.DBRuntime``) satisfies this
structurally and is injected at assembly time; it does not import this module.
"""

from __future__ import annotations

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
