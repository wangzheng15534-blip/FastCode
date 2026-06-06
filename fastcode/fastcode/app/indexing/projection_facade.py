"""
ProjectionFacade -- narrow API surface extracted from FastCode.

Wraps ProjectionService with 4 public methods.  build_projection acquires
the RuntimeState write lock; the other three are pure delegation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastcode.app.indexing.projection.service import ProjectionService
    from fastcode.runtime_support.runtime_state import RuntimeState


class ProjectionFacade:
    """Facade for projection-related operations.

    Delegates calls to ProjectionService.  ``build_projection`` holds the
    RuntimeState write lock; ``get_projection_layer``,
    ``get_projection_chunk``, and ``get_session_prefix`` are lock-free
    delegations.
    """

    def __init__(
        self,
        service: ProjectionService,
        state: RuntimeState,
    ) -> None:
        self._service = service
        self._state = state

    # ------------------------------------------------------------------
    # Lock-guarded method
    # ------------------------------------------------------------------

    def build_projection(
        self,
        scope_kind: str,
        snapshot_id: str | None = None,
        repo_name: str | None = None,
        ref_name: str | None = None,
        query: str | None = None,
        target_id: str | None = None,
        filters: dict[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        with self._state.write_lock():
            return self._service.build_projection(
                scope_kind=scope_kind,
                snapshot_id=snapshot_id,
                repo_name=repo_name,
                ref_name=ref_name,
                query=query,
                target_id=target_id,
                filters=filters,
                force=force,
            )

    # ------------------------------------------------------------------
    # Simple delegations
    # ------------------------------------------------------------------

    def get_projection_layer(self, projection_id: str, layer: str) -> dict[str, Any]:
        return self._service.get_projection_layer(projection_id, layer)

    def get_projection_chunk(self, projection_id: str, chunk_id: str) -> dict[str, Any]:
        return self._service.get_projection_chunk(projection_id, chunk_id)

    def get_session_prefix(self, snapshot_id: str) -> dict[str, Any]:
        """Return L0+L1 projection data for system prompt injection."""
        return self._service.get_session_prefix(snapshot_id)
