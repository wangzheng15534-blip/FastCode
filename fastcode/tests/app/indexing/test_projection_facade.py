"""Tests for ProjectionFacade -- extracted projection methods."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from fastcode.app.indexing.projection_facade import ProjectionFacade
from fastcode.main.runtime_state import RuntimeState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FacadeHarness:
    """Test harness that holds the facade and its mock dependencies."""

    def __init__(self) -> None:
        self.projection_service = MagicMock()
        self.state = RuntimeState()
        self.facade = ProjectionFacade(
            service=self.projection_service,
            state=self.state,
        )


# ---------------------------------------------------------------------------
# Simple delegation tests (no lock needed)
# ---------------------------------------------------------------------------


class TestGetProjectionLayer:
    def test_delegates_to_service(self) -> None:
        h = _FacadeHarness()
        h.projection_service.get_projection_layer.return_value = {
            "projection_id": "p1",
            "layer": "l0",
            "data": {},
        }
        result = h.facade.get_projection_layer("p1", "l0")
        assert result["projection_id"] == "p1"
        h.projection_service.get_projection_layer.assert_called_once_with("p1", "l0")

    def test_passes_through_service_error(self) -> None:
        h = _FacadeHarness()
        h.projection_service.get_projection_layer.side_effect = RuntimeError(
            "not found"
        )
        with pytest.raises(RuntimeError, match="not found"):
            h.facade.get_projection_layer("p1", "l0")


class TestGetProjectionChunk:
    def test_delegates_to_service(self) -> None:
        h = _FacadeHarness()
        h.projection_service.get_projection_chunk.return_value = {
            "chunk_id": "c1",
            "content": "data",
        }
        result = h.facade.get_projection_chunk("p1", "c1")
        assert result["chunk_id"] == "c1"
        h.projection_service.get_projection_chunk.assert_called_once_with("p1", "c1")

    def test_passes_through_service_error(self) -> None:
        h = _FacadeHarness()
        h.projection_service.get_projection_chunk.side_effect = RuntimeError(
            "missing"
        )
        with pytest.raises(RuntimeError, match="missing"):
            h.facade.get_projection_chunk("p1", "c1")


class TestGetSessionPrefix:
    def test_delegates_to_service(self) -> None:
        h = _FacadeHarness()
        h.projection_service.get_session_prefix.return_value = {
            "snapshot_id": "snap:repo:abc",
            "projection_id": "p1",
            "l0": {},
            "l1": {},
        }
        result = h.facade.get_session_prefix("snap:repo:abc")
        assert result["snapshot_id"] == "snap:repo:abc"
        h.projection_service.get_session_prefix.assert_called_once_with(
            "snap:repo:abc"
        )

    def test_passes_through_service_error(self) -> None:
        h = _FacadeHarness()
        h.projection_service.get_session_prefix.side_effect = RuntimeError(
            "no snapshot"
        )
        with pytest.raises(RuntimeError, match="no snapshot"):
            h.facade.get_session_prefix("snap:repo:abc")

    def test_is_callable(self) -> None:
        """get_session_prefix must be usable as a bare callable (for QueryPipeline wiring)."""
        h = _FacadeHarness()
        h.projection_service.get_session_prefix.return_value = {"snapshot_id": "s"}
        fn = h.facade.get_session_prefix
        result = fn("s")
        assert result["snapshot_id"] == "s"


# ---------------------------------------------------------------------------
# build_projection (lock-guarded)
# ---------------------------------------------------------------------------


class TestBuildProjection:
    def test_delegates_to_service(self) -> None:
        h = _FacadeHarness()
        h.projection_service.build_projection.return_value = {
            "projection_id": "p1",
            "status": "built",
        }
        result = h.facade.build_projection(
            scope_kind="snapshot",
            snapshot_id="snap:repo:abc",
        )
        assert result["status"] == "built"
        h.projection_service.build_projection.assert_called_once_with(
            scope_kind="snapshot",
            snapshot_id="snap:repo:abc",
            repo_name=None,
            ref_name=None,
            query=None,
            target_id=None,
            filters=None,
            force=False,
        )

    def test_acquires_write_lock(self) -> None:
        """Verify build_projection holds the write lock during execution."""
        import threading

        h = _FacadeHarness()
        lock_acquired = False

        def _side_effect(**kwargs: Any) -> dict[str, Any]:
            nonlocal lock_acquired
            # Inside the service call, the lock should be held (write-locked
            # by the current thread).  RuntimeState._lock._writer stores the
            # native thread ident, matching threading.get_ident().
            assert h.state._lock._writer == threading.get_ident()
            lock_acquired = True
            return {"status": "ok"}

        h.projection_service.build_projection.side_effect = _side_effect
        h.facade.build_projection(scope_kind="snapshot", snapshot_id="snap:1")
        assert lock_acquired

    def test_forwards_all_parameters(self) -> None:
        h = _FacadeHarness()
        h.projection_service.build_projection.return_value = {"status": "ok"}

        h.facade.build_projection(
            scope_kind="query",
            snapshot_id="snap:repo:abc",
            repo_name="repo",
            ref_name="main",
            query="find auth",
            target_id="t1",
            filters={"path": "src/"},
            force=True,
        )

        h.projection_service.build_projection.assert_called_once_with(
            scope_kind="query",
            snapshot_id="snap:repo:abc",
            repo_name="repo",
            ref_name="main",
            query="find auth",
            target_id="t1",
            filters={"path": "src/"},
            force=True,
        )

    def test_passes_through_service_error(self) -> None:
        h = _FacadeHarness()
        h.projection_service.build_projection.side_effect = RuntimeError(
            "projection store not configured"
        )
        with pytest.raises(RuntimeError, match="projection store not configured"):
            h.facade.build_projection(scope_kind="snapshot")

    def test_releases_lock_on_error(self) -> None:
        h = _FacadeHarness()
        h.projection_service.build_projection.side_effect = RuntimeError("fail")

        with pytest.raises(RuntimeError):
            h.facade.build_projection(scope_kind="snapshot")

        # Lock should be released -- acquiring again should succeed immediately
        with h.state.write_lock():
            pass  # If lock was held, this would deadlock
