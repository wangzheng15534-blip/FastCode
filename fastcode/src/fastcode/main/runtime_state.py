"""Runtime state holder extracted from FastCode for injection into facades."""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

_STATE_LOCK_LOGGER = logging.getLogger(f"{__name__}.state_lock")


class _ReadWriteStateLock:
    """Reentrant writer lock with shared read sections for immutable queries."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._readers = 0
        self._reader_depths: dict[int, int] = {}
        self._writer: int | None = None
        self._write_depth = 0

    def __enter__(self) -> _ReadWriteStateLock:
        self._acquire_write()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._release_write()

    @contextmanager
    def read_lock(self) -> Generator[None, None, None]:
        self._acquire_read()
        try:
            yield
        finally:
            self._release_read()

    @contextmanager
    def write_lock(self) -> Generator[None, None, None]:
        self._acquire_write()
        try:
            yield
        finally:
            self._release_write()

    def _acquire_read(self) -> None:
        ident = threading.get_ident()
        _STATE_LOCK_LOGGER.debug(
            "Acquiring service state read lock",
            extra={"fc_event": "service_lock_acquire", "lock_mode": "read"},
        )
        with self._condition:
            if self._writer == ident:
                self._readers += 1
                self._reader_depths[ident] = self._reader_depths.get(ident, 0) + 1
                _STATE_LOCK_LOGGER.debug(
                    "Acquired service state read lock",
                    extra={
                        "fc_event": "service_lock_acquired",
                        "lock_mode": "read",
                        "reader_count": self._readers,
                    },
                )
                return
            while self._writer is not None:
                self._condition.wait()
            self._readers += 1
            self._reader_depths[ident] = self._reader_depths.get(ident, 0) + 1
            _STATE_LOCK_LOGGER.debug(
                "Acquired service state read lock",
                extra={
                    "fc_event": "service_lock_acquired",
                    "lock_mode": "read",
                    "reader_count": self._readers,
                },
            )

    def _release_read(self) -> None:
        ident = threading.get_ident()
        with self._condition:
            depth = self._reader_depths.get(ident, 0)
            if depth <= 1:
                self._reader_depths.pop(ident, None)
            else:
                self._reader_depths[ident] = depth - 1
            self._readers -= 1
            _STATE_LOCK_LOGGER.debug(
                "Released service state read lock",
                extra={
                    "fc_event": "service_lock_released",
                    "lock_mode": "read",
                    "reader_count": self._readers,
                },
            )
            if self._readers == 0:
                self._condition.notify_all()

    def _acquire_write(self) -> None:
        ident = threading.get_ident()
        _STATE_LOCK_LOGGER.debug(
            "Acquiring service state write lock",
            extra={"fc_event": "service_lock_acquire", "lock_mode": "write"},
        )
        with self._condition:
            if self._writer == ident:
                self._write_depth += 1
                _STATE_LOCK_LOGGER.debug(
                    "Acquired service state write lock",
                    extra={
                        "fc_event": "service_lock_acquired",
                        "lock_mode": "write",
                        "write_depth": self._write_depth,
                    },
                )
                return
            own_read_depth = self._reader_depths.get(ident, 0)
            while self._writer is not None or self._readers > own_read_depth:
                self._condition.wait()
            self._writer = ident
            self._write_depth = 1
            _STATE_LOCK_LOGGER.debug(
                "Acquired service state write lock",
                extra={
                    "fc_event": "service_lock_acquired",
                    "lock_mode": "write",
                    "write_depth": self._write_depth,
                },
            )

    def _release_write(self) -> None:
        ident = threading.get_ident()
        with self._condition:
            if self._writer != ident:
                raise RuntimeError(
                    "cannot release state write lock not owned by thread"
                )
            self._write_depth -= 1
            _STATE_LOCK_LOGGER.debug(
                "Released service state write lock",
                extra={
                    "fc_event": "service_lock_released",
                    "lock_mode": "write",
                    "write_depth": self._write_depth,
                },
            )
            if self._write_depth == 0:
                self._writer = None
                self._condition.notify_all()


class RuntimeState:
    """Mutable runtime state shared between FastCode and extracted facades.

    Holds the five state booleans/dicts that previously lived directly on
    the FastCode instance, plus the read-write lock that serializes access.
    """

    __slots__ = (
        "_lock",
        "loaded_repositories",
        "multi_repo_mode",
        "repo_indexed",
        "repo_info",
        "repo_loaded",
    )

    def __init__(self) -> None:
        self._lock = _ReadWriteStateLock()
        self.repo_loaded: bool = False
        self.repo_indexed: bool = False
        self.repo_info: dict[str, Any] = {}
        self.multi_repo_mode: bool = False
        self.loaded_repositories: dict[str, dict[str, Any]] = {}

    def read_lock(self) -> AbstractContextManager[None]:
        return self._lock.read_lock()

    def write_lock(self) -> AbstractContextManager[None]:
        return self._lock.write_lock()
