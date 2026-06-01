"""Tests for RuntimeState extracted from FastCode."""

from __future__ import annotations

import threading
import time

from fastcode.runtime_support.runtime_state import RuntimeState, _ReadWriteStateLock


class TestRuntimeStateDefaults:
    def test_repo_loaded_defaults_false(self) -> None:
        state = RuntimeState()
        assert state.repo_loaded is False

    def test_repo_indexed_defaults_false(self) -> None:
        state = RuntimeState()
        assert state.repo_indexed is False

    def test_repo_info_defaults_empty_dict(self) -> None:
        state = RuntimeState()
        assert state.repo_info == {}

    def test_multi_repo_mode_defaults_false(self) -> None:
        state = RuntimeState()
        assert state.multi_repo_mode is False

    def test_loaded_repositories_defaults_empty_dict(self) -> None:
        state = RuntimeState()
        assert state.loaded_repositories == {}

    def test_lock_is_read_write_lock(self) -> None:
        state = RuntimeState()
        assert isinstance(state._lock, _ReadWriteStateLock)


class TestRuntimeStateMutation:
    def test_repo_loaded_mutable(self) -> None:
        state = RuntimeState()
        state.repo_loaded = True
        assert state.repo_loaded is True

    def test_repo_indexed_mutable(self) -> None:
        state = RuntimeState()
        state.repo_indexed = True
        assert state.repo_indexed is True

    def test_repo_info_mutable(self) -> None:
        state = RuntimeState()
        state.repo_info = {"name": "test-repo", "file_count": 10}
        assert state.repo_info["name"] == "test-repo"

    def test_multi_repo_mode_mutable(self) -> None:
        state = RuntimeState()
        state.multi_repo_mode = True
        assert state.multi_repo_mode is True

    def test_loaded_repositories_mutable(self) -> None:
        state = RuntimeState()
        state.loaded_repositories["repo-a"] = {"file_count": 5}
        assert "repo-a" in state.loaded_repositories


class TestRuntimeStateLockDelegates:
    def test_read_lock_context_manager(self) -> None:
        state = RuntimeState()
        with state.read_lock():
            assert True  # no error acquiring read lock

    def test_write_lock_context_manager(self) -> None:
        state = RuntimeState()
        with state.write_lock():
            assert True  # no error acquiring write lock


class TestReadWriteStateLock:
    def test_write_lock_is_reentrant(self) -> None:
        lock = _ReadWriteStateLock()
        with lock, lock:
            assert True  # double-acquire should not deadlock

    def test_read_lock_is_reentrant(self) -> None:
        lock = _ReadWriteStateLock()
        with lock.read_lock(), lock.read_lock():
            assert True

    def test_read_and_write_from_same_thread(self) -> None:
        lock = _ReadWriteStateLock()
        with lock, lock.read_lock():
            assert True

    def test_write_lock_serializes_with_other_threads(self) -> None:
        lock = _ReadWriteStateLock()
        concurrent_count = 0
        max_concurrent = 0
        count_lock = threading.Lock()
        barrier = threading.Barrier(2, timeout=5)
        errors: list[Exception] = []

        def _enter_critical(name: str) -> None:
            nonlocal concurrent_count, max_concurrent
            with count_lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
            time.sleep(0.05)
            with count_lock:
                concurrent_count -= 1

        def _writer1() -> None:
            try:
                barrier.wait(timeout=5)
                with lock:
                    _enter_critical("w1")
            except Exception as exc:
                errors.append(exc)

        def _writer2() -> None:
            try:
                barrier.wait(timeout=5)
                with lock:
                    _enter_critical("w2")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_writer1)
        t2 = threading.Thread(target=_writer2)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Threads raised: {errors}"
        assert max_concurrent <= 1

    def test_concurrent_readers_allowed(self) -> None:
        lock = _ReadWriteStateLock()
        concurrent_count = 0
        max_concurrent = 0
        count_lock = threading.Lock()
        barrier = threading.Barrier(2, timeout=5)
        errors: list[Exception] = []

        def _reader() -> None:
            nonlocal concurrent_count, max_concurrent
            try:
                barrier.wait(timeout=5)
                with lock.read_lock():
                    with count_lock:
                        concurrent_count += 1
                        max_concurrent = max(max_concurrent, concurrent_count)
                    time.sleep(0.05)
                    with count_lock:
                        concurrent_count -= 1
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_reader)
        t2 = threading.Thread(target=_reader)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Threads raised: {errors}"
        assert max_concurrent == 2

    def test_read_blocks_write(self) -> None:
        lock = _ReadWriteStateLock()
        write_acquired = threading.Event()
        read_started = threading.Event()
        errors: list[Exception] = []

        def _reader() -> None:
            try:
                with lock.read_lock():
                    read_started.set()
                    time.sleep(0.1)
            except Exception as exc:
                errors.append(exc)

        def _writer() -> None:
            try:
                read_started.wait(timeout=5)
                with lock:
                    write_acquired.set()
            except Exception as exc:
                errors.append(exc)

        reader_thread = threading.Thread(target=_reader)
        writer_thread = threading.Thread(target=_writer)
        reader_thread.start()
        writer_thread.start()
        reader_thread.join(timeout=10)
        writer_thread.join(timeout=10)

        assert not errors, f"Threads raised: {errors}"
        assert write_acquired.is_set()

    def test_release_write_lock_by_non_owner_raises(self) -> None:
        lock = _ReadWriteStateLock()
        errors: list[Exception] = []

        def _bad_release() -> None:
            try:
                lock._release_write()
            except RuntimeError as exc:
                errors.append(exc)

        t = threading.Thread(target=_bad_release)
        t.start()
        t.join(timeout=5)

        assert len(errors) == 1
        assert "not owned by thread" in str(errors[0])


class TestRuntimeStateIntegrationWithFastCode:
    """Verify RuntimeState works as drop-in for direct field access."""

    def test_state_fields_replace_direct_attributes(self) -> None:
        """Simulate the FastCode pattern: self.state = RuntimeState()."""
        state = RuntimeState()

        # Simulate load_repository
        state.repo_loaded = True
        state.repo_info = {"name": "test-repo", "file_count": 10}

        # Simulate index_repository
        state.repo_indexed = True

        # Simulate multi-repo
        state.multi_repo_mode = True
        state.loaded_repositories["repo-a"] = {"file_count": 5}

        # Simulate get_status_info reading state
        assert state.repo_loaded is True
        assert state.repo_indexed is True
        assert state.repo_info["name"] == "test-repo"
        assert state.multi_repo_mode is True
        assert "repo-a" in state.loaded_repositories

        # Simulate reindex reset
        state.repo_indexed = False
        state.loaded_repositories.clear()
        assert state.repo_indexed is False
        assert state.loaded_repositories == {}
