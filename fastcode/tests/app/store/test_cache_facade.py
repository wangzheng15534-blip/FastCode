"""Tests for CacheFacade — cache operations extracted from FastCode."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

from fastcode.app.store.cache_facade import CacheFacade
from fastcode.runtime_support.runtime_state import RuntimeState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cache_facade() -> tuple[CacheFacade, dict[str, MagicMock]]:
    """Create a CacheFacade with all mocked dependencies."""
    cache_manager = MagicMock()
    vector_store = MagicMock()
    embedder = MagicMock()
    retriever = MagicMock()
    graph_builder = MagicMock()
    graph_artifact_store = MagicMock()
    state = RuntimeState()

    facade = CacheFacade(
        cache_manager=cache_manager,
        vector_store=vector_store,
        embedder=embedder,
        retriever=retriever,
        graph_builder=graph_builder,
        graph_artifact_store=graph_artifact_store,
        state=state,
    )

    return facade, {
        "cache_manager": cache_manager,
        "vector_store": vector_store,
        "embedder": embedder,
        "retriever": retriever,
        "graph_builder": graph_builder,
        "graph_artifact_store": graph_artifact_store,
        "state": state,
    }


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:
    def test_returns_true_on_success(self) -> None:
        facade, deps = _make_cache_facade()
        deps["cache_manager"].clear.return_value = True

        result = facade.clear_cache()

        assert result is True
        deps["cache_manager"].clear.assert_called_once()

    def test_returns_false_on_failure(self) -> None:
        facade, deps = _make_cache_facade()
        deps["cache_manager"].clear.return_value = False

        result = facade.clear_cache()

        assert result is False

    def test_returns_bool_for_truthy_int(self) -> None:
        facade, deps = _make_cache_facade()
        deps["cache_manager"].clear.return_value = 42

        result = facade.clear_cache()

        assert result is True

    def test_acquires_state_lock(self) -> None:
        facade, deps = _make_cache_facade()
        lock_held = False

        def _clear() -> bool:
            nonlocal lock_held
            lock_held = deps["state"]._lock._writer == threading.get_ident()
            return True

        deps["cache_manager"].clear.side_effect = _clear
        result = facade.clear_cache()

        assert result is True
        assert lock_held is True
        deps["cache_manager"].clear.assert_called_once()


# ---------------------------------------------------------------------------
# get_cache_stats
# ---------------------------------------------------------------------------


class TestGetCacheStats:
    def test_delegates_to_cache_manager(self) -> None:
        facade, deps = _make_cache_facade()
        expected_stats: dict[str, Any] = {
            "enabled": True,
            "backend": "memory",
            "items": 5,
            "size": 1024,
        }
        deps["cache_manager"].get_stats.return_value = expected_stats

        result = facade.get_cache_stats()

        assert result == expected_stats
        deps["cache_manager"].get_stats.assert_called_once()

    def test_returns_empty_dict_when_no_stats(self) -> None:
        facade, deps = _make_cache_facade()
        deps["cache_manager"].get_stats.return_value = {}

        result = facade.get_cache_stats()

        assert result == {}


# ---------------------------------------------------------------------------
# invalidate_scan_cache
# ---------------------------------------------------------------------------


class TestInvalidateScanCache:
    def test_delegates_to_vector_store(self) -> None:
        facade, deps = _make_cache_facade()

        facade.invalidate_scan_cache()

        deps["vector_store"].invalidate_scan_cache.assert_called_once()

    def test_does_not_acquire_lock(self) -> None:
        """invalidate_scan_cache is lock-free (pure delegation)."""
        facade, deps = _make_cache_facade()

        # Just verify it doesn't need the lock
        facade.invalidate_scan_cache()
        deps["vector_store"].invalidate_scan_cache.assert_called_once()


# ---------------------------------------------------------------------------
# refresh_index_cache
# ---------------------------------------------------------------------------


class TestRefreshIndexCache:
    def test_invalidates_and_rescans(self) -> None:
        facade, deps = _make_cache_facade()
        expected: list[dict[str, Any]] = [
            {"name": "repo1", "element_count": 100},
            {"name": "repo2", "element_count": 200},
        ]
        deps["vector_store"].scan_available_indexes.return_value = expected

        result = facade.refresh_index_cache()

        assert result == expected
        deps["vector_store"].invalidate_scan_cache.assert_called_once()
        deps["vector_store"].scan_available_indexes.assert_called_once_with(
            use_cache=False
        )

    def test_returns_empty_list_when_no_indexes(self) -> None:
        facade, deps = _make_cache_facade()
        deps["vector_store"].scan_available_indexes.return_value = []

        result = facade.refresh_index_cache()

        assert result == []

    def test_acquires_state_lock(self) -> None:
        facade, deps = _make_cache_facade()
        lock_held = False

        def _scan(*, use_cache: bool) -> list[dict[str, Any]]:
            nonlocal lock_held
            assert use_cache is False
            lock_held = deps["state"]._lock._writer == threading.get_ident()
            return []

        deps["vector_store"].scan_available_indexes.side_effect = _scan

        facade.refresh_index_cache()

        assert lock_held is True
        deps["vector_store"].invalidate_scan_cache.assert_called_once()
        deps["vector_store"].scan_available_indexes.assert_called_once_with(
            use_cache=False
        )


# ---------------------------------------------------------------------------
# load_cached_repos
# ---------------------------------------------------------------------------


class TestLoadCachedRepos:
    @patch("fastcode.app.store.cache_facade._load_multi_repo_cache_impl")
    def test_delegates_to_rehydration(self, mock_impl: MagicMock) -> None:
        facade, deps = _make_cache_facade()
        mock_impl.return_value = True

        result = facade.load_cached_repos(repo_names=["repo1", "repo2"])

        assert result is True
        mock_impl.assert_called_once_with(
            repo_names=["repo1", "repo2"],
            vector_store=deps["vector_store"],
            embedder=deps["embedder"],
            retriever=deps["retriever"],
            graph_builder=deps["graph_builder"],
            graph_artifact_store=deps["graph_artifact_store"],
            loaded_repositories=deps["state"].loaded_repositories,
        )

    @patch("fastcode.app.store.cache_facade._load_multi_repo_cache_impl")
    def test_passes_none_repo_names(self, mock_impl: MagicMock) -> None:
        facade, deps = _make_cache_facade()
        mock_impl.return_value = True

        result = facade.load_cached_repos()

        assert result is True
        mock_impl.assert_called_once_with(
            repo_names=None,
            vector_store=deps["vector_store"],
            embedder=deps["embedder"],
            retriever=deps["retriever"],
            graph_builder=deps["graph_builder"],
            graph_artifact_store=deps["graph_artifact_store"],
            loaded_repositories=deps["state"].loaded_repositories,
        )

    @patch("fastcode.app.store.cache_facade._load_multi_repo_cache_impl")
    def test_returns_false_on_failure(self, mock_impl: MagicMock) -> None:
        facade, _deps = _make_cache_facade()
        mock_impl.return_value = False

        result = facade.load_cached_repos()

        assert result is False

    @patch("fastcode.app.store.cache_facade._load_multi_repo_cache_impl")
    def test_marks_runtime_ready_on_success(self, mock_impl: MagicMock) -> None:
        facade, deps = _make_cache_facade()
        mock_impl.return_value = True

        result = facade.load_cached_repos(repo_names=["repo1"])

        assert result is True
        assert deps["state"].repo_loaded is True
        assert deps["state"].repo_indexed is True
        assert deps["state"].multi_repo_mode is True

    @patch("fastcode.app.store.cache_facade._load_multi_repo_cache_impl")
    def test_does_not_mark_runtime_ready_on_failure(self, mock_impl: MagicMock) -> None:
        facade, deps = _make_cache_facade()
        mock_impl.return_value = False

        result = facade.load_cached_repos(repo_names=["missing"])

        assert result is False
        assert deps["state"].repo_loaded is False
        assert deps["state"].repo_indexed is False
        assert deps["state"].multi_repo_mode is False

    @patch("fastcode.app.store.cache_facade._load_multi_repo_cache_impl")
    def test_acquires_state_lock(self, mock_impl: MagicMock) -> None:
        facade, deps = _make_cache_facade()
        lock_held = False

        def _load_impl(**kwargs: Any) -> bool:
            nonlocal lock_held
            lock_held = deps["state"]._lock._writer == threading.get_ident()
            return True

        mock_impl.side_effect = _load_impl

        facade.load_cached_repos()
        assert lock_held is True
        mock_impl.assert_called_once()
