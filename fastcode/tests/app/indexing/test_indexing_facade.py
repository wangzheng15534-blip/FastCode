"""Tests for IndexingFacade -- indexing operations extracted from FastCode."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fastcode.app.indexing.facade import IndexingFacade
from fastcode.runtime_support.runtime_state import RuntimeState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FacadeHarness:
    """Test harness that holds the facade and its mock dependencies."""

    def __init__(self, *, config: dict[str, Any] | None = None) -> None:
        self.loader = MagicMock()
        self.loader.repo_path = "/tmp/repo"
        self.loader.get_repository_info.return_value = {
            "name": "test-repo",
            "file_count": 10,
            "total_size_mb": 1.5,
        }
        self.pipeline = MagicMock()
        self.state = RuntimeState()
        self.vector_store = MagicMock()
        self.store = MagicMock()
        self.direct_indexer = MagicMock()
        self.multi_repo_direct_indexer = MagicMock()
        self.graph_runtime = None
        self.retriever = MagicMock()
        self.config = config or {}
        self.eval_config: dict[str, Any] = {}
        self.logger = MagicMock()
        self.set_repo_root_fn = MagicMock()
        self.apply_env_ignore_patterns_fn = MagicMock()

        self.facade = IndexingFacade(
            loader=self.loader,
            pipeline=self.pipeline,
            state=self.state,
            vector_store=self.vector_store,
            store=self.store,
            direct_indexer=self.direct_indexer,
            multi_repo_direct_indexer=self.multi_repo_direct_indexer,
            graph_runtime=self.graph_runtime,
            retriever=self.retriever,
            config=self.config,
            eval_config=self.eval_config,
            logger=self.logger,
            set_repo_root_fn=self.set_repo_root_fn,
            apply_env_ignore_patterns_fn=self.apply_env_ignore_patterns_fn,
        )


# ---------------------------------------------------------------------------
# TestLoadRepository
# ---------------------------------------------------------------------------


class TestLoadRepository:
    def test_load_from_path(self) -> None:
        h = _FacadeHarness()
        h.facade.load_repository("/tmp/my-repo", is_url=False)
        h.loader.load_from_path.assert_called_once_with("/tmp/my-repo")
        assert h.state.repo_loaded is True
        assert h.state.repo_info["name"] == "test-repo"

    def test_load_from_url(self) -> None:
        h = _FacadeHarness()
        h.facade.load_repository("https://github.com/org/repo.git", is_url=True)
        h.loader.load_from_url.assert_called_once_with(
            "https://github.com/org/repo.git"
        )

    def test_load_from_zip(self) -> None:
        h = _FacadeHarness()
        h.facade.load_repository("/tmp/repo.zip", is_zip=True)
        h.loader.load_from_zip.assert_called_once_with("/tmp/repo.zip")

    def test_auto_detect_url(self) -> None:
        h = _FacadeHarness()
        h.facade.load_repository("https://github.com/org/repo.git")
        h.loader.load_from_url.assert_called_once()

    def test_auto_detect_local_path(self) -> None:
        h = _FacadeHarness()
        h.facade.load_repository("/tmp/local-repo")
        h.loader.load_from_path.assert_called_once_with("/tmp/local-repo")

    def test_set_repo_root_callback_called(self) -> None:
        h = _FacadeHarness()
        h.facade.load_repository("/tmp/my-repo", is_url=False)
        h.set_repo_root_fn.assert_called_once_with("/tmp/repo")

    def test_retriever_set_repo_root_called(self) -> None:
        h = _FacadeHarness()
        h.facade.load_repository("/tmp/my-repo", is_url=False)
        h.retriever.set_repo_root.assert_called_once_with("/tmp/repo")

    def test_load_error_raises(self) -> None:
        h = _FacadeHarness()
        h.loader.load_from_path.side_effect = FileNotFoundError("not found")
        with pytest.raises(FileNotFoundError, match="not found"):
            h.facade.load_repository("/tmp/missing", is_url=False)


# ---------------------------------------------------------------------------
# TestUploadRepositoryZip
# ---------------------------------------------------------------------------


class TestUploadRepositoryZip:
    def test_upload_success(self) -> None:
        h = _FacadeHarness()
        file_bytes = b"fake zip content"
        mock_zip = MagicMock()
        with patch("fastcode.app.indexing.facade.zipfile.ZipFile", return_value=mock_zip), \
             patch("fastcode.app.indexing.facade.safe_extract_zip"), \
             patch("fastcode.app.indexing.facade.safe_repo_name_from_archive", return_value="my-repo"):
            result = h.facade.upload_repository_zip(file_bytes, "my-repo.zip")
        assert result["status"] == "success"
        assert "repo_info" in result

    def test_upload_too_large(self) -> None:
        h = _FacadeHarness()
        big_bytes = b"x" * (101 * 1024 * 1024)  # 101 MB
        with pytest.raises(ValueError, match="too large"):
            h.facade.upload_repository_zip(big_bytes, "big.zip")


# ---------------------------------------------------------------------------
# TestIndexRepository
# ---------------------------------------------------------------------------


class TestIndexRepository:
    def test_index_pipeline_path(self) -> None:
        h = _FacadeHarness()
        h.state.repo_loaded = True
        h.facade.index_repository(force=False)
        h.pipeline.run_index_pipeline.assert_called_once()
        h.vector_store.invalidate_scan_cache.assert_called_once()

    def test_index_direct_path(self) -> None:
        h = _FacadeHarness(config={"indexing": {"allow_direct_index": True}})
        h.direct_indexer.run.return_value = (True, MagicMock(), MagicMock(), MagicMock())
        h.facade.index_repository(force=False)
        h.direct_indexer.run.assert_called_once()

    def test_index_without_loaded_repo_raises(self) -> None:
        h = _FacadeHarness()
        h.state.repo_loaded = False
        with pytest.raises(RuntimeError, match="No repository loaded"):
            h.facade.index_repository()


# ---------------------------------------------------------------------------
# TestLoadAndIndex
# ---------------------------------------------------------------------------


class TestLoadAndIndex:
    def test_load_and_index_success(self) -> None:
        h = _FacadeHarness()
        h.state.repo_loaded = True
        h.store.get_repository_summary.return_value = "summary"
        result = h.facade.load_and_index("/tmp/repo", is_url=False)
        assert result["status"] == "success"
        h.vector_store.invalidate_scan_cache.assert_called_once()

    def test_load_and_index_with_force(self) -> None:
        h = _FacadeHarness()
        h.state.repo_loaded = True
        h.store.get_repository_summary.return_value = "summary"
        result = h.facade.load_and_index("/tmp/repo", is_url=False, force=True)
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# TestUploadAndIndex
# ---------------------------------------------------------------------------


class TestUploadAndIndex:
    def test_upload_and_index_success(self) -> None:
        h = _FacadeHarness()
        h.state.repo_loaded = True
        h.store.get_repository_summary.return_value = "summary"
        mock_zip = MagicMock()
        with patch("fastcode.app.indexing.facade.zipfile.ZipFile", return_value=mock_zip), \
             patch("fastcode.app.indexing.facade.safe_extract_zip"), \
             patch("fastcode.app.indexing.facade.safe_repo_name_from_archive", return_value="repo"):
            result = h.facade.upload_and_index(b"zipdata", "repo.zip")
        assert result["status"] == "success"
        h.vector_store.invalidate_scan_cache.assert_called_once()


# ---------------------------------------------------------------------------
# TestLoadMultipleRepositories
# ---------------------------------------------------------------------------


class TestLoadMultipleRepositories:
    def test_pipeline_path(self) -> None:
        h = _FacadeHarness()
        h.pipeline.run_index_pipeline.return_value = {"repo_name": "r1"}
        sources = [{"source": "/tmp/r1", "is_url": False}]
        result = h.facade.load_multiple_repositories(sources)
        assert result["status"] == "succeeded"
        assert "r1" in result["repositories"]

    def test_pipeline_path_with_error(self) -> None:
        h = _FacadeHarness()
        h.pipeline.run_index_pipeline.side_effect = RuntimeError("fail")
        sources = [{"source": "/tmp/r1", "is_url": False}]
        result = h.facade.load_multiple_repositories(sources)
        assert result["status"] == "failed"
        assert len(result["errors"]) == 1

    def test_direct_path(self) -> None:
        h = _FacadeHarness(config={"indexing": {"allow_direct_index": True}})
        h.multi_repo_direct_indexer.run.return_value = {"has_success": True}
        sources = [{"source": "/tmp/r1"}]
        h.facade.load_multiple_repositories(sources)
        assert h.state.multi_repo_mode is True
        assert h.state.repo_indexed is True


# ---------------------------------------------------------------------------
# TestReindexRepository
# ---------------------------------------------------------------------------


class TestReindexRepository:
    def test_reindex_local_path(self) -> None:
        h = _FacadeHarness()
        h.store.repo_name_from_source.return_value = "my-repo"
        h.vector_store.get_count.return_value = 42
        with patch("os.path.isdir", return_value=True), \
             patch("os.path.abspath", return_value="/tmp/my-repo"):
            result = h.facade.reindex_repository("/tmp/my-repo")
        assert "42 elements indexed" in result
        assert h.state.repo_indexed is False

    def test_reindex_nonexistent_path(self) -> None:
        h = _FacadeHarness()
        h.store.repo_name_from_source.return_value = "my-repo"
        with patch("os.path.isdir", return_value=False), \
             patch("os.path.abspath", return_value="/tmp/missing"):
            result = h.facade.reindex_repository("/tmp/missing")
        assert "Error" in result

    def test_reindex_calls_env_ignore_patterns(self) -> None:
        h = _FacadeHarness()
        h.store.repo_name_from_source.return_value = "my-repo"
        h.vector_store.get_count.return_value = 0
        with patch("os.path.isdir", return_value=True), \
             patch("os.path.abspath", return_value="/tmp/my-repo"):
            h.facade.reindex_repository("/tmp/my-repo")
        h.apply_env_ignore_patterns_fn.assert_called_once()


# ---------------------------------------------------------------------------
# TestRunIndexPipeline
# ---------------------------------------------------------------------------


class TestRunIndexPipeline:
    def test_delegates_to_pipeline(self) -> None:
        h = _FacadeHarness()
        h.pipeline.run_index_pipeline.return_value = {"status": "ok"}
        result = h.facade.run_index_pipeline(
            source="/tmp/repo", is_url=False, force=True
        )
        assert result == {"status": "ok"}
        call_kwargs = h.pipeline.run_index_pipeline.call_args
        assert call_kwargs.kwargs["source"] == "/tmp/repo"
        assert call_kwargs.kwargs["force"] is True


# ---------------------------------------------------------------------------
# TestIncrementalReindex
# ---------------------------------------------------------------------------


class TestIncrementalReindex:
    def test_valid_path(self) -> None:
        h = _FacadeHarness()
        h.pipeline.run_index_pipeline.return_value = {"status": "ok"}
        with patch("os.path.isdir", return_value=True):
            result = h.facade.incremental_reindex("repo", "/tmp/repo")
        assert result == {"status": "ok"}

    def test_invalid_path(self) -> None:
        h = _FacadeHarness()
        result = h.facade.incremental_reindex("repo", None)
        assert result["status"] == "path_not_found"

    def test_nonexistent_path(self) -> None:
        h = _FacadeHarness()
        with patch("os.path.isdir", return_value=False):
            result = h.facade.incremental_reindex("repo", "/tmp/missing")
        assert result["status"] == "path_not_found"


# ---------------------------------------------------------------------------
# TestDirectIndexEnabled
# ---------------------------------------------------------------------------


class TestDirectIndexEnabled:
    def test_default_disabled(self) -> None:
        h = _FacadeHarness()
        assert h.facade._direct_index_enabled() is False

    def test_enabled_via_config(self) -> None:
        h = _FacadeHarness(config={"indexing": {"allow_direct_index": True}})
        assert h.facade._direct_index_enabled() is True

    def test_non_dict_indexing_config(self) -> None:
        h = _FacadeHarness(config={"indexing": "not-a-dict"})
        assert h.facade._direct_index_enabled() is False


# ---------------------------------------------------------------------------
# TestIndexRepositoryDirectUnlocked
# ---------------------------------------------------------------------------


class TestIndexRepositoryDirectUnlocked:
    def test_sets_resolver_attributes(self) -> None:
        h = _FacadeHarness(config={"indexing": {"allow_direct_index": True}})
        mock_gib = MagicMock()
        mock_mr = MagicMock()
        mock_sr = MagicMock()
        h.direct_indexer.run.return_value = (True, mock_gib, mock_mr, mock_sr)
        h.facade.index_repository(force=False)
        assert h.facade.global_index_builder is mock_gib
        assert h.facade.module_resolver is mock_mr
        assert h.facade.symbol_resolver is mock_sr

    def test_none_resolvers_not_stored(self) -> None:
        h = _FacadeHarness(config={"indexing": {"allow_direct_index": True}})
        h.direct_indexer.run.return_value = (True, None, None, None)
        h.facade.index_repository(force=False)
        assert h.facade.global_index_builder is None
        assert h.facade.module_resolver is None
        assert h.facade.symbol_resolver is None
