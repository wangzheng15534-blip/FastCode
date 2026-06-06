"""IndexingFacade -- repository loading and indexing operations.

Extracted from FastCode during Phase 7 of the FCIS split. Owns all repository
loading, indexing, ZIP upload, and multi-repo orchestration methods.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastcode.app.indexing.pipeline.service import IndexPipeline
from fastcode.utils.archive import safe_extract_zip, safe_repo_name_from_archive

if TYPE_CHECKING:
    import logging

    from fastcode.app.indexing.pipeline.direct_indexer import DirectIndexer
    from fastcode.app.indexing.pipeline.multi_repo_direct import MultiRepoDirectIndexer
    from fastcode.app.store.facade import StoreFacade
    from fastcode.app.store.vectors.vector import VectorStore
    from fastcode.scip.global_builder import GlobalIndexBuilder
    from fastcode.infrastructure.graph_runtime.contracts import DocumentGraphRuntime
    from fastcode.runtime_support.runtime_state import RuntimeState
    from fastcode.retrieval.hybrid_retriever import HybridRetriever
    from fastcode.scip.module_resolver import ModuleResolver
    from fastcode.scip.symbol_resolver import SymbolResolver


class IndexingFacade:
    """Facade wrapping all repository loading and indexing operations."""

    def __init__(
        self,
        loader: Any,
        pipeline: IndexPipeline,
        state: RuntimeState,
        vector_store: VectorStore,
        store: StoreFacade,
        direct_indexer: DirectIndexer,
        multi_repo_direct_indexer: MultiRepoDirectIndexer,
        graph_runtime: DocumentGraphRuntime | None,
        retriever: HybridRetriever,
        config: dict[str, Any],
        eval_config: dict[str, Any],
        logger: logging.Logger,
        set_repo_root_fn: Callable[[str], None],
        apply_env_ignore_patterns_fn: Callable[[], None],
    ) -> None:
        self._loader = loader
        self._pipeline = pipeline
        self._state = state
        self._vector_store = vector_store
        self._store = store
        self._direct_indexer = direct_indexer
        self._multi_repo_direct_indexer = multi_repo_direct_indexer
        self._graph_runtime = graph_runtime
        self._retriever = retriever
        self._config = config
        self._eval_config = eval_config
        self._logger = logger
        self._set_repo_root_fn = set_repo_root_fn
        self._apply_env_ignore_patterns_fn = apply_env_ignore_patterns_fn

        # Set by _index_repository_direct_unlocked
        self.global_index_builder: GlobalIndexBuilder | None = None
        self.module_resolver: ModuleResolver | None = None
        self.symbol_resolver: SymbolResolver | None = None

    # ------------------------------------------------------------------
    # Lock helper
    # ------------------------------------------------------------------

    def _state_lock(self) -> Any:
        return self._state._lock

    # ------------------------------------------------------------------
    # URL inference helper
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_is_url(source: str) -> bool:
        return IndexPipeline._infer_is_url(source)

    def _direct_index_enabled(self) -> bool:
        indexing_config = self._config.get("indexing", {})
        if not isinstance(indexing_config, dict):
            return False
        return bool(indexing_config.get("allow_direct_index", False))

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def load_repository(
        self, source: str, is_url: bool | None = None, is_zip: bool = False
    ):
        with self._state_lock():
            return self._load_repository_unlocked(source, is_url, is_zip)

    def upload_repository_zip(
        self, file_bytes: bytes, filename: str
    ) -> dict[str, Any]:
        with self._state_lock():
            return self._upload_repository_zip_unlocked(file_bytes, filename)

    def index_repository(self, force: bool = False):
        with self._state_lock():
            result = self._index_repository_unlocked(force=force)
            self._vector_store.invalidate_scan_cache()
        return result

    def load_and_index(
        self, source: str, is_url: bool | None = None, *, force: bool = False
    ) -> dict[str, Any]:
        with self._state_lock():
            self._load_repository_unlocked(source, is_url)
            self._index_repository_unlocked(force=force)
            self._vector_store.invalidate_scan_cache()
            return {
                "status": "success",
                "message": "Repository loaded and indexed successfully",
                "summary": self._store.get_repository_summary(),
            }

    def upload_and_index(
        self, file_bytes: bytes, filename: str, *, force: bool = False
    ) -> dict[str, Any]:
        with self._state_lock():
            upload_result = self._upload_repository_zip_unlocked(
                file_bytes, filename
            )
            if upload_result.get("status") != "success":
                return upload_result
            self._index_repository_unlocked(force=force)
            self._vector_store.invalidate_scan_cache()
            return {
                "status": "success",
                "message": "Repository uploaded and indexed successfully",
                "summary": self._store.get_repository_summary(),
            }

    def reindex_repository(self, source: str) -> str:
        self._apply_env_ignore_patterns_fn()
        resolved_is_url = self._infer_is_url(source)
        name = self._store.repo_name_from_source(source, resolved_is_url)
        self._logger.info("Force re-indexing '%s' from %s", name, source)

        if resolved_is_url:
            self.load_repository(source, is_url=True)
        else:
            abs_path = os.path.abspath(source)
            if not os.path.isdir(abs_path):
                return f"Error: Local path does not exist: {abs_path}"
            self.load_repository(abs_path, is_url=False)

        self.index_repository(force=True)
        count = self._vector_store.get_count()

        # Reset in-memory state for clean reload
        self._state.repo_indexed = False
        self._state.loaded_repositories.clear()

        return f"Successfully re-indexed '{name}': {count} elements indexed."

    def run_index_pipeline(
        self,
        source: str,
        is_url: bool | None = None,
        ref: str | None = None,
        commit: str | None = None,
        force: bool = False,
        publish: bool = True,
        scip_artifact_path: str | None = None,
        enable_scip: bool = True,
    ) -> dict[str, Any]:
        with self._state_lock():
            return self._pipeline.run_index_pipeline(
                source=source,
                is_url=is_url,
                ref=ref,
                commit=commit,
                force=force,
                publish=publish,
                scip_artifact_path=scip_artifact_path,
                enable_scip=enable_scip,
                load_repository_cb=self.load_repository,
                get_loaded_repositories=lambda: self._state.loaded_repositories,
                graph_runtime=self._graph_runtime,
            )

    def load_multiple_repositories(self, sources: list[dict[str, Any]]):
        with self._state_lock():
            return self._load_multiple_repositories_unlocked(sources)

    def incremental_reindex(
        self, repo_name: str, repo_path: str | None = None
    ) -> dict[str, Any]:
        if not repo_path or not os.path.isdir(repo_path):
            self._logger.warning(
                f"Invalid repo path for '{repo_name}': {repo_path}"
            )
            return {"status": "path_not_found", "changes": 0}
        return self.run_index_pipeline(
            source=repo_path,
            is_url=False,
            publish=True,
            enable_scip=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers (unlocked)
    # ------------------------------------------------------------------

    def _load_repository_unlocked(
        self,
        source: str,
        is_url: bool | None = None,
        is_zip: bool = False,
    ):
        self._logger.info(f"Loading repository: {source}")

        try:
            resolved_is_url = is_url
            if not is_zip and resolved_is_url is None:
                resolved_is_url = self._infer_is_url(source)
                source_type = "URL" if resolved_is_url else "local path"
                self._logger.info(
                    f"Auto-detected source type as {source_type}: {source}"
                )

            if is_zip:
                self._loader.load_from_zip(source)
            elif resolved_is_url:
                self._loader.load_from_url(source)
            else:
                self._loader.load_from_path(source)

            self._state.repo_loaded = True
            self._state.repo_info = self._loader.get_repository_info()

            if self._loader.repo_path:
                self._set_repo_root_fn(self._loader.repo_path)
                self._logger.info(f"Set repo_root to: {self._loader.repo_path}")
                self._retriever.set_repo_root(self._loader.repo_path)

            self._logger.info(
                f"Loaded repository: {self._state.repo_info.get('name')}"
            )
            self._logger.info(
                f"Files: {self._state.repo_info.get('file_count')}, "
                f"Size: {self._state.repo_info.get('total_size_mb', 0):.2f} MB"
            )

        except Exception as e:
            self._logger.error(f"Failed to load repository: {e}")
            raise

    def _upload_repository_zip_unlocked(
        self, file_bytes: bytes, filename: str
    ) -> dict[str, Any]:
        max_size = 100 * 1024 * 1024  # 100MB
        if len(file_bytes) > max_size:
            raise ValueError(
                f"File too large. Maximum size is {max_size / (1024 * 1024)}MB"
            )

        archive_filename = Path(filename).name
        repo_name = safe_repo_name_from_archive(archive_filename)

        repo_workspace = getattr(self._loader, "safe_repo_root", "./repos")
        repos_dir = Path(repo_workspace)
        repos_dir.mkdir(parents=True, exist_ok=True)
        repo_path = repos_dir / repo_name

        if repo_path.exists():
            self._loader._backup_existing_repo(str(repo_path))

        temp_dir = tempfile.mkdtemp(prefix="fastcode_upload_")
        try:
            zip_path = Path(temp_dir) / archive_filename

            self._logger.info(
                "Saving uploaded ZIP file: %s (%d bytes)",
                archive_filename,
                len(file_bytes),
            )
            zip_path.write_bytes(file_bytes)

            extract_dir = Path(temp_dir) / "extracted"
            extract_dir.mkdir(exist_ok=True)

            self._logger.info(
                "Extracting ZIP file to temporary directory: %s", extract_dir
            )
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                safe_extract_zip(zip_ref, extract_dir)

            extracted_items = list(extract_dir.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                source_repo_path = extracted_items[0]
            else:
                source_repo_path = extract_dir

            self._logger.info("Moving repository to: %s", repo_path)
            shutil.move(str(source_repo_path), str(repo_path))

            self._logger.info("Loading repository from: %s", repo_path)
            self._load_repository_unlocked(str(repo_path), False)

            return {
                "status": "success",
                "message": f"ZIP file '{archive_filename}' uploaded and extracted to repos/{repo_name}",
                "repo_info": self._state.repo_info,
                "repo_path": str(repo_path),
            }
        finally:
            try:
                shutil.rmtree(temp_dir)
                self._logger.info(
                    "Cleaned up temporary directory: %s", temp_dir
                )
            except Exception as cleanup_error:
                self._logger.warning(
                    "Failed to clean up temp directory: %s", cleanup_error
                )

    def _index_repository_unlocked(self, force: bool = False):
        if self._direct_index_enabled():
            return self._index_repository_direct_unlocked(force=force)
        force = force or self._eval_config.get("force_reindex", False)
        if not self._state.repo_loaded:
            raise RuntimeError(
                "No repository loaded. Call load_repository() first."
            )
        if not self._loader.repo_path:
            raise RuntimeError(
                "No repository path available for indexing."
            )
        return self._pipeline.run_index_pipeline(
            source=self._loader.repo_path,
            is_url=False,
            force=force,
            publish=True,
            enable_scip=True,
            load_repository_cb=None,
            get_loaded_repositories=lambda: self._state.loaded_repositories,
            graph_runtime=self._graph_runtime,
        )

    def _index_repository_direct_unlocked(self, force: bool = False):
        indexed, gib, mr, sr = self._direct_indexer.run(
            repo_loaded=self._state.repo_loaded,
            repo_info=self._state.repo_info,
            eval_config=self._eval_config,
            force=force,
        )
        if gib is not None:
            self.global_index_builder = gib
        if mr is not None:
            self.module_resolver = mr
        if sr is not None:
            self.symbol_resolver = sr
        self._state.repo_indexed = indexed

    def _load_multiple_repositories_unlocked(
        self, sources: list[dict[str, Any]]
    ):
        if not self._direct_index_enabled():
            return self._load_multiple_repositories_pipeline_unlocked(sources)
        return self._load_multiple_repositories_direct_unlocked(sources)

    def _load_multiple_repositories_pipeline_unlocked(
        self, sources: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self._logger.info(f"Loading {len(sources)} repositories")
        self._state.multi_repo_mode = True
        successfully_indexed: list[str] = []
        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        for i, source_info in enumerate(sources):
            source = str(source_info.get("source") or "")
            is_url: bool | None = source_info.get("is_url")
            is_zip = bool(source_info.get("is_zip", False))
            try:
                self._logger.info(
                    f"[{i + 1}/{len(sources)}] Loading repository: {source}"
                )
                resolved_is_url = (
                    self._infer_is_url(source)
                    if not is_zip and is_url is None
                    else is_url
                )
                pipeline_source = source
                pipeline_is_url = resolved_is_url
                load_repository_cb: Callable[..., None] | None
                if is_zip:
                    self._load_repository_unlocked(
                        source, is_url=False, is_zip=True
                    )
                    if not self._loader.repo_path:
                        raise RuntimeError(
                            "zip load did not produce a repository path"
                        )
                    pipeline_source = self._loader.repo_path
                    pipeline_is_url = False
                    load_repository_cb = None
                else:

                    def _load_repository_cb(
                        loaded_source: str, is_url: bool | None = None
                    ) -> None:
                        self._load_repository_unlocked(
                            loaded_source, is_url=is_url, is_zip=False
                        )

                    load_repository_cb = _load_repository_cb

                result = self._pipeline.run_index_pipeline(
                    source=pipeline_source,
                    is_url=pipeline_is_url,
                    force=bool(source_info.get("force", False)),
                    publish=True,
                    enable_scip=True,
                    load_repository_cb=load_repository_cb,
                    get_loaded_repositories=lambda: self._state.loaded_repositories,
                    graph_runtime=self._graph_runtime,
                )
                results.append(result)
                repo_name = str(result.get("repo_name") or "")
                if repo_name:
                    successfully_indexed.append(repo_name)
            except Exception as e:
                self._logger.error(
                    f"Failed to index repository {source}: {e}"
                )
                errors.append({"source": source, "error": str(e)})
                continue

        self._state.repo_indexed = len(successfully_indexed) > 0
        self._state.repo_loaded = len(successfully_indexed) > 0
        return {
            "status": "succeeded" if successfully_indexed else "failed",
            "repositories": successfully_indexed,
            "results": results,
            "errors": errors,
        }

    def _load_multiple_repositories_direct_unlocked(
        self, sources: list[dict[str, Any]]
    ):
        self._state.multi_repo_mode = True
        result = self._multi_repo_direct_indexer.run(
            sources,
            loaded_repositories=self._state.loaded_repositories,
        )
        self._state.repo_indexed = result["has_success"]
        self._state.repo_loaded = result["has_success"]
