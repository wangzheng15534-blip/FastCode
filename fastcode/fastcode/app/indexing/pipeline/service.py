"""
IndexPipeline - Extracted indexing pipeline from FastCode.

Handles repository checkout, snapshot resolution, AST+SCIP dual-source
extraction, IR merge, semantic resolution, and artifact persistence.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import tempfile
import threading
import tracemalloc
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from datetime import datetime
from time import perf_counter
from typing import Any, cast
from urllib.parse import urlparse

import numpy as np
from git import GitCommandError, Repo

from fastcode.app.indexing.doc_ingester import KeyDocIngester
from fastcode.app.indexing.embedder import CodeEmbedder
from fastcode.app.indexing.file_inventory import FileInventory
from fastcode.app.indexing.graph_mapper import document_overlay_node_records
from fastcode.app.indexing.loader import RepositoryLoader
from fastcode.app.query.selection.retriever import HybridRetriever
from fastcode.app.store.artifacts.graph import GraphArtifactStore
from fastcode.app.store.runs.index_run import IndexRunStore
from fastcode.app.store.snapshots.manifest import ManifestStore
from fastcode.app.store.snapshots.snapshot import SnapshotStore
from fastcode.app.store.vectors.pg_retrieval import PgRetrievalStore
from fastcode.app.store.vectors.vector import VectorStore
from fastcode.app.store.vectors.vector_math import as_float32_matrix, as_float32_vector
from fastcode.common.identifiers import RepoName, SnapshotId
from fastcode.graph.build import CodeGraphBuilder
from fastcode.infrastructure.execution.ports import (
    ScipFileInfoView,
    ScipIndexerRuntime,
    SemanticHelperRuntime,
)
from fastcode.infrastructure.graph_runtime.contracts import DocumentGraphRuntime
from fastcode.ir.element import (
    CodeElement,
    CodeElementMeta,
    deserialize_code_element,
    serialize_code_element,
)
from fastcode.ir.graph import IRGraphBuilder, IRGraphs
from fastcode.ir.merge import merge_ir
from fastcode.ir.types import IRRelation, IRSnapshot, IRUnitSupport
from fastcode.ir.validate import validate_snapshot
from fastcode.ports.artifacts import (
    FileArtifactStore as FileArtifactStorePort,
)
from fastcode.ports.artifacts import (
    UnitArtifactStore as UnitArtifactStorePort,
)
from fastcode.ports.publishing import LineagePublisher
from fastcode.scip.ast_adapter import build_ir_from_ast
from fastcode.scip.global_builder import GlobalIndexBuilder
from fastcode.scip.models import SCIPIndex
from fastcode.scip.module_resolver import ModuleResolver
from fastcode.scip.scip_adapter import build_ir_from_scip
from fastcode.scip.symbol_resolver import SymbolResolver
from fastcode.semantic.contracts import SemanticGraphContext
from fastcode.semantic.resolvers.engine.patching import apply_resolution_patch
from fastcode.semantic.resolvers.engine.registry import (
    build_default_semantic_resolver_registry,
)
from fastcode.semantic.symbol_index import SnapshotSymbolIndex
from fastcode.utils.filesystem import compute_file_hash, ensure_dir, normalize_path
from fastcode.utils.materialization import (
    BOUNDARY_JSON_DECODE,
    BOUNDARY_JSON_ENCODE,
    BOUNDARY_PICKLE_LOAD,
    MaterializationCounters,
    increment_materialization_boundary,
    reset_materialization_counters,
    set_materialization_counters,
)

from .incremental import (
    FileChangeSet,
    PlanChanges,
    apply_incremental_update,
    diff_changed_files,
)
from .indexer import CodeIndexer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedSnapshotArtifacts:
    """Read-only serving handles for immutable snapshot query requests.

    The contained vector store, retriever, and graph builder are safe to share
    across concurrent snapshot reads after construction. Mutation, repair, and
    rebuild paths must publish a new handle and invalidate the old artifact-key
    cache entry instead of modifying an active handle in place.
    """

    artifact_key: str
    snapshot_id: str | None
    vector_store: VectorStore
    retriever: HybridRetriever
    graph_builder: CodeGraphBuilder


@dataclass(frozen=True)
class ScopedSCIPCacheEntry:
    key: str
    path: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class IncrementalGraphDeltaContext:
    elements: list[CodeElement]
    affected_path_keys: set[str]
    reusable_path_keys: set[str]
    loaded_affected_path_keys: set[str]
    missing_affected_path_keys: set[str]


class IndexPipeline:
    """Encapsulates the full snapshot-oriented indexing pipeline."""

    _PACKAGE_SCOPE_MARKERS = (
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "package.json",
        "tsconfig.json",
        "go.mod",
        "Cargo.toml",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "composer.json",
        "Project.toml",
    )

    def __init__(
        self,
        *,
        config: dict[str, Any],
        logger: Any,
        loader: RepositoryLoader,
        snapshot_store: SnapshotStore,
        manifest_store: ManifestStore,
        index_run_store: IndexRunStore,
        unit_artifact_store: UnitArtifactStorePort,
        snapshot_symbol_index: SnapshotSymbolIndex,
        vector_store: VectorStore,
        embedder: CodeEmbedder,
        indexer: CodeIndexer,
        retriever: HybridRetriever,
        graph_builder: CodeGraphBuilder,
        ir_graph_builder: IRGraphBuilder,
        pg_retrieval_store: PgRetrievalStore | None,
        terminus_publisher: LineagePublisher,
        doc_ingester: KeyDocIngester,
        semantic_resolver_registry: Any,
        # Callbacks for mutable FastCode state
        set_repo_indexed: Callable[[bool], None],
        set_repo_loaded: Callable[[bool], None],
        set_repo_info: Callable[[dict[str, Any]], None],
        graph_artifact_store: GraphArtifactStore | None = None,
        semantic_helper_runtime: SemanticHelperRuntime | None = None,
        scip_indexer_runtime: ScipIndexerRuntime | None = None,
        file_artifact_store: FileArtifactStorePort | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.loader = loader
        self.snapshot_store = snapshot_store
        self.manifest_store = manifest_store
        self.index_run_store = index_run_store
        self.unit_artifact_store = unit_artifact_store
        self.file_artifact_store = file_artifact_store
        self.snapshot_symbol_index = snapshot_symbol_index
        self.vector_store = vector_store
        self.embedder = embedder
        self.indexer = indexer
        self.retriever = retriever
        self.graph_builder = graph_builder
        self.ir_graph_builder = ir_graph_builder
        self.graph_artifact_store = graph_artifact_store or GraphArtifactStore(config)
        self.pg_retrieval_store = pg_retrieval_store
        self.terminus_publisher = terminus_publisher
        self.doc_ingester = doc_ingester
        self.semantic_resolver_registry = semantic_resolver_registry
        self.semantic_helper_runtime = semantic_helper_runtime
        self.scip_indexer_runtime = scip_indexer_runtime
        self._set_repo_indexed = set_repo_indexed
        self._set_repo_loaded = set_repo_loaded
        self._set_repo_info = set_repo_info
        self._artifact_lock = threading.RLock()
        self._artifact_handle_cache: OrderedDict[str, LoadedSnapshotArtifacts] = (
            OrderedDict()
        )
        self._artifact_handle_cache_hits = 0
        self._artifact_handle_cache_misses = 0
        self._last_file_inventory_metrics: dict[str, Any] = {}
        self._last_repository_inventory_metrics: dict[str, Any] = {}
        self._active_pipeline_profile: dict[str, Any] | None = None

    def _run_scip_indexer(
        self,
        language: str,
        repo_path: str,
        output_path: str,
    ) -> str:
        if self.scip_indexer_runtime is None:
            msg = "SCIP indexer runtime is not configured"
            raise RuntimeError(msg)
        return self.scip_indexer_runtime.run_indexer(language, repo_path, output_path)

    def _scip_profile(self, language: str) -> Any | None:
        if self.scip_indexer_runtime is None:
            return None
        return self.scip_indexer_runtime.get_profile(language)

    def _detect_scip_languages_from_file_infos(
        self,
        file_infos: Sequence[ScipFileInfoView],
    ) -> tuple[str, ...]:
        if self.scip_indexer_runtime is None:
            return ()
        return self.scip_indexer_runtime.detect_languages_from_file_infos(file_infos)

    def _detect_scip_languages_in_paths(
        self,
        repo_path: str,
        relative_paths: Sequence[str],
    ) -> tuple[str, ...]:
        if self.scip_indexer_runtime is None:
            return ()
        return self.scip_indexer_runtime.detect_languages_in_paths(
            repo_path, relative_paths
        )

    @staticmethod
    def _experimental_scip_languages(languages: Sequence[str]) -> list[str]:
        experimental: list[str] = []
        for language in languages:
            if language in {"zig", "fortran", "julia"}:
                experimental.append(language)
        return experimental

    def _load_scip_artifact(self, path: str) -> SCIPIndex:
        if self.scip_indexer_runtime is None:
            msg = "SCIP indexer runtime is not configured"
            raise RuntimeError(msg)
        return self.scip_indexer_runtime.load_artifact(path)

    def _run_scip_for_language(
        self,
        language: str,
        repo_path: str,
        output_dir: str,
    ) -> SCIPIndex | None:
        output_path = os.path.join(output_dir, f"{language}.scip")
        try:
            artifact_path = self._run_scip_indexer(language, repo_path, output_path)
            return self._load_scip_artifact(artifact_path)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            self.logger.warning("SCIP indexer for %s unavailable: %s", language, exc)
            return None

    def _run_scip_python_index(self, repo_path: str, output_path: str) -> str:
        return self._run_scip_indexer("python", repo_path, output_path)

    def _load_graph_artifact_into(
        self,
        builder: CodeGraphBuilder,
        artifact_key: str,
    ) -> bool:
        graph_artifact_store = getattr(self, "graph_artifact_store", None)
        load_graph_artifact = getattr(graph_artifact_store, "load", None)
        if callable(load_graph_artifact):
            return bool(load_graph_artifact(builder, artifact_key))

        legacy_load = getattr(builder, "load", None)
        if callable(legacy_load):
            return bool(legacy_load(artifact_key))
        return False

    # ------------------------------------------------------------------
    # URL inference (pure static utility)
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_is_url(source: str) -> bool:
        """
        Infer whether source should be treated as URL.

        Priority rule: existing local paths always win over URL heuristics.
        """
        normalized = (source or "").strip()
        if not normalized:
            return False

        if os.path.exists(normalized):
            return False

        parsed = urlparse(normalized)
        if parsed.scheme in {"http", "https", "ssh", "git", "file"}:
            return True

        # SCP-like git syntax, e.g. git@github.com:user/repo.git
        return bool(re.match(r"^[^@\s]+@[^:\s]+:[^\s]+$", normalized))

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------

    def _checkout_target_ref(
        self, ref: str | None = None, commit: str | None = None
    ) -> None:
        """Checkout requested ref/commit inside loaded repository workspace."""
        target = commit or ref
        if not target or not self.loader.repo_path:
            return
        if getattr(self.loader, "repo_load_mode", None) == "in_place":
            msg = (
                "Ref/commit checkout would mutate an in-place local repository. "
                "Set repository.local_source_mode='copy' or 'hardlink', or load "
                "a workspace clone before indexing a different ref."
            )
            raise RuntimeError(msg)
        try:
            repo = Repo(self.loader.repo_path)
            repo.git.checkout(target)
            invalidate_inventory = getattr(
                self.loader, "invalidate_preloaded_file_inventory", None
            )
            if callable(invalidate_inventory):
                invalidate_inventory()
            self.logger.info(f"Checked out target: {target}")
        except (GitCommandError, Exception) as e:
            msg = f"Failed to checkout target '{target}': {e}"
            raise RuntimeError(msg)

    def _resolve_snapshot_ref(
        self,
        repo_name: str,
        requested_ref: str | None = None,
        requested_commit: str | None = None,
        current_files: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Resolve repo snapshot identity from git metadata or file hashes."""

        def file_identity(file_info: Mapping[str, Any]) -> str:
            precomputed = file_info.get("content_hash") or file_info.get("blob_oid")
            if precomputed:
                return str(precomputed)
            return compute_file_hash(str(file_info["path"]))

        repo_path = self.loader.repo_path or ""
        try:
            repo = Repo(repo_path)
            commit_obj = repo.commit(requested_commit or requested_ref or "HEAD")
            tree_id = commit_obj.tree.hexsha
            commit_id = commit_obj.hexsha
            branch = requested_ref
            if branch is None:
                try:
                    branch = repo.active_branch.name
                except Exception:
                    branch = None
            dirty_suffix = ""
            try:
                if repo.is_dirty(untracked_files=True):
                    files = (
                        current_files
                        if current_files is not None
                        else self.loader.scan_files()
                    )
                    digest = hashlib.sha1()
                    for f in sorted(files, key=lambda x: x["relative_path"]):
                        digest.update(f["relative_path"].encode("utf-8"))
                        digest.update(file_identity(f).encode("utf-8"))
                    working_tree_hash = digest.hexdigest()
                    tree_id = f"{tree_id}:dirty:{working_tree_hash}"
                    dirty_suffix = f":dirty:{working_tree_hash}"
            except Exception:
                dirty_suffix = ""
            snapshot_id = SnapshotId.build(
                RepoName.parse(repo_name).as_str(),
                f"{commit_id}{dirty_suffix}",
            ).as_str()
            return {
                "repo_name": repo_name,
                "branch": branch,
                "commit_id": commit_id,
                "tree_id": tree_id,
                "snapshot_id": snapshot_id,
            }
        except Exception:
            files = (
                current_files if current_files is not None else self.loader.scan_files()
            )
            if not files:
                synthetic = "empty"
            else:
                digest = hashlib.sha1()
                for f in sorted(files, key=lambda x: x["relative_path"]):
                    digest.update(f["relative_path"].encode("utf-8"))
                    try:
                        digest.update(file_identity(f).encode("utf-8"))
                    except Exception:
                        digest.update(str(f.get("size", 0)).encode("utf-8"))
                synthetic = digest.hexdigest()
            return {
                "repo_name": repo_name,
                "branch": requested_ref,
                "commit_id": requested_commit,
                "tree_id": synthetic,
                "snapshot_id": SnapshotId.build(
                    RepoName.parse(repo_name).as_str(),
                    synthetic,
                ).as_str(),
            }

    def _build_git_meta(self, snapshot_ref: dict[str, Any]) -> dict[str, Any]:
        git_meta = dict(snapshot_ref or {})
        commit_id = git_meta.get("commit_id")
        if not commit_id or not self.loader.repo_path:
            return git_meta
        try:
            repo = Repo(self.loader.repo_path)
            commit_obj = repo.commit(commit_id)
            parent_ids = [p.hexsha for p in commit_obj.parents]
            git_meta["parent_commit_id"] = parent_ids[0] if parent_ids else None
            git_meta["parent_commit_ids"] = parent_ids
        except (ValueError, KeyError) as e:
            self.logger.warning(f"Failed to resolve commit parent metadata: {e}")
        return git_meta

    def _previous_snapshot_symbol_versions(
        self,
        repo_name: str,
        ref_name: str,
        current_snapshot_id: str,
    ) -> dict[str, str] | None:
        previous_manifest = self.manifest_store.get_branch_manifest_record(
            repo_name, ref_name
        )
        if not previous_manifest:
            return None
        previous_snapshot_id = previous_manifest.snapshot_id
        if not previous_snapshot_id or previous_snapshot_id == current_snapshot_id:
            return None
        previous_snapshot = self.snapshot_store.load_snapshot(previous_snapshot_id)
        if not previous_snapshot:
            return None
        out: dict[str, str] = {}
        for symbol in previous_snapshot.symbols:
            if not symbol.external_symbol_id:
                continue
            out[symbol.external_symbol_id] = (
                f"symbol:{previous_snapshot_id}:{symbol.symbol_id}"
            )
        return out

    # ------------------------------------------------------------------
    # Artifact loading (shared with query + projection services)
    # ------------------------------------------------------------------

    def _load_artifacts_by_key(self, artifact_key: str) -> bool:
        """Load vector/BM25/graph artifacts for a snapshot artifact key."""
        with self._artifact_lock:
            loaded = self._load_artifacts_by_key_locked(artifact_key)
        log = getattr(self, "logger", logger)
        log.info(
            "Loaded mutable artifact handles",
            extra={
                "fc_event": "artifact_handle_swap",
                "artifact_key": artifact_key,
                "artifact_handle_kind": "mutable",
                "artifact_loaded": loaded,
            },
        )
        return loaded

    def _artifact_handle_cache_limit(self) -> int:
        configured = self.config.get("query", {}).get("snapshot_handle_cache_size", 4)
        try:
            return max(1, int(configured))
        except (TypeError, ValueError):
            return 4

    def _invalidate_loaded_artifact_handle(self, artifact_key: str) -> None:
        removed = self._artifact_handle_cache.pop(artifact_key, None) is not None
        log = getattr(self, "logger", logger)
        log.info(
            "Invalidated cached artifact handle",
            extra={
                "fc_event": "artifact_handle_invalidate",
                "artifact_key": artifact_key,
                "artifact_handle_removed": removed,
            },
        )

    def _configure_retriever_ir_graph_backend(
        self,
        retriever: Any,
        *,
        snapshot_id: str | None,
    ) -> None:
        if not snapshot_id:
            retriever.set_ir_graphs(None, snapshot_id=None)
            return
        set_loader = getattr(retriever, "set_ir_graph_loader", None)
        if callable(set_loader):
            set_loader(
                self.snapshot_store.load_ir_graphs,
                snapshot_id=snapshot_id,
            )
            return
        ir_graphs = self.snapshot_store.load_ir_graphs(snapshot_id)
        retriever.set_ir_graphs(ir_graphs, snapshot_id=snapshot_id)

    def load_snapshot_artifacts_handle(
        self,
        artifact_key: str,
        *,
        snapshot_id: str | None = None,
    ) -> LoadedSnapshotArtifacts | None:
        with self._artifact_lock:
            handle = self._load_snapshot_artifacts_handle_locked(
                artifact_key,
                snapshot_id=snapshot_id,
            )
        log = getattr(self, "logger", logger)
        log.info(
            "Loaded snapshot artifact handle",
            extra={
                "fc_event": "artifact_handle_swap",
                "artifact_key": artifact_key,
                "snapshot_id": snapshot_id,
                "artifact_handle_kind": "snapshot",
                "artifact_loaded": handle is not None,
            },
        )
        return handle

    def _load_snapshot_artifacts_handle_locked(
        self,
        artifact_key: str,
        *,
        snapshot_id: str | None = None,
    ) -> LoadedSnapshotArtifacts | None:
        cached = self._artifact_handle_cache.get(artifact_key)
        if cached is not None:
            self._increment_artifact_handle_cache_hit()
            self._artifact_handle_cache.move_to_end(artifact_key)
            return cached
        self._increment_artifact_handle_cache_miss()

        resolved_snapshot_id = snapshot_id
        if resolved_snapshot_id is None and artifact_key.startswith("snap_"):
            resolved_snapshot_id = self._snapshot_id_for_artifact_key(artifact_key)

        vector_store = VectorStore(self.config)
        if not vector_store.load(artifact_key):
            return None

        graph_builder = CodeGraphBuilder(self.config)
        graph_loaded = self._load_graph_artifact_into(graph_builder, artifact_key)
        retriever = HybridRetriever(
            self.config,
            vector_store,
            self.embedder,
            graph_builder,
            repo_root=self.loader.repo_path,
        )
        retriever.set_pg_retrieval_store(self.pg_retrieval_store)
        bm25_loaded = retriever.load_bm25(artifact_key)
        self._configure_retriever_ir_graph_backend(
            retriever,
            snapshot_id=resolved_snapshot_id,
        )
        retriever.build_repo_overview_bm25()

        if not bm25_loaded or not graph_loaded:
            elements = self._reconstruct_elements_from_metadata_view(
                vector_store.metadata
            )
            if elements:
                if not bm25_loaded:
                    retriever.index_for_bm25(elements)
                if not graph_loaded:
                    graph_builder.build_graphs(elements)

        handle = LoadedSnapshotArtifacts(
            artifact_key=artifact_key,
            snapshot_id=resolved_snapshot_id,
            vector_store=vector_store,
            retriever=retriever,
            graph_builder=graph_builder,
        )
        self._artifact_handle_cache[artifact_key] = handle
        while len(self._artifact_handle_cache) > self._artifact_handle_cache_limit():
            self._artifact_handle_cache.popitem(last=False)
        return handle

    def _load_artifacts_by_key_locked(self, artifact_key: str) -> bool:
        if not self.vector_store.load(artifact_key):
            return False

        bm25_loaded = self.retriever.load_bm25(artifact_key)
        graph_loaded = self._load_graph_artifact_into(self.graph_builder, artifact_key)
        if artifact_key.startswith("snap_"):
            snapshot_id = self._snapshot_id_for_artifact_key(artifact_key)
            self._configure_retriever_ir_graph_backend(
                self.retriever,
                snapshot_id=snapshot_id,
            )
        else:
            self.retriever.set_ir_graphs(None, snapshot_id=None)
        self.retriever.build_repo_overview_bm25()

        if not bm25_loaded or not graph_loaded:
            elements = self._reconstruct_elements_from_metadata()
            if elements:
                if not bm25_loaded:
                    self.retriever.index_for_bm25(elements)
                if not graph_loaded:
                    self.graph_builder.build_graphs(elements)

        self._set_repo_indexed(True)
        self._set_repo_loaded(True)
        return True

    def _snapshot_id_for_artifact_key(self, artifact_key: str) -> str | None:
        record = self.snapshot_store.find_by_artifact_key_record(artifact_key)
        candidate = getattr(record, "snapshot_id", None)
        if isinstance(candidate, str) and candidate:
            return candidate
        return None

    @staticmethod
    def _reconstruct_elements_from_metadata_view(
        metadata_rows: Sequence[Mapping[str, Any]],
    ) -> list[CodeElement]:
        """
        Reconstruct CodeElement objects from vector store metadata.
        Excludes repository_overview elements.
        """
        elements: list[CodeElement] = []
        for meta in metadata_rows:
            try:
                if meta.get("type") == "repository_overview":
                    continue

                element = CodeElement(
                    id=meta.get("id", ""),
                    type=meta.get("type", ""),
                    name=meta.get("name", ""),
                    file_path=meta.get("file_path", ""),
                    relative_path=meta.get("relative_path", ""),
                    language=meta.get("language", ""),
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    code=meta.get("code", ""),
                    signature=meta.get("signature"),
                    docstring=meta.get("docstring"),
                    summary=meta.get("summary"),
                    metadata=meta.get("metadata", {}),
                    repo_name=meta.get("repo_name"),
                    repo_url=meta.get("repo_url"),
                )
                elements.append(element)
            except (TypeError, ValueError):
                continue
        return elements

    def _reconstruct_elements_from_metadata(self) -> list[CodeElement]:
        elements = self._reconstruct_elements_from_metadata_view(
            self.vector_store.metadata
        )
        self.logger.info(
            f"Reconstructed {len(elements)} elements from metadata"
            " (excluding repository_overview)"
        )
        return elements

    def _unit_artifact_rows(
        self,
        elements: Sequence[Any],
        *,
        target_paths: set[str] | None = None,
        repair_frontier_summary: dict[str, Any] | None = None,
        scoped_tool_ref: str | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        repo_root = self.loader.repo_path or ""
        target_paths_norm = (
            {normalize_path(path) for path in target_paths} if target_paths else None
        )
        for elem in elements:
            row = self._unit_artifact_row_payload(elem)
            rel_path = normalize_path(
                row.get("relative_path") or row.get("file_path") or ""
            )
            if target_paths_norm is not None and rel_path not in target_paths_norm:
                continue
            metadata = dict(cast(dict[str, Any], row.get("metadata", {}) or {}))
            if rel_path:
                metadata.setdefault(
                    "package_root", self._package_scope_root(repo_root, rel_path)
                )
            embedding_artifact_ref = row.get("embedding_artifact_ref")
            if embedding_artifact_ref is None:
                embedding_artifact_ref = metadata.get("embedding_artifact_ref")
            if embedding_artifact_ref is None and row.get("embedding_text"):
                embedding_artifact_ref_factory = getattr(
                    self.embedder, "embedding_artifact_ref", None
                )
                if callable(embedding_artifact_ref_factory):
                    embedding_artifact_ref = embedding_artifact_ref_factory(
                        str(row["embedding_text"])
                    )
            if embedding_artifact_ref is not None:
                metadata["embedding_artifact_ref"] = embedding_artifact_ref
            if scoped_tool_ref is not None:
                metadata["scoped_tool_ref"] = scoped_tool_ref
            if repair_frontier_summary is not None:
                metadata["repair_frontier_summary"] = json.dumps(
                    repair_frontier_summary,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            row["metadata"] = metadata
            if embedding_artifact_ref is not None:
                row["embedding_artifact_ref"] = embedding_artifact_ref
            if scoped_tool_ref is not None:
                row["scoped_tool_ref"] = scoped_tool_ref
            if rel_path:
                row["package_root"] = metadata.get("package_root")
            if repair_frontier_summary is not None:
                row["repair_frontier_summary"] = metadata["repair_frontier_summary"]
            rows.append(row)
        return rows

    @staticmethod
    def _stable_unit_id_from_artifact_row(row: Mapping[str, Any]) -> str:
        metadata = row.get("metadata")
        if isinstance(metadata, Mapping):
            stable_unit_id = metadata.get("stable_unit_id")
            if stable_unit_id:
                return str(stable_unit_id)
        stable_unit_id = row.get("stable_unit_id")
        return str(stable_unit_id) if stable_unit_id else ""

    @staticmethod
    def _stable_unit_id_from_ir_unit(unit: Any) -> str:
        metadata = getattr(unit, "metadata", None)
        if isinstance(metadata, Mapping):
            stable_unit_id = metadata.get("stable_unit_id")
            if stable_unit_id:
                return str(stable_unit_id)
        return ""

    @classmethod
    def _current_unit_artifact_rows(
        cls,
        *,
        rows: Sequence[dict[str, Any]],
        snapshot: IRSnapshot,
        target_paths: set[str],
    ) -> list[dict[str, Any]]:
        if not snapshot.units:
            return [dict(row) for row in rows]
        target_paths_norm = {normalize_path(path) for path in target_paths}
        units_by_stable_id: dict[str, Any] = {}
        for unit in snapshot.units:
            stable_unit_id = cls._stable_unit_id_from_ir_unit(unit)
            if not stable_unit_id:
                continue
            unit_path = normalize_path(getattr(unit, "path", ""))
            if unit_path not in target_paths_norm:
                continue
            units_by_stable_id[stable_unit_id] = unit

        refreshed_rows: list[dict[str, Any]] = []
        metadata_fields = (
            "content_hash",
            "syntax_hash",
            "signature_hash",
            "edge_surface_hash",
            "embedding_text_hash",
            "api_surface_hash",
            "embedding_artifact_ref",
            "scoped_tool_ref",
            "package_root",
            "repair_frontier_summary",
        )
        for row in rows:
            stable_unit_id = cls._stable_unit_id_from_artifact_row(row)
            if not stable_unit_id or stable_unit_id not in units_by_stable_id:
                continue
            unit = units_by_stable_id[stable_unit_id]
            row_payload = dict(row)
            row_metadata = cls._unit_artifact_metadata_payload(
                row_payload.get("metadata")
            )
            unit_metadata = cls._unit_artifact_metadata_payload(
                getattr(unit, "metadata", None)
            )
            row_metadata.update(unit_metadata)
            row_payload["metadata"] = row_metadata
            unit_path = getattr(unit, "path", None)
            if unit_path:
                row_payload["relative_path"] = unit_path
            for field_name in metadata_fields:
                if row_metadata.get(field_name) is not None:
                    row_payload[field_name] = row_metadata.get(field_name)
            refreshed_rows.append(row_payload)
        return refreshed_rows

    @staticmethod
    def _unit_artifact_metadata_payload(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        return dict(cast(Mapping[str, Any], value))

    @classmethod
    def _legacy_element_mapping(cls, element: Any) -> Mapping[str, Any] | None:
        to_dict = getattr(element, "to_dict", None)
        if not callable(to_dict):
            return None
        try:
            payload = to_dict()
        except Exception:
            return None
        if not isinstance(payload, Mapping):
            return None
        return cast(Mapping[str, Any], payload)

    @classmethod
    def _unit_artifact_row_payload(cls, element: Any) -> dict[str, Any]:
        if isinstance(element, CodeElement):
            payload = cast(dict[str, Any], serialize_code_element(element))
        elif isinstance(element, Mapping):
            mapping = cast(Mapping[str, Any], element)
            payload = {
                field_name: mapping.get(field_name)
                for field_name in (
                    "id",
                    "type",
                    "name",
                    "file_path",
                    "relative_path",
                    "language",
                    "start_line",
                    "end_line",
                    "code",
                    "signature",
                    "docstring",
                    "summary",
                    "repo_name",
                    "repo_url",
                    "embedding_text",
                    "embedding_artifact_ref",
                    "scoped_tool_ref",
                    "package_root",
                    "repair_frontier_summary",
                    "stable_unit_id",
                    "content_hash",
                    "syntax_hash",
                    "signature_hash",
                    "edge_surface_hash",
                    "embedding_text_hash",
                    "api_surface_hash",
                )
            }
            payload["metadata"] = cls._unit_artifact_metadata_payload(
                mapping.get("metadata")
            )
        else:
            payload = {
                "id": getattr(element, "id", None),
                "type": getattr(element, "type", None),
                "name": getattr(element, "name", None),
                "file_path": getattr(element, "file_path", None),
                "relative_path": getattr(element, "relative_path", None),
                "language": getattr(element, "language", None),
                "start_line": getattr(element, "start_line", None),
                "end_line": getattr(element, "end_line", None),
                "code": getattr(element, "code", None),
                "signature": getattr(element, "signature", None),
                "docstring": getattr(element, "docstring", None),
                "summary": getattr(element, "summary", None),
                "repo_name": getattr(element, "repo_name", None),
                "repo_url": getattr(element, "repo_url", None),
                "embedding_text": getattr(element, "embedding_text", None),
                "embedding_artifact_ref": getattr(
                    element, "embedding_artifact_ref", None
                ),
                "scoped_tool_ref": getattr(element, "scoped_tool_ref", None),
                "package_root": getattr(element, "package_root", None),
                "repair_frontier_summary": getattr(
                    element, "repair_frontier_summary", None
                ),
                "stable_unit_id": getattr(element, "stable_unit_id", None),
                "content_hash": getattr(element, "content_hash", None),
                "syntax_hash": getattr(element, "syntax_hash", None),
                "signature_hash": getattr(element, "signature_hash", None),
                "edge_surface_hash": getattr(element, "edge_surface_hash", None),
                "embedding_text_hash": getattr(element, "embedding_text_hash", None),
                "api_surface_hash": getattr(element, "api_surface_hash", None),
                "metadata": cls._unit_artifact_metadata_payload(
                    getattr(element, "metadata", None)
                ),
            }
            legacy_payload = cls._legacy_element_mapping(element)
            if legacy_payload is not None:
                for field_name in (
                    "id",
                    "type",
                    "name",
                    "file_path",
                    "relative_path",
                    "language",
                    "start_line",
                    "end_line",
                    "code",
                    "signature",
                    "docstring",
                    "summary",
                    "repo_name",
                    "repo_url",
                    "embedding_text",
                    "embedding_artifact_ref",
                    "scoped_tool_ref",
                    "package_root",
                    "repair_frontier_summary",
                    "stable_unit_id",
                    "content_hash",
                    "syntax_hash",
                    "signature_hash",
                    "edge_surface_hash",
                    "embedding_text_hash",
                    "api_surface_hash",
                ):
                    if (
                        payload.get(field_name) is None
                        and legacy_payload.get(field_name) is not None
                    ):
                        payload[field_name] = legacy_payload.get(field_name)
                if not payload["metadata"]:
                    payload["metadata"] = cls._unit_artifact_metadata_payload(
                        legacy_payload.get("metadata")
                    )
        payload = cast(dict[str, Any], payload)
        metadata = cls._unit_artifact_metadata_payload(payload.get("metadata"))
        payload["metadata"] = metadata
        if payload.get("embedding_text") is None:
            payload["embedding_text"] = metadata.get("embedding_text")
        return payload

    @classmethod
    def _code_element_like_payload(cls, element: Any) -> CodeElementMeta:
        if isinstance(element, CodeElement):
            return serialize_code_element(element)
        if isinstance(element, Mapping):
            mapping = cast(Mapping[str, Any], element)
            metadata = cls._unit_artifact_metadata_payload(mapping.get("metadata"))
            return {
                "id": str(mapping.get("id") or ""),
                "type": str(mapping.get("type") or ""),
                "name": str(mapping.get("name") or ""),
                "file_path": str(mapping.get("file_path") or ""),
                "relative_path": str(mapping.get("relative_path") or ""),
                "language": str(mapping.get("language") or ""),
                "start_line": int(mapping.get("start_line") or 0),
                "end_line": int(mapping.get("end_line") or 0),
                "code": str(mapping.get("code") or ""),
                "signature": (
                    None
                    if mapping.get("signature") is None
                    else str(mapping.get("signature"))
                ),
                "docstring": (
                    None
                    if mapping.get("docstring") is None
                    else str(mapping.get("docstring"))
                ),
                "summary": (
                    None
                    if mapping.get("summary") is None
                    else str(mapping.get("summary"))
                ),
                "metadata": metadata,
                "repo_name": (
                    None
                    if mapping.get("repo_name") is None
                    else str(mapping.get("repo_name"))
                ),
                "repo_url": (
                    None
                    if mapping.get("repo_url") is None
                    else str(mapping.get("repo_url"))
                ),
            }
        metadata = cls._unit_artifact_metadata_payload(
            getattr(element, "metadata", None)
        )
        signature = getattr(element, "signature", None)
        docstring = getattr(element, "docstring", None)
        summary = getattr(element, "summary", None)
        repo_name = getattr(element, "repo_name", None)
        repo_url = getattr(element, "repo_url", None)
        payload: CodeElementMeta = {
            "id": str(getattr(element, "id", "") or ""),
            "type": str(getattr(element, "type", "") or ""),
            "name": str(getattr(element, "name", "") or ""),
            "file_path": str(getattr(element, "file_path", "") or ""),
            "relative_path": str(getattr(element, "relative_path", "") or ""),
            "language": str(getattr(element, "language", "") or ""),
            "start_line": int(getattr(element, "start_line", 0) or 0),
            "end_line": int(getattr(element, "end_line", 0) or 0),
            "code": str(getattr(element, "code", "") or ""),
            "signature": None if signature is None else str(signature),
            "docstring": None if docstring is None else str(docstring),
            "summary": None if summary is None else str(summary),
            "metadata": metadata,
            "repo_name": None if repo_name is None else str(repo_name),
            "repo_url": None if repo_url is None else str(repo_url),
        }
        legacy_payload = cls._legacy_element_mapping(element)
        if legacy_payload is None:
            return payload
        if not payload["metadata"]:
            payload["metadata"] = cls._unit_artifact_metadata_payload(
                legacy_payload.get("metadata")
            )
        for field_name in (
            "id",
            "type",
            "name",
            "file_path",
            "relative_path",
            "language",
            "code",
        ):
            if not payload[field_name] and legacy_payload.get(field_name) is not None:
                payload[field_name] = str(legacy_payload.get(field_name) or "")
        for field_name in ("start_line", "end_line"):
            if not payload[field_name] and legacy_payload.get(field_name) is not None:
                payload[field_name] = int(legacy_payload.get(field_name) or 0)
        for field_name in (
            "signature",
            "docstring",
            "summary",
            "repo_name",
            "repo_url",
        ):
            if (
                payload[field_name] is None
                and legacy_payload.get(field_name) is not None
            ):
                payload[field_name] = str(legacy_payload.get(field_name))
        return payload

    def _has_active_doc_persistence(
        self, graph_runtime: DocumentGraphRuntime | None
    ) -> bool:
        """Return True when doc ingestion has at least one active sink."""
        return self.snapshot_store.db_runtime.backend == "postgres" or bool(
            graph_runtime is not None and graph_runtime.enabled
        )

    def _should_ingest_docs(self, graph_runtime: DocumentGraphRuntime | None) -> bool:
        """Only ingest docs when the feature is enabled and results can be persisted."""
        return bool(
            getattr(self.doc_ingester, "enabled", False)
        ) and self._has_active_doc_persistence(graph_runtime)

    def _sync_doc_overlay(
        self,
        graph_runtime: DocumentGraphRuntime | None,
        *,
        chunks: list[dict[str, Any]],
        mentions: list[dict[str, Any]],
        warnings: list[str],
    ) -> None:
        """Best-effort Ladybug sync with explicit failure reporting."""
        if not chunks or graph_runtime is None or not graph_runtime.enabled:
            return
        try:
            synced = graph_runtime.sync_nodes(
                nodes=document_overlay_node_records(chunks=chunks, mentions=mentions)
            )
        except Exception as e:
            warnings.append(f"ladybug_doc_sync_failed: {e}")
            return
        if not synced:
            warnings.append("ladybug_doc_sync_failed")

    # ------------------------------------------------------------------
    # Semantic resolvers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_layer_record(
        *,
        name: str,
        ordinal: int,
        enabled: bool,
        source: str,
        description: str,
        status: str = "pending",
        strict: bool = False,
        conditional: bool = False,
        reason: str | None = None,
        metrics: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "ordinal": ordinal,
            "enabled": enabled,
            "source": source,
            "description": description,
            "status": status,
            "strict": strict,
            "conditional": conditional,
            "reason": reason,
            "metrics": dict(metrics or {}),
            "warnings": list(warnings or []),
        }

    @staticmethod
    def _finalize_layer_metrics(
        snapshot: IRSnapshot | None,
        layer: dict[str, Any],
        *,
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        metrics = dict(layer.get("metrics") or {})
        if snapshot is not None:
            metrics.update(
                {
                    "units": len(snapshot.units),
                    "supports": len(snapshot.supports),
                    "relations": len(snapshot.relations),
                    "embeddings": len(snapshot.embeddings),
                }
            )
        if extra_metrics:
            metrics.update(extra_metrics)
        layer["metrics"] = metrics

    def _snapshot_layer_metadata(self, snapshot: IRSnapshot) -> dict[str, Any]:
        metadata = dict(snapshot.metadata or {})
        record = self.snapshot_store.get_snapshot_record(snapshot.snapshot_id)
        if record and record.metadata_json:
            try:
                stored_metadata = json.loads(record.metadata_json)
            except (TypeError, json.JSONDecodeError):
                stored_metadata = {}
            if isinstance(stored_metadata, dict):
                metadata = {**stored_metadata, **metadata}
        return metadata

    @staticmethod
    def _layer3_quality_metrics(snapshot: IRSnapshot) -> dict[str, Any]:
        relations = list(snapshot.relations)
        total_relations = len(relations)
        structural = 0
        anchored = 0
        semantic = 0
        pending = 0
        for relation in relations:
            state = relation.resolution_state
            if state == "structural":
                structural += 1
            elif state == "anchored":
                anchored += 1
            elif state in {"semantic", "semantically_resolved"}:
                semantic += 1
            if relation.pending_capabilities:
                pending += 1
        return {
            "total_relations": total_relations,
            "structural_relations": structural,
            "anchored_relations": anchored,
            "semantic_relations": semantic,
            "relations_with_pending_capabilities": pending,
        }

    def _default_pipeline_layers(self, *, enable_scip: bool) -> list[dict[str, Any]]:
        return [
            self._make_layer_record(
                name="plain_ast_embedding",
                ordinal=1,
                enabled=True,
                source="hKUDS_fastcode_upstream",
                description="Original tree-sitter/code-index graph plus embedding pipeline without SCIP patching.",
                strict=True,
            ),
            self._make_layer_record(
                name="unified_ir_scip_merge",
                ordinal=2,
                enabled=enable_scip,
                source="fastcode_ir_patch",
                description="Canonical IR built from AST and optionally merged with SCIP precision anchors.",
                strict=False,
                conditional=True,
                reason="disabled_by_config" if not enable_scip else None,
                status="skipped" if not enable_scip else "pending",
            ),
            self._make_layer_record(
                name="language_specific_semantic_upgrade",
                ordinal=3,
                enabled=True,
                source="language_specific_ast_resolvers",
                description="Language-specific semantic resolver layer that upgrades graph relations beyond universal AST and SCIP anchors.",
                strict=False,
                conditional=True,
            ),
        ]

    def _backfill_result_layer_metadata(
        self,
        *,
        snapshot_id: str,
        result: dict[str, Any],
        enable_scip: bool,
    ) -> dict[str, Any]:
        snapshot = self.snapshot_store.load_snapshot(snapshot_id)
        if snapshot is None:
            return result
        metadata = self._snapshot_layer_metadata(snapshot)
        changed = False
        layers = metadata.get("pipeline_layers")
        if not layers:
            layers = self._default_pipeline_layers(enable_scip=enable_scip)
            metadata["pipeline_layers"] = layers
            changed = True
        metrics = metadata.get("pipeline_metrics")
        if not metrics:
            metrics = {
                "never_silent_fallback": True,
                "degraded": result.get("status") == "degraded",
                "warning_count": len(result.get("warnings", [])),
                "layer_statuses": {layer["name"]: layer["status"] for layer in layers},
            }
            metadata["pipeline_metrics"] = metrics
            changed = True

        if (
            "scip_artifact_ref" not in result
            and metadata.get("scip_artifact_ref") is not None
        ):
            result["scip_artifact_ref"] = metadata.get("scip_artifact_ref")
        if "scip_artifact_refs" not in result:
            if metadata.get("scip_artifact_refs") is not None:
                result["scip_artifact_refs"] = metadata.get("scip_artifact_refs")
            elif metadata.get("scip_artifact_ref") is not None:
                result["scip_artifact_refs"] = [metadata.get("scip_artifact_ref")]
        result["pipeline_layers"] = layers
        result["pipeline_metrics"] = metrics

        if changed:
            self.snapshot_store.update_snapshot_metadata(snapshot_id, metadata)
        return result

    @staticmethod
    def _with_materialization_metrics(
        metrics: dict[str, Any] | None,
        counters: MaterializationCounters,
    ) -> dict[str, Any]:
        merged = dict(metrics or {})
        merged.update(counters.as_metrics())
        return merged

    def _reset_embedding_metrics(self) -> None:
        reset_metrics = getattr(self.embedder, "reset_embedding_metrics", None)
        if callable(reset_metrics):
            reset_metrics()

    def _embedding_metrics_payload(self) -> dict[str, Any]:
        embedding_metrics = getattr(self.embedder, "embedding_metrics", None)
        if not callable(embedding_metrics):
            return {}
        payload = embedding_metrics()
        return dict(payload) if isinstance(payload, Mapping) else {}

    def _apply_semantic_resolvers(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        graph_context: CodeGraphBuilder | None,
        target_paths: set[str],
        warnings: list[str],
        budget: str = "changed_files",
    ) -> IRSnapshot:
        if not target_paths:
            return snapshot

        upgraded = snapshot
        registry = getattr(self, "semantic_resolver_registry", None)
        if registry is None:
            registry = build_default_semantic_resolver_registry(
                semantic_helper_runtime=getattr(self, "semantic_helper_runtime", None)
            )

        # Collect pending capabilities from unresolved relations so we can
        # capability-gate which resolvers actually run.
        pending_caps: set[str] = set()
        for relation in upgraded.relations:
            if relation.pending_capabilities:
                pending_caps |= relation.pending_capabilities

        # Use capability-gated selection when there are pending capabilities;
        # otherwise run all applicable resolvers (initial index path).
        if pending_caps:
            resolvers = registry.applicable_for_capabilities(
                snapshot=upgraded,
                elements=elements,
                target_paths=target_paths,
                required_capabilities=frozenset(pending_caps),
            )
        else:
            resolvers = registry.applicable(
                snapshot=upgraded,
                elements=elements,
                target_paths=target_paths,
            )

        semantic_graph_context = cast(SemanticGraphContext | None, graph_context)
        for resolver in resolvers:
            try:
                patch = resolver.resolve(
                    snapshot=upgraded,
                    elements=elements,
                    target_paths=target_paths,
                    graph_context=semantic_graph_context,
                )
            except Exception as exc:
                warnings.append(f"{resolver.language}_resolver_failed: {exc}")
                continue
            warnings.extend(patch.warnings)
            upgraded = apply_resolution_patch(upgraded, patch)
        return upgraded

    # ------------------------------------------------------------------
    # Incremental extraction planning
    # ------------------------------------------------------------------

    def _manifest_path_for_artifact(self, artifact_key: str) -> str:
        return os.path.join(
            self.vector_store.persist_dir, f"{artifact_key}_manifest.json"
        )

    def _metadata_path_for_artifact(self, artifact_key: str) -> str:
        return os.path.join(
            self.vector_store.persist_dir, f"{artifact_key}_metadata.pkl"
        )

    def _scan_files_for_pipeline(self) -> list[dict[str, Any]]:
        preloaded_inventory = getattr(self.loader, "_preloaded_file_inventory", None)
        preloaded_repo_path = getattr(
            self.loader, "_preloaded_file_inventory_repo_path", None
        )
        repo_path = getattr(self.loader, "repo_path", None)
        source_inventory_reused = (
            preloaded_inventory is not None
            and isinstance(preloaded_repo_path, str)
            and isinstance(repo_path, str)
            and os.path.abspath(repo_path) == os.path.abspath(preloaded_repo_path)
        )
        scan_inventory = getattr(self.loader, "scan_file_inventory", None)
        if callable(scan_inventory):
            try:
                inventory = scan_inventory(include_fingerprints=True)
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
            else:
                if isinstance(inventory, FileInventory):
                    metrics = inventory.metrics()
                    self._last_file_inventory_metrics = metrics
                    self._last_repository_inventory_metrics = (
                        self._repository_inventory_cache_metrics(
                            metrics,
                            reused_source_inventory=source_inventory_reused,
                        )
                    )
                    return inventory.to_file_info_list()
                metrics = getattr(inventory, "metrics", None)
                to_file_info_list = getattr(inventory, "to_file_info_list", None)
                if callable(metrics) and callable(to_file_info_list):
                    metrics_payload = metrics()
                    self._last_file_inventory_metrics = (
                        dict(metrics_payload)
                        if isinstance(metrics_payload, Mapping)
                        else {}
                    )
                    self._last_repository_inventory_metrics = (
                        self._repository_inventory_cache_metrics(
                            self._last_file_inventory_metrics,
                            reused_source_inventory=source_inventory_reused,
                        )
                    )
                    return cast(list[dict[str, Any]], to_file_info_list())
        try:
            files = self.loader.scan_files(include_fingerprints=True)
        except TypeError as exc:
            if "unexpected keyword" not in str(exc):
                raise
            # Some tests and legacy integrations replace scan_files with a
            # zero-argument callable. The planner still computes missing
            # fingerprints at use sites in that compatibility path.
            files = self.loader.scan_files()
        self._last_file_inventory_metrics = self._derive_file_inventory_metrics(files)
        self._last_repository_inventory_metrics = (
            self._repository_inventory_cache_metrics(
                self._last_file_inventory_metrics,
                from_legacy_scan=True,
            )
        )
        return files

    def _repository_info_for_pipeline(
        self, current_files: list[dict[str, Any]]
    ) -> dict[str, Any]:
        try:
            return self.loader.get_repository_info(files=current_files)
        except TypeError as exc:
            if "unexpected keyword" not in str(exc):
                raise
            return self.loader.get_repository_info()

    @staticmethod
    def _derive_file_inventory_metrics(
        current_files: Sequence[Mapping[str, Any]] | None,
    ) -> dict[str, Any]:
        files = list(current_files or [])
        total_size_bytes = sum(int(file.get("size") or 0) for file in files)
        git_blob_count = sum(1 for file in files if file.get("git_blob_oid"))
        content_hash_count = sum(
            1
            for file in files
            if file.get("content_hash") and not file.get("git_blob_oid")
        )
        hashed_bytes = sum(
            int(file.get("size") or 0)
            for file in files
            if file.get("content_hash") and not file.get("git_blob_oid")
        )
        return {
            "file_count": len(files),
            "total_size_bytes": total_size_bytes,
            "scanned_bytes": total_size_bytes,
            "hashed_bytes": hashed_bytes,
            "git_blob_oid_count": git_blob_count,
            "content_hash_count": content_hash_count,
            "fingerprinted_file_count": git_blob_count + content_hash_count,
            "supported_tool_eligible_count": sum(
                1 for file in files if file.get("supported_tool_eligible")
            ),
        }

    def _file_inventory_metrics_payload(
        self,
        current_files: Sequence[Mapping[str, Any]] | None,
    ) -> dict[str, Any]:
        metrics = dict(self._last_file_inventory_metrics or {})
        if metrics:
            return metrics
        return self._derive_file_inventory_metrics(current_files)

    @staticmethod
    def _repository_inventory_cache_metrics(
        inventory_metrics: Mapping[str, Any],
        *,
        from_legacy_scan: bool = False,
        reused_source_inventory: bool = False,
    ) -> dict[str, Any]:
        copied_from_source = bool(reused_source_inventory)
        return {
            "hit_count": int(copied_from_source),
            "miss_count": 0 if copied_from_source else 1,
            "reused": copied_from_source,
            "source": "source_inventory" if copied_from_source else "repo_scan",
            "legacy_scan": bool(from_legacy_scan),
        }

    @staticmethod
    def _int_metric(payload: Mapping[str, Any], key: str) -> int:
        try:
            return int(payload.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _artifact_handle_cache_metrics(self) -> dict[str, Any]:
        return {
            "hit_count": int(getattr(self, "_artifact_handle_cache_hits", 0) or 0),
            "miss_count": int(getattr(self, "_artifact_handle_cache_misses", 0) or 0),
            "size": len(getattr(self, "_artifact_handle_cache", {}) or {}),
            "limit": self._artifact_handle_cache_limit(),
        }

    def _increment_artifact_handle_cache_hit(self) -> None:
        self._artifact_handle_cache_hits = (
            int(getattr(self, "_artifact_handle_cache_hits", 0) or 0) + 1
        )

    def _increment_artifact_handle_cache_miss(self) -> None:
        self._artifact_handle_cache_misses = (
            int(getattr(self, "_artifact_handle_cache_misses", 0) or 0) + 1
        )

    def _parse_cache_metrics_payload(
        self,
        incremental_plan: Mapping[str, Any] | None,
        *,
        current_files: Sequence[Mapping[str, Any]] | None = None,
        reused_existing_snapshot: bool = False,
    ) -> dict[str, Any]:
        current_file_count = len(list(current_files or []))
        if incremental_plan is None:
            return {
                "hit_count": current_file_count if reused_existing_snapshot else 0,
                "miss_count": 0 if reused_existing_snapshot else current_file_count,
                "reused_files": current_file_count if reused_existing_snapshot else 0,
            }
        reused_files = self._int_metric(incremental_plan, "ast_ir_reused_files")
        rebuilt_elements = self._int_metric(incremental_plan, "ast_ir_rebuilt_elements")
        changed_file_count = len(
            [path for path in incremental_plan.get("changed_paths", []) if path]
        )
        parsed_reuse = incremental_plan.get("parsed_artifact_reuse")
        parsed_artifact_hits = (
            self._int_metric(parsed_reuse, "reused_files")
            if isinstance(parsed_reuse, Mapping)
            else 0
        )
        parsed_artifact_misses = (
            self._int_metric(parsed_reuse, "missed_files")
            if isinstance(parsed_reuse, Mapping)
            else changed_file_count
        )
        return {
            "hit_count": reused_files + parsed_artifact_hits,
            "miss_count": parsed_artifact_misses,
            "reused_files": reused_files,
            "rebuilt_elements": rebuilt_elements,
        }

    @classmethod
    def _embedding_cache_metrics_payload(
        cls,
        embedding_metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "hit_count": cls._int_metric(embedding_metrics, "cache_hit_count")
            + cls._int_metric(embedding_metrics, "reuse_index_hit_count"),
            "miss_count": cls._int_metric(embedding_metrics, "cache_miss_count"),
            "write_count": cls._int_metric(embedding_metrics, "cache_write_count"),
            "reuse_index_hit_count": cls._int_metric(
                embedding_metrics, "reuse_index_hit_count"
            ),
        }

    @classmethod
    def _incremental_update_metrics_payload(
        cls,
        incremental_plan: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if incremental_plan is None:
            return {
                "planned": False,
                "added": 0,
                "modified": 0,
                "removed": 0,
                "unchanged": 0,
                "changed": 0,
                "reused_elements": 0,
                "reindexed_elements": 0,
            }
        added = cls._int_metric(incremental_plan, "added")
        modified = cls._int_metric(incremental_plan, "modified")
        removed = cls._int_metric(incremental_plan, "removed")
        return {
            "planned": True,
            "added": added,
            "modified": modified,
            "removed": removed,
            "unchanged": cls._int_metric(incremental_plan, "unchanged"),
            "changed": added + modified + removed,
            "reused_elements": cls._int_metric(incremental_plan, "reused_elements"),
            "reindexed_elements": cls._int_metric(
                incremental_plan, "reindexed_elements"
            ),
        }

    def _attach_cache_update_metrics(
        self,
        metrics: dict[str, Any],
        *,
        embedding_metrics: Mapping[str, Any] | None = None,
        incremental_plan: Mapping[str, Any] | None = None,
        current_files: Sequence[Mapping[str, Any]] | None = None,
        reused_existing_snapshot: bool = False,
    ) -> dict[str, Any]:
        embedding_payload = (
            dict(embedding_metrics)
            if isinstance(embedding_metrics, Mapping)
            else self._embedding_metrics_payload()
        )
        metrics["cache_update"] = {
            "repository_inventory": dict(
                getattr(self, "_last_repository_inventory_metrics", {})
                or self._repository_inventory_cache_metrics(
                    self._file_inventory_metrics_payload(None)
                )
            ),
            "parse_cache": self._parse_cache_metrics_payload(
                incremental_plan,
                current_files=current_files,
                reused_existing_snapshot=reused_existing_snapshot,
            ),
            "embedding_cache": self._embedding_cache_metrics_payload(embedding_payload),
            "artifact_handle_cache": self._artifact_handle_cache_metrics(),
            "incremental_planner": self._incremental_update_metrics_payload(
                incremental_plan
            ),
        }
        return metrics

    def _pipeline_profiling_enabled(self) -> bool:
        indexing_config = self.config.get("indexing", {})
        performance_config = self.config.get("performance", {})
        return bool(
            self.config.get("profile_pipeline")
            or (
                isinstance(indexing_config, dict)
                and indexing_config.get("profile_pipeline")
            )
            or (
                isinstance(performance_config, dict)
                and performance_config.get("profile_pipeline")
            )
        )

    def _new_pipeline_profile(self) -> dict[str, Any] | None:
        if not self._pipeline_profiling_enabled():
            return None
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        return {
            "enabled": True,
            "stages": {},
            "io": {
                "temporary_directories": 0,
                "scoped_tool_copied_bytes": 0,
                "scoped_tool_copied_files": 0,
                "deleted_bytes": 0,
            },
            "snapshot_store_bytes_written": {},
        }

    @contextmanager
    def _profile_stage(self, profile: dict[str, Any] | None, name: str) -> Any:
        if profile is None:
            yield
            return
        if tracemalloc.is_tracing():
            tracemalloc.reset_peak()
        started = perf_counter()
        try:
            yield
        finally:
            duration_ms = round((perf_counter() - started) * 1000, 3)
            peak_bytes = 0
            if tracemalloc.is_tracing():
                _current, peak_bytes = tracemalloc.get_traced_memory()
            stages = cast(dict[str, dict[str, Any]], profile.setdefault("stages", {}))
            stage = stages.setdefault(
                name,
                {
                    "calls": 0,
                    "duration_ms": 0.0,
                    "allocation_peak_bytes": 0,
                },
            )
            stage["calls"] = int(stage.get("calls", 0)) + 1
            stage["duration_ms"] = round(
                float(stage.get("duration_ms", 0.0)) + duration_ms, 3
            )
            stage["allocation_peak_bytes"] = max(
                int(stage.get("allocation_peak_bytes", 0)),
                int(peak_bytes),
            )

    @staticmethod
    def _path_size_bytes(path: str | None) -> int:
        if not path or not os.path.exists(path):
            return 0
        if os.path.isfile(path):
            try:
                return os.path.getsize(path)
            except OSError:
                return 0
        total = 0
        for root, _dirs, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                try:
                    total += os.path.getsize(file_path)
                except OSError:
                    continue
        return total

    @contextmanager
    def _profile_store_surface(
        self,
        profile: dict[str, Any] | None,
        surface: str,
        *paths: str | None,
    ) -> Any:
        if profile is None:
            yield
            return
        before = sum(self._path_size_bytes(path) for path in paths)
        with self._profile_stage(profile, f"store:{surface}"):
            yield
        after = sum(self._path_size_bytes(path) for path in paths)
        bytes_written = max(0, after - before)
        store_bytes = cast(
            dict[str, int], profile.setdefault("snapshot_store_bytes_written", {})
        )
        store_bytes[surface] = store_bytes.get(surface, 0) + bytes_written

    def _profile_record_loader_io(self, profile: dict[str, Any] | None) -> None:
        if profile is None:
            return
        stats = getattr(self.loader, "last_load_stats", {}) or {}
        io = cast(dict[str, Any], profile.setdefault("io", {}))
        io["repository_copied_bytes"] = int(stats.get("copied_bytes", 0) or 0)
        io["repository_copied_files"] = int(stats.get("copied_files", 0) or 0)
        io["repository_hard_linked_bytes"] = int(stats.get("linked_bytes", 0) or 0)
        io["repository_hard_linked_files"] = int(stats.get("linked_files", 0) or 0)
        io["repository_copy_cache_hit"] = bool(stats.get("copy_cache_hit", False))
        io["repository_copy_cache_key"] = stats.get("copy_cache_key")

    def _profile_add_io(
        self, key: str, amount: int, *, profile: dict[str, Any] | None = None
    ) -> None:
        active = profile if profile is not None else self._active_pipeline_profile
        if active is None:
            return
        io = cast(dict[str, Any], active.setdefault("io", {}))
        io[key] = int(io.get(key, 0) or 0) + int(amount)

    def _profile_record_temp_dir(self, path: str) -> None:
        if self._active_pipeline_profile is None:
            return
        self._profile_add_io("temporary_directories", 1)
        temp_dirs = cast(
            list[str], self._active_pipeline_profile.setdefault("temporary_paths", [])
        )
        temp_dirs.append(path)

    @staticmethod
    def _attach_pipeline_profile(
        metrics: dict[str, Any], profile: dict[str, Any] | None
    ) -> dict[str, Any]:
        if profile is not None:
            metrics["profiling"] = profile
        return metrics

    def _file_fingerprint(
        self, abs_path: str, file_info: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | None:
        if file_info is not None:
            content_hash_value = file_info.get("content_hash")
            blob_oid_value = file_info.get("git_blob_oid") or file_info.get("blob_oid")
            if content_hash_value or blob_oid_value:
                return {
                    "mtime": float(file_info.get("mtime") or 0.0),
                    "size": int(file_info.get("size") or 0),
                    "content_hash": str(content_hash_value)
                    if content_hash_value
                    else None,
                    "blob_oid": str(blob_oid_value) if blob_oid_value else None,
                }
        try:
            stat = os.stat(abs_path)
        except OSError:
            return None
        content_hash = None
        try:
            content_hash = compute_file_hash(abs_path)
        except Exception as e:
            self.logger.warning(f"Failed to hash file for incremental manifest: {e}")
        if not content_hash:
            return None
        return {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "content_hash": content_hash,
            "blob_oid": content_hash,
        }

    def _embedding_fingerprint_payload(self) -> dict[str, Any]:
        embedding_fingerprint_record = getattr(
            self.embedder, "embedding_fingerprint_record", None
        )
        embedding_fingerprint = getattr(self.embedder, "embedding_fingerprint", None)
        if callable(embedding_fingerprint_record):
            fingerprint = embedding_fingerprint_record()
            to_payload = getattr(fingerprint, "to_payload", None)
            embedding_identity = (
                to_payload()
                if callable(to_payload)
                else embedding_fingerprint()
                if callable(embedding_fingerprint)
                else self._fallback_embedding_identity()
            )
        elif callable(embedding_fingerprint):
            embedding_identity = embedding_fingerprint()
        else:
            embedding_identity = self._fallback_embedding_identity()
        return (
            dict(cast(Mapping[str, Any], embedding_identity))
            if isinstance(embedding_identity, Mapping)
            else self._fallback_embedding_identity()
        )

    def _incremental_compatibility_payload(self) -> dict[str, Any]:
        embedding_identity = self._embedding_fingerprint_payload()
        return {
            "schema_version": 2,
            "embedding": embedding_identity,
            "indexing": {
                "levels": list(getattr(self.indexer, "levels", []) or []),
                "include_imports": getattr(self.indexer, "include_imports", None),
                "include_class_context": getattr(
                    self.indexer, "include_class_context", None
                ),
                "generate_repo_overview": getattr(
                    self.indexer, "generate_repo_overview", None
                ),
            },
            "parser": self.config.get("parser", {}),
        }

    def _fallback_embedding_identity(self) -> dict[str, Any]:
        return {
            "provider": getattr(self.embedder, "provider", None),
            "model": getattr(self.embedder, "model_name", None),
            "dimension": (
                getattr(self.embedder, "_configured_embedding_dim", None)
                or getattr(self.embedder, "_embedding_dim", None)
            ),
            "max_seq_length": getattr(self.embedder, "max_seq_length", None),
            "normalize": getattr(self.embedder, "normalize", None),
        }

    def _incremental_compatibility_hash(self) -> str:
        payload = json.dumps(
            self._incremental_compatibility_payload(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _manifest_is_incrementally_compatible(
        self, manifest: dict[str, Any], artifact_key: str
    ) -> bool:
        expected = self._incremental_compatibility_hash()
        actual = manifest.get("compatibility_hash")
        if actual == expected:
            return True
        self.logger.info(
            "incremental prefilter disabled for %s: compatibility mismatch",
            artifact_key,
        )
        return False

    def _manifest_has_required_fingerprints(
        self, manifest: dict[str, Any], artifact_key: str
    ) -> bool:
        files = manifest.get("files", {})
        if not isinstance(files, dict):
            self.logger.info(
                "incremental prefilter disabled for %s: missing file manifest",
                artifact_key,
            )
            return False
        missing = [
            str(path)
            for path, entry in cast(dict[str, Any], files).items()
            if not isinstance(entry, dict)
            or not (entry.get("blob_oid") or entry.get("content_hash"))
        ]
        if missing:
            self.logger.info(
                "incremental prefilter disabled for %s: %d files lack required "
                "fingerprints",
                artifact_key,
                len(missing),
            )
            return False
        return True

    def _file_info_by_relative_path(
        self, current_files: Sequence[Mapping[str, Any]] | None
    ) -> dict[str, Mapping[str, Any]]:
        if current_files is None:
            return {}
        return {
            normalize_path(str(file_info.get("relative_path") or "")): file_info
            for file_info in current_files
            if file_info.get("relative_path")
        }

    def _build_file_manifest(
        self,
        elements: list[CodeElement],
        repo_root: str,
        current_files: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "created_at": datetime.now().isoformat(),
            "embedding_fingerprint": self._embedding_fingerprint_payload(),
            "compatibility": self._incremental_compatibility_payload(),
            "compatibility_hash": self._incremental_compatibility_hash(),
            "files": {},
        }
        embedding_fingerprint = cast(dict[str, Any], manifest["embedding_fingerprint"])
        file_info_by_path = self._file_info_by_relative_path(current_files)
        interface_digests = self._interface_digests_for_elements(elements)

        for elem in elements:
            rel_path = normalize_path(elem.relative_path or elem.file_path)
            if not rel_path:
                continue
            if rel_path not in manifest["files"]:
                abs_path = os.path.join(repo_root, rel_path)
                fingerprint = self._file_fingerprint(
                    abs_path, file_info=file_info_by_path.get(rel_path)
                )
                if fingerprint is None:
                    manifest["files"][rel_path] = {
                        "mtime": 0.0,
                        "size": 0,
                        "content_hash": None,
                        "blob_oid": None,
                        "embedding_fingerprint": dict(embedding_fingerprint),
                        "interface_digest": interface_digests.get(rel_path),
                        "element_ids": [],
                    }
                else:
                    manifest["files"][rel_path] = {
                        **fingerprint,
                        "embedding_fingerprint": dict(embedding_fingerprint),
                        "interface_digest": interface_digests.get(rel_path),
                        "element_ids": [],
                    }
            if interface_digests.get(rel_path):
                manifest["files"][rel_path]["interface_digest"] = interface_digests[
                    rel_path
                ]
            manifest["files"][rel_path]["element_ids"].append(elem.id)

        return manifest

    def _build_file_manifest_delta(
        self,
        *,
        artifact_key: str,
        elements: Sequence[CodeElement],
        repo_root: str,
        current_files: Sequence[Mapping[str, Any]],
        incremental_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        manifest = self._build_file_manifest(
            list(elements),
            repo_root,
            current_files=current_files,
        )
        previous_key = str(incremental_plan.get("previous_artifact_key") or "")
        previous_manifest = self._load_file_manifest(previous_key)
        if previous_manifest is None:
            manifest["fallback_reason"] = "previous_file_manifest_missing"
            return manifest

        previous_files = cast(
            dict[str, dict[str, Any]], previous_manifest.get("files", {})
        )
        current_by_path = self._file_info_by_relative_path(current_files)
        unchanged_paths = [
            normalize_path(str(path))
            for path in incremental_plan.get("unchanged_paths", [])
            if path
        ]
        changed_paths = {
            normalize_path(str(path))
            for path in (
                list(incremental_plan.get("added_paths", []) or [])
                + list(incremental_plan.get("modified_paths", []) or [])
            )
            if path
        }
        removed_paths = {
            normalize_path(str(path))
            for path in incremental_plan.get("removed_paths", [])
            if path
        }
        files = cast(dict[str, Any], manifest["files"])
        for rel_path in unchanged_paths:
            previous_entry = previous_files.get(rel_path)
            if previous_entry is None or rel_path in removed_paths:
                continue
            entry = dict(previous_entry)
            current_info = current_by_path.get(rel_path)
            if current_info is not None:
                fingerprint = self._file_fingerprint(
                    str(current_info.get("path") or ""),
                    file_info=current_info,
                )
                if fingerprint is not None:
                    entry.update(fingerprint)
            files[rel_path] = entry

        for rel_path, file_info in current_by_path.items():
            if rel_path in files or rel_path in removed_paths:
                continue
            fingerprint = self._file_fingerprint(
                str(file_info.get("path") or ""),
                file_info=file_info,
            )
            files[rel_path] = {
                **(fingerprint or {}),
                "embedding_fingerprint": dict(
                    cast(dict[str, Any], manifest["embedding_fingerprint"])
                ),
                "element_ids": [],
            }
        for rel_path in changed_paths:
            files.setdefault(
                rel_path,
                {
                    "embedding_fingerprint": dict(
                        cast(dict[str, Any], manifest["embedding_fingerprint"])
                    ),
                    "interface_digest": None,
                    "element_ids": [],
                },
            )
        manifest["delta"] = {
            "artifact_key": artifact_key,
            "previous_artifact_key": previous_key,
            "reused_files": len(unchanged_paths),
            "changed_files": len(changed_paths),
            "removed_files": len(removed_paths),
        }
        return manifest

    def _incremental_graph_delta_context(
        self,
        *,
        changed_elements: Sequence[CodeElement],
        incremental_plan: Mapping[str, Any],
        current_files: Sequence[Mapping[str, Any]],
        repo_name: str,
        repo_url: str,
    ) -> IncrementalGraphDeltaContext:
        previous_key = str(incremental_plan.get("previous_artifact_key") or "")
        changed_path_keys = {
            normalize_path(str(path))
            for path in incremental_plan.get("changed_paths", [])
            if path
        }
        removed_path_keys = {
            normalize_path(str(path))
            for path in incremental_plan.get("removed_paths", [])
            if path
        }
        unchanged_path_keys = {
            normalize_path(str(path))
            for path in incremental_plan.get("unchanged_paths", [])
            if path
        }
        affected_path_keys = set(changed_path_keys) | set(removed_path_keys)
        affected_loader = getattr(
            self.graph_artifact_store, "affected_path_keys_for_delta", None
        )
        if callable(affected_loader) and previous_key:
            try:
                affected_path_keys = {
                    normalize_path(str(path))
                    for path in cast(
                        set[str],
                        affected_loader(
                            previous_key,
                            changed_path_keys=changed_path_keys,
                            removed_path_keys=removed_path_keys,
                        ),
                    )
                    if path
                }
            except Exception as exc:
                self.logger.warning(
                    "graph affected-path detection failed for %s: %s",
                    previous_key,
                    exc,
                )

        affected_unchanged_path_keys = sorted(
            (affected_path_keys & unchanged_path_keys) - removed_path_keys
        )
        graph_elements = list(changed_elements)
        loaded_affected_path_keys: set[str] = set()
        if affected_unchanged_path_keys:
            reused_elements, hit_paths, _reuse_stats = (
                self._reuse_parsed_element_content_artifacts(
                    repo_name=repo_name,
                    current_files=current_files,
                    paths=affected_unchanged_path_keys,
                )
            )
            graph_elements.extend(reused_elements)
            loaded_affected_path_keys.update(hit_paths)

            missing_paths = [
                path
                for path in affected_unchanged_path_keys
                if path not in loaded_affected_path_keys
            ]
            previous_manifest = self._load_file_manifest(previous_key)
            if previous_manifest is not None and missing_paths:
                existing_metadata = self._load_existing_metadata(previous_key)
                unchanged_metadata, _expected_ids = self._collect_unchanged_metadata(
                    previous_manifest,
                    missing_paths,
                    existing_metadata,
                )
                current_lookup = self._file_info_by_relative_path(current_files)
                for meta in unchanged_metadata:
                    rel_path = normalize_path(
                        str(meta.get("relative_path") or meta.get("file_path") or "")
                    )
                    if not rel_path or rel_path in loaded_affected_path_keys:
                        continue
                    file_info = current_lookup.get(rel_path)
                    elem = self._reconstruct_code_element(
                        meta,
                        file_info=dict(file_info)
                        if isinstance(file_info, Mapping)
                        else None,
                        repo_name=repo_name,
                        repo_url=repo_url,
                    )
                    if elem is not None:
                        graph_elements.append(elem)
                        loaded_affected_path_keys.add(rel_path)

        missing_affected_path_keys = (
            set(affected_unchanged_path_keys) - loaded_affected_path_keys
        )
        reusable_path_keys = unchanged_path_keys - set(affected_unchanged_path_keys)
        reusable_path_keys -= removed_path_keys
        return IncrementalGraphDeltaContext(
            elements=graph_elements,
            affected_path_keys=affected_path_keys,
            reusable_path_keys=reusable_path_keys,
            loaded_affected_path_keys=loaded_affected_path_keys,
            missing_affected_path_keys=missing_affected_path_keys,
        )

    def _build_incremental_graph_resolver_index(
        self,
        *,
        graph_elements: Sequence[CodeElement],
        incremental_plan: Mapping[str, Any] | None,
        excluded_path_keys: set[str],
        repo_root: str,
    ) -> GlobalIndexBuilder:
        gib = GlobalIndexBuilder(self.config)
        previous_key = (
            str(incremental_plan.get("previous_artifact_key") or "")
            if incremental_plan is not None
            else ""
        )
        resolver_loader = getattr(
            self.graph_artifact_store, "load_resolver_index", None
        )
        if callable(resolver_loader) and previous_key:
            try:
                resolver_payload = resolver_loader(
                    previous_key,
                    excluded_path_keys=excluded_path_keys,
                    repo_root=repo_root,
                )
            except Exception as exc:
                self.logger.warning(
                    "graph resolver-index load failed for %s: %s",
                    previous_key,
                    exc,
                )
                resolver_payload = None
            if isinstance(resolver_payload, Mapping):
                file_map = resolver_payload.get("file_map")
                module_map = resolver_payload.get("module_map")
                export_map = resolver_payload.get("export_map")
                if isinstance(file_map, Mapping):
                    gib.file_map.update(
                        {str(key): str(value) for key, value in file_map.items()}
                    )
                if isinstance(module_map, Mapping):
                    gib.module_map.update(
                        {str(key): str(value) for key, value in module_map.items()}
                    )
                if isinstance(export_map, Mapping):
                    for module_path, symbols in export_map.items():
                        if not isinstance(symbols, Mapping):
                            continue
                        gib.export_map[str(module_path)] = {
                            str(name): str(symbol_id)
                            for name, symbol_id in symbols.items()
                        }

        overlay = GlobalIndexBuilder(self.config)
        overlay.build_maps(list(graph_elements), repo_root)
        gib.file_map.update(overlay.file_map)
        gib.module_map.update(overlay.module_map)
        for module_path, symbols in overlay.export_map.items():
            gib.export_map.setdefault(module_path, {}).update(symbols)
        gib.stats = {
            "files_processed": len(gib.file_map),
            "modules_created": len(gib.module_map),
            "symbols_exported": sum(
                len(symbols) for symbols in gib.export_map.values()
            ),
            "errors": 0,
        }
        return gib

    def _reuse_file_ir_content_artifacts(
        self,
        *,
        repo_name: str,
        snapshot_id: str,
        current_files: Sequence[Mapping[str, Any]],
        incremental_plan: Mapping[str, Any],
        excluded_paths: Sequence[str],
    ) -> tuple[list[dict[str, Any]], dict[str, int | str]]:
        store = self.file_artifact_store
        if store is None:
            return [], {"mode": "unavailable", "reused_shards": 0}
        excluded_path_set = {
            normalize_path(str(path)) for path in excluded_paths if path
        }
        reusable_paths = sorted(
            {
                normalize_path(str(path))
                for path in incremental_plan.get("unchanged_paths", [])
                if path and normalize_path(str(path)) not in excluded_path_set
            }
        )
        if not reusable_paths:
            return [], {"mode": "content_addressed", "requested_shards": 0}
        records = store.list_file_ir_records_for_file_infos(
            repo_name=repo_name,
            file_infos=current_files,
            paths=reusable_paths,
        )
        return (
            [
                store.file_ir_payload_from_record(record, snapshot_id=snapshot_id)
                for record in records
            ],
            {
                "mode": "content_addressed",
                "requested_shards": len(reusable_paths),
                "reused_shards": len(records),
            },
        )

    def _reuse_parsed_element_content_artifacts(
        self,
        *,
        repo_name: str,
        current_files: Sequence[Mapping[str, Any]],
        paths: Sequence[str],
    ) -> tuple[list[CodeElement], set[str], dict[str, int | str]]:
        requested_paths = sorted({normalize_path(str(path)) for path in paths if path})
        if not requested_paths:
            return (
                [],
                set(),
                {
                    "mode": "content_addressed",
                    "requested_files": 0,
                    "reused_files": 0,
                    "missed_files": 0,
                    "reused_elements": 0,
                },
            )
        store = self.file_artifact_store
        if store is None:
            return (
                [],
                set(),
                {
                    "mode": "unavailable",
                    "requested_files": len(requested_paths),
                    "reused_files": 0,
                    "missed_files": len(requested_paths),
                    "reused_elements": 0,
                },
            )
        records = store.list_parsed_element_records_for_file_infos(
            repo_name=repo_name,
            file_infos=current_files,
            paths=requested_paths,
        )
        elements: list[CodeElement] = []
        reused_paths: set[str] = set()
        for record in records:
            payload = store.parsed_elements_payload_from_record(record)
            raw_elements = payload.get("elements")
            if not isinstance(raw_elements, list):
                continue
            decoded_for_path = 0
            for raw_element in raw_elements:
                try:
                    elements.append(deserialize_code_element(raw_element))
                except (TypeError, ValueError):
                    continue
                decoded_for_path += 1
            if decoded_for_path:
                reused_paths.add(record.relative_path)
        return (
            elements,
            reused_paths,
            {
                "mode": "content_addressed",
                "requested_files": len(requested_paths),
                "reused_files": len(reused_paths),
                "missed_files": len(set(requested_paths) - reused_paths),
                "reused_elements": len(elements),
            },
        )

    def _publish_parsed_element_content_artifacts(
        self,
        *,
        repo_name: str,
        elements: Sequence[CodeElement],
        current_files: Sequence[Mapping[str, Any]],
        paths: Sequence[str] | None = None,
    ) -> dict[str, int | str]:
        store = self.file_artifact_store
        if store is None:
            return {"mode": "unavailable", "written_records": 0}
        path_filter = (
            {normalize_path(str(path)) for path in paths if path}
            if paths is not None
            else None
        )
        payloads = [
            self._code_element_like_payload(element)
            for element in elements
            if path_filter is None
            or normalize_path(element.relative_path or element.file_path) in path_filter
        ]
        summary, _records = store.upsert_parsed_elements(
            repo_name=repo_name,
            elements=payloads,
            file_infos=current_files,
            paths=paths,
        )
        summary["candidate_elements"] = len(payloads)
        return summary

    def _publish_file_ir_content_artifacts(
        self,
        *,
        repo_name: str,
        shards: Sequence[Mapping[str, Any]],
        current_files: Sequence[Mapping[str, Any]],
        paths: Sequence[str] | None = None,
    ) -> dict[str, int | str]:
        store = self.file_artifact_store
        if store is None:
            return {"mode": "unavailable", "written_records": 0}
        path_filter = (
            {normalize_path(str(path)) for path in paths if path}
            if paths is not None
            else None
        )
        publish_shards = [
            shard
            for shard in shards
            if path_filter is None
            or normalize_path(
                str(shard.get("relative_path") or shard.get("path") or "")
            )
            in path_filter
        ]
        summary, _records = store.upsert_file_ir_shards(
            repo_name=repo_name,
            shards=publish_shards,
            file_infos=current_files,
        )
        summary["candidate_shards"] = len(publish_shards)
        return summary

    def _publish_embedding_ref_content_artifacts(
        self,
        *,
        repo_name: str,
        rows: Sequence[Mapping[str, Any]],
        current_files: Sequence[Mapping[str, Any]],
        paths: Sequence[str] | None = None,
    ) -> dict[str, int | str]:
        store = self.file_artifact_store
        if store is None:
            return {"mode": "unavailable", "written_records": 0}
        summary, _records = store.upsert_embedding_refs(
            repo_name=repo_name,
            rows=rows,
            file_infos=current_files,
            paths=paths,
        )
        return summary

    def _publish_semantic_fact_content_artifacts(
        self,
        *,
        repo_name: str,
        shards: Sequence[Mapping[str, Any]],
        current_files: Sequence[Mapping[str, Any]],
        paths: Sequence[str] | None = None,
    ) -> dict[str, int | str]:
        store = self.file_artifact_store
        if store is None:
            return {"mode": "unavailable", "written_records": 0}
        summary, _records = store.upsert_semantic_fact_shards(
            repo_name=repo_name,
            shards=shards,
            file_infos=current_files,
            paths=paths,
        )
        return summary

    def _save_file_manifest(self, artifact_key: str, manifest: dict[str, Any]) -> None:
        with open(
            self._manifest_path_for_artifact(artifact_key), "w", encoding="utf-8"
        ) as f:
            increment_materialization_boundary(
                BOUNDARY_JSON_ENCODE,
                items=len(cast(dict[str, Any], manifest.get("files", {}))),
            )
            json.dump(manifest, f, indent=2)

    def _save_relational_facts_for_index(
        self,
        snapshot: IRSnapshot,
        incremental_plan: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if incremental_plan is not None:
            previous_snapshot_id = str(
                incremental_plan.get("previous_snapshot_id") or ""
            )
            semantic_widened = int(
                incremental_plan.get("semantic_frontier_widened", 0) or 0
            )
            changed_paths = [
                normalize_path(str(path))
                for path in (
                    list(incremental_plan.get("added_paths", []) or [])
                    + list(incremental_plan.get("modified_paths", []) or [])
                )
                if path
            ]
            removed_paths = [
                normalize_path(str(path))
                for path in incremental_plan.get("removed_paths", []) or []
                if path
            ]
            save_delta = getattr(
                self.snapshot_store, "save_relational_facts_delta", None
            )
            if (
                previous_snapshot_id
                and semantic_widened == 0
                and callable(save_delta)
                and save_delta(
                    snapshot,
                    previous_snapshot_id=previous_snapshot_id,
                    changed_paths=changed_paths,
                    removed_paths=removed_paths,
                )
            ):
                return {
                    "mode": "delta",
                    "previous_snapshot_id": previous_snapshot_id,
                    "changed_path_count": len(set(changed_paths)),
                    "removed_path_count": len(set(removed_paths)),
                }
            if semantic_widened:
                self.snapshot_store.save_relational_facts(snapshot)
                return {
                    "mode": "full",
                    "fallback_reason": "semantic_frontier_widened",
                    "previous_snapshot_id": previous_snapshot_id,
                    "changed_path_count": len(set(changed_paths)),
                    "removed_path_count": len(set(removed_paths)),
                }
            if not previous_snapshot_id:
                self.snapshot_store.save_relational_facts(snapshot)
                return {"mode": "full", "fallback_reason": "missing_previous_snapshot"}
            if not callable(save_delta):
                self.snapshot_store.save_relational_facts(snapshot)
                return {"mode": "full", "fallback_reason": "delta_api_unavailable"}

        self.snapshot_store.save_relational_facts(snapshot)
        return {"mode": "full"}

    @staticmethod
    def _incremental_delta_paths(
        incremental_plan: dict[str, Any] | None,
    ) -> tuple[list[str], list[str]]:
        if incremental_plan is None:
            return [], []
        changed_paths = [
            normalize_path(str(path))
            for path in (
                list(incremental_plan.get("added_paths", []) or [])
                + list(incremental_plan.get("modified_paths", []) or [])
            )
            if path
        ]
        removed_paths = [
            normalize_path(str(path))
            for path in incremental_plan.get("removed_paths", []) or []
            if path
        ]
        return changed_paths, removed_paths

    def _save_ir_graphs_for_index(
        self,
        snapshot: IRSnapshot,
        incremental_plan: dict[str, Any] | None,
    ) -> tuple[Any, dict[str, Any]]:
        previous_snapshot_id = (
            str(incremental_plan.get("previous_snapshot_id") or "")
            if incremental_plan is not None
            else ""
        )
        changed_paths, removed_paths = self._incremental_delta_paths(incremental_plan)
        previous_graphs: IRGraphs | None = None
        fallback_reason = "missing_previous_snapshot"
        if previous_snapshot_id:
            loaded_graphs = self.snapshot_store.load_ir_graphs(previous_snapshot_id)
            if isinstance(loaded_graphs, IRGraphs):
                previous_graphs = loaded_graphs
            else:
                fallback_reason = "previous_ir_graphs_unavailable_or_legacy"

        if previous_snapshot_id and previous_graphs is not None:
            ir_graphs, graph_delta = self.ir_graph_builder.build_graph_delta(
                snapshot,
                previous_graphs=previous_graphs,
                changed_paths=changed_paths,
                removed_paths=removed_paths,
            )
        else:
            ir_graphs = self.ir_graph_builder.build_graphs(snapshot)
            graph_delta = {
                "mode": "full",
                "fallback_reason": fallback_reason,
                "changed_path_count": len(set(changed_paths)),
                "removed_path_count": len(set(removed_paths)),
                "reusable_graphs": [],
                "rebuilt_graphs": [],
            }

        save_stats: dict[str, Any] = {}
        save_delta = getattr(self.snapshot_store, "save_ir_graphs_delta", None)
        if previous_snapshot_id and callable(save_delta):
            delta_result = save_delta(
                snapshot.snapshot_id,
                ir_graphs,
                previous_snapshot_id=previous_snapshot_id,
                reusable_graphs=cast(
                    Sequence[str], graph_delta.get("reusable_graphs", [])
                ),
            )
            if (
                isinstance(delta_result, tuple)
                and len(delta_result) == 2
                and isinstance(delta_result[1], Mapping)
            ):
                save_stats.update(dict(delta_result[1]))
        else:
            self.snapshot_store.save_ir_graphs(snapshot.snapshot_id, ir_graphs)
            save_stats.update(
                {
                    "ir_graph_shards_reused": 0,
                    "ir_graph_shards_written": 0,
                    "fallback_full_rewrite": int(bool(previous_snapshot_id)),
                }
            )

        metrics = {
            **graph_delta,
            **save_stats,
            "previous_snapshot_id": previous_snapshot_id or None,
            "changed_path_count": len(set(changed_paths)),
            "removed_path_count": len(set(removed_paths)),
        }
        pipeline_metrics = cast(
            dict[str, Any], snapshot.metadata.setdefault("pipeline_metrics", {})
        )
        pipeline_metrics["ir_graph_delta"] = metrics
        return ir_graphs, metrics

    def _load_file_manifest(self, artifact_key: str) -> dict[str, Any] | None:
        manifest_path = self._manifest_path_for_artifact(artifact_key)
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, encoding="utf-8") as f:
                increment_materialization_boundary(BOUNDARY_JSON_DECODE)
                return cast(dict[str, Any], json.load(f))
        except Exception as e:
            self.logger.warning(
                f"Failed to load incremental manifest for '{artifact_key}': {e}"
            )
            return None

    def _load_existing_metadata(self, artifact_key: str) -> list[dict[str, Any]]:
        load_metadata_payload = getattr(
            self.vector_store, "load_metadata_payload", None
        )
        if callable(load_metadata_payload):
            try:
                data = load_metadata_payload(artifact_key)
            except Exception as e:
                self.logger.warning(
                    f"Failed to load incremental metadata for '{artifact_key}': {e}"
                )
            else:
                metadata = data.get("metadata", []) if isinstance(data, dict) else []
                if isinstance(metadata, list):
                    return cast(list[dict[str, Any]], metadata)
        metadata_path = self._metadata_path_for_artifact(artifact_key)
        if not os.path.exists(metadata_path):
            return []
        try:
            with open(metadata_path, "rb") as f:
                increment_materialization_boundary(BOUNDARY_PICKLE_LOAD)
                data = pickle.load(f)
            metadata = data.get("metadata", [])
            if isinstance(metadata, list):
                return cast(list[dict[str, Any]], metadata)
        except Exception as e:
            self.logger.warning(
                f"Failed to load incremental metadata for '{artifact_key}': {e}"
            )
        return []

    def _detect_file_changes(
        self,
        manifest: dict[str, Any],
        current_files: list[dict[str, Any]],
    ) -> dict[str, Any]:
        manifest_files = cast(dict[str, dict[str, Any]], manifest.get("files", {}))
        current_lookup: dict[str, dict[str, Any]] = {}

        for file_info in current_files:
            rel_path = str(file_info.get("relative_path") or "")
            abs_path = str(file_info.get("path") or "")
            if not rel_path or not abs_path:
                continue
            rel_path = normalize_path(rel_path)
            fingerprint = self._file_fingerprint(abs_path, file_info=file_info)
            if fingerprint is None:
                current_lookup[rel_path] = {
                    "fingerprint_missing": True,
                    "file_info": file_info,
                }
                continue
            current_lookup[rel_path] = {
                **fingerprint,
                "file_info": file_info,
            }

        added: list[str] = []
        modified: list[str] = []
        deleted: list[str] = []
        unchanged: list[str] = []
        missing_fingerprints: list[str] = []

        for rel_path, info in current_lookup.items():
            if info.get("fingerprint_missing"):
                missing_fingerprints.append(rel_path)
                continue
            saved = manifest_files.get(rel_path)
            if saved is None:
                added.append(rel_path)
                continue
            if saved.get("blob_oid") and info.get("blob_oid"):
                saved_identity = saved.get("blob_oid")
                current_identity = info.get("blob_oid")
            elif saved.get("content_hash") and info.get("content_hash"):
                saved_identity = saved.get("content_hash")
                current_identity = info.get("content_hash")
            else:
                missing_fingerprints.append(rel_path)
                continue
            changed = current_identity != saved_identity
            if changed:
                modified.append(rel_path)
            else:
                unchanged.append(rel_path)

        for rel_path in manifest_files:
            if rel_path not in current_lookup:
                deleted.append(rel_path)

        return {
            "added": added,
            "modified": modified,
            "deleted": deleted,
            "unchanged": unchanged,
            "current_lookup": current_lookup,
            "fingerprints_complete": not missing_fingerprints,
            "missing_fingerprint_paths": sorted(missing_fingerprints),
        }

    def _collect_unchanged_metadata(
        self,
        manifest: dict[str, Any],
        unchanged_files: list[str],
        existing_metadata: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], set[str]]:
        unchanged_element_ids: set[str] = set()
        manifest_files = cast(dict[str, dict[str, Any]], manifest.get("files", {}))
        for rel_path in unchanged_files:
            file_entry = manifest_files.get(rel_path, {})
            for elem_id in cast(list[str], file_entry.get("element_ids", [])):
                unchanged_element_ids.add(elem_id)
        return (
            [
                meta
                for meta in existing_metadata
                if str(meta.get("id")) in unchanged_element_ids
            ],
            unchanged_element_ids,
        )

    @staticmethod
    def _metadata_by_stable_unit_id(
        metadata_rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for row in metadata_rows:
            meta = row.get("metadata", {}) or {}
            stable_unit_id = meta.get("stable_unit_id") or row.get("stable_unit_id")
            if stable_unit_id:
                indexed[str(stable_unit_id)] = row
        return indexed

    @staticmethod
    def _metadata_payload(value: Any) -> dict[str, Any]:
        return (
            dict(cast(dict[str, Any], value or {})) if isinstance(value, dict) else {}
        )

    @staticmethod
    def _interface_payload_from_element(element: Any) -> dict[str, Any] | None:
        rel_path = normalize_path(
            str(
                getattr(element, "relative_path", None)
                or getattr(element, "file_path", None)
                or ""
            )
        )
        if not rel_path:
            return None
        metadata = IndexPipeline._metadata_payload(getattr(element, "metadata", None))
        stable_unit_id = metadata.get("stable_unit_id") or getattr(element, "id", None)
        return {
            "path": rel_path,
            "stable_unit_id": str(stable_unit_id or ""),
            "kind": str(getattr(element, "type", "") or ""),
            "name": str(getattr(element, "name", "") or ""),
            "signature_hash": metadata.get("signature_hash"),
            "api_surface_hash": metadata.get("api_surface_hash"),
            "edge_surface_hash": metadata.get("edge_surface_hash"),
        }

    @staticmethod
    def _interface_payload_from_metadata(
        row: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        rel_path = normalize_path(
            str(row.get("relative_path") or row.get("file_path") or "")
        )
        if not rel_path:
            return None
        metadata = IndexPipeline._metadata_payload(row.get("metadata"))
        stable_unit_id = metadata.get("stable_unit_id") or row.get("id")
        return {
            "path": rel_path,
            "stable_unit_id": str(stable_unit_id or ""),
            "kind": str(row.get("type") or ""),
            "name": str(row.get("name") or ""),
            "signature_hash": metadata.get("signature_hash"),
            "api_surface_hash": metadata.get("api_surface_hash"),
            "edge_surface_hash": metadata.get("edge_surface_hash"),
        }

    @staticmethod
    def _interface_digest(payloads: Sequence[Mapping[str, Any]]) -> str | None:
        if not payloads:
            return None
        normalized = sorted(
            [dict(payload) for payload in payloads],
            key=lambda payload: (
                str(payload.get("path") or ""),
                str(payload.get("stable_unit_id") or ""),
                str(payload.get("kind") or ""),
                str(payload.get("name") or ""),
            ),
        )
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        return f"iface:{digest}"

    @classmethod
    def _interface_digests_for_elements(cls, elements: Sequence[Any]) -> dict[str, str]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for element in elements:
            payload = cls._interface_payload_from_element(element)
            if payload is None:
                continue
            grouped.setdefault(str(payload["path"]), []).append(payload)
        return {
            path: digest
            for path, payloads in grouped.items()
            if (digest := cls._interface_digest(payloads)) is not None
        }

    @classmethod
    def _interface_digests_from_metadata(
        cls, metadata_rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, str]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in metadata_rows:
            payload = cls._interface_payload_from_metadata(row)
            if payload is None:
                continue
            grouped.setdefault(str(payload["path"]), []).append(payload)
        return {
            path: digest
            for path, payloads in grouped.items()
            if (digest := cls._interface_digest(payloads)) is not None
        }

    @classmethod
    def _previous_interface_digests(
        cls,
        manifest: Mapping[str, Any],
        metadata_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, str]:
        metadata_digests = cls._interface_digests_from_metadata(metadata_rows)
        files = manifest.get("files", {})
        if not isinstance(files, Mapping):
            return metadata_digests
        digests = dict(metadata_digests)
        for raw_path, raw_entry in cast(Mapping[Any, Any], files).items():
            if not isinstance(raw_entry, Mapping):
                continue
            digest = raw_entry.get("interface_digest")
            if digest:
                digests[normalize_path(str(raw_path))] = str(digest)
        return digests

    @staticmethod
    def _interface_digest_changed_paths(
        *,
        changed_paths: Sequence[str],
        previous_digests: Mapping[str, str],
        current_digests: Mapping[str, str],
    ) -> list[str]:
        changed: list[str] = []
        for path in sorted(
            {normalize_path(str(path)) for path in changed_paths if path}
        ):
            previous = previous_digests.get(path)
            current = current_digests.get(path)
            if previous is None or current is None or previous != current:
                changed.append(path)
        return changed

    @staticmethod
    def _incremental_degraded_reasons(
        *,
        changed_paths: Sequence[str],
        removed_paths: Sequence[str],
        semantic_frontier_widened: bool,
        previous_interface_digests: Mapping[str, str],
        current_interface_digests: Mapping[str, str],
        new_elements: Sequence[Any],
        existing_by_stable_unit_id: Mapping[str, Mapping[str, Any]],
    ) -> list[str]:
        reasons: set[str] = set()
        if removed_paths:
            reasons.add("file_delete_requires_full_tool_rerun")
        if semantic_frontier_widened:
            reasons.add("semantic_frontier_widened")
        for path in {normalize_path(str(path)) for path in changed_paths if path}:
            if path not in previous_interface_digests:
                reasons.add("interface_digest_missing_previous")
            if path not in current_interface_digests:
                reasons.add("interface_digest_missing_current")
        for element in new_elements:
            metadata = IndexPipeline._metadata_payload(
                getattr(element, "metadata", None)
            )
            stable_unit_id = metadata.get("stable_unit_id")
            if not stable_unit_id:
                reasons.add("stable_unit_id_missing")
                continue
            if str(stable_unit_id) not in existing_by_stable_unit_id:
                reasons.add("stable_unit_id_new")
        return sorted(reasons)

    @staticmethod
    def _dependency_frontier_metadata(
        *,
        changed_paths: Sequence[str],
        interface_changed_paths: Sequence[str],
        package_scope_roots: Sequence[str],
        change_kinds: Sequence[str],
        degraded_reasons: Sequence[str],
    ) -> dict[str, Any]:
        interface_change_set = {
            normalize_path(str(path)) for path in interface_changed_paths if path
        }
        change_kind_set = {str(kind) for kind in change_kinds if kind}
        if interface_change_set or change_kind_set & {
            "api_surface_hash",
            "signature_hash",
            "edge_surface_hash",
        }:
            radius = "dependent_neighborhood"
            strategy = "package"
            target_paths = sorted(interface_change_set)
        else:
            radius = "file_local"
            strategy = "file"
            target_paths = sorted(
                {normalize_path(str(path)) for path in changed_paths if path}
            )
        return {
            "radius": radius,
            "strategy": strategy,
            "target_paths": target_paths,
            "scope_roots": sorted(
                {normalize_path(str(root)) for root in package_scope_roots if root}
            ),
            "change_kinds": sorted(change_kind_set),
            "degraded": bool(degraded_reasons),
            "degraded_reasons": sorted({str(reason) for reason in degraded_reasons}),
        }

    def _reuse_changed_unit_embeddings(
        self,
        *,
        new_elements: list[CodeElement],
        existing_by_stable_unit_id: dict[str, dict[str, Any]],
    ) -> int:
        reused = 0
        current_embedding_identity = self._incremental_compatibility_payload().get(
            "embedding"
        )
        for elem in new_elements:
            stable_unit_id = (elem.metadata or {}).get("stable_unit_id")
            if not stable_unit_id:
                continue
            existing = existing_by_stable_unit_id.get(str(stable_unit_id))
            if not existing:
                continue
            existing_meta = dict(
                cast(dict[str, Any], existing.get("metadata", {}) or {})
            )
            existing_text_hash = existing_meta.get("embedding_text_hash")
            current_text_hash = (elem.metadata or {}).get("embedding_text_hash")
            if not existing_text_hash or existing_text_hash != current_text_hash:
                continue
            existing_embedding_identity = existing_meta.get("embedding_fingerprint")
            if not self._embedding_identity_matches(
                existing_embedding_identity,
                current_embedding_identity,
            ):
                continue
            previous_embedding = existing_meta.get("embedding")
            previous_text = existing_meta.get("embedding_text")
            if previous_embedding is None or previous_text is None:
                continue
            elem.metadata["embedding"] = previous_embedding
            elem.metadata["embedding_text"] = previous_text
            elem.metadata["embedding_text_hash"] = existing_text_hash
            if "embedding_artifact_ref" in existing_meta:
                elem.metadata["embedding_artifact_ref"] = existing_meta[
                    "embedding_artifact_ref"
                ]
            if "embedding_fingerprint" in existing_meta:
                elem.metadata["embedding_fingerprint"] = existing_meta[
                    "embedding_fingerprint"
                ]
            reused += 1
        return reused

    @staticmethod
    def _embedding_identity_matches(existing: Any, current: Any) -> bool:
        if not isinstance(existing, Mapping) or not isinstance(current, Mapping):
            return False
        for field_name, expected_value in current.items():
            existing_value = existing.get(field_name)
            if field_name == "dimension" and (
                expected_value is None or existing_value is None
            ):
                continue
            if existing_value != expected_value:
                return False
        return True

    def _semantic_frontier_widened(
        self,
        *,
        new_elements: list[CodeElement],
        existing_by_stable_unit_id: dict[str, dict[str, Any]],
    ) -> bool:
        for elem in new_elements:
            stable_unit_id = (elem.metadata or {}).get("stable_unit_id")
            if not stable_unit_id:
                return True
            existing = existing_by_stable_unit_id.get(str(stable_unit_id))
            if not existing:
                return True
            existing_meta = dict(
                cast(dict[str, Any], existing.get("metadata", {}) or {})
            )
            current_meta = dict(elem.metadata or {})
            for field_name in (
                "signature_hash",
                "edge_surface_hash",
                "api_surface_hash",
            ):
                if current_meta.get(field_name) != existing_meta.get(field_name):
                    return True
        return False

    def _api_frontier_changed_paths(
        self,
        *,
        new_elements: list[CodeElement],
        existing_by_stable_unit_id: dict[str, dict[str, Any]],
    ) -> list[str]:
        changed_paths: set[str] = set()
        for elem in new_elements:
            rel_path = elem.relative_path or elem.file_path
            stable_unit_id = (elem.metadata or {}).get("stable_unit_id")
            if not rel_path:
                continue
            if not stable_unit_id:
                changed_paths.add(rel_path)
                continue
            existing = existing_by_stable_unit_id.get(str(stable_unit_id))
            if not existing:
                changed_paths.add(rel_path)
                continue
            existing_meta = dict(
                cast(dict[str, Any], existing.get("metadata", {}) or {})
            )
            current_meta = dict(elem.metadata or {})
            if current_meta.get("api_surface_hash") != existing_meta.get(
                "api_surface_hash"
            ):
                changed_paths.add(rel_path)
        return sorted(changed_paths)

    _CHANGE_KIND_FIELDS = (
        "signature_hash",
        "api_surface_hash",
        "edge_surface_hash",
        "embedding_text_hash",
    )

    def _classify_change_kinds(
        self,
        *,
        new_elements: list[CodeElement],
        existing_by_stable_unit_id: dict[str, dict[str, Any]],
    ) -> set[str]:
        kinds: set[str] = set()
        for elem in new_elements:
            stable_unit_id = (elem.metadata or {}).get("stable_unit_id")
            if not stable_unit_id:
                return set(self._CHANGE_KIND_FIELDS)
            existing = existing_by_stable_unit_id.get(str(stable_unit_id))
            if not existing:
                return set(self._CHANGE_KIND_FIELDS)
            existing_meta = dict(
                cast(dict[str, Any], existing.get("metadata", {}) or {})
            )
            current_meta = dict(elem.metadata or {})
            for field_name in self._CHANGE_KIND_FIELDS:
                if current_meta.get(field_name) != existing_meta.get(field_name):
                    kinds.add(field_name)
        return kinds

    def _package_scope_root(self, repo_root: str, rel_path: str) -> str:
        repo_root_abs = os.path.abspath(repo_root or ".")
        current_dir = os.path.dirname(os.path.join(repo_root_abs, rel_path))
        last_match = ""
        while current_dir.startswith(repo_root_abs):
            for marker in self._PACKAGE_SCOPE_MARKERS:
                if os.path.exists(os.path.join(current_dir, marker)):
                    rel_dir = os.path.relpath(current_dir, repo_root_abs)
                    return "." if rel_dir == "." else normalize_path(rel_dir)
            if current_dir == repo_root_abs:
                break
            last_match = current_dir
            current_dir = os.path.dirname(current_dir)
            if current_dir == last_match:
                break
        rel_dir = os.path.dirname(rel_path)
        return normalize_path(rel_dir) if rel_dir else "."

    def _package_scope_roots(
        self, repo_root: str, changed_paths: list[str]
    ) -> list[str]:
        return sorted(
            {
                self._package_scope_root(repo_root, rel_path)
                for rel_path in changed_paths
                if rel_path
            }
        )

    def _reconstruct_code_element(
        self,
        meta: dict[str, Any],
        *,
        file_info: dict[str, Any] | None = None,
        repo_name: str | None = None,
        repo_url: str | None = None,
    ) -> CodeElement | None:
        metadata = dict(cast(dict[str, Any], meta.get("metadata", {}) or {}))
        for stale_key in ("snapshot_id", "source_priority", "ir_symbol_id"):
            metadata.pop(stale_key, None)
        file_path = str(meta.get("file_path", ""))
        relative_path = str(meta.get("relative_path", ""))
        if file_info is not None:
            file_path = str(file_info.get("path") or file_path)
            relative_path = str(file_info.get("relative_path") or relative_path)
        try:
            return CodeElement(
                id=str(meta.get("id", "")),
                type=str(meta.get("type", "")),
                name=str(meta.get("name", "")),
                file_path=file_path,
                relative_path=relative_path,
                language=str(meta.get("language", "")),
                start_line=int(meta.get("start_line", 0) or 0),
                end_line=int(meta.get("end_line", 0) or 0),
                code=str(meta.get("code", "")),
                signature=cast(str | None, meta.get("signature")),
                docstring=cast(str | None, meta.get("docstring")),
                summary=cast(str | None, meta.get("summary")),
                metadata=metadata,
                repo_name=repo_name or cast(str | None, meta.get("repo_name")),
                repo_url=repo_url or cast(str | None, meta.get("repo_url")),
            )
        except Exception as e:
            self.logger.warning(f"Failed to reconstruct cached code element: {e}")
            return None

    @classmethod
    def _materialize_indexed_element_payload(
        cls,
        element: Any,
        *,
        snapshot_id: str,
    ) -> CodeElementMeta:
        payload = cls._code_element_like_payload(element)
        payload["snapshot_id"] = snapshot_id
        return payload

    @classmethod
    def _materialize_indexed_elements_for_storage(
        cls,
        elements: Sequence[CodeElement],
        *,
        snapshot_id: str,
    ) -> tuple[np.ndarray, list[CodeElementMeta], list[CodeElementMeta]]:
        vectors: list[Any] = []
        metadata: list[CodeElementMeta] = []
        all_payloads: list[CodeElementMeta] = []

        for element in elements:
            payload = cls._materialize_indexed_element_payload(
                element, snapshot_id=snapshot_id
            )
            all_payloads.append(payload)
            embedding = element.metadata.get("embedding")
            if embedding is None:
                continue
            vectors.append(embedding)
            metadata.append(payload)

        return (
            as_float32_matrix(vectors, copy_policy="contiguous"),
            metadata,
            all_payloads,
        )

    def _embedding_is_usable(self, embedding: Any) -> bool:
        vector = as_float32_vector(embedding, copy_policy="view")
        if vector is None:
            return False
        expected_dim = int(getattr(self.embedder, "embedding_dim", 0) or 0)
        return expected_dim <= 0 or int(vector.shape[0]) == expected_dim

    def _embed_missing_indexed_element_embeddings(
        self,
        elements: Sequence[CodeElement],
    ) -> int:
        missing: list[tuple[CodeElement, CodeElementMeta]] = []
        for element in elements:
            if self._embedding_is_usable((element.metadata or {}).get("embedding")):
                continue
            payload = serialize_code_element(element)
            payload_metadata = dict(payload.get("metadata", {}) or {})
            payload_metadata.pop("embedding", None)
            payload["metadata"] = payload_metadata
            missing.append((element, payload))

        if not missing:
            return 0

        embed_elements = getattr(self.embedder, "embed_elements", None)
        if not callable(embed_elements):
            return 0

        embedded_payloads = cast(
            list[CodeElementMeta],
            embed_elements([payload for _element, payload in missing]),
        )
        embedded_count = 0
        for (element, _payload), embedded_payload in zip(
            missing,
            embedded_payloads,
            strict=True,
        ):
            vector = as_float32_vector(
                embedded_payload.get("embedding"),
                copy_policy="contiguous",
            )
            if vector is None:
                continue
            element.metadata["embedding"] = vector
            embedding_text = embedded_payload.get("embedding_text")
            if isinstance(embedding_text, str):
                element.metadata["embedding_text"] = embedding_text
            artifact_ref = embedded_payload.get("embedding_artifact_ref")
            if isinstance(artifact_ref, str):
                element.metadata["embedding_artifact_ref"] = artifact_ref
            embedded_metadata = embedded_payload.get("metadata", {})
            for field_name in (
                "embedding_text_hash",
                "embedding_fingerprint",
            ):
                if field_name in embedded_metadata:
                    element.metadata[field_name] = embedded_metadata[field_name]
            embedded_count += 1
        return embedded_count

    def _plan_incremental_elements(
        self,
        *,
        repo_name: str,
        repo_url: str,
        snapshot_id: str,
        snapshot_ref: dict[str, Any],
        ref: str | None,
        current_files: list[dict[str, Any]] | None = None,
    ) -> tuple[list[CodeElement] | None, dict[str, Any] | None]:
        ref_name = snapshot_ref.get("branch") or ref or "HEAD"
        previous_manifest = self.manifest_store.get_branch_manifest_record(
            repo_name, ref_name
        )
        if not previous_manifest:
            return None, None

        previous_snapshot_id = previous_manifest.snapshot_id
        if not previous_snapshot_id or previous_snapshot_id == snapshot_id:
            return None, None

        previous_snapshot = self.snapshot_store.get_snapshot_record(
            previous_snapshot_id
        )
        if not previous_snapshot:
            return None, None

        previous_artifact_key = previous_snapshot.artifact_key
        manifest = self._load_file_manifest(previous_artifact_key)
        if manifest is None:
            return None, None
        if not self._manifest_is_incrementally_compatible(
            manifest, previous_artifact_key
        ):
            return None, None
        if not self._manifest_has_required_fingerprints(
            manifest, previous_artifact_key
        ):
            return None, None

        existing_metadata = self._load_existing_metadata(previous_artifact_key)
        if not existing_metadata:
            return None, None

        inventory = (
            current_files if current_files is not None else self.loader.scan_files()
        )
        changes = self._detect_file_changes(manifest, inventory)
        if not bool(changes.get("fingerprints_complete", True)):
            self.logger.info(
                "incremental prefilter disabled for %s: current files lack "
                "required fingerprints: %s",
                previous_artifact_key,
                ", ".join(
                    cast(list[str], changes.get("missing_fingerprint_paths", []))[:5]
                ),
            )
            return None, None
        added = cast(list[str], changes["added"])
        modified = cast(list[str], changes["modified"])
        deleted = cast(list[str], changes["deleted"])
        unchanged = cast(list[str], changes["unchanged"])

        unchanged_metadata, expected_unchanged_ids = self._collect_unchanged_metadata(
            manifest, unchanged, existing_metadata
        )
        found_unchanged_ids = {str(meta.get("id")) for meta in unchanged_metadata}
        if expected_unchanged_ids - found_unchanged_ids:
            self.logger.warning(
                "incremental prefilter fallback: missing cached metadata for "
                "%d unchanged elements",
                len(expected_unchanged_ids - found_unchanged_ids),
            )
            return None, None
        existing_by_stable_unit_id = self._metadata_by_stable_unit_id(existing_metadata)
        current_lookup = cast(dict[str, dict[str, Any]], changes["current_lookup"])

        changed_file_infos: list[dict[str, Any]] = []
        for rel_path in added + modified:
            lookup = current_lookup.get(rel_path)
            file_info = lookup.get("file_info") if lookup else None
            if isinstance(file_info, dict):
                changed_file_infos.append(file_info)

        changed_paths = sorted(set(added + modified))
        (
            cached_changed_elements,
            parsed_artifact_hit_paths,
            parsed_artifact_reuse,
        ) = self._reuse_parsed_element_content_artifacts(
            repo_name=repo_name,
            current_files=inventory,
            paths=changed_paths,
        )
        uncached_changed_file_infos = [
            file_info
            for file_info in changed_file_infos
            if normalize_path(str(file_info.get("relative_path") or ""))
            not in parsed_artifact_hit_paths
        ]
        parsed_new_elements = (
            self.indexer.index_files(uncached_changed_file_infos, repo_name, repo_url)
            if uncached_changed_file_infos
            else []
        )
        new_elements = cached_changed_elements + parsed_new_elements
        parsed_artifact_publish = {
            "mode": "content_addressed",
            "artifact_type": (
                self.file_artifact_store.parsed_elements_artifact_type
                if self.file_artifact_store is not None
                else "parsed_elements"
            ),
            "candidate_files": len(uncached_changed_file_infos),
            "written_records": 0,
            "candidate_elements": len(parsed_new_elements),
        }
        reused_changed_embeddings = self._reuse_changed_unit_embeddings(
            new_elements=new_elements,
            existing_by_stable_unit_id=existing_by_stable_unit_id,
        )
        semantic_frontier_widened = self._semantic_frontier_widened(
            new_elements=new_elements,
            existing_by_stable_unit_id=existing_by_stable_unit_id,
        )
        api_frontier_changed_paths = self._api_frontier_changed_paths(
            new_elements=new_elements,
            existing_by_stable_unit_id=existing_by_stable_unit_id,
        )
        change_kinds = self._classify_change_kinds(
            new_elements=new_elements,
            existing_by_stable_unit_id=existing_by_stable_unit_id,
        )
        previous_interface_digests = self._previous_interface_digests(
            manifest,
            existing_metadata,
        )
        current_interface_digests = self._interface_digests_for_elements(new_elements)
        interface_changed_paths = self._interface_digest_changed_paths(
            changed_paths=changed_paths,
            previous_digests=previous_interface_digests,
            current_digests=current_interface_digests,
        )
        api_frontier_changed_paths = sorted(
            set(api_frontier_changed_paths) | set(interface_changed_paths)
        )
        package_scope_roots = self._package_scope_roots(
            self.loader.repo_path or "",
            api_frontier_changed_paths,
        )
        degraded_reasons = self._incremental_degraded_reasons(
            changed_paths=changed_paths,
            removed_paths=deleted,
            semantic_frontier_widened=semantic_frontier_widened,
            previous_interface_digests=previous_interface_digests,
            current_interface_digests=current_interface_digests,
            new_elements=new_elements,
            existing_by_stable_unit_id=existing_by_stable_unit_id,
        )
        dependency_frontier = self._dependency_frontier_metadata(
            changed_paths=changed_paths,
            interface_changed_paths=interface_changed_paths,
            package_scope_roots=package_scope_roots,
            change_kinds=sorted(change_kinds),
            degraded_reasons=degraded_reasons,
        )
        plan_changes = PlanChanges(
            previous_snapshot_id=str(previous_snapshot_id),
            previous_artifact_key=previous_artifact_key,
            added_paths=tuple(sorted(added)),
            modified_paths=tuple(sorted(modified)),
            removed_paths=tuple(sorted(deleted)),
            unchanged_paths=tuple(sorted(unchanged)),
            reused_elements=len(expected_unchanged_ids),
            reindexed_elements=len(new_elements),
            reused_changed_embeddings=reused_changed_embeddings,
            semantic_frontier_widened=semantic_frontier_widened,
            api_frontier_changed_paths=tuple(api_frontier_changed_paths),
            package_scope_roots=tuple(package_scope_roots),
            change_kinds=tuple(sorted(change_kinds)),
            interface_digest_changed_paths=tuple(interface_changed_paths),
            interface_digests={
                path: current_interface_digests[path]
                for path in sorted(current_interface_digests)
                if path in set(changed_paths)
            },
            dependency_frontier=dependency_frontier,
            degraded_reasons=tuple(degraded_reasons),
        )
        prefilter_payload = plan_changes.to_prefilter_payload()
        prefilter_payload["parsed_artifact_reuse"] = dict(parsed_artifact_reuse)
        prefilter_payload["parsed_artifact_publish"] = dict(parsed_artifact_publish)
        return new_elements, prefilter_payload

    @staticmethod
    def _preservable_incremental_sources(
        incremental_plan: dict[str, Any] | None,
    ) -> set[str] | None:
        if incremental_plan is None:
            return None
        change_kinds = {
            str(kind) for kind in incremental_plan.get("change_kinds", []) if kind
        }
        unstable_tool_surfaces = {
            "api_surface_hash",
            "signature_hash",
            "edge_surface_hash",
        }
        if change_kinds & unstable_tool_surfaces:
            return None
        if int(incremental_plan.get("removed", 0) or 0) > 0:
            return None
        if int(incremental_plan.get("modified", 0) or 0) <= 0:
            return None
        return {"scip"}

    def _incremental_scip_scope(
        self,
        incremental_plan: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if incremental_plan is None:
            return None
        changed_paths = [
            normalize_path(str(path))
            for path in incremental_plan.get("changed_paths", [])
            if path
        ]
        if not changed_paths:
            return {
                "mode": "skip",
                "reason": "no_incremental_scip_frontier",
                "target_paths": [],
                "scope_roots": [],
            }
        if self._preservable_incremental_sources(incremental_plan):
            return {
                "mode": "skip",
                "reason": "source_owned_evidence_preserved",
                "target_paths": changed_paths,
                "scope_roots": [],
            }
        scope_roots = list(
            incremental_plan.get("package_scope_roots")
            or self._package_scope_roots(self.loader.repo_path or "", changed_paths)
        )
        return {
            "mode": "package",
            "reason": "incremental_surface_frontier",
            "target_paths": changed_paths,
            "scope_roots": scope_roots,
        }

    @staticmethod
    def _scip_degraded_reasons(
        incremental_plan: dict[str, Any] | None,
        scip_scope: Mapping[str, Any] | None,
    ) -> list[str]:
        if incremental_plan is None:
            return []

        reasons: set[str] = set()
        change_kinds = {
            str(kind) for kind in incremental_plan.get("change_kinds", []) if kind
        }
        if int(incremental_plan.get("removed", 0) or 0) > 0:
            reasons.add("scip_full_rerun:file_delete")
        if int(incremental_plan.get("semantic_frontier_widened", 0) or 0) > 0:
            reasons.add("scip_frontier_widened")
        if change_kinds & {
            "api_surface_hash",
            "signature_hash",
            "edge_surface_hash",
        }:
            reasons.add("scip_dependency_frontier_changed")
        if scip_scope is None and not reasons:
            reasons.add("scip_full_rerun:unsupported_incremental_scope")
        return sorted(reasons)

    @staticmethod
    def _build_repair_frontier_task(
        *,
        snapshot_id: str,
        repo_name: str,
        source: str,
        changed_paths: list[str],
        modified_count: int,
        widened: bool,
        scope_kind: str,
        scope_roots: list[str],
        change_kinds: list[str] | None = None,
    ) -> dict[str, Any] | None:
        if not changed_paths or not widened:
            return None
        return {
            "task_type": "semantic_repair_frontier",
            "payload": {
                "snapshot_id": snapshot_id,
                "repo_name": repo_name,
                "source": source,
                "changed_paths": changed_paths,
                "reason": "api_or_edge_surface_changed",
                "modified_count": modified_count,
                "widened": widened,
                "scope_kind": scope_kind,
                "scope_roots": scope_roots,
                "change_kinds": list(change_kinds or []),
            },
        }

    def _repair_scope_paths(
        self,
        *,
        elements: list[CodeElement],
        changed_paths: list[str],
        scope_kind: str,
        scope_roots: list[str],
    ) -> set[str]:
        candidate_paths = {
            normalize_path(elem.relative_path or elem.file_path)
            for elem in elements
            if (elem.relative_path or elem.file_path)
        }
        if scope_kind == "package" and scope_roots:
            selected: set[str] = set()
            for root in scope_roots:
                normalized_root = normalize_path(root)
                if normalized_root == ".":
                    selected |= candidate_paths
                    continue
                prefix = f"{normalized_root}/"
                selected |= {
                    path
                    for path in candidate_paths
                    if path == normalized_root or path.startswith(prefix)
                }
            if selected:
                return selected
        changed = {normalize_path(path) for path in changed_paths if path}
        return changed or candidate_paths

    @staticmethod
    def _candidate_scope_paths(elements: list[CodeElement]) -> set[str]:
        return {
            normalize_path(elem.relative_path or elem.file_path)
            for elem in elements
            if (elem.relative_path or elem.file_path)
        }

    def _expand_repair_target_paths(
        self,
        *,
        elements: list[CodeElement],
        changed_paths: list[str],
        scope_paths: set[str],
        max_hops: int | None = None,
        change_kinds: set[str] | None = None,
        package_roots: list[str] | None = None,
        degraded_reasons: list[str] | None = None,
    ) -> set[str]:
        if change_kinds is not None and change_kinds:
            if change_kinds == {"embedding_text_hash"}:
                return scope_paths
            if change_kinds <= {"edge_surface_hash"} and max_hops is None:
                max_hops = 1

        path_by_element_id = {
            str(elem.id): normalize_path(elem.relative_path or elem.file_path)
            for elem in elements
            if (elem.relative_path or elem.file_path)
        }
        seed_ids = {
            element_id
            for element_id, path in path_by_element_id.items()
            if path in {normalize_path(p) for p in changed_paths}
        }
        if not seed_ids:
            return scope_paths

        frontier_ids = set(seed_ids)
        visited_ids = set(seed_ids)
        hop_budget = (
            int(max_hops)
            if max_hops is not None
            else max(1, int(self.config.get("graph", {}).get("max_depth", 5)))
        )
        graphs = [
            getattr(self.graph_builder, "dependency_graph", None),
            getattr(self.graph_builder, "call_graph", None),
            getattr(self.graph_builder, "inheritance_graph", None),
        ]

        for _ in range(hop_budget):
            next_ids: set[str] = set()
            for graph in graphs:
                if graph is None:
                    continue
                for node_id in frontier_ids:
                    if node_id not in graph:
                        continue
                    try:
                        next_ids.update(
                            str(pred) for pred in graph.predecessors(node_id)
                        )
                    except Exception:
                        continue
            next_ids -= visited_ids
            if not next_ids:
                break
            visited_ids |= next_ids
            frontier_ids = next_ids

        expanded_paths = {
            path_by_element_id[element_id]
            for element_id in visited_ids
            if element_id in path_by_element_id
            and path_by_element_id[element_id] in scope_paths
        }
        if expanded_paths and package_roots and degraded_reasons is not None:
            root_set = {
                normalize_path(root) for root in package_roots if normalize_path(root)
            }
            if root_set and any(
                not any(
                    path == root or path.startswith(f"{root}/") for root in root_set
                )
                for path in expanded_paths
            ):
                degraded_reasons.append("expansion_widened_past_package")
        return expanded_paths or scope_paths

    @staticmethod
    def _evidence_source_owned(
        source: str,
        owned_sources: set[str] | None,
    ) -> bool:
        if owned_sources is None:
            return source not in {"fc_structure", "scip"}
        return source in owned_sources

    def _drop_owned_evidence(
        self,
        *,
        snapshot: IRSnapshot,
        target_paths: set[str],
        owned_sources: set[str] | None,
    ) -> IRSnapshot:
        target_paths_norm = {normalize_path(path) for path in target_paths}
        owned_supports: list[IRUnitSupport] = []
        removed_support_ids: set[str] = set()
        support_sources_by_id = {
            support.support_id: support.source for support in snapshot.supports
        }
        for support in snapshot.supports:
            support_path = normalize_path(support.path or "")
            owned = support_path in target_paths_norm and self._evidence_source_owned(
                support.source, owned_sources
            )
            if owned:
                removed_support_ids.add(support.support_id)
                continue
            owned_supports.append(support)

        unit_paths = {
            unit.unit_id: normalize_path(unit.path) for unit in snapshot.units
        }
        owned_relations: list[IRRelation] = []
        for relation in snapshot.relations:
            src_path = unit_paths.get(relation.src_unit_id, "")
            relation_sources = (
                set(relation.support_sources)
                if relation.support_sources
                else {relation.source}
            )
            relation_owned = src_path in target_paths_norm and all(
                self._evidence_source_owned(source, owned_sources)
                for source in relation_sources
                if source
            )
            if relation_owned:
                continue
            removed_relation_support_ids = [
                support_id
                for support_id in relation.support_ids
                if support_id in removed_support_ids
            ]
            if removed_relation_support_ids:
                removed_sources = {
                    support_sources_by_id.get(support_id, "")
                    for support_id in removed_relation_support_ids
                }
                remaining_support_ids = [
                    support_id
                    for support_id in relation.support_ids
                    if support_id not in removed_support_ids
                ]
                remaining_support_sources = set(relation.support_sources) - {
                    source for source in removed_sources if source
                }
                if not remaining_support_ids and not remaining_support_sources:
                    continue
                owned_relations.append(
                    dc_replace(
                        relation,
                        support_ids=remaining_support_ids,
                        support_sources=remaining_support_sources,
                    )
                )
                continue
            owned_relations.append(relation)

        return IRSnapshot(
            repo_name=snapshot.repo_name,
            snapshot_id=snapshot.snapshot_id,
            branch=snapshot.branch,
            commit_id=snapshot.commit_id,
            tree_id=snapshot.tree_id,
            units=list(snapshot.units),
            supports=owned_supports,
            relations=owned_relations,
            embeddings=list(snapshot.embeddings),
            metadata=dict(snapshot.metadata or {}),
        )

    def _drop_owned_semantic_evidence(
        self,
        *,
        snapshot: IRSnapshot,
        target_paths: set[str],
    ) -> IRSnapshot:
        return self._drop_owned_evidence(
            snapshot=snapshot,
            target_paths=target_paths,
            owned_sources=None,
        )

    @staticmethod
    def _filter_scip_index_to_paths(
        scip_index: SCIPIndex | None, target_paths: set[str]
    ) -> SCIPIndex | None:
        if scip_index is None:
            return None
        filtered_docs = [
            doc
            for doc in scip_index.documents
            if normalize_path(doc.path) in target_paths
        ]
        if not filtered_docs:
            return None
        return SCIPIndex(
            documents=filtered_docs,
            indexer_name=scip_index.indexer_name,
            indexer_version=scip_index.indexer_version,
            metadata=dict(scip_index.metadata or {}),
        )

    def _scoped_scip_cache_dir(self) -> str:
        scip_config = self.config.get("scip", {})
        configured = (
            scip_config.get("artifact_cache_dir")
            if isinstance(scip_config, dict)
            else None
        )
        cache_dir = (
            str(configured)
            if configured
            else os.path.join(self.vector_store.persist_dir, "scip_tool_cache")
        )
        ensure_dir(cache_dir)
        return cache_dir

    def _scoped_scip_file_fingerprint(
        self, repo_root: str, rel_path: str
    ) -> dict[str, Any] | None:
        normalized = normalize_path(rel_path)
        abs_path = os.path.join(repo_root, normalized)
        try:
            stat = os.stat(abs_path)
        except OSError:
            return None
        content_hash = compute_file_hash(abs_path) or None
        if content_hash is None:
            return None
        return {
            "path": normalized,
            "size": int(stat.st_size),
            "content_hash": content_hash,
        }

    def _scoped_scip_package_marker_paths(
        self, repo_root: str, scope_root: str
    ) -> list[str]:
        normalized_scope = normalize_path(scope_root)
        candidates: set[str] = set()
        for marker in self._PACKAGE_SCOPE_MARKERS:
            candidates.add(marker)
            if normalized_scope and normalized_scope != ".":
                candidates.add(normalize_path(os.path.join(normalized_scope, marker)))
        return sorted(
            path for path in candidates if os.path.exists(os.path.join(repo_root, path))
        )

    def _scoped_scip_cache_entry(
        self,
        *,
        repo_root: str,
        language: str,
        scope_root: str,
        target_paths: set[str],
    ) -> ScopedSCIPCacheEntry:
        normalized_scope = normalize_path(scope_root or ".") or "."
        profile = self._scip_profile(language)
        file_fingerprints = [
            fingerprint
            for path in sorted(normalize_path(path) for path in target_paths if path)
            if (fingerprint := self._scoped_scip_file_fingerprint(repo_root, path))
            is not None
        ]
        marker_fingerprints = [
            fingerprint
            for path in self._scoped_scip_package_marker_paths(
                repo_root, normalized_scope
            )
            if (fingerprint := self._scoped_scip_file_fingerprint(repo_root, path))
            is not None
        ]
        payload: dict[str, Any] = {
            "version": 1,
            "language": language,
            "scope_root": normalized_scope,
            "target_paths": sorted(
                normalize_path(path) for path in target_paths if path
            ),
            "files": file_fingerprints,
            "package_markers": marker_fingerprints,
            "tool": {
                "binary_name": profile.binary_name if profile else None,
                "extra_args": list(profile.extra_args) if profile else [],
                "experimental": bool(profile.experimental) if profile else False,
            },
            "mode": "repo_root_filtered",
        }
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return ScopedSCIPCacheEntry(
            key=key,
            path=os.path.join(self._scoped_scip_cache_dir(), f"{key}.json"),
            payload=payload,
        )

    def _load_scoped_scip_cache(self, entry: ScopedSCIPCacheEntry) -> SCIPIndex | None:
        if not os.path.exists(entry.path):
            return None
        try:
            with open(entry.path, encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning(
                "Failed to load scoped SCIP cache %s: %s", entry.key, exc
            )
            return None
        if not isinstance(raw, dict) or raw.get("key") != entry.key:
            return None
        if raw.get("identity") != entry.payload:
            return None
        index_payload = raw.get("scip_index")
        if not isinstance(index_payload, dict):
            return None
        return SCIPIndex.from_dict(index_payload)

    def _save_scoped_scip_cache(
        self, entry: ScopedSCIPCacheEntry, scip_index: SCIPIndex
    ) -> None:
        payload = {
            "key": entry.key,
            "identity": entry.payload,
            "scip_index": scip_index.to_dict(),
        }
        tmp_path = f"{entry.path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            os.replace(tmp_path, entry.path)
        except OSError as exc:
            self.logger.warning(
                "Failed to save scoped SCIP cache %s: %s", entry.key, exc
            )
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _scoped_scip_runs_from_repo_root(self) -> bool:
        scip_config = self.config.get("scip", {})
        if not isinstance(scip_config, dict):
            return True
        return bool(scip_config.get("scoped_repo_root_mode", True))

    def _copy_scope_root(self, repo_root: str, scope_root: str, temp_root: str) -> str:
        normalized_root = normalize_path(scope_root)
        if normalized_root == ".":
            return repo_root
        source_root = os.path.join(repo_root, normalized_root)
        dest_root = os.path.join(temp_root, normalized_root)
        ensure_dir(os.path.dirname(dest_root))

        def _copy2_counting(src: str, dst: str) -> str:
            try:
                self._profile_add_io("scoped_tool_copied_bytes", os.path.getsize(src))
                self._profile_add_io("scoped_tool_copied_files", 1)
            except OSError:
                pass
            return shutil.copy2(src, dst)

        shutil.copytree(
            source_root, dest_root, symlinks=True, copy_function=_copy2_counting
        )
        for marker in self._PACKAGE_SCOPE_MARKERS:
            source_marker = os.path.join(repo_root, marker)
            if os.path.exists(source_marker):
                try:
                    self._profile_add_io(
                        "scoped_tool_copied_bytes", os.path.getsize(source_marker)
                    )
                    self._profile_add_io("scoped_tool_copied_files", 1)
                except OSError:
                    pass
                shutil.copy2(source_marker, os.path.join(temp_root, marker))
        return temp_root

    def _run_scoped_scip_frontier(
        self,
        *,
        snapshot: IRSnapshot,
        repo_name: str,
        scope_kind: str,
        scope_roots: list[str],
        target_paths: set[str],
        warnings: list[str],
        degraded_reasons: list[str] | None = None,
    ) -> tuple[IRSnapshot | None, list[str]]:
        repo_root = self.loader.repo_path or ""
        if scope_kind != "package" or not repo_root or not scope_roots:
            return None, []
        languages = list(
            self._detect_scip_languages_in_paths(repo_root, sorted(target_paths))
        )
        if not languages:
            return None, []

        filtered_indexes: list[SCIPIndex] = []
        cache_hits = 0
        cache_misses = 0
        scope_copies = 0
        for scope_root in scope_roots:
            materialized_root = repo_root
            temp_root: str | None = None
            try:
                use_repo_root = self._scoped_scip_runs_from_repo_root()
                if normalize_path(scope_root) != "." and not use_repo_root:
                    temp_root = tempfile.mkdtemp(prefix="fastcode_scope_repo_")
                    self._profile_record_temp_dir(temp_root)
                    materialized_root = self._copy_scope_root(
                        repo_root, scope_root, temp_root
                    )
                    scope_copies += 1
                for language in languages:
                    cache_entry = self._scoped_scip_cache_entry(
                        repo_root=repo_root,
                        language=language,
                        scope_root=scope_root,
                        target_paths=target_paths,
                    )
                    cached_index = self._load_scoped_scip_cache(cache_entry)
                    if cached_index is not None:
                        cache_hits += 1
                        filtered_indexes.append(cached_index)
                        continue
                    cache_misses += 1
                    scip_temp_root = tempfile.mkdtemp(prefix="fastcode_scope_scip_")
                    self._profile_record_temp_dir(scip_temp_root)
                    scoped_index = self._run_scip_for_language(
                        language,
                        materialized_root,
                        scip_temp_root,
                    )
                    filtered = self._filter_scip_index_to_paths(
                        scoped_index, target_paths
                    )
                    if filtered is not None:
                        filtered.metadata["scoped_scip_cache_key"] = cache_entry.key
                        self._save_scoped_scip_cache(cache_entry, filtered)
                        filtered_indexes.append(filtered)
            except Exception as exc:
                warnings.append(f"scoped_scip_failed:{scope_root}:{exc}")
                if degraded_reasons is not None:
                    degraded_reasons.append(f"scoped_scip_failed:{scope_root}")
            finally:
                if temp_root:
                    self._profile_add_io(
                        "deleted_bytes", self._path_size_bytes(temp_root)
                    )
                    shutil.rmtree(temp_root, ignore_errors=True)
        if not filtered_indexes:
            return None, languages

        combined = SCIPIndex(
            documents=[doc for index in filtered_indexes for doc in index.documents],
            indexer_name=filtered_indexes[0].indexer_name,
            indexer_version=filtered_indexes[0].indexer_version,
            metadata={
                "scoped": True,
                "scope_roots": scope_roots,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "scope_copies": scope_copies,
                "working_mode": (
                    "repo_root_filtered"
                    if self._scoped_scip_runs_from_repo_root()
                    else "copied_scope_root"
                ),
            },
        )
        scip_snapshot = build_ir_from_scip(
            repo_name=repo_name,
            snapshot_id=snapshot.snapshot_id,
            scip_index=combined,
            branch=snapshot.branch,
            commit_id=snapshot.commit_id,
            tree_id=snapshot.tree_id,
        )
        scip_snapshot.metadata["scoped_scip_cache"] = {
            "hits": cache_hits,
            "misses": cache_misses,
            "scope_copies": scope_copies,
            "working_mode": combined.metadata["working_mode"],
        }
        return scip_snapshot, languages

    def run_semantic_repair_frontier(
        self,
        *,
        snapshot_id: str,
        scope_kind: str,
        scope_roots: list[str],
        changed_paths: list[str],
        repo_name: str | None = None,
        change_kinds: list[str] | None = None,
    ) -> dict[str, Any]:
        record = self.snapshot_store.get_snapshot_record(snapshot_id)
        if not record:
            msg = f"snapshot not found: {snapshot_id}"
            raise RuntimeError(msg)
        snapshot = self.snapshot_store.load_snapshot(snapshot_id)
        if snapshot is None:
            msg = f"snapshot IR not found: {snapshot_id}"
            raise RuntimeError(msg)
        if not self._load_artifacts_by_key(record.artifact_key):
            msg = f"failed to load artifacts for snapshot: {snapshot_id}"
            raise RuntimeError(msg)

        elements = self._reconstruct_elements_from_metadata()
        change_kind_set = set(change_kinds) if change_kinds else None
        candidate_paths = self._candidate_scope_paths(elements)
        target_paths = self._repair_scope_paths(
            elements=elements,
            changed_paths=changed_paths,
            scope_kind=scope_kind,
            scope_roots=scope_roots,
        )
        warnings: list[str] = []
        degraded_reasons: list[str] = []
        if scope_kind != "package":
            expansion_scope_paths = (
                candidate_paths
                if change_kind_set and change_kind_set - {"embedding_text_hash"}
                else target_paths
            )
            target_paths = self._expand_repair_target_paths(
                elements=elements,
                changed_paths=changed_paths,
                scope_paths=expansion_scope_paths,
                change_kinds=change_kind_set,
                package_roots=(
                    scope_roots
                    or self._package_scope_roots(
                        self.loader.repo_path or "",
                        changed_paths,
                    )
                ),
                degraded_reasons=degraded_reasons,
            )
        scoped_tool_languages: list[str] = []
        if not target_paths:
            return {
                "status": "skipped",
                "snapshot_id": snapshot_id,
                "repo_name": repo_name or snapshot.repo_name,
                "warnings": warnings,
                "degraded_reasons": degraded_reasons,
                "repair_frontier": {
                    "scope_kind": scope_kind,
                    "scope_roots": scope_roots,
                    "changed_paths": changed_paths,
                    "target_paths": [],
                    "tool_rerun_languages": scoped_tool_languages,
                },
            }

        if scope_kind == "package" and (self.loader.repo_path or ""):
            try:
                scoped_tool_languages = list(
                    self._detect_scip_languages_in_paths(
                        self.loader.repo_path or "",
                        sorted(target_paths),
                    )
                )
            except Exception as exc:
                warnings.append(f"scoped_tool_language_detection_failed: {exc}")
        elif (
            scope_kind != "package"
            and change_kind_set
            and change_kind_set - {"embedding_text_hash"}
        ):
            degraded_reasons.append("tooling_repo_fallback")

        base_snapshot = self._drop_owned_semantic_evidence(
            snapshot=snapshot,
            target_paths=target_paths,
        )
        scoped_scip_snapshot = None
        if scope_kind == "package":
            scoped_scip_snapshot, scoped_tool_languages = (
                self._run_scoped_scip_frontier(
                    snapshot=base_snapshot,
                    repo_name=repo_name or snapshot.repo_name,
                    scope_kind=scope_kind,
                    scope_roots=scope_roots,
                    target_paths=target_paths,
                    warnings=warnings,
                    degraded_reasons=degraded_reasons,
                )
            )
        if scoped_scip_snapshot is not None:
            base_snapshot = self._drop_owned_evidence(
                snapshot=base_snapshot,
                target_paths=target_paths,
                owned_sources={"scip"},
            )
            base_snapshot = merge_ir(base_snapshot, scoped_scip_snapshot)

        repair_frontier_summary = {
            "snapshot_id": snapshot_id,
            "scope_kind": scope_kind,
            "scope_roots": scope_roots,
            "changed_paths": changed_paths,
            "target_paths": sorted(target_paths),
            "change_kinds": sorted(change_kind_set or []),
            "tool_rerun_languages": list(scoped_tool_languages),
            "degraded_reasons": list(degraded_reasons),
        }
        scoped_tool_ref = hashlib.sha1(
            json.dumps(repair_frontier_summary, sort_keys=True).encode("utf-8")
        ).hexdigest()
        affected_rows = self._unit_artifact_rows(
            elements,
            target_paths=target_paths,
            repair_frontier_summary=repair_frontier_summary,
            scoped_tool_ref=scoped_tool_ref,
        )
        affected_stable_unit_ids: list[str] = []
        for row in affected_rows:
            metadata = cast(dict[str, Any], row.get("metadata", {}) or {})
            stable_unit_id = str(metadata.get("stable_unit_id") or "")
            if stable_unit_id:
                affected_stable_unit_ids.append(stable_unit_id)
        affected_stable_unit_ids = sorted(set(affected_stable_unit_ids))
        repaired_snapshot = self._apply_semantic_resolvers(
            snapshot=base_snapshot,
            elements=elements,
            graph_context=self.graph_builder,
            target_paths=target_paths,
            warnings=warnings,
            budget="repair_frontier",
        )
        metadata = (
            json.loads(record.metadata_json) if record.metadata_json is not None else {}
        )
        self._invalidate_loaded_artifact_handle(record.artifact_key)
        save_snapshot_delta = getattr(self.snapshot_store, "save_snapshot_delta", None)
        if callable(save_snapshot_delta):
            save_snapshot_delta(
                repaired_snapshot,
                previous_snapshot_id=snapshot_id,
                changed_paths=sorted(target_paths),
                removed_paths=[],
                metadata=metadata,
            )
        else:
            self.snapshot_store.save_snapshot(repaired_snapshot, metadata=metadata)
        refreshed_rows = self._current_unit_artifact_rows(
            rows=affected_rows,
            snapshot=repaired_snapshot,
            target_paths=target_paths,
        )
        if affected_stable_unit_ids:
            self.unit_artifact_store.refresh_units(
                snapshot_id=snapshot_id,
                stable_unit_ids=affected_stable_unit_ids,
                elements=refreshed_rows,
            )
        self.snapshot_symbol_index.register_snapshot(repaired_snapshot)
        self.snapshot_store.save_relational_facts(repaired_snapshot)
        _repaired_ir_graphs, ir_graph_persistence = self._save_ir_graphs_for_index(
            repaired_snapshot,
            {
                "previous_snapshot_id": snapshot_id,
                "added_paths": [],
                "modified_paths": sorted(target_paths),
                "removed_paths": [],
            },
        )
        metadata["pipeline_metrics"] = repaired_snapshot.metadata.get(
            "pipeline_metrics",
            metadata.get("pipeline_metrics", {}),
        )
        self.snapshot_store.update_snapshot_metadata(snapshot_id, metadata)
        self._load_artifacts_by_key(record.artifact_key)
        return {
            "status": "repaired",
            "snapshot_id": snapshot_id,
            "repo_name": repo_name or snapshot.repo_name,
            "warnings": warnings,
            "degraded_reasons": degraded_reasons,
            "repair_frontier": {
                "scope_kind": scope_kind,
                "scope_roots": scope_roots,
                "changed_paths": changed_paths,
                "target_paths": sorted(target_paths),
                "tool_rerun_languages": scoped_tool_languages,
            },
            "ir_graph_persistence": ir_graph_persistence,
        }

    # ------------------------------------------------------------------
    # The big one: run_index_pipeline
    # ------------------------------------------------------------------

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
        # These are set by FastCode.load_repository() and stored externally;
        # we receive them via callbacks. But load_repository itself lives on
        # FastCode still, so the pipeline needs a reference to call it.
        load_repository_cb: Callable[..., None] | None = None,
        # Access to loaded_repositories dict for storing results
        get_loaded_repositories: Callable[[], dict[str, dict[str, Any]]] | None = None,
        graph_runtime: DocumentGraphRuntime | None = None,
    ) -> dict[str, Any]:
        """
        Run snapshot-oriented indexing pipeline with AST + optional SCIP merge.
        """
        resolved_is_url = self._infer_is_url(source) if is_url is None else is_url
        pipeline_profile = self._new_pipeline_profile()
        previous_profile = self._active_pipeline_profile
        self._active_pipeline_profile = pipeline_profile

        # Load repository via callback (FastCode owns the loader lifecycle)
        if load_repository_cb is not None:
            with self._profile_stage(pipeline_profile, "load_repository"):
                load_repository_cb(source, is_url=resolved_is_url)
            self._profile_record_loader_io(pipeline_profile)

        with self._profile_stage(pipeline_profile, "checkout"):
            self._checkout_target_ref(ref=ref, commit=commit)
        with self._profile_stage(pipeline_profile, "file_inventory"):
            current_files = self._scan_files_for_pipeline()
        repo_info = self._repository_info_for_pipeline(current_files)
        self._set_repo_info(repo_info)

        repo_name = repo_info.get("name", "default")
        repo_url = repo_info.get("url", source)
        snapshot_ref = self._resolve_snapshot_ref(
            repo_name,
            requested_ref=ref,
            requested_commit=commit,
            current_files=current_files,
        )
        git_meta = self._build_git_meta(snapshot_ref)
        snapshot_id = snapshot_ref["snapshot_id"]
        warnings: list[str] = []
        degraded = False
        pipeline_layers = self._default_pipeline_layers(enable_scip=enable_scip)

        existing = self.snapshot_store.get_snapshot_record(snapshot_id)
        if existing and not force:
            materialization_counters = MaterializationCounters()
            materialization_token = set_materialization_counters(
                materialization_counters
            )
            try:
                artifact_key = existing.artifact_key
                loaded = self._load_artifacts_by_key(artifact_key)
                result = self._backfill_result_layer_metadata(
                    snapshot_id=snapshot_id,
                    enable_scip=enable_scip,
                    result={
                        "status": "reused",
                        "repo_name": repo_name,
                        "snapshot_id": snapshot_id,
                        "artifact_key": artifact_key,
                        "loaded": loaded,
                    },
                )
                result["pipeline_metrics"] = self._with_materialization_metrics(
                    cast(dict[str, Any], result.get("pipeline_metrics", {})),
                    materialization_counters,
                )
                self._attach_cache_update_metrics(
                    result["pipeline_metrics"],
                    current_files=current_files,
                    reused_existing_snapshot=True,
                )
                self._attach_pipeline_profile(
                    result["pipeline_metrics"],
                    pipeline_profile,
                )
                return result
            finally:
                self._active_pipeline_profile = previous_profile
                reset_materialization_counters(materialization_token)

        idempotency_key = hashlib.sha1(
            f"{repo_name}:{snapshot_id}:{bool(publish)}:{bool(enable_scip)}".encode()
        ).hexdigest()
        run_id = self.index_run_store.create_run(
            repo_name=repo_name,
            snapshot_id=snapshot_id,
            branch=snapshot_ref.get("branch"),
            commit_id=snapshot_ref.get("commit_id"),
            idempotency_key=idempotency_key,
        )
        existing_run = self.index_run_store.get_run_record(run_id)
        existing_run_status = existing_run.status if existing_run is not None else None
        existing_run_warnings_json = (
            existing_run.warnings_json if existing_run is not None else None
        )
        if (
            existing_run
            and existing_run_status
            in {"published", "succeeded", "degraded", "publish_pending"}
            and not force
        ):
            existing_snapshot = self.snapshot_store.get_snapshot_record(snapshot_id)
            if existing_snapshot:
                materialization_counters = MaterializationCounters()
                materialization_token = set_materialization_counters(
                    materialization_counters
                )
                try:
                    loaded = self._load_artifacts_by_key(existing_snapshot.artifact_key)
                    increment_materialization_boundary(BOUNDARY_JSON_DECODE)
                    result = self._backfill_result_layer_metadata(
                        snapshot_id=snapshot_id,
                        enable_scip=enable_scip,
                        result={
                            "status": existing_run_status,
                            "run_id": run_id,
                            "repo_name": repo_name,
                            "snapshot_id": snapshot_id,
                            "artifact_key": existing_snapshot.artifact_key,
                            "loaded": loaded,
                            "warnings": json.loads(existing_run_warnings_json or "[]"),
                        },
                    )
                    result["pipeline_metrics"] = self._with_materialization_metrics(
                        cast(dict[str, Any], result.get("pipeline_metrics", {})),
                        materialization_counters,
                    )
                    self._attach_cache_update_metrics(
                        result["pipeline_metrics"],
                        current_files=current_files,
                        reused_existing_snapshot=True,
                    )
                    return result
                finally:
                    reset_materialization_counters(materialization_token)
        self.index_run_store.mark_started(run_id)
        lock_name = f"index:{snapshot_id}"
        fencing_token: int | None = self.snapshot_store.acquire_lock(
            lock_name, owner_id=run_id, ttl_seconds=600
        )
        if fencing_token is None:
            msg = f"snapshot is currently locked for indexing: {snapshot_id}"
            raise RuntimeError(msg)
        lock_token: int = fencing_token
        stage_id: str | None = None
        materialization_counters = MaterializationCounters()
        materialization_token = set_materialization_counters(materialization_counters)

        try:
            self._reset_embedding_metrics()
            self.index_run_store.mark_status(run_id, "extracting")
            artifact_key = self.snapshot_store.artifact_key_for_snapshot(snapshot_id)
            incremental_plan: dict[str, Any] | None = None
            planned_elements, incremental_plan = self._plan_incremental_elements(
                repo_name=repo_name,
                repo_url=repo_url,
                snapshot_id=snapshot_id,
                snapshot_ref=snapshot_ref,
                ref=ref,
                current_files=current_files,
            )
            if planned_elements is None:
                elements = self.indexer.extract_elements(
                    repo_name=repo_name,
                    repo_url=repo_url,
                    file_infos=current_files,
                )
            else:
                elements = planned_elements
                if incremental_plan is None:
                    msg = "incremental prefilter returned no plan"
                    raise RuntimeError(msg)
                embedded_from_plan_cache = (
                    self._embed_missing_indexed_element_embeddings(elements)
                )
                if embedded_from_plan_cache:
                    incremental_plan["recomputed_changed_embeddings"] = (
                        embedded_from_plan_cache
                    )
                self.logger.info(
                    "incremental prefilter: +%d ~%d -%d =%d, reused=%d, reindexed=%d",
                    incremental_plan["added"],
                    incremental_plan["modified"],
                    incremental_plan["removed"],
                    incremental_plan["unchanged"],
                    incremental_plan["reused_elements"],
                    incremental_plan["reindexed_elements"],
                )

            self.index_run_store.mark_status(run_id, "materializing")
            artifact_delta_mode = incremental_plan is not None and bool(
                incremental_plan.get("artifact_delta_mode")
            )
            graph_delta_context: IncrementalGraphDeltaContext | None = None
            graph_elements = elements
            if artifact_delta_mode and incremental_plan is not None:
                graph_delta_context = self._incremental_graph_delta_context(
                    changed_elements=elements,
                    incremental_plan=incremental_plan,
                    current_files=current_files,
                    repo_name=repo_name,
                    repo_url=repo_url,
                )
                graph_elements = graph_delta_context.elements
                incremental_plan["graph_affected_paths"] = sorted(
                    graph_delta_context.affected_path_keys
                )
                incremental_plan["graph_reusable_paths"] = sorted(
                    graph_delta_context.reusable_path_keys
                )
                plan_changes_payload = incremental_plan.get("plan_changes")
                if isinstance(plan_changes_payload, dict):
                    frontier_payload = plan_changes_payload.get("frontier")
                    if isinstance(frontier_payload, dict):
                        frontier_payload["graph_affected_paths"] = sorted(
                            graph_delta_context.affected_path_keys
                        )
                        frontier_payload["graph_reusable_paths"] = sorted(
                            graph_delta_context.reusable_path_keys
                        )
                if graph_delta_context.missing_affected_path_keys:
                    incremental_plan["graph_delta_degraded_reason"] = (
                        "missing_affected_graph_elements"
                    )
                    incremental_plan["graph_delta_missing_affected_paths"] = sorted(
                        graph_delta_context.missing_affected_path_keys
                    )
                    if isinstance(plan_changes_payload, dict):
                        frontier_payload = plan_changes_payload.get("frontier")
                        if isinstance(frontier_payload, dict):
                            frontier_payload["graph_delta_degraded_reason"] = (
                                "missing_affected_graph_elements"
                            )
                            frontier_payload["graph_delta_missing_affected_paths"] = (
                                sorted(graph_delta_context.missing_affected_path_keys)
                            )
            temp_store = VectorStore(self.config)
            temp_store.initialize(self.embedder.embedding_dim)
            vectors, metadata, all_elem_payloads = (
                self._materialize_indexed_elements_for_storage(
                    elements, snapshot_id=snapshot_id
                )
            )
            if vectors.size == 0 and not artifact_delta_mode:
                msg = "No embeddings produced during indexing"
                raise RuntimeError(msg)
            if vectors.size > 0:
                temp_store.add_vectors(vectors, metadata)

            temp_graph = CodeGraphBuilder(self.config)
            module_resolver = None
            symbol_resolver = None
            try:
                repo_root = self.loader.repo_path or ""
                if artifact_delta_mode and incremental_plan is not None:
                    excluded_path_keys = (
                        set(graph_delta_context.affected_path_keys)
                        if graph_delta_context is not None
                        else {
                            normalize_path(str(path))
                            for path in (
                                list(incremental_plan.get("changed_paths", []) or [])
                                + list(incremental_plan.get("removed_paths", []) or [])
                            )
                            if path
                        }
                    )
                    gib = self._build_incremental_graph_resolver_index(
                        graph_elements=graph_elements,
                        incremental_plan=incremental_plan,
                        excluded_path_keys=excluded_path_keys,
                        repo_root=repo_root,
                    )
                else:
                    gib = GlobalIndexBuilder(self.config)
                    gib.build_maps(graph_elements, repo_root)
                module_resolver = ModuleResolver(gib)
                symbol_resolver = SymbolResolver(gib, module_resolver)
            except Exception as e:
                warnings.append(f"resolver_init_failed: {e}")
            temp_graph.build_graphs(graph_elements, module_resolver, symbol_resolver)

            temp_retriever = HybridRetriever(
                self.config,
                temp_store,
                self.embedder,
                CodeGraphBuilder(self.config),
                repo_root=self.loader.repo_path,
            )
            temp_retriever.index_for_bm25(elements)
            temp_retriever.build_repo_overview_bm25()

            layer1 = pipeline_layers[0]
            layer1["status"] = "succeeded"
            self._finalize_layer_metrics(
                None,
                layer1,
                extra_metrics={
                    "elements": len(elements),
                    "graph_elements": len(graph_elements),
                    "embedded_elements": len(vectors),
                    "bm25_indexed": len(elements),
                    "dependency_graph_nodes": temp_graph.dependency_graph.number_of_nodes(),
                    "inheritance_graph_nodes": temp_graph.inheritance_graph.number_of_nodes(),
                    "call_graph_nodes": temp_graph.call_graph.number_of_nodes(),
                    "dependency_graph_edges": temp_graph.dependency_graph.number_of_edges(),
                    "inheritance_graph_edges": temp_graph.inheritance_graph.number_of_edges(),
                    "call_graph_edges": temp_graph.call_graph.number_of_edges(),
                    "embedding_provider": self._embedding_metrics_payload(),
                },
            )

            self.index_run_store.mark_status(run_id, "validating")
            ast_elements = elements
            if incremental_plan is not None:
                changed_paths_for_ast = {
                    normalize_path(str(path))
                    for path in incremental_plan.get("changed_paths", [])
                    if path
                }
                ast_elements = [
                    elem
                    for elem in elements
                    if normalize_path(elem.relative_path or elem.file_path)
                    in changed_paths_for_ast
                ]
                incremental_plan["ast_ir_rebuilt_elements"] = len(ast_elements)
                incremental_plan["ast_ir_reused_files"] = int(
                    incremental_plan.get("unchanged", 0) or 0
                )
            ast_snapshot: IRSnapshot = build_ir_from_ast(
                repo_name=repo_name,
                snapshot_id=snapshot_id,
                elements=ast_elements,
                repo_root=self.loader.repo_path or "",
                branch=snapshot_ref.get("branch"),
                commit_id=snapshot_ref.get("commit_id"),
                tree_id=snapshot_ref.get("tree_id"),
                file_fingerprints=self._file_info_by_relative_path(current_files),
            )

            ast_snapshot.metadata["repo_root"] = self.loader.repo_path or ""
            ast_snapshot.metadata["pipeline_layers"] = pipeline_layers
            scip_snapshot = None
            scip_artifact_ref = None
            scip_artifact_refs: list[dict[str, Any]] = []
            scip_degraded_reasons: list[str] = []
            layer2 = pipeline_layers[1]
            if enable_scip:
                layer2_start = perf_counter()
                try:
                    scip_artifact_paths: list[str] = []
                    experimental_scip_languages: list[str] = []
                    scip_data: SCIPIndex | None = None
                    if scip_artifact_path:
                        scip_data = self._load_scip_artifact(scip_artifact_path)
                        scip_artifact_paths = [scip_artifact_path]
                        scip_snapshot = build_ir_from_scip(
                            repo_name=repo_name,
                            snapshot_id=snapshot_id,
                            scip_index=scip_data,
                            branch=snapshot_ref.get("branch"),
                            commit_id=snapshot_ref.get("commit_id"),
                            tree_id=snapshot_ref.get("tree_id"),
                        )
                    else:
                        scip_scope = self._incremental_scip_scope(incremental_plan)
                        scip_degraded_reasons = self._scip_degraded_reasons(
                            incremental_plan,
                            scip_scope,
                        )
                        if scip_scope and scip_scope["mode"] == "skip":
                            warning = f"incremental_scip_skipped:{scip_scope['reason']}"
                            warnings.append(warning)
                            layer2["warnings"].append(warning)
                            scip_snapshot = IRSnapshot(
                                repo_name=repo_name,
                                snapshot_id=snapshot_id,
                                branch=snapshot_ref.get("branch"),
                                commit_id=snapshot_ref.get("commit_id"),
                                tree_id=snapshot_ref.get("tree_id"),
                                metadata={
                                    "scip_languages": [],
                                    "experimental_scip_languages": [],
                                    "incremental_scip_scope": scip_scope,
                                    "scip_degraded_reasons": list(
                                        scip_degraded_reasons
                                    ),
                                },
                            )
                        elif scip_scope and scip_scope["mode"] == "package":
                            scoped_snapshot, scoped_languages = (
                                self._run_scoped_scip_frontier(
                                    snapshot=ast_snapshot,
                                    repo_name=repo_name,
                                    scope_kind="package",
                                    scope_roots=cast(
                                        list[str], scip_scope.get("scope_roots", [])
                                    ),
                                    target_paths=set(
                                        cast(
                                            list[str],
                                            scip_scope.get("target_paths", []),
                                        )
                                    ),
                                    warnings=warnings,
                                )
                            )
                            experimental_scip_languages = (
                                self._experimental_scip_languages(scoped_languages)
                            )
                            if experimental_scip_languages:
                                warning = "experimental_scip_languages: " + ", ".join(
                                    sorted(experimental_scip_languages)
                                )
                                warnings.append(warning)
                                layer2["warnings"].append(warning)
                            if scoped_snapshot is None:
                                warning = "incremental_scip_no_scoped_artifacts"
                                warnings.append(warning)
                                layer2["warnings"].append(warning)
                                scip_snapshot = IRSnapshot(
                                    repo_name=repo_name,
                                    snapshot_id=snapshot_id,
                                    branch=snapshot_ref.get("branch"),
                                    commit_id=snapshot_ref.get("commit_id"),
                                    tree_id=snapshot_ref.get("tree_id"),
                                    metadata={
                                        "scip_languages": scoped_languages,
                                        "experimental_scip_languages": list(
                                            experimental_scip_languages
                                        ),
                                        "incremental_scip_scope": scip_scope,
                                        "scip_degraded_reasons": list(
                                            scip_degraded_reasons
                                        ),
                                    },
                                )
                            else:
                                scoped_snapshot.metadata["scip_languages"] = (
                                    scoped_languages
                                )
                                scoped_snapshot.metadata[
                                    "experimental_scip_languages"
                                ] = list(experimental_scip_languages)
                                scoped_snapshot.metadata["incremental_scip_scope"] = (
                                    scip_scope
                                )
                                scoped_snapshot.metadata["scip_degraded_reasons"] = (
                                    list(scip_degraded_reasons)
                                )
                                scip_snapshot = scoped_snapshot
                        else:
                            out_dir = tempfile.mkdtemp(prefix="fastcode_scip_")
                            self._profile_record_temp_dir(out_dir)
                            scip_indexes = []
                            detected_languages = list(
                                self._detect_scip_languages_from_file_infos(
                                    cast(Sequence[ScipFileInfoView], current_files)
                                )
                            )
                            experimental_scip_languages = (
                                self._experimental_scip_languages(detected_languages)
                            )
                            if experimental_scip_languages:
                                warning = "experimental_scip_languages: " + ", ".join(
                                    sorted(experimental_scip_languages)
                                )
                                warnings.append(warning)
                                layer2["warnings"].append(warning)
                            for language in detected_languages:
                                scip_index = self._run_scip_for_language(
                                    language, self.loader.repo_path or "", out_dir
                                )
                                if scip_index is not None:
                                    scip_indexes.append((language, scip_index))
                                    artifact_path = os.path.join(
                                        out_dir, f"{language}.scip"
                                    )
                                    if os.path.exists(artifact_path):
                                        scip_artifact_paths.append(artifact_path)
                            if not scip_indexes:
                                out_path = os.path.join(out_dir, "index.scip.json")
                                self._run_scip_python_index(
                                    self.loader.repo_path or "", out_path
                                )
                                scip_indexes.append(
                                    ("python", self._load_scip_artifact(out_path))
                                )
                                scip_artifact_paths.append(out_path)
                            scip_snapshots = [
                                build_ir_from_scip(
                                    repo_name=repo_name,
                                    snapshot_id=snapshot_id,
                                    scip_index=index,
                                    branch=snapshot_ref.get("branch"),
                                    commit_id=snapshot_ref.get("commit_id"),
                                    tree_id=snapshot_ref.get("tree_id"),
                                    language_hint=language,
                                )
                                for language, index in scip_indexes
                            ]
                            scip_snapshot = IRSnapshot(
                                repo_name=repo_name,
                                snapshot_id=snapshot_id,
                                branch=snapshot_ref.get("branch"),
                                commit_id=snapshot_ref.get("commit_id"),
                                tree_id=snapshot_ref.get("tree_id"),
                                units=[
                                    unit
                                    for snap in scip_snapshots
                                    for unit in snap.units
                                ],
                                supports=[
                                    support
                                    for snap in scip_snapshots
                                    for support in snap.supports
                                ],
                                relations=[
                                    relation
                                    for snap in scip_snapshots
                                    for relation in snap.relations
                                ],
                                embeddings=[
                                    embedding
                                    for snap in scip_snapshots
                                    for embedding in snap.embeddings
                                ],
                                metadata={
                                    "scip_languages": [
                                        language for language, _ in scip_indexes
                                    ],
                                    "experimental_scip_languages": list(
                                        experimental_scip_languages
                                    ),
                                    "incremental_scip_scope": {
                                        "mode": "full",
                                        "reason": "no_incremental_plan",
                                    },
                                    "scip_degraded_reasons": list(
                                        scip_degraded_reasons
                                    ),
                                },
                            )
                            scip_data = scip_indexes[0][1]
                    scip_snapshot.metadata["scip_degraded_reasons"] = list(
                        scip_degraded_reasons
                    )
                    # Preserve ALL generated SCIP artifacts (not just the first)
                    if scip_artifact_paths:
                        import shutil

                        scip_dir = os.path.join(
                            self.snapshot_store.snapshot_dir(snapshot_id),
                            "scip",
                        )
                        ensure_dir(scip_dir)
                        preserved_paths: list[str] = []
                        for artifact_src in scip_artifact_paths:
                            if not os.path.exists(artifact_src):
                                continue
                            basename = os.path.basename(artifact_src)
                            preserved_path = os.path.join(scip_dir, basename)
                            shutil.copy2(artifact_src, preserved_path)
                            preserved_paths.append(preserved_path)
                        # Compute checksum from the primary artifact for the ref
                        primary_path = preserved_paths[0] if preserved_paths else None
                        artifact_records: list[dict[str, Any]] = []
                        for artifact_path in preserved_paths:
                            digest = hashlib.sha256()
                            with open(artifact_path, "rb") as fh:
                                for chunk in iter(lambda: fh.read(8192), b""):
                                    digest.update(chunk)
                            language_name = os.path.splitext(
                                os.path.basename(artifact_path)
                            )[0]
                            artifact_records.append(
                                {
                                    "indexer_name": (
                                        (scip_data.indexer_name or "scip-python")
                                        if scip_data is not None
                                        else "scip-python"
                                    ),
                                    "indexer_version": (
                                        scip_data.indexer_version
                                        if scip_data is not None
                                        else None
                                    ),
                                    "artifact_path": artifact_path,
                                    "checksum": digest.hexdigest(),
                                    "language": language_name,
                                }
                            )
                        primary_path = preserved_paths[0] if preserved_paths else None
                        if primary_path:
                            scip_artifact_refs = (
                                self.snapshot_store.save_scip_artifact_refs(
                                    snapshot_id=snapshot_id,
                                    artifacts=artifact_records,
                                )
                            )
                            scip_artifact_ref = scip_artifact_refs[0]
                    layer2["status"] = "succeeded"
                    self._finalize_layer_metrics(
                        scip_snapshot,
                        layer2,
                        extra_metrics={
                            "duration_ms": round(
                                (perf_counter() - layer2_start) * 1000, 3
                            ),
                            "artifact_count": len(scip_artifact_paths),
                            "scip_enabled": True,
                            "scip_languages": list(
                                (scip_snapshot.metadata or {}).get("scip_languages", [])
                            ),
                            "experimental_scip_languages": list(
                                (scip_snapshot.metadata or {}).get(
                                    "experimental_scip_languages", []
                                )
                            ),
                            "experimental_language_count": len(
                                (scip_snapshot.metadata or {}).get(
                                    "experimental_scip_languages", []
                                )
                            ),
                            "scip_degraded_reasons": list(scip_degraded_reasons),
                        },
                    )
                except Exception as e:
                    degraded = True
                    message = f"scip_unavailable_or_failed: {e}"
                    warnings.append(message)
                    layer2["status"] = "degraded"
                    layer2["reason"] = "scip_failed"
                    layer2["warnings"].append(message)
                    self._finalize_layer_metrics(
                        None,
                        layer2,
                        extra_metrics={
                            "scip_enabled": True,
                            "artifact_count": 0,
                            "error": str(e),
                            "experimental_scip_languages": [],
                            "experimental_language_count": 0,
                            "scip_degraded_reasons": ["scip_failed"],
                        },
                    )
            else:
                layer2["warnings"].append("layer_disabled: enable_scip=false")
                self._finalize_layer_metrics(
                    None,
                    layer2,
                    extra_metrics={
                        "scip_enabled": False,
                        "artifact_count": 0,
                        "experimental_scip_languages": [],
                        "experimental_language_count": 0,
                        "scip_degraded_reasons": ["scip_disabled"],
                    },
                )

            merged_snapshot = merge_ir(ast_snapshot, scip_snapshot)
            merged_snapshot.metadata["repo_root"] = self.loader.repo_path or ""
            merged_snapshot.metadata["pipeline_layers"] = pipeline_layers
            merged_snapshot.metadata["pipeline_layer_contract"] = {
                "layer_1": "plain_ast_embedding",
                "layer_2": "unified_ir_scip_merge",
                "layer_3": "language_specific_semantic_upgrade",
                "never_silent_fallback": True,
            }

            # Incremental update: if a previous snapshot exists for this branch,
            # diff blob_oids and merge only changed file content.
            incremental_change_set = None
            ref_name_for_inc = snapshot_ref.get("branch") or ref or "HEAD"
            prev_manifest = self.manifest_store.get_branch_manifest_record(
                repo_name, ref_name_for_inc
            )
            if prev_manifest:
                prev_snap_id = prev_manifest.snapshot_id
                if prev_snap_id and prev_snap_id != snapshot_id:
                    prev_snapshot = self.snapshot_store.load_snapshot(prev_snap_id)
                    if prev_snapshot:
                        if incremental_plan is not None:
                            incremental_change_set = FileChangeSet(
                                added=cast(
                                    list[str], incremental_plan.get("added_paths", [])
                                ),
                                removed=cast(
                                    list[str],
                                    incremental_plan.get("removed_paths", []),
                                ),
                                modified=cast(
                                    list[str],
                                    incremental_plan.get("modified_paths", []),
                                ),
                                unchanged=[
                                    doc.path
                                    for doc in prev_snapshot.documents
                                    if doc.path
                                    not in set(
                                        cast(
                                            list[str],
                                            incremental_plan.get("removed_paths", []),
                                        )
                                        + cast(
                                            list[str],
                                            incremental_plan.get("modified_paths", []),
                                        )
                                    )
                                ],
                            )
                        else:
                            incremental_change_set = diff_changed_files(
                                prev_snapshot, merged_snapshot
                            )
                        changed_count = (
                            len(incremental_change_set.added)
                            + len(incremental_change_set.modified)
                            + len(incremental_change_set.removed)
                        )
                        if changed_count == 0:
                            self.logger.info(
                                "incremental: no file changes detected vs %s",
                                prev_snap_id,
                            )
                            merged_snapshot = IRSnapshot(
                                repo_name=merged_snapshot.repo_name,
                                snapshot_id=merged_snapshot.snapshot_id,
                                branch=merged_snapshot.branch,
                                commit_id=merged_snapshot.commit_id,
                                tree_id=merged_snapshot.tree_id,
                                units=list(prev_snapshot.units),
                                supports=list(prev_snapshot.supports),
                                relations=list(prev_snapshot.relations),
                                embeddings=list(prev_snapshot.embeddings),
                                metadata=merged_snapshot.metadata,
                            )
                        else:
                            self.logger.info(
                                "incremental: %d added, %d modified, %d removed "
                                "(%d unchanged) vs %s",
                                len(incremental_change_set.added),
                                len(incremental_change_set.modified),
                                len(incremental_change_set.removed),
                                len(incremental_change_set.unchanged),
                                prev_snap_id,
                            )
                            merged_snapshot = apply_incremental_update(
                                prev_snapshot,
                                merged_snapshot,
                                incremental_change_set,
                                preserve_sources_for_modified_paths=(
                                    self._preservable_incremental_sources(
                                        incremental_plan
                                    )
                                ),
                            )

            target_paths = (
                set(incremental_change_set.added + incremental_change_set.modified)
                if incremental_change_set is not None
                else {elem.relative_path or elem.file_path for elem in elements}
            )
            layer3 = pipeline_layers[2]
            layer3_start = perf_counter()
            merged_snapshot = self._apply_semantic_resolvers(
                snapshot=merged_snapshot,
                elements=elements,
                graph_context=temp_graph,
                target_paths=target_paths,
                warnings=warnings,
            )
            semantic_runs = list(
                (merged_snapshot.metadata or {}).get("semantic_resolver_runs", [])
            )
            layer3_quality = self._layer3_quality_metrics(merged_snapshot)
            if semantic_runs and (
                layer3_quality["semantic_relations"] > 0
                or layer3_quality["anchored_relations"] > 0
                or layer3_quality["relations_with_pending_capabilities"] > 0
            ):
                layer3["status"] = "succeeded"
                layer3["reason"] = None
            else:
                layer3["status"] = "degraded"
                layer3["reason"] = (
                    "no_semantic_resolver_runs_recorded"
                    if not semantic_runs
                    else "semantic_resolver_runs_without_graph_upgrade_signal"
                )
                layer3["warnings"].append(layer3["reason"])
                degraded = True
            self._finalize_layer_metrics(
                merged_snapshot,
                layer3,
                extra_metrics={
                    "duration_ms": round((perf_counter() - layer3_start) * 1000, 3),
                    "target_paths": len(target_paths),
                    "resolver_runs": len(semantic_runs),
                    **layer3_quality,
                },
            )
            merged_snapshot.metadata["pipeline_layers"] = pipeline_layers
            merged_snapshot.metadata["pipeline_metrics"] = {
                "never_silent_fallback": True,
                "degraded": degraded,
                "warning_count": len(warnings),
                "layer_statuses": {
                    layer["name"]: layer["status"] for layer in pipeline_layers
                },
                "file_inventory": self._file_inventory_metrics_payload(current_files),
            }
            errors = validate_snapshot(merged_snapshot)
            if errors:
                msg = f"IR validation failed: {errors[:5]}"
                raise RuntimeError(msg)

            self.snapshot_symbol_index.register_snapshot(merged_snapshot)

            doc_chunks_payload: list[dict[str, Any]] = []
            doc_mentions_payload: list[dict[str, Any]] = []
            doc_elements_payload: list[dict[str, Any]] = []
            if self._should_ingest_docs(graph_runtime):
                try:
                    doc_ingest = self.doc_ingester.ingest(
                        repo_path=self.loader.repo_path or "",
                        repo_name=repo_name,
                        snapshot_id=snapshot_id,
                        snapshot=merged_snapshot,
                    )
                    doc_chunks_payload = [
                        {
                            "chunk_id": c.chunk_id,
                            "snapshot_id": c.snapshot_id,
                            "repo_name": c.repo_name,
                            "path": c.path,
                            "title": c.title,
                            "heading": c.heading,
                            "doc_type": c.doc_type,
                            "content": c.text,
                            "start_line": c.start_line,
                            "end_line": c.end_line,
                        }
                        for c in (doc_ingest.get("chunks") or [])
                    ]
                    doc_mentions_payload = list(doc_ingest.get("mentions") or [])
                    doc_elements_payload = list(doc_ingest.get("elements") or [])
                except Exception as e:
                    warnings.append(f"doc_ingestion_failed: {e}")

            # Backfill canonical IR symbol IDs into vector metadata for
            # IR-aware retrieval.
            ast_id_to_ir: dict[str, str] = {}
            for sym in merged_snapshot.symbols:
                meta = sym.metadata or {}
                ast_elem_id = meta.get("ast_element_id")
                if ast_elem_id:
                    ast_id_to_ir[str(ast_elem_id)] = sym.symbol_id
                for alias in (
                    meta.get("aliases", [])
                    if isinstance(meta.get("aliases", []), list)
                    else []
                ):
                    # alias can be an AST symbol id; keep as an extra hint only
                    if alias:
                        ast_id_to_ir.setdefault(str(alias), sym.symbol_id)
            for row in temp_store.metadata:
                elem_id = row.get("id")
                ir_symbol_id = ast_id_to_ir.get(str(elem_id))
                if ir_symbol_id:
                    row["ir_symbol_id"] = ir_symbol_id
                    row_meta = row.get("metadata") or {}
                    row_meta["ir_symbol_id"] = ir_symbol_id
                    row["metadata"] = row_meta
            for elem in elements:
                ir_symbol_id = ast_id_to_ir.get(str(elem.id))
                if ir_symbol_id:
                    elem.metadata["ir_symbol_id"] = ir_symbol_id

            self.index_run_store.mark_status(run_id, "persisting")
            if not self.snapshot_store.validate_fencing_token(lock_name, lock_token):
                msg = f"stale_lock_detected_for_snapshot:{snapshot_id}"
                raise RuntimeError(msg)

            # Artifact persistence — only after fencing confirmed valid.
            self._invalidate_loaded_artifact_handle(artifact_key)
            artifact_reuse_stats: dict[str, Any] = {}
            incremental_previous_artifact_key = (
                str(incremental_plan.get("previous_artifact_key") or "")
                if incremental_plan is not None
                else ""
            )
            reusable_path_keys = (
                {
                    normalize_path(str(path))
                    for path in incremental_plan.get("unchanged_paths", [])
                    if path
                }
                if incremental_plan is not None and incremental_previous_artifact_key
                else set()
            )
            active_incremental_plan = incremental_plan
            vector_persist_dir = cast(
                str | None, getattr(temp_store, "persist_dir", None)
            )
            lexical_persist_dir = cast(
                str | None, getattr(temp_retriever, "persist_dir", None)
            )
            graph_persist_dir = cast(
                str | None, getattr(self.graph_artifact_store, "persist_dir", None)
            )
            if (
                incremental_previous_artifact_key
                and active_incremental_plan is not None
            ):
                publish_vector_delta = getattr(temp_store, "publish_delta", None)
                save_vector_incremental = getattr(temp_store, "save_incremental", None)
                with self._profile_store_surface(
                    pipeline_profile, "vector", vector_persist_dir
                ):
                    if callable(publish_vector_delta):
                        artifact_reuse_stats.update(
                            cast(
                                dict[str, Any],
                                publish_vector_delta(
                                    artifact_key,
                                    previous_name=incremental_previous_artifact_key,
                                    reusable_path_keys=reusable_path_keys,
                                    snapshot_id=snapshot_id,
                                ),
                            )
                        )
                    elif callable(save_vector_incremental):
                        artifact_reuse_stats.update(
                            cast(
                                dict[str, Any],
                                save_vector_incremental(
                                    artifact_key,
                                    previous_name=incremental_previous_artifact_key,
                                    reusable_path_keys=reusable_path_keys,
                                ),
                            )
                        )
                    else:
                        temp_store.save(artifact_key)
                save_bm25_incremental = getattr(
                    temp_retriever, "save_bm25_incremental", None
                )
                publish_bm25_delta = getattr(temp_retriever, "publish_bm25_delta", None)
                with self._profile_store_surface(
                    pipeline_profile, "lexical", lexical_persist_dir
                ):
                    if callable(publish_bm25_delta):
                        artifact_reuse_stats.update(
                            cast(
                                dict[str, Any],
                                publish_bm25_delta(
                                    artifact_key,
                                    previous_name=incremental_previous_artifact_key,
                                    reusable_path_keys=reusable_path_keys,
                                ),
                            )
                        )
                    elif callable(save_bm25_incremental):
                        artifact_reuse_stats.update(
                            cast(
                                dict[str, Any],
                                save_bm25_incremental(
                                    artifact_key,
                                    previous_name=incremental_previous_artifact_key,
                                    reusable_path_keys=reusable_path_keys,
                                ),
                            )
                        )
                    else:
                        temp_retriever.save_bm25(artifact_key)
                graph_reuse_path_keys = (
                    set(graph_delta_context.reusable_path_keys)
                    if graph_delta_context is not None
                    else set(reusable_path_keys)
                )
                publish_graph_delta = getattr(
                    self.graph_artifact_store, "publish_delta", None
                )
                save_graph_incremental = getattr(
                    self.graph_artifact_store, "save_incremental", None
                )
                with self._profile_store_surface(
                    pipeline_profile, "graph", graph_persist_dir
                ):
                    if callable(publish_graph_delta):
                        artifact_reuse_stats.update(
                            cast(
                                dict[str, Any],
                                publish_graph_delta(
                                    temp_graph,
                                    artifact_key,
                                    previous_name=incremental_previous_artifact_key,
                                    reusable_path_keys=graph_reuse_path_keys,
                                ),
                            )
                        )
                    elif callable(save_graph_incremental):
                        artifact_reuse_stats.update(
                            cast(
                                dict[str, Any],
                                save_graph_incremental(
                                    temp_graph,
                                    artifact_key,
                                    previous_name=incremental_previous_artifact_key,
                                    reusable_path_keys=graph_reuse_path_keys,
                                ),
                            )
                        )
                    else:
                        self.graph_artifact_store.save(temp_graph, artifact_key)
                        artifact_reuse_stats["graph_fallback_reason"] = (
                            "graph_delta_api_unavailable"
                        )
            else:
                with self._profile_store_surface(
                    pipeline_profile, "vector", vector_persist_dir
                ):
                    temp_store.save(artifact_key)
                with self._profile_store_surface(
                    pipeline_profile, "lexical", lexical_persist_dir
                ):
                    temp_retriever.save_bm25(artifact_key)
                with self._profile_store_surface(
                    pipeline_profile, "graph", graph_persist_dir
                ):
                    self.graph_artifact_store.save(temp_graph, artifact_key)
            with self._profile_store_surface(
                pipeline_profile,
                "unit_artifact",
                getattr(self.vector_store, "persist_dir", None),
            ):
                file_manifest = (
                    self._build_file_manifest_delta(
                        artifact_key=artifact_key,
                        elements=elements,
                        repo_root=self.loader.repo_path or "",
                        current_files=current_files,
                        incremental_plan=active_incremental_plan,
                    )
                    if active_incremental_plan is not None
                    and incremental_previous_artifact_key
                    else self._build_file_manifest(
                        elements,
                        self.loader.repo_path or "",
                        current_files=current_files,
                    )
                )
                self._save_file_manifest(
                    artifact_key,
                    file_manifest,
                )

            snapshot_store_dir = getattr(self.snapshot_store, "persist_dir", None)
            snapshot_db_path = getattr(self.snapshot_store, "db_path", None)
            with self._profile_store_surface(
                pipeline_profile, "ir", snapshot_store_dir, snapshot_db_path
            ):
                snapshot_metadata = {
                    "run_id": run_id,
                    "artifact_key": artifact_key,
                    "warnings": warnings,
                    "scip_artifact_ref": scip_artifact_ref,
                    "scip_artifact_refs": scip_artifact_refs,
                    "pipeline_layers": pipeline_layers,
                    "pipeline_metrics": merged_snapshot.metadata.get(
                        "pipeline_metrics", {}
                    ),
                    "fencing_token": lock_token,
                }
                save_snapshot_delta = getattr(
                    self.snapshot_store,
                    "save_snapshot_delta",
                    None,
                )
                if (
                    callable(save_snapshot_delta)
                    and active_incremental_plan is not None
                    and active_incremental_plan.get("previous_snapshot_id")
                ):
                    save_snapshot_delta(
                        merged_snapshot,
                        previous_snapshot_id=str(
                            active_incremental_plan.get("previous_snapshot_id")
                        ),
                        changed_paths=[
                            normalize_path(str(path))
                            for path in (
                                list(
                                    active_incremental_plan.get("added_paths", []) or []
                                )
                                + list(
                                    active_incremental_plan.get("modified_paths", [])
                                    or []
                                )
                            )
                        ],
                        removed_paths=[
                            normalize_path(str(path))
                            for path in (
                                active_incremental_plan.get("removed_paths", []) or []
                            )
                        ],
                        metadata=snapshot_metadata,
                    )
                else:
                    self.snapshot_store.save_snapshot(
                        merged_snapshot,
                        metadata=snapshot_metadata,
                    )
            with self._profile_store_surface(
                pipeline_profile, "unit_artifact", snapshot_db_path
            ):
                unit_artifact_rows = self._unit_artifact_rows(elements)
                changed_artifact_paths = (
                    [
                        normalize_path(str(path))
                        for path in (
                            list(active_incremental_plan.get("added_paths", []) or [])
                            + list(
                                active_incremental_plan.get("modified_paths", []) or []
                            )
                        )
                        if path
                    ]
                    if active_incremental_plan is not None
                    else []
                )
                removed_artifact_paths = (
                    [
                        normalize_path(str(path))
                        for path in (
                            active_incremental_plan.get("removed_paths", []) or []
                        )
                        if path
                    ]
                    if active_incremental_plan is not None
                    else []
                )
                file_ir_changed_paths = (
                    SnapshotStore.file_ir_shard_paths_for_paths(
                        merged_snapshot,
                        changed_artifact_paths,
                        removed_paths=removed_artifact_paths,
                    )
                    if active_incremental_plan is not None
                    else []
                )
                file_ir_shards = SnapshotStore.file_ir_shard_payloads(
                    merged_snapshot,
                    paths=file_ir_changed_paths
                    if active_incremental_plan is not None
                    else None,
                )
                reused_file_ir_shards: list[dict[str, Any]] = []
                file_ir_content_reuse: dict[str, int | str] = {
                    "mode": "skipped",
                    "reused_shards": 0,
                }
                file_ir_content_publish_paths: Sequence[str] | None = None
                parsed_element_content_publish_paths: Sequence[str] | None = None
                if active_incremental_plan is not None:
                    reused_file_ir_shards, file_ir_content_reuse = (
                        self._reuse_file_ir_content_artifacts(
                            repo_name=str(repo_name),
                            snapshot_id=str(snapshot_id),
                            current_files=current_files,
                            incremental_plan=active_incremental_plan,
                            excluded_paths=[
                                *file_ir_changed_paths,
                                *removed_artifact_paths,
                            ],
                        )
                    )
                    file_ir_content_publish_paths = changed_artifact_paths
                    parsed_element_content_publish_paths = changed_artifact_paths
                parsed_element_content_publish = (
                    self._publish_parsed_element_content_artifacts(
                        repo_name=str(repo_name),
                        elements=elements,
                        current_files=current_files,
                        paths=parsed_element_content_publish_paths,
                    )
                )
                file_ir_content_publish = self._publish_file_ir_content_artifacts(
                    repo_name=str(repo_name),
                    shards=file_ir_shards,
                    current_files=current_files,
                    paths=file_ir_content_publish_paths,
                )
                embedding_ref_content_publish = (
                    self._publish_embedding_ref_content_artifacts(
                        repo_name=str(repo_name),
                        rows=unit_artifact_rows,
                        current_files=current_files,
                        paths=file_ir_content_publish_paths,
                    )
                )
                semantic_fact_content_publish = (
                    self._publish_semantic_fact_content_artifacts(
                        repo_name=str(repo_name),
                        shards=file_ir_shards,
                        current_files=current_files,
                        paths=file_ir_content_publish_paths,
                    )
                )
                publish_units_delta = getattr(
                    self.unit_artifact_store, "publish_snapshot_units_delta", None
                )
                publish_file_ir_delta = getattr(
                    self.unit_artifact_store,
                    "publish_snapshot_file_ir_shards_delta",
                    None,
                )
                if (
                    callable(publish_units_delta)
                    and active_incremental_plan is not None
                    and active_incremental_plan.get("previous_snapshot_id")
                ):
                    unit_artifact_persistence = cast(
                        dict[str, Any],
                        publish_units_delta(
                            snapshot_id=snapshot_id,
                            previous_snapshot_id=str(
                                active_incremental_plan.get("previous_snapshot_id")
                            ),
                            changed_paths=changed_artifact_paths,
                            removed_paths=removed_artifact_paths,
                            elements=unit_artifact_rows,
                        ),
                    )
                    if callable(publish_file_ir_delta):
                        unit_artifact_persistence["file_ir_shards"] = cast(
                            dict[str, Any],
                            publish_file_ir_delta(
                                snapshot_id=snapshot_id,
                                previous_snapshot_id=str(
                                    active_incremental_plan.get("previous_snapshot_id")
                                ),
                                changed_paths=file_ir_changed_paths,
                                removed_paths=removed_artifact_paths,
                                shards=file_ir_shards,
                                reused_shards=reused_file_ir_shards,
                            ),
                        )
                        unit_artifact_persistence["file_ir_shards"][
                            "content_artifact_reuse"
                        ] = dict(file_ir_content_reuse)
                        unit_artifact_persistence["file_ir_shards"][
                            "content_artifact_publish"
                        ] = dict(file_ir_content_publish)
                    unit_artifact_persistence["embedding_ref_artifacts"] = dict(
                        embedding_ref_content_publish
                    )
                    unit_artifact_persistence["semantic_fact_artifacts"] = dict(
                        semantic_fact_content_publish
                    )
                    unit_artifact_persistence["parsed_element_artifacts"] = dict(
                        parsed_element_content_publish
                    )
                else:
                    unit_artifact_persistence: dict[str, Any] = {"mode": "full"}
                    self.unit_artifact_store.replace_snapshot_units(
                        snapshot_id=snapshot_id,
                        elements=unit_artifact_rows,
                    )
                    replace_file_ir_shards = getattr(
                        self.unit_artifact_store,
                        "replace_snapshot_file_ir_shards",
                        None,
                    )
                    if callable(replace_file_ir_shards):
                        unit_artifact_persistence["file_ir_shards"] = cast(
                            dict[str, Any],
                            replace_file_ir_shards(
                                snapshot_id=snapshot_id,
                                shards=file_ir_shards,
                            ),
                        )
                        unit_artifact_persistence["file_ir_shards"][
                            "content_artifact_publish"
                        ] = dict(file_ir_content_publish)
                    unit_artifact_persistence["embedding_ref_artifacts"] = dict(
                        embedding_ref_content_publish
                    )
                    unit_artifact_persistence["semantic_fact_artifacts"] = dict(
                        semantic_fact_content_publish
                    )
                    unit_artifact_persistence["parsed_element_artifacts"] = dict(
                        parsed_element_content_publish
                    )
            with self._profile_store_surface(
                pipeline_profile, "relational_fact", snapshot_db_path
            ):
                self.snapshot_store.import_git_backbone(
                    merged_snapshot, git_meta=git_meta
                )
                relational_fact_persistence = self._save_relational_facts_for_index(
                    merged_snapshot,
                    active_incremental_plan,
                )
            if doc_chunks_payload:
                mentions_by_chunk: dict[str, list[dict[str, Any]]] = {}
                for mention in doc_mentions_payload:
                    chunk_id = mention.get("chunk_id")
                    if not chunk_id:
                        continue
                    mentions_by_chunk.setdefault(str(chunk_id), []).append(
                        dict(mention)
                    )
                for elem in doc_elements_payload:
                    chunk_id = elem.get("id")
                    elem_meta = elem.get("metadata") or {}
                    elem_meta["trace_links"] = mentions_by_chunk.get(str(chunk_id), [])
                    elem["metadata"] = elem_meta
                self.snapshot_store.save_design_documents(
                    snapshot_id=snapshot_id,
                    repo_name=repo_name,
                    chunks=doc_chunks_payload,
                    mentions=doc_mentions_payload,
                )
            with self._profile_store_surface(
                pipeline_profile, "ir_graph", snapshot_store_dir, snapshot_db_path
            ):
                _ir_graphs, ir_graph_persistence = self._save_ir_graphs_for_index(
                    merged_snapshot,
                    active_incremental_plan,
                )
            with self._profile_store_surface(
                pipeline_profile, "ir", snapshot_store_dir, snapshot_db_path
            ):
                stage_id = self.snapshot_store.stage_snapshot(
                    merged_snapshot,
                    metadata={"run_id": run_id, "artifact_key": artifact_key},
                )
            all_pg_elements: list[dict[str, Any]] = cast(
                list[dict[str, Any]], list(all_elem_payloads)
            )
            if doc_elements_payload:
                all_pg_elements.extend(doc_elements_payload)
            if not self.pg_retrieval_store:
                msg = "pg_retrieval_store not initialized"
                raise RuntimeError(msg)
            with self._profile_store_surface(pipeline_profile, "pg", snapshot_db_path):
                publish_pg_delta = getattr(
                    self.pg_retrieval_store, "publish_elements_delta", None
                )
                if (
                    callable(publish_pg_delta)
                    and active_incremental_plan is not None
                    and active_incremental_plan.get("previous_snapshot_id")
                ):
                    pg_retrieval_persistence = publish_pg_delta(
                        snapshot_id=snapshot_id,
                        previous_snapshot_id=str(
                            active_incremental_plan.get("previous_snapshot_id")
                        ),
                        changed_paths=[
                            normalize_path(str(path))
                            for path in (
                                list(
                                    active_incremental_plan.get("added_paths", []) or []
                                )
                                + list(
                                    active_incremental_plan.get("modified_paths", [])
                                    or []
                                )
                            )
                            if path
                        ],
                        removed_paths=[
                            normalize_path(str(path))
                            for path in active_incremental_plan.get("removed_paths", [])
                            or []
                            if path
                        ],
                        elements=all_pg_elements,
                    )
                else:
                    pg_retrieval_persistence = {"mode": "full"}
                    self.pg_retrieval_store.upsert_elements(
                        snapshot_id=snapshot_id,
                        elements=all_pg_elements,
                    )
            self._sync_doc_overlay(
                graph_runtime,
                chunks=doc_chunks_payload,
                mentions=doc_mentions_payload,
                warnings=warnings,
            )

            self._load_artifacts_by_key(artifact_key)

            # Store result in FastCode.loaded_repositories
            if get_loaded_repositories is not None:
                get_loaded_repositories()[repo_name] = repo_info

            manifest = None
            status = "degraded" if degraded else "succeeded"

            if publish:
                self.index_run_store.mark_status(run_id, "publishing")
                ref_name = snapshot_ref.get("branch") or ref or "HEAD"
                previous_snapshot_symbols = self._previous_snapshot_symbol_versions(
                    repo_name=repo_name,
                    ref_name=ref_name,
                    current_snapshot_id=snapshot_id,
                )
                manifest = self.manifest_store.publish_record(
                    repo_name=repo_name,
                    ref_name=ref_name,
                    snapshot_id=snapshot_id,
                    index_run_id=run_id,
                    status="published",
                )
                if self.terminus_publisher.is_configured():
                    try:
                        self.terminus_publisher.publish_snapshot_lineage_for_snapshot(
                            snapshot=merged_snapshot,
                            manifest=manifest,
                            git_meta=git_meta,
                            previous_snapshot_symbols=previous_snapshot_symbols,
                            idempotency_key=f"lineage:{run_id}:{snapshot_id}",
                        )
                        status = "published" if not degraded else "degraded"
                    except Exception as e:
                        warnings.append(f"terminus_publish_failed: {e}")
                        self.index_run_store.enqueue_publish_retry(
                            run_id=run_id,
                            snapshot_id=snapshot_id,
                            manifest_id=manifest.manifest_id,
                            error_message=str(e),
                        )
                        status = "publish_pending"
                else:
                    warnings.append("terminus_not_configured")
                if stage_id:
                    self.snapshot_store.promote_staged_snapshot(
                        snapshot_id=snapshot_id, stage_id=stage_id
                    )

            pipeline_metrics = self._with_materialization_metrics(
                cast(
                    dict[str, Any],
                    merged_snapshot.metadata.get("pipeline_metrics", {}),
                ),
                materialization_counters,
            )
            embedding_metrics = self._embedding_metrics_payload()
            if embedding_metrics:
                pipeline_metrics["embedding_provider"] = embedding_metrics
            self._attach_cache_update_metrics(
                pipeline_metrics,
                embedding_metrics=embedding_metrics,
                incremental_plan=incremental_plan,
                current_files=current_files,
            )
            self._attach_pipeline_profile(pipeline_metrics, pipeline_profile)
            merged_snapshot.metadata["pipeline_metrics"] = pipeline_metrics
            plan_changes_payload = (
                incremental_plan.get("plan_changes")
                if incremental_plan is not None
                else None
            )
            snapshot_metadata = {
                "run_id": run_id,
                "artifact_key": artifact_key,
                "warnings": warnings,
                "scip_artifact_ref": scip_artifact_ref,
                "scip_artifact_refs": scip_artifact_refs,
                "pipeline_layers": pipeline_layers,
                "pipeline_metrics": merged_snapshot.metadata.get(
                    "pipeline_metrics", {}
                ),
                "fencing_token": lock_token,
            }
            if isinstance(plan_changes_payload, dict):
                merged_snapshot.metadata["plan_changes"] = dict(plan_changes_payload)
                snapshot_metadata["plan_changes"] = dict(plan_changes_payload)
            self.snapshot_store.update_snapshot_metadata(snapshot_id, snapshot_metadata)
            self.index_run_store.mark_completed(
                run_id, status=status, warnings=warnings
            )
            result: dict[str, Any] = {
                "status": status,
                "run_id": run_id,
                "repo_name": repo_name,
                "snapshot_id": snapshot_id,
                "artifact_key": artifact_key,
                "manifest": manifest,
                "warnings": warnings,
                "scip_artifact_ref": scip_artifact_ref,
                "scip_artifact_refs": scip_artifact_refs,
                "pipeline_layers": pipeline_layers,
                "pipeline_metrics": merged_snapshot.metadata.get(
                    "pipeline_metrics", {}
                ),
                "relational_fact_persistence": relational_fact_persistence,
                "ir_graph_persistence": ir_graph_persistence,
                "unit_artifact_persistence": unit_artifact_persistence,
                "pg_retrieval_persistence": pg_retrieval_persistence,
            }
            if incremental_change_set is not None:
                result["incremental"] = {
                    "added": len(incremental_change_set.added),
                    "modified": len(incremental_change_set.modified),
                    "removed": len(incremental_change_set.removed),
                    "unchanged": len(incremental_change_set.unchanged),
                }
            if incremental_plan is not None:
                if artifact_reuse_stats:
                    incremental_plan["artifact_shard_reuse"] = dict(
                        artifact_reuse_stats
                    )
                if isinstance(plan_changes_payload, dict):
                    result["plan_changes"] = dict(plan_changes_payload)
                result["incremental_prefilter"] = dict(incremental_plan)
                preserved_sources = self._preservable_incremental_sources(
                    incremental_plan
                )
                if preserved_sources:
                    result["incremental_prefilter"][
                        "preserved_source_owned_evidence"
                    ] = sorted(preserved_sources)
            widened_repair_needed = bool(
                incremental_plan is None
                or incremental_plan.get("semantic_frontier_widened", 0) > 0
            )
            repair_task = self._build_repair_frontier_task(
                snapshot_id=snapshot_id,
                repo_name=repo_name,
                source=source,
                changed_paths=sorted(target_paths),
                modified_count=(
                    incremental_plan["modified"] if incremental_plan is not None else 0
                ),
                widened=widened_repair_needed,
                scope_kind=(
                    "package"
                    if incremental_plan is not None
                    and incremental_plan.get("api_frontier_changed", 0) > 0
                    else "path"
                ),
                scope_roots=(
                    cast(list[str], incremental_plan.get("package_scope_roots", []))
                    if incremental_plan is not None
                    else []
                ),
                change_kinds=(
                    cast(list[str], incremental_plan.get("change_kinds", []))
                    if incremental_plan is not None
                    else []
                ),
            )
            if repair_task is not None:
                task_id = self.snapshot_store.enqueue_redo_task(
                    task_type=str(repair_task["task_type"]),
                    payload=cast(dict[str, Any], repair_task["payload"]),
                )
                result["repair_queue"] = {
                    "pending": 1,
                    "task_ids": [task_id],
                    "task_type": repair_task["task_type"],
                    "scope_kind": repair_task["payload"]["scope_kind"],
                    "scope_roots": repair_task["payload"]["scope_roots"],
                }
            else:
                result["repair_queue"] = {"pending": 0}
            return result
        except Exception as e:
            self.index_run_store.mark_failed(run_id, str(e))
            self.snapshot_store.enqueue_redo_task(
                task_type="index_run_recovery",
                payload={
                    "run_id": run_id,
                    "snapshot_id": snapshot_id,
                    "source": source,
                    "is_url": resolved_is_url,
                    "ref": ref,
                    "commit": commit,
                    "publish": publish,
                    "enable_scip": enable_scip,
                    "scip_artifact_path": scip_artifact_path,
                },
                error=str(e),
            )
            raise
        finally:
            self._active_pipeline_profile = previous_profile
            reset_materialization_counters(materialization_token)
            self.snapshot_store.release_lock(lock_name, owner_id=run_id)
