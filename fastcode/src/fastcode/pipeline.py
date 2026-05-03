"""
IndexPipeline - Extracted indexing pipeline from FastCode.

Handles repository checkout, snapshot resolution, AST+SCIP dual-source
extraction, IR merge, semantic resolution, and artifact persistence.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Callable
from time import perf_counter
from typing import Any, cast
from urllib.parse import urlparse

import numpy as np
from git import GitCommandError, Repo

from .adapters.ast_to_ir import build_ir_from_ast
from .adapters.scip_to_ir import build_ir_from_scip
from .doc_ingester import KeyDocIngester
from .embedder import CodeEmbedder
from .global_index_builder import GlobalIndexBuilder
from .graph_builder import CodeGraphBuilder
from .incremental_update import apply_incremental_update, diff_changed_files
from .index_run import IndexRunStore
from .indexer import CodeElement, CodeElementMeta, CodeIndexer
from .ir_graph_builder import IRGraphBuilder
from .ir_merge import merge_ir
from .ir_validators import validate_snapshot
from .loader import RepositoryLoader
from .manifest_store import ManifestStore
from .module_resolver import ModuleResolver
from .pg_retrieval import PgRetrievalStore
from .retriever import HybridRetriever
from .scip_indexers import (
    detect_scip_languages,
    get_scip_indexer_profile,
    run_scip_for_language,
)
from .scip_loader import load_scip_artifact, run_scip_python_index
from .semantic_ir import IRSnapshot
from .semantic_resolvers import (
    apply_resolution_patch,
    build_default_semantic_resolver_registry,
)
from .snapshot_store import SnapshotStore
from .snapshot_symbol_index import SnapshotSymbolIndex
from .symbol_resolver import SymbolResolver
from .terminus_publisher import TerminusPublisher
from .utils import compute_file_hash, ensure_dir
from .vector_store import VectorStore


class IndexPipeline:
    """Encapsulates the full snapshot-oriented indexing pipeline."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        logger: Any,
        loader: RepositoryLoader,
        snapshot_store: SnapshotStore,
        manifest_store: ManifestStore,
        index_run_store: IndexRunStore,
        snapshot_symbol_index: SnapshotSymbolIndex,
        vector_store: VectorStore,
        embedder: CodeEmbedder,
        indexer: CodeIndexer,
        retriever: HybridRetriever,
        graph_builder: CodeGraphBuilder,
        ir_graph_builder: IRGraphBuilder,
        pg_retrieval_store: PgRetrievalStore | None,
        terminus_publisher: TerminusPublisher,
        doc_ingester: KeyDocIngester,
        semantic_resolver_registry: Any,
        # Callbacks for mutable FastCode state
        set_repo_indexed: Callable[[bool], None],
        set_repo_loaded: Callable[[bool], None],
        set_repo_info: Callable[[dict[str, Any]], None],
    ) -> None:
        self.config = config
        self.logger = logger
        self.loader = loader
        self.snapshot_store = snapshot_store
        self.manifest_store = manifest_store
        self.index_run_store = index_run_store
        self.snapshot_symbol_index = snapshot_symbol_index
        self.vector_store = vector_store
        self.embedder = embedder
        self.indexer = indexer
        self.retriever = retriever
        self.graph_builder = graph_builder
        self.ir_graph_builder = ir_graph_builder
        self.pg_retrieval_store = pg_retrieval_store
        self.terminus_publisher = terminus_publisher
        self.doc_ingester = doc_ingester
        self.semantic_resolver_registry = semantic_resolver_registry
        self._set_repo_indexed = set_repo_indexed
        self._set_repo_loaded = set_repo_loaded
        self._set_repo_info = set_repo_info
        self._artifact_lock = threading.RLock()

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
        try:
            repo = Repo(self.loader.repo_path)
            repo.git.checkout(target)
            self.logger.info(f"Checked out target: {target}")
        except (GitCommandError, Exception) as e:
            raise RuntimeError(f"Failed to checkout target '{target}': {e}")

    def _resolve_snapshot_ref(
        self,
        repo_name: str,
        requested_ref: str | None = None,
        requested_commit: str | None = None,
    ) -> dict[str, Any]:
        """Resolve repo snapshot identity from git metadata or file hashes."""
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
            snapshot_id = f"snap:{repo_name}:{commit_id}"
            return {
                "repo_name": repo_name,
                "branch": branch,
                "commit_id": commit_id,
                "tree_id": tree_id,
                "snapshot_id": snapshot_id,
            }
        except Exception:
            files = self.loader.scan_files()
            if not files:
                synthetic = "empty"
            else:
                digest = hashlib.sha1()
                for f in sorted(files, key=lambda x: x["relative_path"]):
                    digest.update(f["relative_path"].encode("utf-8"))
                    try:
                        digest.update(compute_file_hash(f["path"]).encode("utf-8"))
                    except Exception:
                        digest.update(str(f.get("size", 0)).encode("utf-8"))
                synthetic = digest.hexdigest()
            return {
                "repo_name": repo_name,
                "branch": requested_ref,
                "commit_id": requested_commit,
                "tree_id": synthetic,
                "snapshot_id": f"snap:{repo_name}:{synthetic}",
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
            return self._load_artifacts_by_key_locked(artifact_key)

    def _load_artifacts_by_key_locked(self, artifact_key: str) -> bool:
        if not self.vector_store.load(artifact_key):
            return False

        bm25_loaded = self.retriever.load_bm25(artifact_key)
        graph_loaded = self.graph_builder.load(artifact_key)
        ir_graphs = None
        if artifact_key.startswith("snap_"):
            record = self.snapshot_store.find_by_artifact_key(artifact_key)
            snapshot_id = record["snapshot_id"] if record else None
            if snapshot_id:
                ir_graphs = self.snapshot_store.load_ir_graphs(snapshot_id)
                self.retriever.set_ir_graphs(ir_graphs, snapshot_id=snapshot_id)
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

    def _reconstruct_elements_from_metadata(self) -> list[CodeElement]:
        """
        Reconstruct CodeElement objects from vector store metadata.
        Excludes repository_overview elements.
        """
        elements: list[CodeElement] = []
        for meta in self.vector_store.metadata:
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
            except Exception as e:
                self.logger.warning(f"Failed to reconstruct element: {e}")
                continue

        self.logger.info(
            f"Reconstructed {len(elements)} elements from metadata"
            " (excluding repository_overview)"
        )
        return elements

    def _has_active_doc_persistence(self, graph_runtime: Any) -> bool:
        """Return True when doc ingestion has at least one active sink."""
        return self.snapshot_store.db_runtime.backend == "postgres" or bool(
            getattr(graph_runtime, "enabled", False)
        )

    def _should_ingest_docs(self, graph_runtime: Any) -> bool:
        """Only ingest docs when the feature is enabled and results can be persisted."""
        return bool(
            getattr(self.doc_ingester, "enabled", False)
        ) and self._has_active_doc_persistence(graph_runtime)

    def _sync_doc_overlay(
        self,
        graph_runtime: Any,
        *,
        chunks: list[dict[str, Any]],
        mentions: list[dict[str, Any]],
        warnings: list[str],
    ) -> None:
        """Best-effort Ladybug sync with explicit failure reporting."""
        if not chunks or not getattr(graph_runtime, "enabled", False):
            return
        try:
            if graph_runtime is None:
                raise RuntimeError("Graph runtime not available")
            synced = graph_runtime.sync_docs(chunks=chunks, mentions=mentions)
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

    def _apply_semantic_resolvers(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        legacy_graph_builder: CodeGraphBuilder | None,
        target_paths: set[str],
        warnings: list[str],
        budget: str = "changed_files",
    ) -> IRSnapshot:
        if not target_paths:
            return snapshot

        upgraded = snapshot
        registry = getattr(
            self,
            "semantic_resolver_registry",
            build_default_semantic_resolver_registry(),
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

        for resolver in resolvers:
            try:
                patch = resolver.resolve(
                    snapshot=upgraded,
                    elements=elements,
                    target_paths=target_paths,
                    legacy_graph_builder=legacy_graph_builder,
                )
            except Exception as exc:
                warnings.append(f"{resolver.language}_resolver_failed: {exc}")
                continue
            warnings.extend(patch.warnings)
            upgraded = apply_resolution_patch(upgraded, patch)
        return upgraded

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
        # Access to graph_runtime for doc sync
        graph_runtime: Any = None,
    ) -> dict[str, Any]:
        """
        Run snapshot-oriented indexing pipeline with AST + optional SCIP merge.
        """
        resolved_is_url = self._infer_is_url(source) if is_url is None else is_url

        # Load repository via callback (FastCode owns the loader lifecycle)
        if load_repository_cb is not None:
            load_repository_cb(source, is_url=resolved_is_url)

        self._checkout_target_ref(ref=ref, commit=commit)
        repo_info = self.loader.get_repository_info()
        self._set_repo_info(repo_info)

        repo_name = repo_info.get("name", "default")
        repo_url = repo_info.get("url", source)
        snapshot_ref = self._resolve_snapshot_ref(
            repo_name, requested_ref=ref, requested_commit=commit
        )
        git_meta = self._build_git_meta(snapshot_ref)
        snapshot_id = snapshot_ref["snapshot_id"]
        warnings: list[str] = []
        degraded = False
        pipeline_layers = self._default_pipeline_layers(enable_scip=enable_scip)

        existing = self.snapshot_store.get_snapshot_record(snapshot_id)
        if existing and not force:
            artifact_key = existing.artifact_key
            loaded = self._load_artifacts_by_key(artifact_key)
            return self._backfill_result_layer_metadata(
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
        existing_run = self.index_run_store.get_run(run_id)
        if (
            existing_run
            and existing_run.get("status")
            in {"published", "succeeded", "degraded", "publish_pending"}
            and not force
        ):
            existing_snapshot = self.snapshot_store.get_snapshot_record(snapshot_id)
            if existing_snapshot:
                loaded = self._load_artifacts_by_key(existing_snapshot.artifact_key)
                return self._backfill_result_layer_metadata(
                    snapshot_id=snapshot_id,
                    enable_scip=enable_scip,
                    result={
                        "status": existing_run.get("status"),
                        "run_id": run_id,
                        "repo_name": repo_name,
                        "snapshot_id": snapshot_id,
                        "artifact_key": existing_snapshot.artifact_key,
                        "loaded": loaded,
                        "warnings": json.loads(
                            existing_run.get("warnings_json") or "[]"
                        ),
                    },
                )
        self.index_run_store.mark_started(run_id)
        lock_name = f"index:{snapshot_id}"
        fencing_token: int | None = self.snapshot_store.acquire_lock(
            lock_name, owner_id=run_id, ttl_seconds=600
        )
        if fencing_token is None:
            raise RuntimeError(
                f"snapshot is currently locked for indexing: {snapshot_id}"
            )
        lock_token: int = fencing_token
        stage_id: str | None = None

        try:
            self.index_run_store.mark_status(run_id, "extracting")
            elements: list[CodeElement] = self.indexer.extract_elements(
                repo_name=repo_name, repo_url=repo_url
            )

            artifact_key = self.snapshot_store.artifact_key_for_snapshot(snapshot_id)

            self.index_run_store.mark_status(run_id, "materializing")
            temp_store = VectorStore(self.config)
            temp_store.initialize(self.embedder.embedding_dim)
            vectors: list[list[float]] = []
            metadata: list[CodeElementMeta] = []
            for elem in elements:
                embedding = elem.metadata.get("embedding")
                if embedding is not None:
                    elem.metadata["snapshot_id"] = snapshot_id
                    elem.metadata["source_priority"] = 10
                    vectors.append(embedding)
                    elem_dict = elem.to_dict()
                    elem_dict["snapshot_id"] = snapshot_id
                    metadata.append(elem_dict)
            if not vectors:
                raise RuntimeError("No embeddings produced during indexing")
            temp_store.add_vectors(np.array(vectors), metadata)

            temp_graph = CodeGraphBuilder(self.config)
            module_resolver = None
            symbol_resolver = None
            try:
                gib = GlobalIndexBuilder(self.config)
                gib.build_maps(elements, self.loader.repo_path or "")
                module_resolver = ModuleResolver(gib)
                symbol_resolver = SymbolResolver(gib, module_resolver)
            except Exception as e:
                warnings.append(f"resolver_init_failed: {e}")
            temp_graph.build_graphs(elements, module_resolver, symbol_resolver)

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
                    "embedded_elements": len(vectors),
                    "bm25_indexed": len(elements),
                    "dependency_graph_nodes": temp_graph.dependency_graph.number_of_nodes(),
                    "inheritance_graph_nodes": temp_graph.inheritance_graph.number_of_nodes(),
                    "call_graph_nodes": temp_graph.call_graph.number_of_nodes(),
                    "dependency_graph_edges": temp_graph.dependency_graph.number_of_edges(),
                    "inheritance_graph_edges": temp_graph.inheritance_graph.number_of_edges(),
                    "call_graph_edges": temp_graph.call_graph.number_of_edges(),
                },
            )

            self.index_run_store.mark_status(run_id, "validating")
            ast_snapshot: IRSnapshot = build_ir_from_ast(
                repo_name=repo_name,
                snapshot_id=snapshot_id,
                elements=elements,
                repo_root=self.loader.repo_path or "",
                branch=snapshot_ref.get("branch"),
                commit_id=snapshot_ref.get("commit_id"),
                tree_id=snapshot_ref.get("tree_id"),
            )

            ast_snapshot.metadata["repo_root"] = self.loader.repo_path or ""
            ast_snapshot.metadata["pipeline_layers"] = pipeline_layers
            scip_snapshot = None
            scip_artifact_ref = None
            scip_artifact_refs: list[dict[str, Any]] = []
            layer2 = pipeline_layers[1]
            if enable_scip:
                layer2_start = perf_counter()
                try:
                    scip_artifact_paths: list[str] = []
                    experimental_scip_languages: list[str] = []
                    if scip_artifact_path:
                        scip_data = load_scip_artifact(scip_artifact_path)
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
                        out_dir = tempfile.mkdtemp(prefix="fastcode_scip_")
                        scip_indexes = []
                        detected_languages = detect_scip_languages(
                            self.loader.repo_path or ""
                        )
                        experimental_scip_languages = [
                            language
                            for language in detected_languages
                            if (profile := get_scip_indexer_profile(language))
                            is not None
                            and profile.experimental
                        ]
                        if experimental_scip_languages:
                            warning = "experimental_scip_languages: " + ", ".join(
                                sorted(experimental_scip_languages)
                            )
                            warnings.append(warning)
                            layer2["warnings"].append(warning)
                        for language in detected_languages:
                            scip_index = run_scip_for_language(
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
                            run_scip_python_index(self.loader.repo_path or "", out_path)
                            scip_indexes.append(
                                ("python", load_scip_artifact(out_path))
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
                                unit for snap in scip_snapshots for unit in snap.units
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
                            },
                        )
                        scip_data = scip_indexes[0][1]
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
                                    "indexer_name": scip_data.indexer_name
                                    or "scip-python",
                                    "indexer_version": scip_data.indexer_version,
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
            prev_manifest = self.manifest_store.get_branch_manifest(
                repo_name, ref_name_for_inc
            )
            if prev_manifest:
                prev_snap_id = prev_manifest.get("snapshot_id")
                if prev_snap_id and prev_snap_id != snapshot_id:
                    prev_snapshot = self.snapshot_store.load_snapshot(prev_snap_id)
                    if prev_snapshot:
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
                legacy_graph_builder=temp_graph,
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
            }
            errors = validate_snapshot(merged_snapshot)
            if errors:
                raise RuntimeError(f"IR validation failed: {errors[:5]}")

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
                raise RuntimeError(f"stale_lock_detected_for_snapshot:{snapshot_id}")

            # Artifact persistence — only after fencing confirmed valid.
            temp_store.save(artifact_key)
            temp_retriever.save_bm25(artifact_key)
            temp_graph.save(artifact_key)

            self.snapshot_store.save_snapshot(
                merged_snapshot,
                metadata={
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
                },
            )
            self.snapshot_store.import_git_backbone(merged_snapshot, git_meta=git_meta)
            self.snapshot_store.save_relational_facts(merged_snapshot)
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
            ir_graphs = self.ir_graph_builder.build_graphs(merged_snapshot)
            self.snapshot_store.save_ir_graphs(snapshot_id, ir_graphs)
            stage_id = self.snapshot_store.stage_snapshot(
                merged_snapshot,
                metadata={"run_id": run_id, "artifact_key": artifact_key},
            )
            all_pg_elements: list[dict[str, Any]] = [
                cast(dict[str, Any], elem.to_dict()) for elem in elements
            ]
            if doc_elements_payload:
                all_pg_elements.extend(doc_elements_payload)
            if not self.pg_retrieval_store:
                raise RuntimeError("pg_retrieval_store not initialized")
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
                manifest = self.manifest_store.publish(
                    repo_name=repo_name,
                    ref_name=ref_name,
                    snapshot_id=snapshot_id,
                    index_run_id=run_id,
                    status="published",
                )
                if self.terminus_publisher.is_configured():
                    try:
                        self.terminus_publisher.publish_snapshot_lineage(
                            snapshot=merged_snapshot.to_dict(),
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
                            manifest_id=manifest.get("manifest_id")
                            if manifest
                            else None,
                            error_message=str(e),
                        )
                        status = "publish_pending"
                else:
                    warnings.append("terminus_not_configured")
                if stage_id:
                    self.snapshot_store.promote_staged_snapshot(
                        snapshot_id=snapshot_id, stage_id=stage_id
                    )

            self.snapshot_store.update_snapshot_metadata(
                snapshot_id,
                {
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
                },
            )
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
            }
            if incremental_change_set is not None:
                result["incremental"] = {
                    "added": len(incremental_change_set.added),
                    "modified": len(incremental_change_set.modified),
                    "removed": len(incremental_change_set.removed),
                    "unchanged": len(incremental_change_set.unchanged),
                }
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
            self.snapshot_store.release_lock(lock_name, owner_id=run_id)
