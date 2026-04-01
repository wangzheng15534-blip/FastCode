"""
Main FastCode Class - Orchestrate all components
"""

import os
import pickle
import logging
import json
import hashlib
import re
import tempfile
from datetime import datetime
from urllib.parse import urlparse
from typing import Optional, Dict, Any, List, Callable
import numpy as np
import networkx as nx
from rank_bm25 import BM25Okapi
from git import Repo, GitCommandError

from .utils import load_config, resolve_config_paths, setup_logging, ensure_dir, compute_file_hash, safe_jsonable
from .loader import RepositoryLoader
from .parser import CodeParser
from .embedder import CodeEmbedder
from .indexer import CodeIndexer, CodeElement
from .vector_store import VectorStore
from .graph_builder import CodeGraphBuilder
from .symbol_resolver import SymbolResolver
from .module_resolver import ModuleResolver
from .global_index_builder import GlobalIndexBuilder
from .retriever import HybridRetriever
from .query_processor import QueryProcessor
from .answer_generator import AnswerGenerator
from .cache import CacheManager
from .adapters.ast_to_ir import build_ir_from_ast
from .adapters.scip_to_ir import build_ir_from_scip
from .ir_graph_builder import IRGraphBuilder
from .ir_merge import merge_ir
from .ir_validators import validate_snapshot
from .manifest_store import ManifestStore
from .index_run import IndexRunStore
from .scip_loader import load_scip_artifact, run_scip_python_index
from .scip_models import SCIPIndex
from .snapshot_store import SnapshotStore
from .terminus_publisher import TerminusPublisher
from .projection_models import ProjectionScope
from .projection_store import ProjectionStore
from .projection_transform import ProjectionTransformer
from .snapshot_symbol_index import SnapshotSymbolIndex
from .pg_retrieval import PgRetrievalStore
from .redo_worker import RedoWorker
from .semantic_ir import IREdge
from .doc_ingester import KeyDocIngester
from .graph_runtime import LadybugGraphRuntime


class FastCode:
    """Main FastCode system for repository-level code understanding"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize FastCode system
        
        Args:
            config_path: Path to configuration file (default: config/config.yaml)
        """
        # Resolve FastCode project root from package location.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Load configuration
        if config_path is None:
            # Try to find config in standard locations
            possible_paths = [
                "config/config.yaml",
                "../config/config.yaml",
                os.path.join(os.path.dirname(__file__), "../config/config.yaml"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        if config_path and os.path.exists(config_path):
            self.config = load_config(config_path)
        else:
            # Use default configuration
            self.config = self._get_default_config()
            self.config = resolve_config_paths(self.config, project_root)
        
        # Evaluation-specific overrides (keep core system decoupled)
        self.eval_config = self.config.get("evaluation", {})
        self.eval_mode = self.eval_config.get("enabled", False)
        self.in_memory_index = self.eval_config.get("in_memory_index", False)
        
        # Ensure in-memory mode disables disk-based caches/persistence
        if self.in_memory_index:
            self.config.setdefault("vector_store", {})["in_memory"] = True
            self.config.setdefault("cache", {})["enabled"] = False
        
        # Allow explicit cache disable via evaluation config
        if self.eval_config.get("disable_cache", False):
            self.config.setdefault("cache", {})["enabled"] = False
        
        # Setup logging
        self.logger = setup_logging(self.config)
        self.logger.info("Initializing FastCode system")

        # Initialize resolver attributes (will be set in index_repository)
        self.global_index_builder = None
        self.module_resolver = None
        self.symbol_resolver = None
        
        # Initialize components
        self.loader = RepositoryLoader(self.config)
        self.parser = CodeParser(self.config)
        self.embedder = CodeEmbedder(self.config)
        self.vector_store = VectorStore(self.config)
        self.indexer = CodeIndexer(self.config, self.loader, self.parser, self.embedder, self.vector_store)
        self.graph_builder = CodeGraphBuilder(self.config)
        self.ir_graph_builder = IRGraphBuilder()
        
        # Get repo_root from config if available
        config_repo_root = self.config.get("repo_root")
        config_repo_root = os.path.abspath(config_repo_root)
        ensure_dir(config_repo_root)
        self.logger.info(f"Configured repo_root: {config_repo_root}")
        
        self.retriever = HybridRetriever(self.config, self.vector_store, 
                                         self.embedder, self.graph_builder,
                                         repo_root=config_repo_root)
        self.query_processor = QueryProcessor(self.config)
        self.answer_generator = AnswerGenerator(self.config)
        self.cache_manager = CacheManager(self.config)

        persist_dir = self.vector_store.persist_dir
        storage_cfg = self.config.get("storage", {})
        self.snapshot_store = SnapshotStore(persist_dir, storage_cfg=storage_cfg)
        self.manifest_store = ManifestStore(self.snapshot_store.db_runtime)
        self.index_run_store = IndexRunStore(self.snapshot_store.db_runtime)
        self.terminus_publisher = TerminusPublisher(self.config)
        self.projection_transformer = ProjectionTransformer(self.config)
        self.projection_store = ProjectionStore(self.config)
        self.snapshot_symbol_index = SnapshotSymbolIndex()
        self.pg_retrieval_store = PgRetrievalStore(self.snapshot_store.db_runtime, self.config)
        self.retriever.set_pg_retrieval_store(self.pg_retrieval_store)
        self.doc_ingester = KeyDocIngester(self.config, self.embedder)
        self.graph_runtime = LadybugGraphRuntime(self.config)

        self._redo_worker: Optional[RedoWorker] = None
        if self.snapshot_store.db_runtime.backend == "postgres":
            storage_cfg = self.config.get("storage", {}) or {}
            poll_interval = int(storage_cfg.get("redo_poll_interval_seconds", 30))
            self._redo_worker = RedoWorker(self, poll_interval_seconds=poll_interval)
            self._redo_worker.start()
        
        # State
        self.repo_loaded = False
        self.repo_indexed = False
        self.repo_info = {}
        
        # Multi-repository state
        self.multi_repo_mode = False
        self.loaded_repositories = {}  # {repo_name: repo_info}

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
    
    def load_repository(self, source: str, is_url: Optional[bool] = None, is_zip: bool = False):
        """
        Load repository from URL, local path, or ZIP file
        
        Args:
            source: Repository URL, local path, or ZIP file path
            is_url: True if source is a URL, False if local path.
                    If None, FastCode auto-detects source type.
            is_zip: True if source is a ZIP file, False otherwise
        """
        self.logger.info(f"Loading repository: {source}")
        
        try:
            resolved_is_url = is_url
            if not is_zip and resolved_is_url is None:
                resolved_is_url = self._infer_is_url(source)
                source_type = "URL" if resolved_is_url else "local path"
                self.logger.info(f"Auto-detected source type as {source_type}: {source}")

            if is_zip:
                self.loader.load_from_zip(source)
            elif resolved_is_url:
                self.loader.load_from_url(source)
            else:
                self.loader.load_from_path(source)
            
            self.repo_loaded = True
            self.repo_info = self.loader.get_repository_info()
            
            # CRITICAL: Update config with the actual repo path.
            # This ensures path_utils can correctly normalize paths relative to the root.
            if self.loader.repo_path:
                self.config["repo_root"] = self.loader.repo_path
                self.logger.info(f"Set repo_root to: {self.loader.repo_path}")
                
                # Initialize retriever agents if agency mode is enabled
                self.retriever.set_repo_root(self.loader.repo_path)
            
            self.logger.info(f"Loaded repository: {self.repo_info.get('name')}")
            self.logger.info(f"Files: {self.repo_info.get('file_count')}, "
                           f"Size: {self.repo_info.get('total_size_mb', 0):.2f} MB")
            
        except Exception as e:
            self.logger.error(f"Failed to load repository: {e}")
            raise
    
    def index_repository(self, force: bool = False):
        """
        Index the loaded repository
        
        Args:
            force: Force re-indexing even if cache exists
        """
        # Evaluation can request forced re-indexing to respect commit checkouts
        force = force or self.eval_config.get("force_reindex", False)
        
        if not self.repo_loaded:
            raise RuntimeError("No repository loaded. Call load_repository() first.")
        
        self.logger.info("Indexing repository")
        
        repo_name = self.repo_info.get("name", "default")
        
        # Check cache
        if not force and self._should_use_cache():
            loaded = self._try_load_from_cache()
            if loaded:
                self.repo_indexed = True
                return
        
        try:
            # Get repository name for indexing
            repo_url = self.repo_info.get("url")
            
            # Index code elements with repository information
            elements = self.indexer.extract_elements(repo_name=repo_name, repo_url=repo_url)
            
            # Initialize vector store if not already done
            if self.vector_store.dimension is None:
                self.vector_store.initialize(self.embedder.embedding_dim)
            
            # Add embeddings to vector store
            vectors = []
            metadata = []
            
            for elem in elements:
                embedding = elem.metadata.get("embedding")
                if embedding is not None:
                    vectors.append(embedding)
                    metadata.append(elem.to_dict())
            
            if vectors:
                vectors_array = np.array(vectors)
                self.vector_store.add_vectors(vectors_array, metadata)
            
            # Initialize resolvers for complete graph building
            # This fixes the "0 edges" issue by providing the necessary context for resolution
            try:
                self.logger.info("Initializing resolvers for precise graph building...")
                
                # Ensure repo_root is set
                repo_root = self.config.get("repo_root")
                if not repo_root and self.loader.repo_path:
                    repo_root = self.loader.repo_path
                    self.config["repo_root"] = repo_root

                # 1. Create GlobalIndexBuilder
                self.global_index_builder = GlobalIndexBuilder(self.config)

                # 2. Build global maps
                self.logger.info(f"Building global index maps (Repo Root: {repo_root})...")
                self.global_index_builder.build_maps(elements, repo_root or "")
                self.logger.info(f"  - Mapped {len(self.global_index_builder.file_map)} files")
                self.logger.info(f"  - Mapped {len(self.global_index_builder.module_map)} modules")

                # 3. Create ModuleResolver
                self.module_resolver = ModuleResolver(self.global_index_builder)

                # 4. Create SymbolResolver
                self.symbol_resolver = SymbolResolver(self.global_index_builder, self.module_resolver)
                
                self.logger.info("Resolvers initialized successfully")

            except Exception as e:
                self.logger.warning(f"Resolver initialization failed: {e}")
                self.logger.warning("Using fallback graph building (less accurate)")
                import traceback
                self.logger.error(traceback.format_exc())
                self.module_resolver = None
                self.symbol_resolver = None

            # Build code graphs with resolvers
            # This will now use the initialized resolvers to build precise graphs
            self.graph_builder.build_graphs(elements, self.module_resolver, self.symbol_resolver)
            
            # Index for BM25
            self.retriever.index_for_bm25(elements)
            
            # Build separate BM25 index for repository overviews
            self.retriever.build_repo_overview_bm25()
            
            # Save artifacts only when persistence is enabled
            if self._should_persist_indexes():
                # Save to cache with repository-specific name
                self._save_to_cache(cache_name=repo_name)

                # Save BM25 and graph data
                self.retriever.save_bm25(repo_name)
                self.graph_builder.save(repo_name)
                self._save_file_manifest(repo_name, self._build_file_manifest(elements, self.loader.repo_path))
            else:
                self.logger.info("Skipping on-disk persistence (ephemeral/evaluation mode)")

            self.repo_indexed = True
            self.logger.info(f"Repository indexing complete for {repo_name}")
            
            # Log statistics
            self._log_statistics()
            
        except Exception as e:
            self.logger.error(f"Failed to index repository: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _checkout_target_ref(self, ref: Optional[str] = None, commit: Optional[str] = None) -> None:
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
        requested_ref: Optional[str] = None,
        requested_commit: Optional[str] = None,
    ) -> Dict[str, Any]:
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

    def _build_git_meta(self, snapshot_ref: Dict[str, Any]) -> Dict[str, Any]:
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
        except Exception as e:
            self.logger.warning(f"Failed to resolve commit parent metadata: {e}")
        return git_meta

    def _previous_snapshot_symbol_versions(
        self,
        repo_name: str,
        ref_name: str,
        current_snapshot_id: str,
    ) -> Optional[Dict[str, str]]:
        previous_manifest = self.manifest_store.get_branch_manifest(repo_name, ref_name)
        if not previous_manifest:
            return None
        previous_snapshot_id = previous_manifest.get("snapshot_id")
        if not previous_snapshot_id or previous_snapshot_id == current_snapshot_id:
            return None
        previous_snapshot = self.snapshot_store.load_snapshot(previous_snapshot_id)
        if not previous_snapshot:
            return None
        out: Dict[str, str] = {}
        for symbol in previous_snapshot.symbols:
            if not symbol.external_symbol_id:
                continue
            out[symbol.external_symbol_id] = f"symbol:{previous_snapshot_id}:{symbol.symbol_id}"
        return out

    def _load_artifacts_by_key(self, artifact_key: str) -> bool:
        """Load vector/BM25/graph artifacts for a snapshot artifact key."""
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

        self.repo_indexed = True
        self.repo_loaded = True
        return True

    def run_index_pipeline(
        self,
        source: str,
        is_url: Optional[bool] = None,
        ref: Optional[str] = None,
        commit: Optional[str] = None,
        force: bool = False,
        publish: bool = True,
        scip_artifact_path: Optional[str] = None,
        enable_scip: bool = True,
    ) -> Dict[str, Any]:
        """
        Run snapshot-oriented indexing pipeline with AST + optional SCIP merge.
        """
        resolved_is_url = self._infer_is_url(source) if is_url is None else is_url
        self.load_repository(source, is_url=resolved_is_url)
        self._checkout_target_ref(ref=ref, commit=commit)
        self.repo_info = self.loader.get_repository_info()

        repo_name = self.repo_info.get("name", "default")
        repo_url = self.repo_info.get("url", source)
        snapshot_ref = self._resolve_snapshot_ref(repo_name, requested_ref=ref, requested_commit=commit)
        git_meta = self._build_git_meta(snapshot_ref)
        snapshot_id = snapshot_ref["snapshot_id"]
        warnings: List[str] = []
        degraded = False

        existing = self.snapshot_store.get_snapshot_record(snapshot_id)
        if existing and not force:
            artifact_key = existing["artifact_key"]
            loaded = self._load_artifacts_by_key(artifact_key)
            return {
                "status": "reused",
                "repo_name": repo_name,
                "snapshot_id": snapshot_id,
                "artifact_key": artifact_key,
                "loaded": loaded,
            }

        idempotency_key = hashlib.sha1(
            f"{repo_name}:{snapshot_id}:{bool(publish)}:{bool(enable_scip)}".encode("utf-8")
        ).hexdigest()
        run_id = self.index_run_store.create_run(
            repo_name=repo_name,
            snapshot_id=snapshot_id,
            branch=snapshot_ref.get("branch"),
            commit_id=snapshot_ref.get("commit_id"),
            idempotency_key=idempotency_key,
        )
        existing_run = self.index_run_store.get_run(run_id)
        if existing_run and existing_run.get("status") in {"published", "succeeded", "degraded", "publish_pending"} and not force:
            existing_snapshot = self.snapshot_store.get_snapshot_record(snapshot_id)
            if existing_snapshot:
                loaded = self._load_artifacts_by_key(existing_snapshot["artifact_key"])
                return {
                    "status": existing_run.get("status"),
                    "run_id": run_id,
                    "repo_name": repo_name,
                    "snapshot_id": snapshot_id,
                    "artifact_key": existing_snapshot["artifact_key"],
                    "loaded": loaded,
                    "warnings": json.loads(existing_run.get("warnings_json") or "[]"),
                }
        self.index_run_store.mark_started(run_id)
        lock_name = f"index:{snapshot_id}"
        fencing_token = self.snapshot_store.acquire_lock(lock_name, owner_id=run_id, ttl_seconds=600)
        if fencing_token is None:
            raise RuntimeError(f"snapshot is currently locked for indexing: {snapshot_id}")
        stage_id: Optional[str] = None

        try:
            self.index_run_store.mark_status(run_id, "extracting")
            elements = self.indexer.extract_elements(repo_name=repo_name, repo_url=repo_url)

            artifact_key = self.snapshot_store.artifact_key_for_snapshot(snapshot_id)

            self.index_run_store.mark_status(run_id, "materializing")
            temp_store = VectorStore(self.config)
            temp_store.initialize(self.embedder.embedding_dim)
            vectors = []
            metadata = []
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
            call_graph_edges_raw = list(temp_graph.call_graph.edges(data=True))

            temp_retriever = HybridRetriever(
                self.config,
                temp_store,
                self.embedder,
                CodeGraphBuilder(self.config),
                repo_root=self.loader.repo_path,
            )
            temp_retriever.index_for_bm25(elements)
            temp_retriever.build_repo_overview_bm25()

            temp_store.save(artifact_key)
            temp_retriever.save_bm25(artifact_key)
            temp_graph.save(artifact_key)

            self.index_run_store.mark_status(run_id, "validating")
            ast_snapshot = build_ir_from_ast(
                repo_name=repo_name,
                snapshot_id=snapshot_id,
                elements=elements,
                repo_root=self.loader.repo_path or "",
                branch=snapshot_ref.get("branch"),
                commit_id=snapshot_ref.get("commit_id"),
                tree_id=snapshot_ref.get("tree_id"),
            )
            ast_element_to_symbol: Dict[str, str] = {}
            for symbol in ast_snapshot.symbols:
                ast_element_id = (symbol.metadata or {}).get("ast_element_id")
                if ast_element_id:
                    ast_element_to_symbol[str(ast_element_id)] = symbol.symbol_id
            for caller_id, callee_id, data in call_graph_edges_raw:
                src_id = ast_element_to_symbol.get(str(caller_id))
                dst_id = ast_element_to_symbol.get(str(callee_id))
                if not src_id or not dst_id:
                    continue
                edge_id = f"edge:call:{hashlib.md5(f'{src_id}->{dst_id}'.encode('utf-8')).hexdigest()[:20]}"
                ast_snapshot.edges.append(
                    IREdge(
                        edge_id=edge_id,
                        src_id=src_id,
                        dst_id=dst_id,
                        edge_type="call",
                        source="ast",
                        confidence="heuristic",
                        doc_id=None,
                        metadata={
                            "extractor": "fastcode.graph_builder.call_graph_bridge",
                            "call_name": (data or {}).get("call_name"),
                            "call_type": (data or {}).get("call_type"),
                            "file_path": (data or {}).get("file_path"),
                        },
                    )
                )

            scip_snapshot = None
            scip_artifact_ref = None
            if enable_scip:
                try:
                    raw_scip_path = None
                    if scip_artifact_path:
                        scip_data = load_scip_artifact(scip_artifact_path)
                        raw_scip_path = scip_artifact_path
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
                        out_path = os.path.join(out_dir, "index.scip.json")
                        run_scip_python_index(self.loader.repo_path or "", out_path)
                        scip_data = load_scip_artifact(out_path)
                        raw_scip_path = out_path
                        scip_snapshot = build_ir_from_scip(
                            repo_name=repo_name,
                            snapshot_id=snapshot_id,
                            scip_index=scip_data,
                            branch=snapshot_ref.get("branch"),
                            commit_id=snapshot_ref.get("commit_id"),
                            tree_id=snapshot_ref.get("tree_id"),
                        )
                    if raw_scip_path and os.path.exists(raw_scip_path):
                        import shutil

                        scip_dir = os.path.join(self.snapshot_store.snapshot_dir(snapshot_id), "scip")
                        ensure_dir(scip_dir)
                        ext = os.path.splitext(raw_scip_path)[1] or ".json"
                        preserved_path = os.path.join(scip_dir, f"raw{ext}")
                        shutil.copy2(raw_scip_path, preserved_path)
                        digest = hashlib.sha256()
                        with open(preserved_path, "rb") as fh:
                            for chunk in iter(lambda: fh.read(8192), b""):
                                digest.update(chunk)
                        scip_artifact_ref = self.snapshot_store.save_scip_artifact_ref(
                            snapshot_id=snapshot_id,
                            indexer_name=(scip_data.indexer_name if isinstance(scip_data, SCIPIndex) else None)
                            or "scip-python",
                            indexer_version=scip_data.indexer_version if isinstance(scip_data, SCIPIndex) else None,
                            artifact_path=preserved_path,
                            checksum=digest.hexdigest(),
                        )
                except Exception as e:
                    degraded = True
                    warnings.append(f"scip_unavailable_or_failed: {e}")

            merged_snapshot = merge_ir(ast_snapshot, scip_snapshot)
            errors = validate_snapshot(merged_snapshot)
            if errors:
                raise RuntimeError(f"IR validation failed: {errors[:5]}")
            self.snapshot_symbol_index.register_snapshot(merged_snapshot)

            doc_chunks_payload: List[Dict[str, Any]] = []
            doc_mentions_payload: List[Dict[str, Any]] = []
            doc_elements_payload: List[Dict[str, Any]] = []
            if self._should_ingest_docs():
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

            # Backfill canonical IR symbol IDs into vector metadata for IR-aware retrieval.
            ast_id_to_ir: Dict[str, str] = {}
            for sym in merged_snapshot.symbols:
                meta = sym.metadata or {}
                ast_elem_id = meta.get("ast_element_id")
                if ast_elem_id:
                    ast_id_to_ir[str(ast_elem_id)] = sym.symbol_id
                for alias in meta.get("aliases", []) if isinstance(meta.get("aliases", []), list) else []:
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
            temp_store.save(artifact_key)

            self.index_run_store.mark_status(run_id, "persisting")
            if fencing_token is not None and not self.snapshot_store.validate_fencing_token(lock_name, fencing_token):
                raise RuntimeError(f"stale_lock_detected_for_snapshot:{snapshot_id}")
            self.snapshot_store.save_snapshot(
                merged_snapshot,
                metadata={
                    "run_id": run_id,
                    "artifact_key": artifact_key,
                    "warnings": warnings,
                    "scip_artifact_ref": scip_artifact_ref,
                    "fencing_token": fencing_token,
                },
            )
            self.snapshot_store.import_git_backbone(merged_snapshot, git_meta=git_meta)
            self.snapshot_store.save_relational_facts(merged_snapshot)
            if doc_chunks_payload:
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
            all_pg_elements = [elem.to_dict() for elem in elements]
            if doc_elements_payload:
                all_pg_elements.extend(doc_elements_payload)
            self.pg_retrieval_store.upsert_elements(
                snapshot_id=snapshot_id,
                elements=all_pg_elements,
            )
            self._sync_doc_overlay(
                chunks=doc_chunks_payload,
                mentions=doc_mentions_payload,
                warnings=warnings,
            )

            self._load_artifacts_by_key(artifact_key)
            self.loaded_repositories[repo_name] = self.repo_info

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
                            manifest_id=manifest.get("manifest_id") if manifest else None,
                            error_message=str(e),
                        )
                        status = "publish_pending"
                else:
                    warnings.append("terminus_not_configured")
                if stage_id:
                    self.snapshot_store.promote_staged_snapshot(snapshot_id=snapshot_id, stage_id=stage_id)

            self.snapshot_store.update_snapshot_metadata(
                snapshot_id,
                {
                    "run_id": run_id,
                    "artifact_key": artifact_key,
                    "warnings": warnings,
                    "scip_artifact_ref": scip_artifact_ref,
                    "fencing_token": fencing_token,
                },
            )
            self.index_run_store.mark_completed(run_id, status=status, warnings=warnings)
            return {
                "status": status,
                "run_id": run_id,
                "repo_name": repo_name,
                "snapshot_id": snapshot_id,
                "artifact_key": artifact_key,
                "manifest": manifest,
                "warnings": warnings,
            }
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

    def get_index_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self.index_run_store.get_run(run_id)

    def publish_index_run(self, run_id: str, ref_name: Optional[str] = None) -> Dict[str, Any]:
        run = self.index_run_store.get_run(run_id)
        if not run:
            raise RuntimeError(f"index run not found: {run_id}")
        snapshot = self.snapshot_store.load_snapshot(run["snapshot_id"])
        if not snapshot:
            raise RuntimeError(f"snapshot not found for run: {run_id}")

        manifest = self.manifest_store.publish(
            repo_name=run["repo_name"],
            ref_name=ref_name or run.get("branch") or "HEAD",
            snapshot_id=run["snapshot_id"],
            index_run_id=run_id,
            status="published",
        )
        status = "published"
        if self.terminus_publisher.is_configured():
            try:
                git_meta = self._build_git_meta(
                    {
                        "repo_name": run["repo_name"],
                        "branch": run.get("branch"),
                        "commit_id": run.get("commit_id"),
                    }
                )
                branch_name = manifest.get("ref_name") or run.get("branch") or "HEAD"
                previous_snapshot_symbols = self._previous_snapshot_symbol_versions(
                    repo_name=run["repo_name"],
                    ref_name=branch_name,
                    current_snapshot_id=run["snapshot_id"],
                )
                self.terminus_publisher.publish_snapshot_lineage(
                    snapshot=snapshot.to_dict(),
                    manifest=manifest,
                    git_meta=git_meta,
                    previous_snapshot_symbols=previous_snapshot_symbols,
                    idempotency_key=f"lineage:{run_id}:{run['snapshot_id']}",
                )
            except Exception as e:
                self.index_run_store.enqueue_publish_retry(
                    run_id=run_id,
                    snapshot_id=run["snapshot_id"],
                    manifest_id=manifest.get("manifest_id"),
                    error_message=str(e),
                )
                status = "publish_pending"
        self.index_run_store.mark_completed(run_id, status=status)
        return {"status": status, "manifest": manifest, "run_id": run_id}

    def retry_pending_publishes(self, limit: int = 10) -> Dict[str, Any]:
        if not self.terminus_publisher.is_configured():
            return {"processed": 0, "succeeded": 0, "failed": 0, "message": "terminus_not_configured"}

        processed = 0
        succeeded = 0
        failed = 0

        while processed < limit:
            task = self.index_run_store.claim_next_publish_task()
            if not task:
                break
            processed += 1
            task_id = task["task_id"]
            run_id = task["run_id"]
            try:
                run = self.index_run_store.get_run(run_id)
                if not run:
                    raise RuntimeError(f"run not found: {run_id}")
                snapshot = self.snapshot_store.load_snapshot(run["snapshot_id"])
                if not snapshot:
                    raise RuntimeError(f"snapshot not found: {run['snapshot_id']}")

                ref_name = run.get("branch") or "HEAD"
                manifest = self.manifest_store.get_branch_manifest(run["repo_name"], ref_name)
                if not manifest:
                    manifest = self.manifest_store.publish(
                        repo_name=run["repo_name"],
                        ref_name=ref_name,
                        snapshot_id=run["snapshot_id"],
                        index_run_id=run_id,
                        status="published",
                    )

                git_meta = self._build_git_meta(
                    {
                        "repo_name": run["repo_name"],
                        "branch": run.get("branch"),
                        "commit_id": run.get("commit_id"),
                    }
                )
                previous_snapshot_symbols = self._previous_snapshot_symbol_versions(
                    repo_name=run["repo_name"],
                    ref_name=ref_name,
                    current_snapshot_id=run["snapshot_id"],
                )
                self.terminus_publisher.publish_snapshot_lineage(
                    snapshot=snapshot.to_dict(),
                    manifest=manifest,
                    git_meta=git_meta,
                    previous_snapshot_symbols=previous_snapshot_symbols,
                    idempotency_key=f"lineage:{run_id}:{run['snapshot_id']}",
                )
                self.index_run_store.mark_publish_task_done(task_id)
                self.index_run_store.mark_completed(run_id, status="published")
                succeeded += 1
            except Exception as e:
                self.index_run_store.mark_publish_task_failed(task_id, str(e))
                failed += 1

        return {
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed,
        }

    def retry_index_run_recovery(self, run_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        source = payload.get("source")
        if not source:
            raise RuntimeError(f"redo recovery payload missing source for run {run_id}")
        return self.run_index_pipeline(
            source=source,
            is_url=payload.get("is_url"),
            ref=payload.get("ref"),
            commit=payload.get("commit"),
            force=True,
            publish=bool(payload.get("publish", True)),
            scip_artifact_path=payload.get("scip_artifact_path"),
            enable_scip=bool(payload.get("enable_scip", True)),
        )

    def process_redo_tasks(self, limit: int = 10) -> Dict[str, Any]:
        if not self._redo_worker:
            return {"processed": 0, "succeeded": 0, "failed": 0, "message": "redo_worker_disabled"}
        processed = 0
        succeeded = 0
        failed = 0
        while processed < max(1, int(limit)):
            status = self._redo_worker.process_once_status()
            if status == "none":
                break
            processed += 1
            if status == "succeeded":
                succeeded += 1
            elif status == "failed":
                failed += 1
        return {"processed": processed, "succeeded": succeeded, "failed": failed}

    def list_repo_refs(self, repo_name: str) -> List[Dict[str, Any]]:
        with self.snapshot_store.db_runtime.connect() as conn:
            rows = self.snapshot_store.db_runtime.execute(
                conn,
                """
                SELECT branch, commit_id, tree_id, snapshot_id, created_at
                FROM snapshot_refs
                WHERE repo_name=?
                ORDER BY created_at DESC
                """,
                (repo_name,),
            ).fetchall()
        return [self.snapshot_store.db_runtime.row_to_dict(r) for r in rows if r]

    def find_symbol(
        self,
        snapshot_id: str,
        *,
        symbol_id: Optional[str] = None,
        name: Optional[str] = None,
        path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        resolved = self.resolve_snapshot_symbol(snapshot_id, symbol_id=symbol_id, name=name, path=path)
        if not resolved:
            return None
        snapshot = self.snapshot_store.load_snapshot(snapshot_id)
        if not snapshot:
            return None
        for symbol in snapshot.symbols:
            if symbol.symbol_id == resolved:
                return symbol.to_dict()
        return None

    def get_graph_callees(self, snapshot_id: str, symbol_id: str, max_hops: int = 1) -> List[Dict[str, Any]]:
        ir_graphs = self.snapshot_store.load_ir_graphs(snapshot_id)
        if not ir_graphs:
            return []
        g = ir_graphs.call_graph
        if symbol_id not in g:
            return []
        dist = nx.single_source_shortest_path_length(g, symbol_id, cutoff=max_hops)
        return [{"symbol_id": node, "distance": d} for node, d in dist.items() if node != symbol_id]

    def get_graph_callers(self, snapshot_id: str, symbol_id: str, max_hops: int = 1) -> List[Dict[str, Any]]:
        ir_graphs = self.snapshot_store.load_ir_graphs(snapshot_id)
        if not ir_graphs:
            return []
        g = ir_graphs.call_graph.reverse(copy=False)
        if symbol_id not in g:
            return []
        dist = nx.single_source_shortest_path_length(g, symbol_id, cutoff=max_hops)
        return [{"symbol_id": node, "distance": d} for node, d in dist.items() if node != symbol_id]

    def get_graph_dependencies(self, snapshot_id: str, doc_id: str, max_hops: int = 1) -> List[Dict[str, Any]]:
        ir_graphs = self.snapshot_store.load_ir_graphs(snapshot_id)
        if not ir_graphs:
            return []
        g = ir_graphs.dependency_graph
        if doc_id not in g:
            return []
        dist = nx.single_source_shortest_path_length(g, doc_id, cutoff=max_hops)
        return [{"doc_id": node, "distance": d} for node, d in dist.items() if node != doc_id]

    def get_branch_manifest(self, repo_name: str, ref_name: str) -> Optional[Dict[str, Any]]:
        return self.manifest_store.get_branch_manifest(repo_name, ref_name)

    def get_snapshot_manifest(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        return self.manifest_store.get_snapshot_manifest(snapshot_id)

    def get_scip_artifact_ref(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        return self.snapshot_store.get_scip_artifact_ref(snapshot_id)

    def resolve_snapshot_symbol(
        self,
        snapshot_id: str,
        *,
        symbol_id: Optional[str] = None,
        name: Optional[str] = None,
        path: Optional[str] = None,
    ) -> Optional[str]:
        if not self.snapshot_symbol_index.has_snapshot(snapshot_id):
            snap = self.snapshot_store.load_snapshot(snapshot_id)
            if snap:
                self.snapshot_symbol_index.register_snapshot(snap)
        return self.snapshot_symbol_index.resolve_symbol(
            snapshot_id,
            symbol_id=symbol_id,
            name=name,
            path=path,
        )

    @staticmethod
    def _projection_scope_key(
        scope_kind: str,
        snapshot_id: str,
        query: Optional[str],
        target_id: Optional[str],
        filters: Optional[Dict[str, Any]],
    ) -> str:
        base = {
            "scope_kind": scope_kind,
            "snapshot_id": snapshot_id,
            "query": query or "",
            "target_id": target_id or "",
            "filters": filters or {},
        }
        payload = json.dumps(base, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _projection_params_hash(scope: ProjectionScope, projection_algo_version: str = "v1") -> str:
        payload = json.dumps(
            {
                "scope": scope.to_dict(),
                "projection_algo_version": projection_algo_version,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _resolve_snapshot_id(
        self,
        snapshot_id: Optional[str],
        repo_name: Optional[str],
        ref_name: Optional[str],
    ) -> str:
        if snapshot_id:
            return snapshot_id
        if not repo_name or not ref_name:
            raise RuntimeError("projection requires snapshot_id or repo_name+ref_name")
        manifest = self.manifest_store.get_branch_manifest(repo_name, ref_name)
        if not manifest:
            raise RuntimeError(f"manifest not found for {repo_name}:{ref_name}")
        return manifest["snapshot_id"]

    def _mirror_projection_artifacts(self, snapshot_id: str, result: Dict[str, Any]) -> str:
        import os

        root = os.path.join(self.snapshot_store.snapshot_dir(snapshot_id), "projection", result["projection_id"])
        ensure_dir(root)
        chunk_dir = os.path.join(root, "chunks")
        ensure_dir(chunk_dir)
        with open(os.path.join(root, "node.l0.json"), "w", encoding="utf-8") as f:
            json.dump(result["l0"], f, ensure_ascii=False, indent=2)
        with open(os.path.join(root, "node.l1.json"), "w", encoding="utf-8") as f:
            json.dump(result["l1"], f, ensure_ascii=False, indent=2)
        with open(os.path.join(root, "node.l2.index.json"), "w", encoding="utf-8") as f:
            json.dump(result["l2_index"], f, ensure_ascii=False, indent=2)
        for chunk in result.get("chunks", []):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            with open(os.path.join(chunk_dir, f"{chunk_id}.json"), "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
        return root

    def build_projection(
        self,
        scope_kind: str,
        snapshot_id: Optional[str] = None,
        repo_name: Optional[str] = None,
        ref_name: Optional[str] = None,
        query: Optional[str] = None,
        target_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        if scope_kind not in {"snapshot", "query", "entity"}:
            raise RuntimeError("scope_kind must be one of: snapshot, query, entity")
        if not self.projection_store.enabled:
            raise RuntimeError("projection store is not configured (set projection.postgres_dsn)")

        resolved_snapshot_id = self._resolve_snapshot_id(snapshot_id, repo_name, ref_name)
        snapshot_record = self.snapshot_store.get_snapshot_record(resolved_snapshot_id)
        if not snapshot_record:
            raise RuntimeError(f"snapshot not found: {resolved_snapshot_id}")

        if not self._load_artifacts_by_key(snapshot_record["artifact_key"]):
            raise RuntimeError(f"failed to load artifacts for snapshot: {resolved_snapshot_id}")

        snapshot = self.snapshot_store.load_snapshot(resolved_snapshot_id)
        if not snapshot:
            raise RuntimeError(f"IR snapshot not found: {resolved_snapshot_id}")
        ir_graphs = self.snapshot_store.load_ir_graphs(resolved_snapshot_id)

        scope_key = self._projection_scope_key(
            scope_kind=scope_kind,
            snapshot_id=resolved_snapshot_id,
            query=query,
            target_id=target_id,
            filters=filters,
        )
        scope = ProjectionScope(
            scope_kind=scope_kind,
            snapshot_id=resolved_snapshot_id,
            scope_key=scope_key,
            query=query,
            target_id=target_id,
            filters=filters or {},
        )
        params_hash = self._projection_params_hash(
            scope,
            projection_algo_version=getattr(self.projection_transformer, "ALGO_VERSION", "v1"),
        )

        if not force:
            cached_id = self.projection_store.find_cached_projection_id(scope, params_hash)
            if cached_id:
                l0 = self.projection_store.get_layer(cached_id, "L0")
                l1 = self.projection_store.get_layer(cached_id, "L1")
                l2 = self.projection_store.get_layer(cached_id, "L2")
                if l0 and l1 and l2:
                    return {
                        "status": "reused",
                        "projection_id": cached_id,
                        "snapshot_id": resolved_snapshot_id,
                        "scope_kind": scope_kind,
                        "scope_key": scope_key,
                        "l0": l0,
                        "l1": l1,
                        "l2_index": l2,
                        "warnings": [],
                    }

        build = self.projection_transformer.build(scope=scope, snapshot=snapshot, ir_graphs=ir_graphs)
        self.projection_store.save(build, params_hash=params_hash)
        payload = build.to_dict()
        mirror_root = self._mirror_projection_artifacts(resolved_snapshot_id, payload)
        payload["status"] = "built"
        payload["mirror_path"] = mirror_root
        return payload

    def get_projection_layer(self, projection_id: str, layer: str) -> Dict[str, Any]:
        if not self.projection_store.enabled:
            raise RuntimeError("projection store is not configured (set projection.postgres_dsn)")
        layer_payload = self.projection_store.get_layer(projection_id, layer)
        if not layer_payload:
            raise RuntimeError(f"projection layer not found: {projection_id}:{layer}")
        build = self.projection_store.get_build(projection_id)
        return {
            "projection_id": projection_id,
            "layer": layer.upper(),
            "node": layer_payload,
            "build": build,
        }

    def get_projection_chunk(self, projection_id: str, chunk_id: str) -> Dict[str, Any]:
        if not self.projection_store.enabled:
            raise RuntimeError("projection store is not configured (set projection.postgres_dsn)")
        chunk_payload = self.projection_store.get_chunk(projection_id, chunk_id)
        if not chunk_payload:
            raise RuntimeError(f"projection chunk not found: {projection_id}:{chunk_id}")
        build = self.projection_store.get_build(projection_id)
        return {
            "projection_id": projection_id,
            "chunk_id": chunk_id,
            "chunk": chunk_payload,
            "build": build,
        }

    def query_snapshot(
        self,
        question: str,
        repo_name: Optional[str] = None,
        ref_name: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        enable_multi_turn: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if not snapshot_id:
            if not repo_name or not ref_name:
                raise RuntimeError("query_snapshot requires snapshot_id or repo_name+ref_name")
            manifest = self.manifest_store.get_branch_manifest(repo_name, ref_name)
            if not manifest:
                raise RuntimeError(f"manifest not found for {repo_name}:{ref_name}")
            snapshot_id = manifest["snapshot_id"]

        snapshot_record = self.snapshot_store.get_snapshot_record(snapshot_id)
        if not snapshot_record:
            raise RuntimeError(f"snapshot not found: {snapshot_id}")

        if not self._load_artifacts_by_key(snapshot_record["artifact_key"]):
            raise RuntimeError(f"failed to load artifacts for snapshot: {snapshot_id}")
        if not self.snapshot_symbol_index.has_snapshot(snapshot_id):
            loaded_snapshot = self.snapshot_store.load_snapshot(snapshot_id)
            if loaded_snapshot:
                self.snapshot_symbol_index.register_snapshot(loaded_snapshot)

        merged_filters = dict(filters or {})
        merged_filters["snapshot_id"] = snapshot_id

        result = self.query(
            question=question,
            filters=merged_filters,
            repo_filter=None,
            session_id=session_id,
            enable_multi_turn=enable_multi_turn,
        )
        result["snapshot_id"] = snapshot_id
        result["artifact_key"] = snapshot_record["artifact_key"]
        return result
    
    def query(self, question: str, filters: Optional[Dict[str, Any]] = None, 
              repo_filter: Optional[List[str]] = None,
              session_id: Optional[str] = None,
              enable_multi_turn: Optional[bool] = None,
              use_agency_mode: Optional[bool] = None,
              prompt_builder: Optional[Callable[[str, str, Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]], str]] = None) -> Dict[str, Any]:
        """
        Query the repository (or multiple repositories)
        
        Args:
            question: User question
            filters: Optional filters for retrieval
            repo_filter: Optional list of repository names to search in
            session_id: Optional session ID for multi-turn dialogue
            enable_multi_turn: Override config setting for multi-turn mode
            prompt_builder: Optional callable to build a custom LLM prompt using
                (question, prepared_context, query_info, dialogue_history)
        
        Returns:
            Dictionary with answer and metadata (including summary if multi-turn)
        """
        if not self.repo_indexed:
            raise RuntimeError("Repository not indexed. Call index_repository() first.")
        
        # Determine if multi-turn mode is enabled
        if enable_multi_turn is None:
            enable_multi_turn = self.config.get("generation", {}).get("enable_multi_turn", False)
        
        if repo_filter:
            self.logger.info(f"Processing query: {question} in repositories: {repo_filter}")
        else:
            self.logger.info(f"Processing query: {question}")
        
        # Get dialogue history if in multi-turn mode
        dialogue_history = []
        if enable_multi_turn and session_id:
            # Get recent summaries from cache (last 10 turns for iterative agent)
            history_summary_rounds = self.config.get("query", {}).get("history_summary_rounds", 10)
            dialogue_history = self.cache_manager.get_recent_summaries(session_id, history_summary_rounds)

            if dialogue_history:
                self.logger.info(f"Retrieved {len(dialogue_history)} previous dialogue summaries")
        
        # NOTE: Query result caching is disabled to ensure full iterative_agent flow
        # Original cache logic (disabled):
        # use_cache = (not enable_multi_turn or not session_id) and self._should_use_cache()
        # cached_result = None
        # cache_key = None
        # repo_hash = None
        # if use_cache:
        #     repo_hash = self._get_repo_hash()
        #     cache_key = f"{question}_{','.join(sorted(repo_filter)) if repo_filter else 'all'}"
        #     cached_result = self.cache_manager.get_query_result(cache_key, repo_hash)
        #     if cached_result:
        #         self.logger.info("Returning cached result")
        # result = cached_result

        result = None  # Always process through full flow
        processed_query = None
        retrieved: List[CodeElement] = []

        try:
            if result is None:
                # Determine if iterative enhancement should be used
                use_iterative_enhancement = (
                    self.retriever.enable_agency_mode
                    and self.retriever.iterative_agent is not None
                )

                # Process query: skip query_processor entirely in iterative mode
                if use_iterative_enhancement:
                    # Iterative agent will handle all query enhancement
                    # Create minimal ProcessedQuery object
                    from .query_processor import ProcessedQuery
                    processed_query = ProcessedQuery(
                        original=question,
                        expanded=question,
                        keywords=[],
                        intent="unknown",
                        subqueries=[],
                        filters=filters or {},
                        rewritten_query=None,
                        pseudocode_hints=None,
                        search_strategy=None
                    )
                    self.logger.info("Iterative mode: skipping query_processor, all enhancements handled by iterative_agent")
                else:
                    # Standard mode: use full query processing
                    processed_query = self.query_processor.process(
                        question,
                        dialogue_history,
                        use_llm_enhancement=True
                    )
                    self.logger.info(f"Query intent: {processed_query.intent}")
                    self.logger.info(f"Keywords: {processed_query.keywords}")

                # Retrieve relevant code (with repository filter and agency mode)
                # Pass ProcessedQuery object for enhanced retrieval
                # Pass dialogue_history for multi-turn context in iterative mode
                retrieved = self.retriever.retrieve(
                    processed_query,  # Pass full ProcessedQuery object for multi-repo support
                    filters=filters,
                    repo_filter=repo_filter,
                    use_agency_mode=use_agency_mode,
                    dialogue_history=dialogue_history if enable_multi_turn else None
                )
                
                # Generate answer (with dialogue history for multi-turn)
                result = self.answer_generator.generate(
                    question,
                    retrieved,
                    query_info=processed_query.to_dict(),
                    dialogue_history=self._get_full_dialogue_history(session_id, enable_multi_turn),
                    prompt_builder=prompt_builder
                )
            
            # Add repository information to result
            if repo_filter:
                result["searched_repositories"] = repo_filter
            
            # Persist dialogue for any session (even single-turn) so users keep history
            if session_id:
                turn_number = self._get_next_turn_number(session_id)
                summary = result.get("summary", "")
                # Use formatted sources from result instead of raw retrieved elements
                # This ensures proper display format when loading history
                sources = result.get("sources", [])
                # Ensure sources are fully JSON-serializable
                serializable_sources = safe_jsonable(sources)

                # Ensure metadata is JSON-serializable
                metadata = {
                    "intent": getattr(processed_query, "intent", None),
                    "keywords": getattr(processed_query, "keywords", None),
                    "repo_filter": repo_filter,
                    "multi_turn": enable_multi_turn,
                }
                serializable_metadata = safe_jsonable(metadata)

                self.cache_manager.save_dialogue_turn(
                    session_id=session_id,
                    turn_number=turn_number,
                    query=question,
                    answer=result.get("answer", ""),
                    summary=summary,
                    retrieved_elements=serializable_sources,
                    metadata=serializable_metadata
                )

                self.logger.info(f"Saved dialogue turn {turn_number} for session {session_id}")
            
            # Cache result for stateless flows (including single-turn sessions)
            # NOTE: Query result caching is disabled to ensure full iterative_agent flow
            # if use_cache and result is not None and cache_key and repo_hash:
            #     self.cache_manager.set_query_result(cache_key, repo_hash, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "query": question,
                "error": str(e),
            }

    def query_stream(self, question: str, filters: Optional[Dict[str, Any]] = None,
                    repo_filter: Optional[List[str]] = None,
                    session_id: Optional[str] = None,
                    enable_multi_turn: Optional[bool] = None,
                    use_agency_mode: Optional[bool] = None,
                    prompt_builder: Optional[Callable[[str, str, Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]], str]] = None):
        """
        Query the repository with streaming response (yields answer chunks)

        Args:
            question: User question
            filters: Optional filters for retrieval
            repo_filter: Optional list of repository names to search in
            session_id: Optional session ID for multi-turn dialogue
            enable_multi_turn: Override config setting for multi-turn mode
            use_agency_mode: Override config setting for agency mode
            prompt_builder: Optional callable to build custom LLM prompt

        Yields:
            Tuples of (chunk_text or None, metadata_dict or None)
            - First yield: (None, {"status": "retrieving"})
            - After retrieval: (None, {"status": "generating", "sources": [...], ...})
            - During generation: (text_chunk, None)
            - Final yield: (None, {"status": "complete", "summary": ..., ...})
        """
        if not self.repo_indexed:
            yield None, {"error": "Repository not indexed. Call index_repository() first."}
            return

        # Determine if multi-turn mode is enabled
        if enable_multi_turn is None:
            enable_multi_turn = self.config.get("generation", {}).get("enable_multi_turn", False)

        if repo_filter:
            self.logger.info(f"Processing streaming query: {question} in repositories: {repo_filter}")
        else:
            self.logger.info(f"Processing streaming query: {question}")

        # Get dialogue history if in multi-turn mode
        dialogue_history = []
        if enable_multi_turn and session_id:
            history_summary_rounds = self.config.get("query", {}).get("history_summary_rounds", 10)
            dialogue_history = self.cache_manager.get_recent_summaries(session_id, history_summary_rounds)
            if dialogue_history:
                self.logger.info(f"Retrieved {len(dialogue_history)} previous dialogue summaries")

        try:
            # Notify start of retrieval
            yield None, {"status": "retrieving", "query": question}

            # Retrieval phase (same as query method)
            use_iterative_enhancement = (
                self.retriever.enable_agency_mode
                and self.retriever.iterative_agent is not None
            )

            if use_iterative_enhancement:
                from .query_processor import ProcessedQuery
                processed_query = ProcessedQuery(
                    original=question,
                    expanded=question,
                    keywords=[],
                    intent="unknown",
                    subqueries=[],
                    filters=filters or {},
                    rewritten_query=None,
                    pseudocode_hints=None,
                    search_strategy=None
                )
                self.logger.info("Iterative mode: skipping query_processor, all enhancements handled by iterative_agent")
            else:
                processed_query = self.query_processor.process(
                    question,
                    dialogue_history,
                    use_llm_enhancement=True
                )
                self.logger.info(f"Query intent: {processed_query.intent}")
                self.logger.info(f"Keywords: {processed_query.keywords}")

            # Retrieve relevant code
            retrieved = self.retriever.retrieve(
                processed_query,
                filters=filters,
                repo_filter=repo_filter,
                use_agency_mode=use_agency_mode,
                dialogue_history=dialogue_history if enable_multi_turn else None
            )

            # Notify start of generation
            yield None, {"status": "generating", "retrieved_count": len(retrieved)}

            # Stream answer generation
            full_answer_parts = []
            answer_metadata = {}

            for chunk, metadata in self.answer_generator.generate_stream(
                question,
                retrieved,
                query_info=processed_query.to_dict(),
                dialogue_history=self._get_full_dialogue_history(session_id, enable_multi_turn),
                prompt_builder=prompt_builder
            ):
                if chunk:
                    full_answer_parts.append(chunk)
                    yield chunk, None
                if metadata:
                    answer_metadata.update(metadata)

            # Build complete result
            full_answer = "".join(full_answer_parts)
            summary = answer_metadata.get("summary")

            result = {
                "status": "complete",
                "answer": full_answer,
                "query": question,
                "context_elements": len(retrieved),
                "sources": answer_metadata.get("sources", self._extract_sources_from_elements(retrieved)),
            }

            if summary:
                result["summary"] = summary

            if repo_filter:
                result["searched_repositories"] = repo_filter

            # Save dialogue turn if session_id provided
            if session_id:
                turn_number = self._get_next_turn_number(session_id)
                serializable_sources = safe_jsonable(result.get("sources", []))
                serializable_metadata = safe_jsonable({
                    "intent": getattr(processed_query, "intent", None),
                    "keywords": getattr(processed_query, "keywords", None),
                    "repo_filter": repo_filter,
                    "multi_turn": enable_multi_turn,
                })

                self.cache_manager.save_dialogue_turn(
                    session_id=session_id,
                    turn_number=turn_number,
                    query=question,
                    answer=full_answer,
                    summary=summary or "",
                    retrieved_elements=serializable_sources,
                    metadata=serializable_metadata
                )

                self.logger.info(f"Saved dialogue turn {turn_number} for session {session_id}")

            # Final yield with complete result
            yield None, result

        except Exception as e:
            self.logger.error(f"Streaming query failed: {e}")
            import traceback
            error_trace = traceback.format_exc()
            self.logger.error(f"Full error traceback:\n{error_trace}")
            yield None, {
                "status": "error",
                "error": str(e),
                "query": question,
            }

    def _extract_sources_from_elements(self, elements: List) -> List[Dict[str, Any]]:
        """Extract source information from retrieved elements"""
        sources = []
        for elem_data in elements:
            elem = elem_data.get("element", {})
            source = {
                "file": elem.get("relative_path", ""),
                "repo": elem.get("repo_name", ""),
                "type": elem.get("type", ""),
                "name": elem.get("name", ""),
                "start_line": elem.get("start_line", 0),
                "end_line": elem.get("end_line", 0),
            }
            sources.append(source)
        return sources

    def get_repository_summary(self) -> str:
        """Get summary of the loaded repository"""
        if not self.repo_info:
            return "No repository loaded"
        
        summary_parts = [
            f"Repository: {self.repo_info.get('name', 'Unknown')}",
            f"Files: {self.repo_info.get('file_count', 0)}",
            f"Size: {self.repo_info.get('total_size_mb', 0):.2f} MB",
        ]
        
        if self.repo_indexed:
            summary_parts.append(f"Indexed elements: {self.vector_store.get_count()}")
        
        return "\n".join(summary_parts)
    
    def _try_load_from_cache(self) -> bool:
        """Try to load indexed data from cache for single repository"""
        if not self._should_use_cache():
            self.logger.info("Cache loading disabled (ephemeral/evaluation mode)")
            return False
        
        try:
            cache_name = self._get_cache_name()
            
            # Try to load vector store
            if self.vector_store.load(cache_name):
                self.logger.info(f"Loaded vector store from cache for {cache_name}")
                
                # Load BM25 index
                bm25_loaded = self.retriever.load_bm25(cache_name)
                if not bm25_loaded:
                    self.logger.warning("Failed to load BM25 index, will need to rebuild")
                
                # Build separate repo overview BM25 index
                self.retriever.build_repo_overview_bm25()
                
                # Load graph data
                graph_loaded = self.graph_builder.load(cache_name)
                if not graph_loaded:
                    self.logger.warning("Failed to load graph data, will need to rebuild")
                
                # If BM25 or graph failed to load, reconstruct from metadata
                if not bm25_loaded or not graph_loaded:
                    self.logger.info("Reconstructing missing components from metadata...")
                    elements = self._reconstruct_elements_from_metadata()
                    
                    if elements:
                        if not bm25_loaded:
                            self.retriever.index_for_bm25(elements)
                            self.logger.info(f"Rebuilt BM25 index with {len(elements)} elements")
                        
                        if not graph_loaded:
                            # Note: Rebuilding graph from metadata is a fallback.
                            # Precise linking might be limited if repo_root context is lost.
                            self.graph_builder.build_graphs(elements)
                            self.logger.info("Rebuilt code graph (fallback mode)")
                    else:
                        self.logger.warning("No elements reconstructed from metadata")
                
                self.logger.info("Cache loaded successfully")
                self._log_statistics()
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {e}")
            return False
    
    def _save_to_cache(self, cache_name: Optional[str] = None):
        """Save indexed data to cache"""
        if not self._should_persist_indexes():
            self.logger.info("Cache save disabled (ephemeral/evaluation mode)")
            return
        
        try:
            if cache_name is None:
                cache_name = self._get_cache_name()
            self.vector_store.save(cache_name)
            self.logger.info(f"Saved index to cache: {cache_name}")
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {e}")
    
    def _get_cache_name(self) -> str:
        """Get cache name for current repository"""
        return self.repo_info.get("name", "default")
    
    def _get_repo_hash(self) -> str:
        """Get hash of repository for cache key"""
        return self.repo_info.get("commit", self.repo_info.get("name", "default"))
    
    def _reconstruct_elements_from_metadata(self) -> List[CodeElement]:
        """
        Reconstruct CodeElement objects from vector store metadata
        Excludes repository_overview elements (they're in separate storage)
        
        Returns:
            List of CodeElement objects
        """
        elements = []
        for meta in self.vector_store.metadata:
            try:
                # Skip repository_overview elements
                if meta.get("type") == "repository_overview":
                    continue
                
                # Reconstruct CodeElement from metadata dictionary
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
        
        self.logger.info(f"Reconstructed {len(elements)} elements from metadata (excluding repository_overview)")
        return elements
    
    def _log_statistics(self):
        """Log indexing statistics"""
        stats = {
            "vector_count": self.vector_store.get_count(),
            "graph_stats": self.graph_builder.get_graph_stats(),
        }
        
        self.logger.info(f"Statistics: {stats}")
    
    def _is_ephemeral_mode(self) -> bool:
        """Return True when running in evaluation/in-memory mode."""
        return self.in_memory_index or getattr(self.vector_store, "in_memory", False)

    def _should_use_cache(self) -> bool:
        """Determine whether cache/index reuse is allowed."""
        if self.eval_config.get("disable_cache", False):
            return False
        return not self._is_ephemeral_mode()

    def _should_persist_indexes(self) -> bool:
        """Determine whether indexes should be persisted to disk."""
        if self.eval_config.get("disable_persistence", False):
            return False
        return not self._is_ephemeral_mode()

    def _has_active_doc_persistence(self) -> bool:
        """Return True when doc ingestion has at least one active sink."""
        return (
            self.snapshot_store.db_runtime.backend == "postgres"
            or bool(getattr(self.graph_runtime, "enabled", False))
        )

    def _should_ingest_docs(self) -> bool:
        """Only ingest docs when the feature is enabled and results can be persisted."""
        return bool(getattr(self.doc_ingester, "enabled", False)) and self._has_active_doc_persistence()

    def _sync_doc_overlay(
        self,
        *,
        chunks: List[Dict[str, Any]],
        mentions: List[Dict[str, Any]],
        warnings: List[str],
    ) -> None:
        """Best-effort Ladybug sync with explicit failure reporting."""
        if not chunks or not getattr(self.graph_runtime, "enabled", False):
            return
        try:
            synced = self.graph_runtime.sync_docs(chunks=chunks, mentions=mentions)
        except Exception as e:
            warnings.append(f"ladybug_doc_sync_failed: {e}")
            return
        if not synced:
            warnings.append("ladybug_doc_sync_failed")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "storage": {
                "backend": "sqlite",
                "postgres_dsn": "",
                "pool_min": 1,
                "pool_max": 8,
            },
            "repository": {
                "clone_depth": 1,
                "max_file_size_mb": 5,
                "backup_directory": "./repo_backup",
                "ignore_patterns": ["*.pyc", "__pycache__", "node_modules", ".git"],
                "supported_extensions": [".py", ".js", ".ts", ".java", ".go"],
            },
            "parser": {
                "extract_docstrings": True,
                "extract_comments": True,
                "extract_imports": True,
            },
            "embedding": {
                "provider": "ollama",
                "model": "bge-large-en-v1.5",
                "ollama_url": "http://127.0.0.1:11434/api/embeddings",
                "device": "cpu",
                "batch_size": 32,
            },
            "indexing": {
                "levels": ["file", "class", "function", "documentation"],
            },
            "vector_store": {
                "persist_directory": "./data/vector_store",
                "distance_metric": "cosine",
            },
            "retrieval": {
                "semantic_weight": 0.6,
                "keyword_weight": 0.3,
                "graph_weight": 0.1,
                "max_results": 5,
                "backend": "pg_hybrid",
                "graph_backend": "ir",
                "allow_legacy_graph_fallback": True,
            },
            "generation": {
                "provider": "openai",
                "model": "gpt-4-turbo-preview",
                "temperature": 0.1,
                "max_tokens": 2000,
            },
            "evaluation": {
                "enabled": False,
                "in_memory_index": False,
                "disable_cache": False,
                "disable_persistence": False,
                "force_reindex": False,
            },
            "cache": {
                "enabled": True,
                "backend": "disk",
                "cache_directory": "./data/cache",
                "cache_queries": False,
            },
            "logging": {
                "level": "INFO",
                "console": True,
            },
            "terminus": {
                "endpoint": "",
                "api_key": "",
                "timeout_seconds": 15,
            },
            "projection": {
                "postgres_dsn": "",
                "enable_leiden": True,
                "llm_enabled": True,
                "llm_timeout_seconds": 8,
                "llm_max_tokens": 180,
                "llm_temperature": 0.2,
                "max_entity_hops": 2,
                "max_query_hops": 2,
                "max_chunk_count": 64,
            },
        }
    
    def load_multiple_repositories(self, sources: List[Dict[str, Any]]):
        """
        Load and index multiple repositories (saves each repository separately)
        
        Args:
            sources: List of dictionaries with 'source', 'is_url', and optionally 'is_zip' keys
                    Example: [{'source': 'https://github.com/user/repo1', 'is_url': True},
                             {'source': '/path/to/repo2', 'is_url': False},
                             {'source': '/path/to/repo3.zip', 'is_url': False, 'is_zip': True}]
        """
        self.logger.info(f"Loading {len(sources)} repositories")
        self.multi_repo_mode = True
        
        successfully_indexed = []
        
        for i, source_info in enumerate(sources):
            source = source_info.get('source')
            is_url = source_info.get('is_url')
            is_zip = source_info.get('is_zip', False)
            
            try:
                self.logger.info(f"[{i+1}/{len(sources)}] Loading repository: {source}")

                resolved_is_url = is_url
                if not is_zip and resolved_is_url is None:
                    resolved_is_url = self._infer_is_url(source)
                    source_type = "URL" if resolved_is_url else "local path"
                    self.logger.info(f"[{i+1}/{len(sources)}] Auto-detected source type as {source_type}")
                
                # Load repository
                if is_zip:
                    self.loader.load_from_zip(source)
                elif resolved_is_url:
                    self.loader.load_from_url(source)
                else:
                    self.loader.load_from_path(source)
                
                repo_info = self.loader.get_repository_info()
                repo_name = repo_info.get('name')
                repo_url = repo_info.get('url', source)
                
                # Update config with repo_root for each repo (Critical for graph building)
                if self.loader.repo_path:
                    self.config["repo_root"] = self.loader.repo_path
                
                # Store repository info
                self.loaded_repositories[repo_name] = repo_info
                
                self.logger.info(f"Indexing repository: {repo_name}")
                
                # Create a fresh vector store for this repository
                temp_vector_store = VectorStore(self.config)
                temp_vector_store.initialize(self.embedder.embedding_dim)
                
                # Create a temporary indexer with the temp vector store for this repo
                temp_indexer = CodeIndexer(self.config, self.loader, self.parser, 
                                          self.embedder, temp_vector_store)
                
                # Index with repository information
                elements = temp_indexer.extract_elements(repo_name=repo_name, repo_url=repo_url)
                
                # Add to temporary vector store
                vectors = []
                metadata = []
                
                for elem in elements:
                    embedding = elem.metadata.get("embedding")
                    if embedding is not None:
                        vectors.append(embedding)
                        metadata.append(elem.to_dict())
                
                if vectors:
                    vectors_array = np.array(vectors)
                    temp_vector_store.add_vectors(vectors_array, metadata)
                    
                    # Save this repository's vector index separately
                    temp_vector_store.save(repo_name)
                    
                    # Build and save BM25 index for this repository
                    temp_retriever = HybridRetriever(self.config, temp_vector_store, 
                                                     self.embedder, self.graph_builder,
                                                     repo_root=self.loader.repo_path)
                    temp_retriever.index_for_bm25(elements)
                    temp_retriever.save_bm25(repo_name)
                    self.logger.info(f"Saved BM25 index for {repo_name}")
                    
                    # Build separate BM25 index for repository overviews
                    temp_retriever.build_repo_overview_bm25()
                    self.logger.info(f"Built repo overview BM25 index")
                    
                    # Build and save graph for this repository (Using temporary graph builder)
                    # We need a fresh graph builder to avoid mixing graphs between repos during this loop
                    # unless we want to support cross-repo graphs immediately
                    temp_graph_builder = CodeGraphBuilder(self.config)
                    
                    # Initialize resolvers for precise graph building
                    repo_root = self.loader.repo_path
                    temp_module_resolver = None
                    temp_symbol_resolver = None
                    
                    try:
                        self.logger.info(f"Initializing resolvers for {repo_name}...")
                        temp_global_index = GlobalIndexBuilder(self.config)
                        temp_global_index.build_maps(elements, repo_root)
                        temp_module_resolver = ModuleResolver(temp_global_index)
                        temp_symbol_resolver = SymbolResolver(temp_global_index, temp_module_resolver)
                        self.logger.info(f"Resolvers initialized for {repo_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize resolvers for {repo_name}: {e}")
                        temp_module_resolver = None
                        temp_symbol_resolver = None

                    temp_graph_builder.build_graphs(elements, temp_module_resolver, temp_symbol_resolver)
                    temp_graph_builder.save(repo_name)
                    self.logger.info(f"Saved graph data for {repo_name}")
                    
                    successfully_indexed.append(repo_name)
                    
                    self.logger.info(f"Successfully indexed and saved {repo_name}: {len(elements)} elements")
                else:
                    self.logger.warning(f"No vectors generated for {repo_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load repository {source}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # Continue with next repository
                continue
        
        if successfully_indexed:
            self.logger.info(f"Successfully indexed {len(successfully_indexed)} repositories:")
            for repo_name in successfully_indexed:
                self.logger.info(f"  - {repo_name}")
            
            # Merge all indexed repositories into the main vector store for statistics
            self.logger.info("Merging repositories into main vector store for statistics...")
            if self.vector_store.dimension is None:
                self.vector_store.initialize(self.embedder.embedding_dim)
            
            for repo_name in successfully_indexed:
                if self.vector_store.merge_from_index(repo_name):
                    self.logger.info(f"Merged {repo_name} into main store")
                else:
                    self.logger.warning(f"Failed to merge {repo_name}")
        else:
            self.logger.error("No repositories were successfully indexed")
        
        self.repo_indexed = len(successfully_indexed) > 0
        self.repo_loaded = len(successfully_indexed) > 0
        
        self.logger.info(f"Indexing complete. Each repository saved separately.")
    
    def list_repositories(self) -> List[Dict[str, Any]]:
        """
        List all indexed repositories
        
        Returns:
            List of repository information dictionaries
        """
        repo_names = self.vector_store.get_repository_names()
        repo_counts = self.vector_store.get_count_by_repository()
        
        repositories = []
        for repo_name in repo_names:
            repo_info = self.loaded_repositories.get(repo_name, {})
            repositories.append({
                'name': repo_name,
                'element_count': repo_counts.get(repo_name, 0),
                'file_count': repo_info.get('file_count', 0),
                'size_mb': repo_info.get('total_size_mb', 0),
                'url': repo_info.get('url', 'N/A'),
            })
        
        return repositories
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all indexed repositories
        
        Returns:
            Dictionary with repository statistics
        """
        repo_counts = self.vector_store.get_count_by_repository()
        repo_names = self.vector_store.get_repository_names()
        
        stats = {
            'total_repositories': len(repo_names),
            'total_elements': self.vector_store.get_count(),
            'repositories': []
        }
        
        for repo_name in repo_names:
            repo_info = self.loaded_repositories.get(repo_name, {})
            stats['repositories'].append({
                'name': repo_name,
                'elements': repo_counts.get(repo_name, 0),
                'files': repo_info.get('file_count', 0),
                'size_mb': repo_info.get('total_size_mb', 0),
            })
        
        return stats
    
    def _load_multi_repo_cache(self, repo_names: Optional[List[str]] = None) -> bool:
        """
        Load multi-repository index from cache by merging individual repository indices
        
        Args:
            repo_names: Optional list of specific repository names to load.
                       If None, loads all available repositories.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Discover available repository indexes
            persist_dir = self.vector_store.persist_dir
            available_repos = []
            
            if os.path.exists(persist_dir):
                for file in os.listdir(persist_dir):
                    if file.endswith('.faiss'):
                        repo_name = file.replace('.faiss', '')
                        metadata_file = os.path.join(persist_dir, f"{repo_name}_metadata.pkl")
                        if os.path.exists(metadata_file):
                            available_repos.append(repo_name)
            
            if not available_repos:
                self.logger.error("No repository indexes found")
                return False
            
            # Filter repositories if specific ones are requested
            if repo_names:
                repos_to_load = [r for r in available_repos if r in repo_names]
                if not repos_to_load:
                    self.logger.error(f"None of the requested repositories found: {repo_names}")
                    return False
            else:
                repos_to_load = available_repos
            
            self.logger.info(f"Found {len(repos_to_load)} repository indexes: {', '.join(repos_to_load)}")
            
            # Always reinitialize for clean merge
            self.vector_store.initialize(self.embedder.embedding_dim)
            
            # Load each repository index and merge them
            for repo_name in repos_to_load:
                self.logger.info(f"Loading index for {repo_name}...")
                try:
                    # Merge this repository's index into the main vector store
                    if self.vector_store.merge_from_index(repo_name):
                        self.logger.info(f"Successfully merged {repo_name}")
                    else:
                        self.logger.warning(f"Failed to merge index for {repo_name}")
                        
                except Exception as e:
                    self.logger.error(f"Error loading {repo_name}: {e}")
                    continue
            
            # Check if we successfully loaded any repositories
            if self.vector_store.get_count() == 0:
                self.logger.error("Failed to load any repository indexes")
                return False
            
            # Register loaded repositories
            # We know which repos were successfully loaded from repos_to_load
            for repo_name in repos_to_load:
                if repo_name not in self.loaded_repositories:
                    self.loaded_repositories[repo_name] = {
                        "name": repo_name,
                        "file_count": 0,  # Will be updated if needed
                        "total_size_mb": 0,
                    }
            
            # Try to load BM25 and graph data from saved files
            # For multi-repo, we merge BM25 data from all loaded repositories
            self.logger.info("Loading BM25 and graph data...")
            
            all_bm25_elements = []
            all_bm25_corpus = []
            graphs_loaded = False
            
            for repo_name in repos_to_load:
                # Try loading BM25 for each repo
                bm25_path = os.path.join(self.retriever.persist_dir, f"{repo_name}_bm25.pkl")
                if os.path.exists(bm25_path):
                    try:
                        with open(bm25_path, 'rb') as f:
                            data = pickle.load(f)
                            all_bm25_corpus.extend(data["bm25_corpus"])
                            
                            # Reconstruct CodeElement objects
                            for elem_dict in data["bm25_elements"]:
                                all_bm25_elements.append(CodeElement(**elem_dict))
                        
                        self.logger.info(f"Loaded BM25 data for {repo_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load BM25 data for {repo_name}: {e}")
                
                # Load graph data (merge into main graph)
                if not graphs_loaded:
                    # Load the first repository's graph as base
                    if self.graph_builder.load(repo_name):
                        graphs_loaded = True
                        self.logger.info(f"Loaded graph data from {repo_name} as base")
                else:
                    # Merge additional repository graphs
                    if self.graph_builder.merge_from_file(repo_name):
                        self.logger.info(f"Merged graph data from {repo_name}")
                    else:
                        self.logger.warning(f"Failed to merge graph data from {repo_name}")
                    # TODO: Merge additional repository graphs if needed
            # Rebuild FULL BM25 index with merged data (for repository selection)
            if all_bm25_elements and all_bm25_corpus:
                self.retriever.full_bm25_elements = all_bm25_elements
                self.retriever.full_bm25_corpus = all_bm25_corpus
                self.retriever.full_bm25 = BM25Okapi(all_bm25_corpus)
                self.logger.info(f"Rebuilt full BM25 index with {len(all_bm25_elements)} merged elements")
            else:
                # Fallback: reconstruct from metadata
                self.logger.info("No BM25 data found, reconstructing from metadata...")
                elements = self._reconstruct_elements_from_metadata()
                
                if elements:
                    self.retriever.index_for_bm25(elements)
                    self.logger.info(f"Rebuilt BM25 index with {len(elements)} elements")
                    
                    if not graphs_loaded:
                        self.graph_builder.build_graphs(elements)
                        self.logger.info("Rebuilt code graph")
                else:
                    self.logger.warning("No elements reconstructed from metadata")
            
            # Build separate BM25 index for repository overviews
            self.retriever.build_repo_overview_bm25()
            self.logger.info("Built separate BM25 index for repository overviews")
            
            self.multi_repo_mode = True
            self.repo_indexed = True
            self.repo_loaded = True
            
            self.logger.info(f"Successfully loaded {len(repos_to_load)} repositories with {self.vector_store.get_count()} total vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load multi-repo cache: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    # ------------------------------------------------------------------
    # Incremental indexing
    # ------------------------------------------------------------------

    def _build_file_manifest(self, elements, repo_root) -> dict:
        """Build a file manifest mapping files to their mtime/size and element IDs."""
        manifest = {
            "repo_name": self.repo_info.get("name", ""),
            "created_at": datetime.now().isoformat(),
            "files": {},
        }

        for elem in elements:
            rel_path = elem.relative_path
            if rel_path not in manifest["files"]:
                abs_path = os.path.join(repo_root, rel_path)
                try:
                    stat = os.stat(abs_path)
                    manifest["files"][rel_path] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "element_ids": [],
                    }
                except OSError:
                    manifest["files"][rel_path] = {
                        "mtime": 0.0,
                        "size": 0,
                        "element_ids": [],
                    }
            manifest["files"][rel_path]["element_ids"].append(elem.id)

        return manifest

    def _save_file_manifest(self, repo_name, manifest) -> None:
        """Save file manifest to disk as JSON."""
        manifest_path = os.path.join(
            self.vector_store.persist_dir, f"{repo_name}_manifest.json"
        )
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        self.logger.info(f"Saved file manifest: {manifest_path}")

    def _load_file_manifest(self, repo_name):
        """Load file manifest from disk. Returns None if missing."""
        manifest_path = os.path.join(
            self.vector_store.persist_dir, f"{repo_name}_manifest.json"
        )
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load manifest for '{repo_name}': {e}")
            return None

    def _load_existing_metadata(self, repo_name: str) -> list:
        """Load existing vector store metadata for a repo directly from disk."""
        meta_path = os.path.join(
            self.vector_store.persist_dir, f"{repo_name}_metadata.pkl"
        )
        if not os.path.exists(meta_path):
            return []
        try:
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
            return data.get("metadata", [])
        except Exception as e:
            self.logger.warning(f"Failed to load metadata for '{repo_name}': {e}")
            return []

    def _detect_file_changes(self, repo_name, current_files):
        """Compare current files against saved manifest to detect changes.

        Returns dict with added/modified/deleted/unchanged lists, or None
        if no manifest exists.
        """
        manifest = self._load_file_manifest(repo_name)
        if manifest is None:
            return None

        manifest_files = manifest.get("files", {})

        # Build lookup of current files with stat info
        current_lookup = {}
        for file_info in current_files:
            rel_path = file_info["relative_path"]
            abs_path = file_info["path"]
            try:
                stat = os.stat(abs_path)
                current_lookup[rel_path] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "file_info": file_info,
                }
            except OSError:
                continue

        added, modified, deleted, unchanged = [], [], [], []

        for rel_path, info in current_lookup.items():
            if rel_path not in manifest_files:
                added.append(rel_path)
            else:
                saved = manifest_files[rel_path]
                if info["mtime"] != saved["mtime"] or info["size"] != saved["size"]:
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
            "manifest": manifest,
            "current_lookup": current_lookup,
        }

    def _collect_unchanged_elements(self, manifest, unchanged_files, existing_metadata) -> tuple:
        """Collect element dicts and IDs for unchanged files from existing metadata."""
        unchanged_element_ids = set()
        for rel_path in unchanged_files:
            file_entry = manifest.get("files", {}).get(rel_path, {})
            for elem_id in file_entry.get("element_ids", []):
                unchanged_element_ids.add(elem_id)

        unchanged_elements = [
            meta for meta in existing_metadata
            if meta.get("id") in unchanged_element_ids
        ]

        return unchanged_elements, list(unchanged_element_ids)

    def incremental_reindex(self, repo_name: str, repo_path: str = None) -> dict:
        """Perform incremental reindexing: only re-embed changed files.

        Unchanged files reuse their existing embeddings. FAISS, BM25, and
        graphs are rebuilt from the combined element set (fast, since no
        model inference is needed for unchanged elements).

        Args:
            repo_name: Canonical repository name.
            repo_path: Local filesystem path to the repository.

        Returns:
            Dict with status and change summary.
        """
        self.logger.info(f"Starting incremental reindex for '{repo_name}'")

        # 1. Load manifest (skip if missing)
        manifest = self._load_file_manifest(repo_name)
        if manifest is None:
            self.logger.info(f"No manifest for '{repo_name}', skipping incremental")
            return {"status": "no_manifest", "changes": 0}

        # 2. Set up loader for this repo
        if not repo_path or not os.path.isdir(repo_path):
            self.logger.warning(f"Invalid repo path for '{repo_name}': {repo_path}")
            return {"status": "path_not_found", "changes": 0}

        self.loader.load_from_path(repo_path)
        self.config["repo_root"] = repo_path

        # 3. Scan current files and detect changes
        current_files = self.loader.scan_files()
        changes = self._detect_file_changes(repo_name, current_files)
        if changes is None:
            return {"status": "no_manifest", "changes": 0}

        added = changes["added"]
        modified = changes["modified"]
        deleted = changes["deleted"]
        unchanged = changes["unchanged"]
        total_changes = len(added) + len(modified) + len(deleted)

        self.logger.info(
            f"Changes: +{len(added)} ~{len(modified)} -{len(deleted)} ={len(unchanged)}"
        )

        if total_changes == 0:
            return {"status": "no_changes", "changes": 0}

        # 4. Load existing metadata from disk
        existing_metadata = self._load_existing_metadata(repo_name)
        if not existing_metadata:
            self.logger.warning(f"No existing metadata for '{repo_name}'")
            return {"status": "no_metadata", "changes": 0}

        # 5. Collect unchanged elements (with pre-computed embeddings)
        unchanged_elements, _ = self._collect_unchanged_elements(
            changes["manifest"], unchanged, existing_metadata
        )
        self.logger.info(
            f"Preserved {len(unchanged_elements)} elements from {len(unchanged)} unchanged files"
        )

        # 6. Parse & embed changed files
        changed_file_infos = []
        for rp in added + modified:
            lookup = changes["current_lookup"].get(rp)
            if lookup and lookup.get("file_info"):
                changed_file_infos.append(lookup["file_info"])

        new_elements = []
        if changed_file_infos:
            repo_url = self.loaded_repositories.get(repo_name, {}).get("url")
            new_elements = self.indexer.index_files(
                changed_file_infos, repo_name, repo_url
            )
            self.logger.info(
                f"Indexed {len(new_elements)} elements from {len(changed_file_infos)} changed files"
            )

        # 7. Combine: convert unchanged metadata dicts → CodeElement objects
        all_elements = []
        for meta in unchanged_elements:
            try:
                all_elements.append(CodeElement(
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
                ))
            except Exception as e:
                self.logger.warning(f"Failed to reconstruct element: {e}")
        all_elements.extend(new_elements)
        self.logger.info(f"Total elements after merge: {len(all_elements)}")

        # 8. Rebuild FAISS (temporary store — main instance untouched)
        temp_store = VectorStore(self.config)
        temp_store.initialize(self.embedder.embedding_dim)

        vectors, metadata_list = [], []
        for elem in all_elements:
            embedding = elem.metadata.get("embedding")
            if embedding is not None:
                vectors.append(embedding)
                metadata_list.append(elem.to_dict())

        if vectors:
            temp_store.add_vectors(np.array(vectors), metadata_list)

        # 9. Rebuild BM25 (temporary retriever)
        temp_retriever = HybridRetriever(
            self.config, temp_store, self.embedder,
            CodeGraphBuilder(self.config), repo_root=repo_path,
        )
        temp_retriever.index_for_bm25(all_elements)

        # 10. Rebuild graphs (temporary builder)
        temp_graph = CodeGraphBuilder(self.config)
        module_resolver, symbol_resolver = None, None
        try:
            gib = GlobalIndexBuilder(self.config)
            gib.build_maps(all_elements, repo_path)
            module_resolver = ModuleResolver(gib)
            symbol_resolver = SymbolResolver(gib, module_resolver)
        except Exception as e:
            self.logger.warning(f"Resolver init failed during incremental reindex: {e}")

        temp_graph.build_graphs(all_elements, module_resolver, symbol_resolver)

        # 11. Save all artifacts
        if self._should_persist_indexes():
            temp_store.save(repo_name)
            temp_retriever.save_bm25(repo_name)
            temp_graph.save(repo_name)
            new_manifest = self._build_file_manifest(all_elements, repo_path)
            self._save_file_manifest(repo_name, new_manifest)
            self.logger.info(f"Saved all artifacts for '{repo_name}'")

        return {
            "status": "success",
            "changes": total_changes,
            "added_files": len(added),
            "modified_files": len(modified),
            "deleted_files": len(deleted),
            "unchanged_files": len(unchanged),
            "total_elements": len(all_elements),
            "new_elements": len(new_elements),
            "preserved_elements": len(unchanged_elements),
        }

    def cleanup(self):
        """Cleanup resources"""
        self.shutdown()
        self.loader.cleanup()
        self.logger.info("Cleanup complete")

    def shutdown(self):
        """Stop background workers."""
        if self._redo_worker:
            self._redo_worker.stop()
        if self.graph_runtime:
            self.graph_runtime.close()
    
    def _get_full_dialogue_history(self, session_id: Optional[str], enable_multi_turn: bool) -> Optional[List[Dict[str, Any]]]:
        """
        Get full dialogue history for answer generation
        
        Args:
            session_id: Session ID
            enable_multi_turn: Whether multi-turn is enabled
        
        Returns:
            List of dialogue turns or None
        """
        if not enable_multi_turn or not session_id:
            return None

        context_rounds = self.config.get("generation", {}).get("context_rounds", 10)
        history = self.cache_manager.get_dialogue_history(session_id, max_turns=context_rounds)

        return history if history else None
    
    def _get_next_turn_number(self, session_id: str) -> int:
        """
        Get the next turn number for a session
        
        Args:
            session_id: Session ID
        
        Returns:
            Next turn number (1-indexed)
        """
        session_index = self.cache_manager._get_session_index(session_id)
        if session_index:
            return session_index.get("total_turns", 0) + 1
        else:
            return 1
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get dialogue history for a session
        
        Args:
            session_id: Session ID
        
        Returns:
            List of dialogue turns
        """
        return self.cache_manager.get_dialogue_history(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a dialogue session
        
        Args:
            session_id: Session ID
        
        Returns:
            True if successful
        """
        return self.cache_manager.delete_session(session_id)
    
    def remove_repository(self, repo_name: str, delete_source: bool = True) -> Dict[str, Any]:
        """
        Fully remove a repository: vector index files, BM25, graphs,
        repo overview, and optionally the cloned source code.

        Args:
            repo_name: Name of the repository to remove
            delete_source: If True, also remove ./repos/<repo_name>

        Returns:
            Dict with deleted files and freed bytes
        """
        import shutil

        persist_dir = self.vector_store.persist_dir
        deleted_files = []
        freed_bytes = 0

        # Files to delete from vector_store directory
        file_patterns = [
            f"{repo_name}.faiss",
            f"{repo_name}_metadata.pkl",
            f"{repo_name}_bm25.pkl",
            f"{repo_name}_graphs.pkl",
        ]

        for fname in file_patterns:
            fpath = os.path.join(persist_dir, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                os.remove(fpath)
                deleted_files.append(fname)
                freed_bytes += size
                self.logger.info(f"Deleted {fpath} ({size / (1024*1024):.2f} MB)")

        # Remove overview entry from repo_overviews.pkl
        if self.vector_store.delete_repo_overview(repo_name):
            deleted_files.append("repo_overviews.pkl (entry)")
            self.logger.info(f"Deleted overview entry for {repo_name}")

        # Remove cloned source code
        if delete_source:
            repo_root = getattr(self.loader, "safe_repo_root", self.config.get("repo_root", "./repos"))
            repo_dir = os.path.join(repo_root, repo_name)
            if os.path.isdir(repo_dir):
                dir_size = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fns in os.walk(repo_dir)
                    for f in fns
                )
                shutil.rmtree(repo_dir)
                deleted_files.append(f"repos/{repo_name}/")
                freed_bytes += dir_size
                self.logger.info(f"Deleted source directory {repo_dir}")

        # Invalidate scan cache
        self.vector_store.invalidate_scan_cache()

        return {
            "repo_name": repo_name,
            "deleted_files": deleted_files,
            "freed_bytes": freed_bytes,
            "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all dialogue sessions with enriched metadata

        Returns:
            List of session metadata with first query as title
        """
        sessions = self.cache_manager.list_sessions()

        # Enrich each session with the first query as a title
        enriched_sessions = []
        for session in sessions:
            session_id = session.get("session_id", "")
            if session_id:
                # Get the first turn to use its query as title
                first_turn = self.cache_manager.get_dialogue_turn(session_id, 1)
                if first_turn:
                    first_query = first_turn.get("query", "")
                    # Truncate long queries
                    if len(first_query) > 80:
                        title = first_query[:77] + "..."
                    else:
                        title = first_query
                    session["title"] = title
                else:
                    session["title"] = f"Session {session_id}"
            else:
                session["title"] = "Unknown Session"

            enriched_sessions.append(session)

        return enriched_sessions
