"""
End-to-end tests for the FastCode indexing pipeline.

Exercises run_index_pipeline() in two configurations:
1. SQLite-only with real Ollama embeddings
2. PostgreSQL + doc ingestion with real Ollama embeddings

Requirements:
- Ollama embeddings endpoint at FASTCODE_E2E_OLLAMA_URL
  (defaults to http://127.0.0.1:11434/api/embeddings)
- PostgreSQL with pgvector extension at PG_E2E_DSN
  (defaults to postgresql://jacob:jacob@/var/run/postgresql?dbname=fastcode_e2e)

These are real-service e2e tests. They should fail fast when the required
services are unavailable rather than being silently skipped.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import subprocess
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from fastcode.graph.build import CodeGraphBuilder
from fastcode.inbound.config_mapper import config_from_mapping
from fastcode.indexing.doc_ingester import KeyDocIngester
from fastcode.indexing.embedder import CodeEmbedder
from fastcode.indexing.indexer import CodeIndexer
from fastcode.indexing.loader import RepositoryLoader
from fastcode.indexing.parser import CodeParser
from fastcode.indexing.pipeline import IndexPipeline
from fastcode.indexing.terminus import TerminusPublisher
from fastcode.ir.graph import IRGraphBuilder
from fastcode.main.config import config_to_runtime_mapping
from fastcode.main.fastcode import FastCode
from fastcode.semantic.resolvers.registry import (
    build_default_semantic_resolver_registry,
)
from fastcode.semantic.symbol_index import SnapshotSymbolIndex
from fastcode.store.index_run import IndexRunStore
from fastcode.store.infrastructure.graph_runtime import LadybugGraphRuntime
from fastcode.store.infrastructure.runtime import DBRuntime
from fastcode.store.manifest import ManifestStore
from fastcode.store.pg_retrieval import PgRetrievalStore
from fastcode.store.snapshot import SnapshotStore
from fastcode.store.vector import VectorStore

pytestmark = [pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Test repo fixture
# ---------------------------------------------------------------------------

_TEST_PYTHON_SOURCE = '''\
"""Math utilities for vector operations."""

import pathlib
import math
from typing import List, Tuple, Any


class Vector:
    """A simple 2D vector."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def magnitude(self) -> float:
        """Return the Euclidean magnitude."""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self) -> "Vector":
        """Return a unit vector in the same direction."""
        mag = self.magnitude()
        return Vector(self.x / mag, self.y / mag)

    def dot(self, other: "Vector") -> float:
        """Dot product with another vector."""
        return self.x * other.x + self.y * other.y


def add_vectors(a: Vector, b: Vector) -> Vector:
    """Add two vectors component-wise."""
    return Vector(a.x + b.x, a.y + b.y)


def cross_2d(a: Vector, b: Vector) -> float:
    """2D cross product (scalar)."""
    return a.x * b.y - a.y * b.x
'''

_TEST_README = """\
# Math Utils

A tiny library for 2D vector arithmetic.

## Usage

```python
from math_utils import Vector
v = Vector(3, 4)
print(v.magnitude())  # 5.0
```
"""

_TEST_DESIGN_DOC = """\
# Architecture: Vector Module

## Overview

The vector module provides 2D vector operations for geometric algorithms.

## Design Decisions

- Immutable input pattern: methods return new Vector instances.
- No Z-axis support: keep the API minimal for 2D use cases.
"""


def _build_test_repo(tmp_path: pathlib.Path) -> Any:
    """Create a minimal git repo with Python source + docs."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    (repo_dir / "math_utils.py").write_text(_TEST_PYTHON_SOURCE, encoding="utf-8")
    (repo_dir / "README.md").write_text(_TEST_README, encoding="utf-8")
    (repo_dir / ".e2e_case_id").write_text(tmp_path.name, encoding="utf-8")
    docs_dir = repo_dir / "docs" / "design"
    docs_dir.mkdir(parents=True)
    (docs_dir / "arch.md").write_text(_TEST_DESIGN_DOC, encoding="utf-8")

    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "e2e@test.com"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "E2E Test"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "-A"], cwd=str(repo_dir), check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
    )
    return str(repo_dir)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _base_config(
    tmp_path: pathlib.Path,
    *,
    backend: Any = "sqlite",
    pg_dsn: Any = "",
    enable_docs: Any = False,
    ollama_url: Any = "http://127.0.0.1:11434/api/embeddings",
) -> dict[str, Any]:
    """Return a minimal config dict with all paths under tmp_path."""
    persist_dir = str(tmp_path / "persist")
    repo_root = str(tmp_path / "repos")
    cache_dir = str(tmp_path / "cache")
    backup_dir = str(tmp_path / "backup")
    for d in (persist_dir, repo_root, cache_dir, backup_dir):
        os.makedirs(d, exist_ok=True)

    config = {
        "storage": {
            "backend": backend,
            "postgres_dsn": pg_dsn,
            "pool_min": 2,
            "pool_max": 30,
        },
        "repository": {
            "clone_depth": 1,
            "max_file_size_mb": 5,
            "backup_directory": backup_dir,
            "ignore_patterns": ["*.pyc", "__pycache__", "node_modules", ".git"],
            "supported_extensions": [
                ".py",
                ".js",
                ".ts",
                ".java",
                ".go",
                ".rs",
                ".cs",
                ".c",
                ".h",
                ".cpp",
                ".hpp",
                ".zig",
                ".f90",
                ".jl",
            ],
        },
        "parser": {
            "extract_docstrings": True,
            "extract_comments": True,
            "extract_imports": True,
        },
        "embedding": {
            "provider": "ollama",
            "model": "all-minilm:l6-v2",
            "ollama_url": ollama_url,
            "device": "cpu",
            "batch_size": 32,
        },
        "indexing": {
            "levels": ["file", "class", "function", "documentation"],
            "generate_repo_overview": False,
        },
        "vector_store": {
            "persist_directory": persist_dir,
            "distance_metric": "cosine",
        },
        "retrieval": {
            "semantic_weight": 0.6,
            "keyword_weight": 0.3,
            "graph_weight": 0.1,
            "max_results": 5,
            "retrieval_backend": "pg_hybrid",
            "graph_expansion_backend": "ir",
        },
        "cache": {"enabled": False},
        "repo_root": repo_root,
        "terminus": {"endpoint": "", "api_key": ""},
        "graph_overlay": {"enabled": False},
        "logging": {"level": "WARNING", "console": False},
    }
    if enable_docs:
        config["docs_integration"] = {"enabled": True}
    return config


# ---------------------------------------------------------------------------
# FastCode builder (bypasses __init__ to avoid heavy setup)
# ---------------------------------------------------------------------------


def _build_fastcode(config: dict[str, Any]) -> Any:
    """Construct a FastCode instance, wiring real components from config."""
    fc = FastCode.__new__(FastCode)
    fc.runtime_config = config_from_mapping(config)
    fc.config = config_to_runtime_mapping(fc.runtime_config)
    fc.eval_config = fc.config.get("evaluation", {})
    fc.eval_mode = False
    fc.in_memory_index = False
    fc.global_index_builder = None
    fc.module_resolver = None
    fc.symbol_resolver = None
    fc.logger = MagicMock()

    # Real components.
    fc.loader = RepositoryLoader(fc.config)
    fc.parser = CodeParser(fc.config)
    fc.embedder = CodeEmbedder(fc.config)  # Real Ollama embedder
    fc.vector_store = VectorStore(fc.config)
    fc.graph_builder = CodeGraphBuilder(fc.config)
    fc.ir_graph_builder = IRGraphBuilder()

    fc.indexer = CodeIndexer(
        fc.config,
        fc.loader,
        fc.parser,
        fc.embedder,
        fc.vector_store,
    )

    from fastcode.query.processor import QueryProcessor
    from fastcode.query.retriever import HybridRetriever
    from fastcode.store.cache import CacheManager

    config_repo_root = fc.config.get("repo_root", "./repos")
    fc.retriever = HybridRetriever(
        fc.config,
        fc.vector_store,
        fc.embedder,
        fc.graph_builder,
        repo_root=config_repo_root,
    )
    fc.query_processor = QueryProcessor(fc.config)
    fc.answer_generator = MagicMock()
    fc.cache_manager = CacheManager(fc.config)

    # Persistence.
    persist_dir = fc.vector_store.persist_dir
    storage_cfg = fc.config.get("storage", {}) or {}
    db_runtime = DBRuntime.from_storage_config(
        sqlite_path=os.path.join(os.path.abspath(persist_dir), "lineage.db"),
        storage_cfg=storage_cfg,
    )
    fc.snapshot_store = SnapshotStore(persist_dir, db_runtime=db_runtime)
    fc.manifest_store = ManifestStore(db_runtime)
    fc.index_run_store = IndexRunStore(db_runtime)
    fc.terminus_publisher = TerminusPublisher(fc.config)

    from fastcode.indexing.projection_transform import ProjectionTransformer
    from fastcode.store.projection import ProjectionStore

    fc.projection_transformer = ProjectionTransformer(fc.config)
    fc.projection_store = ProjectionStore(fc.config)
    fc.snapshot_symbol_index = SnapshotSymbolIndex()
    fc.pg_retrieval_store = PgRetrievalStore(db_runtime, fc.config)
    fc.retriever.set_pg_retrieval_store(fc.pg_retrieval_store)
    fc.doc_ingester = KeyDocIngester(fc.config, fc.embedder)
    fc.graph_runtime = LadybugGraphRuntime(fc.config)

    # State.
    fc.repo_loaded = False
    fc.repo_indexed = False
    fc.repo_info = {}
    fc.multi_repo_mode = False
    fc.loaded_repositories = {}
    fc._redo_worker = None

    # Pipeline.
    fc.semantic_resolver_registry = build_default_semantic_resolver_registry()
    fc.pipeline = IndexPipeline(
        config=fc.config,
        logger=fc.logger,
        loader=fc.loader,
        snapshot_store=fc.snapshot_store,
        manifest_store=fc.manifest_store,
        index_run_store=fc.index_run_store,
        unit_artifact_store=SimpleNamespace(
            replace_snapshot_units=lambda **kwargs: None,
            list_snapshot_units=lambda snapshot_id: [],
        ),
        snapshot_symbol_index=fc.snapshot_symbol_index,
        vector_store=fc.vector_store,
        embedder=fc.embedder,
        indexer=fc.indexer,
        retriever=fc.retriever,
        graph_builder=fc.graph_builder,
        ir_graph_builder=fc.ir_graph_builder,
        pg_retrieval_store=fc.pg_retrieval_store,
        terminus_publisher=fc.terminus_publisher,
        doc_ingester=fc.doc_ingester,
        semantic_resolver_registry=fc.semantic_resolver_registry,
        set_repo_indexed=lambda v: setattr(fc, "repo_indexed", v),
        set_repo_loaded=lambda v: setattr(fc, "repo_loaded", v),
        set_repo_info=lambda v: setattr(fc, "repo_info", v),
    )

    return fc


# ---------------------------------------------------------------------------
# Test 1: SQLite-only E2E (real Ollama embeddings)
# ---------------------------------------------------------------------------


def test_e2e_indexing_sqlite_real_embeddings(
    tmp_path: pathlib.Path,
    require_ollama_all_minilm_embeddings: str,
):
    """Full indexing pipeline with SQLite backend and real Ollama embeddings."""
    repo_path = _build_test_repo(tmp_path)
    config = _base_config(
        tmp_path,
        backend="sqlite",
        ollama_url=require_ollama_all_minilm_embeddings,
    )
    fc = _build_fastcode(config)

    # Verify embedder is real Ollama, not mocked.
    assert fc.embedder.provider == "ollama"
    assert (fc.embedder.embedding_dim or 0) > 0

    result = fc.run_index_pipeline(
        source=repo_path,
        enable_scip=False,
        publish=True,
    )

    # Return shape.
    assert result["status"] in ("succeeded", "degraded")
    assert result["run_id"]
    assert result["snapshot_id"]
    assert result["artifact_key"]

    if result["status"] == "degraded":
        assert "terminus_not_configured" in result["warnings"]

    snapshot_id = result["snapshot_id"]
    run_id = result["run_id"]

    # Snapshot persisted and loadable.
    snapshot = fc.snapshot_store.load_snapshot(snapshot_id)
    assert snapshot is not None
    assert snapshot.snapshot_id == snapshot_id
    assert len(snapshot.documents) >= 1
    assert len(snapshot.symbols) >= 1
    assert len(snapshot.occurrences) >= 1
    assert len(snapshot.edges) >= 1

    # IR graphs saved (5 types).
    ir_graphs = fc.snapshot_store.load_ir_graphs(snapshot_id)
    assert ir_graphs is not None
    graph_handles = [
        ir_graphs.dependency_graph,
        ir_graphs.call_graph,
        ir_graphs.inheritance_graph,
        ir_graphs.reference_graph,
        ir_graphs.containment_graph,
    ]
    for graph in graph_handles:
        assert callable(getattr(graph, "number_of_nodes", None))
        assert callable(getattr(graph, "number_of_edges", None))
    assert ir_graphs.containment_graph.number_of_edges() >= 1
    total_edges = sum(g.number_of_edges() for g in graph_handles)
    assert total_edges >= 1

    # Manifest published.
    head = fc.manifest_store.get_branch_manifest_record("test_repo", "main")
    assert head is not None
    assert head.snapshot_id == snapshot_id

    # Index run completed.
    run = fc.index_run_store.get_run_record(run_id)
    assert run is not None
    assert run.status in ("succeeded", "degraded")


# ---------------------------------------------------------------------------
# Test 2: PostgreSQL + doc ingestion (real Ollama + real PG)
# ---------------------------------------------------------------------------


def test_e2e_indexing_pg_real_embeddings(
    tmp_path: pathlib.Path,
    require_ollama_all_minilm_embeddings: str,
    require_postgres_e2e: str,
):
    """Full indexing pipeline with real PG backend and real Ollama embeddings.

    Verifies:
    - PgRetrievalStore.upsert_elements writes to real PG tables
    - Doc ingestion produces chunks stored via snapshot_store
    - IR snapshot is persisted and loadable
    - Manifest is published
    """
    pg_dsn = require_postgres_e2e
    repo_path = _build_test_repo(tmp_path)
    config = _base_config(
        tmp_path,
        backend="postgres",
        pg_dsn=pg_dsn,
        enable_docs=True,
        ollama_url=require_ollama_all_minilm_embeddings,
    )
    fc = _build_fastcode(config)

    # Verify real services are wired.
    assert fc.embedder.provider == "ollama"
    assert fc.snapshot_store.db_runtime.backend == "postgres"
    assert fc.pg_retrieval_store.enabled is True
    assert fc.doc_ingester.enabled is True

    # Clean up any prior test data in PG tables for this snapshot.
    _cleanup_pg_tables(pg_dsn)

    result = fc.run_index_pipeline(
        source=repo_path,
        enable_scip=False,
        publish=True,
    )

    snapshot_id = result["snapshot_id"]

    # Pipeline succeeded.
    assert result["status"] in ("succeeded", "degraded")
    assert result["run_id"]
    assert snapshot_id

    # Snapshot persisted and loadable.
    snapshot = fc.snapshot_store.load_snapshot(snapshot_id)
    assert snapshot is not None
    assert len(snapshot.symbols) >= 1

    # Verify PG actually received upserted elements.
    _verify_pg_elements(pg_dsn, snapshot_id)

    # Verify PG search_documents table has rows.
    _verify_pg_search_documents(pg_dsn, snapshot_id)

    # Manifest published.
    head = fc.manifest_store.get_branch_manifest_record("test_repo", "main")
    assert head is not None
    assert head.snapshot_id == snapshot_id

    # Clean up PG test data.
    _cleanup_pg_tables(pg_dsn)


# ---------------------------------------------------------------------------
# PG verification helpers
# ---------------------------------------------------------------------------


def _pg_execute(
    dsn: str, sql: str, params: dict[str, Any] | None = None
) -> list[tuple[Any, ...]] | None:
    """Execute a query against the test PG database."""
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())  # type: ignore[arg-type]
        if cur.description:
            return cur.fetchall()
        conn.commit()
        return None


def _cleanup_pg_tables(dsn: str) -> None:
    """Remove test data from PG tables."""
    for table in (
        "embedding_vectors",
        "search_documents",
        "design_documents",
        "design_doc_mentions",
    ):
        with contextlib.suppress(Exception):
            _pg_execute(
                dsn, f"DELETE FROM {table} WHERE snapshot_id LIKE 'snap:test_repo:%'"
            )


def _verify_pg_elements(dsn: str, snapshot_id: str) -> None:
    """Assert embedding_vectors table has real rows for the snapshot."""
    rows = _pg_execute(
        dsn,
        "SELECT COUNT(*) FROM embedding_vectors WHERE snapshot_id = %s",
        (snapshot_id,),
    )
    assert rows is not None
    count = rows[0][0]
    assert count >= 1, f"Expected >= 1 row in embedding_vectors, got {count}"

    rows = _pg_execute(
        dsn,
        """
        SELECT element_id, metadata_json, vector_dims(embedding), embedding_arr
        FROM embedding_vectors
        WHERE snapshot_id = %s
        ORDER BY element_id
        """,
        (snapshot_id,),
    )
    assert rows is not None
    dimensions: set[int] = set()
    for element_id, metadata_json, vector_dim, embedding_arr in rows:
        assert embedding_arr is None, f"legacy embedding_arr populated: {element_id}"
        assert isinstance(vector_dim, int)
        assert vector_dim > 0
        dimensions.add(vector_dim)
        metadata = _metadata_payload(metadata_json)
        bad_paths = _bad_embedding_payload_paths(metadata)
        assert bad_paths == [], f"raw embedding leaked in metadata: {bad_paths}"
        nested_metadata = metadata.get("metadata")
        if not isinstance(nested_metadata, dict):
            nested_metadata = {}
        assert metadata.get("embedding_artifact_ref") or nested_metadata.get(
            "embedding_artifact_ref"
        ), f"embedding artifact ref missing: {element_id}"
        assert metadata.get("embedding_fingerprint") or nested_metadata.get(
            "embedding_fingerprint"
        ), f"embedding fingerprint missing: {element_id}"

    assert len(dimensions) == 1, f"mixed embedding dimensions found: {dimensions}"


def _verify_pg_search_documents(dsn: str, snapshot_id: str) -> None:
    """Assert search_documents table has real rows for the snapshot."""
    rows = _pg_execute(
        dsn,
        "SELECT COUNT(*) FROM search_documents WHERE snapshot_id = %s",
        (snapshot_id,),
    )
    assert rows is not None
    count = rows[0][0]
    assert count >= 1, f"Expected >= 1 row in search_documents, got {count}"


def _metadata_payload(raw_metadata: Any) -> dict[str, Any]:
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if isinstance(raw_metadata, str):
        parsed = json.loads(raw_metadata)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _bad_embedding_payload_paths(value: Any, path: str = "$") -> list[str]:
    bad_paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key == "embedding":
                bad_paths.append(child)
            if (
                isinstance(item, list)
                and item
                and all(
                    isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in item
                )
                and ("embedding" in key.lower() or key.lower() in {"vector", "vectors"})
            ):
                bad_paths.append(child)
            bad_paths.extend(_bad_embedding_payload_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            bad_paths.extend(_bad_embedding_payload_paths(item, f"{path}[{index}]"))
    return bad_paths
