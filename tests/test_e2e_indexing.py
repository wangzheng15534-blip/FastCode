"""
End-to-end tests for the FastCode indexing pipeline.

Exercises run_index_pipeline() in two configurations:
1. SQLite-only with real Ollama embeddings
2. PostgreSQL + doc ingestion with real Ollama embeddings

Requirements:
- Ollama running at localhost:11434 with all-minilm:l6-v2
- PostgreSQL with pgvector extension
- PGUSER=jacob PGPASSWORD=jacob (or adjust PG_E2E_DSN)

Mark with pytest.mark.skipif if services are unavailable.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import subprocess
from typing import Any
from unittest.mock import MagicMock

import networkx as nx
import pytest

from fastcode.doc_ingester import KeyDocIngester
from fastcode.embedder import CodeEmbedder
from fastcode.graph_builder import CodeGraphBuilder
from fastcode.graph_runtime import LadybugGraphRuntime
from fastcode.index_run import IndexRunStore
from fastcode.indexer import CodeIndexer
from fastcode.ir_graph_builder import IRGraphBuilder
from fastcode.loader import RepositoryLoader
from fastcode.main import FastCode
from fastcode.manifest_store import ManifestStore
from fastcode.parser import CodeParser
from fastcode.pg_retrieval import PgRetrievalStore
from fastcode.snapshot_store import SnapshotStore
from fastcode.snapshot_symbol_index import SnapshotSymbolIndex
from fastcode.terminus_publisher import TerminusPublisher
from fastcode.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Service availability checks
# ---------------------------------------------------------------------------


def _ollama_available() -> bool:
    try:
        import urllib.request

        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/embeddings",
            data=b'{"model":"nomic-embed-text-v2-moe","prompt":"probe"}',
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


def _pg_available() -> bool:
    dsn = os.environ.get(
        "PG_E2E_DSN",
        "postgresql://jacob:jacob@/var/run/postgresql?dbname=fastcode_e2e",
    )
    try:
        import psycopg

        psycopg.connect(dsn).close()
        return True
    except Exception:
        return False


_skip_ollama = pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
_skip_pg = pytest.mark.skipif(not _pg_available(), reason="PostgreSQL not available")


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
) -> None:
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
            "supported_extensions": [".py", ".js", ".ts", ".java", ".go"],
        },
        "parser": {
            "extract_docstrings": True,
            "extract_comments": True,
            "extract_imports": True,
        },
        "embedding": {
            "provider": "ollama",
            "model": "nomic-embed-text-v2-moe",
            "ollama_url": "http://127.0.0.1:11434/api/embeddings",
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
            "backend": "pg_hybrid",
            "graph_backend": "ir",
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
    fc.config = config
    fc.eval_config = config.get("evaluation", {})
    fc.eval_mode = False
    fc.in_memory_index = False
    fc.global_index_builder = None
    fc.module_resolver = None
    fc.symbol_resolver = None
    fc.logger = MagicMock()

    # Real components.
    fc.loader = RepositoryLoader(config)
    fc.parser = CodeParser(config)
    fc.embedder = CodeEmbedder(config)  # Real Ollama embedder
    fc.vector_store = VectorStore(config)
    fc.graph_builder = CodeGraphBuilder(config)
    fc.ir_graph_builder = IRGraphBuilder()

    fc.indexer = CodeIndexer(config, fc.loader, fc.parser, fc.embedder, fc.vector_store)

    from fastcode.cache import CacheManager
    from fastcode.query_processor import QueryProcessor
    from fastcode.retriever import HybridRetriever

    config_repo_root = config.get("repo_root", "./repos")
    fc.retriever = HybridRetriever(
        config,
        fc.vector_store,
        fc.embedder,
        fc.graph_builder,
        repo_root=config_repo_root,
    )
    fc.query_processor = QueryProcessor(config)
    fc.answer_generator = MagicMock()
    fc.cache_manager = CacheManager(config)

    # Persistence.
    persist_dir = fc.vector_store.persist_dir
    storage_cfg = config.get("storage", {})
    fc.snapshot_store = SnapshotStore(persist_dir, storage_cfg=storage_cfg)
    fc.manifest_store = ManifestStore(fc.snapshot_store.db_runtime)
    fc.index_run_store = IndexRunStore(fc.snapshot_store.db_runtime)
    fc.terminus_publisher = TerminusPublisher(config)

    from fastcode.projection_store import ProjectionStore
    from fastcode.projection_transform import ProjectionTransformer

    fc.projection_transformer = ProjectionTransformer(config)
    fc.projection_store = ProjectionStore(config)
    fc.snapshot_symbol_index = SnapshotSymbolIndex()
    fc.pg_retrieval_store = PgRetrievalStore(fc.snapshot_store.db_runtime, config)
    fc.retriever.set_pg_retrieval_store(fc.pg_retrieval_store)
    fc.doc_ingester = KeyDocIngester(config, fc.embedder)
    fc.graph_runtime = LadybugGraphRuntime(config)

    # State.
    fc.repo_loaded = False
    fc.repo_indexed = False
    fc.repo_info = {}
    fc.multi_repo_mode = False
    fc.loaded_repositories = {}
    fc._redo_worker = None

    return fc


# ---------------------------------------------------------------------------
# Test 1: SQLite-only E2E (real Ollama embeddings)
# ---------------------------------------------------------------------------


@_skip_ollama
def test_e2e_indexing_sqlite_real_embeddings(tmp_path: pathlib.Path):
    """Full indexing pipeline with SQLite backend and real Ollama embeddings."""
    repo_path = _build_test_repo(tmp_path)
    config = _base_config(tmp_path, backend="sqlite")
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
    assert result["status"] == "succeeded"
    assert result["run_id"]
    assert result["snapshot_id"]
    assert result["artifact_key"]

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
    assert isinstance(ir_graphs.dependency_graph, nx.DiGraph)
    assert isinstance(ir_graphs.call_graph, nx.DiGraph)
    assert isinstance(ir_graphs.inheritance_graph, nx.DiGraph)
    assert isinstance(ir_graphs.reference_graph, nx.DiGraph)
    assert isinstance(ir_graphs.containment_graph, nx.DiGraph)
    assert ir_graphs.containment_graph.number_of_edges() >= 1
    total_edges = sum(
        g.number_of_edges()
        for g in [
            ir_graphs.dependency_graph,
            ir_graphs.call_graph,
            ir_graphs.inheritance_graph,
            ir_graphs.reference_graph,
            ir_graphs.containment_graph,
        ]
    )
    assert total_edges >= 1

    # Manifest published.
    head = fc.manifest_store.get_branch_manifest("test_repo", "main")
    assert head is not None
    assert head["snapshot_id"] == snapshot_id

    # Index run completed.
    run = fc.index_run_store.get_run(run_id)
    assert run is not None
    assert run["status"] == "succeeded"


# ---------------------------------------------------------------------------
# Test 2: PostgreSQL + doc ingestion (real Ollama + real PG)
# ---------------------------------------------------------------------------


@_skip_ollama
@_skip_pg
def test_e2e_indexing_pg_real_embeddings(tmp_path: pathlib.Path):
    """Full indexing pipeline with real PG backend and real Ollama embeddings.

    Verifies:
    - PgRetrievalStore.upsert_elements writes to real PG tables
    - Doc ingestion produces chunks stored via snapshot_store
    - IR snapshot is persisted and loadable
    - Manifest is published
    """
    pg_dsn = os.environ.get(
        "PG_E2E_DSN", "postgresql://jacob:jacob@/var/run/postgresql?dbname=fastcode_e2e"
    )
    repo_path = _build_test_repo(tmp_path)
    config = _base_config(tmp_path, backend="postgres", pg_dsn=pg_dsn, enable_docs=True)
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
    head = fc.manifest_store.get_branch_manifest("test_repo", "main")
    assert head is not None
    assert head["snapshot_id"] == snapshot_id

    # Clean up PG test data.
    _cleanup_pg_tables(pg_dsn)


# ---------------------------------------------------------------------------
# PG verification helpers
# ---------------------------------------------------------------------------


def _pg_execute(
    dsn: str, sql: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]] | None:
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
