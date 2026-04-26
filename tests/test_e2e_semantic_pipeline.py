"""
End-to-end tests for the semantic chunking pipeline integration.

Exercises:
1. Chonkie SemanticChunker uses the project's configured embedding model
2. Full index pipeline with PostgreSQL backend produces semantic chunks
3. Ladybug graph sync stores doc chunks and mentions
4. Query pipeline retrieves semantically relevant results

Requirements:
- Ollama running at localhost:11434 with nomic-embed-text-v2-moe
- PostgreSQL with pgvector extension
- LadybugDB (optional, graceful skip)
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import subprocess
from typing import Any
from unittest.mock import MagicMock

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


def _ladybug_available() -> bool:
    try:
        from real_ladybug import Connection  # noqa: F401

        return True
    except ImportError:
        return False


_skip_ollama = pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
_skip_pg = pytest.mark.skipif(not _pg_available(), reason="PostgreSQL not available")
_skip_ladybug = pytest.mark.skipif(
    not _ladybug_available(), reason="LadybugDB not installed"
)


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
'''

_TEST_DESIGN_DOC = """\
# Architecture: Vector Module

## Overview

The vector module provides 2D vector operations for geometric algorithms.
It supports magnitude, normalization, dot product, and cross product.

## Design Decisions

- Immutable input pattern: methods return new Vector instances.
- No Z-axis support: keep the API minimal for 2D use cases.
- All methods use float precision for coordinate values.

## Performance Characteristics

The magnitude calculation uses the standard Euclidean formula.
For large batches, consider using numpy vectorized operations.
The normalize method calls magnitude internally, so cache results if needed.

## Testing Strategy

Unit tests cover each method independently.
Integration tests verify the full arithmetic pipeline.
Property-based tests check mathematical invariants like associativity.
"""


def _build_test_repo(tmp_path: pathlib.Path) -> Any:
    """Create a minimal git repo with Python source + docs."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    (repo_dir / "math_utils.py").write_text(_TEST_PYTHON_SOURCE, encoding="utf-8")
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
        ["git", "add", "-A"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
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
    enable_docs: Any = True,
    enable_ladybug: Any = False,
    ladybug_db_path: Any = "",
) -> dict:
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
        config["docs_integration"] = {
            "enabled": True,
            "curated_paths": ["docs/design/**"],
            "chunk_token_size": 128,
            "similarity_threshold": 0.5,
        }

    if enable_ladybug:
        config["graph"] = {
            "ladybug": {
                "enabled": True,
                "db_path": ladybug_db_path or str(tmp_path / "ladybug" / "test.lb"),
            },
        }
    else:
        config["graph"] = {"ladybug": {"enabled": False}}

    return config


# ---------------------------------------------------------------------------
# FastCode builder
# ---------------------------------------------------------------------------


def _build_fastcode(config: dict) -> Any:
    """Construct a FastCode instance with real components from config."""
    fc = FastCode.__new__(FastCode)
    fc.config = config
    fc.eval_config = config.get("evaluation", {})
    fc.eval_mode = False
    fc.in_memory_index = False
    fc.global_index_builder = None
    fc.module_resolver = None
    fc.symbol_resolver = None
    fc.logger = MagicMock()

    fc.loader = RepositoryLoader(config)
    fc.parser = CodeParser(config)
    fc.embedder = CodeEmbedder(config)
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

    fc.repo_loaded = False
    fc.repo_indexed = False
    fc.repo_info = {}
    fc.multi_repo_mode = False
    fc.loaded_repositories = {}
    fc._redo_worker = None

    return fc


# ---------------------------------------------------------------------------
# PG helpers
# ---------------------------------------------------------------------------


def _pg_execute(dsn: str, sql: str, params: dict = None) -> None:
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())
        if cur.description:
            return cur.fetchall()
        conn.commit()
        return None


def _cleanup_pg_tables(dsn: str) -> None:
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


# ===========================================================================
# TEST 1: Chonkie uses project's configured embedding model
# ===========================================================================


@_skip_ollama
def test_semantic_chunker_uses_configured_embedding_model(tmp_path: pathlib.Path):
    """SemanticChunker should use sentence-transformers model from config,
    not the hardcoded default minishlab/potion-base-32M."""
    config = _base_config(tmp_path, enable_docs=True)
    config["embedding"] = {
        "provider": "sentence_transformers",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "batch_size": 32,
    }
    ingester = KeyDocIngester(config, CodeEmbedder(config))

    # Force chunker initialization
    chunker = ingester._ensure_chunker()
    assert chunker is not None, "SemanticChunker should initialize"

    # The chunker should use sentence-transformers model, not the default potion model.
    # Verify by checking that the embedding model object's class name contains
    # "SentenceTransformer" (not the default Model2Vec).
    model_obj = getattr(chunker, "embedding_model", None)
    assert model_obj is not None, "Chunker should have an embedding_model"
    model_cls = type(model_obj).__name__
    assert "SentenceTransformer" in model_cls, (
        f"Expected SentenceTransformerEmbeddings, got {model_cls}. "
        "The chunker should use the project's configured ST model, "
        "not the default minishlab/potion-base-32M."
    )

    # Chunk real text to verify semantic boundaries work
    text = """# Architecture

The system uses a microservices pattern. Each service owns its data.
Requests enter through the API gateway.

## Data Flow

Events flow through a message broker. Services consume and produce events.
The gateway handles authentication and rate limiting.
"""
    chunks = ingester._chunk_document(text)
    assert len(chunks) >= 2, f"Expected >= 2 semantic chunks, got {len(chunks)}"

    # Verify heading metadata preserved
    headings = {c["heading"] for c in chunks if c["heading"]}
    assert "Architecture" in headings
    assert "Data Flow" in headings


# ===========================================================================
# TEST 2: Full pipeline with PostgreSQL + semantic chunking
# ===========================================================================


@_skip_ollama
@_skip_pg
def test_e2e_semantic_indexing_with_postgres(tmp_path: pathlib.Path):
    """Full index pipeline with PostgreSQL backend + Chonkie semantic chunking.

    Verifies:
    - SemanticChunker produces chunks (not word-based fallback)
    - Chunks are stored in PG embedding_vectors table
    - Doc ingestion produces semantic chunks with headings
    - Snapshot persisted with correct symbols and documents
    """
    pg_dsn = os.environ.get(
        "PG_E2E_DSN",
        "postgresql://jacob:jacob@/var/run/postgresql?dbname=fastcode_e2e",
    )
    repo_path = _build_test_repo(tmp_path)
    config = _base_config(tmp_path, backend="postgres", pg_dsn=pg_dsn, enable_docs=True)
    fc = _build_fastcode(config)

    # Verify real services wired
    assert fc.embedder.provider == "ollama"
    assert fc.doc_ingester.enabled is True
    assert fc.snapshot_store.db_runtime.backend == "postgres"

    _cleanup_pg_tables(pg_dsn)

    result = fc.run_index_pipeline(
        source=repo_path,
        enable_scip=False,
        publish=True,
    )

    snapshot_id = result["snapshot_id"]
    assert result["status"] in ("succeeded", "degraded")
    assert snapshot_id

    # Snapshot persisted
    snapshot = fc.snapshot_store.load_snapshot(snapshot_id)
    assert snapshot is not None
    assert len(snapshot.documents) >= 1, "Should have at least code files"
    assert len(snapshot.symbols) >= 1, "Should have extracted symbols"

    # PG has embedding rows
    rows = _pg_execute(
        pg_dsn,
        "SELECT COUNT(*) FROM embedding_vectors WHERE snapshot_id = %s",
        (snapshot_id,),
    )
    assert rows is not None
    count = rows[0][0]
    assert count >= 1, f"Expected >= 1 embedding row, got {count}"

    # PG has search document rows
    rows = _pg_execute(
        pg_dsn,
        "SELECT COUNT(*) FROM search_documents WHERE snapshot_id = %s",
        (snapshot_id,),
    )
    assert rows is not None
    count = rows[0][0]
    assert count >= 1, f"Expected >= 1 search_document row, got {count}"

    # Doc ingester produced chunks via semantic chunking
    # Verify by checking chunk text content — semantic chunks should contain
    # complete sentences, not arbitrary word splits
    doc_result = fc.doc_ingester.ingest(
        repo_path=repo_path,
        repo_name="test_repo",
        snapshot_id=snapshot_id,
        snapshot=snapshot,
    )
    chunks = doc_result["chunks"]
    assert len(chunks) >= 1, "Should have doc chunks"

    for chunk in chunks:
        # Semantic chunks should end with sentence boundaries (for multi-sentence text)
        text = chunk.text.strip()
        if text and len(text) > 80:
            # Longer chunks should end at a sentence boundary or be the last chunk
            # (headings/short sections are exempt)
            assert text[-1] in ".!?\n`:", (
                f"Semantic chunk should end at sentence boundary: '...{text[-40:]}'"
            )

    _cleanup_pg_tables(pg_dsn)


# ===========================================================================
# TEST 3: Ladybug graph sync with semantic chunks
# ===========================================================================


@_skip_ollama
@_skip_ladybug
def test_e2e_semantic_indexing_with_ladybug(tmp_path: pathlib.Path):
    """Index pipeline with LadybugDB graph backend + semantic chunking.

    Verifies:
    - Ladybug runtime initializes and is enabled
    - Doc chunks are synced to Ladybug tables
    - Mentions are synced to Ladybug tables
    - Semantic chunking produces meaningful chunks
    """
    ladybug_path = str(tmp_path / "ladybug" / "test.lb")
    repo_path = _build_test_repo(tmp_path)
    config = _base_config(
        tmp_path,
        backend="sqlite",
        enable_docs=True,
        enable_ladybug=True,
        ladybug_db_path=ladybug_path,
    )
    fc = _build_fastcode(config)

    # Verify Ladybug is wired and enabled
    assert fc.graph_runtime.enabled is True, "Ladybug should be enabled"

    result = fc.run_index_pipeline(
        source=repo_path,
        enable_scip=False,
        publish=True,
    )

    snapshot_id = result["snapshot_id"]
    assert result["status"] in ("succeeded", "degraded")

    # Snapshot persisted
    snapshot = fc.snapshot_store.load_snapshot(snapshot_id)
    assert snapshot is not None
    assert len(snapshot.symbols) >= 1

    # Run doc ingestion and verify Ladybug sync
    doc_result = fc.doc_ingester.ingest(
        repo_path=repo_path,
        repo_name="test_repo",
        snapshot_id=snapshot_id,
        snapshot=snapshot,
    )
    chunks = doc_result["chunks"]
    mentions = doc_result["mentions"]

    assert len(chunks) >= 1, "Should have doc chunks"

    # Sync to Ladybug
    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "snapshot_id": c.snapshot_id,
            "repo_name": c.repo_name,
            "path": c.path,
            "title": c.title,
            "heading": c.heading,
            "doc_type": c.doc_type,
            "content": c.text,
        }
        for c in chunks
    ]
    synced = fc.graph_runtime.sync_docs(chunks=chunk_dicts, mentions=mentions)
    assert synced is True, "Ladybug sync should succeed"

    # Verify data in Ladybug via query_docs
    docs = fc.graph_runtime.query_docs(snapshot_id=chunks[0].snapshot_id)
    assert len(docs) >= 1, f"Expected >= 1 Ladybug doc, got {len(docs)}"

    # Verify mentions synced (sync should not error)
    assert isinstance(mentions, list)

    # Semantic chunks should have heading metadata
    headings = [c.heading for c in chunks if c.heading]
    assert len(headings) >= 1, "Semantic chunks should preserve heading metadata"


# ===========================================================================
# TEST 4: Full query pipeline retrieves semantically chunked docs
# ===========================================================================


@_skip_ollama
@_skip_pg
def test_e2e_semantic_query_pipeline_with_postgres(tmp_path: pathlib.Path):
    """Index with semantic chunking, then query and verify retrieval.

    Verifies:
    - Index produces semantic chunks stored in PG
    - HybridRetriever returns relevant results for a code query
    - Results include doc chunks with proper metadata
    """
    pg_dsn = os.environ.get(
        "PG_E2E_DSN",
        "postgresql://jacob:jacob@/var/run/postgresql?dbname=fastcode_e2e",
    )
    repo_path = _build_test_repo(tmp_path)
    config = _base_config(tmp_path, backend="postgres", pg_dsn=pg_dsn, enable_docs=True)
    fc = _build_fastcode(config)

    _cleanup_pg_tables(pg_dsn)

    # Index
    result = fc.run_index_pipeline(
        source=repo_path,
        enable_scip=False,
        publish=True,
    )
    snapshot_id = result["snapshot_id"]
    assert result["status"] in ("succeeded", "degraded")

    # Retrieve using semantic query
    query_result = fc.retriever.retrieve(
        query="vector magnitude calculation",
        filters={"snapshot_id": snapshot_id},
    )

    # Should return results
    assert len(query_result) >= 1, (
        f"Expected >= 1 retrieval result for 'vector magnitude', got {len(query_result)}"
    )

    # At least one result should reference Vector or magnitude
    # Results have structure: {"element": {...actual fields...}, "total_score": ...}
    all_text = " ".join(
        str(r.get("element", {}).get(k, ""))
        for r in query_result
        for k in (
            "code",
            "text",
            "docstring",
            "name",
            "summary",
            "file_path",
            "relative_path",
            "content",
        )
    ).lower()
    relevant = (
        "vector" in all_text
        or "magnitude" in all_text
        or "math" in all_text
        or "math_utils" in all_text
    )
    assert relevant, (
        f"Expected at least one result mentioning 'vector', 'magnitude', or 'math'. "
        f"Got names: {[r.get('element', {}).get('name', '?') for r in query_result[:5]]}"
    )

    _cleanup_pg_tables(pg_dsn)
