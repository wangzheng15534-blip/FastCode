"""Tests for supporting_docs annotation in L2 projection chunks."""

from __future__ import annotations

from typing import Any

import networkx as nx

from fastcode.ir_graph_builder import IRGraphs
from fastcode.projection_models import ProjectionScope
from fastcode.projection_transform import ProjectionTransformer
from fastcode.semantic_ir import IRDocument, IREdge, IRSnapshot, IRSymbol


def _sample_snapshot() -> IRSnapshot:
    doc = IRDocument(
        doc_id="doc:1", path="app/service.py", language="python", source_set={"ast"}
    )
    sym_a = IRSymbol(
        symbol_id="ast:snap:repo:abc:python:app/service.py:function:login:10:20",
        external_symbol_id=None,
        path="app/service.py",
        display_name="login",
        kind="function",
        language="python",
        start_line=10,
        end_line=20,
        source_priority=10,
        source_set={"ast"},
    )
    sym_b = IRSymbol(
        symbol_id="ast:snap:repo:abc:python:app/service.py:function:validate:22:30",
        external_symbol_id=None,
        path="app/service.py",
        display_name="validate",
        kind="function",
        language="python",
        start_line=22,
        end_line=30,
        source_priority=10,
        source_set={"ast"},
    )
    edges = [
        IREdge(
            edge_id="edge:contain:1",
            src_id=doc.doc_id,
            dst_id=sym_a.symbol_id,
            edge_type="contain",
            source="ast",
            confidence="resolved",
            doc_id=doc.doc_id,
        ),
        IREdge(
            edge_id="edge:contain:2",
            src_id=doc.doc_id,
            dst_id=sym_b.symbol_id,
            edge_type="contain",
            source="ast",
            confidence="resolved",
            doc_id=doc.doc_id,
        ),
        IREdge(
            edge_id="edge:call:1",
            src_id=sym_a.symbol_id,
            dst_id=sym_b.symbol_id,
            edge_type="call",
            source="ast",
            confidence="heuristic",
            doc_id=doc.doc_id,
        ),
    ]
    return IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:repo:abc",
        branch="main",
        commit_id="abc",
        documents=[doc],
        symbols=[sym_a, sym_b],
        edges=edges,
    )


def _sample_graphs() -> IRGraphs:
    return IRGraphs(
        dependency_graph=nx.DiGraph(),
        call_graph=nx.DiGraph(),
        inheritance_graph=nx.DiGraph(),
        reference_graph=nx.DiGraph(),
        containment_graph=nx.DiGraph(),
    )


def _config() -> dict[str, Any]:
    return {
        "projection": {"enable_leiden": True, "leiden_resolutions": [1.0]},
        "embedding": {"enabled": False},
    }


def _build(transformer: Any, snapshot: IRSnapshot, doc_mentions: Any = None) -> None:
    scope = ProjectionScope(
        scope_kind="snapshot",
        snapshot_id=snapshot.snapshot_id,
        scope_key=snapshot.snapshot_id,
    )
    return transformer.build(
        scope=scope,
        snapshot=snapshot,
        ir_graphs=_sample_graphs(),
        doc_mentions=doc_mentions,
    )


def test_l2_no_supporting_docs_when_no_mentions():
    """supporting_docs should be absent when doc_mentions is None."""
    transformer = ProjectionTransformer(_config())
    snapshot = _sample_snapshot()
    result = _build(transformer, snapshot)
    chunks = result.chunks
    assert len(chunks) > 0
    for chunk in chunks:
        assert "supporting_docs" not in chunk.get("content", {})


def test_l2_supporting_docs_with_mentions():
    """supporting_docs should appear when doc_mentions reference cluster symbols."""
    transformer = ProjectionTransformer(_config())
    snapshot = _sample_snapshot()
    sym_a = snapshot.symbols[0]
    sym_b = snapshot.symbols[1]
    doc_mentions = [
        {
            "chunk_id": "doc_adr_001",
            "symbol_id": sym_a.symbol_id,
            "symbol_name": "login",
            "confidence": "exact_name",
        },
        {
            "chunk_id": "doc_adr_002",
            "symbol_id": sym_b.symbol_id,
            "symbol_name": "validate",
            "confidence": "exact_name",
        },
    ]
    result = _build(transformer, snapshot, doc_mentions=doc_mentions)
    chunks = result.chunks
    assert len(chunks) > 0
    chunks_with_docs = [c for c in chunks if "supporting_docs" in c.get("content", {})]
    assert len(chunks_with_docs) > 0
    for chunk in chunks_with_docs:
        docs = chunk["content"]["supporting_docs"]
        assert isinstance(docs, list)
        for d in docs:
            assert "chunk_id" in d
            assert "mentioned_symbols" in d


def test_l2_supporting_docs_capping():
    """supporting_docs should be capped at max_supporting_docs_per_cluster."""
    config = _config()
    config["projection"]["max_supporting_docs_per_cluster"] = 2
    transformer = ProjectionTransformer(config)
    snapshot = _sample_snapshot()
    sym_a = snapshot.symbols[0]
    doc_mentions = [
        {
            "chunk_id": f"doc_{i}",
            "symbol_id": sym_a.symbol_id,
            "symbol_name": "login",
            "confidence": "exact_name",
        }
        for i in range(5)
    ]
    result = _build(transformer, snapshot, doc_mentions=doc_mentions)
    chunks = result.chunks
    for chunk in chunks:
        docs = chunk.get("content", {}).get("supporting_docs", [])
        assert len(docs) <= 2


def test_l2_supporting_docs_no_match():
    """supporting_docs should be absent when doc_mentions reference unknown symbols."""
    transformer = ProjectionTransformer(_config())
    snapshot = _sample_snapshot()
    doc_mentions = [
        {
            "chunk_id": "doc_unrelated",
            "symbol_id": "nonexistent_symbol_id",
            "symbol_name": "phantom",
            "confidence": "exact_name",
        },
    ]
    result = _build(transformer, snapshot, doc_mentions=doc_mentions)
    chunks = result.chunks
    for chunk in chunks:
        assert "supporting_docs" not in chunk.get("content", {})
