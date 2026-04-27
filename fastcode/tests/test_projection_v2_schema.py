from __future__ import annotations

from typing import Any

import networkx as nx

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
        metadata={"source": "ast"},
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
        metadata={"source": "ast"},
    )
    return IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:repo:abc",
        branch="main",
        commit_id="abc",
        documents=[doc],
        symbols=[sym_a, sym_b],
        edges=[
            IREdge(
                edge_id="edge:contain:1",
                src_id=doc.doc_id,
                dst_id=sym_a.symbol_id,
                edge_type="contain",
                source="ast",
                confidence="resolved",
            ),
            IREdge(
                edge_id="edge:call:1",
                src_id=sym_a.symbol_id,
                dst_id=sym_b.symbol_id,
                edge_type="call",
                source="ast",
                confidence="heuristic",
            ),
        ],
    )


def _sample_graphs() -> Any:
    from fastcode.ir_graph_builder import IRGraphs

    return IRGraphs(
        dependency_graph=nx.DiGraph(),
        call_graph=nx.DiGraph(),
        inheritance_graph=nx.DiGraph(),
        reference_graph=nx.DiGraph(),
        containment_graph=nx.DiGraph(),
    )


def test_projection_schema_has_hierarchy_and_relations():
    snapshot = _sample_snapshot()
    transformer = ProjectionTransformer(config={"projection": {"enable_leiden": False}})
    scope = ProjectionScope(
        scope_kind="snapshot", snapshot_id=snapshot.snapshot_id, scope_key="k2"
    )
    result = transformer.build(
        scope=scope, snapshot=snapshot, ir_graphs=_sample_graphs()
    )

    l1_content = result.l1["content"]
    assert "relations" in l1_content
    assert "xref" in l1_content["relations"]
    assert "hierarchy" in l1_content["relations"]
    assert "backbone" in l1_content["relations"]
    assert "related_code" in l1_content
    assert "related_memory" in l1_content

    chunk = result.chunks[0]
    assert "chunk_id" in chunk
    assert "content" in chunk
    assert chunk["version"] == "v1"
    assert chunk["layer"] == "L2"
    assert "source" in chunk
    assert "render" in chunk
    assert "meta" in chunk
    assert "aggregation" in chunk["meta"]


def test_projection_l2_chunk_has_all_required_metadata():
    snapshot = _sample_snapshot()
    transformer = ProjectionTransformer(config={"projection": {"enable_leiden": False}})
    scope = ProjectionScope(
        scope_kind="snapshot", snapshot_id=snapshot.snapshot_id, scope_key="k3"
    )
    result = transformer.build(
        scope=scope, snapshot=snapshot, ir_graphs=_sample_graphs()
    )

    for chunk in result.chunks:
        assert chunk["version"] == "v1", f"chunk {chunk['chunk_id']} missing version"
        assert chunk["layer"] == "L2", f"chunk {chunk['chunk_id']} missing layer"
        assert "id" in chunk, f"chunk {chunk['chunk_id']} missing id"
        assert "path" in chunk, f"chunk {chunk['chunk_id']} missing path"
        assert "title" in chunk, f"chunk {chunk['chunk_id']} missing title"
        assert "source" in chunk, f"chunk {chunk['chunk_id']} missing source"
        assert "render" in chunk, f"chunk {chunk['chunk_id']} missing render"
        assert "meta" in chunk, f"chunk {chunk['chunk_id']} missing meta"


def test_projection_l1_xref_relations_have_confidence():
    snapshot = _sample_snapshot()
    transformer = ProjectionTransformer(config={"projection": {"enable_leiden": False}})
    scope = ProjectionScope(
        scope_kind="snapshot", snapshot_id=snapshot.snapshot_id, scope_key="k4"
    )
    result = transformer.build(
        scope=scope, snapshot=snapshot, ir_graphs=_sample_graphs()
    )

    l1_content = result.l1["content"]
    xrefs = l1_content["relations"]["xref"]
    for rel in xrefs:
        assert "id" in rel
        assert "type" in rel
        assert 0.0 <= rel["confidence"] <= 1.0
