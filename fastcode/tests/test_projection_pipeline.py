import networkx as nx

from fastcode.adapters.scip_to_ir import build_ir_from_scip
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
    edges = [
        IREdge(
            edge_id="edge:contain:1",
            src_id=doc.doc_id,
            dst_id=sym_a.symbol_id,
            edge_type="contain",
            source="ast",
            confidence="resolved",
            doc_id=doc.doc_id,
            metadata={},
        ),
        IREdge(
            edge_id="edge:contain:2",
            src_id=doc.doc_id,
            dst_id=sym_b.symbol_id,
            edge_type="contain",
            source="ast",
            confidence="resolved",
            doc_id=doc.doc_id,
            metadata={},
        ),
        IREdge(
            edge_id="edge:call:1",
            src_id=sym_a.symbol_id,
            dst_id=sym_b.symbol_id,
            edge_type="call",
            source="ast",
            confidence="heuristic",
            doc_id=doc.doc_id,
            metadata={},
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
        metadata={"source_modes": ["ast"]},
    )


def _sample_graphs() -> IRGraphs:
    dep = nx.DiGraph()
    call = nx.DiGraph()
    inherit = nx.DiGraph()
    ref = nx.DiGraph()
    contain = nx.DiGraph()
    return IRGraphs(
        dependency_graph=dep,
        call_graph=call,
        inheritance_graph=inherit,
        reference_graph=ref,
        containment_graph=contain,
    )


def test_projection_transform_emits_required_layers_and_meta():
    snapshot = _sample_snapshot()
    transformer = ProjectionTransformer(config={"projection": {"enable_leiden": False}})
    scope = ProjectionScope(
        scope_kind="snapshot", snapshot_id=snapshot.snapshot_id, scope_key="k1"
    )
    result = transformer.build(
        scope=scope, snapshot=snapshot, ir_graphs=_sample_graphs()
    )

    assert result.l0["layer"] == "L0"
    assert result.l1["layer"] == "L1"
    assert result.l2_index["layer"] == "L2"
    assert "covers_nodes" in result.l0["meta"]
    assert "covers_edges" in result.l1["meta"]
    assert "projection_method" in result.l2_index["meta"]
    assert result.l1["meta"]["algo_version"] == transformer.ALGO_VERSION
    assert result.chunks


def test_scip_adapter_uses_snapshot_prefixed_symbol_ids_and_ref_edges():
    scip = {
        "indexer_name": "scip-python",
        "indexer_version": "0.1.0",
        "documents": [
            {
                "path": "app/service.py",
                "language": "python",
                "symbols": [
                    {
                        "symbol": "pkg app/service.py login().",
                        "name": "login",
                        "kind": "function",
                    }
                ],
                "occurrences": [
                    {
                        "symbol": "pkg app/service.py login().",
                        "role": "reference",
                        "range": [10, 0, 10, 5],
                    },
                    {
                        "symbol": "pkg app/service.py login().",
                        "role": "type_definition",
                        "range": [12, 0, 12, 5],
                    },
                ],
            }
        ],
    }
    snap = build_ir_from_scip(
        repo_name="repo", snapshot_id="snap:repo:abc", scip_index=scip
    )
    assert snap.symbols
    assert snap.symbols[0].symbol_id.startswith("scip:snap:repo:abc:")
    assert any(e.edge_type == "contain" and e.source == "scip" for e in snap.edges)
    assert any(e.edge_type == "ref" and e.source == "scip" for e in snap.edges)
    assert any(o.role == "type_definition" for o in snap.occurrences)


def test_steiner_prune_removes_non_terminal_leaves():
    transformer = ProjectionTransformer(
        config={"projection": {"enable_leiden": False, "steiner_prune": True}}
    )
    tree = nx.Graph()
    tree.add_edge("t1", "mid")
    tree.add_edge("mid", "t2")
    tree.add_edge("mid", "leaf")
    pruned = transformer._prune_steiner_leaves(tree, {"t1", "t2"})
    assert "leaf" not in pruned.nodes
    assert {"t1", "mid", "t2"} == set(pruned.nodes)
