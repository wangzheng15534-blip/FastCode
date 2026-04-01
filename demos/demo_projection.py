"""
Demo: Projection Transform -- L0/L1/L2 generation from IR graph.

Usage:
    cd /home/jacob/develop/FastCode
    python -m demos.demo_projection

Shows:
    1. Building a multi-file IR snapshot
    2. Building IR graphs
    3. Generating L0 (summary), L1 (navigation), L2 (chunks) projections
    4. Printing each layer's structure
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastcode.semantic_ir import IRDocument, IREdge, IRSnapshot, IRSymbol
from fastcode.ir_graph_builder import IRGraphBuilder
from fastcode.projection_models import ProjectionScope
from fastcode.projection_transform import ProjectionTransformer


def _build_sample_snapshot() -> IRSnapshot:
    """Build a small multi-file, multi-symbol snapshot."""
    docs = [
        IRDocument(doc_id="doc:auth", path="app/auth.py", language="python", source_set={"ast"}),
        IRDocument(doc_id="doc:user", path="app/user.py", language="python", source_set={"ast"}),
        IRDocument(doc_id="doc:db", path="app/db.py", language="python", source_set={"ast"}),
    ]
    symbols = [
        IRSymbol(symbol_id="sym:auth_svc", external_symbol_id=None, path="app/auth.py",
                 display_name="AuthService", kind="class", language="python",
                 start_line=1, source_priority=10, source_set={"ast"}, metadata={"source": "ast"}),
        IRSymbol(symbol_id="sym:login", external_symbol_id=None, path="app/auth.py",
                 display_name="login", kind="function", language="python",
                 start_line=20, source_priority=10, source_set={"ast"}, metadata={"source": "ast"}),
        IRSymbol(symbol_id="sym:validate_token", external_symbol_id=None, path="app/auth.py",
                 display_name="validate_token", kind="function", language="python",
                 start_line=40, source_priority=10, source_set={"ast"}, metadata={"source": "ast"}),
        IRSymbol(symbol_id="sym:user_model", external_symbol_id=None, path="app/user.py",
                 display_name="UserModel", kind="class", language="python",
                 start_line=1, source_priority=10, source_set={"ast"}, metadata={"source": "ast"}),
        IRSymbol(symbol_id="sym:get_user", external_symbol_id=None, path="app/user.py",
                 display_name="get_user", kind="function", language="python",
                 start_line=10, source_priority=10, source_set={"ast"}, metadata={"source": "ast"}),
        IRSymbol(symbol_id="sym:db_conn", external_symbol_id=None, path="app/db.py",
                 display_name="get_connection", kind="function", language="python",
                 start_line=1, source_priority=10, source_set={"ast"}, metadata={"source": "ast"}),
    ]
    edges = [
        # Contain edges
        IREdge(edge_id="e:c1", src_id="doc:auth", dst_id="sym:auth_svc", edge_type="contain", source="ast", confidence="resolved"),
        IREdge(edge_id="e:c2", src_id="doc:auth", dst_id="sym:login", edge_type="contain", source="ast", confidence="resolved"),
        IREdge(edge_id="e:c3", src_id="doc:auth", dst_id="sym:validate_token", edge_type="contain", source="ast", confidence="resolved"),
        IREdge(edge_id="e:c4", src_id="doc:user", dst_id="sym:user_model", edge_type="contain", source="ast", confidence="resolved"),
        IREdge(edge_id="e:c5", src_id="doc:user", dst_id="sym:get_user", edge_type="contain", source="ast", confidence="resolved"),
        IREdge(edge_id="e:c6", src_id="doc:db", dst_id="sym:db_conn", edge_type="contain", source="ast", confidence="resolved"),
        # Call edges
        IREdge(edge_id="e:call1", src_id="sym:login", dst_id="sym:validate_token", edge_type="call", source="ast", confidence="heuristic"),
        IREdge(edge_id="e:call2", src_id="sym:login", dst_id="sym:get_user", edge_type="call", source="ast", confidence="heuristic"),
        IREdge(edge_id="e:call3", src_id="sym:get_user", dst_id="sym:db_conn", edge_type="call", source="ast", confidence="heuristic"),
        # Import edges
        IREdge(edge_id="e:imp1", src_id="doc:auth", dst_id="doc:user", edge_type="import", source="ast", confidence="heuristic"),
        IREdge(edge_id="e:imp2", src_id="doc:user", dst_id="doc:db", edge_type="import", source="ast", confidence="heuristic"),
    ]
    return IRSnapshot(
        repo_name="demo", snapshot_id="snap:demo:proj",
        documents=docs, symbols=symbols, edges=edges,
        metadata={"source_modes": ["ast"]},
    )


def main():
    snapshot = _build_sample_snapshot()
    print(f"Snapshot: {len(snapshot.documents)} docs, {len(snapshot.symbols)} symbols, {len(snapshot.edges)} edges")

    graphs = IRGraphBuilder().build_graphs(snapshot)
    print(f"Graphs: dep={graphs.dependency_graph.number_of_edges()}, "
          f"call={graphs.call_graph.number_of_edges()}, "
          f"contain={graphs.containment_graph.number_of_edges()}")

    config = {"projection": {"enable_leiden": True, "llm_enabled": False}}
    transformer = ProjectionTransformer(config=config)
    scope = ProjectionScope(scope_kind="snapshot", snapshot_id="snap:demo:proj", scope_key="demo_key")

    result = transformer.build(scope=scope, snapshot=snapshot, ir_graphs=graphs)

    print(f"\n=== L0 (Summary) ===")
    print(json.dumps(result.l0, indent=2, ensure_ascii=False)[:500])

    print(f"\n=== L1 (Navigation) ===")
    l1_content = result.l1.get("content", {})
    print(f"  relations keys: {list(l1_content.get('relations', {}).keys())}")
    print(f"  relations_v2 keys: {list(l1_content.get('relations_v2', {}).keys())}")
    v2_xrefs = l1_content.get("relations_v2", {}).get("xref", [])
    print(f"  relations_v2.xref count: {len(v2_xrefs)}")
    if v2_xrefs:
        print(f"  first xref: {v2_xrefs[0]}")
    print(f"  related_code: {len(l1_content.get('related_code', []))} refs")

    print(f"\n=== L2 Index ===")
    print(json.dumps(result.l2_index, indent=2, ensure_ascii=False)[:500])

    print(f"\n=== Chunks: {len(result.chunks)} ===")
    for chunk in result.chunks:
        print(f"  {chunk['chunk_id']}: {chunk['kind']} | version={chunk.get('version')} "
              f"layer={chunk.get('layer')} title={chunk.get('title', '?')}")
        if chunk.get('meta'):
            print(f"    meta: {chunk['meta']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
