"""
Demo: PostgreSQL Hardening Features -- fencing tokens, redo worker, lineage edges.

Usage:
    cd /home/jacob/develop/FastCode
    python -m demos.demo_hardening

Shows:
    1. Fencing token acquisition and validation
    2. Redo worker lifecycle (start/stop/process)
    3. TerminusDB lineage edges (commit_parent, symbol_version_from)
    4. Graph API (callees, callers, dependencies via NetworkX)
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from fastcode.snapshot_store import SnapshotStore
from fastcode.terminus_publisher import TerminusPublisher
from fastcode.semantic_ir import IREdge, IRSnapshot, IRSymbol
from fastcode.ir_graph_builder import IRGraphBuilder
from fastcode.redo_worker import RedoWorker
from unittest.mock import MagicMock


def main():
    print("=" * 60)
    print("FastCode Hardening Features Demo")
    print("=" * 60)

    # --- 1. Fencing Tokens ---
    print("\n--- 1. Fencing Tokens ---")
    with tempfile.TemporaryDirectory(prefix="fc_hardening_") as tmp:
        store = SnapshotStore(tmp)
        token1 = store.acquire_lock("index:snap:repo:1", owner_id="run1", ttl_seconds=60)
        print(f"  Acquired lock: token={token1}")
        print(f"  Validate token={token1}: {store.validate_fencing_token('index:snap:repo:1', token1)}")
        print(f"  Validate stale token=999: {store.validate_fencing_token('index:snap:repo:1', 999)}")

        token2 = store.acquire_lock("index:snap:repo:1", owner_id="run2", ttl_seconds=60)
        print(f"  Re-acquired by run2: token={token2} (SQLite always returns 1)")

    # --- 2. Redo Worker ---
    print("\n--- 2. Redo Worker ---")
    fake_fc = MagicMock()
    fake_fc.snapshot_store.claim_redo_task.return_value = None
    worker = RedoWorker(fake_fc, poll_interval_seconds=1)
    print(f"  Created worker: poll_interval={worker.poll_interval_seconds}s")
    worker.start()
    print(f"  Thread started: alive={worker._thread.is_alive()}, daemon={worker._thread.daemon}")
    result = worker.process_once_status()
    print(f"  process_once_status (no tasks): {result}")
    worker.stop()
    print(f"  Stopped: event set={worker._stop_event.is_set()}")

    # --- 3. Lineage Edges ---
    print("\n--- 3. TerminusDB Lineage Edges ---")
    publisher = TerminusPublisher({"terminus": {"endpoint": "http://localhost"}})
    payload = publisher.build_lineage_payload(
        snapshot={
            "repo_name": "my-repo",
            "snapshot_id": "snap:my-repo:c2",
            "branch": "main",
            "commit_id": "c2",
            "documents": [],
            "symbols": [
                {
                    "symbol_id": "sym:ext:1",
                    "external_symbol_id": "ext:sym:1",
                    "display_name": "authenticate",
                    "kind": "function",
                    "path": "auth.py",
                }
            ],
        },
        manifest={"manifest_id": "m2"},
        git_meta={"parent_commit_ids": ["c1"]},
        previous_snapshot_symbols={"ext:sym:1": "symbol:snap:my-repo:c1:sym:ext:1"},
    )
    edge_types = [e["type"] for e in payload["edges"]]
    print(f"  Edge types: {edge_types}")
    commit_parent_edges = [e for e in payload["edges"] if e["type"] == "commit_parent"]
    version_edges = [e for e in payload["edges"] if e["type"] == "symbol_version_from"]
    print(f"  commit_parent edges: {len(commit_parent_edges)}")
    print(f"  symbol_version_from edges: {len(version_edges)}")
    if version_edges:
        print(f"    {version_edges[0]['src']} -> {version_edges[0]['dst']}")

    # --- 4. Graph API ---
    print("\n--- 4. Graph API (NetworkX Traversal) ---")
    sym_a = IRSymbol(
        symbol_id="sym:auth", external_symbol_id=None, path="auth.py",
        display_name="authenticate", kind="function", language="python",
        start_line=1, source_priority=10, source_set={"ast"}, metadata={"source": "ast"},
    )
    sym_b = IRSymbol(
        symbol_id="sym:validate", external_symbol_id=None, path="auth.py",
        display_name="validate_token", kind="function", language="python",
        start_line=20, source_priority=10, source_set={"ast"}, metadata={"source": "ast"},
    )
    sym_c = IRSymbol(
        symbol_id="sym:db", external_symbol_id=None, path="db.py",
        display_name="get_connection", kind="function", language="python",
        start_line=1, source_priority=10, source_set={"ast"}, metadata={"source": "ast"},
    )
    snapshot = IRSnapshot(
        repo_name="demo", snapshot_id="snap:demo:graph",
        documents=[], symbols=[sym_a, sym_b, sym_c],
        edges=[
            IREdge(edge_id="e:1", src_id="sym:auth", dst_id="sym:validate",
                   edge_type="call", source="ast", confidence="heuristic"),
            IREdge(edge_id="e:2", src_id="sym:validate", dst_id="sym:db",
                   edge_type="call", source="ast", confidence="heuristic"),
        ],
    )
    graphs = IRGraphBuilder().build_graphs(snapshot)
    print(f"  Call graph: {graphs.call_graph.number_of_nodes()} nodes, {graphs.call_graph.number_of_edges()} edges")

    # Callees from sym:auth
    dist = nx.single_source_shortest_path_length(graphs.call_graph, "sym:auth", cutoff=2)
    callees = [{"symbol_id": n, "distance": d} for n, d in dist.items() if n != "sym:auth"]
    print(f"  Callees from authenticate (2 hops):")
    for c in callees:
        print(f"    -> {c['symbol_id']} (distance {c['distance']})")

    # Callers of sym:db
    rev = graphs.call_graph.reverse(copy=False)
    dist_rev = nx.single_source_shortest_path_length(rev, "sym:db", cutoff=2)
    callers = [{"symbol_id": n, "distance": d} for n, d in dist_rev.items() if n != "sym:db"]
    print(f"  Callers of get_connection (2 hops):")
    for c in callers:
        print(f"    <- {c['symbol_id']} (distance {c['distance']})")

    print("\nDone.")


if __name__ == "__main__":
    main()
