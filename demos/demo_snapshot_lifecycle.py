"""
Demo: Snapshot Store Lifecycle -- save, load, manifest publish, ref resolution.

Usage:
    cd /home/jacob/develop/FastCode
    python -m demos.demo_snapshot_lifecycle

Shows:
    1. Creating and saving an IRSnapshot
    2. Loading it back
    3. Publishing manifests for two snapshots on the same branch
    4. Verifying branch head tracks the latest
    5. Manifest previous-chain (supersession)
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastcode.semantic_ir import IRDocument, IREdge, IRSymbol, IRSnapshot
from fastcode.snapshot_store import SnapshotStore
from fastcode.manifest_store import ManifestStore
from fastcode.index_run import IndexRunStore
from fastcode.ir_graph_builder import IRGraphBuilder


def _make_snapshot(repo: str, snap_id: str, commit: str, branch: str) -> IRSnapshot:
    doc = IRDocument(doc_id="doc:1", path="main.py", language="python", source_set={"ast"})
    sym = IRSymbol(
        symbol_id=f"sym:{snap_id}:main", external_symbol_id=None, path="main.py",
        display_name="main", kind="function", language="python",
        start_line=1, source_priority=10, source_set={"ast"}, metadata={"source": "ast"},
    )
    edge = IREdge(
        edge_id=f"e:contain:{snap_id}", src_id="doc:1", dst_id=sym.symbol_id,
        edge_type="contain", source="ast", confidence="resolved",
    )
    return IRSnapshot(
        repo_name=repo, snapshot_id=snap_id, branch=branch, commit_id=commit,
        documents=[doc], symbols=[sym], edges=[edge],
        metadata={"source_modes": ["ast"]},
    )


def main():
    with tempfile.TemporaryDirectory(prefix="fc_demo_snap_") as tmp:
        store = SnapshotStore(tmp)
        manifest_store = ManifestStore(store.db_path)
        run_store = IndexRunStore(store.db_path)

        # 1. Save first snapshot
        snap1 = _make_snapshot("my-repo", "snap:my-repo:aaa", "aaa", "main")
        meta1 = store.save_snapshot(snap1, metadata={"run": 1})
        graphs1 = IRGraphBuilder().build_graphs(snap1)
        store.save_ir_graphs(snap1.snapshot_id, graphs1)
        print(f"Saved snapshot 1: {snap1.snapshot_id} -> {meta1['artifact_key']}")

        # 2. Load it back
        loaded1 = store.load_snapshot("snap:my-repo:aaa")
        assert loaded1 is not None
        print(f"Loaded snapshot 1: {loaded1.snapshot_id}, {len(loaded1.symbols)} symbols")

        # 3. Publish manifest for snapshot 1
        run1 = run_store.create_run("my-repo", "snap:my-repo:aaa", "main", "aaa")
        m1 = manifest_store.publish("my-repo", "main", "snap:my-repo:aaa", run1)
        print(f"Published manifest 1: {m1['manifest_id']}")

        # 4. Save and publish second snapshot (simulates new commit on main)
        snap2 = _make_snapshot("my-repo", "snap:my-repo:bbb", "bbb", "main")
        store.save_snapshot(snap2, metadata={"run": 2})
        run2 = run_store.create_run("my-repo", "snap:my-repo:bbb", "main", "bbb")
        m2 = manifest_store.publish("my-repo", "main", "snap:my-repo:bbb", run2)
        print(f"Published manifest 2: {m2['manifest_id']}")

        # 5. Verify branch head
        head = manifest_store.get_branch_manifest("my-repo", "main")
        assert head is not None
        assert head["snapshot_id"] == "snap:my-repo:bbb"
        assert head["previous_manifest_id"] == m1["manifest_id"]
        print(f"Branch head: {head['snapshot_id']} (previous: {head['previous_manifest_id']})")

        # 6. Test idempotent run creation
        run3 = run_store.create_run("my-repo", "snap:my-repo:bbb", "main", "bbb", idempotency_key="key1")
        run4 = run_store.create_run("my-repo", "snap:my-repo:bbb", "main", "bbb", idempotency_key="key1")
        assert run3 == run4
        print(f"Idempotent run: {run3} == {run4}")

        # 7. Load IR graphs
        loaded_graphs = store.load_ir_graphs("snap:my-repo:aaa")
        assert loaded_graphs is not None
        print(f"Loaded IR graphs: containment has {loaded_graphs.containment_graph.number_of_edges()} edges")

        print("\nDone.")


if __name__ == "__main__":
    main()
