from fastcode.terminus_publisher import TerminusPublisher


def test_terminus_payload_contains_expected_node_and_edge_types():
    publisher = TerminusPublisher({"terminus": {"endpoint": "http://localhost:9999"}})
    snapshot = {
        "repo_name": "repo",
        "snapshot_id": "snap:repo:abc",
        "branch": "main",
        "commit_id": "abc",
        "documents": [{"doc_id": "d1", "path": "a.py", "language": "python"}],
        "symbols": [{"symbol_id": "sym:1", "display_name": "foo", "path": "a.py", "kind": "function"}],
    }
    manifest = {
        "manifest_id": "m1",
        "index_run_id": "run1",
        "ref_name": "main",
        "status": "published",
        "published_at": "2026-01-01T00:00:00Z",
        "previous_manifest_id": "m0",
    }
    payload = publisher.build_lineage_payload(snapshot=snapshot, manifest=manifest, git_meta={})
    node_types = {n["type"] for n in payload["nodes"]}
    edge_types = {e["type"] for e in payload["edges"]}
    assert "Repository" in node_types
    assert "Snapshot" in node_types
    assert "Manifest" in node_types
    assert "DocumentVersion" in node_types
    assert "SymbolVersion" in node_types
    assert "snapshot_manifest" in edge_types
    assert "manifest_supersedes" in edge_types
