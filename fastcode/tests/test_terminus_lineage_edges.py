from fastcode.terminus_publisher import TerminusPublisher


def test_terminus_payload_includes_commit_parent_edges():
    publisher = TerminusPublisher({"terminus": {"endpoint": "http://localhost"}})
    payload = publisher.build_lineage_payload(
        snapshot={
            "repo_name": "repo",
            "snapshot_id": "snap:repo:c1",
            "branch": "main",
            "commit_id": "c1",
            "documents": [],
            "symbols": [],
        },
        manifest={"manifest_id": "m1"},
        git_meta={"parent_commit_ids": ["p1", "p2"]},
    )
    commit_parent_edges = [e for e in payload["edges"] if e["type"] == "commit_parent"]
    assert len(commit_parent_edges) == 2


def test_terminus_payload_includes_symbol_version_from_edges():
    publisher = TerminusPublisher({"terminus": {"endpoint": "http://localhost"}})
    payload = publisher.build_lineage_payload(
        snapshot={
            "repo_name": "repo",
            "snapshot_id": "snap:repo:c2",
            "branch": "main",
            "commit_id": "c2",
            "documents": [],
            "symbols": [
                {
                    "symbol_id": "scip:snap:repo:c2:ext:sym:1",
                    "external_symbol_id": "ext:sym:1",
                    "display_name": "foo",
                    "kind": "function",
                    "path": "a.py",
                }
            ],
        },
        manifest={"manifest_id": "m2"},
        git_meta={},
        previous_snapshot_symbols={"ext:sym:1": "symbol:snap:repo:c1:scip:snap:repo:c1:ext:sym:1"},
    )
    version_edges = [e for e in payload["edges"] if e["type"] == "symbol_version_from"]
    assert len(version_edges) == 1
