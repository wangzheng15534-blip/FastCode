"""Tests for TerminusPublisher code graph payload (symbols + relations)."""

import pytest

from fastcode.terminus_publisher import TerminusPublisher


def _make_publisher() -> TerminusPublisher:
    return TerminusPublisher({"terminus": {"endpoint": "http://localhost:9999"}})


def _make_snapshot_v2(
    snapshot_id: str = "snap:repo:abc",
    units: list[dict] | None = None,
    relations: list[dict] | None = None,
) -> dict:
    """Build a minimal ir.v2 snapshot dict."""
    return {
        "schema_version": "ir.v2",
        "repo_name": "repo",
        "snapshot_id": snapshot_id,
        "branch": "main",
        "commit_id": "abc",
        "units": units or [],
        "relations": relations or [],
    }


def test_build_code_graph_payload_empty_snapshot():
    publisher = _make_publisher()
    result = publisher.build_code_graph_payload(_make_snapshot_v2())
    assert result["nodes"] == []
    assert result["edges"] == []


def test_build_code_graph_payload_skips_file_and_doc_units():
    publisher = _make_publisher()
    units = [
        {
            "unit_id": "f1",
            "kind": "file",
            "path": "a.py",
            "display_name": "a.py",
            "language": "python",
        },
        {
            "unit_id": "d1",
            "kind": "doc",
            "path": "docs.md",
            "display_name": "docs",
            "language": "markdown",
        },
        {
            "unit_id": "fn1",
            "kind": "function",
            "path": "a.py",
            "display_name": "foo",
            "language": "python",
        },
    ]
    result = publisher.build_code_graph_payload(_make_snapshot_v2(units=units))
    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["id"] == "sym:snap:repo:abc:fn1"
    assert result["nodes"][0]["type"] == "Symbol"
    assert result["nodes"][0]["props"]["kind"] == "function"


def test_build_code_graph_payload_symbol_node_props():
    publisher = _make_publisher()
    units = [
        {
            "unit_id": "cls1",
            "kind": "class",
            "path": "b.py",
            "display_name": "MyClass",
            "language": "python",
            "start_line": 10,
            "end_line": 50,
            "qualified_name": "pkg.b.MyClass",
            "primary_anchor_symbol_id": "scip python pkg.b.MyClass",
            "source_set": ["scip", "fc_structure"],
        },
    ]
    result = publisher.build_code_graph_payload(_make_snapshot_v2(units=units))
    node = result["nodes"][0]
    props = node["props"]
    assert props["unit_id"] == "cls1"
    assert props["display_name"] == "MyClass"
    assert props["kind"] == "class"
    assert props["path"] == "b.py"
    assert props["language"] == "python"
    assert props["start_line"] == 10
    assert props["end_line"] == 50
    assert props["qualified_name"] == "pkg.b.MyClass"
    assert props["scip_symbol"] == "scip python pkg.b.MyClass"
    assert set(props["source_set"]) == {"fc_structure", "scip"}


def test_build_code_graph_payload_relation_edges():
    publisher = _make_publisher()
    relations = [
        {
            "relation_id": "r1",
            "src_unit_id": "cls1",
            "dst_unit_id": "fn1",
            "relation_type": "contain",
            "resolution_state": "anchored",
            "support_sources": ["scip"],
        },
        {
            "relation_id": "r2",
            "src_unit_id": "fn1",
            "dst_unit_id": "fn2",
            "relation_type": "call",
            "resolution_state": "structural",
            "support_sources": ["fc_structure"],
        },
    ]
    result = publisher.build_code_graph_payload(_make_snapshot_v2(relations=relations))
    assert len(result["edges"]) == 2

    contain_edge = result["edges"][0]
    assert contain_edge["id"] == "rel:snap:repo:abc:r1"
    assert contain_edge["type"] == "contain"
    assert contain_edge["src"] == "sym:snap:repo:abc:cls1"
    assert contain_edge["dst"] == "sym:snap:repo:abc:fn1"
    assert contain_edge["confidence"] == "precise"  # anchored -> precise
    assert contain_edge["resolution_state"] == "anchored"
    assert contain_edge["source_set"] == ["scip"]

    call_edge = result["edges"][1]
    assert call_edge["confidence"] == "resolved"  # structural -> resolved


def test_build_code_graph_payload_resolution_confidence_mapping():
    publisher = _make_publisher()
    for resolution_state, expected_confidence in [
        ("anchored", "precise"),
        ("structural", "resolved"),
        ("candidate", "heuristic"),
        ("unknown", "derived"),
        ("", "derived"),
    ]:
        relations = [
            {
                "relation_id": f"r_{resolution_state}",
                "src_unit_id": "a",
                "dst_unit_id": "b",
                "relation_type": "ref",
                "resolution_state": resolution_state,
                "support_sources": [],
            },
        ]
        result = publisher.build_code_graph_payload(
            _make_snapshot_v2(relations=relations)
        )
        assert result["edges"][0]["confidence"] == expected_confidence, (
            f"{resolution_state!r} should map to {expected_confidence!r}"
        )


def test_build_code_graph_payload_skips_relations_missing_ids():
    publisher = _make_publisher()
    relations = [
        {
            "relation_id": "",
            "src_unit_id": "a",
            "dst_unit_id": "b",
            "relation_type": "ref",
        },
        {
            "relation_id": "r1",
            "src_unit_id": "",
            "dst_unit_id": "b",
            "relation_type": "ref",
        },
        {
            "relation_id": "r2",
            "src_unit_id": "a",
            "dst_unit_id": None,
            "relation_type": "ref",
        },
    ]
    result = publisher.build_code_graph_payload(_make_snapshot_v2(relations=relations))
    assert result["edges"] == []


def test_build_code_graph_payload_support_sources_as_set():
    """Ensure support_sources serialized as set (from to_dict) are handled."""
    publisher = _make_publisher()
    # When IRRelation.to_dict() is called, support_sources is sorted list,
    # but defensive: handle if a set slips through
    relations = [
        {
            "relation_id": "r1",
            "src_unit_id": "a",
            "dst_unit_id": "b",
            "relation_type": "import",
            "resolution_state": "anchored",
            "support_sources": {"scip", "fc_structure"},  # set, not list
        },
    ]
    result = publisher.build_code_graph_payload(_make_snapshot_v2(relations=relations))
    assert isinstance(result["edges"][0]["source_set"], list)
    assert set(result["edges"][0]["source_set"]) == {"scip", "fc_structure"}


def test_build_lineage_payload_includes_code_graph():
    """Code graph nodes/edges should be merged into the lineage payload."""
    publisher = _make_publisher()
    units = [
        {
            "unit_id": "fn1",
            "kind": "function",
            "path": "a.py",
            "display_name": "foo",
            "language": "python",
        },
    ]
    relations = [
        {
            "relation_id": "r1",
            "src_unit_id": "fn1",
            "dst_unit_id": "fn2",
            "relation_type": "call",
            "resolution_state": "structural",
            "support_sources": [],
        },
    ]
    snapshot = _make_snapshot_v2(units=units, relations=relations)
    payload = publisher.build_lineage_payload(
        snapshot=snapshot,
        manifest={"manifest_id": "m1"},
        git_meta={},
    )
    node_types = {n["type"] for n in payload["nodes"]}
    edge_types = {e["type"] for e in payload["edges"]}

    # Lineage types should still be present
    assert "Repository" in node_types
    assert "Snapshot" in node_types

    # Code graph types should also be present
    assert "Symbol" in node_types
    assert "call" in edge_types

    # Verify the symbol node has correct ID format
    sym_nodes = [n for n in payload["nodes"] if n["type"] == "Symbol"]
    assert len(sym_nodes) == 1
    assert sym_nodes[0]["id"] == "sym:snap:repo:abc:fn1"


def test_build_lineage_payload_legacy_no_units_is_safe():
    """Legacy snapshots without units/relations should not break."""
    publisher = _make_publisher()
    snapshot = {
        "repo_name": "repo",
        "snapshot_id": "snap:repo:old",
        "documents": [{"doc_id": "d1", "path": "a.py", "language": "python"}],
        "symbols": [
            {
                "symbol_id": "s1",
                "display_name": "foo",
                "path": "a.py",
                "kind": "function",
            }
        ],
    }
    payload = publisher.build_lineage_payload(
        snapshot=snapshot,
        manifest={"manifest_id": "m1"},
        git_meta={},
    )
    # Legacy SymbolVersion nodes should be present
    sym_ver_nodes = [n for n in payload["nodes"] if n["type"] == "SymbolVersion"]
    assert len(sym_ver_nodes) == 1
    # No new Symbol nodes (no units)
    assert not [n for n in payload["nodes"] if n["type"] == "Symbol"]


def test_load_graph_nodes_unconfigured_returns_empty():
    publisher = TerminusPublisher({"terminus": {}})
    assert publisher.load_graph_nodes("snap:repo:abc") == []


def test_load_graph_nodes_configured_raises_not_implemented():
    publisher = _make_publisher()
    with pytest.raises(NotImplementedError, match="query endpoint"):
        publisher.load_graph_nodes("snap:repo:abc")


def test_load_graph_edges_unconfigured_returns_empty():
    publisher = TerminusPublisher({"terminus": {}})
    assert publisher.load_graph_edges("snap:repo:abc") == []
    assert publisher.load_graph_edges("snap:repo:abc", edge_type="call") == []


def test_load_graph_edges_configured_raises_not_implemented():
    publisher = _make_publisher()
    with pytest.raises(NotImplementedError, match="query endpoint"):
        publisher.load_graph_edges("snap:repo:abc", edge_type="call")
