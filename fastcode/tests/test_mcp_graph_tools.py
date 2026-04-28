"""Tests for mcp_graph_tools module.

Tests the pure compute functions from fastcode.mcp_graph_tools, which are
the production code extracted from mcp_server.py.
"""

from fastcode.ir_graph_builder import IRGraphBuilder
from fastcode.mcp_graph_tools import (
    compute_directed_path,
    compute_find_callers,
    compute_impact_analysis,
    compute_steiner_path,
    extract_cluster_data,
    format_path_node,
    resolve_unit_id,
)
from fastcode.semantic_ir import IRCodeUnit, IRRelation, IRSnapshot

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _unit(
    unit_id: str,
    name: str = "foo",
    kind: str = "function",
    path: str = "a.py",
    qualified_name: str | None = None,
    start_line: int = 10,
) -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=unit_id,
        kind=kind,
        path=path,
        language="python",
        display_name=name,
        qualified_name=qualified_name,
        start_line=start_line,
        end_line=start_line + 10,
        source_set={"scip"},
    )


def _rel(
    src: str,
    dst: str,
    rtype: str = "call",
) -> IRRelation:
    return IRRelation(
        relation_id=f"rel:{src}:{dst}",
        src_unit_id=src,
        dst_unit_id=dst,
        relation_type=rtype,
        resolution_state="anchored",
        support_sources={"scip"},
    )


def _snapshot(units: list[IRCodeUnit], relations: list[IRRelation]) -> IRSnapshot:
    return IRSnapshot(
        repo_name="test",
        snapshot_id="snap:test:abc123",
        units=units,
        relations=relations,
    )


# ===========================================================================
# impact_analysis tests
# ===========================================================================


class TestImpactAnalysis:
    def test_direct_caller(self):
        """A calls B; impact of B should find A at distance 1."""
        units = [
            _unit("u:A", name="caller"),
            _unit("u:B", name="callee"),
        ]
        rels = [_rel("u:A", "u:B", "call")]
        snap = _snapshot(units, rels)
        result = compute_impact_analysis("callee", snap)
        assert result["error"] is None
        assert result["total_count"] == 1
        assert result["affected"][0]["display_name"] == "caller"
        assert result["affected"][0]["distance"] == 1

    def test_transitive_callers(self):
        """A->B->C; impact of C should find B at d=1 and A at d=2."""
        units = [
            _unit("u:A", name="top"),
            _unit("u:B", name="mid"),
            _unit("u:C", name="bottom"),
        ]
        rels = [
            _rel("u:A", "u:B", "call"),
            _rel("u:B", "u:C", "call"),
        ]
        snap = _snapshot(units, rels)
        result = compute_impact_analysis("bottom", snap)
        assert result["error"] is None
        assert result["total_count"] == 2
        names = [a["display_name"] for a in result["affected"]]
        assert "mid" in names
        assert "top" in names
        # Sorted by distance
        assert result["affected"][0]["distance"] == 1
        assert result["affected"][1]["distance"] == 2

    def test_max_hops_limit(self):
        """A->B->C->D; impact of D with max_hops=1 should only find C."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
            _unit("u:C", name="c"),
            _unit("u:D", name="d"),
        ]
        rels = [
            _rel("u:A", "u:B", "call"),
            _rel("u:B", "u:C", "call"),
            _rel("u:C", "u:D", "call"),
        ]
        snap = _snapshot(units, rels)
        result = compute_impact_analysis("d", snap, max_hops=1)
        assert result["total_count"] == 1
        assert result["affected"][0]["display_name"] == "c"

    def test_dependency_edges(self):
        """A imports B; impact of B with dependency graph should find A."""
        units = [
            _unit("u:A", name="mod_a"),
            _unit("u:B", name="mod_b"),
        ]
        rels = [_rel("u:A", "u:B", "import")]
        snap = _snapshot(units, rels)
        result = compute_impact_analysis("mod_b", snap, graph_types=["dependency"])
        assert result["total_count"] == 1
        assert result["affected"][0]["display_name"] == "mod_a"

    def test_symbol_not_found(self):
        snap = _snapshot([_unit("u:A", name="foo")], [])
        result = compute_impact_analysis("nonexistent", snap)
        assert result["total_count"] == 0
        assert "Symbol not found" in result["error"]

    def test_symbol_not_in_graph(self):
        """Symbol exists but has no edges."""
        units = [
            _unit("u:A", name="connected"),
            _unit("u:B", name="isolated"),
        ]
        rels = [_rel("u:A", "u:A", "call")]  # self-loop just to get A in graph
        snap = _snapshot(units, rels)
        result = compute_impact_analysis("isolated", snap)
        assert "not in graph" in result["error"]

    def test_leaf_symbol_no_callers(self):
        """Leaf symbol (nothing depends on it) has zero impact."""
        units = [
            _unit("u:A", name="leaf"),
            _unit("u:B", name="caller"),
        ]
        rels = [_rel("u:B", "u:A", "call")]  # B calls A; nothing calls B
        snap = _snapshot(units, rels)
        result = compute_impact_analysis("caller", snap)
        assert result["total_count"] == 0
        assert result["error"] is None

    def test_edge_types_recorded(self):
        """Edge types are correctly recorded for affected nodes."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
            _unit("u:C", name="c"),
        ]
        rels = [
            _rel("u:A", "u:B", "call"),
            _rel("u:A", "u:C", "import"),
        ]
        snap = _snapshot(units, rels)
        result = compute_impact_analysis("b", snap)
        assert result["total_count"] == 1
        assert "call" in result["affected"][0]["edge_types"]

    def test_invalid_graph_type(self):
        snap = _snapshot([_unit("u:A", name="foo")], [])
        result = compute_impact_analysis("foo", snap, graph_types=["invalid"])
        assert "Invalid graph types" in result["error"]

    def test_multi_edge_types(self):
        """A calls B and A imports B; impact of B records both edge types."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
        ]
        rels = [
            _rel("u:A", "u:B", "call"),
            _rel("u:A", "u:B", "import"),
        ]
        snap = _snapshot(units, rels)
        result = compute_impact_analysis("b", snap, graph_types=["call", "dependency"])
        assert result["total_count"] == 1
        assert sorted(result["affected"][0]["edge_types"]) == ["call", "dependency"]


# ===========================================================================
# find_callers tests
# ===========================================================================


class TestFindCallers:
    def test_direct_caller(self):
        """A calls B; find_callers(B) should find A."""
        units = [
            _unit("u:A", name="caller"),
            _unit("u:B", name="callee"),
        ]
        rels = [_rel("u:A", "u:B", "call")]
        snap = _snapshot(units, rels)
        result = compute_find_callers("callee", snap)
        assert result["error"] is None
        assert result["total_count"] == 1
        assert result["callers"][0]["display_name"] == "caller"
        assert result["callers"][0]["distance"] == 1

    def test_no_callers(self):
        """B has no callers."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
        ]
        rels = [_rel("u:A", "u:B", "import")]  # import, not call
        snap = _snapshot(units, rels)
        result = compute_find_callers("b", snap)
        assert result["total_count"] == 0

    def test_transitive_callers(self):
        """A->B->C; find_callers(C) finds B at d=1, A at d=2."""
        units = [
            _unit("u:A", name="top"),
            _unit("u:B", name="mid"),
            _unit("u:C", name="bot"),
        ]
        rels = [
            _rel("u:A", "u:B", "call"),
            _rel("u:B", "u:C", "call"),
        ]
        snap = _snapshot(units, rels)
        result = compute_find_callers("bot", snap)
        assert result["total_count"] == 2
        assert result["callers"][0]["distance"] == 1
        assert result["callers"][1]["distance"] == 2

    def test_max_hops(self):
        """A->B->C; find_callers(C) with max_hops=1 only finds B."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
            _unit("u:C", name="c"),
        ]
        rels = [
            _rel("u:A", "u:B", "call"),
            _rel("u:B", "u:C", "call"),
        ]
        snap = _snapshot(units, rels)
        result = compute_find_callers("c", snap, max_hops=1)
        assert result["total_count"] == 1
        assert result["callers"][0]["display_name"] == "b"

    def test_symbol_not_found(self):
        snap = _snapshot([_unit("u:A", name="foo")], [])
        result = compute_find_callers("nonexistent", snap)
        assert "Symbol not found" in result["error"]

    def test_symbol_not_in_call_graph(self):
        """Symbol exists but not in call graph."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
            _unit("u:C", name="c"),
        ]
        # C has a call edge (making call_graph non-empty), but B does not
        rels = [
            _rel("u:A", "u:C", "call"),
            _rel("u:A", "u:B", "import"),
        ]
        snap = _snapshot(units, rels)
        result = compute_find_callers("b", snap)
        assert "not in call graph" in result["error"]

    def test_multiple_callers(self):
        """Both A and B call C."""
        units = [
            _unit("u:A", name="caller1"),
            _unit("u:B", name="caller2"),
            _unit("u:C", name="target"),
        ]
        rels = [
            _rel("u:A", "u:C", "call"),
            _rel("u:B", "u:C", "call"),
        ]
        snap = _snapshot(units, rels)
        result = compute_find_callers("target", snap)
        assert result["total_count"] == 2
        names = {c["display_name"] for c in result["callers"]}
        assert names == {"caller1", "caller2"}

    def test_caller_metadata(self):
        """Caller entries include full metadata."""
        units = [
            _unit("u:A", name="caller", path="main.py", kind="function", start_line=42),
            _unit("u:B", name="callee", path="util.py", kind="function", start_line=10),
        ]
        rels = [_rel("u:A", "u:B", "call")]
        snap = _snapshot(units, rels)
        result = compute_find_callers("callee", snap)
        assert result["total_count"] == 1
        c = result["callers"][0]
        assert c["symbol_id"] == "u:A"
        assert c["path"] == "main.py"
        assert c["start_line"] == 42

    def test_ignores_non_call_edges(self):
        """Dependency edges are ignored in find_callers."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
            _unit("u:C", name="c"),
        ]
        rels = [
            _rel("u:A", "u:C", "call"),
            _rel("u:B", "u:C", "import"),  # not a call
        ]
        snap = _snapshot(units, rels)
        result = compute_find_callers("c", snap)
        assert result["total_count"] == 1
        assert result["callers"][0]["display_name"] == "a"


# ===========================================================================
# steiner_path tests
# ===========================================================================


class TestSteinerPath:
    def test_two_terminals_directly_connected(self):
        """A->B; steiner_path([A, B]) returns both nodes and the edge."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
        ]
        rels = [_rel("u:A", "u:B", "call")]
        snap = _snapshot(units, rels)
        result = compute_steiner_path(["a", "b"], snap)
        assert result["found"] is True
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        assert result["error"] is None

    def test_three_terminals_with_intermediate(self):
        """A->B->C, A->C; steiner_path([A, C]) returns direct edge."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
            _unit("u:C", name="c"),
        ]
        rels = [
            _rel("u:A", "u:B", "call"),
            _rel("u:B", "u:C", "call"),
            _rel("u:A", "u:C", "call"),
        ]
        snap = _snapshot(units, rels)
        result = compute_steiner_path(["a", "c"], snap)
        assert result["found"] is True
        # Steiner should find the direct A-C connection
        node_ids = {n["symbol_id"] for n in result["nodes"]}
        assert "u:A" in node_ids
        assert "u:C" in node_ids

    def test_three_terminals_all_needed(self):
        """A->B->C; steiner_path([A, C]) needs B as intermediate."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
            _unit("u:C", name="c"),
        ]
        rels = [
            _rel("u:A", "u:B", "call"),
            _rel("u:B", "u:C", "call"),
        ]
        snap = _snapshot(units, rels)
        result = compute_steiner_path(["a", "c"], snap)
        assert result["found"] is True
        node_ids = {n["symbol_id"] for n in result["nodes"]}
        assert "u:A" in node_ids
        assert "u:B" in node_ids
        assert "u:C" in node_ids

    def test_prune_non_terminal_leaves(self):
        """A->B->C, D->B; steiner_path([A, C]) should prune D."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
            _unit("u:C", name="c"),
            _unit("u:D", name="d"),
        ]
        rels = [
            _rel("u:A", "u:B", "call"),
            _rel("u:B", "u:C", "call"),
            _rel("u:D", "u:B", "call"),
        ]
        snap = _snapshot(units, rels)
        result = compute_steiner_path(["a", "c"], snap)
        assert result["found"] is True
        node_ids = {n["symbol_id"] for n in result["nodes"]}
        assert "u:D" not in node_ids

    def test_fewer_than_two_terminals(self):
        snap = _snapshot([_unit("u:A", name="a")], [])
        result = compute_steiner_path(["a"], snap)
        assert result["found"] is False
        assert "At least 2" in result["error"]

    def test_too_many_terminals(self):
        snap = _snapshot([], [])
        terms = ["a"] * 9
        result = compute_steiner_path(terms, snap)
        assert result["found"] is False
        assert "Maximum 8" in result["error"]

    def test_symbol_not_found(self):
        snap = _snapshot([_unit("u:A", name="a")], [])
        result = compute_steiner_path(["a", "nonexistent"], snap)
        assert "Symbol not found" in result["error"]

    def test_disconnected_terminals(self):
        """A and B are not connected."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
        ]
        snap = _snapshot(units, [])  # no edges
        result = compute_steiner_path(["a", "b"], snap)
        # Both not in graph (no edges)
        assert result["found"] is False

    def test_edge_types_in_result(self):
        """Edges carry type information."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
        ]
        rels = [_rel("u:A", "u:B", "call")]
        snap = _snapshot(units, rels)
        result = compute_steiner_path(["a", "b"], snap)
        assert result["found"] is True
        assert result["edges"][0]["type"] == "call"

    def test_duplicate_terminals(self):
        """Duplicate terminal names should be deduplicated."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
        ]
        rels = [_rel("u:A", "u:B", "call")]
        snap = _snapshot(units, rels)
        result = compute_steiner_path(["a", "a", "b"], snap)
        assert result["found"] is True
        # Should still find the path with deduped terminals
        assert len(result["nodes"]) >= 2

    def test_multi_edge_type(self):
        """A->B via both call and import; edge type shows both."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
        ]
        rels = [
            _rel("u:A", "u:B", "call"),
            _rel("u:A", "u:B", "import"),
        ]
        snap = _snapshot(units, rels)
        result = compute_steiner_path(["a", "b"], snap)
        assert result["found"] is True
        assert "call" in result["edges"][0]["type"]
        assert "dependency" in result["edges"][0]["type"]


# ===========================================================================
# leiden_clusters tests (unit tests for extract_cluster_data)
# ===========================================================================


class TestLeidenClusters:
    def test_extract_cluster_data_basic(self):
        """Extract cluster data from a well-formed L1 projection."""
        snap = _snapshot([], [])
        l1_data = {
            "content_extra": {
                "sections": [
                    {"name": "Core Engine", "text": "12 nodes"},
                    {"name": "API Layer", "text": "8 nodes"},
                ],
                "navigation": [
                    {"label": "Core Engine", "ref": {"display_name": "FastCode"}},
                    {"label": "API Layer", "ref": {"display_name": "query"}},
                ],
                "relations": {
                    "xref": [
                        {"id": "Core Engine->API Layer", "confidence": 0.75},
                    ],
                },
            },
        }
        result = extract_cluster_data(l1_data, snap)
        assert result["error"] is None
        assert result["total_clusters"] == 2
        assert result["clusters"][0]["label"] == "Core Engine"
        assert result["clusters"][0]["node_count"] == 12
        assert result["clusters"][0]["representative"] == "FastCode"
        assert result["clusters"][1]["node_count"] == 8
        assert len(result["xrefs"]) == 1
        assert result["xrefs"][0]["from_cluster"] == "Core Engine"
        assert result["xrefs"][0]["to_cluster"] == "API Layer"
        assert result["xrefs"][0]["weight"] == 0.75

    def test_extract_cluster_data_empty(self):
        """Empty L1 data returns empty clusters."""
        snap = _snapshot([], [])
        l1_data = {"content_extra": {}}
        result = extract_cluster_data(l1_data, snap)
        assert result["total_clusters"] == 0
        assert result["clusters"] == []
        assert result["xrefs"] == []

    def test_extract_cluster_data_no_node_count(self):
        """Section text without parseable node count defaults to 0."""
        snap = _snapshot([], [])
        l1_data = {
            "content_extra": {
                "sections": [
                    {"name": "Unknown", "text": "many nodes"},
                ],
                "navigation": [],
            },
        }
        result = extract_cluster_data(l1_data, snap)
        assert result["clusters"][0]["node_count"] == 0

    def test_extract_cluster_data_no_representative(self):
        """Cluster without navigation entry has no representative."""
        snap = _snapshot([], [])
        l1_data = {
            "content_extra": {
                "sections": [
                    {"name": "Orphan", "text": "3 nodes"},
                ],
                "navigation": [],
            },
        }
        result = extract_cluster_data(l1_data, snap)
        assert result["clusters"][0]["representative"] is None

    def test_extract_cluster_data_multiple_xrefs(self):
        """Multiple cross-cluster references are extracted."""
        snap = _snapshot([], [])
        l1_data = {
            "content_extra": {
                "sections": [],
                "navigation": [],
                "relations": {
                    "xref": [
                        {"id": "A->B", "confidence": 0.5},
                        {"id": "B->C", "confidence": 0.8},
                        {"id": "A->C", "confidence": 0.3},
                    ],
                },
            },
        }
        result = extract_cluster_data(l1_data, snap)
        assert len(result["xrefs"]) == 3
        assert result["xrefs"][1]["from_cluster"] == "B"
        assert result["xrefs"][1]["to_cluster"] == "C"


# ===========================================================================
# resolve_unit_id tests
# ===========================================================================


class TestResolveUnitId:
    def test_exact_unit_id(self):
        snap = _snapshot(
            [_unit("u:1", name="foo"), _unit("u:2", name="bar")],
            [],
        )
        assert resolve_unit_id("u:2", snap) == "u:2"

    def test_display_name(self):
        snap = _snapshot(
            [_unit("u:1", name="foo"), _unit("u:2", name="bar")],
            [],
        )
        assert resolve_unit_id("bar", snap) == "u:2"

    def test_qualified_name(self):
        snap = _snapshot(
            [_unit("u:1", name="foo", qualified_name="mod.foo")],
            [],
        )
        assert resolve_unit_id("mod.foo", snap) == "u:1"

    def test_case_insensitive_display_name(self):
        snap = _snapshot(
            [_unit("u:1", name="MyClass")],
            [],
        )
        assert resolve_unit_id("myclass", snap) == "u:1"

    def test_not_found(self):
        snap = _snapshot([_unit("u:1", name="foo")], [])
        assert resolve_unit_id("nonexistent", snap) is None


# ===========================================================================
# format_path_node tests
# ===========================================================================


class TestFormatPathNode:
    def test_known_unit(self):
        snap = _snapshot([_unit("u:1", name="foo", path="a.py", start_line=10)], [])
        node = format_path_node("u:1", snap)
        assert node["symbol_id"] == "u:1"
        assert node["display_name"] == "foo"
        assert node["kind"] == "function"
        assert node["path"] == "a.py"
        assert node["start_line"] == 10

    def test_unknown_unit(self):
        snap = _snapshot([], [])
        node = format_path_node("u:missing", snap)
        assert node["symbol_id"] == "u:missing"
        assert node["display_name"] is None


# ===========================================================================
# Directed path integration tests
# ===========================================================================


class TestDirectedPath:
    def test_simple_call_path(self):
        """A -> B -> C via call edges."""
        units = [
            _unit("u:A", name="main"),
            _unit("u:B", name="process"),
            _unit("u:C", name="render"),
        ]
        rels = [
            _rel("u:A", "u:B", "call"),
            _rel("u:B", "u:C", "call"),
        ]
        snap = _snapshot(units, rels)
        result = compute_directed_path("main", "render", snap)
        assert result["found"] is True
        assert result["path_length"] == 2
        assert [n["display_name"] for n in result["path"]] == [
            "main",
            "process",
            "render",
        ]

    def test_direct_call(self):
        """A -> B directly."""
        units = [
            _unit("u:A", name="caller"),
            _unit("u:B", name="callee"),
        ]
        rels = [_rel("u:A", "u:B", "call")]
        snap = _snapshot(units, rels)
        result = compute_directed_path("caller", "callee", snap)
        assert result["found"] is True
        assert result["path_length"] == 1
        assert result["error"] is None

    def test_no_path(self):
        """A and B are in the graph but not connected."""
        units = [
            _unit("u:A", name="alpha"),
            _unit("u:B", name="beta"),
            _unit("u:C", name="gamma"),
        ]
        # C->A creates nodes in the graph, but B has no edges to/from it.
        # NetworkX graphs only have nodes that appear in edges.
        # We need an edge that touches A but not B to get "no path" vs "empty".
        _rels_unused = [_rel("u:A", "u:C", "call")]
        _ = _snapshot(units, _rels_unused)  # unused; kept for documentation
        # B is not in the graph at all (no edges touch it), so this hits
        # "not in graph" rather than "no path". Test with both in graph but
        # unreachable instead.
        rels2 = [
            _rel("u:A", "u:C", "call"),
            _rel("u:B", "u:C", "call"),  # both A and B point to C, not to each other
        ]
        snap2 = _snapshot(units, rels2)
        result = compute_directed_path("alpha", "beta", snap2)
        assert result["found"] is False
        assert "No directed path" in result["error"]

    def test_reverse_path_exists(self):
        """A <- B (edge from B to A), querying A -> B reports reverse."""
        units = [
            _unit("u:A", name="alpha"),
            _unit("u:B", name="beta"),
        ]
        rels = [_rel("u:B", "u:A", "call")]
        snap = _snapshot(units, rels)
        result = compute_directed_path("alpha", "beta", snap)
        assert result["found"] is False
        assert "reverse path exists" in result["error"]

    def test_same_symbol(self):
        """from and to are the same symbol."""
        units = [_unit("u:A", name="foo")]
        snap = _snapshot(units, [])
        result = compute_directed_path("foo", "foo", snap)
        assert result["found"] is True
        assert result["path_length"] == 0
        assert len(result["path"]) == 1

    def test_symbol_not_found(self):
        """Query a symbol that does not exist."""
        snap = _snapshot([_unit("u:A", name="foo")], [])
        result = compute_directed_path("foo", "nonexistent", snap)
        assert result["found"] is False
        assert result["error"] == "Symbol not found: nonexistent"

    def test_max_hops_exceeded(self):
        """Path exists but exceeds max_hops."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
            _unit("u:C", name="c"),
            _unit("u:D", name="d"),
        ]
        rels = [
            _rel("u:A", "u:B", "call"),
            _rel("u:B", "u:C", "call"),
            _rel("u:C", "u:D", "call"),
        ]
        snap = _snapshot(units, rels)
        result = compute_directed_path("a", "d", snap, max_hops=2)
        assert result["found"] is False
        assert "exceeds max_hops=2" in result["error"]
        assert result["path_length"] == 3

    def test_dependency_graph_type(self):
        """Traverse import/dependency edges."""
        units = [
            _unit("u:A", name="mod_a"),
            _unit("u:B", name="mod_b"),
        ]
        rels = [_rel("u:A", "u:B", "import")]
        snap = _snapshot(units, rels)
        result = compute_directed_path(
            "mod_a", "mod_b", snap, graph_types=["dependency"]
        )
        assert result["found"] is True
        assert result["path_length"] == 1

    def test_multi_graph_union(self):
        """Path uses both call and import edges."""
        units = [
            _unit("u:A", name="entry"),
            _unit("u:B", name="middle"),
            _unit("u:C", name="target"),
        ]
        rels = [
            _rel("u:A", "u:B", "call"),
            _rel("u:B", "u:C", "import"),
        ]
        snap = _snapshot(units, rels)
        # With both call and dependency
        result = compute_directed_path(
            "entry", "target", snap, graph_types=["call", "dependency"]
        )
        assert result["found"] is True
        assert result["path_length"] == 2

        # With only call (should fail since B->C is import)
        result2 = compute_directed_path("entry", "target", snap, graph_types=["call"])
        assert result2["found"] is False

    def test_invalid_graph_type(self):
        snap = _snapshot([_unit("u:A", name="foo")], [])
        result = compute_directed_path("foo", "foo", snap, graph_types=["invalid_type"])
        assert result["found"] is False
        assert "Invalid graph types" in result["error"]

    def test_empty_graph_type(self):
        """Selecting a graph type that has no edges returns empty-graph error."""
        units = [_unit("u:A", name="foo"), _unit("u:B", name="bar")]
        snap = _snapshot(units, [])  # no relations at all
        result = compute_directed_path("foo", "bar", snap, graph_types=["inheritance"])
        assert result["found"] is False
        assert "empty" in result["error"]

    def test_symbol_not_in_graph(self):
        """Unit exists but has no edges so it is not in the graph."""
        units = [
            _unit("u:A", name="connected"),
            _unit("u:B", name="peer"),
            _unit("u:C", name="isolated"),
        ]
        rels = [_rel("u:A", "u:B", "call")]
        snap = _snapshot(units, rels)
        result = compute_directed_path("connected", "isolated", snap)
        assert result["found"] is False
        assert "not in graph" in result["error"]

    def test_resolve_by_unit_id(self):
        """Resolve symbols by their unit_id rather than display_name."""
        units = [
            _unit("u:A", name="foo"),
            _unit("u:B", name="bar"),
        ]
        rels = [_rel("u:A", "u:B", "call")]
        snap = _snapshot(units, rels)
        result = compute_directed_path("u:A", "u:B", snap)
        assert result["found"] is True
        assert result["path_length"] == 1

    def test_path_node_metadata(self):
        """Each node in the path includes full metadata."""
        units = [
            _unit("u:A", name="start", path="main.py", kind="function", start_line=5),
            _unit("u:B", name="end", path="util.py", kind="function", start_line=20),
        ]
        rels = [_rel("u:A", "u:B", "call")]
        snap = _snapshot(units, rels)
        result = compute_directed_path("start", "end", snap)
        assert result["found"] is True
        path = result["path"]
        assert path[0]["symbol_id"] == "u:A"
        assert path[0]["display_name"] == "start"
        assert path[0]["path"] == "main.py"
        assert path[0]["start_line"] == 5
        assert path[1]["symbol_id"] == "u:B"
        assert path[1]["path"] == "util.py"


# ===========================================================================
# IRGraphBuilder integration (ensures graph building works with relations)
# ===========================================================================


class TestGraphBuilderIntegration:
    def test_call_graph_from_snapshot(self):
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
        ]
        rels = [_rel("u:A", "u:B", "call")]
        snap = _snapshot(units, rels)
        graphs = IRGraphBuilder().build_graphs(snap)
        assert graphs.call_graph.has_edge("u:A", "u:B")
        assert not graphs.call_graph.has_edge("u:B", "u:A")

    def test_dependency_graph_from_snapshot(self):
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
        ]
        rels = [_rel("u:A", "u:B", "import")]
        snap = _snapshot(units, rels)
        graphs = IRGraphBuilder().build_graphs(snap)
        assert graphs.dependency_graph.has_edge("u:A", "u:B")
        assert graphs.call_graph.number_of_edges() == 0

    def test_mixed_relation_types(self):
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
            _unit("u:C", name="c"),
        ]
        rels = [
            _rel("u:A", "u:B", "call"),
            _rel("u:B", "u:C", "import"),
            _rel("u:A", "u:C", "inherit"),  # not in call or dependency
        ]
        snap = _snapshot(units, rels)
        graphs = IRGraphBuilder().build_graphs(snap)
        assert graphs.call_graph.number_of_edges() == 1
        assert graphs.dependency_graph.number_of_edges() == 1
        assert graphs.inheritance_graph.number_of_edges() == 1
