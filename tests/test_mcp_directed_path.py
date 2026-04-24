"""Tests for the directed_path MCP tool.

The tool logic in mcp_server.py uses module-level helpers (_resolve_unit_id,
_format_path_node, etc.). Since mcp_server.py lives at the project root and
tests may not reliably import it (xdist workers, path issues), the helpers are
replicated here for testing. The production code is in mcp_server.py and should
be kept in sync.
"""


import networkx as nx

from fastcode.ir_graph_builder import IRGraphBuilder
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


# ---------------------------------------------------------------------------
# Replicated helpers (mirror of mcp_server.py module-level functions)
# ---------------------------------------------------------------------------


_VALID_GRAPH_TYPES = {"call", "dependency", "inheritance", "reference", "containment"}

_GRAPH_TYPE_MAP = {
    "call": "call_graph",
    "dependency": "dependency_graph",
    "inheritance": "inheritance_graph",
    "reference": "reference_graph",
    "containment": "containment_graph",
}


def _resolve_unit_id(query: str, snapshot) -> str | None:
    for unit in snapshot.units:
        if unit.unit_id == query:
            return unit.unit_id
    for unit in snapshot.units:
        if unit.display_name == query:
            return unit.unit_id
    for unit in snapshot.units:
        if unit.qualified_name and unit.qualified_name == query:
            return unit.unit_id
    query_lower = query.lower()
    for unit in snapshot.units:
        if unit.display_name.lower() == query_lower:
            return unit.unit_id
    return None


def _format_path_node(unit_id: str, snapshot) -> dict[str, str | int | None]:
    for unit in snapshot.units:
        if unit.unit_id == unit_id:
            return {
                "symbol_id": unit.unit_id,
                "display_name": unit.display_name,
                "kind": unit.kind,
                "path": unit.path,
                "start_line": unit.start_line,
            }
    return {
        "symbol_id": unit_id,
        "display_name": None,
        "kind": None,
        "path": None,
        "start_line": None,
    }


def _run_directed_path(snapshot, from_sym: str, to_sym: str, **kwargs) -> dict:
    """Execute directed_path logic and return result dict."""
    max_hops = kwargs.get("max_hops", 5)
    graph_types = kwargs.get("graph_types", ["call", "dependency"])

    invalid = [gt for gt in graph_types if gt not in _VALID_GRAPH_TYPES]
    if invalid:
        return {
            "found": False,
            "path": [],
            "path_length": 0,
            "error": f"Invalid graph types: {invalid}. Valid: {sorted(_VALID_GRAPH_TYPES)}",
        }

    from_id = _resolve_unit_id(from_sym, snapshot)
    if not from_id:
        return {
            "found": False,
            "path": [],
            "path_length": 0,
            "error": f"Symbol not found: {from_sym}",
        }

    to_id = _resolve_unit_id(to_sym, snapshot)
    if not to_id:
        return {
            "found": False,
            "path": [],
            "path_length": 0,
            "error": f"Symbol not found: {to_sym}",
        }

    if from_id == to_id:
        return {
            "found": True,
            "path": [_format_path_node(from_id, snapshot)],
            "path_length": 0,
            "error": None,
        }

    graphs = IRGraphBuilder().build_graphs(snapshot)
    combined = None
    for gt in graph_types:
        attr = _GRAPH_TYPE_MAP[gt]
        g: nx.DiGraph = getattr(graphs, attr)
        if combined is None:
            combined = g.copy()
        else:
            combined = nx.compose(combined, g)

    if combined is None or combined.number_of_nodes() == 0:
        return {
            "found": False,
            "path": [],
            "path_length": 0,
            "error": "Selected graph types are empty (no nodes or edges).",
        }

    if from_id not in combined or to_id not in combined:
        missing = []
        if from_id not in combined:
            missing.append(from_sym)
        if to_id not in combined:
            missing.append(to_sym)
        return {
            "found": False,
            "path": [],
            "path_length": 0,
            "error": f"Symbol(s) not in graph: {missing}",
        }

    try:
        path = nx.shortest_path(combined, source=from_id, target=to_id)
    except nx.NetworkXNoPath:
        try:
            nx.shortest_path(combined, source=to_id, target=from_id)
            return {
                "found": False,
                "path": [],
                "path_length": 0,
                "error": f"No directed path from '{from_sym}' to '{to_sym}'. "
                f"A reverse path exists (from '{to_sym}' to '{from_sym}').",
            }
        except nx.NetworkXNoPath:
            return {
                "found": False,
                "path": [],
                "path_length": 0,
                "error": f"No directed path between '{from_sym}' and '{to_sym}' in either direction.",
            }

    if len(path) - 1 > max_hops:
        return {
            "found": False,
            "path": [],
            "path_length": len(path) - 1,
            "error": f"Shortest path length {len(path) - 1} exceeds max_hops={max_hops}.",
        }

    path_nodes = [_format_path_node(nid, snapshot) for nid in path]
    return {
        "found": True,
        "path": path_nodes,
        "path_length": len(path) - 1,
        "error": None,
    }


# ---------------------------------------------------------------------------
# _resolve_unit_id tests
# ---------------------------------------------------------------------------


class TestResolveUnitId:
    def test_exact_unit_id(self):
        snap = _snapshot(
            [_unit("u:1", name="foo"), _unit("u:2", name="bar")],
            [],
        )
        assert _resolve_unit_id("u:2", snap) == "u:2"

    def test_display_name(self):
        snap = _snapshot(
            [_unit("u:1", name="foo"), _unit("u:2", name="bar")],
            [],
        )
        assert _resolve_unit_id("bar", snap) == "u:2"

    def test_qualified_name(self):
        snap = _snapshot(
            [_unit("u:1", name="foo", qualified_name="mod.foo")],
            [],
        )
        assert _resolve_unit_id("mod.foo", snap) == "u:1"

    def test_case_insensitive_display_name(self):
        snap = _snapshot(
            [_unit("u:1", name="MyClass")],
            [],
        )
        assert _resolve_unit_id("myclass", snap) == "u:1"

    def test_not_found(self):
        snap = _snapshot([_unit("u:1", name="foo")], [])
        assert _resolve_unit_id("nonexistent", snap) is None


# ---------------------------------------------------------------------------
# _format_path_node tests
# ---------------------------------------------------------------------------


class TestFormatPathNode:
    def test_known_unit(self):
        snap = _snapshot([_unit("u:1", name="foo", path="a.py", start_line=10)], [])
        node = _format_path_node("u:1", snap)
        assert node["symbol_id"] == "u:1"
        assert node["display_name"] == "foo"
        assert node["kind"] == "function"
        assert node["path"] == "a.py"
        assert node["start_line"] == 10

    def test_unknown_unit(self):
        snap = _snapshot([], [])
        node = _format_path_node("u:missing", snap)
        assert node["symbol_id"] == "u:missing"
        assert node["display_name"] is None


# ---------------------------------------------------------------------------
# Directed path integration tests
# ---------------------------------------------------------------------------


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
        result = _run_directed_path(snap, "main", "render")
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
        result = _run_directed_path(snap, "caller", "callee")
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
        rels = [_rel("u:A", "u:C", "call")]
        snap = _snapshot(units, rels)
        # B is not in the graph at all (no edges touch it), so this hits
        # "not in graph" rather than "no path". Test with both in graph but
        # unreachable instead.
        rels2 = [
            _rel("u:A", "u:C", "call"),
            _rel("u:B", "u:C", "call"),  # both A and B point to C, not to each other
        ]
        snap2 = _snapshot(units, rels2)
        result = _run_directed_path(snap2, "alpha", "beta")
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
        result = _run_directed_path(snap, "alpha", "beta")
        assert result["found"] is False
        assert "reverse path exists" in result["error"]

    def test_same_symbol(self):
        """from and to are the same symbol."""
        units = [_unit("u:A", name="foo")]
        snap = _snapshot(units, [])
        result = _run_directed_path(snap, "foo", "foo")
        assert result["found"] is True
        assert result["path_length"] == 0
        assert len(result["path"]) == 1

    def test_symbol_not_found(self):
        """Query a symbol that does not exist."""
        snap = _snapshot([_unit("u:A", name="foo")], [])
        result = _run_directed_path(snap, "foo", "nonexistent")
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
        result = _run_directed_path(snap, "a", "d", max_hops=2)
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
        result = _run_directed_path(snap, "mod_a", "mod_b", graph_types=["dependency"])
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
        result = _run_directed_path(
            snap, "entry", "target", graph_types=["call", "dependency"]
        )
        assert result["found"] is True
        assert result["path_length"] == 2

        # With only call (should fail since B->C is import)
        result2 = _run_directed_path(snap, "entry", "target", graph_types=["call"])
        assert result2["found"] is False

    def test_invalid_graph_type(self):
        snap = _snapshot([_unit("u:A", name="foo")], [])
        result = _run_directed_path(snap, "foo", "foo", graph_types=["invalid_type"])
        assert result["found"] is False
        assert "Invalid graph types" in result["error"]

    def test_empty_graph_type(self):
        """Selecting a graph type that has no edges returns empty-graph error."""
        units = [_unit("u:A", name="foo"), _unit("u:B", name="bar")]
        snap = _snapshot(units, [])  # no relations at all
        result = _run_directed_path(snap, "foo", "bar", graph_types=["inheritance"])
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
        result = _run_directed_path(snap, "connected", "isolated")
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
        result = _run_directed_path(snap, "u:A", "u:B")
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
        result = _run_directed_path(snap, "start", "end")
        assert result["found"] is True
        path = result["path"]
        assert path[0]["symbol_id"] == "u:A"
        assert path[0]["display_name"] == "start"
        assert path[0]["path"] == "main.py"
        assert path[0]["start_line"] == 5
        assert path[1]["symbol_id"] == "u:B"
        assert path[1]["path"] == "util.py"


# ---------------------------------------------------------------------------
# IRGraphBuilder integration (ensures graph building works with relations)
# ---------------------------------------------------------------------------


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
