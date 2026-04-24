"""Tests for impact_analysis, leiden_clusters, steiner_path, and find_callers MCP tools.

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


# ---------------------------------------------------------------------------
# impact_analysis logic (mirrors mcp_server.py)
# ---------------------------------------------------------------------------


def _run_impact_analysis(snapshot, symbol: str, **kwargs) -> dict:
    """Execute impact_analysis logic and return result dict."""
    from collections import deque

    max_hops = kwargs.get("max_hops", 3)
    graph_types = kwargs.get("graph_types", ["call", "dependency"])

    invalid = [gt for gt in graph_types if gt not in _VALID_GRAPH_TYPES]
    if invalid:
        return {
            "affected": [],
            "total_count": 0,
            "error": f"Invalid graph types: {invalid}. Valid: {sorted(_VALID_GRAPH_TYPES)}",
        }

    unit_id = _resolve_unit_id(symbol, snapshot)
    if not unit_id:
        return {
            "affected": [],
            "total_count": 0,
            "error": f"Symbol not found: {symbol}",
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
            "affected": [],
            "total_count": 0,
            "error": "Selected graph types are empty (no nodes or edges).",
        }

    if unit_id not in combined:
        return {
            "affected": [],
            "total_count": 0,
            "error": f"Symbol not in graph: {symbol}",
        }

    visited: dict[str, int] = {}
    edge_types_map: dict[str, set[str]] = {}
    queue = deque()
    queue.append((unit_id, 0))
    visited[unit_id] = 0

    while queue:
        node, dist = queue.popleft()
        if dist >= max_hops:
            continue
        for pred in combined.predecessors(node):
            if pred not in visited:
                visited[pred] = dist + 1
                edge_types_map[pred] = set()
                queue.append((pred, dist + 1))
            for gt in graph_types:
                attr = _GRAPH_TYPE_MAP[gt]
                g: nx.DiGraph = getattr(graphs, attr)
                if g.has_edge(pred, node):
                    edge_types_map.setdefault(pred, set()).add(gt)

    affected = []
    for nid, dist in sorted(visited.items(), key=lambda kv: kv[1]):
        if nid == unit_id:
            continue
        node_info = _format_path_node(nid, snapshot)
        node_info["distance"] = dist
        node_info["edge_types"] = sorted(edge_types_map.get(nid, set()))
        affected.append(node_info)

    return {"affected": affected, "total_count": len(affected), "error": None}


# ---------------------------------------------------------------------------
# steiner_path logic (mirrors mcp_server.py)
# ---------------------------------------------------------------------------


def _run_steiner_path(snapshot, terminals: list[str]) -> dict:
    """Execute steiner_path logic and return result dict."""
    if not terminals or len(terminals) < 2:
        return {
            "found": False,
            "nodes": [],
            "edges": [],
            "error": "At least 2 terminal symbols required.",
        }

    if len(terminals) > 8:
        return {
            "found": False,
            "nodes": [],
            "edges": [],
            "error": "Maximum 8 terminal symbols allowed.",
        }

    terminal_ids = []
    for t in terminals:
        tid = _resolve_unit_id(t, snapshot)
        if not tid:
            return {
                "found": False,
                "nodes": [],
                "edges": [],
                "error": f"Symbol not found: {t}",
            }
        terminal_ids.append(tid)

    terminal_ids = list(dict.fromkeys(terminal_ids))

    graphs = IRGraphBuilder().build_graphs(snapshot)
    undirected = None
    for attr_name in [
        "call_graph",
        "dependency_graph",
        "inheritance_graph",
        "reference_graph",
        "containment_graph",
    ]:
        g: nx.DiGraph = getattr(graphs, attr_name)
        ug = g.to_undirected()
        if undirected is None:
            undirected = ug.copy()
        else:
            undirected = nx.compose(undirected, ug)

    if undirected is None or undirected.number_of_nodes() == 0:
        return {
            "found": False,
            "nodes": [],
            "edges": [],
            "error": "Graph is empty (no nodes or edges).",
        }

    missing = [t for t, tid in zip(terminals, terminal_ids) if tid not in undirected]
    if missing:
        return {
            "found": False,
            "nodes": [],
            "edges": [],
            "error": f"Symbol(s) not in graph: {missing}",
        }

    if len(terminal_ids) == 1:
        node = _format_path_node(terminal_ids[0], snapshot)
        return {"found": True, "nodes": [node], "edges": [], "error": None}

    try:
        steiner = nx.approximation.steiner_tree(undirected, terminal_ids)
    except nx.NetworkXError as exc:
        return {
            "found": False,
            "nodes": [],
            "edges": [],
            "error": f"Steiner tree computation failed: {exc}",
        }

    terminal_set = set(terminal_ids)
    changed = True
    while changed:
        changed = False
        leaves = [
            n
            for n in steiner.nodes()
            if steiner.degree(n) == 1 and n not in terminal_set
        ]
        for leaf in leaves:
            steiner.remove_node(leaf)
            changed = True

    if steiner.number_of_nodes() == 0:
        return {
            "found": False,
            "nodes": [],
            "edges": [],
            "error": "Steiner tree is empty after pruning.",
        }

    nodes = [_format_path_node(nid, snapshot) for nid in steiner.nodes()]

    edges = []
    edge_type_map = {
        "call_graph": "call",
        "dependency_graph": "dependency",
        "inheritance_graph": "inheritance",
        "reference_graph": "reference",
        "containment_graph": "containment",
    }
    seen_edges = set()
    for u, v in steiner.edges():
        edge_key = (min(u, v), max(u, v))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edge_types = []
        for attr_name, etype in edge_type_map.items():
            g: nx.DiGraph = getattr(graphs, attr_name)
            if g.has_edge(u, v) or g.has_edge(v, u):
                edge_types.append(etype)
        edges.append(
            {
                "from": u,
                "to": v,
                "type": "+".join(edge_types) if edge_types else "unknown",
            }
        )

    return {"found": True, "nodes": nodes, "edges": edges, "error": None}


# ---------------------------------------------------------------------------
# find_callers logic (mirrors mcp_server.py)
# ---------------------------------------------------------------------------


def _run_find_callers(snapshot, symbol: str, **kwargs) -> dict:
    """Execute find_callers logic and return result dict."""
    from collections import deque

    max_hops = kwargs.get("max_hops", 2)

    unit_id = _resolve_unit_id(symbol, snapshot)
    if not unit_id:
        return {"callers": [], "total_count": 0, "error": f"Symbol not found: {symbol}"}

    graphs = IRGraphBuilder().build_graphs(snapshot)
    call_g = graphs.call_graph

    if call_g.number_of_nodes() == 0:
        return {"callers": [], "total_count": 0, "error": "Call graph is empty."}

    if unit_id not in call_g:
        return {
            "callers": [],
            "total_count": 0,
            "error": f"Symbol not in call graph: {symbol}",
        }

    visited: dict[str, int] = {}
    queue = deque()
    queue.append((unit_id, 0))
    visited[unit_id] = 0

    while queue:
        node, dist = queue.popleft()
        if dist >= max_hops:
            continue
        for pred in call_g.predecessors(node):
            if pred not in visited:
                visited[pred] = dist + 1
                queue.append((pred, dist + 1))

    callers = []
    for nid, dist in sorted(visited.items(), key=lambda kv: kv[1]):
        if nid == unit_id:
            continue
        node_info = _format_path_node(nid, snapshot)
        node_info["distance"] = dist
        callers.append(node_info)

    return {"callers": callers, "total_count": len(callers), "error": None}


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
        result = _run_impact_analysis(snap, "callee")
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
        result = _run_impact_analysis(snap, "bottom")
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
        result = _run_impact_analysis(snap, "d", max_hops=1)
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
        result = _run_impact_analysis(snap, "mod_b", graph_types=["dependency"])
        assert result["total_count"] == 1
        assert result["affected"][0]["display_name"] == "mod_a"

    def test_symbol_not_found(self):
        snap = _snapshot([_unit("u:A", name="foo")], [])
        result = _run_impact_analysis(snap, "nonexistent")
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
        result = _run_impact_analysis(snap, "isolated")
        assert "not in graph" in result["error"]

    def test_leaf_symbol_no_callers(self):
        """Leaf symbol (nothing depends on it) has zero impact."""
        units = [
            _unit("u:A", name="leaf"),
            _unit("u:B", name="caller"),
        ]
        rels = [_rel("u:B", "u:A", "call")]  # B calls A; nothing calls B
        snap = _snapshot(units, rels)
        result = _run_impact_analysis(snap, "caller")
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
        result = _run_impact_analysis(snap, "b")
        assert result["total_count"] == 1
        assert "call" in result["affected"][0]["edge_types"]

    def test_invalid_graph_type(self):
        snap = _snapshot([_unit("u:A", name="foo")], [])
        result = _run_impact_analysis(snap, "foo", graph_types=["invalid"])
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
        result = _run_impact_analysis(snap, "b", graph_types=["call", "dependency"])
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
        result = _run_find_callers(snap, "callee")
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
        result = _run_find_callers(snap, "b")
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
        result = _run_find_callers(snap, "bot")
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
        result = _run_find_callers(snap, "c", max_hops=1)
        assert result["total_count"] == 1
        assert result["callers"][0]["display_name"] == "b"

    def test_symbol_not_found(self):
        snap = _snapshot([_unit("u:A", name="foo")], [])
        result = _run_find_callers(snap, "nonexistent")
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
        result = _run_find_callers(snap, "b")
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
        result = _run_find_callers(snap, "target")
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
        result = _run_find_callers(snap, "callee")
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
        result = _run_find_callers(snap, "c")
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
        result = _run_steiner_path(snap, ["a", "b"])
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
        result = _run_steiner_path(snap, ["a", "c"])
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
        result = _run_steiner_path(snap, ["a", "c"])
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
        result = _run_steiner_path(snap, ["a", "c"])
        assert result["found"] is True
        node_ids = {n["symbol_id"] for n in result["nodes"]}
        assert "u:D" not in node_ids

    def test_fewer_than_two_terminals(self):
        snap = _snapshot([_unit("u:A", name="a")], [])
        result = _run_steiner_path(snap, ["a"])
        assert result["found"] is False
        assert "At least 2" in result["error"]

    def test_too_many_terminals(self):
        snap = _snapshot([], [])
        terms = ["a"] * 9
        result = _run_steiner_path(snap, terms)
        assert result["found"] is False
        assert "Maximum 8" in result["error"]

    def test_symbol_not_found(self):
        snap = _snapshot([_unit("u:A", name="a")], [])
        result = _run_steiner_path(snap, ["a", "nonexistent"])
        assert "Symbol not found" in result["error"]

    def test_disconnected_terminals(self):
        """A and B are not connected."""
        units = [
            _unit("u:A", name="a"),
            _unit("u:B", name="b"),
        ]
        snap = _snapshot(units, [])  # no edges
        result = _run_steiner_path(snap, ["a", "b"])
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
        result = _run_steiner_path(snap, ["a", "b"])
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
        result = _run_steiner_path(snap, ["a", "a", "b"])
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
        result = _run_steiner_path(snap, ["a", "b"])
        assert result["found"] is True
        assert "call" in result["edges"][0]["type"]
        assert "dependency" in result["edges"][0]["type"]


# ===========================================================================
# leiden_clusters tests (unit tests for _extract_cluster_data)
# ===========================================================================


def _extract_cluster_data(l1_data: dict, snapshot) -> dict:
    """Replicated from mcp_server.py for testing."""
    clusters = []
    xrefs = []

    content_extra = l1_data.get("content_extra", {})
    sections = content_extra.get("sections", [])
    navigation = content_extra.get("navigation", [])

    relations = content_extra.get("relations", {})
    xref_list = relations.get("xref", [])
    for xref in xref_list:
        xref_id = xref.get("id", "")
        parts = xref_id.split("->")
        if len(parts) == 2:
            xrefs.append(
                {
                    "from_cluster": parts[0],
                    "to_cluster": parts[1],
                    "weight": xref.get("confidence", 0),
                }
            )

    for i, section in enumerate(sections):
        cluster_info = {
            "cluster_id": str(i),
            "label": section.get("name", f"Cluster {i}"),
            "node_count": 0,
            "representative": None,
            "top_members": [],
        }
        text = section.get("text", "")
        try:
            cluster_info["node_count"] = int(text.split()[0])
        except (ValueError, IndexError):
            pass
        if i < len(navigation):
            nav = navigation[i]
            rep_ref = nav.get("ref", {})
            if rep_ref:
                rep_display = rep_ref.get("display_name")
                if rep_display:
                    cluster_info["representative"] = rep_display
        clusters.append(cluster_info)

    return {
        "clusters": clusters,
        "xrefs": xrefs,
        "total_clusters": len(clusters),
        "error": None,
    }


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
        result = _extract_cluster_data(l1_data, snap)
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
        result = _extract_cluster_data(l1_data, snap)
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
        result = _extract_cluster_data(l1_data, snap)
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
        result = _extract_cluster_data(l1_data, snap)
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
        result = _extract_cluster_data(l1_data, snap)
        assert len(result["xrefs"]) == 3
        assert result["xrefs"][1]["from_cluster"] == "B"
        assert result["xrefs"][1]["to_cluster"] == "C"
