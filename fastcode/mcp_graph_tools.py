"""Pure functions for MCP graph tools.

Extracted from mcp_server.py to enable unit testing without importing the
MCP server module (which has heavy side effects).  Each compute_* function
accepts an IRSnapshot (and optionally a FastCode instance) and returns a
plain Python dict.  The @mcp.tool() wrappers in mcp_server.py handle
json.dumps().
"""

from __future__ import annotations

import contextlib
from collections import deque

import networkx as nx

from .ir_graph_builder import IRGraphBuilder
from .semantic_ir import IRSnapshot

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_GRAPH_TYPES = {"call", "dependency", "inheritance", "reference", "containment"}

GRAPH_TYPE_MAP = {
    "call": "call_graph",
    "dependency": "dependency_graph",
    "inheritance": "inheritance_graph",
    "reference": "reference_graph",
    "containment": "containment_graph",
}

_ALL_GRAPH_ATTRS = [
    "call_graph",
    "dependency_graph",
    "inheritance_graph",
    "reference_graph",
    "containment_graph",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_unit_id(query: str, snapshot: IRSnapshot) -> str | None:
    """Resolve a symbol query to a unit_id from snapshot units.

    Tries exact unit_id, then display_name, then qualified_name,
    then case-insensitive display_name.  Returns the first match or None.
    """
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


def format_path_node(unit_id: str, snapshot: IRSnapshot) -> dict[str, str | int | None]:
    """Build a metadata dict for a graph node."""
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


def build_combined_graph(
    snapshot: IRSnapshot,
    graph_types: list[str] | None = None,
    undirected: bool = False,
) -> nx.Graph | nx.DiGraph:
    """Build graphs from snapshot and compose selected types into one.

    Args:
        snapshot: The IRSnapshot to build graphs from.
        graph_types: Which graph type names to include.  If None, all five.
        undirected: If True, convert each DiGraph to undirected before
            composing and return a plain Graph.

    Returns:
        A composed networkx Graph (undirected=True) or DiGraph.
    """
    if graph_types is None:
        graph_types = sorted(VALID_GRAPH_TYPES)

    graphs = IRGraphBuilder().build_graphs(snapshot)
    combined: nx.Graph | nx.DiGraph | None = None

    for gt in graph_types:
        attr = GRAPH_TYPE_MAP[gt]
        g: nx.DiGraph = getattr(graphs, attr)
        if undirected:
            g = g.to_undirected()
        combined = g.copy() if combined is None else nx.compose(combined, g)
    return (
        combined
        if combined is not None
        else (nx.Graph() if undirected else nx.DiGraph())
    )


# ---------------------------------------------------------------------------
# Compute functions (one per MCP tool)
# ---------------------------------------------------------------------------


def compute_directed_path(
    from_symbol: str,
    to_symbol: str,
    snapshot: IRSnapshot,
    max_hops: int = 5,
    graph_types: list[str] | None = None,
) -> dict:
    """Find the directed shortest path between two symbols."""
    if graph_types is None:
        graph_types = ["call", "dependency"]

    invalid = [gt for gt in graph_types if gt not in VALID_GRAPH_TYPES]
    if invalid:
        return {
            "found": False,
            "path": [],
            "path_length": 0,
            "error": f"Invalid graph types: {invalid}. Valid: {sorted(VALID_GRAPH_TYPES)}",
        }

    from_id = resolve_unit_id(from_symbol, snapshot)
    if not from_id:
        return {
            "found": False,
            "path": [],
            "path_length": 0,
            "error": f"Symbol not found: {from_symbol}",
        }

    to_id = resolve_unit_id(to_symbol, snapshot)
    if not to_id:
        return {
            "found": False,
            "path": [],
            "path_length": 0,
            "error": f"Symbol not found: {to_symbol}",
        }

    if from_id == to_id:
        return {
            "found": True,
            "path": [format_path_node(from_id, snapshot)],
            "path_length": 0,
            "error": None,
        }

    combined = build_combined_graph(snapshot, graph_types, undirected=False)

    if combined.number_of_nodes() == 0:
        return {
            "found": False,
            "path": [],
            "path_length": 0,
            "error": "Selected graph types are empty (no nodes or edges).",
        }

    if from_id not in combined or to_id not in combined:
        missing = []
        if from_id not in combined:
            missing.append(from_symbol)
        if to_id not in combined:
            missing.append(to_symbol)
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
                "error": (
                    f"No directed path from '{from_symbol}' to '{to_symbol}'. "
                    f"A reverse path exists (from '{to_symbol}' to '{from_symbol}')."
                ),
            }
        except nx.NetworkXNoPath:
            return {
                "found": False,
                "path": [],
                "path_length": 0,
                "error": f"No directed path between '{from_symbol}' and '{to_symbol}' in either direction.",
            }

    if len(path) - 1 > max_hops:
        return {
            "found": False,
            "path": [],
            "path_length": len(path) - 1,
            "error": f"Shortest path length {len(path) - 1} exceeds max_hops={max_hops}.",
        }

    path_nodes = [format_path_node(nid, snapshot) for nid in path]
    return {
        "found": True,
        "path": path_nodes,
        "path_length": len(path) - 1,
        "error": None,
    }


def compute_impact_analysis(
    symbol: str,
    snapshot: IRSnapshot,
    max_hops: int = 3,
    graph_types: list[str] | None = None,
) -> dict:
    """Analyze what would be affected if a symbol changes (BFS on predecessors)."""
    if graph_types is None:
        graph_types = ["call", "dependency"]

    invalid = [gt for gt in graph_types if gt not in VALID_GRAPH_TYPES]
    if invalid:
        return {
            "affected": [],
            "total_count": 0,
            "error": f"Invalid graph types: {invalid}. Valid: {sorted(VALID_GRAPH_TYPES)}",
        }

    unit_id = resolve_unit_id(symbol, snapshot)
    if not unit_id:
        return {
            "affected": [],
            "total_count": 0,
            "error": f"Symbol not found: {symbol}",
        }

    graphs = IRGraphBuilder().build_graphs(snapshot)
    combined: nx.DiGraph | None = None
    for gt in graph_types:
        attr = GRAPH_TYPE_MAP[gt]
        g: nx.DiGraph = getattr(graphs, attr)
        combined = g.copy() if combined is None else nx.compose(combined, g)
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
    queue: deque[tuple[str, int]] = deque()
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
                attr = GRAPH_TYPE_MAP[gt]
                g: nx.DiGraph = getattr(graphs, attr)
                if g.has_edge(pred, node):
                    edge_types_map.setdefault(pred, set()).add(gt)

    affected = []
    for nid, dist in sorted(visited.items(), key=lambda kv: kv[1]):
        if nid == unit_id:
            continue
        node_info = format_path_node(nid, snapshot)
        node_info["distance"] = dist
        node_info["edge_types"] = sorted(edge_types_map.get(nid, set()))
        affected.append(node_info)

    return {"affected": affected, "total_count": len(affected), "error": None}


def compute_leiden_clusters(snapshot: IRSnapshot, fc: object | None = None) -> dict:
    """Get module boundaries (Leiden community detection) for a snapshot.

    Args:
        snapshot: The IRSnapshot.
        fc: Optional FastCode instance for accessing projection store/transformer.

    Returns:
        Dict with clusters, xrefs, total_clusters, error.
    """
    if fc is None:
        return {
            "clusters": [],
            "xrefs": [],
            "total_clusters": 0,
            "error": "No FastCode instance provided.",
        }

    snapshot_id = snapshot.snapshot_id

    # Try to load cached projection first
    projection_store = getattr(fc, "projection_store", None)
    projection_transformer = getattr(fc, "projection_transformer", None)

    if projection_store is not None and projection_store.enabled:
        from .projection_models import ProjectionScope

        scope = ProjectionScope(
            scope_kind="full",
            snapshot_id=snapshot_id,
            scope_key="full",
        )
        cached_id = projection_store.find_cached_projection_id(scope, "default")
        if cached_id:
            l1_data = projection_store.get_layer(cached_id, "L1")
            if l1_data:
                return extract_cluster_data(l1_data, snapshot)

    # No cached projection; try to build one
    if projection_transformer is not None:
        try:
            from .projection_models import ProjectionScope

            scope = ProjectionScope(
                scope_kind="full",
                snapshot_id=snapshot_id,
                scope_key="full",
            )
            result = projection_transformer.build(scope, snapshot)
            if result and result.l1:
                return extract_cluster_data(result.l1, snapshot)
        except Exception as exc:
            return {
                "clusters": [],
                "xrefs": [],
                "total_clusters": 0,
                "error": f"Failed to build projection: {exc}",
            }

    return {
        "clusters": [],
        "xrefs": [],
        "total_clusters": 0,
        "error": "Projection store not configured and projection transformer not available.",
    }


def compute_steiner_path(
    terminals: list[str],
    snapshot: IRSnapshot,
) -> dict:
    """Find a small undirected explanatory subgraph connecting terminal symbols."""
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
        tid = resolve_unit_id(t, snapshot)
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
    undirected: nx.Graph | None = None
    for attr_name in _ALL_GRAPH_ATTRS:
        g: nx.DiGraph = getattr(graphs, attr_name)
        ug = g.to_undirected()
        undirected = ug.copy() if undirected is None else nx.compose(undirected, ug)
    if undirected is None or undirected.number_of_nodes() == 0:
        return {
            "found": False,
            "nodes": [],
            "edges": [],
            "error": "Graph is empty (no nodes or edges).",
        }

    missing = [
        t
        for t, tid in zip(terminals, terminal_ids, strict=False)
        if tid not in undirected
    ]
    if missing:
        return {
            "found": False,
            "nodes": [],
            "edges": [],
            "error": f"Symbol(s) not in graph: {missing}",
        }

    if len(terminal_ids) == 1:
        node = format_path_node(terminal_ids[0], snapshot)
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

    nodes = [format_path_node(nid, snapshot) for nid in steiner.nodes()]

    edge_type_map = {
        "call_graph": "call",
        "dependency_graph": "dependency",
        "inheritance_graph": "inheritance",
        "reference_graph": "reference",
        "containment_graph": "containment",
    }
    seen_edges: set[tuple[str, str]] = set()
    edges = []
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


def compute_find_callers(
    symbol: str,
    snapshot: IRSnapshot,
    max_hops: int = 2,
) -> dict:
    """Find all symbols that call the given symbol (BFS on reversed call graph)."""
    unit_id = resolve_unit_id(symbol, snapshot)
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
    queue: deque[tuple[str, int]] = deque()
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
        node_info = format_path_node(nid, snapshot)
        node_info["distance"] = dist
        callers.append(node_info)

    return {"callers": callers, "total_count": len(callers), "error": None}


# ---------------------------------------------------------------------------
# Shared data extraction
# ---------------------------------------------------------------------------


def extract_cluster_data(l1_data: dict, snapshot: IRSnapshot) -> dict:
    """Extract structured cluster data from L1 projection data."""
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
        with contextlib.suppress(ValueError, IndexError):
            cluster_info["node_count"] = int(text.split()[0])
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
