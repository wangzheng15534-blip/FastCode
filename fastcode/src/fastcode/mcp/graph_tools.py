"""Pure functions for MCP graph tools.

Extracted from mcp_server.py to enable unit testing without importing the
MCP server module (which has heavy side effects).  Each compute_* function
accepts an IRSnapshot (and optionally a FastCode instance) and returns a
plain Python dict.  The @mcp.tool() wrappers in mcp_server.py handle
json.dumps().
"""

from __future__ import annotations

import builtins
import contextlib
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, cast

import networkx as nx

from ..ir.graph import IRGraphBuilder
from ..ir.types import IRDocument, IRSnapshot, IRSymbol
from ..utils.materialization import (
    BOUNDARY_GRAPH_FULL_LOAD,
    BOUNDARY_SNAPSHOT_FULL_LOAD,
    increment_materialization_boundary,
)

# ---------------------------------------------------------------------------
# Type aliases for networkx functions with unknown **kwargs in stubs.
# Direct attribute access triggers reportUnknownMemberType because the
# function overload signatures include **kwargs: Unknown.  We use
# builtins.getattr (not the bare getattr builtin) which ruff-format
# will not rewrite to dot-access.
# ---------------------------------------------------------------------------
# ruff-format: off
_NXShortestPath: Any = builtins.getattr(nx, "shortest_path")  # noqa: B009
_NXSteinerTree: Any = builtins.getattr(nx.approximation, "steiner_tree")  # noqa: B009
# ruff-format: on

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


def _empty_str_map() -> dict[str, str]:
    return {}


def _empty_record_map() -> dict[str, dict[str, Any]]:
    return {}


@dataclass
class GraphToolContext:
    """Compact graph-tool view backed by saved graph and symbol sidecars."""

    snapshot_id: str
    graphs: Any
    canonical_by_alias: dict[str, str] = field(default_factory=_empty_str_map)
    canonical_by_name: dict[str, str] = field(default_factory=_empty_str_map)
    canonical_by_lower_name: dict[str, str] = field(default_factory=_empty_str_map)
    records_by_id: dict[str, dict[str, Any]] = field(default_factory=_empty_record_map)

    @classmethod
    def from_symbol_payload(
        cls,
        *,
        snapshot_id: str,
        graphs: Any,
        payload: Mapping[str, Any],
    ) -> GraphToolContext:
        context = cls(snapshot_id=snapshot_id, graphs=graphs)
        for item in _sequence_items(payload.get("symbols")):
            if not isinstance(item, Mapping):
                continue
            item_payload = cast("Mapping[str, Any]", item)
            canonical = _text_or_none(item_payload.get("canonical"))
            if not canonical:
                continue
            names = _string_items(item_payload.get("names"))
            display_name = _text_or_none(item_payload.get("display_name")) or (
                names[0] if names else None
            )
            qualified_name = _text_or_none(item_payload.get("qualified_name"))
            if qualified_name is None:
                qualified_name = next(
                    (name for name in names if name != display_name),
                    None,
                )
            aliases = {canonical}
            aliases.update(_string_items(item_payload.get("aliases")))
            for alias in aliases:
                context.canonical_by_alias[alias] = canonical

            context._register_name(display_name, canonical)
            context._register_name(qualified_name, canonical)
            for name in names:
                context._register_name(name, canonical)

            context.records_by_id[canonical] = {
                "symbol_id": canonical,
                "display_name": display_name,
                "kind": _text_or_none(item_payload.get("kind")),
                "path": _text_or_none(item_payload.get("path")),
                "start_line": _int_or_none(item_payload.get("start_line")),
            }

        raw_records = payload.get("records")
        if isinstance(raw_records, Mapping):
            records_payload = cast("Mapping[str, Any]", raw_records)
            for key, record_value in records_payload.items():
                if not isinstance(record_value, Mapping):
                    continue
                record = cast("Mapping[str, Any]", record_value)
                record_symbol = _text_or_none(record.get("symbol_id")) or str(key)
                canonical = (
                    context.canonical_by_alias.get(record_symbol)
                    or context.canonical_by_alias.get(str(key))
                    or record_symbol
                )
                context._merge_record(canonical, record)

        return context

    def _register_name(self, name: str | None, canonical: str) -> None:
        if not name:
            return
        self.canonical_by_name.setdefault(name, canonical)
        self.canonical_by_lower_name.setdefault(name.lower(), canonical)

    def _merge_record(self, canonical: str, record: Mapping[str, Any]) -> None:
        existing = self.records_by_id.setdefault(
            canonical,
            {
                "symbol_id": canonical,
                "display_name": None,
                "kind": None,
                "path": None,
                "start_line": None,
            },
        )
        for key in ("display_name", "kind", "path"):
            value = _text_or_none(record.get(key))
            if value and not existing.get(key):
                existing[key] = value
        start_line = _int_or_none(record.get("start_line"))
        if start_line is not None and existing.get("start_line") is None:
            existing["start_line"] = start_line

    def resolve_unit_id(self, query: str) -> str | None:
        if query in self.canonical_by_alias:
            return self.canonical_by_alias[query]
        if query in self.records_by_id or _graphs_contain(self.graphs, query):
            return query
        if query in self.canonical_by_name:
            return self.canonical_by_name[query]
        return self.canonical_by_lower_name.get(query.lower())

    def format_node(self, unit_id: str) -> dict[str, Any]:
        record = self.records_by_id.get(unit_id)
        if record is None:
            return {
                "symbol_id": unit_id,
                "display_name": None,
                "kind": None,
                "path": None,
                "start_line": None,
            }
        return {
            "symbol_id": unit_id,
            "display_name": record.get("display_name"),
            "kind": record.get("kind"),
            "path": record.get("path"),
            "start_line": record.get("start_line"),
        }


class _CombinedGraph:
    def __init__(self, graphs: Any, graph_types: Iterable[str]) -> None:
        self._items: list[tuple[str, Any]] = []
        for graph_type in graph_types:
            attr = GRAPH_TYPE_MAP.get(graph_type)
            if not attr:
                continue
            graph = getattr(graphs, attr, None)
            if graph is not None:
                self._items.append((graph_type, graph))

    def number_of_nodes(self) -> int:
        total = 0
        for _, graph in self._items:
            count = getattr(graph, "number_of_nodes", None)
            if callable(count):
                count_value: Any = count()
                total += int(count_value)
        return total

    def __contains__(self, node: object) -> bool:
        return isinstance(node, str) and any(
            _graph_contains(graph, node) for _, graph in self._items
        )

    def successors(self, node: str) -> list[str]:
        return _unique_neighbors(
            neighbor
            for _, graph in self._items
            for neighbor in _graph_neighbors(graph, node, "successors")
        )

    def predecessors(self, node: str) -> list[str]:
        return _unique_neighbors(
            neighbor
            for _, graph in self._items
            for neighbor in _graph_neighbors(graph, node, "predecessors")
        )

    def undirected_neighbors(self, node: str) -> list[str]:
        return _unique_neighbors([*self.successors(node), *self.predecessors(node)])

    def edge_types(self, source: str, target: str) -> list[str]:
        edge_types: list[str] = []
        for graph_type, graph in self._items:
            has_edge = getattr(graph, "has_edge", None)
            if callable(has_edge) and bool(has_edge(source, target)):
                edge_types.append(graph_type)
        return edge_types

    def undirected_edge_types(self, source: str, target: str) -> list[str]:
        return sorted(
            {*self.edge_types(source, target), *self.edge_types(target, source)}
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sequence_items(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        return cast("Sequence[Any]", value)
    if isinstance(value, (set, frozenset)):
        return tuple(cast("Iterable[Any]", value))
    return ()


def _string_items(value: Any) -> list[str]:
    return [str(item) for item in _sequence_items(value) if item]


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _graph_contains(graph: Any, node: str) -> bool:
    try:
        return node in graph
    except TypeError:
        nodes = getattr(graph, "nodes", None)
        if callable(nodes):
            node_values: Any = nodes()
            return node in {str(item) for item in node_values}
    return False


def _graphs_contain(graphs: Any, node: str) -> bool:
    return any(
        _graph_contains(getattr(graphs, attr, None), node) for attr in _ALL_GRAPH_ATTRS
    )


def _graph_neighbors(graph: Any, node: str, method_name: str) -> list[str]:
    method = getattr(graph, method_name, None)
    if not callable(method):
        return []
    try:
        neighbor_values: Any = method(node)
        return [str(neighbor) for neighbor in neighbor_values]
    except Exception:
        return []


def _unique_neighbors(neighbors: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for neighbor in neighbors:
        neighbor_id = str(neighbor)
        if neighbor_id in seen:
            continue
        seen.add(neighbor_id)
        result.append(neighbor_id)
    return result


def _shortest_path(
    graph: _CombinedGraph,
    source: str,
    target: str,
    *,
    max_hops: int | None = None,
    undirected: bool = False,
) -> list[str] | None:
    if source == target:
        return [source]
    queue: deque[list[str]] = deque([[source]])
    visited: set[str] = {source}
    while queue:
        path = queue.popleft()
        if max_hops is not None and len(path) - 1 >= max_hops:
            continue
        node = path[-1]
        neighbors = (
            graph.undirected_neighbors(node) if undirected else graph.successors(node)
        )
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            next_path = [*path, neighbor]
            if neighbor == target:
                return next_path
            visited.add(neighbor)
            queue.append(next_path)
    return None


def _distances_within(
    graph: _CombinedGraph,
    seed: str,
    max_hops: int,
    *,
    direction: str,
) -> dict[str, int]:
    distances: dict[str, int] = {seed: 0}
    queue: deque[str] = deque([seed])
    while queue:
        node = queue.popleft()
        distance = distances[node]
        if distance >= max_hops:
            continue
        neighbors = (
            graph.predecessors(node) if direction == "in" else graph.successors(node)
        )
        for neighbor in neighbors:
            if neighbor in distances:
                continue
            distances[neighbor] = distance + 1
            queue.append(neighbor)
    return {node: distance for node, distance in distances.items() if node != seed}


def resolve_unit_id(query: str, snapshot: IRSnapshot | GraphToolContext) -> str | None:
    """Resolve a symbol query to a unit_id from snapshot units.

    Tries exact unit_id, then display_name, then qualified_name,
    then case-insensitive display_name.  Returns the first match or None.
    """
    if isinstance(snapshot, GraphToolContext):
        return snapshot.resolve_unit_id(query)

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


def format_path_node(
    unit_id: str, snapshot: IRSnapshot | GraphToolContext
) -> dict[str, Any]:
    """Build a metadata dict for a graph node."""
    if isinstance(snapshot, GraphToolContext):
        return snapshot.format_node(unit_id)

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
) -> nx.Graph[str] | nx.DiGraph[str]:
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

    increment_materialization_boundary(BOUNDARY_GRAPH_FULL_LOAD)
    increment_materialization_boundary(BOUNDARY_GRAPH_FULL_LOAD)
    graphs = IRGraphBuilder().build_graphs(snapshot)
    combined: nx.Graph[str] | nx.DiGraph[str] | None = None

    for gt in graph_types:
        attr = GRAPH_TYPE_MAP[gt]
        g: nx.DiGraph[str] = getattr(graphs, attr)
        if undirected:
            ug: nx.Graph[str] = g.to_undirected()
            g = ug  # type: ignore[assignment]
        combined = g.copy() if combined is None else nx.compose(combined, g)
    return (
        combined
        if combined is not None
        else (nx.Graph() if undirected else nx.DiGraph())
    )


def _compatibility_fallback(reason: str) -> dict[str, Any]:
    return {
        "compatibility_fallback": True,
        "degraded_reason": reason,
        "materialization": {
            "snapshot_full_load": True,
            "graph_full_rebuild": True,
        },
    }


def _with_compatibility_fallback(
    result: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {**result, **_compatibility_fallback(reason)}


# ---------------------------------------------------------------------------
# Compute functions (one per MCP tool)
# ---------------------------------------------------------------------------


def compute_directed_path(
    from_symbol: str,
    to_symbol: str,
    snapshot: IRSnapshot | GraphToolContext,
    max_hops: int = 5,
    graph_types: list[str] | None = None,
) -> dict[str, Any]:
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

    if isinstance(snapshot, GraphToolContext):
        combined_view = _CombinedGraph(snapshot.graphs, graph_types)
        if combined_view.number_of_nodes() == 0:
            return {
                "found": False,
                "path": [],
                "path_length": 0,
                "error": "Selected graph types are empty (no nodes or edges).",
            }
        if from_id not in combined_view or to_id not in combined_view:
            missing: list[str] = []
            if from_id not in combined_view:
                missing.append(from_symbol)
            if to_id not in combined_view:
                missing.append(to_symbol)
            return {
                "found": False,
                "path": [],
                "path_length": 0,
                "error": f"Symbol(s) not in graph: {missing}",
            }

        compact_path = _shortest_path(
            combined_view,
            from_id,
            to_id,
            max_hops=max_hops + 1,
        )
        if compact_path is None:
            reverse_path = _shortest_path(
                combined_view,
                to_id,
                from_id,
                max_hops=max_hops + 1,
            )
            if reverse_path is not None:
                return {
                    "found": False,
                    "path": [],
                    "path_length": 0,
                    "error": (
                        f"No directed path from '{from_symbol}' to '{to_symbol}'. "
                        f"A reverse path exists (from '{to_symbol}' to '{from_symbol}')."
                    ),
                }
            return {
                "found": False,
                "path": [],
                "path_length": 0,
                "error": f"No directed path between '{from_symbol}' and '{to_symbol}' within max_hops={max_hops}.",
            }
        if len(compact_path) - 1 > max_hops:
            return {
                "found": False,
                "path": [],
                "path_length": len(compact_path) - 1,
                "error": f"Shortest path length {len(compact_path) - 1} exceeds max_hops={max_hops}.",
            }
        return {
            "found": True,
            "path": [format_path_node(nid, snapshot) for nid in compact_path],
            "path_length": len(compact_path) - 1,
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
        missing: list[str] = []
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
        path: list[str] = list(_NXShortestPath(combined, source=from_id, target=to_id))
    except nx.NetworkXNoPath:
        try:
            list(_NXShortestPath(combined, source=to_id, target=from_id))
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
    snapshot: IRSnapshot | GraphToolContext,
    max_hops: int = 3,
    graph_types: list[str] | None = None,
) -> dict[str, Any]:
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

    if isinstance(snapshot, GraphToolContext):
        combined_view = _CombinedGraph(snapshot.graphs, graph_types)
        if combined_view.number_of_nodes() == 0:
            return {
                "affected": [],
                "total_count": 0,
                "error": "Selected graph types are empty (no nodes or edges).",
            }
        if unit_id not in combined_view:
            return {
                "affected": [],
                "total_count": 0,
                "error": f"Symbol not in graph: {symbol}",
            }

        visited = _distances_within(
            combined_view,
            unit_id,
            max_hops,
            direction="in",
        )
        affected: list[dict[str, Any]] = []
        for nid, dist in sorted(visited.items(), key=lambda kv: kv[1]):
            node_info = format_path_node(nid, snapshot)
            node_info["distance"] = dist
            edge_types: set[str] = set()
            for successor in combined_view.successors(nid):
                if successor in visited or successor == unit_id:
                    edge_types.update(combined_view.edge_types(nid, successor))
            node_info["edge_types"] = sorted(edge_types)
            affected.append(node_info)

        return {"affected": affected, "total_count": len(affected), "error": None}

    increment_materialization_boundary(BOUNDARY_GRAPH_FULL_LOAD)
    graphs = IRGraphBuilder().build_graphs(snapshot)
    combined: nx.DiGraph[str] | None = None
    for gt in graph_types:
        attr = GRAPH_TYPE_MAP[gt]
        g: nx.DiGraph[str] = getattr(graphs, attr)
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
                g: nx.DiGraph[str] = getattr(graphs, attr)
                if g.has_edge(pred, node):
                    edge_types_map.setdefault(pred, set()).add(gt)

    affected: list[dict[str, Any]] = []
    for nid, dist in sorted(visited.items(), key=lambda kv: kv[1]):
        if nid == unit_id:
            continue
        node_info: dict[str, Any] = format_path_node(nid, snapshot)
        node_info["distance"] = dist
        node_info["edge_types"] = sorted(edge_types_map.get(nid, set()))
        affected.append(node_info)

    return {"affected": affected, "total_count": len(affected), "error": None}


def compute_leiden_clusters(
    snapshot: IRSnapshot,
    fc: object | None = None,
) -> dict[str, Any]:
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
        from ..ir.projection import ProjectionScope

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
            from ..ir.projection import ProjectionScope

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
    snapshot: IRSnapshot | GraphToolContext,
) -> dict[str, Any]:
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

    terminal_ids: list[str] = []
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

    if isinstance(snapshot, GraphToolContext):
        combined_view = _CombinedGraph(snapshot.graphs, VALID_GRAPH_TYPES)
        if combined_view.number_of_nodes() == 0:
            return {
                "found": False,
                "nodes": [],
                "edges": [],
                "error": "Graph is empty (no nodes or edges).",
            }
        missing = [
            t
            for t, tid in zip(terminals, terminal_ids, strict=False)
            if tid not in combined_view
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

        tree_nodes: set[str] = {terminal_ids[0]}
        tree_edges: list[tuple[str, str]] = []
        for terminal_id in terminal_ids[1:]:
            best_path: list[str] | None = None
            for existing in sorted(tree_nodes):
                path = _shortest_path(
                    combined_view,
                    existing,
                    terminal_id,
                    undirected=True,
                )
                if path is None:
                    continue
                if best_path is None or len(path) < len(best_path):
                    best_path = path
            if best_path is None:
                return {
                    "found": False,
                    "nodes": [],
                    "edges": [],
                    "error": "No connecting path found for all terminals.",
                }
            tree_nodes.update(best_path)
            tree_edges.extend(pairwise(best_path))

        nodes = [format_path_node(node_id, snapshot) for node_id in sorted(tree_nodes)]
        seen_edges: set[tuple[str, str]] = set()
        edges: list[dict[str, str]] = []
        for source, target in tree_edges:
            edge_key = (min(source, target), max(source, target))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edge_types = combined_view.undirected_edge_types(source, target)
            edges.append(
                {
                    "from": source,
                    "to": target,
                    "type": "+".join(edge_types) if edge_types else "unknown",
                }
            )
        return {"found": True, "nodes": nodes, "edges": edges, "error": None}

    increment_materialization_boundary(BOUNDARY_GRAPH_FULL_LOAD)
    graphs = IRGraphBuilder().build_graphs(snapshot)
    undirected: nx.Graph[str] | None = None
    for attr_name in _ALL_GRAPH_ATTRS:
        g: nx.DiGraph[str] = getattr(graphs, attr_name)
        ug: nx.Graph[str] = g.to_undirected()
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
        steiner_g: Any = _NXSteinerTree(undirected, terminal_ids)
    except nx.NetworkXError as exc:
        return {
            "found": False,
            "nodes": [],
            "edges": [],
            "error": f"Steiner tree computation failed: {exc}",
        }

    terminal_set: set[str] = set(terminal_ids)
    changed = True
    while changed:
        changed = False
        leaves: list[str] = [
            n
            for n in steiner_g.nodes()
            if steiner_g.degree(n) == 1 and n not in terminal_set
        ]
        for leaf in leaves:
            steiner_g.remove_node(leaf)
            changed = True

    if steiner_g.number_of_nodes() == 0:
        return {
            "found": False,
            "nodes": [],
            "edges": [],
            "error": "Steiner tree is empty after pruning.",
        }

    nodes = [format_path_node(str(nid), snapshot) for nid in steiner_g.nodes()]

    edge_type_map = {
        "call_graph": "call",
        "dependency_graph": "dependency",
        "inheritance_graph": "inheritance",
        "reference_graph": "reference",
        "containment_graph": "containment",
    }
    seen_edges: set[tuple[str, str]] = set()
    edges: list[dict[str, str]] = []
    for u, v in steiner_g.edges():
        u_str, v_str = str(u), str(v)
        edge_key = (min(u_str, v_str), max(u_str, v_str))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edge_types: list[str] = []
        for attr_name, etype in edge_type_map.items():
            g: nx.DiGraph[str] = getattr(graphs, attr_name)
            if g.has_edge(u_str, v_str) or g.has_edge(v_str, u_str):
                edge_types.append(etype)
        edges.append(
            {
                "from": u_str,
                "to": v_str,
                "type": "+".join(edge_types) if edge_types else "unknown",
            },
        )

    return {"found": True, "nodes": nodes, "edges": edges, "error": None}


def compute_find_callers(
    symbol: str,
    snapshot: IRSnapshot | GraphToolContext,
    max_hops: int = 2,
) -> dict[str, Any]:
    """Find all symbols that call the given symbol (BFS on reversed call graph)."""
    unit_id = resolve_unit_id(symbol, snapshot)
    if not unit_id:
        return {"callers": [], "total_count": 0, "error": f"Symbol not found: {symbol}"}

    if isinstance(snapshot, GraphToolContext):
        call_view = _CombinedGraph(snapshot.graphs, ["call"])
        if call_view.number_of_nodes() == 0:
            return {"callers": [], "total_count": 0, "error": "Call graph is empty."}
        if unit_id not in call_view:
            return {
                "callers": [],
                "total_count": 0,
                "error": f"Symbol not in call graph: {symbol}",
            }
        visited = _distances_within(call_view, unit_id, max_hops, direction="in")
        callers: list[dict[str, Any]] = []
        for nid, dist in sorted(visited.items(), key=lambda kv: kv[1]):
            node_info = format_path_node(nid, snapshot)
            node_info["distance"] = dist
            callers.append(node_info)
        return {"callers": callers, "total_count": len(callers), "error": None}

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

    callers: list[dict[str, Any]] = []
    for nid, dist in sorted(visited.items(), key=lambda kv: kv[1]):
        if nid == unit_id:
            continue
        node_info: dict[str, Any] = format_path_node(nid, snapshot)
        node_info["distance"] = dist
        callers.append(node_info)

    return {"callers": callers, "total_count": len(callers), "error": None}


def load_graph_tool_context(fc: object, snapshot_id: str) -> GraphToolContext | None:
    snapshot_store = getattr(fc, "snapshot_store", None)
    if snapshot_store is None:
        return None
    load_graphs = getattr(snapshot_store, "load_ir_graphs", None)
    load_symbols = getattr(snapshot_store, "load_snapshot_symbol_index_payload", None)
    if not callable(load_graphs) or not callable(load_symbols):
        return None
    graphs = load_graphs(snapshot_id)
    if graphs is None:
        return None
    payload = load_symbols(snapshot_id)
    if not isinstance(payload, Mapping):
        return None
    payload_mapping = cast("Mapping[str, Any]", payload)
    return GraphToolContext.from_symbol_payload(
        snapshot_id=snapshot_id,
        graphs=graphs,
        payload=payload_mapping,
    )


def _load_snapshot(fc: object, snapshot_id: str) -> IRSnapshot | None:
    snapshot_store = getattr(fc, "snapshot_store", None)
    if snapshot_store is None:
        return None
    load_snapshot = getattr(snapshot_store, "load_snapshot", None)
    if not callable(load_snapshot):
        return None
    increment_materialization_boundary(BOUNDARY_SNAPSHOT_FULL_LOAD)
    snapshot = load_snapshot(snapshot_id)
    return cast("IRSnapshot | None", snapshot)


def compute_directed_path_for_snapshot(
    fc: object,
    from_symbol: str,
    to_symbol: str,
    snapshot_id: str,
    max_hops: int = 5,
    graph_types: list[str] | None = None,
) -> dict[str, Any]:
    context = load_graph_tool_context(fc, snapshot_id)
    if context is not None:
        return compute_directed_path(
            from_symbol,
            to_symbol,
            context,
            max_hops,
            graph_types,
        )
    snapshot = _load_snapshot(fc, snapshot_id)
    if not snapshot:
        return {
            "found": False,
            "path": [],
            "path_length": 0,
            "error": f"Snapshot not found: {snapshot_id}",
        }
    if graph_types is None:
        graph_types = ["call", "dependency"]
    return _with_compatibility_fallback(
        compute_directed_path(
            from_symbol,
            to_symbol,
            snapshot,
            max_hops,
            graph_types,
        ),
        "compact_graph_context_unavailable",
    )


def compute_impact_analysis_for_snapshot(
    fc: object,
    symbol: str,
    snapshot_id: str,
    max_hops: int = 3,
    graph_types: list[str] | None = None,
) -> dict[str, Any]:
    context = load_graph_tool_context(fc, snapshot_id)
    if context is not None:
        return compute_impact_analysis(symbol, context, max_hops, graph_types)
    snapshot = _load_snapshot(fc, snapshot_id)
    if not snapshot:
        return {
            "affected": [],
            "total_count": 0,
            "error": f"Snapshot not found: {snapshot_id}",
        }
    if graph_types is None:
        graph_types = ["call", "dependency"]
    return _with_compatibility_fallback(
        compute_impact_analysis(symbol, snapshot, max_hops, graph_types),
        "compact_graph_context_unavailable",
    )


def _compute_cached_leiden_clusters(
    fc: object,
    context: GraphToolContext,
) -> dict[str, Any] | None:
    projection_store = getattr(fc, "projection_store", None)
    if projection_store is None or not getattr(projection_store, "enabled", False):
        return None
    from ..ir.projection import ProjectionScope

    scope = ProjectionScope(
        scope_kind="full",
        snapshot_id=context.snapshot_id,
        scope_key="full",
    )
    cached_id = projection_store.find_cached_projection_id(scope, "default")
    if not cached_id:
        return None
    l1_data = projection_store.get_layer(cached_id, "L1")
    if not l1_data:
        return None
    return extract_cluster_data(l1_data, context)


def _snapshot_from_graph_tool_context(context: GraphToolContext) -> IRSnapshot:
    docs_by_path: dict[str, IRDocument] = {}
    symbols: list[IRSymbol] = []
    for unit_id, record in sorted(context.records_by_id.items()):
        path = _text_or_none(record.get("path")) or ""
        if path and path not in docs_by_path:
            docs_by_path[path] = IRDocument(
                doc_id=f"doc:{path}",
                path=path,
                language="",
                source_set={"symbol_index"},
            )
        symbols.append(
            IRSymbol(
                symbol_id=unit_id,
                external_symbol_id=None,
                path=path,
                display_name=_text_or_none(record.get("display_name")) or unit_id,
                kind=_text_or_none(record.get("kind")) or "symbol",
                language="",
                start_line=_int_or_none(record.get("start_line")),
                source_set={"symbol_index"},
            )
        )
    return IRSnapshot(
        repo_name="",
        snapshot_id=context.snapshot_id,
        documents=list(docs_by_path.values()),
        symbols=symbols,
        metadata={
            "mcp_projection_source": "compact_graph_context",
            "compact_graph_context": True,
        },
    )


def _build_leiden_clusters_from_context(
    fc: object,
    context: GraphToolContext,
) -> dict[str, Any] | None:
    projection_transformer = getattr(fc, "projection_transformer", None)
    if projection_transformer is None:
        return None
    try:
        from ..ir.projection import ProjectionScope

        scope = ProjectionScope(
            scope_kind="full",
            snapshot_id=context.snapshot_id,
            scope_key="full",
        )
        snapshot = _snapshot_from_graph_tool_context(context)
        result = projection_transformer.build(
            scope=scope,
            snapshot=snapshot,
            ir_graphs=context.graphs,
        )
        if result and result.l1:
            payload = extract_cluster_data(result.l1, context)
            payload["compact_graph_context"] = True
            return payload
    except Exception as exc:
        return {
            "clusters": [],
            "xrefs": [],
            "total_clusters": 0,
            "error": f"Failed to build compact projection: {exc}",
            "compact_graph_context": True,
        }
    return None


def compute_leiden_clusters_for_snapshot(
    fc: object,
    snapshot_id: str,
) -> dict[str, Any]:
    context = load_graph_tool_context(fc, snapshot_id)
    if context is not None:
        cached = _compute_cached_leiden_clusters(fc, context)
        if cached is not None:
            return cached
        compact = _build_leiden_clusters_from_context(fc, context)
        if compact is not None:
            return compact
    snapshot = _load_snapshot(fc, snapshot_id)
    if not snapshot:
        return {
            "clusters": [],
            "xrefs": [],
            "total_clusters": 0,
            "error": f"Snapshot not found: {snapshot_id}",
        }
    return _with_compatibility_fallback(
        compute_leiden_clusters(snapshot, fc),
        "compact_graph_context_unavailable",
    )


def compute_steiner_path_for_snapshot(
    fc: object,
    terminals: list[str],
    snapshot_id: str,
) -> dict[str, Any]:
    context = load_graph_tool_context(fc, snapshot_id)
    if context is not None:
        return compute_steiner_path(terminals, context)
    snapshot = _load_snapshot(fc, snapshot_id)
    if not snapshot:
        return {
            "found": False,
            "nodes": [],
            "edges": [],
            "error": f"Snapshot not found: {snapshot_id}",
        }
    return _with_compatibility_fallback(
        compute_steiner_path(terminals, snapshot),
        "compact_graph_context_unavailable",
    )


def compute_find_callers_for_snapshot(
    fc: object,
    symbol: str,
    snapshot_id: str,
    max_hops: int = 2,
) -> dict[str, Any]:
    context = load_graph_tool_context(fc, snapshot_id)
    if context is not None:
        return compute_find_callers(symbol, context, max_hops)
    snapshot = _load_snapshot(fc, snapshot_id)
    if not snapshot:
        return {
            "callers": [],
            "total_count": 0,
            "error": f"Snapshot not found: {snapshot_id}",
        }
    return _with_compatibility_fallback(
        compute_find_callers(symbol, snapshot, max_hops),
        "compact_graph_context_unavailable",
    )


# ---------------------------------------------------------------------------
# Shared data extraction
# ---------------------------------------------------------------------------


def extract_cluster_data(
    l1_data: dict[str, Any],
    snapshot: IRSnapshot | GraphToolContext,
) -> dict[str, Any]:
    """Extract structured cluster data from L1 projection data."""
    clusters: list[dict[str, Any]] = []
    xrefs: list[dict[str, Any]] = []

    content_extra = cast(
        "dict[str, Any]",
        l1_data.get("content_extra") or l1_data.get("content") or {},
    )
    sections = cast("list[dict[str, Any]]", content_extra.get("sections") or [])
    navigation = cast("list[dict[str, Any]]", content_extra.get("navigation") or [])

    relations = cast("dict[str, Any]", content_extra.get("relations") or {})
    xref_list = cast("list[dict[str, Any]]", relations.get("xref") or [])
    for xref in xref_list:
        xref_id: str = xref.get("id", "")
        parts = xref_id.split("->")
        if len(parts) == 2:
            xrefs.append(
                {
                    "from_cluster": parts[0],
                    "to_cluster": parts[1],
                    "weight": xref.get("confidence", 0),
                },
            )

    for i, section in enumerate(sections):
        cluster_info: dict[str, Any] = {
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
                rep_display = (
                    rep_ref.get("display_name")
                    or rep_ref.get("label")
                    or rep_ref.get("id")
                )
                if rep_display:
                    cluster_info["representative"] = rep_display
        clusters.append(cluster_info)

    return {
        "clusters": clusters,
        "xrefs": xrefs,
        "total_clusters": len(clusters),
        "error": None,
    }
