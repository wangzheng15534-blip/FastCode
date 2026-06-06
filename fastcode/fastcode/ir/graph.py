"""
Build graph materializations from canonical IR relations.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

import igraph as ig
import networkx as nx
from igraph._igraph import InternalError as IGraphInternalError

from fastcode.utils.materialization import (
    BOUNDARY_NETWORKX_CONVERSION,
    increment_materialization_boundary,
)

from .types import IRSnapshot


def _normalize_path_key(path: str | None) -> str:
    return str(path or "").replace("\\", "/").strip("./")


class IRGraphView:
    """Compact directed graph view backed by python-igraph."""

    STORAGE_VERSION = "ir_graph_view.v1"

    def __init__(
        self,
        *,
        nodes: Iterable[str] = (),
        edges: Iterable[tuple[str, str, dict[str, Any] | None]] = (),
    ) -> None:
        node_list: list[str] = []
        seen: set[str] = set()
        edge_list: list[tuple[str, str, dict[str, Any]]] = []
        for node in nodes:
            node_id = str(node)
            if node_id not in seen:
                seen.add(node_id)
                node_list.append(node_id)
        edge_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for src, dst, attrs in edges:
            src_id = str(src)
            dst_id = str(dst)
            for node_id in (src_id, dst_id):
                if node_id not in seen:
                    seen.add(node_id)
                    node_list.append(node_id)
            edge_by_pair[(src_id, dst_id)] = dict(attrs or {})
        edge_list.extend(
            (src, dst, attrs) for (src, dst), attrs in edge_by_pair.items()
        )

        self._graph: Any = ig.Graph(directed=True)
        self.graph: dict[str, Any] = {}
        self._node_names: tuple[str, ...] = tuple(node_list)
        self._name_to_index: dict[str, int] = {
            name: index for index, name in enumerate(self._node_names)
        }
        self._graph.add_vertices(node_list)
        if edge_list:
            self._graph.add_edges([(src, dst) for src, dst, _ in edge_list])
            for attr_name in sorted(
                {key for _, _, attrs in edge_list for key in attrs}
            ):
                self._graph.es[attr_name] = [
                    attrs.get(attr_name) for _, _, attrs in edge_list
                ]

    def _names(self) -> list[str]:
        return list(self._node_names)

    @classmethod
    def from_networkx(cls, graph: nx.Graph[str]) -> IRGraphView:
        increment_materialization_boundary(
            BOUNDARY_NETWORKX_CONVERSION,
            items=graph.number_of_nodes() + graph.number_of_edges(),
        )
        return cls(
            nodes=(str(node) for node in graph.nodes()),
            edges=(
                (str(src), str(dst), dict(attrs))
                for src, dst, attrs in graph.edges(data=True)
            ),
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> IRGraphView:
        raw_nodes = payload.get("nodes") or []
        raw_edges = payload.get("edges") or []
        edges: list[tuple[str, str, dict[str, Any] | None]] = []
        if isinstance(raw_edges, list):
            for edge in raw_edges:
                if not isinstance(edge, dict):
                    continue
                src = edge.get("source")
                dst = edge.get("target")
                if src is None or dst is None:
                    continue
                attrs = edge.get("attrs") if isinstance(edge.get("attrs"), dict) else {}
                edges.append((str(src), str(dst), attrs))
        return cls(nodes=(str(node) for node in raw_nodes), edges=edges)

    @classmethod
    def union(cls, graphs: Iterable[Any]) -> IRGraphView:
        nodes: set[str] = set()
        edges: list[tuple[str, str, dict[str, Any] | None]] = []
        for graph in graphs:
            if isinstance(graph, IRGraphView):
                nodes.update(graph.nodes())
                edges.extend(
                    (src, dst, attrs) for src, dst, attrs in graph.edges(data=True)
                )
            elif isinstance(graph, nx.Graph):
                nodes.update(str(node) for node in graph.nodes())
                edges.extend(
                    (str(src), str(dst), dict(attrs))
                    for src, dst, attrs in graph.edges(data=True)
                )
        return cls(nodes=nodes, edges=edges)

    def __contains__(self, node: object) -> bool:
        return isinstance(node, str) and node in self._name_to_index

    def __iter__(self) -> Any:
        return iter(self.nodes())

    def is_directed(self) -> bool:
        return True

    def is_multigraph(self) -> bool:
        return False

    @staticmethod
    def _normalized_mode(mode: str) -> str:
        return mode if mode in {"in", "out", "all"} else "out"

    def nodes(self, data: bool = False) -> list[Any]:
        names = self._names()
        if data:
            return [(name, {}) for name in names]
        return names

    def edges(self, data: bool = False) -> list[Any]:
        names = self._names()
        rows: list[Any] = []
        for edge in self._graph.es:
            src = str(names[edge.source])
            dst = str(names[edge.target])
            if data:
                rows.append((src, dst, dict(edge.attributes())))
            else:
                rows.append((src, dst))
        return rows

    def number_of_nodes(self) -> int:
        return int(self._graph.vcount())

    def number_of_edges(self) -> int:
        return int(self._graph.ecount())

    def has_edge(self, source: str, target: str) -> bool:
        try:
            self._graph.get_eid(str(source), str(target), directed=True)
        except (ValueError, IGraphInternalError):
            return False
        return True

    def predecessors(self, node: str) -> Any:
        vertex = self._name_to_index.get(str(node))
        if vertex is None:
            return iter(())
        names = self._names()
        return iter(
            str(names[index]) for index in self._graph.neighbors(vertex, mode="in")
        )

    def successors(self, node: str) -> Any:
        vertex = self._name_to_index.get(str(node))
        if vertex is None:
            return iter(())
        names = self._names()
        return iter(
            str(names[index]) for index in self._graph.neighbors(vertex, mode="out")
        )

    def neighbors(self, node: str, *, mode: str = "out") -> Any:
        """Iterate node neighbors without materializing a NetworkX graph."""
        vertex = self._name_to_index.get(str(node))
        if vertex is None:
            return iter(())
        names = self._names()
        normalized_mode = self._normalized_mode(mode)
        return iter(
            str(names[index])
            for index in self._graph.neighbors(vertex, mode=normalized_mode)
        )

    def undirected_neighbors(self, node: str) -> Any:
        return self.neighbors(node, mode="all")

    def degree(self, node: str | None = None, *, mode: str = "all") -> Any:
        normalized_mode = self._normalized_mode(mode)
        if node is not None:
            vertex = self._name_to_index.get(str(node))
            if vertex is None:
                return 0
            return int(self._graph.degree(vertex, mode=normalized_mode))
        return [
            (name, int(self._graph.degree(index, mode=normalized_mode)))
            for index, name in enumerate(self._node_names)
        ]

    def in_degree(self, node: str | None = None) -> Any:
        return self.degree(node, mode="in")

    def out_degree(self, node: str | None = None) -> Any:
        return self.degree(node, mode="out")

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        mode: str = "out",
        max_hops: int | None = None,
    ) -> list[str]:
        source_index = self._name_to_index.get(str(source))
        target_index = self._name_to_index.get(str(target))
        if source_index is None or target_index is None:
            return []
        if source_index == target_index:
            return [self._node_names[source_index]]
        normalized_mode = self._normalized_mode(mode)
        try:
            paths = self._graph.get_shortest_paths(
                source_index,
                to=target_index,
                mode=normalized_mode,
                output="vpath",
            )
        except IGraphInternalError:
            return []
        if not paths or not paths[0]:
            return []
        path = [self._node_names[int(index)] for index in paths[0]]
        if max_hops is not None and len(path) - 1 > max_hops:
            return []
        return path

    def component_stats(self, *, mode: str = "weak") -> dict[str, Any]:
        component_mode = "strong" if mode == "strong" else "weak"
        components = self._graph.connected_components(mode=component_mode)
        sizes = [int(size) for size in components.sizes()]
        return {
            "mode": component_mode,
            "component_count": len(sizes),
            "largest_component_size": max(sizes, default=0),
            "isolated_node_count": sum(1 for size in sizes if size == 1),
        }

    def to_undirected_view(self) -> IRGraphView:
        edges: list[tuple[str, str, dict[str, Any]]] = []
        for src, dst, attrs in self.edges(data=True):
            attrs_payload = dict(attrs)
            edges.append((src, dst, attrs_payload))
            edges.append((dst, src, attrs_payload))
        return IRGraphView(nodes=self.nodes(), edges=edges)

    def distances_within(
        self, seed: str, max_hops: int, *, mode: str = "out"
    ) -> dict[str, int]:
        seed_index = self._name_to_index.get(str(seed))
        if seed_index is None or max_hops <= 0:
            return {}
        mode = self._normalized_mode(mode)
        distances: dict[int, int] = {seed_index: 0}
        queue: list[int] = [seed_index]
        cursor = 0
        while cursor < len(queue):
            vertex = queue[cursor]
            cursor += 1
            distance = distances[vertex]
            if distance >= max_hops:
                continue
            for neighbor in self._graph.neighbors(vertex, mode=mode):
                neighbor_index = int(neighbor)
                if neighbor_index in distances:
                    continue
                distances[neighbor_index] = distance + 1
                queue.append(neighbor_index)
        return {
            self._node_names[index]: distance
            for index, distance in distances.items()
            if index != seed_index
        }

    def reachable_within(self, seed: str, max_hops: int) -> set[str]:
        seed_index = self._name_to_index.get(str(seed))
        if seed_index is None or max_hops <= 0:
            return set()
        try:
            indexes = self._graph.neighborhood(
                vertices=seed_index,
                order=int(max_hops),
                mode="all",
                mindist=1,
            )
        except TypeError:
            indexes = [
                index
                for index in self._graph.neighborhood(
                    vertices=seed_index,
                    order=int(max_hops),
                    mode="all",
                )
                if index != seed_index
            ]
        return {self._node_names[int(index)] for index in indexes}

    def to_networkx(self) -> nx.DiGraph[str]:
        increment_materialization_boundary(
            BOUNDARY_NETWORKX_CONVERSION,
            items=self.number_of_nodes() + self.number_of_edges(),
        )
        graph: nx.DiGraph[str] = nx.DiGraph()
        graph.add_nodes_from(self.nodes())
        graph.add_edges_from(self.edges(data=True))
        return graph

    def copy(self) -> nx.DiGraph[str]:
        return self.to_networkx()

    def to_undirected(self) -> nx.Graph[str]:
        return self.to_networkx().to_undirected()

    def to_payload(self) -> dict[str, Any]:
        return {
            "storage_version": self.STORAGE_VERSION,
            "nodes": self.nodes(),
            "edges": [
                {"source": src, "target": dst, "attrs": attrs}
                for src, dst, attrs in self.edges(data=True)
            ],
        }


@dataclass
class IRGraphs:
    dependency_graph: Any
    call_graph: Any
    inheritance_graph: Any
    reference_graph: Any
    containment_graph: Any

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            "dependency": {
                "nodes": self.dependency_graph.number_of_nodes(),
                "edges": self.dependency_graph.number_of_edges(),
            },
            "call": {
                "nodes": self.call_graph.number_of_nodes(),
                "edges": self.call_graph.number_of_edges(),
            },
            "inheritance": {
                "nodes": self.inheritance_graph.number_of_nodes(),
                "edges": self.inheritance_graph.number_of_edges(),
            },
            "reference": {
                "nodes": self.reference_graph.number_of_nodes(),
                "edges": self.reference_graph.number_of_edges(),
            },
            "containment": {
                "nodes": self.containment_graph.number_of_nodes(),
                "edges": self.containment_graph.number_of_edges(),
            },
        }


class IRGraphBuilder:
    _RELATION_GRAPH_KEYS: ClassVar[dict[str, str]] = {
        "import": "dependency_graph",
        "call": "call_graph",
        "inherit": "inheritance_graph",
        "ref": "reference_graph",
        "contain": "containment_graph",
    }

    def build_graphs(self, snapshot: IRSnapshot) -> IRGraphs:
        edge_rows: dict[str, list[tuple[str, str, dict[str, Any]]]] = {
            "import": [],
            "call": [],
            "inherit": [],
            "ref": [],
            "contain": [],
        }

        for relation in snapshot.relations:
            rows = edge_rows.get(relation.relation_type)
            if rows is None:
                continue
            rows.append(
                (
                    relation.src_unit_id,
                    relation.dst_unit_id,
                    {
                        "relation_id": relation.relation_id,
                        "source": relation.source,
                        "resolution_state": relation.resolution_state,
                        "metadata": relation.metadata,
                    },
                )
            )

        return IRGraphs(
            dependency_graph=IRGraphView(edges=edge_rows["import"]),
            call_graph=IRGraphView(edges=edge_rows["call"]),
            inheritance_graph=IRGraphView(edges=edge_rows["inherit"]),
            reference_graph=IRGraphView(edges=edge_rows["ref"]),
            containment_graph=IRGraphView(edges=edge_rows["contain"]),
        )

    @staticmethod
    def _graph_touches_units(graph: Any, unit_ids: set[str]) -> bool:
        if not unit_ids:
            return False
        for unit_id in unit_ids:
            try:
                if unit_id in graph and int(graph.degree(unit_id)) > 0:
                    return True
            except (TypeError, ValueError, AttributeError):
                continue
        return False

    def build_graph_delta(
        self,
        snapshot: IRSnapshot,
        *,
        previous_graphs: IRGraphs | None,
        changed_paths: Iterable[str],
        removed_paths: Iterable[str] = (),
        edge_change_threshold: int = 10000,
    ) -> tuple[IRGraphs, dict[str, Any]]:
        changed_path_keys = {_normalize_path_key(path) for path in changed_paths}
        removed_path_keys = {_normalize_path_key(path) for path in removed_paths}
        affected_paths = changed_path_keys | removed_path_keys
        all_graphs = sorted(self._RELATION_GRAPH_KEYS.values())
        if previous_graphs is None:
            return self.build_graphs(snapshot), {
                "mode": "full",
                "fallback_reason": "missing_previous_graphs",
                "reusable_graphs": [],
                "rebuilt_graphs": all_graphs,
            }
        if not affected_paths:
            return previous_graphs, {
                "mode": "delta",
                "changed_relation_count": 0,
                "affected_path_count": 0,
                "reusable_graphs": all_graphs,
                "rebuilt_graphs": [],
                "fallback_reason": None,
            }
        if removed_path_keys:
            return self.build_graphs(snapshot), {
                "mode": "full",
                "fallback_reason": "removed_paths_require_graph_rebuild",
                "changed_relation_count": 0,
                "affected_path_count": len(affected_paths),
                "removed_path_count": len(removed_path_keys),
                "reusable_graphs": [],
                "rebuilt_graphs": all_graphs,
            }

        unit_paths = {
            unit.unit_id: _normalize_path_key(unit.path) for unit in snapshot.units
        }
        affected_units = {
            unit_id for unit_id, path in unit_paths.items() if path in affected_paths
        }
        changed_relations = [
            relation
            for relation in snapshot.relations
            if relation.src_unit_id in affected_units
            or relation.dst_unit_id in affected_units
        ]
        if len(changed_relations) > edge_change_threshold:
            return self.build_graphs(snapshot), {
                "mode": "full",
                "fallback_reason": "edge_change_threshold_exceeded",
                "changed_relation_count": len(changed_relations),
                "reusable_graphs": [],
                "rebuilt_graphs": all_graphs,
            }

        changed_relation_types = {
            relation.relation_type for relation in changed_relations
        }
        for relation_type, graph_attr in self._RELATION_GRAPH_KEYS.items():
            if self._graph_touches_units(
                getattr(previous_graphs, graph_attr), affected_units
            ):
                changed_relation_types.add(relation_type)
        graph_payload: dict[str, Any] = {}
        reusable_graphs: list[str] = []
        rebuilt_graphs: list[str] = []
        for relation_type, graph_attr in self._RELATION_GRAPH_KEYS.items():
            if relation_type not in changed_relation_types:
                graph_payload[graph_attr] = getattr(previous_graphs, graph_attr)
                reusable_graphs.append(graph_attr)
                continue
            edges = [
                (
                    relation.src_unit_id,
                    relation.dst_unit_id,
                    {
                        "relation_id": relation.relation_id,
                        "source": relation.source,
                        "resolution_state": relation.resolution_state,
                        "metadata": relation.metadata,
                    },
                )
                for relation in snapshot.relations
                if relation.relation_type == relation_type
            ]
            graph_payload[graph_attr] = IRGraphView(edges=edges)
            rebuilt_graphs.append(graph_attr)

        return IRGraphs(
            dependency_graph=graph_payload["dependency_graph"],
            call_graph=graph_payload["call_graph"],
            inheritance_graph=graph_payload["inheritance_graph"],
            reference_graph=graph_payload["reference_graph"],
            containment_graph=graph_payload["containment_graph"],
        ), {
            "mode": "delta",
            "changed_relation_count": len(changed_relations),
            "affected_path_count": len(affected_paths),
            "reusable_graphs": sorted(reusable_graphs),
            "rebuilt_graphs": sorted(rebuilt_graphs),
            "fallback_reason": None,
        }
