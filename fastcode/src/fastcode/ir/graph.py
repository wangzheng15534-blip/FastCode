"""
Build graph materializations from canonical IR relations.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import igraph as ig
import networkx as nx
from igraph._igraph import InternalError as IGraphInternalError

from fastcode.utils.materialization import (
    BOUNDARY_NETWORKX_CONVERSION,
    increment_materialization_boundary,
)

from .types import IRSnapshot


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
