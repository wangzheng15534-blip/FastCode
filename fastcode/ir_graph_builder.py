"""
Build graph materializations from canonical IR edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import networkx as nx

from .semantic_ir import IRSnapshot


@dataclass
class IRGraphs:
    dependency_graph: nx.DiGraph
    call_graph: nx.DiGraph
    inheritance_graph: nx.DiGraph
    reference_graph: nx.DiGraph
    containment_graph: nx.DiGraph

    def stats(self) -> Dict[str, Dict[str, int]]:
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
        dep = nx.DiGraph()
        call = nx.DiGraph()
        inherit = nx.DiGraph()
        ref = nx.DiGraph()
        contain = nx.DiGraph()

        graph_by_edge = {
            "import": dep,
            "call": call,
            "inherit": inherit,
            "ref": ref,
            "contain": contain,
        }

        for edge in snapshot.edges:
            graph = graph_by_edge.get(edge.edge_type)
            if graph is None:
                continue
            graph.add_edge(
                edge.src_id,
                edge.dst_id,
                edge_id=edge.edge_id,
                source=edge.source,
                confidence=edge.confidence,
                metadata=edge.metadata,
            )

        return IRGraphs(
            dependency_graph=dep,
            call_graph=call,
            inheritance_graph=inherit,
            reference_graph=ref,
            containment_graph=contain,
        )

