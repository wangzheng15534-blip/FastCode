"""
Build graph materializations from canonical IR relations.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .semantic_ir import IRSnapshot


@dataclass
class IRGraphs:
    dependency_graph: nx.DiGraph[str]
    call_graph: nx.DiGraph[str]
    inheritance_graph: nx.DiGraph[str]
    reference_graph: nx.DiGraph[str]
    containment_graph: nx.DiGraph[str]

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
        dep = nx.DiGraph()
        call = nx.DiGraph()
        inherit = nx.DiGraph()
        ref = nx.DiGraph()
        contain = nx.DiGraph()

        graph_by_relation = {
            "import": dep,
            "call": call,
            "inherit": inherit,
            "ref": ref,
            "contain": contain,
        }

        for relation in snapshot.relations:
            graph = graph_by_relation.get(relation.relation_type)
            if graph is None:
                continue
            graph.add_edge(
                relation.src_unit_id,
                relation.dst_unit_id,
                relation_id=relation.relation_id,
                source=relation.source,
                resolution_state=relation.resolution_state,
                metadata=relation.metadata,
            )

        return IRGraphs(
            dependency_graph=dep,
            call_graph=call,
            inheritance_graph=inherit,
            reference_graph=ref,
            containment_graph=contain,
        )
