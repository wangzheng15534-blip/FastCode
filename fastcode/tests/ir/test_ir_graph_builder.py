"""Property-based tests for ir_graph_builder module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.ir.graph import IRGraphBuilder, IRGraphs, IRGraphView
from fastcode.ir.types import IRCodeUnit, IREdge, IRRelation, IRSnapshot

# --- Helpers ---


def _edge(edge_id: str, src: str, dst: str, edge_type: str = "call") -> IREdge:
    return IREdge(
        edge_id=edge_id,
        src_id=src,
        dst_id=dst,
        edge_type=edge_type,
        source="ast",
        confidence="resolved",
    )


def _snapshot(edges: list[IREdge]) -> IRSnapshot:
    return IRSnapshot(repo_name="repo", snapshot_id="snap:1", edges=edges)


def _unit(unit_id: str, path: str) -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=unit_id,
        kind="function",
        path=path,
        language="python",
        display_name=unit_id,
        source_set={"fc_structure"},
    )


def _relation(relation_id: str, src: str, dst: str, relation_type: str) -> IRRelation:
    return IRRelation(
        relation_id=relation_id,
        src_unit_id=src,
        dst_unit_id=dst,
        relation_type=relation_type,
        resolution_state="structural",
    )


edge_type_st = st.sampled_from(
    ["import", "call", "inherit", "ref", "contain", "unknown"]
)
node_id_st = st.text(alphabet="abc", min_size=1, max_size=3)


# --- Properties ---


class TestIRGraphBuilder:
    def test_empty_snapshot_empty_graphs_property(self):
        """HAPPY: empty snapshot produces empty graphs."""
        builder = IRGraphBuilder()
        graphs = builder.build_graphs(IRSnapshot(repo_name="r", snapshot_id="s"))
        assert graphs.dependency_graph.number_of_nodes() == 0
        assert graphs.call_graph.number_of_nodes() == 0

    def test_single_call_edge_property(self):
        """HAPPY: single call edge creates node pair in call graph."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "a", "b", "call")])
        graphs = builder.build_graphs(snap)
        assert graphs.call_graph.number_of_nodes() == 2
        assert graphs.call_graph.number_of_edges() == 1

    def test_ir_graph_view_reachable_within_respects_hop_cutoff(self):
        graph = IRGraphView(
            edges=[
                ("a", "b", {}),
                ("b", "c", {}),
                ("c", "d", {}),
            ]
        )

        assert graph.reachable_within("a", 0) == set()
        assert graph.reachable_within("a", 1) == {"b"}
        assert graph.reachable_within("a", 2) == {"b", "c"}

    def test_ir_graph_view_distances_within_respects_direction(self):
        graph = IRGraphView(
            edges=[
                ("a", "b", {}),
                ("b", "c", {}),
                ("x", "a", {}),
            ]
        )

        assert graph.distances_within("a", 2, mode="out") == {"b": 1, "c": 2}
        assert graph.distances_within("a", 2, mode="in") == {"x": 1}
        assert graph.distances_within("a", 1, mode="all") == {"b": 1, "x": 1}

    def test_ir_graph_view_shortest_path_is_native_and_bounded(self):
        graph = IRGraphView(
            edges=[
                ("a", "b", {}),
                ("b", "c", {}),
                ("x", "a", {}),
            ]
        )

        assert graph.shortest_path("a", "c") == ["a", "b", "c"]
        assert graph.shortest_path("c", "x", mode="all") == ["c", "b", "a", "x"]
        assert graph.shortest_path("a", "c", max_hops=1) == []
        assert graph.shortest_path("missing", "c") == []

    def test_ir_graph_view_degree_neighbors_component_stats_and_undirected_view(self):
        graph = IRGraphView(
            nodes=["isolated"],
            edges=[
                ("a", "b", {}),
                ("b", "c", {}),
                ("x", "a", {}),
            ],
        )

        assert list(graph.neighbors("a", mode="out")) == ["b"]
        assert list(graph.neighbors("a", mode="in")) == ["x"]
        assert set(graph.undirected_neighbors("a")) == {"b", "x"}
        assert graph.degree("a") == 2
        assert graph.in_degree("a") == 1
        assert graph.out_degree("a") == 1
        assert dict(graph.degree())["isolated"] == 0
        assert graph.component_stats() == {
            "mode": "weak",
            "component_count": 2,
            "largest_component_size": 4,
            "isolated_node_count": 1,
        }

        undirected = graph.to_undirected_view()

        assert isinstance(undirected, IRGraphView)
        assert undirected.has_edge("a", "b")
        assert undirected.has_edge("b", "a")

    def test_import_edge_goes_to_dependency_property(self):
        """HAPPY: import edges populate dependency graph."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "a", "b", "import")])
        graphs = builder.build_graphs(snap)
        assert graphs.dependency_graph.number_of_edges() == 1

    def test_inherit_edge_goes_to_inheritance_property(self):
        """HAPPY: inherit edges populate inheritance graph."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "a", "b", "inherit")])
        graphs = builder.build_graphs(snap)
        assert graphs.inheritance_graph.number_of_edges() == 1

    def test_ref_edge_goes_to_reference_property(self):
        """HAPPY: ref edges populate reference graph."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "a", "b", "ref")])
        graphs = builder.build_graphs(snap)
        assert graphs.reference_graph.number_of_edges() == 1

    def test_contain_edge_goes_to_containment_property(self):
        """HAPPY: contain edges populate containment graph."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "a", "b", "contain")])
        graphs = builder.build_graphs(snap)
        assert graphs.containment_graph.number_of_edges() == 1

    @pytest.mark.edge
    def test_unknown_edge_type_skipped_property(self):
        """EDGE: unknown edge types are silently skipped (line 67)."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "a", "b", "unknown_type")])
        graphs = builder.build_graphs(snap)
        for g in [
            graphs.dependency_graph,
            graphs.call_graph,
            graphs.inheritance_graph,
            graphs.reference_graph,
            graphs.containment_graph,
        ]:
            assert g.number_of_edges() == 0

    @pytest.mark.edge
    def test_self_loop_edge_property(self):
        """EDGE: self-referencing edge (src == dst) doesn't crash."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "a", "a", "call")])
        graphs = builder.build_graphs(snap)
        assert graphs.call_graph.number_of_nodes() == 1
        assert graphs.call_graph.number_of_edges() == 1

    @pytest.mark.edge
    def test_duplicate_edges_deduped_in_graph_property(self):
        """EDGE: duplicate edges produce single graph edge (DiGraph dedup)."""
        builder = IRGraphBuilder()
        snap = _snapshot(
            [
                _edge("e1", "a", "b", "call"),
                _edge("e2", "a", "b", "call"),
            ]
        )
        graphs = builder.build_graphs(snap)
        assert graphs.call_graph.number_of_edges() == 1

    @pytest.mark.edge
    def test_mixed_edge_types_isolated_graphs_property(self):
        """EDGE: different edge types only populate their respective graph."""
        builder = IRGraphBuilder()
        snap = _snapshot(
            [
                _edge("e1", "a", "b", "call"),
                _edge("e2", "c", "d", "import"),
            ]
        )
        graphs = builder.build_graphs(snap)
        assert graphs.call_graph.number_of_edges() == 1
        assert graphs.dependency_graph.number_of_edges() == 1
        assert graphs.inheritance_graph.number_of_edges() == 0

    @pytest.mark.edge
    def test_large_node_count_property(self):
        """EDGE: many unique nodes handled without error."""
        builder = IRGraphBuilder()
        edges = [_edge(f"e{i}", f"n{i}", f"n{i + 1}", "call") for i in range(100)]
        snap = _snapshot(edges)
        graphs = builder.build_graphs(snap)
        assert graphs.call_graph.number_of_edges() == 100

    @pytest.mark.edge
    def test_empty_edge_fields_property(self):
        """EDGE: empty string src/dst doesn't crash."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "", "b", "call")])
        graphs = builder.build_graphs(snap)
        assert graphs.call_graph.number_of_nodes() == 2

    @pytest.mark.edge
    def test_stats_populated_graphs_property(self):
        """EDGE: stats returns non-zero for populated graphs."""
        builder = IRGraphBuilder()
        snap = _snapshot(
            [
                _edge("e1", "a", "b", "call"),
                _edge("e2", "c", "d", "import"),
            ]
        )
        graphs = builder.build_graphs(snap)
        stats = graphs.stats()
        assert stats["call"]["edges"] == 1
        assert stats["dependency"]["edges"] == 1
        assert stats["call"]["nodes"] == 2

    @pytest.mark.edge
    def test_builder_reuse_property(self):
        """EDGE: same builder instance used for multiple snapshots."""
        builder = IRGraphBuilder()
        snap1 = _snapshot([_edge("e1", "a", "b", "call")])
        snap2 = _snapshot([_edge("e2", "x", "y", "ref")])
        g1 = builder.build_graphs(snap1)
        g2 = builder.build_graphs(snap2)
        assert g1.call_graph.number_of_edges() == 1
        assert g2.reference_graph.number_of_edges() == 1

    def test_build_graph_delta_reuses_unaffected_graph_families(self):
        builder = IRGraphBuilder()
        previous = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:prev",
            units=[_unit("unit:a", "pkg/a.py"), _unit("unit:b", "pkg/b.py")],
            relations=[
                _relation("rel:import", "unit:a", "unit:b", "import"),
                _relation("rel:call", "unit:a", "unit:b", "call"),
            ],
        )
        current = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:current",
            units=[_unit("unit:a", "pkg/a.py"), _unit("unit:b", "pkg/b.py")],
            relations=[_relation("rel:import", "unit:a", "unit:b", "import")],
        )
        previous_graphs = builder.build_graphs(previous)

        graphs, stats = builder.build_graph_delta(
            current,
            previous_graphs=previous_graphs,
            changed_paths=["pkg/a.py"],
        )

        assert stats["mode"] == "delta"
        assert "dependency_graph" in stats["rebuilt_graphs"]
        assert "call_graph" in stats["rebuilt_graphs"]
        assert "reference_graph" in stats["reusable_graphs"]
        assert graphs.reference_graph is previous_graphs.reference_graph
        assert graphs.call_graph is not previous_graphs.call_graph
        assert graphs.call_graph.number_of_edges() == 0

    def test_build_graph_delta_rebuilds_all_graphs_for_deleted_paths(self):
        builder = IRGraphBuilder()
        previous = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:prev",
            units=[_unit("unit:old", "pkg/old.py"), _unit("unit:b", "pkg/b.py")],
            relations=[_relation("rel:call", "unit:old", "unit:b", "call")],
        )
        current = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:current",
            units=[_unit("unit:b", "pkg/b.py")],
            relations=[],
        )
        previous_graphs = builder.build_graphs(previous)

        graphs, stats = builder.build_graph_delta(
            current,
            previous_graphs=previous_graphs,
            changed_paths=[],
            removed_paths=["pkg/old.py"],
        )

        assert stats["mode"] == "full"
        assert stats["fallback_reason"] == "removed_paths_require_graph_rebuild"
        assert graphs.call_graph is not previous_graphs.call_graph
        assert graphs.call_graph.number_of_edges() == 0

    @given(
        edge_type=edge_type_st,
        n_edges=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20)
    def test_edge_count_matches_property(self, edge_type: str, n_edges: int):
        """HAPPY: number of edges matches snapshot edges."""
        builder = IRGraphBuilder()
        edges = [
            _edge(f"e{i}", f"src{i}", f"dst{i}", edge_type) for i in range(n_edges)
        ]
        snap = _snapshot(edges)
        graphs = builder.build_graphs(snap)
        if edge_type == "import":
            assert graphs.dependency_graph.number_of_edges() == n_edges
        elif edge_type == "call":
            assert graphs.call_graph.number_of_edges() == n_edges
        elif edge_type == "inherit":
            assert graphs.inheritance_graph.number_of_edges() == n_edges
        elif edge_type == "ref":
            assert graphs.reference_graph.number_of_edges() == n_edges
        elif edge_type == "contain":
            assert graphs.containment_graph.number_of_edges() == n_edges


class TestIRGraphsStats:
    def test_stats_keys_property(self):
        """HAPPY: stats returns dict with all graph types."""
        graphs = IRGraphs(
            dependency_graph=__import__("networkx").DiGraph(),
            call_graph=__import__("networkx").DiGraph(),
            inheritance_graph=__import__("networkx").DiGraph(),
            reference_graph=__import__("networkx").DiGraph(),
            containment_graph=__import__("networkx").DiGraph(),
        )
        stats = graphs.stats()
        for key in ("dependency", "call", "inheritance", "reference", "containment"):
            assert key in stats
            assert "nodes" in stats[key]
            assert "edges" in stats[key]

    def test_stats_empty_graphs_property(self):
        """HAPPY: stats on empty graphs returns zeros."""
        graphs = IRGraphBuilder().build_graphs(
            IRSnapshot(repo_name="r", snapshot_id="s")
        )
        stats = graphs.stats()
        for key in stats:
            assert stats[key]["nodes"] == 0
            assert stats[key]["edges"] == 0
