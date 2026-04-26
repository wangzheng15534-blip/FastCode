"""Property-based tests for ir_graph_builder module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.ir_graph_builder import IRGraphBuilder, IRGraphs
from fastcode.semantic_ir import IREdge, IRSnapshot

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


def _snapshot(edges: list) -> None:
    return IRSnapshot(repo_name="repo", snapshot_id="snap:1", edges=edges)


edge_type_st = st.sampled_from(
    ["import", "call", "inherit", "ref", "contain", "unknown"]
)
node_id_st = st.text(alphabet="abc", min_size=1, max_size=3)


# --- Properties ---


@pytest.mark.property
class TestIRGraphBuilder:
    @pytest.mark.happy
    def test_empty_snapshot_empty_graphs(self):
        """HAPPY: empty snapshot produces empty graphs."""
        builder = IRGraphBuilder()
        graphs = builder.build_graphs(IRSnapshot(repo_name="r", snapshot_id="s"))
        assert graphs.dependency_graph.number_of_nodes() == 0
        assert graphs.call_graph.number_of_nodes() == 0

    @pytest.mark.happy
    def test_single_call_edge(self):
        """HAPPY: single call edge creates node pair in call graph."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "a", "b", "call")])
        graphs = builder.build_graphs(snap)
        assert graphs.call_graph.number_of_nodes() == 2
        assert graphs.call_graph.number_of_edges() == 1

    @pytest.mark.happy
    def test_import_edge_goes_to_dependency(self):
        """HAPPY: import edges populate dependency graph."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "a", "b", "import")])
        graphs = builder.build_graphs(snap)
        assert graphs.dependency_graph.number_of_edges() == 1

    @pytest.mark.happy
    def test_inherit_edge_goes_to_inheritance(self):
        """HAPPY: inherit edges populate inheritance graph."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "a", "b", "inherit")])
        graphs = builder.build_graphs(snap)
        assert graphs.inheritance_graph.number_of_edges() == 1

    @pytest.mark.happy
    def test_ref_edge_goes_to_reference(self):
        """HAPPY: ref edges populate reference graph."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "a", "b", "ref")])
        graphs = builder.build_graphs(snap)
        assert graphs.reference_graph.number_of_edges() == 1

    @pytest.mark.happy
    def test_contain_edge_goes_to_containment(self):
        """HAPPY: contain edges populate containment graph."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "a", "b", "contain")])
        graphs = builder.build_graphs(snap)
        assert graphs.containment_graph.number_of_edges() == 1

    @pytest.mark.edge
    def test_unknown_edge_type_skipped(self):
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
    def test_self_loop_edge(self):
        """EDGE: self-referencing edge (src == dst) doesn't crash."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "a", "a", "call")])
        graphs = builder.build_graphs(snap)
        assert graphs.call_graph.number_of_nodes() == 1
        assert graphs.call_graph.number_of_edges() == 1

    @pytest.mark.edge
    def test_duplicate_edges_deduped_in_graph(self):
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
    def test_mixed_edge_types_isolated_graphs(self):
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
    def test_large_node_count(self):
        """EDGE: many unique nodes handled without error."""
        builder = IRGraphBuilder()
        edges = [_edge(f"e{i}", f"n{i}", f"n{i + 1}", "call") for i in range(100)]
        snap = _snapshot(edges)
        graphs = builder.build_graphs(snap)
        assert graphs.call_graph.number_of_edges() == 100

    @pytest.mark.edge
    def test_empty_edge_fields(self):
        """EDGE: empty string src/dst doesn't crash."""
        builder = IRGraphBuilder()
        snap = _snapshot([_edge("e1", "", "b", "call")])
        graphs = builder.build_graphs(snap)
        assert graphs.call_graph.number_of_nodes() == 2

    @pytest.mark.edge
    def test_stats_populated_graphs(self):
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
    def test_builder_reuse(self):
        """EDGE: same builder instance used for multiple snapshots."""
        builder = IRGraphBuilder()
        snap1 = _snapshot([_edge("e1", "a", "b", "call")])
        snap2 = _snapshot([_edge("e2", "x", "y", "ref")])
        g1 = builder.build_graphs(snap1)
        g2 = builder.build_graphs(snap2)
        assert g1.call_graph.number_of_edges() == 1
        assert g2.reference_graph.number_of_edges() == 1

    @given(
        edge_type=edge_type_st,
        n_edges=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_edge_count_matches(self, edge_type: str, n_edges: int):
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


@pytest.mark.property
class TestIRGraphsStats:
    @pytest.mark.happy
    def test_stats_keys(self):
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

    @pytest.mark.happy
    def test_stats_empty_graphs(self):
        """HAPPY: stats on empty graphs returns zeros."""
        graphs = IRGraphBuilder().build_graphs(
            IRSnapshot(repo_name="r", snapshot_id="s")
        )
        stats = graphs.stats()
        for key in stats:
            assert stats[key]["nodes"] == 0
            assert stats[key]["edges"] == 0
