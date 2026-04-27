"""
Baseline performance test: IR graph builder and projection transform.

Run: pytest tests/bench_graph_projection.py -v --benchmark-only
"""

import pytest

from fastcode.ir_graph_builder import IRGraphBuilder
from fastcode.projection_models import ProjectionScope
from fastcode.projection_transform import ProjectionTransformer
from fastcode.semantic_ir import IRDocument, IREdge, IRSnapshot, IRSymbol

pytestmark = [pytest.mark.perf]


def _make_snapshot(num_symbols: int, edges_per_symbol: int = 2) -> IRSnapshot:
    """Generate a synthetic IRSnapshot with controlled size."""
    doc = IRDocument(
        doc_id="doc:1", path="app.py", language="python", source_set={"ast"}
    )
    symbols = []
    edges = []
    for i in range(num_symbols):
        sid = f"sym:{i}"
        symbols.append(
            IRSymbol(
                symbol_id=sid,
                external_symbol_id=None,
                path="app.py",
                display_name=f"func_{i}",
                kind="function",
                language="python",
                start_line=i * 5 + 1,
                source_priority=10,
                source_set={"ast"},
                metadata={"source": "ast"},
            )
        )
        edges.append(
            IREdge(
                edge_id=f"e:contain:{i}",
                src_id="doc:1",
                dst_id=sid,
                edge_type="contain",
                source="ast",
                confidence="resolved",
            )
        )
        for j in range(edges_per_symbol):
            target = f"sym:{(i + j + 1) % num_symbols}"
            edges.append(
                IREdge(
                    edge_id=f"e:call:{i}_{j}",
                    src_id=sid,
                    dst_id=target,
                    edge_type="call",
                    source="ast",
                    confidence="heuristic",
                )
            )
    return IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:bench",
        documents=[doc],
        symbols=symbols,
        edges=edges,
    )


@pytest.mark.parametrize("num_symbols", [10, 100, 500, 1000])
def test_graph_builder_throughput(num_symbols: int, benchmark: pytest.BenchmarkFixture):
    """Benchmark IR graph materialization."""
    snap = _make_snapshot(num_symbols)
    builder = IRGraphBuilder()

    # Verify correctness before benchmarking
    graphs = builder.build_graphs(snap)
    assert graphs.call_graph.number_of_nodes() == num_symbols

    benchmark(builder.build_graphs, snap)


@pytest.mark.parametrize("num_symbols", [10, 100, 500])
def test_projection_transform_throughput(
    num_symbols: int, benchmark: pytest.BenchmarkFixture
):
    """Benchmark projection transform (Leiden disabled, no LLM)."""
    snap = _make_snapshot(num_symbols)
    graphs = IRGraphBuilder().build_graphs(snap)
    config = {"projection": {"enable_leiden": False, "llm_enabled": False}}
    transformer = ProjectionTransformer(config=config)
    scope = ProjectionScope(
        scope_kind="snapshot", snapshot_id="snap:bench", scope_key="k1"
    )

    # Verify correctness before benchmarking
    result = transformer.build(scope=scope, snapshot=snap, ir_graphs=graphs)
    assert result.l0["layer"] == "L0"

    benchmark(transformer.build, scope=scope, snapshot=snap, ir_graphs=graphs)
