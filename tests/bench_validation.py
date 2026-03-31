"""
Baseline performance test: IR validation at scale.

Run: pytest tests/bench_validation.py -v --benchmark-only
"""

import pytest

from fastcode.ir_validators import validate_snapshot
from fastcode.semantic_ir import IREdge, IRDocument, IROccurrence, IRSnapshot, IRSymbol


def _make_snapshot(num_symbols: int, num_occurrences: int) -> IRSnapshot:
    """Generate a valid snapshot with controlled size."""
    doc = IRDocument(
        doc_id="doc:1", path="app.py", language="python", source_set={"ast"},
    )
    symbols = [
        IRSymbol(
            symbol_id=f"sym:{i}",
            external_symbol_id=None,
            path="app.py",
            display_name=f"f{i}",
            kind="function",
            language="python",
            start_line=i + 1,
            source_priority=10,
            source_set={"ast"},
            metadata={"source": "ast"},
        )
        for i in range(num_symbols)
    ]
    occurrences = [
        IROccurrence(
            occurrence_id=f"occ:{i}",
            symbol_id=f"sym:{i % num_symbols}",
            doc_id="doc:1",
            role="definition",
            start_line=i + 1,
            start_col=0,
            end_line=i + 2,
            end_col=0,
            source="ast",
            metadata={},
        )
        for i in range(num_occurrences)
    ]
    edges = [
        IREdge(
            edge_id=f"e:{i}",
            src_id="doc:1",
            dst_id=f"sym:{i % num_symbols}",
            edge_type="contain",
            source="ast",
            confidence="resolved",
        )
        for i in range(min(num_symbols, 50))
    ]
    return IRSnapshot(
        repo_name="r",
        snapshot_id="s",
        documents=[doc],
        symbols=symbols,
        occurrences=occurrences,
        edges=edges,
    )


@pytest.mark.parametrize("num_symbols,num_occurrences", [
    (10, 10),
    (100, 100),
    (1000, 1000),
    (1000, 5000),
])
def test_validation_throughput(num_symbols, num_occurrences, benchmark):
    """Benchmark validate_snapshot throughput."""
    snap = _make_snapshot(num_symbols, num_occurrences)

    # Verify correctness before benchmarking
    errors = validate_snapshot(snap)
    assert errors == []

    benchmark(validate_snapshot, snap)
