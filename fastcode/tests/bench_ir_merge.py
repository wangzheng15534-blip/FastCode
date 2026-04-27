"""
Baseline performance test: IR merge at scale.

Run: pytest tests/bench_ir_merge.py -v --tb=short --benchmark-only
"""

from typing import Any

import pytest

from fastcode.adapters.ast_to_ir import build_ir_from_ast
from fastcode.adapters.scip_to_ir import build_ir_from_scip
from fastcode.indexer import CodeElement
from fastcode.ir_merge import merge_ir

pytestmark = [pytest.mark.perf]


def _make_code_elements(count: int) -> list[CodeElement]:
    """Generate synthetic CodeElement objects."""
    elements = []
    for i in range(count):
        rel_path = f"mod_{i // 10}.py" if count > 10 else "single.py"
        elements.append(
            CodeElement(
                id=f"el_{i}",
                name=f"func_{i}",
                type="function",
                file_path=f"/repo/{rel_path}",
                relative_path=rel_path,
                language="python",
                start_line=i * 10 + 1,
                end_line=i * 10 + 9,
                code=f"def func_{i}(): pass",
                signature=f"func_{i}()",
                docstring=None,
                summary=f"Function func_{i}",
                metadata={"imports": [{"module": "os"}] if i % 5 == 0 else []},
            )
        )
    return elements


def _make_scip_index(doc_count: int, syms_per_doc: int) -> dict[str, Any]:
    """Generate synthetic SCIP payload."""
    docs = []
    for d in range(doc_count):
        path = f"mod_{d}.py"
        symbols = []
        occurrences = []
        for s in range(syms_per_doc):
            sym_str = f"pkg {path} func_{d}_{s}()."
            symbols.append(
                {"symbol": sym_str, "name": f"func_{d}_{s}", "kind": "function"}
            )
            occurrences.append(
                {
                    "symbol": sym_str,
                    "role": "reference",
                    "range": [s * 5, 0, s * 5 + 3, 0],
                }
            )
        docs.append(
            {
                "path": path,
                "language": "python",
                "symbols": symbols,
                "occurrences": occurrences,
            }
        )
    return {"indexer_name": "test", "indexer_version": "0.0.0", "documents": docs}


@pytest.mark.parametrize("element_count", [10, 100, 500, 1000])
def test_merge_throughput(element_count: int, benchmark: pytest.BenchmarkFixture):
    """Benchmark merge_ir throughput for varying AST sizes."""
    elements = _make_code_elements(element_count)
    scip = _make_scip_index(max(1, element_count // 10), 10)
    ast_snap = build_ir_from_ast("repo", "snap:bench", elements, "/repo")
    scip_snap = build_ir_from_scip("repo", "snap:bench", scip)

    # Verify correctness before benchmarking
    merged = merge_ir(ast_snap, scip_snap)
    assert merged.symbols
    assert merged.documents

    benchmark(merge_ir, ast_snap, scip_snap)


@pytest.mark.parametrize("element_count", [10, 100, 500])
def test_ast_adapter_throughput(element_count: int, benchmark: pytest.BenchmarkFixture):
    """Benchmark AST-to-IR adapter throughput."""
    elements = _make_code_elements(element_count)

    # Verify correctness before benchmarking
    snap = build_ir_from_ast("repo", "snap:bench", elements, "/repo")
    assert snap.symbols

    benchmark(build_ir_from_ast, "repo", "snap:bench", elements, "/repo")


@pytest.mark.parametrize(("doc_count", "syms_per_doc"), [(5, 20), (50, 20), (100, 10)])
def test_scip_adapter_throughput(
    doc_count: int, syms_per_doc: int, benchmark: pytest.BenchmarkFixture
):
    """Benchmark SCIP-to-IR adapter throughput."""
    scip = _make_scip_index(doc_count, syms_per_doc)

    # Verify correctness before benchmarking
    snap = build_ir_from_scip("repo", "snap:bench", scip)
    assert snap.symbols

    benchmark(build_ir_from_scip, "repo", "snap:bench", scip)
