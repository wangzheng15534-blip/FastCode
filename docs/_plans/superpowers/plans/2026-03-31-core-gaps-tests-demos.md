# Core Gaps, Tests, and Demos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 4 remaining spec deviations, add baseline performance benchmarks, and create runnable demo scripts for the IR/projection/snapshot pipeline.

**Architecture:** Three independent workstreams — (A) spec gap fixes in IR merge/validators/adapters, (B) baseline performance benchmarks for merge/graph/projection, (C) standalone demo scripts exercising the pipeline end-to-end. Workstreams A and B are TDD; C is integration demonstration.

**Tech Stack:** Python 3.10+, pytest, pytest-benchmark, networkx, python-igraph, FastCode IR layer

---

## Workstream A: Spec Gap Fixes (4 items)

### Task 1: Occurrence deduplication in merge (Rule D)

**Files:**
- Modify: `fastcode/ir_merge.py:54-70`
- Test: `tests/test_ir_core.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_ir_core.py

def test_merge_deduplicates_occurrences_scip_wins():
    """Rule D: when AST and SCIP produce the same occurrence, SCIP wins."""
    doc = _doc("d1", "a.py")
    scip_occ = IROccurrence(
        occurrence_id="occ:scip:1",
        symbol_id="scip:snap:1:foo",
        doc_id="d1",
        role="definition",
        start_line=10,
        start_col=0,
        end_line=20,
        end_col=0,
        source="scip",
        metadata={"source": "scip"},
    )
    ast_occ = IROccurrence(
        occurrence_id="occ:ast:1",
        symbol_id="ast:s1",
        doc_id="d1",
        role="definition",
        start_line=10,
        start_col=0,
        end_line=20,
        end_col=0,
        source="ast",
        metadata={"source": "ast"},
    )
    ast_sym = IRSymbol(
        symbol_id="ast:s1", external_symbol_id=None, path="a.py",
        display_name="foo", kind="function", language="python",
        start_line=10, source_priority=10, source_set={"ast"},
        metadata={"source": "ast"},
    )
    scip_sym = IRSymbol(
        symbol_id="scip:snap:1:foo", external_symbol_id="foo", path="a.py",
        display_name="foo", kind="function", language="python",
        start_line=10, source_priority=100, source_set={"scip"},
        metadata={"source": "scip"},
    )
    ast = IRSnapshot(repo_name="r", snapshot_id="snap:1", documents=[doc], symbols=[ast_sym], occurrences=[ast_occ])
    scip = IRSnapshot(repo_name="r", snapshot_id="snap:1", documents=[doc], symbols=[scip_sym], occurrences=[scip_occ])
    merged = merge_ir(ast, scip)
    # After merge, both occurrences map to canonical symbol "scip:snap:1:foo"
    # They share the same (symbol_id, doc_id, role, start_line, start_col, end_line, end_col)
    # so only one should survive, and it should be the SCIP-sourced one
    assert len(merged.occurrences) == 1
    assert merged.occurrences[0].source == "scip"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ir_core.py::test_merge_deduplicates_occurrences_scip_wins -v`
Expected: FAIL — `merged.occurrences` will have 2 entries (both AST and SCIP)

- [ ] **Step 3: Write minimal implementation**

In `fastcode/ir_merge.py`, replace lines 54-70 (the occurrence merge block) with:

```python
    # Merge occurrences — deduplicate by (symbol_id, doc_id, role, range).
    # SCIP wins when both sources produce the same occurrence (Rule D).
    occ_seen: Dict[tuple, IROccurrence] = {}
    for occ in scip_snapshot.occurrences + ast_snapshot.occurrences:
        symbol_id = ast_to_canonical.get(occ.symbol_id, occ.symbol_id)
        key = (symbol_id, occ.doc_id, occ.role, occ.start_line, occ.start_col, occ.end_line, occ.end_col)
        if key not in occ_seen:
            occ_seen[key] = IROccurrence(
                occurrence_id=occ.occurrence_id,
                symbol_id=symbol_id,
                doc_id=occ.doc_id,
                role=occ.role,
                start_line=occ.start_line,
                start_col=occ.start_col,
                end_line=occ.end_line,
                end_col=occ.end_col,
                source=occ.source,
                metadata=occ.metadata,
            )
    merged_occurrences = list(occ_seen.values())
```

The key trick: iterate `scip_snapshot.occurrences` first so SCIP entries populate `occ_seen` before AST entries. AST entries with the same key are silently skipped.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ir_core.py::test_merge_deduplicates_occurrences_scip_wins -v`
Expected: PASS

- [ ] **Step 5: Run full IR test suite**

Run: `pytest tests/test_ir_core.py -v`
Expected: All 4 tests pass (existing 3 + new 1)

- [ ] **Step 6: Commit**

```bash
git add fastcode/ir_merge.py tests/test_ir_core.py
git commit -m "fix: Deduplicate occurrences in IR merge with SCIP-priority Rule D"
```

---

### Task 2: Fix AST symbol ID format to match spec

**Files:**
- Modify: `fastcode/adapters/ast_to_ir.py:22-26`
- Test: `tests/test_ir_core.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_ir_core.py

def test_ast_symbol_id_uses_qualified_name_and_start_col():
    """Spec: ast:{snapshot_id}:{language}:{file_path}:{kind}:{qualified_name}:{start_line}:{start_col}"""
    from fastcode.indexer import CodeElement
    from fastcode.adapters.ast_to_ir import _ast_symbol_id

    elem = CodeElement(
        id="el1", name="MyClass.my_method", type="method",
        language="python", relative_path="src/app.py",
        file_path="/repo/src/app.py",
        start_line=42, start_col=8, end_line=50, end_col=0,
        metadata={"qualified_name": "pkg.src.app.MyClass.my_method"},
    )
    sid = _ast_symbol_id("snap:abc", elem)
    # Must contain qualified_name value and start_col, not name and end_line
    assert "pkg.src.app.MyClass.my_method" in sid
    assert ":42:8:" in sid
    assert ":50:" not in sid  # end_line should not appear as position
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ir_core.py::test_ast_symbol_id_uses_qualified_name_and_start_col -v`
Expected: FAIL — current format uses `name` and `end_line`

- [ ] **Step 3: Fix the AST adapter symbol ID generation**

In `fastcode/adapters/ast_to_ir.py`, replace lines 22-26:

```python
def _ast_symbol_id(snapshot_id: str, elem: CodeElement) -> str:
    qualified = (elem.metadata or {}).get("qualified_name") or elem.name or "unknown"
    return (
        f"ast:{snapshot_id}:{elem.language or 'unknown'}:{elem.relative_path or ''}:"
        f"{elem.type or 'unknown'}:{qualified}:{int(elem.start_line or 0)}:{int(elem.start_col or 0)}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ir_core.py::test_ast_symbol_id_uses_qualified_name_and_start_col -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add fastcode/adapters/ast_to_ir.py tests/test_ir_core.py
git commit -m "fix: Align AST symbol ID format with spec (qualified_name, start_col)"
```

---

### Task 3: Add document path uniqueness validation

**Files:**
- Modify: `fastcode/ir_validators.py:23-24`
- Test: `tests/test_ir_core.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_ir_core.py

def test_validate_catches_duplicate_doc_paths():
    """Spec: document paths must be unique inside snapshot."""
    snap = IRSnapshot(
        repo_name="r",
        snapshot_id="s",
        documents=[
            IRDocument(doc_id="d1", path="a.py", language="python", source_set={"ast"}),
            IRDocument(doc_id="d2", path="a.py", language="python", source_set={"ast"}),
        ],
        symbols=[
            IRSymbol(
                symbol_id="s1", external_symbol_id=None, path="a.py",
                display_name="x", kind="function", language="python",
                source_priority=0, source_set={"ast"}, metadata={"source": "ast"},
            )
        ],
    )
    errors = validate_snapshot(snap)
    assert any("duplicate document path" in e for e in errors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ir_core.py::test_validate_catches_duplicate_doc_paths -v`
Expected: FAIL — no error about duplicate paths is produced

- [ ] **Step 3: Add path uniqueness check**

In `fastcode/ir_validators.py`, after line 24 (`errors.append("duplicate document IDs detected")`), add:

```python
    doc_paths = [d.path for d in snapshot.documents]
    if len(doc_paths) != len(set(doc_paths)):
        dupes = [p for p in set(doc_paths) if doc_paths.count(p) > 1]
        errors.append(f"duplicate document paths detected: {dupes}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ir_core.py::test_validate_catches_duplicate_doc_paths -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add fastcode/ir_validators.py tests/test_ir_core.py
git commit -m "fix: Add document path uniqueness validation to IR validator"
```

---

### Task 4: Add extractor field to SCIP edge metadata

**Files:**
- Modify: `fastcode/adapters/scip_to_ir.py:104-108, 148-154`
- Test: `tests/test_ir_core.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_ir_core.py

def test_scip_edges_include_extractor_field():
    """SCIP edges should carry extractor field for consistency with AST edges."""
    scip = {
        "indexer_name": "scip-python",
        "indexer_version": "0.1.0",
        "documents": [
            {
                "path": "a.py",
                "language": "python",
                "symbols": [{"symbol": "pkg a/Foo.", "name": "Foo", "kind": "class"}],
                "occurrences": [
                    {"symbol": "pkg a/Foo.", "role": "reference", "range": [5, 0, 5, 3]},
                ],
            }
        ],
    }
    snap = build_ir_from_scip(repo_name="r", snapshot_id="s:1", scip_index=scip)
    for edge in snap.edges:
        assert "extractor" in (edge.metadata or {}), f"missing extractor in {edge.edge_type} edge"
        assert edge.metadata["extractor"] == "fastcode.adapters.scip_to_ir"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ir_core.py::test_scip_edges_include_extractor_field -v`
Expected: FAIL — `extractor` key not in SCIP edge metadata

- [ ] **Step 3: Add extractor to SCIP contain edge metadata**

In `fastcode/adapters/scip_to_ir.py`, line 104, add the `extractor` key to the contain edge metadata dict:

```python
                    metadata={
                        "extractor": "fastcode.adapters.scip_to_ir",
                        "indexer_name": indexer_name,
                        "indexer_version": indexer_version,
                    },
```

- [ ] **Step 4: Add extractor to SCIP ref edge metadata**

In `fastcode/adapters/scip_to_ir.py`, line 148, add the `extractor` key to the ref edge metadata dict:

```python
                    metadata={
                        "extractor": "fastcode.adapters.scip_to_ir",
                        "role": role,
                        "occurrence_id": occ_id,
                        "indexer_name": indexer_name,
                        "indexer_version": indexer_version,
                    },
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_ir_core.py::test_scip_edges_include_extractor_field -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add fastcode/adapters/scip_to_ir.py tests/test_ir_core.py
git commit -m "fix: Add extractor provenance field to SCIP edge metadata"
```

---

## Workstream B: Baseline Performance Tests

### Task 5: IR merge performance benchmark

**Files:**
- Create: `tests/bench_ir_merge.py`

- [ ] **Step 1: Write the merge benchmark**

```python
"""
Baseline performance test: IR merge at scale.

Run: pytest tests/bench_ir_merge.py -v --tb=short --benchmark-only
"""

import time
import pytest
from fastcode.adapters.ast_to_ir import build_ir_from_ast
from fastcode.adapters.scip_to_ir import build_ir_from_scip
from fastcode.ir_merge import merge_ir
from fastcode.indexer import CodeElement


def _make_code_elements(count: int) -> list:
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
                summary=f"Function func_{i}",
                metadata={"imports": [{"module": "os"}] if i % 5 == 0 else []},
            )
        )
    return elements


def _make_scip_index(doc_count: int, syms_per_doc: int) -> dict:
    """Generate synthetic SCIP payload."""
    docs = []
    for d in range(doc_count):
        path = f"mod_{d}.py"
        symbols = []
        occurrences = []
        for s in range(syms_per_doc):
            sym_str = f"pkg {path} func_{d}_{s}()."
            symbols.append({"symbol": sym_str, "name": f"func_{d}_{s}", "kind": "function"})
            occurrences.append({"symbol": sym_str, "role": "reference", "range": [s * 5, 0, s * 5 + 3, 0]})
        docs.append({"path": path, "language": "python", "symbols": symbols, "occurrences": occurrences})
    return {"indexer_name": "test", "indexer_version": "0.0.0", "documents": docs}


@pytest.mark.parametrize("element_count", [10, 100, 500, 1000])
def test_merge_throughput(element_count, benchmark):
    """Benchmark merge_ir throughput for varying AST sizes."""
    elements = _make_code_elements(element_count)
    scip = _make_scip_index(max(1, element_count // 10), 10)
    ast_snap = build_ir_from_ast("repo", "snap:bench", elements, "/repo")
    scip_snap = build_ir_from_scip("repo", "snap:bench", scip)

    def bench():
        merge_ir(ast_snap, scip_snap)

    result = benchmark(bench)
    assert result.symbols
    assert result.documents


@pytest.mark.parametrize("element_count", [10, 100, 500])
def test_ast_adapter_throughput(element_count, benchmark):
    """Benchmark AST-to-IR adapter throughput."""
    elements = _make_code_elements(element_count)

    def bench():
        build_ir_from_ast("repo", "snap:bench", elements, "/repo")

    result = benchmark(bench)
    assert result.symbols


@pytest.mark.parametrize("doc_count,syms_per_doc", [(5, 20), (50, 20), (100, 10)])
def test_scip_adapter_throughput(doc_count, syms_per_doc, benchmark):
    """Benchmark SCIP-to-IR adapter throughput."""
    scip = _make_scip_index(doc_count, syms_per_doc)

    def bench():
        build_ir_from_scip("repo", "snap:bench", scip)

    result = benchmark(bench)
    assert result.symbols
```

- [ ] **Step 2: Install pytest-benchmark**

Run: `pip install pytest-benchmark 2>/dev/null; echo "done"`

- [ ] **Step 3: Run the benchmarks**

Run: `pytest tests/bench_ir_merge.py -v --benchmark-only 2>&1 | head -40`
Expected: Table of timings for each parameterized size

- [ ] **Step 4: Commit**

```bash
git add tests/bench_ir_merge.py
git commit -m "test: Add IR merge and adapter throughput benchmarks"
```

---

### Task 6: Graph builder and projection performance benchmarks

**Files:**
- Create: `tests/bench_graph_projection.py`

- [ ] **Step 1: Write the graph and projection benchmark**

```python
"""
Baseline performance test: IR graph builder and projection transform.

Run: pytest tests/bench_graph_projection.py -v --benchmark-only
"""

import pytest
from fastcode.ir_graph_builder import IRGraphBuilder
from fastcode.projection_models import ProjectionScope
from fastcode.projection_transform import ProjectionTransformer
from fastcode.semantic_ir import IRDocument, IREdge, IRSnapshot, IRSymbol


def _make_snapshot(num_symbols: int, edges_per_symbol: int = 2) -> IRSnapshot:
    """Generate a synthetic IRSnapshot with controlled size."""
    doc = IRDocument(doc_id="doc:1", path="app.py", language="python", source_set={"ast"})
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
        repo_name="repo", snapshot_id="snap:bench",
        documents=[doc], symbols=symbols, edges=edges,
    )


@pytest.mark.parametrize("num_symbols", [10, 100, 500, 1000])
def test_graph_builder_throughput(num_symbols, benchmark):
    """Benchmark IR graph materialization."""
    snap = _make_snapshot(num_symbols)
    builder = IRGraphBuilder()

    def bench():
        builder.build_graphs(snap)

    graphs = benchmark(bench)
    assert graphs.call_graph.number_of_nodes() == num_symbols


@pytest.mark.parametrize("num_symbols", [10, 100, 500])
def test_projection_transform_throughput(num_symbols, benchmark):
    """Benchmark projection transform (Leiden disabled, no LLM)."""
    snap = _make_snapshot(num_symbols)
    graphs = IRGraphBuilder().build_graphs(snap)
    transformer = ProjectionTransformer(config={"projection": {"enable_leiden": False, "llm_enabled": False}})
    scope = ProjectionScope(scope_kind="snapshot", snapshot_id="snap:bench", scope_key="k1")

    def bench():
        transformer.build(scope=scope, snapshot=snap, ir_graphs=graphs)

    result = benchmark(bench)
    assert result.l0["layer"] == "L0"
```

- [ ] **Step 2: Run the benchmarks**

Run: `pytest tests/bench_graph_projection.py -v --benchmark-only 2>&1 | head -40`
Expected: Table of timings

- [ ] **Step 3: Commit**

```bash
git add tests/bench_graph_projection.py
git commit -m "test: Add graph builder and projection transform throughput benchmarks"
```

---

### Task 7: Validation performance benchmark

**Files:**
- Create: `tests/bench_validation.py`

- [ ] **Step 1: Write the validation benchmark**

```python
"""
Baseline performance test: IR validation at scale.

Run: pytest tests/bench_validation.py -v --benchmark-only
"""

import pytest
from fastcode.ir_validators import validate_snapshot
from fastcode.semantic_ir import IRDocument, IREdge, IROccurrence, IRSnapshot, IRSymbol


def _make_snapshot(num_symbols: int, num_occurrences: int) -> IRSnapshot:
    """Generate a valid snapshot with controlled size."""
    doc = IRDocument(doc_id="doc:1", path="app.py", language="python", source_set={"ast"})
    symbols = [
        IRSymbol(
            symbol_id=f"sym:{i}", external_symbol_id=None, path="app.py",
            display_name=f"f{i}", kind="function", language="python",
            start_line=i + 1, source_priority=10, source_set={"ast"},
            metadata={"source": "ast"},
        )
        for i in range(num_symbols)
    ]
    occurrences = [
        IROccurrence(
            occurrence_id=f"occ:{i}", symbol_id=f"sym:{i % num_symbols}",
            doc_id="doc:1", role="definition",
            start_line=i + 1, start_col=0, end_line=i + 2, end_col=0,
            source="ast", metadata={},
        )
        for i in range(num_occurrences)
    ]
    edges = [
        IREdge(
            edge_id=f"e:{i}", src_id="doc:1", dst_id=f"sym:{i % num_symbols}",
            edge_type="contain", source="ast", confidence="resolved",
        )
        for i in range(min(num_symbols, 50))
    ]
    return IRSnapshot(
        repo_name="r", snapshot_id="s",
        documents=[doc], symbols=symbols, occurrences=occurrences, edges=edges,
    )


@pytest.mark.parametrize("num_symbols,num_occurrences", [
    (10, 10), (100, 100), (1000, 1000), (1000, 5000),
])
def test_validation_throughput(num_symbols, num_occurrences, benchmark):
    """Benchmark validate_snapshot throughput."""
    snap = _make_snapshot(num_symbols, num_occurrences)

    def bench():
        validate_snapshot(snap)

    errors = benchmark(bench)
    assert errors == []
```

- [ ] **Step 2: Run the benchmarks**

Run: `pytest tests/bench_validation.py -v --benchmark-only 2>&1 | head -30`
Expected: Table of timings

- [ ] **Step 3: Commit**

```bash
git add tests/bench_validation.py
git commit -m "test: Add IR validation throughput benchmarks"
```

---

## Workstream C: Demo Scripts

### Task 8: IR pipeline demo (AST + SCIP merge → graphs → validate)

**Files:**
- Create: `demos/demo_ir_pipeline.py`

- [ ] **Step 1: Write the demo script**

```python
"""
Demo: IR Pipeline — AST + SCIP merge, graph building, validation.

Usage:
    cd /home/jacob/develop/FastCode
    python -m demos.demo_ir_pipeline

Shows:
    1. Building IR from synthetic AST elements
    2. Building IR from a synthetic SCIP payload
    3. Merging both snapshots (SCIP wins on overlap)
    4. Validating the merged snapshot
    5. Building all 5 graph types from merged IR
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastcode.indexer import CodeElement
from fastcode.adapters.ast_to_ir import build_ir_from_ast
from fastcode.adapters.scip_to_ir import build_ir_from_scip
from fastcode.ir_merge import merge_ir
from fastcode.ir_validators import validate_snapshot
from fastcode.ir_graph_builder import IRGraphBuilder


def main():
    # --- 1. AST elements ---
    elements = [
        CodeElement(
            id="el_1", name="AuthService", type="class",
            file_path="/repo/app/auth.py", relative_path="app/auth.py",
            language="python", start_line=10, end_line=50,
            code="class AuthService: ...", summary="Authentication service",
            metadata={"imports": [{"module": "db"}], "bases": ["BaseService"]},
        ),
        CodeElement(
            id="el_2", name="login", type="function",
            file_path="/repo/app/auth.py", relative_path="app/auth.py",
            language="python", start_line=20, end_line=35,
            code="def login(): ...", summary="Login handler",
            metadata={"class_name": "AuthService"},
        ),
        CodeElement(
            id="el_3", name="BaseService", type="class",
            file_path="/repo/app/base.py", relative_path="app/base.py",
            language="python", start_line=1, end_line=20,
            code="class BaseService: ...", summary="Base service class",
            metadata={},
        ),
    ]

    # --- 2. Build AST IR ---
    ast_snapshot = build_ir_from_ast(
        repo_name="demo-repo",
        snapshot_id="snap:demo:abc123",
        elements=elements,
        repo_root="/repo",
    )
    print(f"AST IR: {len(ast_snapshot.documents)} docs, {len(ast_snapshot.symbols)} symbols, "
          f"{len(ast_snapshot.occurrences)} occurrences, {len(ast_snapshot.edges)} edges")
    for s in ast_snapshot.symbols:
        print(f"  AST symbol: {s.symbol_id} ({s.source_priority})")
    for e in ast_snapshot.edges:
        print(f"  AST edge: {e.edge_type} {e.src_id[:30]}... -> {e.dst_id[:30]}... [{e.source}/{e.confidence}]")

    # --- 3. Build SCIP IR ---
    scip_payload = {
        "indexer_name": "scip-python",
        "indexer_version": "0.5.0",
        "documents": [
            {
                "path": "app/auth.py",
                "language": "python",
                "symbols": [
                    {"symbol": "pkg app/auth.py AuthService.", "name": "AuthService", "kind": "class",
                     "range": [10, 0, 50, 0]},
                    {"symbol": "pkg app/auth.py AuthService.login().", "name": "login", "kind": "method",
                     "range": [20, 4, 35, 0]},
                ],
                "occurrences": [
                    {"symbol": "pkg app/auth.py AuthService.login().", "role": "definition",
                     "range": [20, 4, 35, 0]},
                    {"symbol": "pkg app/auth.py AuthService.login().", "role": "reference",
                     "range": [100, 0, 100, 5]},
                ],
            }
        ],
    }
    scip_snapshot = build_ir_from_scip(
        repo_name="demo-repo",
        snapshot_id="snap:demo:abc123",
        scip_index=scip_payload,
    )
    print(f"\nSCIP IR: {len(scip_snapshot.documents)} docs, {len(scip_snapshot.symbols)} symbols, "
          f"{len(scip_snapshot.occurrences)} occurrences, {len(scip_snapshot.edges)} edges")
    for s in scip_snapshot.symbols:
        print(f"  SCIP symbol: {s.symbol_id} ({s.source_priority})")
    for e in scip_snapshot.edges:
        print(f"  SCIP edge: {e.edge_type} {e.src_id[:30]}... -> {e.dst_id[:30]}... [{e.source}/{e.confidence}]")

    # --- 4. Merge ---
    merged = merge_ir(ast_snapshot, scip_snapshot)
    print(f"\nMerged IR: {len(merged.documents)} docs, {len(merged.symbols)} symbols, "
          f"{len(merged.occurrences)} occurrences, {len(merged.edges)} edges")
    for s in merged.symbols:
        aliases = s.metadata.get("aliases", [])
        alias_str = f" (aliases: {aliases})" if aliases else ""
        print(f"  Merged symbol: {s.symbol_id} source_set={s.source_set}{alias_str}")

    # --- 5. Validate ---
    errors = validate_snapshot(merged)
    if errors:
        print(f"\nValidation FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nValidation PASSED (no errors)")

    # --- 6. Build graphs ---
    builder = IRGraphBuilder()
    graphs = builder.build_graphs(merged)
    print(f"\nGraphs built:")
    for name in ["dependency_graph", "call_graph", "inheritance_graph", "reference_graph", "containment_graph"]:
        g = getattr(graphs, name)
        print(f"  {name}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create demos directory and run**

Run: `mkdir -p demos && python -m demos.demo_ir_pipeline`
Expected: Output showing AST IR, SCIP IR, merged IR, validation pass, and graph stats

- [ ] **Step 3: Commit**

```bash
git add demos/demo_ir_pipeline.py
git commit -m "feat: Add IR pipeline demo (AST+SCIP merge, validation, graph building)"
```

---

### Task 9: Projection demo (L0/L1/L2 generation)

**Files:**
- Create: `demos/demo_projection.py`

- [ ] **Step 1: Write the demo script**

```python
"""
Demo: Projection Transform — L0/L1/L2 generation from IR graph.

Usage:
    cd /home/jacob/develop/FastCode
    python -m demos.demo_projection

Shows:
    1. Building a multi-file IR snapshot
    2. Building IR graphs
    3. Generating L0 (summary), L1 (navigation), L2 (chunks) projections
    4. Printing each layer's structure
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastcode.semantic_ir import IRDocument, IREdge, IRSnapshot, IRSymbol
from fastcode.ir_graph_builder import IRGraphBuilder
from fastcode.projection_models import ProjectionScope
from fastcode.projection_transform import ProjectionTransformer


def _build_sample_snapshot() -> IRSnapshot:
    """Build a small multi-file, multi-symbol snapshot."""
    docs = [
        IRDocument(doc_id="doc:auth", path="app/auth.py", language="python", source_set={"ast"}),
        IRDocument(doc_id="doc:user", path="app/user.py", language="python", source_set={"ast"}),
        IRDocument(doc_id="doc:db", path="app/db.py", language="python", source_set={"ast"}),
    ]
    symbols = [
        IRSymbol(symbol_id="sym:auth_svc", external_symbol_id=None, path="app/auth.py",
                 display_name="AuthService", kind="class", language="python",
                 start_line=1, source_priority=10, source_set={"ast"}, metadata={"source": "ast"}),
        IRSymbol(symbol_id="sym:login", external_symbol_id=None, path="app/auth.py",
                 display_name="login", kind="function", language="python",
                 start_line=20, source_priority=10, source_set={"ast"}, metadata={"source": "ast"}),
        IRSymbol(symbol_id="sym:validate_token", external_symbol_id=None, path="app/auth.py",
                 display_name="validate_token", kind="function", language="python",
                 start_line=40, source_priority=10, source_set={"ast"}, metadata={"source": "ast"}),
        IRSymbol(symbol_id="sym:user_model", external_symbol_id=None, path="app/user.py",
                 display_name="UserModel", kind="class", language="python",
                 start_line=1, source_priority=10, source_set={"ast"}, metadata={"source": "ast"}),
        IRSymbol(symbol_id="sym:get_user", external_symbol_id=None, path="app/user.py",
                 display_name="get_user", kind="function", language="python",
                 start_line=10, source_priority=10, source_set={"ast"}, metadata={"source": "ast"}),
        IRSymbol(symbol_id="sym:db_conn", external_symbol_id=None, path="app/db.py",
                 display_name="get_connection", kind="function", language="python",
                 start_line=1, source_priority=10, source_set={"ast"}, metadata={"source": "ast"}),
    ]
    edges = [
        # Contain edges
        IREdge(edge_id="e:c1", src_id="doc:auth", dst_id="sym:auth_svc", edge_type="contain", source="ast", confidence="resolved"),
        IREdge(edge_id="e:c2", src_id="doc:auth", dst_id="sym:login", edge_type="contain", source="ast", confidence="resolved"),
        IREdge(edge_id="e:c3", src_id="doc:auth", dst_id="sym:validate_token", edge_type="contain", source="ast", confidence="resolved"),
        IREdge(edge_id="e:c4", src_id="doc:user", dst_id="sym:user_model", edge_type="contain", source="ast", confidence="resolved"),
        IREdge(edge_id="e:c5", src_id="doc:user", dst_id="sym:get_user", edge_type="contain", source="ast", confidence="resolved"),
        IREdge(edge_id="e:c6", src_id="doc:db", dst_id="sym:db_conn", edge_type="contain", source="ast", confidence="resolved"),
        # Call edges
        IREdge(edge_id="e:call1", src_id="sym:login", dst_id="sym:validate_token", edge_type="call", source="ast", confidence="heuristic"),
        IREdge(edge_id="e:call2", src_id="sym:login", dst_id="sym:get_user", edge_type="call", source="ast", confidence="heuristic"),
        IREdge(edge_id="e:call3", src_id="sym:get_user", dst_id="sym:db_conn", edge_type="call", source="ast", confidence="heuristic"),
        # Import edges
        IREdge(edge_id="e:imp1", src_id="doc:auth", dst_id="doc:user", edge_type="import", source="ast", confidence="heuristic"),
        IREdge(edge_id="e:imp2", src_id="doc:user", dst_id="doc:db", edge_type="import", source="ast", confidence="heuristic"),
    ]
    return IRSnapshot(
        repo_name="demo", snapshot_id="snap:demo:proj",
        documents=docs, symbols=symbols, edges=edges,
        metadata={"source_modes": ["ast"]},
    )


def main():
    snapshot = _build_sample_snapshot()
    print(f"Snapshot: {len(snapshot.documents)} docs, {len(snapshot.symbols)} symbols, {len(snapshot.edges)} edges")

    graphs = IRGraphBuilder().build_graphs(snapshot)
    print(f"Graphs: dep={graphs.dependency_graph.number_of_edges()}, "
          f"call={graphs.call_graph.number_of_edges()}, "
          f"contain={graphs.containment_graph.number_of_edges()}")

    config = {"projection": {"enable_leiden": True, "llm_enabled": False}}
    transformer = ProjectionTransformer(config=config)
    scope = ProjectionScope(scope_kind="snapshot", snapshot_id="snap:demo:proj", scope_key="demo_key")

    result = transformer.build(scope=scope, snapshot=snapshot, ir_graphs=graphs)

    print(f"\n=== L0 (Summary) ===")
    print(json.dumps(result.l0, indent=2, ensure_ascii=False)[:500])

    print(f"\n=== L1 (Navigation) ===")
    print(json.dumps(result.l1, indent=2, ensure_ascii=False)[:800])

    print(f"\n=== L2 Index ===")
    print(json.dumps(result.l2_index, indent=2, ensure_ascii=False)[:500])

    print(f"\n=== Chunks: {len(result.chunks)} ===")
    for chunk in result.chunks:
        print(f"  chunk {chunk.get('chunk_id', '?')}: {chunk.get('kind', '?')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo**

Run: `python -m demos.demo_projection`
Expected: Output showing L0 summary, L1 navigation structure, L2 index, and chunks

- [ ] **Step 3: Commit**

```bash
git add demos/demo_projection.py
git commit -m "feat: Add projection transform demo (L0/L1/L2 generation)"
```

---

### Task 10: Snapshot store demo (save/load/manifest lifecycle)

**Files:**
- Create: `demos/demo_snapshot_lifecycle.py`

- [ ] **Step 1: Write the demo script**

```python
"""
Demo: Snapshot Store Lifecycle — save, load, manifest publish, ref resolution.

Usage:
    cd /home/jacob/develop/FastCode
    python -m demos.demo_snapshot_lifecycle

Shows:
    1. Creating and saving an IRSnapshot
    2. Loading it back
    3. Publishing manifests for two snapshots on the same branch
    4. Verifying branch head tracks the latest
    5. Manifest previous-chain (supersession)
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastcode.semantic_ir import IRDocument, IREdge, IRSymbol, IRSnapshot
from fastcode.snapshot_store import SnapshotStore
from fastcode.manifest_store import ManifestStore
from fastcode.index_run import IndexRunStore
from fastcode.ir_graph_builder import IRGraphBuilder


def _make_snapshot(repo: str, snap_id: str, commit: str, branch: str) -> IRSnapshot:
    doc = IRDocument(doc_id="doc:1", path="main.py", language="python", source_set={"ast"})
    sym = IRSymbol(
        symbol_id=f"sym:{snap_id}:main", external_symbol_id=None, path="main.py",
        display_name="main", kind="function", language="python",
        start_line=1, source_priority=10, source_set={"ast"}, metadata={"source": "ast"},
    )
    edge = IREdge(
        edge_id=f"e:contain:{snap_id}", src_id="doc:1", dst_id=sym.symbol_id,
        edge_type="contain", source="ast", confidence="resolved",
    )
    return IRSnapshot(
        repo_name=repo, snapshot_id=snap_id, branch=branch, commit_id=commit,
        documents=[doc], symbols=[sym], edges=[edge],
        metadata={"source_modes": ["ast"]},
    )


def main():
    with tempfile.TemporaryDirectory(prefix="fc_demo_snap_") as tmp:
        store = SnapshotStore(tmp)
        manifest_store = ManifestStore(store.db_path)
        run_store = IndexRunStore(store.db_path)

        # 1. Save first snapshot
        snap1 = _make_snapshot("my-repo", "snap:my-repo:aaa", "aaa", "main")
        meta1 = store.save_snapshot(snap1, metadata={"run": 1})
        graphs1 = IRGraphBuilder().build_graphs(snap1)
        store.save_ir_graphs(snap1.snapshot_id, graphs1)
        print(f"Saved snapshot 1: {snap1.snapshot_id} -> {meta1['artifact_key']}")

        # 2. Load it back
        loaded1 = store.load_snapshot("snap:my-repo:aaa")
        assert loaded1 is not None
        print(f"Loaded snapshot 1: {loaded1.snapshot_id}, {len(loaded1.symbols)} symbols")

        # 3. Publish manifest for snapshot 1
        run1 = run_store.create_run("my-repo", "snap:my-repo:aaa", "main", "aaa")
        m1 = manifest_store.publish("my-repo", "main", "snap:my-repo:aaa", run1)
        print(f"Published manifest 1: {m1['manifest_id']}")

        # 4. Save and publish second snapshot (simulates new commit on main)
        snap2 = _make_snapshot("my-repo", "snap:my-repo:bbb", "bbb", "main")
        store.save_snapshot(snap2, metadata={"run": 2})
        run2 = run_store.create_run("my-repo", "snap:my-repo:bbb", "main", "bbb")
        m2 = manifest_store.publish("my-repo", "main", "snap:my-repo:bbb", run2)
        print(f"Published manifest 2: {m2['manifest_id']}")

        # 5. Verify branch head
        head = manifest_store.get_branch_manifest("my-repo", "main")
        assert head is not None
        assert head["snapshot_id"] == "snap:my-repo:bbb"
        assert head["previous_manifest_id"] == m1["manifest_id"]
        print(f"Branch head: {head['snapshot_id']} (previous: {head['previous_manifest_id']})")

        # 6. Test idempotent run creation
        run3 = run_store.create_run("my-repo", "snap:my-repo:bbb", "main", "bbb", idempotency_key="key1")
        run4 = run_store.create_run("my-repo", "snap:my-repo:bbb", "main", "bbb", idempotency_key="key1")
        assert run3 == run4
        print(f"Idempotent run: {run3} == {run4}")

        # 7. Load IR graphs
        loaded_graphs = store.load_ir_graphs("snap:my-repo:aaa")
        assert loaded_graphs is not None
        print(f"Loaded IR graphs: containment has {loaded_graphs.containment_graph.number_of_edges()} edges")

        print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo**

Run: `python -m demos.demo_snapshot_lifecycle`
Expected: Output showing save/load cycle, manifest publish, branch head tracking, idempotent runs, graph persistence

- [ ] **Step 3: Commit**

```bash
git add demos/demo_snapshot_lifecycle.py
git commit -m "feat: Add snapshot lifecycle demo (save/load/manifest/idempotency)"
```

---

## Self-Review Checklist

**1. Spec coverage:**

| Spec Gap | Task | Status |
|----------|------|--------|
| Occurrence dedup (Rule D) | Task 1 | Covered |
| AST symbol ID format | Task 2 | Covered |
| Doc path uniqueness validation | Task 3 | Covered |
| SCIP edge extractor field | Task 4 | Covered |
| Merge/adapter benchmarks | Task 5 | Covered |
| Graph/projection benchmarks | Task 6 | Covered |
| Validation benchmarks | Task 7 | Covered |
| IR pipeline demo | Task 8 | Covered |
| Projection demo | Task 9 | Covered |
| Snapshot lifecycle demo | Task 10 | Covered |

**2. Placeholder scan:** No TBD, TODO, or "similar to Task N" found. All code is complete.

**3. Type consistency:** All type names, method signatures, and file paths are consistent across tasks. Symbol ID format change in Task 2 propagates correctly.
