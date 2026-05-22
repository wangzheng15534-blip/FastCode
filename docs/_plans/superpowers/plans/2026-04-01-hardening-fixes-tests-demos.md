# Hardening Fixes, Test Coverage, and Demo Updates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit pending hardening changes in logical order, fix identified issues, improve test coverage on low-coverage modules, and update demos to showcase new features.

**Architecture:** Five phases — (1) commit existing work as 4 logical commits, (2) fix the `save_scip_artifact_ref` ambiguous signature, (3) add targeted tests to raise coverage on `redo_worker`, `snapshot_store`, `scip_loader`, `projection_transform`, (4) update existing demos and add a new hardening demo, (5) commit all fixes/updates.

**Tech Stack:** Python 3.10, pytest, tempfile (SQLite backend for tests), networkx

---

## Phase 1: Commit Existing Changes (4 commits, dependency order)

### Task 1: Commit typed SCIP models and loader changes

**Files:**
- Create: `fastcode/scip_models.py`
- Modify: `fastcode/scip_loader.py`
- Modify: `fastcode/adapters/scip_to_ir.py`
- Test: `tests/test_scip_models.py`

- [ ] **Step 1: Stage and commit**

```bash
git add fastcode/scip_models.py fastcode/scip_loader.py fastcode/adapters/scip_to_ir.py tests/test_scip_models.py
git commit -m "$(cat <<'EOF'
feat: Add typed SCIP models (dataclass roundtrip) and language_hint fallback

SCIPIndex/SCIPDocument/SCIPSymbol/SCIPOccurrence/SCIPArtifactRef dataclasses
replace raw dicts throughout the pipeline. load_scip_artifact() now returns
SCIPIndex. scip_to_ir adapter accepts typed model and falls back to language_hint
when document-level language is missing.
EOF
)"
```

Expected: commit succeeds with 4 files.

### Task 2: Commit fencing tokens, redo task processing, and snapshot store hardening

**Files:**
- Create: `fastcode/redo_worker.py`
- Modify: `fastcode/snapshot_store.py`
- Test: `tests/test_snapshot_pipeline.py` (fencing token test)

- [ ] **Step 1: Stage and commit**

```bash
git add fastcode/redo_worker.py fastcode/snapshot_store.py tests/test_snapshot_pipeline.py
git commit -m "$(cat <<'EOF'
feat: Add fencing tokens on resource locks and redo task background worker

acquire_lock() returns Optional[int] fencing token instead of bool.
validate_fencing_token() checks token hasn't changed before persist writes.
RedoWorker daemon thread polls claim_redo_task() with FOR UPDATE SKIP LOCKED,
dispatches index_run_recovery tasks, exponential backoff on failure, marks dead
after max_attempts.
EOF
)"
```

Expected: commit succeeds with 3 files.

### Task 3: Commit pipeline wiring — git meta, lineage edges, call graph bridging

**Files:**
- Modify: `fastcode/main.py`
- Modify: `fastcode/terminus_publisher.py`
- Test: `tests/test_terminus_lineage_edges.py`

- [ ] **Step 1: Stage and commit**

```bash
git add fastcode/main.py fastcode/terminus_publisher.py tests/test_terminus_lineage_edges.py
git commit -m "$(cat <<'EOF'
feat: Wire git backbone metadata, cross-snapshot lineage, and call graph bridging

_build_git_meta() resolves commit parents via pygit2.
_previous_snapshot_symbol_versions() builds cross-snapshot symbol mapping.
TerminusDB publisher emits commit_parent and symbol_version_from edges.
AST call_graph edges bridged to IREdge via ast_element_id mapping.
Fencing token validated before persist; enriched redo payload for recovery.
EOF
)"
```

Expected: commit succeeds with 3 files.

### Task 4: Commit API endpoints, projection v2 schema, and graph API tests

**Files:**
- Modify: `api.py`
- Modify: `fastcode/projection_transform.py`
- Test: `tests/test_graph_api.py`
- Test: `tests/test_projection_v2_schema.py`

- [ ] **Step 1: Stage and commit**

```bash
git add api.py fastcode/projection_transform.py tests/test_graph_api.py tests/test_projection_v2_schema.py
git commit -m "$(cat <<'EOF'
feat: Add graph query API, projection v2 schema, and repo refs endpoint

GET /repos/{name}/refs, GET /symbols/find, GET /graph/callees|callers|dependencies,
POST /redo/process. Projection L1 gains relations_v2/related_code/related_memory.
L2 chunks gain version/layer/id/path/title/source/render/meta metadata fields.
Shutdown hook stops redo worker gracefully.
EOF
)"
```

Expected: commit succeeds with 5 files.

- [ ] **Step 2: Verify clean working tree**

```bash
git status
git log --oneline -6
```

Expected: clean working tree, 4 new commits on top of `8a4a8fa`.

---

## Phase 2: Fix Identified Issues

### Task 5: Fix `save_scip_artifact_ref` ambiguous signature

The current `snapshot_id: str | SCIPArtifactRef` union overload is confusing. Split into two clean paths: a typed overload that accepts `SCIPArtifactRef` directly, and keep the positional-args version with explicit `str` type.

**Files:**
- Modify: `fastcode/snapshot_store.py:474-530`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_snapshot_pipeline.py`:

```python
def test_save_scip_artifact_ref_accepts_typed_model():
    from fastcode.scip_models import SCIPArtifactRef
    with tempfile.TemporaryDirectory(prefix="fc_artifact_ref_") as tmp:
        store = SnapshotStore(tmp)
        ref = SCIPArtifactRef(
            snapshot_id="snap:repo:typed",
            indexer_name="scip-python",
            indexer_version="1.0",
            artifact_path="/tmp/index.scip.json",
            checksum="abc123",
            created_at="2026-01-01T00:00:00+00:00",
        )
        result = store.save_scip_artifact_ref(ref)
        assert result["snapshot_id"] == "snap:repo:typed"
        assert result["indexer_name"] == "scip-python"
        loaded = store.get_scip_artifact_ref("snap:repo:typed")
        assert loaded is not None
        assert loaded["checksum"] == "abc123"
```

- [ ] **Step 2: Run test to verify it passes (existing code already supports this)**

```bash
.venv/bin/python -m pytest tests/test_snapshot_pipeline.py::test_save_scip_artifact_ref_accepts_typed_model -v
```

Expected: PASS (the existing `isinstance(snapshot_id, SCIPArtifactRef)` check handles this).

- [ ] **Step 3: Refactor the signature for clarity**

In `fastcode/snapshot_store.py`, replace the current `save_scip_artifact_ref` method signature. The `isinstance` dispatch is correct but the signature is misleading. Change to:

```python
def save_scip_artifact_ref(
    self,
    snapshot_id: str,
    *,
    indexer_name: str = "unknown",
    indexer_version: Optional[str] = None,
    artifact_path: str = "",
    checksum: str = "",
) -> Dict[str, Any]:
```

Remove the `isinstance(snapshot_id, SCIPArtifactRef)` branch. The `SCIPArtifactRef` overload was only used in one call site (`main.py`) — update that call site to pass individual fields from the typed model instead.

In `fastcode/main.py`, find the call to `save_scip_artifact_ref` (around line 677-683) and change from:

```python
scip_artifact_ref = self.snapshot_store.save_scip_artifact_ref(
    snapshot_id=snapshot_id,
    indexer_name=...,
    indexer_version=...,
    artifact_path=preserved_path,
    checksum=digest.hexdigest(),
)
```

This is already using positional args, so no change needed in `main.py`. The method just needs the signature cleanup.

- [ ] **Step 4: Run all tests**

```bash
.venv/bin/python -m pytest tests/ -v --tb=short
```

Expected: all 27 tests pass.

---

## Phase 3: Improve Test Coverage

### Task 6: Add `redo_worker` unit tests (21% → 80%+)

The redo_worker has low coverage because `process_once_status`, `_dispatch_task`, and `stop` aren't tested in isolation. Test with a mock FastCode that provides a controllable snapshot_store.

**Files:**
- Test: `tests/test_redo_worker.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
"""
Tests for RedoWorker background task processing.
"""

import json
from unittest.mock import MagicMock, patch

from fastcode.redo_worker import RedoWorker


class _FakeFastCode:
    def __init__(self):
        self.snapshot_store = MagicMock()


def test_process_once_status_returns_none_when_no_tasks():
    fc = _FakeFastCode()
    fc.snapshot_store.claim_redo_task.return_value = None
    worker = RedoWorker(fc)
    assert worker.process_once_status() == "none"


def test_process_once_status_succeeds_on_valid_task():
    fc = _FakeFastCode()
    fc.snapshot_store.claim_redo_task.return_value = {
        "task_id": "redo_abc",
        "task_type": "index_run_recovery",
        "payload_json": json.dumps({"run_id": "run1", "source": "/tmp/repo"}),
    }
    fc.retry_index_run_recovery = MagicMock(return_value={"status": "published"})
    worker = RedoWorker(fc)
    assert worker.process_once_status() == "succeeded"
    fc.snapshot_store.mark_redo_task_done.assert_called_once_with("redo_abc")


def test_process_once_status_fails_and_marks_failed():
    fc = _FakeFastCode()
    fc.snapshot_store.claim_redo_task.return_value = {
        "task_id": "redo_xyz",
        "task_type": "index_run_recovery",
        "payload_json": json.dumps({"run_id": "run2", "source": "/tmp/repo"}),
    }
    fc.retry_index_run_recovery = MagicMock(side_effect=RuntimeError("boom"))
    worker = RedoWorker(fc)
    assert worker.process_once_status() == "failed"
    fc.snapshot_store.mark_redo_task_failed.assert_called_once()
    call_args = fc.snapshot_store.mark_redo_task_failed.call_args
    assert call_args[1]["task_id"] == "redo_xyz"
    assert "boom" in call_args[1]["error"]


def test_dispatch_task_raises_on_missing_run_id():
    fc = _FakeFastCode()
    worker = RedoWorker(fc)
    task = {"task_id": "redo_bad", "task_type": "index_run_recovery", "payload_json": "{}"}
    import pytest
    with pytest.raises(RuntimeError, match="missing run_id"):
        worker._dispatch_task(task)


def test_dispatch_task_raises_on_unsupported_type():
    fc = _FakeFastCode()
    worker = RedoWorker(fc)
    task = {"task_id": "redo_bad", "task_type": "unknown_type", "payload_json": "{}"}
    import pytest
    with pytest.raises(RuntimeError, match="unsupported redo task type"):
        worker._dispatch_task(task)


def test_stop_sets_event_and_joins():
    fc = _FakeFastCode()
    worker = RedoWorker(fc, poll_interval_seconds=1)
    worker._thread = MagicMock()
    worker.stop()
    assert worker._stop_event.is_set()
    worker._thread.join.assert_called_once_with(timeout=10)


def test_start_creates_daemon_thread():
    fc = _FakeFastCode()
    worker = RedoWorker(fc)
    worker.start()
    assert worker._thread is not None
    assert worker._thread.daemon is True
    assert worker._thread.name == "fastcode-redo-worker"
    worker.stop()


def test_start_idempotent_if_thread_alive():
    fc = _FakeFastCode()
    worker = RedoWorker(fc)
    worker.start()
    original_thread = worker._thread
    worker.start()  # should not create a second thread
    assert worker._thread is original_thread
    worker.stop()
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_redo_worker.py -v --tb=short
```

Expected: 8 tests PASS.

### Task 7: Add `snapshot_store` fencing token and redo task tests (39% → 60%+)

Focus on the new fencing token and redo task methods that have no SQLite-path tests.

**Files:**
- Test: `tests/test_snapshot_pipeline.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_snapshot_pipeline.py`:

```python
def test_fencing_token_increments_on_reacquire():
    with tempfile.TemporaryDirectory(prefix="fc_fence_") as tmp:
        store = SnapshotStore(tmp)
        token1 = store.acquire_lock("index:snap:repo:1", owner_id="run1", ttl_seconds=60)
        assert token1 == 1
        token2 = store.acquire_lock("index:snap:repo:1", owner_id="run2", ttl_seconds=60)
        assert token2 == 2
        assert store.validate_fencing_token("index:snap:repo:1", expected_token=2)
        assert not store.validate_fencing_token("index:snap:repo:1", expected_token=1)


def test_validate_fencing_token_returns_false_for_missing_lock():
    with tempfile.TemporaryDirectory(prefix="fc_fence_") as tmp:
        store = SnapshotStore(tmp)
        assert not store.validate_fencing_token("nonexistent:lock", expected_token=1)


def test_release_lock_removes_entry():
    with tempfile.TemporaryDirectory(prefix="fc_fence_") as tmp:
        store = SnapshotStore(tmp)
        store.acquire_lock("index:snap:repo:1", owner_id="run1", ttl_seconds=60)
        store.release_lock("index:snap:repo:1", owner_id="run1")
        # SQLite backend always returns 1 for acquire_lock, so just verify no error
        token = store.acquire_lock("index:snap:repo:1", owner_id="run2", ttl_seconds=60)
        assert token == 1  # SQLite returns 1 always


def test_enqueue_redo_task_returns_id():
    with tempfile.TemporaryDirectory(prefix="fc_redo_") as tmp:
        store = SnapshotStore(tmp)
        task_id = store.enqueue_redo_task(
            task_type="index_run_recovery",
            payload={"run_id": "run1", "source": "/tmp/repo"},
        )
        assert task_id.startswith("redo_")
        # SQLite backend just returns the ID without persisting
        assert len(task_id) > len("redo_")


def test_claim_redo_task_returns_none_on_sqlite():
    with tempfile.TemporaryDirectory(prefix="fc_redo_") as tmp:
        store = SnapshotStore(tmp)
        assert store.claim_redo_task() is None


def test_mark_redo_task_done_noop_on_sqlite():
    with tempfile.TemporaryDirectory(prefix="fc_redo_") as tmp:
        store = SnapshotStore(tmp)
        store.mark_redo_task_done("redo_fake")  # should not raise


def test_mark_redo_task_failed_noop_on_sqlite():
    with tempfile.TemporaryDirectory(prefix="fc_redo_") as tmp:
        store = SnapshotStore(tmp)
        store.mark_redo_task_failed(task_id="redo_fake", error="test error")
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_snapshot_pipeline.py -v --tb=short
```

Expected: all tests pass (existing 5 + new 6 = 11).

### Task 8: Add `projection_transform` L2 chunk metadata test (already exists, extend)

The projection_transform module has decent coverage but the new chunk metadata fields need explicit assertion.

**Files:**
- Test: `tests/test_projection_v2_schema.py` (extend)

- [ ] **Step 1: Write additional tests**

Append to `tests/test_projection_v2_schema.py`:

```python
def test_projection_l2_chunk_has_all_required_metadata():
    snapshot = _sample_snapshot()
    transformer = ProjectionTransformer(config={"projection": {"enable_leiden": False}})
    scope = ProjectionScope(scope_kind="snapshot", snapshot_id=snapshot.snapshot_id, scope_key="k3")
    result = transformer.build(scope=scope, snapshot=snapshot, ir_graphs=_sample_graphs())

    for chunk in result.chunks:
        assert chunk["version"] == "v1", f"chunk {chunk['chunk_id']} missing version"
        assert chunk["layer"] == "L2", f"chunk {chunk['chunk_id']} missing layer"
        assert "id" in chunk, f"chunk {chunk['chunk_id']} missing id"
        assert "path" in chunk, f"chunk {chunk['chunk_id']} missing path"
        assert "title" in chunk, f"chunk {chunk['chunk_id']} missing title"
        assert "source" in chunk, f"chunk {chunk['chunk_id']} missing source"
        assert "render" in chunk, f"chunk {chunk['chunk_id']} missing render"
        assert "meta" in chunk, f"chunk {chunk['chunk_id']} missing meta"


def test_projection_l1_relations_v2_has_confidence():
    snapshot = _sample_snapshot()
    transformer = ProjectionTransformer(config={"projection": {"enable_leiden": False}})
    scope = ProjectionScope(scope_kind="snapshot", snapshot_id=snapshot.snapshot_id, scope_key="k4")
    result = transformer.build(scope=scope, snapshot=snapshot, ir_graphs=_sample_graphs())

    l1_content = result.l1["content"]
    v2 = l1_content["relations_v2"]["xref"]
    for rel in v2:
        assert "id" in rel
        assert "type" in rel
        assert 0.0 <= rel["confidence"] <= 1.0
```

- [ ] **Step 2: Run tests**

```bash
.venv/bin/python -m pytest tests/test_projection_v2_schema.py -v --tb=short
```

Expected: 3 tests PASS.

### Task 9: Run full test suite and coverage report

- [ ] **Step 1: Run coverage**

```bash
.venv/bin/python -m pytest tests/ -v --tb=short --cov=fastcode --cov-report=term-missing 2>&1 | tail -80
```

Expected: All tests pass. Target modules improved: `redo_worker` 21%→80%+, `snapshot_store` 39%→55%+.

---

## Phase 4: Update Demos

### Task 10: Update `demo_ir_pipeline.py` — use typed SCIP model and call graph bridging

**Files:**
- Modify: `demos/demo_ir_pipeline.py`

- [ ] **Step 1: Update the demo**

Replace the SCIP section (around lines 77-110) to demonstrate typed model usage:

```python
from fastcode.scip_models import SCIPIndex, SCIPDocument, SCIPSymbol, SCIPOccurrence

# --- 3. Build SCIP IR (typed model) ---
scip_index = SCIPIndex(
    indexer_name="scip-python",
    indexer_version="0.5.0",
    documents=[
        SCIPDocument(
            path="app/auth.py",
            language="python",
            symbols=[
                SCIPSymbol(symbol="pkg app/auth.py AuthService.", name="AuthService",
                           kind="class", range=[10, 0, 50, 0]),
                SCIPSymbol(symbol="pkg app/auth.py AuthService.login().", name="login",
                           kind="method", range=[20, 4, 35, 0]),
            ],
            occurrences=[
                SCIPOccurrence(symbol="pkg app/auth.py AuthService.login().",
                               role="definition", range=[20, 4, 35, 0]),
                SCIPOccurrence(symbol="pkg app/auth.py AuthService.login().",
                               role="reference", range=[100, 0, 100, 5]),
            ],
        )
    ],
)
scip_snapshot = build_ir_from_scip(
    repo_name="demo-repo",
    snapshot_id="snap:demo:abc123",
    scip_index=scip_index,
)
print(f"\nSCIP IR (typed model): {len(scip_snapshot.documents)} docs, {len(scip_snapshot.symbols)} symbols, "
      f"{len(scip_snapshot.occurrences)} occurrences, {len(scip_snapshot.edges)} edges")
for s in scip_snapshot.symbols:
    print(f"  SCIP symbol: {s.symbol_id} ({s.source_priority})")
```

Also update the graph section (around line 130-140) to show call graph analysis:

```python
    # --- 6. Build graphs and traverse ---
    builder = IRGraphBuilder()
    graphs = builder.build_graphs(merged)
    print(f"\nGraphs built:")
    for name in [
        "dependency_graph", "call_graph", "inheritance_graph",
        "reference_graph", "containment_graph",
    ]:
        g = getattr(graphs, name)
        print(f"  {name}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # Demonstrate call graph traversal (NetworkX shortest path)
    if graphs.call_graph.number_of_nodes() > 0:
        start_node = list(graphs.call_graph.nodes)[0]
        import networkx as nx
        dist = nx.single_source_shortest_path_length(graphs.call_graph, start_node, cutoff=2)
        print(f"\nCall graph from '{start_node[:40]}...' (max 2 hops):")
        for node, d in dist.items():
            if node != start_node:
                print(f"  -> {node[:50]}... (distance {d})")
```

Add `import networkx as nx` at the top of the file.

- [ ] **Step 2: Run the demo**

```bash
.venv/bin/python -m demos.demo_ir_pipeline
```

Expected: prints AST IR, SCIP IR (typed model), merged IR, validation, graph stats, call graph traversal.

### Task 11: Update `demo_projection.py` — show v2 schema fields

**Files:**
- Modify: `demos/demo_projection.py`

- [ ] **Step 1: Update the demo output section**

Replace the chunk printing section (around lines 101-104):

```python
    print(f"\n=== Chunks: {len(result.chunks)} ===")
    for chunk in result.chunks:
        print(f"  {chunk['chunk_id']}: {chunk['kind']} | version={chunk.get('version')} "
              f"layer={chunk.get('layer')} title={chunk.get('title', '?')}")
        if chunk.get('meta'):
            print(f"    meta: {chunk['meta']}")
```

Replace the L1 printing section (around lines 95-97):

```python
    print(f"\n=== L1 (Navigation) ===")
    l1_content = result.l1.get("content", {})
    print(f"  relations keys: {list(l1_content.get('relations', {}).keys())}")
    print(f"  relations_v2 keys: {list(l1_content.get('relations_v2', {}).keys())}")
    v2_xrefs = l1_content.get("relations_v2", {}).get("xref", [])
    print(f"  relations_v2.xref count: {len(v2_xrefs)}")
    if v2_xrefs:
        print(f"  first xref: {v2_xrefs[0]}")
    print(f"  related_code: {len(l1_content.get('related_code', []))} refs")
```

- [ ] **Step 2: Run the demo**

```bash
.venv/bin/python -m demos.demo_projection
```

Expected: prints L0, L1 with relations_v2 and related_code, L2 index, chunks with version/layer/meta.

### Task 12: Create `demo_hardening.py` — fencing tokens, redo worker, lineage edges

**Files:**
- Create: `demos/demo_hardening.py`

- [ ] **Step 1: Write the demo**

```python
"""
Demo: PostgreSQL Hardening Features -- fencing tokens, redo worker, lineage edges.

Usage:
    cd /home/jacob/develop/FastCode
    python -m demos.demo_hardening

Shows:
    1. Fencing token acquisition and validation
    2. Redo worker lifecycle (start/stop/process)
    3. TerminusDB lineage edges (commit_parent, symbol_version_from)
    4. Graph API (callees, callers, dependencies via NetworkX)
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from fastcode.snapshot_store import SnapshotStore
from fastcode.terminus_publisher import TerminusPublisher
from fastcode.semantic_ir import IRDocument, IREdge, IRSnapshot, IRSymbol
from fastcode.ir_graph_builder import IRGraphBuilder
from fastcode.redo_worker import RedoWorker
from unittest.mock import MagicMock


def main():
    print("=" * 60)
    print("FastCode Hardening Features Demo")
    print("=" * 60)

    # --- 1. Fencing Tokens ---
    print("\n--- 1. Fencing Tokens ---")
    with tempfile.TemporaryDirectory(prefix="fc_hardening_") as tmp:
        store = SnapshotStore(tmp)
        token1 = store.acquire_lock("index:snap:repo:1", owner_id="run1", ttl_seconds=60)
        print(f"  Acquired lock: token={token1}")
        print(f"  Validate token={token1}: {store.validate_fencing_token('index:snap:repo:1', token1)}")
        print(f"  Validate stale token=999: {store.validate_fencing_token('index:snap:repo:1', 999)}")

        token2 = store.acquire_lock("index:snap:repo:1", owner_id="run2", ttl_seconds=60)
        print(f"  Re-acquired by run2: token={token2} (incremented)")
        print(f"  Old token={token1} now invalid: {not store.validate_fencing_token('index:snap:repo:1', token1)}")

    # --- 2. Redo Worker ---
    print("\n--- 2. Redo Worker ---")
    fake_fc = MagicMock()
    fake_fc.snapshot_store.claim_redo_task.return_value = None
    worker = RedoWorker(fake_fc, poll_interval_seconds=1)
    print(f"  Created worker: poll_interval={worker.poll_interval_seconds}s")
    worker.start()
    print(f"  Thread started: alive={worker._thread.is_alive()}, daemon={worker._thread.daemon}")
    result = worker.process_once_status()
    print(f"  process_once_status (no tasks): {result}")
    worker.stop()
    print(f"  Stopped: event set={worker._stop_event.is_set()}")

    # --- 3. Lineage Edges ---
    print("\n--- 3. TerminusDB Lineage Edges ---")
    publisher = TerminusPublisher({"terminus": {"endpoint": "http://localhost"}})
    payload = publisher.build_lineage_payload(
        snapshot={
            "repo_name": "my-repo",
            "snapshot_id": "snap:my-repo:c2",
            "branch": "main",
            "commit_id": "c2",
            "documents": [],
            "symbols": [
                {
                    "symbol_id": "sym:ext:1",
                    "external_symbol_id": "ext:sym:1",
                    "display_name": "authenticate",
                    "kind": "function",
                    "path": "auth.py",
                }
            ],
        },
        manifest={"manifest_id": "m2"},
        git_meta={"parent_commit_ids": ["c1"]},
        previous_snapshot_symbols={"ext:sym:1": "symbol:snap:my-repo:c1:sym:ext:1"},
    )
    edge_types = [e["type"] for e in payload["edges"]]
    print(f"  Edge types: {edge_types}")
    commit_parent_edges = [e for e in payload["edges"] if e["type"] == "commit_parent"]
    version_edges = [e for e in payload["edges"] if e["type"] == "symbol_version_from"]
    print(f"  commit_parent edges: {len(commit_parent_edges)}")
    print(f"  symbol_version_from edges: {len(version_edges)}")
    if version_edges:
        print(f"    {version_edges[0]['src']} -> {version_edges[0]['dst']}")

    # --- 4. Graph API ---
    print("\n--- 4. Graph API (NetworkX Traversal) ---")
    sym_a = IRSymbol(
        symbol_id="sym:auth", external_symbol_id=None, path="auth.py",
        display_name="authenticate", kind="function", language="python",
        start_line=1, source_priority=10, source_set={"ast"}, metadata={"source": "ast"},
    )
    sym_b = IRSymbol(
        symbol_id="sym:validate", external_symbol_id=None, path="auth.py",
        display_name="validate_token", kind="function", language="python",
        start_line=20, source_priority=10, source_set={"ast"}, metadata={"source": "ast"},
    )
    sym_c = IRSymbol(
        symbol_id="sym:db", external_symbol_id=None, path="db.py",
        display_name="get_connection", kind="function", language="python",
        start_line=1, source_priority=10, source_set={"ast"}, metadata={"source": "ast"},
    )
    snapshot = IRSnapshot(
        repo_name="demo", snapshot_id="snap:demo:graph",
        documents=[], symbols=[sym_a, sym_b, sym_c],
        edges=[
            IREdge(edge_id="e:1", src_id="sym:auth", dst_id="sym:validate",
                   edge_type="call", source="ast", confidence="heuristic"),
            IREdge(edge_id="e:2", src_id="sym:validate", dst_id="sym:db",
                   edge_type="call", source="ast", confidence="heuristic"),
        ],
    )
    graphs = IRGraphBuilder().build_graphs(snapshot)
    print(f"  Call graph: {graphs.call_graph.number_of_nodes()} nodes, {graphs.call_graph.number_of_edges()} edges")

    # Callees from sym:auth
    dist = nx.single_source_shortest_path_length(graphs.call_graph, "sym:auth", cutoff=2)
    callees = [{"symbol_id": n, "distance": d} for n, d in dist.items() if n != "sym:auth"]
    print(f"  Callees from authenticate (2 hops):")
    for c in callees:
        print(f"    -> {c['symbol_id']} (distance {c['distance']})")

    # Callers of sym:db
    rev = graphs.call_graph.reverse(copy=False)
    dist_rev = nx.single_source_shortest_path_length(rev, "sym:db", cutoff=2)
    callers = [{"symbol_id": n, "distance": d} for n, d in dist_rev.items() if n != "sym:db"]
    print(f"  Callers of get_connection (2 hops):")
    for c in callers:
        print(f"    <- {c['symbol_id']} (distance {c['distance']})")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo**

```bash
.venv/bin/python -m demos.demo_hardening
```

Expected: prints fencing token flow, redo worker lifecycle, lineage edges, graph traversal.

### Task 13: Update `demo_snapshot_lifecycle.py` — add fencing token and redo enqueue

**Files:**
- Modify: `demos/demo_snapshot_lifecycle.py`

- [ ] **Step 1: Add fencing token and redo task sections**

After the IR graphs section (line 94), add:

```python
        # 8. Fencing token on lock
        token = store.acquire_lock("index:snap:my-repo:aaa", owner_id=run1, ttl_seconds=60)
        print(f"Fencing token for snap:my-repo:aaa: {token}")
        print(f"Token valid: {store.validate_fencing_token('index:snap:my-repo:aaa', token)}")

        # 9. Enqueue redo task (SQLite returns ID without persisting)
        redo_id = store.enqueue_redo_task(
            task_type="index_run_recovery",
            payload={"run_id": run1, "source": "/tmp/repo"},
            error="simulated failure",
        )
        print(f"Enqueued redo task: {redo_id}")
```

- [ ] **Step 2: Run the demo**

```bash
.venv/bin/python -m demos.demo_snapshot_lifecycle
```

Expected: original output plus fencing token and redo task lines.

---

## Phase 5: Final Commit

### Task 14: Commit all fixes, tests, and demo updates

- [ ] **Step 1: Stage and commit**

```bash
git add fastcode/snapshot_store.py tests/test_redo_worker.py tests/test_snapshot_pipeline.py tests/test_projection_v2_schema.py demos/demo_ir_pipeline.py demos/demo_projection.py demos/demo_hardening.py demos/demo_snapshot_lifecycle.py
git commit -m "$(cat <<'EOF'
fix: Clean save_scip_artifact_ref signature, add hardening tests and demo updates

- Remove ambiguous str|SCIPArtifactRef union from save_scip_artifact_ref
- Add 8 redo_worker unit tests (dispatch, start/stop, failure handling)
- Add 6 snapshot_store tests (fencing token increment, validation, redo noop)
- Add 2 projection v2 schema tests (chunk metadata, relations_v2 confidence)
- Update demo_ir_pipeline to use typed SCIP models and call graph traversal
- Update demo_projection to show v2 schema fields (relations_v2, chunk meta)
- Create demo_hardening (fencing tokens, redo worker, lineage, graph API)
- Update demo_snapshot_lifecycle with fencing token and redo enqueue
EOF
)"
```

- [ ] **Step 2: Verify final state**

```bash
git status
git log --oneline -8
.venv/bin/python -m pytest tests/ -v --tb=short
```

Expected: clean working tree, 5 new commits (4 phase 1 + 1 phase 5), all tests pass.
