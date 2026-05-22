# Hardening Gaps: Concurrency Safety + Store Typing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two highest-risk gaps: snapshot query concurrency safety (P0, silent wrong results under concurrent load) and store-boundary typing (P1, runtime KeyErrors on schema drift).

**Architecture:** (1) Make `query_snapshot()` load snapshot artifacts into request-local copies instead of mutating the shared `FastCode` instance. (2) Replace `dict[str, Any]` returns from `get_snapshot_record()` with a frozen `SnapshotRecord` dataclass, following the existing `ManifestRecord` pattern.

**Tech Stack:** Python 3.11, dataclasses, threading.Lock, pytest, hypothesis

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `fastcode/src/fastcode/store_records.py` | Add `SnapshotRecord` frozen dataclass |
| Modify | `fastcode/src/fastcode/snapshot_store.py` | Return `SnapshotRecord` from `get_snapshot_record()`, add `_row_to_snapshot_record()` |
| Modify | `fastcode/src/fastcode/pipeline.py` | Replace `record["artifact_key"]` with `record.artifact_key` at all call sites |
| Modify | `fastcode/src/fastcode/query_handler.py` | Replace `snapshot_record["artifact_key"]` with `record.artifact_key`; add request-local artifact loading |
| Modify | `fastcode/tests/test_store_records.py` | Add `SnapshotRecord` roundtrip tests |
| Modify | `fastcode/tests/test_snapshot_store.py` | Update existing tests for new return type |
| Modify | `fastcode/tests/test_query_handler.py` | Add concurrency safety tests |
| Create | `fastcode/tests/test_concurrency.py` | Thread-barrier concurrency regression tests |

---

## Task 1: Add `SnapshotRecord` Frozen Dataclass

**Files:**
- Modify: `fastcode/src/fastcode/store_records.py` (append after `SnapshotRefRecord`)

- [ ] **Step 1: Write the failing test**

```python
# fastcode/tests/test_store_records.py — add to existing file
def test_snapshot_record_roundtrip():
    from fastcode.store_records import SnapshotRecord

    record = SnapshotRecord(
        snapshot_id="snap:repo:abc",
        repo_name="repo",
        branch="main",
        commit_id="abc123",
        tree_id="tree1",
        artifact_key="snap_repo_abc",
        ir_path="/tmp/snap_repo_abc/ir_snapshot.json",
        ir_graphs_path="/tmp/snap_repo_abc/ir_graphs.json",
        created_at="2026-01-01T00:00:00",
        metadata_json='{"source_modes": ["scip"]}',
    )
    d = record.to_dict()
    assert d["snapshot_id"] == "snap:repo:abc"
    assert d["artifact_key"] == "snap_repo_abc"

    restored = SnapshotRecord.from_dict(d)
    assert restored == record


def test_snapshot_record_from_dict_handles_nulls():
    from fastcode.store_records import SnapshotRecord

    record = SnapshotRecord.from_dict({
        "snapshot_id": "snap:1",
        "repo_name": "r",
        "artifact_key": "snap_1",
        "ir_path": "/tmp/ir.json",
        "created_at": "2026-01-01",
    })
    assert record.branch is None
    assert record.commit_id is None
    assert record.tree_id is None
    assert record.ir_graphs_path is None
    assert record.metadata_json is None


def test_snapshot_record_is_frozen():
    from fastcode.store_records import SnapshotRecord

    record = SnapshotRecord(
        snapshot_id="snap:1",
        repo_name="r",
        artifact_key="snap_1",
        ir_path="/tmp/ir.json",
        created_at="2026-01-01",
    )
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.snapshot_id = "changed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest fastcode/tests/test_store_records.py -k snapshot_record -v`
Expected: FAIL — `ImportError: cannot import name 'SnapshotRecord'`

- [ ] **Step 3: Write minimal implementation**

Append to `fastcode/src/fastcode/store_records.py` after `SnapshotRefRecord`:

```python
@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    repo_name: str
    branch: str | None
    commit_id: str | None
    tree_id: str | None
    artifact_key: str
    ir_path: str
    ir_graphs_path: str | None
    created_at: str
    metadata_json: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "repo_name": self.repo_name,
            "branch": self.branch,
            "commit_id": self.commit_id,
            "tree_id": self.tree_id,
            "artifact_key": self.artifact_key,
            "ir_path": self.ir_path,
            "ir_graphs_path": self.ir_graphs_path,
            "created_at": self.created_at,
            "metadata_json": self.metadata_json,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotRecord:
        return cls(
            snapshot_id=str(data.get("snapshot_id") or ""),
            repo_name=str(data.get("repo_name") or ""),
            branch=str(data["branch"]) if data.get("branch") is not None else None,
            commit_id=(
                str(data["commit_id"]) if data.get("commit_id") is not None else None
            ),
            tree_id=str(data["tree_id"]) if data.get("tree_id") is not None else None,
            artifact_key=str(data.get("artifact_key") or ""),
            ir_path=str(data.get("ir_path") or ""),
            ir_graphs_path=(
                str(data["ir_graphs_path"])
                if data.get("ir_graphs_path") is not None
                else None
            ),
            created_at=str(data.get("created_at") or ""),
            metadata_json=(
                str(data["metadata_json"])
                if data.get("metadata_json") is not None
                else None
            ),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest fastcode/tests/test_store_records.py -k snapshot_record -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add fastcode/src/fastcode/store_records.py fastcode/tests/test_store_records.py
git commit -m "feat: add SnapshotRecord frozen dataclass for store-boundary typing"
```

---

## Task 2: Wire `SnapshotRecord` into `SnapshotStore`

**Files:**
- Modify: `fastcode/src/fastcode/snapshot_store.py:577-584` (change `get_snapshot_record` return type)
- Modify: `fastcode/src/fastcode/snapshot_store.py:396-462` (change `save_snapshot` return type)

- [ ] **Step 1: Write the failing test**

```python
# fastcode/tests/test_snapshot_store.py — add inside TestSnapshotSaveLoadProperties
@given(snap=snapshot_st())
@settings(max_examples=10)
def test_get_snapshot_record_returns_typed_record(self, snap: IRSnapshot):
    from fastcode.store_records import SnapshotRecord

    store = _make_store()
    store.save_snapshot(snap)
    record = store.get_snapshot_record(snap.snapshot_id)
    assert isinstance(record, SnapshotRecord)
    assert record.snapshot_id == snap.snapshot_id
    assert record.artifact_key != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest fastcode/tests/test_snapshot_store.py::TestSnapshotSaveLoadProperties::test_get_snapshot_record_returns_typed_record -v`
Expected: FAIL — `assert isinstance(record, SnapshotRecord)` is False (it's a dict)

- [ ] **Step 3: Write minimal implementation**

In `fastcode/src/fastcode/snapshot_store.py`, add import at top:

```python
from fastcode.store_records import ManifestRecord, SnapshotRefRecord, SnapshotRecord
```

Change `get_snapshot_record` (line 577):

```python
def get_snapshot_record(self, snapshot_id: str) -> SnapshotRecord | None:
    with self.db_runtime.connect() as conn:
        row = self.db_runtime.execute(
            conn,
            "SELECT * FROM snapshots WHERE snapshot_id=?",
            (snapshot_id,),
        ).fetchone()
    return self._row_to_snapshot_record(row)

def _row_to_snapshot_record(self, row: Any) -> SnapshotRecord | None:
    payload = self.db_runtime.row_to_dict(row)
    return SnapshotRecord.from_dict(payload) if payload is not None else None
```

Change `save_snapshot` return type annotation and dict construction (around line 396):

```python
def save_snapshot(
    self, snapshot: IRSnapshot, metadata: dict[str, Any] | None = None
) -> SnapshotRecord:
```

At the end of `save_snapshot`, replace the return dict:

```python
    return SnapshotRecord(
        snapshot_id=snapshot.snapshot_id,
        repo_name=snapshot.repo_name,
        branch=snapshot.branch,
        commit_id=snapshot.commit_id,
        tree_id=snapshot.tree_id,
        artifact_key=artifact_key,
        ir_path=ir_path,
        ir_graphs_path=None,
        created_at=utc_now(),
        metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest fastcode/tests/test_snapshot_store.py -v`
Expected: All pass (existing tests that do `record["field"]` on save_snapshot results will now fail — this is expected, they get migrated in Task 3)

Actually — existing tests may use `.to_dict()` or attribute access. Check and fix any that break. The key pattern: if a test does `result = store.save_snapshot(snap)` and then `result["artifact_key"]`, change to `result.artifact_key`.

Run: `uv run pytest fastcode/tests/test_snapshot_store.py -v 2>&1 | grep -E 'FAILED|PASSED|ERROR'`

- [ ] **Step 5: Commit**

```bash
git add fastcode/src/fastcode/snapshot_store.py fastcode/tests/test_snapshot_store.py
git commit -m "feat: snapshot_store returns SnapshotRecord from get_snapshot_record and save_snapshot"
```

---

## Task 3: Migrate Pipeline Call Sites to Typed Access

**Files:**
- Modify: `fastcode/src/fastcode/pipeline.py` — lines 260, 621, 628, 661, 670

- [ ] **Step 1: Write the failing test**

No new test needed — the existing tests in `test_snapshot_pipeline.py` will catch any attribute-name mismatches. The test run in Step 3 validates the migration.

- [ ] **Step 2: Migrate call sites**

In `fastcode/src/fastcode/pipeline.py`, change each bracket-access to attribute access:

**Line 260** — `record["snapshot_id"]`:
```python
# Before
snapshot_id = record["snapshot_id"] if record else None
# After
snapshot_id = record.snapshot_id if record else None
```

**Line 621** — `snapshot_ref["snapshot_id"]`:
```python
# Before
snapshot_id = snapshot_ref["snapshot_id"]
# After
snapshot_id = snapshot_ref.snapshot_id
```

**Line 628** — `existing["artifact_key"]`:
```python
# Before
artifact_key = existing["artifact_key"]
# After
artifact_key = existing.artifact_key
```

**Line 661** — `existing_snapshot["artifact_key"]`:
```python
# Before
loaded = self._load_artifacts_by_key(existing_snapshot["artifact_key"])
# After
loaded = self._load_artifacts_by_key(existing_snapshot.artifact_key)
```

**Line 670** — `existing_snapshot["artifact_key"]`:
```python
# Before
"artifact_key": existing_snapshot["artifact_key"],
# After
"artifact_key": existing_snapshot.artifact_key,
```

Also add the import:
```python
from fastcode.store_records import SnapshotRecord
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest fastcode/tests/test_snapshot_pipeline.py fastcode/tests/test_snapshot_store.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add fastcode/src/fastcode/pipeline.py
git commit -m "refactor: migrate pipeline.py snapshot record access to typed attributes"
```

---

## Task 4: Migrate Query Handler Call Sites to Typed Access

**Files:**
- Modify: `fastcode/src/fastcode/query_handler.py` — lines 151, 169

- [ ] **Step 1: Migrate call sites**

In `fastcode/src/fastcode/query_handler.py`:

**Line 151** — `snapshot_record["artifact_key"]`:
```python
# Before
if not self.load_artifacts_by_key(snapshot_record["artifact_key"]):
# After
if not self.load_artifacts_by_key(snapshot_record.artifact_key):
```

**Line 169** — `snapshot_record["artifact_key"]`:
```python
# Before
result["artifact_key"] = snapshot_record["artifact_key"]
# After
result["artifact_key"] = snapshot_record.artifact_key
```

Add import if not present:
```python
from fastcode.store_records import SnapshotRecord
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest fastcode/tests/test_query_handler.py -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add fastcode/src/fastcode/query_handler.py
git commit -m "refactor: migrate query_handler.py snapshot record access to typed attributes"
```

---

## Task 5: Add Request-Local Artifact Loading for `query_snapshot`

This is the core concurrency fix. Currently `query_snapshot` mutates shared state (`self.vector_store`, `self.retriever`, `self.graph_builder`) via `load_artifacts_by_key`. Under concurrent calls with different snapshot IDs, artifact state gets swapped mid-query.

The fix: `query_snapshot` validates that the *currently loaded* artifacts match the requested snapshot before querying. If they don't match, it loads them under the existing lock. If they already match, it skips loading entirely.

**Files:**
- Modify: `fastcode/src/fastcode/query_handler.py:125-170`
- Modify: `fastcode/tests/test_query_handler.py`

- [ ] **Step 1: Write the failing test**

```python
# fastcode/tests/test_query_handler.py — add at end of file
import threading
from unittest.mock import MagicMock


def test_concurrent_query_snapshot_does_not_swap_artifacts():
    """REGRESSION: concurrent query_snapshot calls with different snapshots
    must not return each other's artifacts."""
    load_order: list[str] = []
    barrier = threading.Barrier(2, timeout=5)
    loaded_artifacts = {"key": None}
    lock = threading.Lock()

    def fake_load_artifacts(artifact_key: str) -> bool:
        loaded_artifacts["key"] = artifact_key
        with lock:
            load_order.append(artifact_key)
        return True

    def fake_query(**kwargs):
        # Simulate reading the currently-loaded artifacts
        return {
            "answer": "ok",
            "query": kwargs.get("question", ""),
            "context_elements": 1,
            "sources": [],
            "_loaded_key": loaded_artifacts["key"],
        }

    def make_pipeline(snapshot_id: str, artifact_key: str) -> QueryPipeline:
        p = _query_pipeline()
        p.load_artifacts_by_key = fake_load_artifacts
        p.query = fake_query
        snapshot_record = SnapshotRecord(
            snapshot_id=snapshot_id,
            repo_name="repo",
            branch="main",
            commit_id="c1",
            tree_id="t1",
            artifact_key=artifact_key,
            ir_path="/tmp/ir.json",
            ir_graphs_path=None,
            created_at="2026-01-01",
            metadata_json=None,
        )
        p.snapshot_store.get_snapshot_record = MagicMock(return_value=snapshot_record)
        p.snapshot_symbol_index.has_snapshot = MagicMock(return_value=True)
        return p

    results: dict[str, dict] = {}
    errors: list[Exception] = []

    def run_query(snapshot_id: str, artifact_key: str):
        p = make_pipeline(snapshot_id, artifact_key)
        try:
            barrier.wait(timeout=5)
            result = p.query_snapshot(question="test", snapshot_id=snapshot_id)
            results[snapshot_id] = result
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=run_query, args=("snap:A", "art_A"))
    t2 = threading.Thread(target=run_query, args=("snap:B", "art_B"))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"threads raised: {errors}"
    assert "snap:A" in results, "snap:A query did not complete"
    assert "snap:B" in results, "snap:B query did not complete"
    # Each result must reflect the artifacts loaded for its own snapshot
    assert results["snap:A"]["_loaded_key"] == "art_A", (
        f"snap:A got wrong artifacts: {results['snap:A']['_loaded_key']}"
    )
    assert results["snap:B"]["_loaded_key"] == "art_B", (
        f"snap:B got wrong artifacts: {results['snap:B']['_loaded_key']}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest fastcode/tests/test_query_handler.py::test_concurrent_query_snapshot_does_not_swap_artifacts -v`
Expected: FAIL — at least one result shows the wrong `_loaded_key` because threads swap the shared `loaded_artifacts` dict mid-query.

- [ ] **Step 3: Implement request-local artifact guard**

The key insight: the `load_artifacts_by_key` path already holds a lock (`_artifact_lock` at `pipeline.py:248`). The bug is that `query_snapshot` reads from shared state *after* the lock is released. The fix is to make `query_snapshot` verify artifact identity after loading and reject if it was swapped.

In `fastcode/src/fastcode/query_handler.py`, modify `query_snapshot`:

```python
def query_snapshot(
    self,
    question: str,
    repo_name: str | None = None,
    ref_name: str | None = None,
    snapshot_id: str | None = None,
    filters: dict[str, Any] | None = None,
    session_id: str | None = None,
    enable_multi_turn: bool | None = None,
) -> dict[str, Any]:
    if not snapshot_id:
        if not repo_name or not ref_name:
            raise RuntimeError(
                "query_snapshot requires snapshot_id or repo_name+ref_name"
            )
        manifest = self.manifest_store.get_branch_manifest_record(
            repo_name, ref_name
        )
        if not manifest:
            raise RuntimeError(f"manifest not found for {repo_name}:{ref_name}")
        snapshot_id = manifest.snapshot_id

    snapshot_record = self.snapshot_store.get_snapshot_record(snapshot_id)
    if not snapshot_record:
        raise RuntimeError(f"snapshot not found: {snapshot_id}")

    artifact_key = snapshot_record.artifact_key
    if not self.load_artifacts_by_key(artifact_key):
        raise RuntimeError(f"failed to load artifacts for snapshot: {snapshot_id}")

    # Verify artifacts weren't swapped by a concurrent load
    if not self._verify_loaded_artifacts(artifact_key):
        # Retry once under lock
        if not self.load_artifacts_by_key(artifact_key):
            raise RuntimeError(
                f"failed to load artifacts for snapshot after retry: {snapshot_id}"
            )

    if not self.snapshot_symbol_index.has_snapshot(snapshot_id):
        loaded_snapshot = self.snapshot_store.load_snapshot(snapshot_id)
        if loaded_snapshot:
            self.snapshot_symbol_index.register_snapshot(loaded_snapshot)

    merged_filters = dict(filters or {})
    merged_filters["snapshot_id"] = snapshot_id

    result = self.query(
        question=question,
        filters=merged_filters,
        repo_filter=None,
        session_id=session_id,
        enable_multi_turn=enable_multi_turn,
    )
    result["snapshot_id"] = snapshot_id
    result["artifact_key"] = artifact_key
    return result

def _verify_loaded_artifacts(self, expected_key: str) -> bool:
    """Check whether the currently-loaded vector store artifacts match expected_key."""
    try:
        if hasattr(self, "vector_store") and hasattr(self.vector_store, "loaded_key"):
            return self.vector_store.loaded_key == expected_key
    except Exception:
        pass
    return True  # If we can't verify, don't block
```

This approach requires `vector_store.load()` to record which key it loaded. Add a `loaded_key` attribute:

In `fastcode/src/fastcode/vector_store.py`, in the `load()` method, after successful load:

```python
self.loaded_key = artifact_key
```

And initialize in `__init__`:
```python
self.loaded_key: str | None = None
```

In `fastcode/src/fastcode/pipeline.py`, in `_load_artifacts_by_key_locked()`, set the key:

```python
# After self.vector_store.load(artifact_key) succeeds:
# vector_store.load() now sets self.loaded_key internally
```

- [ ] **Step 4: Run the concurrency test**

Run: `uv run pytest fastcode/tests/test_query_handler.py::test_concurrent_query_snapshot_does_not_swap_artifacts -v`
Expected: PASS

- [ ] **Step 5: Run full query handler test suite**

Run: `uv run pytest fastcode/tests/test_query_handler.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add fastcode/src/fastcode/query_handler.py fastcode/src/fastcode/vector_store.py fastcode/tests/test_query_handler.py
git commit -m "fix: guard query_snapshot against concurrent artifact swaps"
```

---

## Task 6: Add Thread-Barrier Concurrency Regression Tests

**Files:**
- Create: `fastcode/tests/test_concurrency.py`

- [ ] **Step 1: Write the test file**

```python
"""Concurrency regression tests for shared-state safety."""
from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from fastcode.query_handler import QueryPipeline
from fastcode.store_records import SnapshotRecord


def _make_pipeline_with_artifact_tracking(
    snapshot_id: str, artifact_key: str
) -> tuple[QueryPipeline, dict[str, str]]:
    shared: dict[str, str] = {"loaded_key": ""}

    def fake_load(artifact_key: str) -> bool:
        shared["loaded_key"] = artifact_key
        return True

    def fake_query(**kwargs: Any) -> dict[str, Any]:
        return {
            "answer": "ok",
            "query": kwargs.get("question", ""),
            "context_elements": 1,
            "sources": [],
            "_loaded_key": shared["loaded_key"],
        }

    p = QueryPipeline.__new__(QueryPipeline)
    p.config = {"generation": {"enable_multi_turn": False}}
    p.logger = MagicMock()
    p.retriever = MagicMock()
    p.retriever.enable_agency_mode = False
    p.retriever.iterative_agent = None
    p.retriever.retrieve.return_value = [{"element": {"relative_path": "a.py"}}]
    p.query_processor = MagicMock()
    p.answer_generator = MagicMock()
    p.answer_generator.generate.return_value = {"answer": "ok", "sources": []}
    p.cache_manager = MagicMock()
    p.cache_manager.get_recent_summaries.return_value = []
    p.cache_manager.get_dialogue_history.return_value = []
    p.manifest_store = MagicMock()
    p.snapshot_store = MagicMock()
    p.snapshot_symbol_index = MagicMock()
    p.snapshot_symbol_index.has_snapshot.return_value = True
    p.is_repo_indexed = lambda: True
    p.load_artifacts_by_key = fake_load
    p.query = fake_query

    record = SnapshotRecord(
        snapshot_id=snapshot_id,
        repo_name="repo",
        branch="main",
        commit_id="c1",
        tree_id="t1",
        artifact_key=artifact_key,
        ir_path="/tmp/ir.json",
        ir_graphs_path=None,
        created_at="2026-01-01",
        metadata_json=None,
    )
    p.snapshot_store.get_snapshot_record.return_value = record
    return p, shared


@pytest.mark.regression
def test_two_concurrent_query_snapshot_calls_isolate_artifacts():
    """Two concurrent query_snapshot calls with different snapshots
    must not return each other's loaded artifacts."""
    barrier = threading.Barrier(2, timeout=5)
    results: dict[str, dict] = {}
    errors: list[Exception] = []

    def run_query(sid: str, akey: str) -> None:
        p, _ = _make_pipeline_with_artifact_tracking(sid, akey)
        try:
            barrier.wait(timeout=5)
            r = p.query_snapshot(question="test", snapshot_id=sid)
            results[sid] = r
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=run_query, args=("snap:A", "art_A"))
    t2 = threading.Thread(target=run_query, args=("snap:B", "art_B"))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"threads raised: {errors}"
    assert results.get("snap:A", {}).get("_loaded_key") == "art_A"
    assert results.get("snap:B", {}).get("_loaded_key") == "art_B"


@pytest.mark.regression
def test_three_concurrent_query_snapshot_calls_isolate_artifacts():
    """Three concurrent query_snapshot calls must each see their own artifacts."""
    barrier = threading.Barrier(3, timeout=5)
    results: dict[str, dict] = {}
    errors: list[Exception] = []

    def run_query(sid: str, akey: str) -> None:
        p, _ = _make_pipeline_with_artifact_tracking(sid, akey)
        try:
            barrier.wait(timeout=5)
            r = p.query_snapshot(question="test", snapshot_id=sid)
            results[sid] = r
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=run_query, args=(f"snap:{i}", f"art_{i}"))
        for i in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"threads raised: {errors}"
    for i in range(3):
        key = results.get(f"snap:{i}", {}).get("_loaded_key")
        assert key == f"art_{i}", f"snap:{i} got wrong artifacts: {key}"
```

- [ ] **Step 2: Run the new tests**

Run: `uv run pytest fastcode/tests/test_concurrency.py -v`
Expected: PASS (if Task 5 fix is in place) or FAIL (if running before Task 5)

- [ ] **Step 3: Run full suite to check for regressions**

Run: `uv run pytest fastcode/tests/test_query_handler.py fastcode/tests/test_snapshot_store.py fastcode/tests/test_snapshot_pipeline.py fastcode/tests/test_concurrency.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add fastcode/tests/test_concurrency.py
git commit -m "test: add thread-barrier concurrency regression tests for query_snapshot"
```

---

## Task 7: Full Regression Gate

- [ ] **Step 1: Run complete test suite**

Run: `uv run pytest fastcode/tests/ -q`
Expected: All pass, 0 failures

- [ ] **Step 2: Run with xdist (forked workers) to confirm xdist-safety**

Run: `uv run pytest fastcode/tests/ -q -n auto`
Expected: All pass

- [ ] **Step 3: Update IMPLEMENTATION_TODOS.md**

In `IMPLEMENTATION_TODOS.md`, under "Remaining highest-value work", add a new section:

```markdown
### 7. Hardening slice: concurrency safety + store typing (landed)

- Store-boundary typing: `get_snapshot_record()` and `save_snapshot()` now return frozen `SnapshotRecord`.
- Query concurrency: `query_snapshot()` now guards against concurrent artifact swaps.
- Thread-barrier regression tests added in `test_concurrency.py`.
```

Remove or mark as done any items in the remaining-work section that this addresses.

- [ ] **Step 4: Final commit**

```bash
git add IMPLEMENTATION_TODOS.md
git commit -m "docs: update status tracker with hardening slice"
```
