# Implementation Plan: Remaining Spec Gaps

**Date:** 2026-04-01
**Branch:** `feat/core-gaps-tests-demos`
**Baseline audit:** `docs/audit-branch-index-pipeline.md` (2026-03-31)
**Spec sources:** `~/.codex/memories/` (7 design documents, 2026-03-30)

---

## 0. Design Decisions (from user)

1. **Redo worker**: automatic background thread (not manual trigger). `POST /redo/process` as admin escape hatch only.
2. **symbol_version_from**: SCIP-only (stable `external_symbol_id`). AST symbols lack cross-run identity.
3. **Projection response shapes**: dual-write transition period. Add new fields alongside old ones, mark old `@deprecated`, remove after one release cycle.

---

## 1. Work Breakdown

### 1.1 Redo Task Consumer (Critical — C1)

**Gap:** `enqueue_redo_task()` exists in `snapshot_store.py:734` but nothing claims or processes tasks. The redo log is write-only.

**Spec reference:** git-like §12.2 (redo markers), hybrid §8.3 (async recovery).

#### Files to create

| File | Purpose |
|------|---------|
| `fastcode/redo_worker.py` | Background thread + claim/process logic |

#### Files to modify

| File | Change |
|------|--------|
| `fastcode/snapshot_store.py` | Add `claim_redo_task()`, `mark_redo_task_done()`, `mark_redo_task_failed()` |
| `fastcode/main.py` | Start redo worker thread in `__init__()`; add `process_redo_tasks()` method |
| `api.py` | Add `POST /redo/process` admin endpoint |

#### `snapshot_store.py` changes

Add three methods:

```python
def claim_redo_task(self) -> Optional[Dict[str, Any]]:
    """
    Claim the next pending redo task.

    Uses FOR UPDATE SKIP LOCKED on PostgreSQL for safe concurrent claiming.
    Returns task dict or None if no tasks pending.
    """
    if self.db_runtime.backend != "postgres":
        return None
    with self.db_runtime.connect() as conn:
        row = self.db_runtime.execute(
            conn,
            """
            SELECT * FROM redo_tasks
            WHERE status='pending' AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
            ORDER BY created_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """,
            (_utc_now(),),
        ).fetchone()
        if not row:
            return None
        self.db_runtime.execute(
            conn,
            """
            UPDATE redo_tasks
            SET status='running', attempts=attempts+1, updated_at=?
            WHERE task_id=?
            """,
            (_utc_now(), row["task_id"]),
        )
        conn.commit()
    return self.db_runtime.row_to_dict(row)

def mark_redo_task_done(self, task_id: str) -> None:
    """Mark redo task completed."""
    with self.db_runtime.connect() as conn:
        self.db_runtime.execute(
            conn,
            """
            UPDATE redo_tasks SET status='completed', updated_at=?
            WHERE task_id=?
            """,
            (_utc_now(), task_id),
        )
        conn.commit()

def mark_redo_task_failed(self, task_id: str, error: str, max_attempts: int = 5) -> None:
    """
    Mark redo task failed with exponential backoff.

    If attempts >= max_attempts, marks as 'dead'. Otherwise resets to 'pending'
    with next_attempt_at = now + 2^attempts seconds.
    """
    with self.db_runtime.connect() as conn:
        row = self.db_runtime.execute(
            conn,
            "SELECT attempts FROM redo_tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()
        attempts = row["attempts"] if row else 0
        if attempts >= max_attempts:
            self.db_runtime.execute(
                conn,
                """
                UPDATE redo_tasks SET status='dead', last_error=?, updated_at=?
                WHERE task_id=?
                """,
                (error, _utc_now(), task_id),
            )
        else:
            import time
            backoff_seconds = 2 ** attempts
            next_at = datetime.fromtimestamp(
                time.time() + backoff_seconds, tz=timezone.utc
            ).isoformat()
            self.db_runtime.execute(
                conn,
                """
                UPDATE redo_tasks
                SET status='pending', last_error=?, next_attempt_at=?, updated_at=?
                WHERE task_id=?
                """,
                (error, next_at, _utc_now(), task_id),
            )
        conn.commit()
```

Schema note: the existing `redo_tasks` table already has `next_attempt_at`, `attempts`, and `last_error` columns. No schema migration needed.

#### `fastcode/redo_worker.py` design

```python
class RedoWorker:
    """
    Background thread that polls redo_tasks and re-executes failed index runs.

    Lifecycle:
    - Started in FastCode.__init__() when backend is postgres
    - Polls every N seconds (configurable, default 30)
    - Claims task via SnapshotStore.claim_redo_task()
    - For task_type="index_run_recovery": re-runs run_index_pipeline with same params
    - On success: mark_redo_task_done()
    - On failure: mark_redo_task_failed() with backoff
    - Stops cleanly on shutdown via stop_event
    """

    def __init__(self, fastcode_instance, poll_interval_seconds=30):
        self.fc = fastcode_instance
        self.poll_interval = poll_interval_seconds
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self._process_one()
            except Exception:
                pass  # logged internally
            self._stop_event.wait(self.poll_interval)

    def _process_one(self):
        task = self.fc.snapshot_store.claim_redo_task()
        if not task:
            return
        # dispatch on task_type
        ...
```

Thread start in `FastCode.__init__()`:

```python
# After existing init code
self._redo_worker = None
storage_cfg = self.config.get("storage", {})
if storage_cfg.get("backend") == "postgres" or os.getenv("FASTCODE_STORAGE_BACKEND") == "postgres":
    from .redo_worker import RedoWorker
    poll_interval = int(storage_cfg.get("redo_poll_interval_seconds", 30))
    self._redo_worker = RedoWorker(self, poll_interval_seconds=poll_interval)
    self._redo_worker.start()
```

Add shutdown hook (important for clean exit):

```python
def shutdown(self):
    """Stop background workers."""
    if self._redo_worker:
        self._redo_worker.stop()
```

#### `api.py` admin endpoint

```python
@app.post("/redo/process")
async def process_redo_tasks(limit: int = 10):
    """Admin escape hatch: manually process pending redo tasks."""
    ...
```

#### Test plan

| Test | File | What it validates |
|------|------|-------------------|
| `test_claim_redo_task_basic` | `tests/test_redo_worker.py` | Claim returns pending task, marks running |
| `test_claim_redo_task_skip_locked` | `tests/test_redo_worker.py` | Concurrent claims don't double-pick |
| `test_mark_redo_task_done` | `tests/test_redo_worker.py` | Task moves to completed |
| `test_mark_redo_task_failed_backoff` | `tests/test_redo_worker.py` | Failed task gets backoff, eventually dead |
| `test_redo_worker_loop` | `tests/test_redo_worker.py` | Worker thread processes enqueued task |

---

### 1.2 Fencing Tokens on Resource Locks (Critical — C2)

**Gap:** Locks use `owner_id` + `expires_at` but no monotonically increasing fencing token. A stale holder could write after losing a lock.

**Spec reference:** git-like §12.1.

#### Files to modify

| File | Change |
|------|--------|
| `fastcode/snapshot_store.py` | Schema migration: add `fencing_token` column; update `acquire_lock()` to return token; add `validate_fencing_token()` |

#### Schema change

```sql
-- Migration: resource_locks v2
ALTER TABLE resource_locks ADD COLUMN fencing_token BIGINT NOT NULL DEFAULT 0;
```

Apply in `_init_pg_schema()` (guarded by `schema_migrations` check).

#### `acquire_lock()` change

Current signature: `acquire_lock(lock_name, owner_id, ttl_seconds=300) -> bool`

New signature: `acquire_lock(lock_name, owner_id, ttl_seconds=300) -> Optional[int]`

- Returns `None` if lock is held by another owner and not expired.
- Returns the new `fencing_token` (incremented) on success.
- The token is monotonically incremented on every successful acquire.

```python
def acquire_lock(self, lock_name: str, owner_id: str, ttl_seconds: int = 300) -> Optional[int]:
    if self.db_runtime.backend != "postgres":
        return 1  # sqlite: always succeed, token=1
    now = datetime.now(timezone.utc)
    expires_at = (now.timestamp() + ttl_seconds)
    expires_iso = datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat()
    with self.db_runtime.connect() as conn:
        row = self.db_runtime.execute(
            conn,
            "SELECT owner_id, expires_at, fencing_token FROM resource_locks WHERE lock_name=?",
            (lock_name,),
        ).fetchone()
        if row:
            current_exp = row["expires_at"]
            # ... existing expiry check logic ...
            if current_exp_dt and current_exp_dt > now and row["owner_id"] != owner_id:
                return None
            new_token = (row["fencing_token"] or 0) + 1
        else:
            new_token = 1
        self.db_runtime.execute(
            conn,
            """
            INSERT INTO resource_locks (lock_name, owner_id, expires_at, updated_at, fencing_token)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(lock_name) DO UPDATE SET
                owner_id=excluded.owner_id,
                expires_at=excluded.expires_at,
                updated_at=excluded.updated_at,
                fencing_token=excluded.fencing_token
            """,
            (lock_name, owner_id, expires_iso, _utc_now(), new_token),
        )
        conn.commit()
    return new_token
```

#### `validate_fencing_token()` method

```python
def validate_fencing_token(self, lock_name: str, expected_token: int) -> bool:
    """Check that the given token is still current for this lock."""
    with self.db_runtime.connect() as conn:
        row = self.db_runtime.execute(
            conn,
            "SELECT fencing_token FROM resource_locks WHERE lock_name=?",
            (lock_name,),
        ).fetchone()
    if not row:
        return False
    return row["fencing_token"] == expected_token
```

#### Caller update in `main.py`

```python
# Before (line 491):
if not self.snapshot_store.acquire_lock(lock_name, owner_id=run_id, ttl_seconds=600):
    raise RuntimeError(...)

# After:
lock_result = self.snapshot_store.acquire_lock(lock_name, owner_id=run_id, ttl_seconds=600)
if lock_result is None:
    raise RuntimeError(f"snapshot is currently locked for indexing: {snapshot_id}")
fencing_token = lock_result
# Optionally pass fencing_token to write operations that need it
```

#### Test plan

| Test | What it validates |
|------|-------------------|
| `test_acquire_lock_returns_token` | Lock returns int token, increments on re-acquire |
| `test_fencing_token_increments` | Each acquire bumps token |
| `test_validate_fencing_token` | Validate matches current token, fails on stale |
| `test_acquire_lock_denied_returns_none` | Expired lock held by other returns None |

---

### 1.3 Terminus commit_parent Edge (Critical — C3)

**Gap:** `git_meta.parent_commit_id` is consumed by `SnapshotStore.import_git_backbone()` for `git_commits` table but never emitted as a TerminusDB edge.

#### Files to modify

| File | Change |
|------|--------|
| `fastcode/terminus_publisher.py` | Add parent commit node + `commit_parent` edge in `build_lineage_payload()` |

#### Implementation in `build_lineage_payload()`

After line 92 (where commit node is appended), add:

```python
parent_commit_id = git_meta.get("parent_commit_id")
if commit_node_id and parent_commit_id:
    parent_node_id = f"commit:{repo_name}:{parent_commit_id}"
    nodes.append({
        "id": parent_node_id,
        "type": "Commit",
        "props": {
            "repo_name": repo_name,
            "commit_id": parent_commit_id,
        },
    })
    edges.append({
        "type": "commit_parent",
        "src": commit_node_id,
        "dst": parent_node_id,
    })
```

Also handle `git_meta.parent_commit_ids` (plural, for merge commits):

```python
parent_commit_ids = git_meta.get("parent_commit_ids", [])
if not parent_commit_ids and parent_commit_id:
    parent_commit_ids = [parent_commit_id]
for pid in parent_commit_ids:
    if not pid:
        continue
    parent_node_id = f"commit:{repo_name}:{pid}"
    nodes.append(...)
    edges.append({"type": "commit_parent", "src": commit_node_id, "dst": parent_node_id})
```

#### Pipeline plumbing in `main.py`

Ensure `git_meta` passed to `publish_snapshot_lineage()` includes `parent_commit_id`. Currently it's passed as `git_meta=snapshot_ref` (line 681). The `snapshot_ref` dict from `_resolve_snapshot_ref()` does not include `parent_commit_id`. Need to enrich it:

```python
# In run_index_pipeline(), around line 647:
git_meta = dict(snapshot_ref)
if commit_id:
    try:
        repo = Repo(self.loader.repo_path)
        commit_obj = repo.commit(commit_id)
        parent_ids = [p.hexsha for p in commit_obj.parents]
        git_meta["parent_commit_id"] = parent_ids[0] if parent_ids else None
        git_meta["parent_commit_ids"] = parent_ids
    except Exception:
        pass
```

Then pass `git_meta` instead of `snapshot_ref` to `publish_snapshot_lineage()` and `import_git_backbone()`.

#### Test plan

| Test | What it validates |
|------|-------------------|
| `test_commit_parent_edge_single` | Single parent produces one `commit_parent` edge |
| `test_commit_parent_edge_merge` | Multiple parents produce multiple edges |
| `test_commit_parent_no_parents` | Root commit produces no edges |

---

### 1.4 AST call Edges (High — H3)

**Gap:** `ast_to_ir.py` produces `contain`, `import`, `inherit` edges but never `call`. The `call_graph` in `IRGraphs` is always empty from AST input.

#### Strategy

The legacy `CodeGraphBuilder` already builds a precise call graph using `CallExtractor` + `SymbolResolver`. Rather than duplicating this logic in the AST adapter, extract call edges from the already-built `CodeGraphBuilder.call_graph` and translate them into IR `call` edges in `run_index_pipeline()`.

This is the approach the spec intended — the AST adapter is one input, and call extraction from the existing graph builder is reused.

#### Files to modify

| File | Change |
|------|--------|
| `fastcode/main.py` | After building `temp_graph`, extract call edges and append to `ast_snapshot.edges` |

#### Implementation in `run_index_pipeline()`

After `temp_graph.build_graphs(elements, module_resolver, symbol_resolver)` (line 529) and before `build_ir_from_ast()` (line 546), capture the call graph. Then after `build_ir_from_ast()`, translate:

```python
# After temp_graph.build_graphs() and before build_ir_from_ast():
call_graph_edges_raw = list(temp_graph.call_graph.edges(data=True))

# After ast_snapshot = build_ir_from_ast(...):
# Translate legacy call graph edges to IR call edges
import hashlib as _hl
ast_call_edges = []
ast_elem_to_ir_symbol = {}
for sym in ast_snapshot.symbols:
    m = sym.metadata or {}
    ast_elem_id = m.get("ast_element_id")
    if ast_elem_id:
        ast_elem_to_ir_symbol[ast_elem_id] = sym.symbol_id

for caller_elem_id, callee_elem_id, data in call_graph_edges_raw:
    ir_caller = ast_elem_to_ir_symbol.get(caller_elem_id)
    ir_callee = ast_elem_to_ir_symbol.get(callee_elem_id)
    if ir_caller and ir_callee:
        edge_id = f"edge:{_hl.md5(f'call:{ir_caller}:{ir_callee}'.encode()).hexdigest()[:20]}"
        ast_call_edges.append(IREdge(
            edge_id=edge_id,
            src_id=ir_caller,
            dst_id=ir_callee,
            edge_type="call",
            source="ast",
            confidence="heuristic",
            doc_id=None,
            metadata={
                "call_name": data.get("call_name"),
                "call_type": data.get("call_type"),
                "file_path": data.get("file_path"),
                "extractor": "fastcode.adapters.ast_to_ir",
                "source": "ast",
            },
        ))

ast_snapshot.edges.extend(ast_call_edges)
```

This ensures the AST snapshot carries `call` edges before the merge step, so `ir_graph_builder` will correctly populate `call_graph` from the merged snapshot.

#### Test plan

| Test | What it validates |
|------|-------------------|
| `test_ast_call_edges_from_graph` | Pipeline produces IR call edges from legacy call graph |
| `test_call_edges_in_merged_graph` | After merge, IRGraphs.call_graph has edges |

---

### 1.5 SCIP Typed Models (High — H1, H2)

**Gap:** `scip_loader.py` returns `Dict[str, Any]`. No `SCIPIndex` or `SCIPArtifactRef` typed models.

#### Files to create

| File | Purpose |
|------|---------|
| `fastcode/scip_models.py` | `SCIPIndex`, `SCIPDocument`, `SCIPSymbol`, `SCIPOccurrence`, `SCIPArtifactRef` dataclasses |

#### Files to modify

| File | Change |
|------|--------|
| `fastcode/scip_loader.py` | `load_scip_artifact()` returns `SCIPIndex` instead of `Dict[str, Any]` |
| `fastcode/adapters/scip_to_ir.py` | Accept `SCIPIndex` (or `Dict` for backward compat); add `language_hint` param |
| `fastcode/snapshot_store.py` | Update `save_scip_artifact_ref()` to accept `SCIPArtifactRef` or construct one |

#### `fastcode/scip_models.py`

```python
@dataclass
class SCIPOccurrence:
    symbol: str
    role: str  # "definition", "reference", "implementation", "type_definition", "unknown"
    range: List[Optional[int]]  # [start_line, start_col, end_line, end_col]

@dataclass
class SCIPSymbol:
    symbol: str
    name: Optional[str] = None
    kind: Optional[str] = None
    qualified_name: Optional[str] = None
    signature: Optional[str] = None
    range: List[Optional[int]] = field(default_factory=lambda: [None, None, None, None])

@dataclass
class SCIPDocument:
    path: str
    language: Optional[str] = None
    symbols: List[SCIPSymbol] = field(default_factory=list)
    occurrences: List[SCIPOccurrence] = field(default_factory=list)

@dataclass
class SCIPIndex:
    documents: List[SCIPDocument] = field(default_factory=list)
    indexer_name: Optional[str] = None
    indexer_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SCIPIndex":
        ...

    def to_dict(self) -> Dict[str, Any]:
        ...

@dataclass
class SCIPArtifactRef:
    snapshot_id: str
    indexer_name: str
    indexer_version: Optional[str]
    artifact_path: str
    checksum: str
    created_at: str
```

#### `scip_loader.py` change

```python
from .scip_models import SCIPIndex

def load_scip_artifact(path: str) -> SCIPIndex:
    raw = _load_raw_json(path)  # existing logic
    return SCIPIndex.from_dict(raw)
```

#### `scip_to_ir.py` change

```python
def build_ir_from_scip(
    repo_name: str,
    snapshot_id: str,
    scip_index: Union[Dict[str, Any], "SCIPIndex"],
    branch: str | None = None,
    commit_id: str | None = None,
    tree_id: str | None = None,
    language_hint: str | None = None,
) -> IRSnapshot:
    ...
```

Accept both types for backward compat (any existing callers passing dicts will still work).

#### Test plan

| Test | What it validates |
|------|-------------------|
| `test_scip_index_round_trip` | `from_dict` -> `to_dict` preserves all fields |
| `test_load_scip_returns_typed` | `load_scip_artifact()` returns `SCIPIndex` |
| `test_scip_to_ir_with_language_hint` | `language_hint` is used as fallback for document language |

---

### 1.6 Terminus symbol_version_from Edges (High — H4)

**Gap:** No edge tracking symbol evolution across snapshots. Required for "symbol evolved into symbol" queries.

**Design decision:** SCIP-only. Match on stable `external_symbol_id` across snapshots.

#### Files to modify

| File | Change |
|------|--------|
| `fastcode/terminus_publisher.py` | Add `symbol_version_from` edges when previous snapshot data is available |

#### Implementation approach

The publisher needs access to symbols from the **previous** snapshot for the same repo+branch. This requires:

1. `build_lineage_payload()` accepts an optional `previous_snapshot_symbols: Dict[str, str]` mapping `external_symbol_id -> symbol_node_id`.
2. When building symbol nodes, check if `external_symbol_id` exists in previous snapshot. If yes, emit `symbol_version_from` edge.

```python
def build_lineage_payload(
    self,
    snapshot: Dict[str, Any],
    manifest: Dict[str, Any],
    git_meta: Dict[str, Any],
    previous_snapshot_symbols: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    ...
    # When building symbol nodes:
    for sym in symbols:
        ext_id = sym.get("external_symbol_id")
        if ext_id and previous_snapshot_symbols and ext_id in previous_snapshot_symbols:
            prev_node_id = previous_snapshot_symbols[ext_id]
            edges.append({
                "type": "symbol_version_from",
                "src": node_id,  # current symbol version
                "dst": prev_node_id,  # previous symbol version
            })
```

#### Plumbing in `main.py`

Before calling `publish_snapshot_lineage()`, look up the previous manifest's snapshot and extract its SCIP symbol `external_symbol_id`s:

```python
# Resolve previous snapshot symbols for symbol_version_from edges
previous_snapshot_symbols = None
if self.terminus_publisher.is_configured():
    prev_manifest = self.manifest_store.get_branch_manifest(repo_name, ref_name)
    if prev_manifest and prev_manifest.get("snapshot_id") != snapshot_id:
        prev_snapshot = self.snapshot_store.load_snapshot(prev_manifest["snapshot_id"])
        if prev_snapshot:
            previous_snapshot_symbols = {}
            for s in prev_snapshot.symbols:
                if s.external_symbol_id:
                    prev_node_id = f"symbol:{prev_manifest['snapshot_id']}:{s.symbol_id}"
                    previous_snapshot_symbols[s.external_symbol_id] = prev_node_id

self.terminus_publisher.publish_snapshot_lineage(
    snapshot=merged_snapshot.to_dict(),
    manifest=manifest,
    git_meta=git_meta,
    previous_snapshot_symbols=previous_snapshot_symbols,
    idempotency_key=...,
)
```

#### Test plan

| Test | What it validates |
|------|-------------------|
| `test_symbol_version_from_edge` | Matching `external_symbol_id` across snapshots produces edge |
| `test_no_version_from_for_ast_symbols` | Symbols without `external_symbol_id` produce no edge |
| `test_no_version_from_first_snapshot` | First snapshot (no previous) has no edges |

---

### 1.7 Projection Schema Dual-Write (Medium — L4, L5)

**Gap:** L1 uses `cross_links` instead of spec's `relations{type: [...]}`. Missing `related_code`/`related_memory`. L2 chunks lack common envelope.

**Strategy:** Dual-write both old and new fields.

#### Files to modify

| File | Change |
|------|--------|
| `fastcode/projection_transform.py` | Add v2 fields alongside v1; update `_envelope()` and chunk building |

#### L1 changes in `_build_l1_summary()` area

```python
l1_content_extra = {
    "sections": sections,
    # v1 (deprecated after one release cycle):
    "relations": {
        "cross_links": [
            {
                "id": f"{src}->{dst}",
                "title": f"{src} -> {dst}",
                "type": "xref",
                "confidence": min(1.0, float(weight) / 4.0),
            }
            for src, dst, weight in xrefs
        ]
    },
    # v2 (spec-compliant):
    "relations_v2": {
        "xref": [
            {
                "id": f"{src}->{dst}",
                "title": f"{src} -> {dst}",
                "type": "xref",
                "confidence": min(1.0, float(weight) / 4.0),
            }
            for src, dst, weight in xrefs
        ],
    },
    "navigation": navigation,
    "decisions": [...],
    "related_code": source_refs,  # from self._source_refs()
    "related_memory": [],  # empty for code-domain projections
}
```

#### L2 chunk envelope

```python
# Wrap chunk in common envelope alongside existing bare format
chunk = {
    # v1 (bare):
    "chunk_id": chunk_id,
    "kind": "cluster_evidence",
    "content": content,
    # v2 (full envelope):
    "version": "v1",
    "layer": "L2",
    "source": {"domain": "code", "refs": refs},
    "render": {"text": content.get("snippet", "")},
    "meta": {"cluster_id": cid},
}
```

#### Deprecation plan

After one release cycle:
- Remove `relations.cross_links` (keep `relations_v2` renamed to `relations`)
- Remove bare chunk fields (keep envelope only)
- Add migration note in changelog

#### Test plan

| Test | What it validates |
|------|-------------------|
| `test_l1_dual_write_relations` | Both `cross_links` and `relations_v2` present |
| `test_l1_related_code_present` | `related_code` field populated |
| `test_l2_chunk_envelope` | Chunk has both bare and envelope fields |

---

### 1.8 API Endpoints (High — H6)

#### Files to modify

| File | Change |
|------|--------|
| `api.py` | Add repo refs, symbol lookup, graph traversal endpoints |

#### New endpoints

| Endpoint | Method | Implementation |
|----------|--------|----------------|
| `GET /repos/{repo_name}/refs` | GET | `snapshot_store.resolve_snapshot_for_ref()` |
| `GET /symbols/find` | GET | `snapshot_symbol_index.resolve_symbol()` |
| `GET /graph/callees/{snapshot_id}/{symbol_id}` | GET | Load IR graphs, walk call_graph successors |
| `GET /graph/callers/{snapshot_id}/{symbol_id}` | GET | Load IR graphs, walk call_graph predecessors |
| `GET /graph/dependencies/{snapshot_id}/{doc_id}` | GET | Load IR graphs, walk dependency_graph |
| `POST /redo/process` | POST | Manual redo task processing (admin) |

#### Graph traversal implementation

```python
@app.get("/graph/callees/{snapshot_id}/{symbol_id}")
async def get_callees(snapshot_id: str, symbol_id: str, max_hops: int = 1):
    ir_graphs = fastcode.snapshot_store.load_ir_graphs(snapshot_id)
    if not ir_graphs:
        raise HTTPException(404, "snapshot or graphs not found")
    g = ir_graphs.call_graph
    if symbol_id not in g:
        return {"callees": []}
    result = []
    for node in nx.single_source_shortest_path_length(g, symbol_id, cutoff=max_hops):
        if node != symbol_id:
            result.append({"symbol_id": node, "distance": ...})
    return {"callees": result}
```

Similar pattern for `/graph/callers` (use `g.reverse()`) and `/graph/dependencies`.

#### Test plan

| Test | What it validates |
|------|-------------------|
| `test_get_repo_refs` | Returns branch -> snapshot mapping |
| `test_get_symbol_find` | Symbol resolution returns IR symbol data |
| `test_get_graph_callees` | Returns callee list for known symbol |
| `test_get_graph_callers` | Returns caller list for known symbol |

---

### 1.9 Test Coverage Gaps (High — H5)

#### Files to create

| File | Tests |
|------|-------|
| `tests/test_redo_worker.py` | Redo task claiming, backoff, worker loop (see §1.1) |
| `tests/test_fencing_token.py` | Token acquire/increment/validate (see §1.2) |
| `tests/test_terminus_commit_parent.py` | commit_parent edge (see §1.3) |
| `tests/test_ast_call_edges.py` | Call edges from legacy graph (see §1.4) |
| `tests/test_scip_models.py` | SCIPIndex round-trip, SCIPArtifactRef (see §1.5) |
| `tests/test_symbol_version_from.py` | Cross-snapshot symbol evolution edges (see §1.6) |
| `tests/test_projection_v2_schema.py` | Dual-write L1/L2 fields (see §1.7) |
| `tests/test_graph_api.py` | Graph traversal endpoints (see §1.8) |
| `tests/test_index_run.py` | Index run lifecycle, publish retry claim/mark |
| `tests/test_db_runtime.py` | SQLite/PG connect, adapt_sql, row_to_dict |
| `tests/test_projection_store.py` | Save/get/cached projection |

#### Integration test

| File | What |
|------|------|
| `tests/test_pipeline_integration.py` | Full `run_index_pipeline()` with mock SCIP + AST, validates end-to-end snapshot/manifest/graph/projection flow |

---

### 1.10 Lower-Priority Items (Informational)

These are tracked but not blockers for the next implementation cycle:

| # | Gap | Priority | Notes |
|---|-----|----------|-------|
| M1 | `SnapshotRef` dataclass | Medium | Data exists in SQL; add typed model for API responses |
| M2 | `PublishedManifest` dataclass | Medium | Returns dict with correct keys; add typed model |
| M3 | `ProjectionStore` bypasses DBRuntime | Medium | Refactor to use `DBRuntime` instead of own psycopg pool |
| L1 | `content_hash` never populated | Low | Need file content access during indexing |
| L2 | `repo_root` unused in AST adapter | Low | Reserved for future module-path resolution |
| L3 | Column info hardcoded to 0 | Low | Use `elem.metadata.start_col` when available |
| L6 | Hash length inconsistency (20 vs 24) | Low | Normalize to one length |
| L7 | `"unknown"` role not handled in SCIP | Low | Skip ref edge generation for unknown roles |

---

## 2. Execution Order

Dependencies between work items determine the order:

```
Phase 1 — Critical hardening (no inter-dependencies, can parallelize):
  1.1 Redo task consumer
  1.2 Fencing tokens
  1.3 commit_parent edge

Phase 2 — High-priority spec alignment (sequential dependencies):
  1.4 AST call edges          (depends on nothing)
  1.5 SCIP typed models       (independent, but scip_to_ir changes should follow)
  1.6 symbol_version_from     (depends on 1.5 for clean model usage)

Phase 3 — Schema and API (depends on Phase 1+2):
  1.7 Projection dual-write
  1.8 API endpoints

Phase 4 — Test coverage (runs alongside all phases):
  1.9 Tests
```

Within each phase, items can be implemented in any order.

---

## 3. Risk Assessment

| Risk | Mitigation |
|------|------------|
| Fencing token migration on existing PG databases | Guarded by `schema_migrations`; `ALTER TABLE ADD COLUMN` with `DEFAULT 0` is non-breaking |
| `acquire_lock()` return type change (`bool` -> `Optional[int]`) | Check all callers; only `main.py:491` calls it. Update call site. |
| `load_scip_artifact()` return type change (`Dict` -> `SCIPIndex`) | `build_ir_from_scip()` accepts `Union[Dict, SCIPIndex]` for backward compat |
| Redo worker thread safety | Worker only accesses `snapshot_store` (which uses connection-per-operation); `FastCode` methods must not be called concurrently without locks |
| Projection dual-write increases response size | Temporary; `relations_v2` is small. Remove v1 fields after one release |
| Background thread lifecycle in web/CLI contexts | `daemon=True` ensures thread dies with process. Add `shutdown()` call in `web_app.py` and `api.py` lifespan handlers |

---

## 4. Acceptance Criteria

The implementation is complete when:

1. `claim_redo_task()` + worker loop successfully recover a failed index run end-to-end.
2. `acquire_lock()` returns a fencing token that increments on each acquire.
3. Terminus lineage payload includes `commit_parent` edges when parent commit is available.
4. Terminus lineage payload includes `symbol_version_from` edges for SCIP symbols that existed in a previous snapshot.
5. AST adapter produces `call` edges that survive through merge and appear in `IRGraphs.call_graph`.
6. `load_scip_artifact()` returns a typed `SCIPIndex`; `build_ir_from_scip()` accepts `language_hint`.
7. L1 projection output includes both `relations.cross_links` (v1) and `relations_v2` (spec-compliant).
8. L2 chunks include both bare fields and full envelope fields.
9. All new API endpoints return correct data.
10. All new tests pass; no existing tests broken.
