# FP Core with Thin I/O Layer — Design Spec

**Date:** 2026-04-26
**Status:** Approved
**Approach:** A — Pure Core, Effect Boundary

## Problem

FastCode's core modules (`FastCode`, `IterativeAgent`, `HybridRetriever`, `SnapshotStore`, `PgRetrievalStore`, `TerminusPublisher`, etc.) mix pure business logic with I/O (PostgreSQL, TerminusDB, LLM, file/git). This creates:

- 17 fake/stub classes across 22 test files, ~80% test code is setup boilerplate
- 5+ `FastCode.__new__()` bypasses to skip `__init__`
- Untestable business logic without heavy mocking infrastructure
- 69 mutable fields in `FastCode` alone, state scattered across method calls

## Design

### Architecture

```
fastcode/core/          ← Pure functions, zero I/O imports, frozen dataclasses
  query.py              ← query processing logic (from main.py)
  retrieval.py          ← scoring, fusion, ranking (from retriever.py)
  iteration.py          ← iteration control flow (from iterative_agent.py)
  graph.py              ← graph algorithms (from graph_builder.py)
  snapshot.py           ← snapshot validation/merging (from snapshot_store.py)
  merge.py              ← IR merge rules (from ir_merge.py)
  projection.py         ← projection transforms (from projection_transform.py)

fastcode/effects/       ← Thin I/O wrappers, each function does ONE I/O operation
  db.py                 ← PostgreSQL queries
  llm.py                ← OpenAI/Ollama API calls
  fs.py                 ← File system and git operations
  graph_db.py           ← TerminusDB/LadybugDB operations

fastcode/               ← Existing orchestrators (modified to wire core + effects)
  main.py
  retriever.py
  iterative_agent.py
  ...
```

### Data Type Strategy

Follow the industry-standard compromise for performance at scale:

| Boundary | Type | Why |
|----------|------|-----|
| FROM outside world (API requests, MCP tools) | **Pydantic** | Must validate untrusted input |
| WITHIN system (core logic) | **Frozen dataclasses** | No validation overhead on trusted data, immutable, fast |
| FROM database | **Frozen dataclasses** | Trust the database, lightweight row mapping |
| TO outside world (API responses) | **Pydantic or dataclass serialization** | Format output, or serialize dataclasses directly |

Frozen dataclasses in `core/` are the canonical internal representation. Pydantic models live only at API boundaries (`api.py`, `mcp_server.py`) for validation and response formatting. Conversion functions at the boundary: `pydantic_to_core()` on input, `core_to_response()` on output.

### Core Principles

1. **`core/` has zero I/O imports.** No `psycopg`, `openai`, `pathlib`, `requests`, `subprocess`. Only stdlib + dataclasses + typing + domain types. Enforceable with a lint rule.

2. **Every function in `core/` is pure.** Same inputs always produce same outputs. No side effects. No mutable state. State flows through as arguments and return values.

3. **`effects/` functions do ONE thing.** Each function performs exactly one I/O operation and returns structured data. No business logic mixed in.

4. **Orchestrators wire core + effects.** Existing classes become thin orchestrators: call effects for data, pass to core for logic, call effects to persist.

5. **Strangler fig migration.** New `core/` and `effects/` packages grow alongside existing code. Each extracted function is tested independently, then wired into the orchestrator. Existing API stays untouched.

### Function Signature Pattern

```python
# core/retrieval.py — pure
from dataclasses import dataclass

@dataclass(frozen=True)
class FusionWeights:
    code_weight: float
    doc_weight: float

@dataclass(frozen=True)
class Hit:
    symbol_id: str
    score: float
    source: str

def fuse_results(
    code_hits: list[Hit],
    doc_hits: list[Hit],
    weights: FusionWeights,
) -> list[Hit]:
    ...  # pure logic, no I/O, no self, no mocks needed

# effects/db.py — thin I/O
def load_code_hits(conn, snapshot_id: str) -> list[Hit]:
    rows = conn.execute(...).fetchall()
    return [Hit(symbol_id=r[0], score=r[1], source=r[2]) for r in rows]

# main.py — orchestrator
def query_snapshot(self, question, ...):
    raw_hits = effects.db.load_code_hits(conn, snap_id)    # I/O
    weights = FusionWeights(code_weight=0.7, doc_weight=0.3)
    return core.retrieval.fuse_results(raw_hits, ..., weights)  # pure
```

### Testing Model

```python
# core/ tests — zero mocks, just data
def test_fuse_results_weights_code_more():
    code = [Hit("a", 0.9, "semantic"), Hit("b", 0.8, "keyword")]
    doc = [Hit("c", 0.95, "doc")]
    result = fuse_results(code, doc, FusionWeights(0.7, 0.3))
    assert result[0].symbol_id == "a"  # code weighted higher

# effects/ tests — test with real or in-memory DB
def test_load_code_hits_maps_rows():
    rows = [("sym:foo:bar", 0.85, "semantic")]
    hits = _rows_to_hits(rows)  # test the mapping logic
    assert hits[0].symbol_id == "sym:foo:bar"

# integration tests — wire everything together
```

### Migration Strategy: Strangler Fig

Build new `core/` and `effects/` alongside existing code. Each extraction follows:

1. Identify pure logic inside an I/O-touching method
2. Extract to `core/` as a pure function with frozen dataclass I/O
3. Extract I/O parts to `effects/` as thin wrappers
4. Add tests for pure function (zero mocks)
5. Wire orchestrator to use core + effects
6. Remove old code path once verified

**Migration order (by test-double pain):**
1. `retriever.py` (HybridRetriever) — retrieval scoring/fusion logic
2. `main.py` (FastCode.query/query_snapshot) — query orchestration
3. `iterative_agent.py` (IterativeAgent) — iteration control flow
4. `snapshot_store.py` — snapshot CRUD + validation
5. `pg_retrieval.py` — vector/FTS query construction
6. `terminus_publisher.py` — graph publishing logic
7. `projection_transform.py` — projection transforms
8. Remaining I/O-touching modules

### Scope

**In scope:** All modules that interact with external services or databases:
- Database layer: `db_runtime.py`, `snapshot_store.py`, `pg_retrieval.py`, `manifest_store.py`, `projection_store.py`
- LLM/Embedding layer: `llm_utils.py`, `answer_generator.py`, `call_extractor.py`, `query_processor.py`, `embedder.py`
- Graph DB layer: `terminus_publisher.py`, `graph_runtime.py`
- File/Git layer: `scip_loader.py`, `repo_overview.py`, `index_run.py`

**Out of scope:**
- Pure data models (`semantic_ir.py` IR dataclasses — already frozen)
- API layer (`api.py`, `mcp_server.py`) — Pydantic boundary, separate concern
- Web UI (`web_app.py`)
- Test infrastructure changes (happens naturally as core is extracted)

### Constraints

- Python >=3.11 (LadybugDB compatibility)
- No new dependencies (stdlib + existing deps only)
- Backward-compatible API (existing callers unaffected during migration)
- `core/` importable without any I/O dependency
