# Architecture Gap Closure Plan

**Date:** 2026-04-24
**Scope:** Close gaps between architecture docs (`review-response-context.md`, `architecture.md`) and codebase
**Base branch:** `feat/scip-multi-language`

---

## Gap Assessment Summary

### Implementation Maturity

| Subsystem | Lines | Maturity | Gap Level |
|-----------|-------|----------|-----------|
| IR Models + Merge | ~1,500 | Production | LOW — closely matches docs |
| Retrieval + Fusion | ~3,500 | Production | LOW — exceeds docs in some areas |
| Projection + Graph Alg | ~1,300 | Production | MEDIUM — simplification needed |
| MCP Graph Tools | 759 | Partial | HIGH — 5 tools missing |
| TerminusDB Publisher | 213 | Partial | HIGH — no outbox, no incremental |
| Doc Ingestion | 424 | Good | MEDIUM — bridge not implemented |
| Infrastructure (API, Config) | ~4,600 | Production | MEDIUM — config bloat |

### Total: 25,707 lines Python, 47 files

---

## Gaps by Priority

### P0 — Architecture Contract Violations

These are features the architecture specifies but implementation does not provide.

| # | Gap | Architecture Says | Code Does | Impact |
|---|-----|-------------------|-----------|--------|
| G1 | **MCP graph tools** | `directed_path`, `impact_analysis`, `leiden_clusters`, `steiner_path`, `find_callers` | Only `get_call_chain` exists. 5 tools missing. | Agent cannot do structural reasoning via MCP. |
| G2 | **Outbox pattern for TerminusDB** | Local intent records + async retry + idempotent upsert | Direct HTTP POST, no retry, no event log | Publishing failures lose data silently. |
| G3 | **Incremental update** | blob_oid diff → re-extract changed files only → tombstone+reinsert | Always full re-index. blob_oid stored but unused | Large repos take minutes per re-index. |

### P1 — Architecture Simplification Not Applied

Docs prescribe simplification, code still has the old complexity.

| # | Gap | Architecture Says | Code Does | Impact |
|---|-----|-------------------|-----------|--------|
| G4 | **Leiden resolution** | Single resolution 1.0 default, hierarchical opt-in | `[0.5, 1.0, 2.0, 4.0]` always, `hierarchical_leiden_enabled: true` default | Over-clustering, unnecessary computation. |
| G5 | **Betweenness centrality** | Remove. PageRank+degree sufficient. | Still computed in `_pick_representatives()`, guarded by 5000-node limit | O(VE) cost, highly correlated with PageRank. |
| G6 | **Config surface** | Reduce to essential (edge_weights, max_hops, enable_leiden) | 83 parameters across 15 categories | Configuration complexity overwhelming. |
| G7 | **Phantom SCIP indexers** | Remove PHP/Dart (don't exist) | Still listed in `scip_indexers.py` | Misleading language support claims. |

### P2 — Partial Implementation Gaps

Features partially done but not matching architecture contract.

| # | Gap | Architecture Says | Code Does | Impact |
|---|-----|-------------------|-----------|--------|
| G8 | **SCIP Resolution Bridge** | 3-strategy cascade: lexical exact → namespace contextual → vector semantic | Regex extraction only (`_extract_mentions()`) | Low doc→code matching recall. |
| G9 | **Collection prior (π_code, π_doc)** | Explicit computation from identifier specificity, design keywords, etc. | Implicit via affinity scoring in alpha computation | Functionally similar but not auditable. |
| G10 | **Doc type weights** | ADR/RFC > design > readme in projection | No doc_type_weight differentiation | All docs treated equally in projection. |
| G11 | **Session prefix injection** | L0/L1 as system prompt prefix, prompt-cached | Projections exist but no injection mechanism | Agent does cold-start exploration every session. |
| G12 | **TerminusDB derived graph** | Materialized graph from canonical facts, branch-aware | Lineage only (repo/branch/commit/snapshot nodes), not code graph (symbols/edges) | TerminusDB stores metadata lineage, not the actual G_dep/G_inh/G_call graph. |

### P3 — Research / Validation Items

Not blocking implementation, but need validation per architecture.

| # | Gap | Notes |
|---|-----|-------|
| G13 | Projection format (D1.1) | DSL vs TOON — research, not blocking |
| G14 | FTS code-awareness | English tsvector only — acceptable per architecture |
| G15 | TerminusDB storage benchmark | "95% dedup" is hypothesis to validate |
| G16 | Leiden resolution tuning | What single value works across repo sizes |

---

## Implementation Plan

### Phase 1: Simplification (G4, G5, G7) — Low Risk, High Value

These are cleanup tasks that reduce complexity without adding new features.

#### Step 1.1: Remove phantom SCIP indexers (G7)
- **File:** `fastcode/scip_indexers.py`
- **Action:** Remove PHP and Dart indexer entries from language map
- **Verify:** `uv run pytest tests/ -v -k "scip"`

#### Step 1.2: Simplify Leiden to single resolution (G4)
- **File:** `fastcode/projection_transform.py`
- **Action:**
  - Change default to `leiden_resolutions: [1.0]` (single)
  - Change `hierarchical_leiden_enabled: false` (default off)
  - Keep multi-resolution as explicit opt-in
  - Remove hierarchy level selection logic when single resolution
- **Verify:** `uv run pytest tests/ -v -k "projection or leiden"`

#### Step 1.3: Remove betweenness centrality (G5)
- **File:** `fastcode/projection_transform.py` — `_pick_representatives()`
- **Action:**
  - Remove `nx.betweenness_centrality()` call
  - Change representative ranking from `(pagerank, betweenness, degree_centrality, degree)` to `(pagerank, degree_centrality, degree)`
  - Remove `centrality_max_nodes` guard (PageRank+degree are fast)
  - Remove `centrality_max_nodes` from config
- **Verify:** `uv run pytest tests/ -v -k "projection or representative"`

### Phase 2: MCP Graph Tools (G1) — Medium Risk, Core Feature

These tools are the primary agent interface for structural reasoning.

#### Step 2.1: Add `directed_path` MCP tool
- **File:** `fastcode/mcp_server.py`
- **Logic:**
  - Resolve `from` and `to` symbols by name/ID
  - Load IR graph (IRGraphs) for snapshot
  - `nx.shortest_path(directed_graph, from_node, to_node)` on union of G_call + G_dep
  - Return: path nodes with metadata (name, file, line range)
- **Dependencies:** IRGraphBuilder (existing)

#### Step 2.2: Add `impact_analysis` MCP tool
- **File:** `fastcode/mcp_server.py`
- **Logic:**
  - Resolve symbol by ID/name
  - Load IR graph
  - BFS on reversed G_call + G_dep edges (callers + dependents)
  - Return: all affected symbols with distance and edge types
- **Dependencies:** IRGraphBuilder (existing)

#### Step 2.3: Add `leiden_clusters` MCP tool
- **File:** `fastcode/mcp_server.py`
- **Logic:**
  - Load projection for snapshot_id
  - Return L1 cluster structure (cluster IDs, labels, representative symbols, node counts)
  - Or run on-demand if no cached projection
- **Dependencies:** ProjectionTransformer (existing)

#### Step 2.4: Add `steiner_path` MCP tool
- **File:** `fastcode/mcp_server.py`
- **Logic:**
  - Resolve terminal symbols from query
  - Load IR graph, symmetrize
  - `nx.approximation.steiner_tree()` with leaf pruning
  - Return: subgraph nodes and edges
- **Dependencies:** ProjectionTransformer._build_steiner_subgraph() (existing)

#### Step 2.5: Add `find_callers` MCP tool
- **File:** `fastcode/mcp_server.py`
- **Logic:**
  - Resolve symbol by ID/name
  - Load IR graph
  - Query predecessors on G_call edges
  - Return: caller symbols with file/line metadata
- **Note:** Replace/supplement existing `get_call_chain` which works differently

#### Verify all: `uv run pytest tests/ -v -k "mcp"`

### Phase 3: Outbox Pattern (G2) — Medium Risk, Reliability

#### Step 3.1: Create outbox table in PostgreSQL
- **File:** `fastcode/snapshot_store.py` or new `fastcode/outbox_store.py`
- **Schema:**
  ```sql
  CREATE TABLE publish_outbox (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,  -- 'lineage_publish'
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, in_progress, published, failed
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    created_at TEXT NOT NULL,
    last_attempt_at TEXT,
    error_message TEXT
  )
  ```

#### Step 3.2: Modify TerminusPublisher to use outbox
- **File:** `fastcode/terminus_publisher.py`
- **Action:**
  - `publish_snapshot_lineage()` writes to outbox table instead of direct HTTP POST
  - Separate `flush_outbox()` method: reads pending events, POSTs to TerminusDB with retry
  - Idempotency: payload hash as dedup key
  - Retry: exponential backoff, max 5 attempts

#### Step 3.3: Add outbox flush to RedoWorker
- **File:** `fastcode/redo_worker.py`
- **Action:** Add `publish_outbox_flush` task type to RedoWorker's poll cycle

#### Verify: `uv run pytest tests/ -v -k "outbox or terminus"`

### Phase 4: Incremental Update (G3) — High Risk, Major Feature

This is the biggest gap. Requires careful design.

#### Step 4.1: Add blob_oid diffing to index pipeline
- **File:** `fastcode/main.py` — `run_index_pipeline()`
- **Logic:**
  - On re-index of same branch, compare new file blob_oids against last snapshot's blob_oids
  - Extract only changed files (SCIP + AST)
  - Merge only changed file units into existing snapshot
  - Tombstone old file-scoped relations → insert new
- **Dependencies:** ManifestStore (to get previous snapshot), blob_oid field (existing)

#### Step 4.2: File-scoped tombstone + reinsert
- **File:** `fastcode/ir_merge.py` or new `fastcode/incremental_update.py`
- **Logic:**
  - Given: old snapshot + changed file paths + new extractions for those files
  - Remove all units where `path in changed_paths` from old snapshot
  - Remove all relations where src or dst unit has changed path
  - Insert new units and relations from fresh extraction
  - Preserve embeddings for unchanged units
- **Key invariant:** File-scoped tombstone is correct because all relations (contain, import, inherit, call) have both endpoints in the same file OR reference a cross-file target by symbol ID.

#### Step 4.3: Update retrieval indices incrementally
- **File:** `fastcode/pg_retrieval.py`
- **Action:** Delete+reinsert rows for changed file paths only, not full rebuild

#### Verify: `uv run pytest tests/ -v -k "incremental"`

### Phase 5: Session Prefix Injection (G11) — Low Risk

#### Step 5.1: Add projection prefix endpoint
- **File:** `fastcode/api.py`
- **Action:** `GET /projection/{snapshot_id}/prefix` returns L0+L1 as compact JSON suitable for system prompt injection

#### Step 5.2: Document consumption contract
- **Action:** Update docs with how agents should consume L0/L1 prefix:
  - Claude Code: inject into CLAUDE.md or system prompt
  - Cursor/Windsurf: inject into rules file
  - MCP: new `get_session_prefix` tool that returns the projection

#### Verify: Manual test with an actual agent session

### Phase 6: TerminusDB Derived Code Graph (G12) — Medium Risk

Currently TerminusDB stores only lineage metadata, not the actual code graph (G_dep, G_inh, G_call edges).

#### Step 6.1: Extend TerminusPublisher to publish code graph edges
- **File:** `fastcode/terminus_publisher.py`
- **Action:**
  - After canonical IR is built, publish symbol nodes with SCIP identity
  - Publish all edges (contain, ref, import, inherit, call) with confidence bands
  - Use branch-aware node IDs: `symbol:{branch}:{symbol_id}`
  - Include provenance metadata on edges

#### Step 6.2: Add graph query methods
- **File:** `fastcode/terminus_publisher.py` or new `fastcode/terminus_query.py`
- **Action:**
  - `load_graph(snapshot_id)` → load all edges for a snapshot into NetworkX
  - `expand_node(symbol_id, hops)` → graph traversal in TerminusDB directly
  - `get_neighbors(symbol_id, edge_type)` → targeted neighbor query

#### Verify: `uv run pytest tests/ -v -k "terminus and graph"`

### Phase 7: SCIP Resolution Bridge (G8) — Medium Risk

#### Step 7.1: Implement 3-strategy cascade
- **File:** `fastcode/doc_ingester.py` — extend `_extract_mentions()`
- **Strategy 1 — Lexical exact:** symbol display_name matches doc entity text (existing regex)
- **Strategy 2 — Namespace contextual:** match within same directory/package context
- **Strategy 3 — Vector semantic:** embed entity text, find nearest code symbol by embedding similarity
- **Cascade:** take highest-confidence match from strategies in order

#### Step 7.2: Add confidence bands to mentions
- **File:** `fastcode/doc_ingester.py`
- **Action:** Each mention gets `match_strategy` and `match_confidence` fields

#### Verify: `uv run pytest tests/ -v -k "bridge or resolution"`

### Phase 8: Config Reduction (G6) — Low Risk, Polish

#### Step 8.1: Audit and reduce config parameters
- **File:** `fastcode/main.py` — `_get_default_config()`
- **Action:**
  - Identify 83 params → categorize as essential vs tunable vs deprecated
  - Essential (~20): backend, dsn, model, alpha_base, max_results, etc.
  - Tunable (~30): keep exposed but document sensible defaults
  - Deprecated/internal (~33): remove from user-facing config, hardcode
  - Focus on retrieval (13 → 6), projection (12 → 5), generation (9 → 4)
- **Principle:** Config should expose "what you'd change per deployment", not "every internal constant"

#### Verify: `uv run pytest tests/ -v` (full regression)

---

## Execution Order and Dependencies

```
Phase 1 (simplification)     ─── no dependencies, start immediately
  G7 phantom indexers          ─── 1 hour
  G4 Leiden simplification     ─── 2 hours
  G5 betweenness removal       ─── 1 hour

Phase 2 (MCP tools)          ─── depends on Phase 1 for Leiden changes
  G1.1 directed_path           ─── 2 hours
  G1.2 impact_analysis         ─── 2 hours
  G1.3 leiden_clusters         ─── 1 hour
  G1.4 steiner_path            ─── 1 hour
  G1.5 find_callers            ─── 1 hour

Phase 3 (outbox)             ─── independent of Phases 1-2
  G2.1 outbox table            ─── 1 hour
  G2.2 publisher refactor      ─── 2 hours
  G2.3 worker integration      ─── 1 hour

Phase 4 (incremental)        ─── independent, highest complexity
  G3.1 blob_oid diffing        ─── 3 hours
  G3.2 tombstone+reinsert      ─── 4 hours
  G3.3 retrieval increment     ─── 2 hours

Phase 5 (session prefix)     ─── depends on Phase 1 for projection format
  G11.1 prefix endpoint        ─── 1 hour
  G11.2 consumption docs       ─── 1 hour

Phase 6 (TerminusDB graph)   ─── depends on Phase 3 (outbox)
  G12.1 edge publishing        ─── 3 hours
  G12.2 graph queries          ─── 2 hours

Phase 7 (SCIP bridge)        ─── independent
  G8.1 3-strategy cascade      ─── 3 hours
  G8.2 confidence bands        ─── 1 hour

Phase 8 (config)             ─── do last, after all features stabilize
  G6.1 config reduction        ─── 2 hours
```

## Estimated Total: ~40 hours

## Deferred Items

| Item | Reason |
|------|--------|
| G9 Collection prior | Functionally achieved via affinity scoring. Not blocking. |
| G10 Doc type weights | Low impact. Can add when doc ingestion sees real use. |
| G13 Projection format | Research item, not blocking. JSON works. |
| G14 FTS code-awareness | Architecture says ripgrep at agent layer, not server. |
| G15 TerminusDB benchmark | Validation, not implementation. |
| G16 Leiden tuning | Needs real repo benchmarks after G4 simplification. |

## What's Already Done Well (No Action Needed)

- IR models: IRCodeUnit with anchor_set, source_set, provenance tracking
- Precision-anchored merge: max weight matching, not "SCIP wins"
- Adaptive RRF: continuous sigmoid, multi-factor, per-channel K
- Doc→code projection: trace links, noisy-or prior, bounded mixture
- Seed merge: deduplicated, provenance-preserving
- 2-hop graph expansion: IR backend with NetworkX
- Branch manifests: PostgreSQL tables with lineage
- Query processing: LLM-enhanced, multi-turn, intent detection
- Projection L0/L1/L2: cached, on-demand, Leiden+arborescence
- Doc ingestion: semantic chunking, entity extraction
