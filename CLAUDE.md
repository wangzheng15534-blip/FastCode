# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
# Install (editable with dev deps)
uv pip install -e ".[dev]"

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_ir_core.py -v

# Filter tests by name
uv run pytest tests/ -v -k "fusion"

# Run with coverage
uv run pytest tests/ --cov=fastcode

# Run benchmarks
uv run pytest tests/bench_*.py -v --benchmark-only
```

## What FastCode Is

FastCode is an **adaptive knowledge graph with git features for AI agents**. Based on the HKUDS scouting-first framework (arxiv:2603.01012v2, Zongwei Li et al.) — preserving the original's three components (semantic-structural representation, structure-aware navigation, cost-aware context management) while hardening for small-team production use.

AI coding agents explore codebases by reading files one by one — slow, shallow, miss structural relationships. FastCode pre-builds code understanding into a queryable knowledge graph so agents get precise answers fast.

Design principle: **canonical facts are truth, graph is derived view**. Code relations (calls, imports, inheritance, containment) are extracted as canonical IR facts (deepest truth), then materialized as derived graph views in TerminusDB for fast traversal. Filesystem views are rendered on-demand from canonical facts, never stored as source of truth.

### Original HKUDS Contributions (Preserved)

| Component | Original | Hardened |
|-----------|----------|----------|
| Semantic-Structural Representation | Hierarchical code units + lightweight metadata | Canonical IR (IRSnapshot/IRDocument/IRSymbol/IREdge) + SCIP + provenance |
| Structure-Aware Navigation | G_dep, G_inh, G_call graph layers | Derived graph views in TerminusDB, loaded into NetworkX for algorithms |
| Cost-Aware Context Management | Dynamic budget B, epistemic confidence κ, IGR, priority P(u) | Adaptive RRF with sigmoid continuous K, Steiner Tree minimal subgraph |
| Multi-Grained Hybrid Indexing | BM25 + dense embeddings (dual index) | pgvector HNSW + GIN FTS + graph expansion + per-channel adaptive K |

## Architecture

Based on the HKUDS scouting-first framework (arxiv:2603.01012v2) — preserving semantic-structural representation, structure-aware navigation, and cost-aware context management. Hardened with SCIP industry standard and multi-branch support.

### Dual-Source Extraction

Two complementary pipelines merge into Canonical IR:

- **SCIP** (precise, deterministic): Symbol identity, definitions, references. 8 languages. Compiler-derived.
- **Original FastCode LLM + Tree-sitter** (semantic, probabilistic): Structural analysis, semantic understanding, code comprehension. 10+ languages. LLM-augmented.

SCIP = "what exists, where it's defined, where it's referenced." Original FC = "what it means, how it relates, why it's structured this way."

### Three-Layer Storage Architecture

| Layer | Backend | Role | Scope |
|-------|---------|------|-------|
| **Canonical facts** | PostgreSQL | Deepest truth: IR facts per snapshot, provenance, branch manifests, vectors, projections | Always |
| **Derived graph views** | TerminusDB | Branch-aware graph for fast traversal. NOT sole truth — derived from canonical facts, confidence-weighted, refreshable | Always in production |
| **Retrieval indices** | PostgreSQL + agent rg | pgvector HNSW (semantic), JSONB (projections), GIN (docs/metadata). Exact code search = ripgrep at agent workspace, not server index | Always |
| **Algorithm engine** | NetworkX | Graph algorithms (Leiden on symmetrized, arborescence on directed, PageRank+degree) against TerminusDB data | On-demand, cached |
| **Docs overlay** | LadybugDB | Architecture docs, ADRs with MENTIONS edges to code. Branch-stable. Off critical path | Optional |

### Data Flow: Index Pipeline (Dual-Source + Branch-Aware)

**Extraction (dual-source, per changed file on incremental update):**

1. **SCIP extract** → `adapters/scip_to_ir.build_ir_from_scip()` — precise symbol identity, definitions, references (8 languages)
2. **Original FC extract** → `adapters/ast_to_ir.build_ir_from_ast()` (tree-sitter structure) + LLM semantic augmentation — semantic understanding, relationships, summaries (10+ languages)
3. **Merge** → `ir_merge.merge_ir()` applies four rules: SCIP wins on overlap (A), original FC fills gaps (B), edges coexist (C), SCIP refs deduplicated first (D)
4. **Validate** → `ir_validators.validate_snapshot()` checks IR integrity
5. **Persist graph** → TerminusDB (branch-aware: nodes, edges, incremental updates via outbox pattern)
6. **Persist metadata** → PostgreSQL (vectors, FTS, projections, branch manifests)

**Document pipeline (probabilistic, optional):**
1. **Ingest docs** → `KeyDocIngester` parses design docs, ADRs, research
2. **LLM extraction** → Named entity recognition from prose
3. **SCIP Resolution Bridge** → Map doc entities to code SCIP IDs (lexical → namespace → vector semantic)
4. **Store** → LadybugDB (doc graph with MENTIONS edges) + PostgreSQL (doc chunks)

**Branch-awareness:** TerminusDB provides git-like graph branching. Each branch has its own graph view. Content-addressable dedup shares unchanged nodes/edges across branches. Incremental updates only re-extract changed files (blob_oid diff).

### Scouting Workflow (HKUDS Section 3.2)

**Retrieval and graph expansion are complementary parts of one scouting loop — not separate layers.** The HKUDS paper integrates them: retrieval finds initial candidates, graph expansion enriches them with structural context, the agent iterates.

**Scouting loop:**
`FastCode.query()` or `FastCode.query_snapshot()`:
1. **Resolve branch** → branch → snapshot → query context
2. **Process query** → `QueryProcessor` extracts intent, keywords, rewrites, collection prior (π_code, π_doc)
3. **Retrieve** → parallel over two collections:
   - Code collection: semantic (pgvector HNSW) + keyword (BM25 GIN) → intra-collection score fusion
   - Doc collection: semantic + keyword → intra-collection score fusion
4. **Cross-collection fusion** → weighted RRF (w_code ∝ π_code, w_doc ∝ π_doc)
5. **Doc→code projection** → grounded mentions → noisy-or doc prior D(v|q), bounded [0,1] (deterministic, no LLM)
6. **Seed merge** → bounded mixture: seed(v) = (1-β)·Ŝ_code(v) + β·D(v), β = β_max · π_doc(q)
7. **Graph expansion** → 2-hop from seeds via G_dep/G_inh/G_call (code-only, docs never enter)
8. **Return** → mixed evidence slate with provenance (code_direct | code_doc_projected | code_graph_expanded | doc_support)

**Agent graph tools (deeper exploration on demand):**
When retrieval + expansion isn't enough, agent calls MCP graph tools:
- `directed_path(from, to)` → directed shortest path through call graph
- `impact_analysis(symbol_id)` → callers + dependents (directed traversal)
- `leiden_clusters(snapshot_id)` → module boundaries (from symmetrized graph)
- `find_callers(symbol_id)` → who calls this symbol
- `steiner_path(from, to, ...)` → small undirected explanatory subgraph (limited use)

**Session Prefix:**
L0/L1 projection (compact JSON) loaded as system prompt prefix at session start. Agent starts every conversation already knowing codebase architecture. No query needed for architectural context.

**Graph reduction at query time:**
- Steiner Tree: minimal path connecting query-relevant symbols (optimal LLM context)
- Leiden: community detection for architecture recovery
- MST/Arborescence: execution backbone from cyclic dependency graph
- Centrality: representative node selection for summaries

### Canonical IR (`fastcode/semantic_ir.py`)

All extraction goes through the IR before storage. Five dataclass models with `to_dict()`/`from_dict()` serialization:

- `IRSnapshot` — top-level container (repo_name, snapshot_id, commit_id, branch, tree_id)
- `IRDocument` — file (doc_id, path, language, blob_oid, source_set)
- `IRSymbol` — symbol (symbol_id, display_name, kind, qualified_name, source_priority, source_set)
- `IROccurrence` — symbol occurrence in a document (role, range, source)
- `IREdge` — relationship between nodes (edge_type, src_id, dst_id, confidence)

Snapshot identity format: `snap:{repo_name}:{commit_id}`

### Key Subsystems

- **`HybridRetriever`** (`retriever.py`): Multi-stage retrieval with adaptive RRF fusion. Per-channel sigmoid continuous K (k_code, k_doc). Alpha modulated by query entropy, domain keyword affinity, confidence. Agency mode for iterative refinement.
- **`PgRetrievalStore`** (`pg_retrieval.py`): PostgreSQL with pgvector HNSW for vector search and GIN for full-text
- **`TerminusPublisher`** (`terminus_publisher.py`): Git-like knowledge graph. Publishes lineage to TerminusDB via outbox pattern (local intent → async retry with idempotent upsert). Branch/commit/snapshot lineage, symbol versioning.
- **`ProjectionTransformer`** (`projection_transform.py`): On-demand L0/L1/L2 projections from graph algorithms (Leiden, Steiner, MST, SNAP). Cached by snapshot_id. LLM optional for labels, deterministic fallback via centrality.
- **`GraphRuntime`** (`graph_runtime.py`): LadybugDB overlay for doc-code graph. Design docs, ADRs with MENTIONS edges to code symbols. Can ATTACH PostgreSQL. Optional.
- **`RedoWorker`** (`redo_worker.py`): Background task consumer for failed index runs
- **`DBRuntime`** (`db_runtime.py`): PostgreSQL-primary abstraction with connection pooling, `connect()` context manager, `execute()`, `row_to_dict()`

### API Surface

- `api.py` — FastAPI REST endpoints (port 8000/8001)
- `web_app.py` — Flask web UI (port 5000)
- `mcp_server.py` — MCP server for AI coding assistants (Cursor, Claude Code, Windsurf)

## Testing Patterns

- **Shared conftest.py** — `fastcode/tests/conftest.py` provides Hypothesis strategies and pytest fixtures used across the suite. Factory functions (`_make_snapshot`, `_make_scip_payload`, `_make_code_elements`) and Hypothesis strategies (`snapshot_st`, `connected_snapshot_st`, etc.) are defined here. Individual test files import what they need.
- **Factory pattern**: `_make_snapshot()`, `_sample_snapshot()`, `_mk_row()` create minimal test data
- **Fake classes**: `_FakeCursor`, `_DummyEmbedder`, `_FakeFastCode` for isolated testing
- **Partial construction**: `FastCode.__new__(FastCode)` + manual attribute setting to bypass `__init__`
- **PostgreSQL tests**: Use `@pytest.mark.skipif` for graceful degradation when PG/Ollama unavailable
- **Benchmarks**: `bench_*.py` files use `pytest-benchmark`
- **E2E**: `test_e2e_indexing.py` tests against real Ollama + PostgreSQL

### xdist-Safe Testing

Tests run under `pytest-xdist` with `-n auto` (forked workers). Heavy ML models (SentenceTransformer, torch) are not fork-safe — loading them in a forked worker causes segfaults.

**Pattern for tests that call code with lazy model loading:**
- If the test only checks structural properties (metadata, keys, line numbers), disable model loading by mocking the lazy initializer:
  ```python
  ingester = _make_ingester()
  ingester._ensure_chunker = lambda: None  # Skip SentenceTransformer, use word-based fallback
  ```
- If the test genuinely needs the ML model (similarity thresholds, semantic boundaries), mark it `@pytest.mark.integration` — these tests load real models and may be flaky under xdist.
- Mark previously-crashing tests with `@pytest.mark.regression` to guard against regressions.

## Environment Variables

Required for LLM features: `OPENAI_API_KEY`, `MODEL`, `BASE_URL`

Optional:
- `FASTCODE_STORAGE_BACKEND` — "sqlite" (default) or "postgres"
- `FASTCODE_POSTGRES_DSN` — PostgreSQL connection string
- `FASTCODE_PROJECTION_POSTGRES_DSN` — separate DSN for projection store
- `NANOBOT_MODEL` — model for nanobot integration

## Python Version

Requires Python >=3.11. LadybugDB wheels only support up to 3.11 — do not upgrade to 3.12+ without verifying LadybugDB compatibility.

## Build & Install

```bash
uv pip install -e ".[dev]"     # dev deps (pytest, pytest-benchmark, etc.)
uv pip install -e ".[ladybug]" # optional graph backend
```

## Configuration

Default config in `FastCode._get_default_config()`. Override via `config/config.yaml` or pass `config_path` to constructor. Key sections: `storage`, `embedding`, `retrieval`, `generation`, `evaluation`, `projection`, `terminus`.

Evaluation mode: set `evaluation.enabled=true` for in-memory, cache-disabled, force-reindex behavior.

## Docker

Two-service architecture in `docker-compose.yml`:
- **fastcode**: Python app on port 8001
- **nanobot**: Gateway on port 18791, depends on fastcode

## Documented Solutions

`docs/solutions/` — documented solutions to past problems (bugs, best practices, workflow patterns), organized by category with YAML frontmatter (`module`, `tags`, `problem_type`). Relevant when implementing or debugging in documented areas.
