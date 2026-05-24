# FastCode Architecture

## Executive Summary

FastCode is a **branch-aware code knowledge graph for AI agents**. Based on the HKUDS scouting-first framework (arxiv:2603.01012v2), it pre-builds code understanding into a queryable knowledge graph so AI agents get precise answers fast — instead of reading files one by one.

**Core thesis:** Code relationships (calls, imports, inheritance, containment) are graph edges. An AI agent needs to query these relationships, not grep for keywords. The graph must be persistent, incrementally updated, and branch-aware.

**Design principle:** Canonical facts are truth, graph is derived view.

## Package Boundary Note

The implementation package follows the FCIS shell split used by the architecture
tests:

- **app-runtime shell:** workflow/runtime use in `indexing/`, `query/`, and most
  of `store/`;
- **capability ports:** shared external capability contracts under
  `fastcode.ports`. App-runtime code and infrastructure adapters may both
  import these compile-time contracts, but ports do not import either side and do
  not own runtime wiring. Current examples include `StoreDatabaseRuntime`,
  `FileArtifactStore`, `EmbeddingProvider`, and `SemanticHelperRuntime`;
- **infrastructure:** concrete DB, filesystem, network, subprocess,
  native-library, and SDK wrappers such as `store/infrastructure/`.

Do not add package-local `ports.py` modules or domain traits for external
capabilities such as DB, network, filesystem, subprocess, event, queue, or
storage. This is separate from network ports listed in the API surface section.

---

## Three-Layer Architecture

```
Layer 1 — Canonical Facts (deepest truth):
  IRSnapshot per commit: documents, symbols, occurrences, edges, attachments
  Provenance: source (scip/fc_structure), source_priority, confidence
  SCIP symbol identity = canonical identity
  Immutable once published per snapshot
  Storage: PostgreSQL

Layer 2 — Derived Graph Views (TerminusDB):
  Branch-aware graph for fast traversal
  NOT sole truth — derived from facts, confidence-weighted, refreshable
  Git-like branching, content-addressable dedup
  Storage: TerminusDB

Layer 3 — Retrieval Indices:
  pgvector HNSW (semantic), GIN (docs/metadata), JSONB (projections)
  Agent-side ripgrep for exact code search
  Hierarchical fusion: intra-collection score fusion → cross-collection weighted RRF
  Storage: PostgreSQL + ripgrep at agent

── THREE LAYERS, NO CROSS-LAYER FUSION ──
```

---

## Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| Language | Python 3.11+ | Core runtime |
| Package Manager | uv | Dependency management |
| API Framework | FastAPI | REST API (port 8001) |
| Web Framework | Flask | Browser UI (port 5000) |
| MCP | mcp[cli] | AI coding assistant integration |
| CLI | Click | Command-line interface |
| Graph Algorithms | NetworkX | Leiden, Steiner, arborescence, PageRank |
| Graph Storage | TerminusDB | Branch-aware persistent graph |
| Relational DB | PostgreSQL | Canonical facts, vectors, FTS, projections |
| Vector Search | pgvector HNSW | Semantic similarity search |
| Full-Text Search | GIN (PostgreSQL) | BM25 keyword search |
| Tree-sitter | 10 languages | Structural code parsing |
| SCIP | 8 languages | Compiler-derived symbol identity |
| Embeddings | Ollama / sentence-transformers | `all-minilm:l6-v2` |
| LLM | OpenAI / Anthropic | Query processing, answer generation |
| Chunking | Chonkie | Semantic document chunking |
| Caching | diskcache / Redis | Embeddings, queries, sessions |
| Doc Graph | LadybugDB (optional) | Architecture docs with MENTIONS edges |
| Agent Framework | Nanobot (vendored) | Multi-channel AI agent gateway |
| Linting | ruff | Code quality |
| Type Checking | pyright (strict) | Static analysis |
| Testing | pytest + hypothesis + syrupy | Unit, property, snapshot, E2E |
| CI Hooks | lefthook | Pre-commit checks |

---

## Precision-Anchored Extraction

One integrated pipeline — not two competing pipelines. Tree-sitter builds the unit skeleton, SCIP anchors precision onto those units, embeddings/LLM attach as derived features.

```
── PRECISION-ANCHORED EXTRACTION (one pipeline, not two rival pipelines) ──

Tree-sitter/AST → build hierarchical code units (File/Class/Function/Doc)
│   Defines unit skeleton: path, span, signature, docstring, structural relations
│   Structural graphs: G_dep (imports), G_inh (inheritance), G_call (call sites)
│   "Scouting surface" — agent scouts metadata, not full code
│   source_priority: 50, confidence: "resolved"/"heuristic"
│
├── SCIP Anchoring (8 languages, when available)
│   Attach precision anchors onto tree-sitter units
│   Alignment: span overlap + name match + kind compat + container compat
│   NOT hard overwrite — anchored unit, not competing symbol
│   Anchored units get: precise identity, definition locations, cross-file references
│   SCIP does NOT give call graphs — tree-sitter discovers calls, SCIP resolves callees
│   source_priority: 100, confidence: "precise"
│
├── Derived Attachments (embeddings, summaries)
│   Dense vectors over enriched unit text → attachment_kind="embedding"
│   Optional LLM summaries/annotations — retrieval features, not graph facts
│   LLM stays in query augmentation (Section 3.2), NOT in extraction/merge
│   confidence: "derived", no source_priority
│
▼
 Enriched Unit → Canonical IR (Layer 1)
    Each unit: skeleton (tree-sitter) + anchor_set (SCIP) + attachments (embedding/LLM)
```

**Key principle: SCIP does not win; it anchors.** The wrong rule is "if keys overlap, SCIP wins." The right rule is "tree-sitter defines the unit skeleton, SCIP provides the precise anchor, embeddings index the semantic projection of that enriched unit."

**SCIP does not give call graphs.** Tree-sitter discovers call sites. SCIP resolves ambiguous callees when precise symbol identity is available. G_call stays tree-sitter-first. SCIP upgrades `call_candidate` to `call_resolved` when anchor match exists.

**LLM role:** Query augmentation, navigation control, optional unit summaries as annotations — NOT symbol identity or graph facts.

### What Each Source Contributes

| Concern | SCIP | FC Structure (tree-sitter) | FC Embedding / LLM |
|---------|------|----------------------------|--------------------|
| Canonical symbol ID | Yes, authoritative | Fallback only | Never |
| Definitions / references | Yes | Definitions only, mostly local | Never |
| Three structural graphs | No | Yes (G_dep, G_inh, G_call) | No |
| Dense retrieval | No | Supplies anchor text | Yes |
| Query augmentation / summaries | No | Anchor context only | Yes |
| First-class canonical fact? | Yes | Yes | No, attachment/annotation only |

### Adapters

| Adapter | Input | Output | ID Scheme |
|---------|-------|--------|-----------|
| `ast_to_ir.py` | CodeElement[] (tree-sitter) | IRSnapshot with unit skeletons | Blake2b (20 hex) |
| `scip_to_ir.py` | SCIPIndex (protobuf/JSON) | IRSnapshot with precision anchors | MD5 (24 hex) |
| `ir_merge.py` | AST snapshot + SCIP snapshot | Canonical IRSnapshot (anchored) | Composite |

### Edge Reliability Policy

| Edge type | Source | Confidence |
|-----------|--------|------------|
| G_dep (import) | tree-sitter import + optional SCIP package match | medium→high |
| G_inh (inheritance) | tree-sitter base-class + optional SCIP type_definition | medium→high |
| G_call (call) | tree-sitter call-site + SCIP callee resolution | candidate→resolved |
| contain | tree-sitter structure | resolved |
| ref | SCIP definitions/references | precise |

---

## Integrated Scouting Workflow

Retrieval and graph expansion are **complementary parts of one scouting loop** — not separate layers. Retrieval finds initial candidates, graph expansion enriches them with structural context, the agent iterates.

```
Agent session startup:
  L0/L1 projection → system prompt prefix (cached, paid once per session)

Scouting loop (query time — HKUDS Section 3.2):

  Stage 1 — Query Processing:
    Intent, keyword expansion, pseudocode hints
    Collection prior: π_code(q), π_doc(q)

  Stage 2 — Parallel Retrieval (two heterogeneous collections):
    Code collection: semantic (pgvector HNSW) + keyword (BM25 GIN)
    Doc collection:  semantic + keyword
    Both run in parallel. ripgrep NOT in this pipeline.

  Stage 3 — Intra-Collection Fusion (normalized weighted score fusion):
    Within each collection, NOT RRF:
      S_code(u|q) = α_code · norm(code_dense) + (1-α_code) · norm(code_sparse)
      S_doc(d|q)  = α_doc · norm(doc_dense) + (1-α_doc) · norm(doc_sparse)
    Normalized to [0,1]. Score fusion preserves magnitude for projection.

  Stage 4 — Cross-Collection Weighted RRF:
    R(x|q) = w_code/(K+rank_code(x)) + w_doc/(K+rank_doc(x))
    w_code ∝ π_code(q), w_doc ∝ π_doc(q)
    RRF on ranks — correct for heterogeneous score scales.

  Stage 5 — Doc→Code Projection (grounded priors):
    Doc prior via noisy-or: D(v|q) = 1 - Π(1 - Ŝ_doc(d|q) · P̂(v|d))
    Seed via bounded mixture: seed(v|q) = (1-β(q))·Ŝ_code + β(q)·D(v|q)
    β(q) = β_max · π_doc(q). Convex combination, guaranteed [0,1].
    Docs NEVER enter graph as nodes — only project grounded code priors.

  Stage 6 — Graph Expansion (code-only, 2-hop):
    Expand weighted seed set via G_dep/G_inh/G_call.
    Provenance: code_direct | code_doc_projected | code_graph_expanded | doc_support

  Stage 7 — Agent Tools (deeper exploration on demand):
    directed_path(from, to) → call flow (directed)
    impact_analysis(symbol_id) → callers + dependents
    leiden_clusters(snapshot_id) → module boundaries (symmetrized)

  Stage 8 — ripgrep verification (agent workspace):
    Exact textual evidence in checked-out repo.

  Iterate until task done.
```

**Why hierarchical fusion, not flat:** Code and docs are different evidence types with different score distributions. Intra-collection score fusion normalizes within homogeneous collections. Cross-collection weighted RRF merges ranked outputs. Doc→code projection grounds doc evidence into code-space. Each step respects data heterogeneity.

---

## Projection Pipeline

4-step deterministic pipeline (LLM optional for labels):

```
Step 1 — Scope + Weight:
  Build undirected (symmetrized) + directed graphs
  Confidence bands multiply weights

Step 2 — Leiden clustering (on SYMMETRIZED graph):
  Groups symbols into code modules from call density
  Resolution: configurable (default 1.0)

Step 3 — Representative selection:
  PageRank + degree centrality pick representative per cluster

Step 4 — Arborescence backbone (on DIRECTED graph):
  Directed navigation tree for L1 hierarchy + cross-cluster xrefs

Output: L0 (summary), L1 (navigation + backbone), L2 (per-cluster evidence)
```

**Critical policy:** Leiden on symmetrized graph (module discovery). Arborescence on directed graph (navigation tree). Steiner tree only for small explanatory subgraphs.

---

## Core Module Organization

### Indexing Pipeline

| Module | LOC | Role |
|--------|-----|------|
| `indexer.py` | 515 | Multi-level code element extraction |
| `adapters/scip_to_ir.py` | 175 | SCIP → Canonical IR |
| `adapters/ast_to_ir.py` | 294 | Tree-sitter/embeddings → Canonical IR |
| `ir_merge.py` | 177 | Precision-anchored merge (alignment scoring) |
| `ir_validators.py` | 78 | IR integrity validation |
| `scip_loader.py` | 152 | Load SCIP artifacts |
| `scip_indexers.py` | 145 | Run SCIP indexers (8 languages) |

### Storage & Graph

| Module | LOC | Role |
|--------|-----|------|
| `snapshot_store.py` | 1,006 | Snapshot persistence (SQLite/PostgreSQL) |
| `manifest_store.py` | 173 | Branch manifest storage |
| `terminus_publisher.py` | 213 | TerminusDB graph publishing |
| `ir_graph_builder.py` | 84 | IR → NetworkX graphs |
| `graph_runtime.py` | 192 | LadybugDB graph overlay |
| `db_runtime.py` | 126 | PostgreSQL connection pool |
| `pg_retrieval.py` | 480 | pgvector + GIN retrieval |

### Retrieval & Query

| Module | LOC | Role |
|--------|-----|------|
| `retriever.py` | 1,895 | Hybrid retrieval + adaptive RRF |
| `query_processor.py` | 828 | Query intent, keyword extraction |
| `answer_generator.py` | 889 | LLM answer synthesis |
| `iterative_agent.py` | 3,335 | Multi-round retrieval agent |
| `vector_store.py` | 755 | FAISS vector store |
| `embedder.py` | 227 | Embedding generation |

### Projection

| Module | LOC | Role |
|--------|-----|------|
| `projection_transform.py` | 984 | Graph → L0/L1/L2 projections |
| `projection_store.py` | 268 | Projection persistence |
| `projection_models.py` | 62 | Projection data models |

### Supporting

| Module | LOC | Role |
|--------|-----|------|
| `loader.py` | 384 | Repository loading (URL/path/ZIP) |
| `cache.py` | 464 | Disk/Redis caching |
| `doc_ingester.py` | 411 | Key document ingestion |
| `repo_selector.py` | 481 | Multi-repo selection |
| `path_utils.py` | 519 | Path utilities |
| `redo_worker.py` | 85 | Background retry worker |

---

## API Surface

| Interface | Port | Auth | Feature Coverage |
|-----------|------|------|-----------------|
| REST API | 8001 | None | Full (snapshots, graphs, projections, SCIP) |
| Web App | 5777 | None | Subset (legacy query, SSE streaming) |
| MCP Server | stdio/SSE | None | Tools for AI coding assistants |
| CLI | Terminal | None | Index, query, interactive REPL |
| Nanobot | 18791 | None | 10 FastCode tools for multi-channel AI |

---

## Testing Strategy

| Type | Tool | Count | Description |
|------|------|-------|-------------|
| Unit | pytest | ~50 files | Function-level tests |
| Property | hypothesis | ~20 files | Invariant-based testing |
| Snapshot | syrupy | 2 files | Contract testing |
| E2E | pytest | 2 files | Real Ollama + PostgreSQL |
| Benchmark | pytest-benchmark | 3 files | IR merge, graph projection, validation |
| Mutation | mutmut | 1 script | Mutation testing |

**Pattern:** No shared conftest.py. Each test file self-contained with factory functions and fake classes. Partial construction via `FastCode.__new__(FastCode)` + manual attribute setting.

---

## Key Assumptions

| # | Assumption | Risk if Wrong |
|---|-----------|--------------|
| A1 | Graph edges more valuable than flat search for agents | Built graph DB for no benefit |
| A2 | TerminusDB derived views worth operational cost | Two DBs for no benefit |
| A3 | Multi-branch graph awareness needed | Branch feature unused |
| A4 | SCIP precision anchoring measurably improves unit map quality | SCIP indexers add complexity without measurable gain |
| A5 | Session prefix projection provides architectural awareness | Wastes context window |
| A6 | Agents actually use graph MCP tools | Graph layer unused |
