# Reference: Doc Parsing Integration & Retrieval Fusion

**Date:** 2026-04-01
**Status:** Draft
**Supersedes:** `design-kuzu-doc-parsing-integration.md`

---

## 1. Architecture: four backends, each with a distinct role

```
DBRuntime (SQLite / PG)     → relational: snapshots, manifests, index runs, projections
GraphRuntime (LadybugDB)    → graph: code entities, edges, doc nodes, Cypher queries
PgRetrievalStore (pgvector) → vector: HNSW embeddings, GIN full-text
NetworkX (in-memory)        → algorithms: Leiden, MST, Steiner, centrality
```

GraphRuntime is a **peer** of DBRuntime, not a subclass. Both are accepted by SnapshotStore:

```python
# fastcode/snapshot_store.py — current constructor (line ~30)
class SnapshotStore:
    def __init__(self, base_path: str):
        self.db_runtime = DBRuntime.from_storage_config(...)

# Target: add optional graph_runtime parameter
class SnapshotStore:
    def __init__(self, base_path: str, graph_runtime: Optional[GraphRuntime] = None):
        self.db_runtime = DBRuntime.from_storage_config(...)
        self.graph_runtime = graph_runtime  # None = NetworkX-only mode
```

---

## 2. Storage decision: PG Primary + LadybugDB Optional

### Decision

PostgreSQL remains the **primary** data store for all existing functionality (snapshots, manifests, index runs, projections, git backbone, embeddings, FTS). LadybugDB is an **optional** graph backend, enabled via config, for graph-native doc-code traversal and multi-hop entity queries.

### Why PG as primary

| Capability | Location | Status |
|---|---|---|
| HNSW vector search | `pg_retrieval.py` `embedding_vectors` table | Production-ready, 1024-dim |
| GIN full-text search | `pg_retrieval.py` `search_documents` table | Production-ready |
| Symbol/occurrence/edge tables | `snapshot_store.py` schema init | Wired and tested |
| Connection pooling | `db_runtime.py` psycopg_pool | Configured (1-8 conns) |
| Dual-mode retrieval | `semantic_search()` + `keyword_search()` | Hybrid scoring |

All infrastructure already exists. Zero new dependencies for PG path.

### Why LadybugDB (not Kuzu)

**Kuzu is archived.** Apple acquired Kuzu in October 2025, the GitHub repo (`kuzudb/kuzu`) is read-only, the website is down, and no community fork has emerged. See:
- [9to5Mac - Kuzu joins Apple's acquisitions](https://9to5mac.com/2026/02/11/kuzu-database-company-joins-apples-list-of-recent-acquisitions/)
- [The Verge - Apple quietly acquired Kuzu](https://www.theverge.com/tech/877360/apple-quietly-acquired-the-graph-database-company-kuzu-late-last-year)
- [GitHub - kuzudb/kuzu (archived)](https://github.com/kuzudb/kuzu)

**LadybugDB is Kuzu's active successor:**

| Aspect | Detail |
|---|---|
| Origin | MIT fork of Kuzu, active since Oct 2025 |
| Install | `pip install real_ladybug` |
| Native vector index | Built-in |
| Native FTS | Built-in |
| Multi-core parallelism | Fixed Kuzu's single-threaded limit |
| Python support | CPython 3.7–3.11 (venv is 3.10 — compatible) |
| Cypher dialect | Identical to Kuzu — schema translates 1:1 |
| License | MIT (permissive, no rug-pull risk) |

See: [LadybugDB GitHub](https://github.com/LadybugDB/ladybug), [LadybugDB Docs](https://docs.ladybugdb.com/get-started/cypher-intro/)

### Config-driven enablement

```yaml
# config.yaml
graph:
  enabled: false              # default: off, zero new deps
  backend: "ladybugdb"        # future: other embedded graph DBs
  db_path: "~/.fastcode/graph.db"
```

When `graph.enabled: false`, FastCode runs entirely on PG + NetworkX. No LadybugDB import, no startup cost.

### Risk: Python 3.10 max

LadybugDB wheels support CPython 3.7–3.11. The venv is 3.10 (compatible), but upgrading to 3.12+ would require building from source. Pin venv Python version in CI.

---

## 3. LadybugDB: capabilities, limitations, and reference implementation

### Why LadybugDB (not Neo4j/FalkorDB)

Embedded, single-file, no server process — matches FastCode's local-first philosophy (same role as SQLite). Native vector search (`FLOAT[]` + `COSINE`). Native FTS. Cypher query language. Multi-core parallelism.

### Known limitations (inherited from Kuzu heritage)

1. No `UNWIND` — bulk inserts must loop
2. No `ALTER TABLE ADD COLUMN` at runtime — schema versioning needed
3. RelatesToNode_ workaround — edge property pattern doubles every logical hop to 2 physical hops
4. Python wheels capped at 3.11 — see §2 risk

### graphiti's Kuzu driver — reference implementation

**Location:** `/home/jacob/develop/graphiti/graphiti_core/driver/kuzu/`

Key patterns to borrow (with line numbers from graphiti source):

| Pattern | graphiti location | What to borrow |
|---------|-------------------|----------------|
| Schema declaration | `kuzu_driver.py:54-132` | `CREATE NODE TABLE` / `CREATE REL TABLE` at startup |
| RelatesToNode_ workaround | `kuzu_driver.py` throughout | Intermediate node for edge properties (2 physical hops = 1 logical) |
| Vector Cypher | `kuzu_driver.py` search methods | `CAST($vec AS FLOAT[N])` with cosine function |
| BFS depth doubling | `kuzu_driver.py` traversal queries | `[:REL*2..{depth*2}]` to account for 2-hop workaround |
| Bulk operations | `kuzu_driver.py` save methods | No UNWIND, iterate individually |
| Async connection | `kuzu_driver.py:__init__` | `kuzu.AsyncConnection` for non-blocking queries |

**LadybugDB dependency:** `pip install real_ladybug`

---

## 4. LadybugDB Schema

### Node tables

Maps directly to FastCode's IR model:

| LadybugDB node | IR source | Key fields from IR |
|---|---|---|
| `CodeEntity` | `fastcode/semantic_ir.py:IRSymbol` | `symbol_id`, `name`, `kind`, `language`, `path`, `display_name`, `start_line`, `end_line`, `source_set` |
| `CodeDocument` | `fastcode/semantic_ir.py:IRDocument` | `doc_id`, `path`, `language` |
| `DesignDocument` | **new** — parsed repo docs | `doc_id`, `title`, `file_path`, `content`, `chunk_id`, `doc_type`, `content_embedding` |
| `CodeSnapshot` | `fastcode/semantic_ir.py:IRSnapshot` | `snapshot_id`, `repo_name`, `branch`, `commit_id` |

```cypher
CREATE NODE TABLE IF NOT EXISTS CodeEntity (
    entity_id     STRING PRIMARY KEY,
    name          STRING,
    kind          STRING,
    language      STRING,
    file_path     STRING,
    display_name  STRING,
    start_line    INT64,
    end_line      INT64,
    source_set    STRING[],
    snapshot_id   STRING,
    repo_name     STRING,
    name_embedding FLOAT[],
    created_at    TIMESTAMP
);

CREATE NODE TABLE IF NOT EXISTS CodeDocument (
    doc_id        STRING PRIMARY KEY,
    path          STRING,
    language      STRING,
    snapshot_id   STRING,
    repo_name     STRING,
    created_at    TIMESTAMP
);

CREATE NODE TABLE IF NOT EXISTS DesignDocument (
    doc_id            STRING PRIMARY KEY,
    title             STRING,
    file_path         STRING,
    content           STRING,
    chunk_id          STRING,
    doc_type          STRING,
    repo_name         STRING,
    snapshot_id       STRING,
    content_embedding FLOAT[],
    created_at        TIMESTAMP
);

CREATE NODE TABLE IF NOT EXISTS CodeSnapshot (
    snapshot_id   STRING PRIMARY KEY,
    repo_name     STRING,
    branch        STRING,
    commit_id     STRING,
    created_at    TIMESTAMP
);
```

### Relationship tables

Maps directly to `fastcode/ir_graph_builder.py:IRGraphs`:

| LadybugDB rel | IR source | Direction |
|---|---|---|
| `CONTAINS` | containment_graph | `CodeDocument → CodeEntity` |
| `CALLS` | call_graph | `CodeEntity → CodeEntity` |
| `DEPENDS_ON` | dependency_graph | `CodeEntity → CodeEntity` |
| `INHERITS` | inheritance_graph | `CodeEntity → CodeEntity` |
| `REFERENCES` | reference_graph | `CodeEntity → CodeEntity` |
| `BELONGS_TO_SNAPSHOT` | IRSnapshot grouping | `CodeEntity → CodeSnapshot` |
| `MENTIONS` + `MENTIONS_CODE` | **new** — doc→code links | `DesignDocument → MentionsCode_ → CodeEntity` |

```cypher
CREATE REL TABLE IF NOT EXISTS CONTAINS (FROM CodeDocument TO CodeEntity);
CREATE REL TABLE IF NOT EXISTS CALLS (FROM CodeEntity TO CodeEntity);
CREATE REL TABLE IF NOT EXISTS DEPENDS_ON (FROM CodeEntity TO CodeEntity);
CREATE REL TABLE IF NOT EXISTS INHERITS (FROM CodeEntity TO CodeEntity);
CREATE REL TABLE IF NOT EXISTS REFERENCES (FROM CodeEntity TO CodeEntity);
CREATE REL TABLE IF NOT EXISTS BELONGS_TO_SNAPSHOT (FROM CodeEntity TO CodeSnapshot);

-- RelatesToNode_ workaround for doc→code links (LadybugDB edge property limitation)
CREATE NODE TABLE IF NOT EXISTS MentionsCode_ (dummy STRING);
CREATE REL TABLE IF NOT EXISTS MENTIONS (FROM DesignDocument TO MentionsCode_);
CREATE REL TABLE IF NOT EXISTS MENTIONS_CODE (FROM MentionsCode_ TO CodeEntity);
```

---

## 5. Document Parsing: borrow from MemOS

### Source files to borrow

| MemOS source | What it does | FastCode target |
|---|---|---|
| `src/memos/parsers/markitdown.py` | Wraps `markitdown.MarkItDown.convert()`, returns markdown text | `fastcode/parsers/markitdown_parser.py` |
| `src/memos/chunkers/sentence_chunker.py` | Chonkie sentence-based chunking, URL protection, configurable tokenizer | `fastcode/chunkers/sentence_chunker.py` |
| `src/memos/chunkers/markdown_chunker.py` | LangChain `MarkdownHeaderTextSplitter`, header hierarchy auto-fix, URL protection | `fastcode/chunkers/markdown_chunker.py` |
| `src/memos/chunkers/base.py` | `BaseChunker` ABC, `Chunk` dataclass, `protect_urls()`/`restore_urls()` | `fastcode/chunkers/base.py` |
| `src/memos/parsers/base.py` | `BaseParser` ABC with `parse(file_path) -> str` | `fastcode/parsers/base.py` |
| `src/memos/chunkers/factory.py` | `ChunkerFactory` config-driven selection | `fastcode/chunkers/factory.py` |
| `src/memos/parsers/factory.py` | `ParserFactory` config-driven selection | `fastcode/parsers/factory.py` |
| `src/memos/mem_reader/read_multi_modal/file_content_parser.py` | Orchestrator: parse → chunk → embed (strip LLM extraction, image processing) | `fastcode/doc_ingester.py` (simplified) |

### What NOT to borrow

- **LLM memory extraction** (`FileContentParser.parse_fine`) — FastCode uses AST/SCIP
- **Image processing** (`ImageParser`) — not needed for code intelligence
- **Memory item models** (`TextualMemoryItem`) — FastCode has its own IR models

### MemOS dependency versions

From `/home/jacob/develop/MemOS/pyproject.toml`:
```toml
[project.optional-dependencies]
mem-reader = [
    "chonkie (>=1.0.7,<2.0.0)",
    "markitdown[docx,pdf,pptx,xls,xlsx] (>=0.1.1,<0.2.0)",
    "langchain-text-splitters (>=1.0.0,<2.0.0)",
]
```

### Pipeline flow

```
Repo docs (README.md, docs/**/*.md, docs/design/**/*.md, docs/research/**/*.md)
  → MarkItDown parser  (PDF/DOCX/PPTX → markdown; .md → passthrough)
  → MarkdownChunker    (split on H1/H2, 512 tokens, 128 overlap)
  → Embed              (existing embedder in fastcode/embedder.py)
  → Store in PG + LadybugDB  (DesignDocument nodes + embedding_vectors table)
  → Auto-link          (scan chunks for symbol names → MENTIONS edges in LadybugDB)
```

### Doc type detection by path

```
docs/design/**, docs/arch/**    → "design"
docs/research/**, docs/rfc/**   → "research"
README.md, docs/**/README.md    → "readme"
docs/adr/**, docs/decisions/**  → "adr"
```

### FastCode-specific additions to MemOS chunkers

1. **Code fence protection** — don't split inside ``` blocks (MemOS doesn't handle this)
2. **Symbol name scanning** — after chunking, match `display_name` values from `IRSnapshot.symbols` to create MENTIONS edges
3. **Snapshot scoping** — tag each `DesignDocument` with `snapshot_id` for versioned queries

---

## 6. Retrieval Fusion Architecture

### Decision: RRF with per-source adaptive continuous k (Heuristic v1)

**Not** a single weight or a hard router. **Not** a learned model (no training data yet).

### Why RRF

Code semantic scores (cosine ~0.6-0.95), BM25 scores (~0-15), and doc/graph scores are **heterogeneous distributions**. A single multiplier assumes they're comparable — they're not.

RRF works on **ranks**, not raw scores:

```
RRF_score(d) = Σ  1 / (k_i(q) + rank_i(d))
                  i∈sources
```

- No normalization needed — ranks are already comparable across sources
- Adding a new source = adding one more term to the sum
- `k` controls top-heaviness per source: low k = only top ranks matter, high k = egalitarian

### Why per-source adaptive continuous k

Multiple weak query signals combine into a smooth, continuous k value per source. No hard thresholds, no cliff effects, no single point of failure.

```python
@dataclass
class FusionConfig:
    """Per-source base k values — tuned once, then adapted at query time."""
    code_semantic_k: float = 60.0
    code_bm25_k: float = 60.0
    doc_vector_k: float = 60.0
    doc_fts_k: float = 60.0
    graph_traversal_k: float = 60.0


def _compute_k(self, base_k: float, query_info: ProcessedQuery,
               source_results: List[Dict], source_type: str) -> float:
    """Continuous k(q) — smooth function of query signals.

    Returns float. RRF works fine with non-integer k.
    Low k  → top-heavy, only rank 1-3 matter.
    High k → egalitarian, rank 30 still contributes.
    """
    k = float(base_k)

    # Signal A: source confidence (continuous sigmoid, no step)
    if source_results:
        top_score = source_results[0].get("score", 0.0)
        k -= 25.0 * _sigmoid(top_score - 0.5)  # smooth decay
    else:
        k += 80.0  # no results → neutralize

    # Signal B: domain keyword affinity (continuous overlap ratio)
    code_terms = {"function", "class", "method", "variable", "return", "import"}
    doc_terms = {"architecture", "design", "decision", "why", "overview",
                 "rationale", "convention", "guideline", "adr"}

    query_tokens = set(query_info.original.lower().split())
    code_affinity = len(query_tokens & code_terms) / max(len(query_tokens), 1)
    doc_affinity = len(query_tokens & doc_terms) / max(len(query_tokens), 1)

    if source_type == "code":
        k -= 20.0 * (code_affinity - doc_affinity)
    else:
        k -= 20.0 * (doc_affinity - code_affinity)

    # Signal C: query entropy proxy (continuous)
    # High entropy (many unique terms) → complex query → broaden all sources
    entropy = _query_entropy(query_info.original)
    k += 10.0 * entropy

    return max(k, 5.0)  # floor to avoid 1/(5+1) dominance


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
```

### Fusion pipeline

```
                    query
                      │
          ┌───────────┼───────────┬──────────────┐
          ▼           ▼           ▼              ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ PG Vector│ │ PG BM25  │ │Ladybug   │ │Ladybug   │
    │ (code    │ │ (code    │ │ Vector   │ │ FTS      │
    │ semantic)│ │ keyword) │ │ (docs)   │ │ (docs)   │
    └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
         │ranked       │ranked       │ranked       │ranked
         └─────────────┴──────┬──────┴─────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Per-Source       │
                    │  Adaptive k       │  ← k modulated by query signals
                    │  RRF Fusion       │     per source independently
                    └─────────┬─────────┘
                              │ unified ranked list
                    ┌─────────▼─────────┐
                    │  Graph Expansion  │  ← LadybugDB Cypher: expand top N
                    │  (post-fusion)    │     with MENTIONS/CALLS/DEPENDS edges
                    └─────────┬─────────┘
                              │ enriched results
                    ┌─────────▼─────────┐
                    │  Proportional     │  ← budget = f(rank_mass per source)
                    │  Context Budget   │     not fixed slots
                    └─────────┬─────────┘
                              │
                         LLM prompt
```

### Context budget: proportional to rank mass

```python
def _allocate_budget(self, fused: List[Dict], total_tokens: int = 4000) -> str:
    """Proportional budget: code with 70% of rank mass gets 70% of tokens."""
    code_mass = sum(r["rrf_score"] for r in fused if r["source_type"] == "code")
    doc_mass = sum(r["rrf_score"] for r in fused if r["source_type"] == "doc")
    total_mass = code_mass + doc_mass or 1

    code_tokens = int(total_tokens * code_mass / total_mass)
    doc_tokens = total_tokens - code_tokens
    ...
```

"Find the auth function" naturally allocates ~95% tokens to code. "Why does the cache use Redis?" might go 60/40 or 40/60 depending on retrieval quality. No routing decision — the scores decide.

### Comparison with alternatives

| Approach | Heterogeneous scores? | Handles ambiguity? | Add new source? | Training data? |
|---|---|---|---|---|
| Single weight | Needs normalization | No (binary) | Retune weight | No |
| Hard router | N/A | No | Add route | No |
| Learned Light Model | Yes | Yes | Retrain | Yes (500+ queries) |
| Full LTR (LambdaMART) | Yes | Yes | Retrain | Yes (10K+ queries) |
| **Per-source adaptive RRF** | **Ranks, comparable** | **Yes (continuous k)** | **Add one term** | **No** |

### Fusion evolution roadmap

| Phase | Fusion strategy | Trigger |
|---|---|---|
| **v1** | Heuristic adaptive RRF + continuous k | Ships day one, zero data needed |
| **v1.5** | Add telemetry: log `k_i(q)` values, RRF scores, implicit relevance (did user quote a source?) | After 500+ queries |
| **v2** | If heuristic thresholds show systematic bias → train Learned Light Model (gradient boosted trees, ~50 features, offline training) | When telemetry reveals patterns |
| **v3** | Full LTR only if competing on retrieval benchmarks | Probably never for a dev tool |

---

## 7. New files

```
fastcode/
├── graph_runtime.py
├── retrieval_fusion.py          # NEW: RRF + adaptive k fusion
├── doc_ingester.py
├── parsers/
│   ├── __init__.py
│   ├── base.py
│   ├── markitdown_parser.py
│   └── factory.py
└── chunkers/
    ├── __init__.py
    ├── base.py
    ├── markdown_chunker.py
    ├── sentence_chunker.py
    └── factory.py

tests/
├── test_graph_runtime.py
├── test_retrieval_fusion.py      # NEW: fusion unit tests
├── test_doc_ingester.py
├── test_markdown_chunker.py
└── test_sentence_chunker.py
```

---

## 8. Files to modify

| File | Change |
|------|--------|
| `fastcode/main.py` | Init `GraphRuntime` in `__init__()` (if enabled), trigger doc ingestion after index pipeline, stop in `shutdown()` |
| `fastcode/snapshot_store.py` | Accept `graph_runtime` param, call `sync_snapshot()` after `save_snapshot()` |
| `fastcode/projection_transform.py` | Query LadybugDB for doc-code links, include in L1/L2 output |
| `fastcode/retriever.py` | Replace `_combine_results()` with RRF fusion from `retrieval_fusion.py`; add LadybugDB retrieval path |
| `fastcode/ir_graph_builder.py` | No change — `IRGraphs` stays as NetworkX source for `sync_snapshot()` |
| `api.py` | Add `GET /graph/query` endpoint for raw Cypher queries |

---

## 9. Integration points in existing code

### Where GraphRuntime connects to the current pipeline

```
fastcode/main.py:__init__()
    ├── self.snapshot_store = SnapshotStore(tmp, graph_runtime=graph_runtime)
    ├── self.pg_retrieval_store = PgRetrievalStore(...)
    └── self.graph_runtime = graph_runtime  # NEW (optional)

fastcode/main.py:run_index_pipeline()
    ├── ... existing pipeline ...
    ├── self.snapshot_store.save_snapshot(merged, ...)        # existing
    ├── self.snapshot_store.import_git_backbone(...)           # existing
    ├── self.snapshot_store.save_relational_facts(...)         # existing
    ├── ir_graphs = self.ir_graph_builder.build_graphs(...)    # existing
    ├── self.snapshot_store.save_ir_graphs(...)                # existing
    ├── if self.graph_runtime:                                 # NEW (conditional)
    │   └── self.graph_runtime.sync_snapshot(merged, ir_graphs)
    └── self.doc_ingester.ingest(repo_path, snapshot_id)       # NEW

fastcode/snapshot_store.py:save_snapshot()
    └── after conn.commit():
        if self.graph_runtime:                                  # NEW
            self.graph_runtime.sync_snapshot(snapshot, graphs)
```

### Where doc chunks connect to code entities

```
fastcode/doc_ingester.py:ingest()
    ├── discover docs (glob patterns)
    ├── for each doc:
    │   ├── parse via MarkItDownParser
    │   ├── chunk via MarkdownChunker
    │   ├── embed each chunk
    │   ├── store DesignDocument nodes in LadybugDB (if enabled)
    │   ├── store chunk embeddings in PG embedding_vectors (always)
    │   └── if LadybugDB enabled:
    │       └── scan chunk text for symbol names from snapshot
    │           └── create MENTIONS edges: DesignDocument → MentionsCode_ → CodeEntity
    └── return ingested count
```

### Where retrieval fusion replaces current scoring

```
fastcode/retriever.py:retrieve()
    ├── ... existing query processing ...
    ├── code_semantic = self._semantic_search(...)              # existing
    ├── code_bm25 = self._keyword_search(...)                   # existing
    ├── doc_vector = self.graph_runtime.doc_vector_search(...)  # NEW (if enabled)
    ├── doc_fts = self.graph_runtime.doc_fts_search(...)        # NEW (if enabled)
    ├── fused = fusion.rrf_fuse(                               # NEW (replaces _combine_results)
    │       code_semantic, code_bm25, doc_vector, doc_fts,
    │       query_info, fusion_config)
    ├── expanded = self._expand_with_graph(fused, ...)          # existing (enriches fused)
    └── return self._rerank(query, expanded)                    # existing
```

---

## 10. Implementation phases

### Phase 1: GraphRuntime + LadybugDB schema

- Create `fastcode/graph_runtime.py` with LadybugDB backend + NetworkX fallback
- `_init_schema()` with all node/rel tables from §4
- `sync_snapshot()` maps `IRSnapshot` + `IRGraphs` → LadybugDB nodes/rels
- Wire into `FastCode.__init__()` via config (`graph.enabled: true/false`)
- Tests with `:memory:` database
- Guard all LadybugDB imports behind `graph.enabled` flag

### Phase 2: Document parsing pipeline

- Borrow parser + chunker files from MemOS (§5 source table)
- Create `fastcode/doc_ingester.py` (simplified FileContentParser)
- Code fence protection in MarkdownChunker
- Doc type detection by path pattern
- Store chunks in PG `embedding_vectors` + `search_documents` (always, even without LadybugDB)

### Phase 3: Doc-code auto-linking

- Symbol name scanning in doc chunks → `MENTIONS` edges (LadybugDB only)
- Store `DesignDocument` nodes in LadybugDB with embeddings
- Snapshot scoping for versioned doc queries

### Phase 4: Retrieval fusion

- Create `fastcode/retrieval_fusion.py` with RRF + adaptive continuous k
- Replace `_combine_results()` in `fastcode/retriever.py` with fusion module
- Add LadybugDB doc retrieval path (vector + FTS) alongside existing PG code retrieval
- Graph expansion as post-fusion enrichment (top-N → Cypher traversal)
- Proportional context budget allocation
- Tests: fusion with mocked retrieval sources, edge cases (empty sources, single-source queries)

### Phase 5: Query integration & API

- Extend `api.py` with `GET /graph/query` endpoint for raw Cypher queries
- Include doc references in projection L1/L2 output
- Demo: `demos/demo_graph_runtime.py`

---

## 11. Risks

| Risk | Mitigation |
|------|------------|
| LadybugDB is young (~6 months) | Optional backend, PG fallback, pin `real_ladybug` version |
| Python 3.10 max for LadybugDB wheels | Pin venv Python version, document in CI |
| RelatesToNode_ doubles hop count | Limit traversal depth; use NetworkX for deep analytics |
| No runtime schema changes | Version schema in `_init_schema()`, recreate tables on version bump |
| MarkItDown output quality varies | Validate markdown output, fall back to raw text on failure |
| Chonkie tokenizer mismatch with embedder | Use same tokenizer config as `fastcode/embedder.py` |
| RRF k thresholds need tuning | Start with k=60 defaults, add telemetry in v1.5 to guide tuning |
| Fusion telemetry privacy | Log aggregate k distributions, not raw queries |
