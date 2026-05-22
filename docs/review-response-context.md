# FastCode: Architecture & Algorithm Review Context

## What FastCode Is

FastCode is a **branch-aware code knowledge graph for AI agents**. Based on the HKUDS scouting-first framework (arxiv:2603.01012v2, Zongwei Li et al.) — preserving its three original contributions while hardening with industry-standard SCIP and multi-branch support.

**The original HKUDS FastCode is too simple:** indexes one snapshot, no branch awareness, no persistent graph. Real development has many branches. Real codebases are too large to re-index on every query.

**Core thesis:** Code relationships (calls, imports, inheritance, containment) are graph edges. An AI agent needs to query these relationships, not grep for keywords. The graph must be persistent, incrementally updated, and branch-aware.

## Architecture — Three-Layer Foundation

```
Layer 1 — Canonical Revisioned Facts (deepest truth):
  Canonical IR per snapshot: documents, symbols, occurrences, edges
  Provenance: source (scip/fc_structure/fc_embedding), source_priority, confidence
  SCIP symbol identity = canonical identity
  Immutable once published per snapshot

Layer 2 — Derived Graph Views (TerminusDB):
  Published branch-aware graph for fast traversal
  NOT sole truth — derived, confidence-weighted, refreshable from facts
  Confidence bands: SCIP (precise), Original FC tree-sitter+LLM (semantic, combined pipeline)
  Git-like branching, incremental updates
  Content-addressable dedup: HYPOTHESIS to benchmark, not assumed

Layer 3 — Retrieval Indices:
  PostgreSQL: pgvector HNSW (semantic), JSONB (projections), GIN (docs)
  Workspace exact search: ripgrep at agent layer (NOT server index — agent-with-workspace model)
  Vanilla RRF fusion as foundation, adaptive K as experimental layer

── PRECISION-ANCHORED EXTRACTION (one pipeline, not two rival pipelines) ──

Extraction (tree-sitter-first, SCIP anchors on):
     │
     Tree-sitter/AST → build hierarchical code units (File/Class/Function/Doc)
     │   Defines unit skeleton: path, span, signature, docstring, structural relations
     │   Structural graphs: G_dep (imports), G_inh (inheritance), G_call (call sites)
     │   "Scouting surface" — agent scouts metadata, not full code
     │
     ├── SCIP Anchoring (8 languages, when available)
     │   Attach precision anchors onto tree-sitter units
     │   Alignment: span overlap + name match + kind compat + container compat
     │   NOT hard overwrite — anchored unit, not competing symbol
     │   Anchored units get: precise identity, definition locations, cross-file references
     │   SCIP does NOT give call graphs — tree-sitter discovers calls, SCIP resolves callees
     │
     ├── Derived Attachments (embeddings, summaries)
     │   Dense vectors over enriched unit text
     │   Optional LLM summaries/annotations — retrieval features, not graph facts
     │
     ▼
  Enriched Unit → Canonical IR (Layer 1)
     Each unit: skeleton (tree-sitter) + anchor_set (SCIP) + attachments (embedding/LLM)
     ▼
  TerminusDB derived graph → Layer 2 (graph views)
     G_dep, G_inh, G_call with confidence-weighted edges
     Branch-aware, incrementally updated

  Retrieval indices → Layer 3 (search)
     Code collection: BM25 + CosSim over enriched units
     Doc collection: BM25 + CosSim over doc chunks
     Fused by hierarchical RRF: intra-collection → cross-collection weighted RRF
     Doc→code projection: grounded mentions → code priors for graph seeding

Agent consumption (scouting workflow — HKUDS Section 3.2):
  L0/L1 prefix → session prompt (cached, paid once)
  L2 → cheap explore subagents with tools
  Query → POST /query → hierarchical fusion → seed merge → graph expansion
  Graph tools → MCP: directed_path, impact_analysis, leiden_clusters
  Docs influence graph ONLY through grounded code priors, never as graph nodes
```

## Why This Architecture

### Why three layers, not "graph is truth" or "flat facts rebuild"

GPT review correctly identified a contradiction: docs described both "late-bound from flat facts" and "persistent TerminusDB graph" as truth. Resolved with three-layer foundation:

- **Facts are deepest truth.** Canonical IR per snapshot. Immutable. SCIP symbol identity canonical. This is what Sourcegraph SCIP was designed for — human-readable symbols, precise identity, incremental indexing.
- **Graph is derived view.** Materialized from facts for fast traversal. Confidence-weighted. Refreshable. Not sole truth — if graph gets corrupted, re-derive from facts.
- **Search is independent.** Retrieval indices built from canonical facts, not from graph. Different access pattern (ranked hits vs relationship traversal).

### Why graph as derived view, not sole truth

Original FC (tree-sitter+LLM) edges carry provenance but are less precise than SCIP. Tree-sitter and LLM are NOT independent sources — tree-sitter provides structure that LLM reasons over. Neither works alone. The dual-source extraction is SCIP vs Original FC (tree-sitter+LLM), NOT triple-source. By treating graph as derived, you can:
- Rebuild graph without re-extracting
- Swap graph backend without losing facts
- Tune confidence thresholds without re-indexing
- Benchmark TerminusDB storage efficiency (not assume "95% dedup")

### Why ripgrep at agent layer, not dedicated code search server

PostgreSQL GIN tsvector is prose-oriented (stems, ignores punctuation). But adding Zoekt-style dedicated code search is over-engineering for an agent-with-workspace architecture. The agent already has a checked-out repo and can call ripgrep directly for exact/regex/pattern search. Zoekt only needed for server-side multi-repo search platform (not our model).

**Three-level toolset contract:**
1. **Semantic + BM25** (server) — "where should I look first?" Broad candidate generation.
2. **Graph tools** (server) — "how are these things related?" Structural reasoning.
3. **ripgrep** (agent workspace) — "show me exact textual evidence." Exact/regex in checkout.

### Why directed paths for flow questions, not undirected Steiner

Leiden runs on symmetrized graph (correct — module discovery doesn't need direction). But "how does X reach Y?" and "what breaks if I change this?" require directed traversal. NetworkX `steiner_tree` is undirected approximation — wrong for call-flow reasoning. Use directed shortest/k-shortest paths for flow questions, Steiner only for "small explanatory subgraph."

### Why vanilla RRF as foundation, not adaptive K as core identity

Adaptive K (sigmoid continuous, 3 signals, per-channel) is experimental complexity. Public hybrid-search guidance (Elastic, Sourcegraph) consistently presents RRF as robust and low-tuning. Start with vanilla RRF, add adaptive K as measured improvement later.

### Why hierarchical fusion, not flat four-list RRF

Code and key-docs are not different fields — they are different evidence types with different length distributions, authority levels, and relevance patterns. Flat RRF over {code_sparse, code_dense, doc_sparse, doc_dense} pretends they're the same kind of thing.

Federated search literature treats heterogeneous collections as a collection-selection + result-merging problem. The right architecture is hierarchical:
1. Intra-collection score fusion normalizes within homogeneous collections (code: sparse+dense, doc: sparse+dense)
2. Cross-collection weighted RRF merges ranked outputs without assuming comparable scores (w_code ∝ π_code(q))
3. Doc→code projection grounds doc evidence into code-space (deterministic mentions → code priors)

This respects heterogeneity at every level. Code gives executable truth, docs give design intent, doc↔code links give structural priors. Each has its own role in the scouting loop.

### Why precision-anchoring, not dual-source merge

GPT review correctly identified that "SCIP wins on overlap" is the wrong merge rule. SCIP should anchor, not overwrite. Tree-sitter builds the unit skeleton (always, for all files), SCIP attaches precision anchors onto those units (when available, 8 languages). This preserves HKUDS Section 3.1's hierarchical unit design — SCIP makes the unit map sharper, not competing.

Key principle: **SCIP does not win; it anchors.** The wrong rule is "if keys overlap, SCIP wins." The right rule is "tree-sitter defines the unit skeleton, SCIP provides the precise anchor, embeddings index the semantic projection of that enriched unit."

**SCIP does not give call graphs.** Tree-sitter discovers call sites. SCIP resolves ambiguous callees when precise symbol identity is available. G_call stays tree-sitter-first. SCIP upgrades `call_candidate` to `call_resolved` when anchor match exists.

**LLM stays in Section 3.2 (query augmentation, navigation control).** NOT in extraction/merge. The paper uses LLM for: query intent classification, keyword expansion, pseudocode hints, exploration control. Optional unit summaries as annotations, not symbol identity.

### Why integrated scouting workflow, not "two separate layers"

The HKUDS paper (Section 3.2) does NOT present retrieval and graph as independent layers. It presents them as complementary stages of one scouting loop:
- Query processing (rewriting, keyword expansion, pseudocode hints)
- Retrieval (hybrid sparse+dense over hierarchical units)
- Graph expansion (from retrieved units, via relation graph)
- Agent tools (directed traversal, impact analysis, module discovery)
- Iteration (agent refines until task done)

Retrieval without graph expansion gives flat results. Graph expansion without retrieval has no seed set. They're designed to work together — retrieval finds candidates, graph expansion gives them structural context. The server exposes separate APIs (query endpoint, MCP graph tools) for implementation modularity, but the architecture is one integrated scouting workflow matching the paper's design.

Post-fusion 2-hop graph expansion is already implemented — retrieval results automatically seed graph expansion. This is the paper's design: retrieval → expansion → deeper exploration on demand.

## The 7 Core Algorithms

### 1. Precision-Anchored Semantic-Structural Representation

**HKUDS Section 3.1, hardened with SCIP anchoring.** Not "dual-source merge." One integrated pipeline:

```
Step 1 — Hierarchical Code Units (tree-sitter/AST):
  Build File/Class/Function/Doc units from tree-sitter
  Unit skeleton: path, span, signature, docstring
  Structural relations: contain, import, inherit, call-site
  This is the token-efficient scouting surface from the paper
  Confidence: "resolved" (contain) / "heuristic" (import/inherit/call)
  source_priority: 50

Step 2 — SCIP Precision Anchoring (when available):
  For each tree-sitter unit, attempt SCIP symbol alignment
  Alignment scoring: span overlap + name match + kind compat + container compat
  High score → attach as canonical anchor (precise identity)
  Medium score → keep as candidate alias
  Low score → keep unanchored AST unit
  Anchored units get: precise symbol ID, definition locations, cross-file references
  Confidence: "precise", source_priority: 100
  SCIP does NOT give call graphs — tree-sitter discovers calls, SCIP resolves callees

Step 3 — Derived Attachments (embeddings, optional LLM annotations):
  Dense vectors over enriched unit text → attachment_kind="embedding"
  Optional LLM summaries/labels → attachment_kind="summary" | "semantic_note"
  NOT graph facts — retrieval features and annotations
  Confidence: "derived", no source_priority

Step 4 — Multi-grained Hybrid Indexing:
  Sparse (BM25) + dense (embedding) over enriched units
  Query augmentation (Section 3.2) steers weights by intent
```

**Canonical enriched unit carries:**
```
unit_id, granularity (file/class/function/doc), path, span, signature, docstring
anchor_set = {SCIP symbols that aligned to this unit}
support = {ast, scip}  (which sources contributed)
reliability = computed from support set
dense vector (embedding attachment)
sparse text fields (BM25 index)
```

**Edge reliability policy:**
| Edge type | Source | Confidence |
|-----------|--------|------------|
| G_dep (import) | tree-sitter import + optional SCIP package match | medium→high |
| G_inh (inheritance) | tree-sitter base-class + optional SCIP type_definition | medium→high |
| G_call (call) | tree-sitter call-site + SCIP callee resolution | candidate→resolved |
| contain | tree-sitter structure | resolved |
| ref | SCIP definitions/references | precise |

### 2. Derived Branch-Aware Graph (TerminusDB)

**Derived view, NOT sole truth.** Canonical IR facts (Layer 1) are deepest truth. Graph is materialized from facts for fast traversal (Layer 2). Refreshable.

- Nodes: documents, symbols (with SCIP identity + provenance metadata)
- Edges: call, import, inherit, ref, contain (5 types, confidence-weighted)
- Provenance on edges: SCIP ("precise"), tree-sitter ("resolved"/"heuristic"), LLM annotations ("derived")
- Git-like branching: branch graph shares unchanged nodes with parent

### 3. Projection Layer (Graph → L0/L1/L2 Summaries)

**SEPARATE from query layer.** Projection generates structured summaries, query returns ranked hits.

**Directed vs undirected policy (critical correction):**
- Leiden runs on SYMMETRIZED graph (undirected) — module discovery doesn't need direction
- Arborescence runs on DIRECTED graph — navigation tree needs direction
- "How does X reach Y?" → directed shortest paths, NOT undirected Steiner
- Steiner tree ONLY for "small explanatory subgraph" — undirected approximation, limited use

**Real pipeline (4 steps):**

Step 1 — Scope + Weight: Build undirected (symmetrized) + directed graphs from TerminusDB. Confidence bands multiply weights. Hub compression.

Step 2 — Leiden clustering (CORE, on SYMMETRIZED graph): Groups symbols into code modules from call density. Single resolution 1.0 sufficient for most repos.

Step 3 — Representative selection (UTILITY): PageRank + degree centrality pick representative per cluster. Betweenness centrality over-engineered.

Step 4 — Arborescence backbone (CORE, on DIRECTED graph): Directed navigation tree for L1 hierarchy + cross-cluster xrefs.

Output: L0 (summary), L1 (navigation + backbone), L2 (per-cluster evidence). LLM optional for labels, deterministic fallback via representatives. L2 chunks may include optional `supporting_docs` — doc chunks whose traceability mentions point at cluster symbols.

### 4. Hierarchical Retrieval + Doc-Code Projection

**Origin:** HKUDS multi-grained hybrid indexing, extended with federated retrieval for heterogeneous evidence collections.

**Core principle:** Code and key-docs are heterogeneous collections — different roles, different authority, different score distributions. NOT flat four-list RRF.

**Pipeline (6 stages):**

```
Stage 1 — Query Processing + Collection Prior:
  Existing rewriting, keyword expansion, pseudocode hints.
  Collection prior: π_code(q) ≈ alpha, π_doc(q) ≈ 1-alpha.
  Signals: identifier specificity, design/ADR keywords, pseudocode richness,
           heading/title affinity, exact-match demand.

Stage 2 — Parallel Retrieval:
  Code collection: semantic (HNSW) + keyword (BM25)
  Doc collection:  semantic + keyword
  Parallel execution.

Stage 3 — Intra-Collection Fusion:
  Within each collection, normalized weighted score fusion:
    S_code(u|q) = α_code · norm(code_dense) + (1-α_code) · norm(code_sparse)
    S_doc(d|q)  = α_doc · norm(doc_dense) + (1-α_doc) · norm(doc_sparse)
  Normalized to [0,1]. Score fusion preserves magnitude for downstream projection.

Stage 4 — Cross-Collection Weighted RRF:
  R(x|q) = w_code/(K+rank_code(x)) + w_doc/(K+rank_doc(x))
  w_code ∝ π_code(q), w_doc ∝ π_doc(q)
  RRF on ranks — correct for heterogeneous score scales.

Stage 5 — Doc→Code Projection (grounded priors):
  Doc prior via noisy-or (bounded [0,1]):
    D(v|q) = 1 - Π(1 - Ŝ_doc(d|q) · P̂(v|d))
  Ŝ_doc normalized [0,1], P̂ normalized [0,1].
  Diminishing returns per additional doc.
  Seed via bounded mixture:
    seed(v|q) = (1-β(q)) · Ŝ_code(v|q) + β(q) · D(v|q)
    β(q) = β_max · π_doc(q)
  Convex combination, guaranteed [0,1]. No scale mismatch.

Stage 6 — Output: direct code hits + doc-projected priors + supporting docs.
  Each result carries provenance channel and contributing doc IDs.
```

**Why hierarchical:** Flat RRF pretends code and docs are the same kind of thing. They're not. Code is executable truth; docs are design intent. Intra-collection fusion normalizes within homogeneous collections first. Cross-collection weighted RRF merges ranked outputs without assuming comparable scores. Doc→code projection grounds doc evidence into code-space for graph seeding.

### 5. Scouting Workflow (Integrated Retrieval + Graph Navigation)

**Origin:** HKUDS structure-aware navigation (Section 3.2). Retrieval, doc→code projection, and graph expansion are complementary stages of one scouting loop.

```
Scouting loop:

  Step 1 — Query Processing (Algorithm 4, Stage 1):
    Rewriting, keyword expansion, pseudocode hints, collection prior.

  Step 2 — Retrieval + Fusion (Algorithm 4, Stages 2-6):
    Parallel retrieval → intra-collection RRF → cross-collection weighted RRF
    → doc→code projection → three result sets.

  Step 3 — Seed Merge:
    seed(v|q) = (1-β(q)) · Ŝ_code(v|q) + β(q) · D(v|q)
    β(q) = β_max · π_doc(q), β_max default 0.3.
    Bounded mixture, no scale mismatch.

  Step 4 — Graph Expansion (code-only, 2-hop):
    Expand seeds via G_dep, G_inh, G_call.
    Docs NEVER enter as graph nodes.
    Provenance: direct | doc-projected | expanded.

  Step 5 — Agent Tools (deeper exploration on demand):
    directed_path, impact_analysis, leiden_clusters, steiner_path.

  Step 6 — Targeted Verification:
    ripgrep for exact textual evidence (agent workspace).

  Step 7 — Iterate.
```

**Final evidence slate:**

| Channel | Role |
|---------|------|
| code_direct | "Here's the code you asked for" |
| code_doc_projected | "Docs mention this code" |
| code_graph_expanded | "This code is structurally related" |
| doc_support | "Design context and rationale" |

Docs influence the graph ONLY through grounded code priors (Step 3). No raw doc enters graph expansion. The code graph stays clean (code-only topology) while doc evidence enriches structural discovery.

Agent interfaces:
- `POST /query` → mixed evidence slate + post-fusion graph expansion
- `MCP tool: directed_path(from, to)` → directed shortest path through call graph
- `MCP tool: impact_analysis(symbol_id)` → callers + dependents (directed traversal)
- `MCP tool: leiden_clusters(snapshot_id)` → module boundaries (from symmetrized graph)
- `MCP tool: steiner_path(...)` → small undirected explanatory subgraph (limited use)
- `GET /projection/{snapshot_id}` → L0/L1 for session prefix injection

### 6. Document Digestion (LadybugDB)

Core docs (architecture, key tech design, ADRs) are branch-stable. Stored in LadybugDB (embedded graph DB, successor to Kuzu, can ATTACH PostgreSQL). Doc graph with MENTIONS edges to code SCIP symbols via SCIP Resolution Bridge. **Off critical path** — store doc chunks+embeddings in PostgreSQL as durable baseline, LadybugDB as accelerator once bridge quality validated.

### 7. SCIP Resolution Bridge (Doc → Code Mapping)

3-strategy cascade: lexical exact → namespace contextual → vector semantic.
Maps doc entity mentions to SCIP symbol IDs in the graph.

## Basic Assumptions to Validate

Before committing to this architecture, these assumptions need evidence:

### A1: Code relationships as graph edges are more valuable than flat search results

**Assumption:** An AI agent gets better outcomes when it can query "how does X connect to Y?" via graph traversal vs grepping for X and Y separately.

**How to validate:** A/B test on SWE-bench or similar: agent with graph tools vs agent with grep-only. Measure task completion rate and context efficiency.

**Risk if wrong:** We built a graph database for a problem that keyword search solves.

### A2: Derived graph views are worth the TerminusDB operational cost

**Assumption:** Building derived graph views from canonical facts and persisting them in TerminusDB with branch semantics is worth the operational overhead of running TerminusDB alongside PostgreSQL. Facts are truth, graph is derived view — but graph views still need a store.

**How to validate:** Benchmark rebuild time for representative repos (100, 1K, 10K files). If materializing graph from facts is <2 seconds, TerminusDB may not be needed — could rebuild derived view per query. Benchmark TerminusDB content-addressable dedup claims (are "95% shared nodes/edges" real?).

**Risk if wrong:** Two databases (PG + TerminusDB) for no benefit. Or TerminusDB storage claims don't hold.

### A3: Multi-branch graph awareness is needed

**Assumption:** AI agents working on feature branches need graph-scoped-to-branch. Re-indexing current branch state is insufficient.

**How to validate:** Survey agent workflows. Do agents actually switch branches mid-session? Or do they work on one branch and re-index when switching?

**Risk if wrong:** TerminusDB branching feature is unused complexity.

### A4: SCIP precision anchoring measurably improves unit map quality

**Assumption:** Attaching SCIP symbol anchors onto tree-sitter units (precision anchoring) measurably improves downstream quality vs tree-sitter-only units. The improvement justifies the complexity of running 8 SCIP indexers.

**How to validate:** Compare retrieval quality and graph accuracy with and without SCIP anchoring. Does anchor_set improve symbol resolution, cross-file reference accuracy, and call-graph precision?

**Risk if wrong:** SCIP indexers add operational complexity (external binaries per language) without measurable quality gain over tree-sitter-only units.

### A5: Session prefix projection provides architectural awareness

**Assumption:** Loading L0/L1 projection as system prompt prefix at session start gives the agent enough architectural context to navigate the codebase effectively, reducing exploration queries. AI coding harnesses (Claude Code, Cursor, Windsurf) always prompt agents to "explore first" — prefix eliminates this cold-start exploration by giving architectural context before any tool call.

**Prompt cache makes prefix nearly free:** The projection sits in the system prompt, which is cached after the first API call. Cost is paid once per session, amortized to zero on every subsequent turn. Tradeoff: small upfront token cost vs. always catching the cache. This means compactness still matters for initial load, but per-turn cost is zero.

**How to validate:** Measure number of exploration queries per session with and without prefix. Measure task completion time. Compare first-turn behavior — does the agent skip the "explore codebase" step?

**Risk if wrong:** Prefix wastes initial context window load without reducing exploration. But since it's cached, the ongoing cost is negligible — the risk is low even if the assumption is partially wrong.

### A6: Graph tools (directed_path, impact_analysis) are actually used by agents

**Assumption:** AI agents will call MCP graph tools when they need relationship-level understanding, not just fall back to file search every time. Directed paths for flow questions, Leiden for module discovery, impact analysis for change propagation.

**How to validate:** Instrument tool call frequency. If graph tools are never called, the graph layer is unused.

**Risk if wrong:** Graph layer built but agents never query it.

---

## Detail Algorithm Research

### D1: Projection Consumption Strategy — DECIDED

**Strategy (not research — decided):**

```
Main agent (Opus/Sonnet):
  System prompt prefix ← L0 + L1 projection (cached)
  - Architectural overview, cluster structure, backbone tree, cross-cluster xrefs
  - Prompt cache: paid once per session, zero per-turn
  - Agent starts session knowing codebase architecture — skips cold-start exploration

Explore subagents (Haiku — cheap model):
  Read L2 JSON on-demand + use tools
  - L2: per-cluster evidence (symbols, files, relationships)
  - Tools: graph queries (steiner_path, impact_analysis, find_callers)
  - Tools: rg/grep for targeted code search
  - Cheap model reads detailed L2 data, returns summarized findings to main agent

For huge monorepos:
  3 layers (L0/L1/L2) may not capture all structure
  Solution: always pick the three projection levels that work for the repo's scale
  Not a research problem — practical tuning per repo
```

**Why this works:**
- L0/L1 is compact enough for system prompt cache. Agent gets architecture for free after first turn.
- L2 is too detailed for prefix — but cheap subagents can read it without wasting main agent's context.
- Subagents with tools (graph + rg) combine structural understanding with targeted search.
- No format research needed — JSON for L2 (subagents read it), compact JSON for L0/L1 prefix (cache handles cost).

### D1.1: Projection Format — RESEARCH NEEDED

**Problem:** L0/L1 prefix format still matters for initial cache priming. JSON baseline ~3000 tokens for 50 clusters. 64% of that is structural tokens (quotes, colons, braces, commas). Need format that is compact AND reliably LLM-parseable.

**Token comparison (50 clusters, BPE tokenization):**

| Format | Tokens | vs JSON | Notes |
|--------|--------|---------|-------|
| JSON (compact) | 3015 | 1.00x | Baseline. Reliable but verbose. |
| XML (compact attributes) | 2385 | 0.79x | Claude trained on XML, but more tokens than JSON. |
| YAML (compact flow) | 2078 | 0.69x | Less structure, but indentation-sensitive. |
| Markdown table | 1942 | 0.64x | Good for tabular data, poor for nested. |
| DSL (pipe-delimited) | 1692 | 0.56x | 44% savings. Unambiguous with header. |
| TSV (tab-separated) | 1592 | 0.53x | 47% savings. Simplest. |
| Positional (space-separated) | 1533 | 0.51x | 49% savings. Ambiguous if values contain spaces. |

**LLM parsing accuracy benchmarks (TOON project, 209 questions, 4 models):**

| Format | Accuracy | Tokens | Efficiency (acc%/1K tok) |
|--------|----------|--------|--------------------------|
| TOON | 76.4% | 2,759 | 27.7 |
| JSON compact | 73.7% | 3,104 | 23.7 |
| YAML | 74.5% | 3,749 | 19.9 |
| JSON pretty | 75.0% | 4,587 | 16.4 |
| XML | 72.1% | 5,221 | 13.8 |

All formats within 4.3% accuracy. No format is dramatically more or less reliable for LLMs.

**Two candidate formats for L0/L1 prefix:**

**Candidate A: Hash-sectioned DSL** (~43% token reduction)
```
#projection L0 proj_abc123
#repo FastCode main b476a04
#summary 347 nodes, 891 edges. Key: authenticate(47), token_validation(23)
#clusters id|label|nodes
c0|authenticate|47
c1|token_validation|23
c2|route_handler|19
#xref src|dst|weight
c0|c1|8.4
c0|c2|6.2
#backbone
c0->c1->c3
c0->c2->c4
#meta algo=leiden backbone=arborescence root=c0
```
Pros: compact, extensible, human-readable. Cons: custom parser needed.

**Candidate B: TOON (Token-Oriented Object Notation)** (~5% reduction for nested, ~35% for tabular)
```
clusters[50]{id,label,members}:
  c0,authenticate,47
  c1,token_validation,23
```
Pros: standardized (spec v3.0), multi-language tooling, lossless JSON round-trip. Cons: smaller savings for non-tabular data, newer format with less LLM training exposure.

**Research needed:**
- Hash-sectioned DSL vs TOON: test actual L0/L1 projection from FastCode on both formats, measure real token counts
- Test Claude parsing accuracy on hash-sectioned DSL with 2-line format description in prompt
- Does the 2-line DSL schema description eat the token savings? (e.g., "# Format: section header, then pipe-delimited rows")
- For L2 chunks (uniform tabular), TOON is clearly better — worth using two formats (DSL for prefix, TOON for L2)?
- Python serialization: toonify library (334 stars) for TOON, custom for DSL. Maintenance burden?

**Sources:** TOON format benchmarks (github.com/toon-format/toon), toonify Python port (github.com/ScrapeGraphAI/toonify)

### D2: SCIP-Tree-sitter Alignment — Edge Case Research

**Alignment scoring replaces hard "SCIP wins" merge.** Edge cases for alignment function (span overlap + name match + kind compat + container compat):

1. **Python decorators producing same-name functions at same line:**
   ```python
   @retry(max_attempts=3)
   def process(): pass  # Two symbols: process and process_with_retry at same line?
   ```
   Does SCIP handle this? Does alignment score disambiguate?

2. **Java method overloading:**
   ```java
   void authenticate(Token t) {}
   void authenticate(String key) {}
   ```
   Same name, same kind, different signatures. Alignment needs signature compat signal. Does SCIP provide distinct symbols with signature info?

3. **Partial SCIP coverage:** Python files have SCIP anchors, TypeScript files don't. TypeScript units stay unanchored AST units. How does Leiden clustering handle mixed-anchored graphs?

4. **Cross-language edges:** Python calls Rust via FFI. Tree-sitter discovers call site in Python, SCIP covers both languages. Can SCIP anchor resolve the cross-language callee?

5. **Alignment threshold tuning:** What scores should be "high → canonical anchor" vs "medium → candidate alias" vs "low → unanchored"? Needs benchmark on real repos.

### D3: TerminusDB Branch Graph — Schema Design

**Research needed:**
- What TerminusDB schema best represents code graphs? Document/Symbol as classes with typed edge properties?
- How does TerminusDB's content-addressable storage handle branch divergence? If 5% of files change, does the branch store 5% new data or 100%?
- Incremental update protocol: when file X changes on branch B, do we:
  a) Delete all edges where src or dst has path=X, then re-insert from new extraction?
  b) Diff old edges vs new edges, compute add/remove delta?
  c) How to handle symbol renames (old ID no longer exists, new ID created)?

### D4: Steiner Tree — Terminal Selection from Natural Language

Current: keyword matching against node titles/paths, top 12 by token overlap + log(degree).

**Research needed:**
- Should terminal selection use embedding similarity instead of keyword matching? Cost: extra embedding computation per query. Benefit: semantic matching.
- How many terminals are optimal? For a "how does X connect to Y" query, 2 terminals suffice. For "explain the auth system", maybe 5-8. Should terminal count adapt to query scope?
- For multi-hop queries, should we first find one terminal via keyword/embedding, then expand via graph (BFS/DFS) to find others?

### D5: Leiden Resolution — Single vs Hierarchical

**Research needed:**
- What single resolution value produces the best module boundaries across different repo sizes? Is 1.0 a good default?
- At what graph size does hierarchical (multi-resolution) become worth the extra computation? 1000 nodes? 5000? 10000?
- Should resolution adapt to graph density (edges/nodes ratio)? Dense graphs may need higher resolution.

### D6: Adaptive Fusion — 5-Source vs 2-Channel

**Research needed:**
- Does per-source K (5 values: code_semantic, code_bm25, doc_vector, doc_fts, graph) produce measurably better results than per-channel K (2 values: code_combined, doc_combined)?
- Is semantic+keyword within the same domain correlated enough that combining them loses signal?
- What's the right way to validate without labeled relevance judgments? Could use LLM-as-judge on retrieval quality.

### D7: SCIP Resolution Bridge — Doc → Code Matching

**Research needed:**
- How to evaluate matching quality without labeled data? Manual annotation on 100 doc-code pairs?
- Ambiguous name resolution: "User" class exists in 5 modules. Does namespace proximity (doc in same directory as code) resolve this reliably?
- Should matching be done at index-time (store MENTIONS edges in LadybugDB) or at query-time (live matching)?

### D8: REMOVED — Merged into D1 (projection strategy is decided, not researched)
