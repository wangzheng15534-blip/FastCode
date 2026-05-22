---
title: "Unit-Grounded IR Model and Doc-Integrated Scouting Loop"
date: "2026-04-24"
category: "docs/solutions/best-practices"
module: "FastCode Canonical IR and Retrieval"
problem_type: best_practice
component: service_object
severity: high
applies_when:
  - "Designing or modifying the canonical IR model (semantic_ir.py, adapters, ir_merge)"
  - "Changing how SCIP, AST, and embeddings integrate into the extraction pipeline"
  - "Modifying the retrieval/scouting loop or doc-channel integration"
  - "Adding new evidence channels or graph expansion logic"
tags:
  - fastcode
  - canonical-ir
  - codeunit
  - scip-alignment
  - doc-projection
  - scouting-loop
  - retrieval-architecture
  - hkuds
---

# Unit-Grounded IR Model and Doc-Integrated Scouting Loop

## Context

FastCode's original IR model used a symbol-first, rule-heavy merge: `IRDocument`/`IRSymbol`/`IROccurrence`/`IREdge` as canonical facts, with confidence enums (`precise`/`resolved`/`heuristic`) baked into the data model. SCIP extraction ran as a separate pipeline beside the original tree-sitter extraction, and the merge applied hard overwrite rules ("SCIP wins on overlap"). The retrieval loop treated design docs as just another code element in a flat mixed list — docs died as a ranked list and never projected grounded priors into code-space before graph expansion.

This didn't match the HKUDS paper's architecture. In the paper (arxiv:2603.01012v2), Section 3.1 is one integrated semantic-structural representation built from hierarchical code units, not from a symbol/document split. Section 3.2 adds LLM query augmentation on top of that scouting map. LLM belongs in query augmentation, not in extraction. And docs should be a separate evidence collection that projects priors into code-space, not flat-fused into the same ranked list.

The user directive was explicit: "drop all legacy code, DIRECT rewrite not just patch, fully refactor, academic and algorithm clean and elegant is most important now." (Codex session `019db273`, Apr 21-22)

## Guidance

### 1. Unit-Grounded Canonical IR (replacing symbol/document/edge)

**Model: four primary objects**

- `IRCodeUnit` — hierarchical unit (`file`, `class`, `function`, `doc`) with stable `unit_id`, `parent_id`, `path`, `kind`, `span`, `language`, `display_name`, `qualified_name`, `source_set`
- `IRUnitSupport` — extractor evidence attached to a unit. For `fc_structure`: parser span/name/signature/context. For `scip`: external symbol ID, occurrences, definitions/references, signature, namespace
- `IRRelation` — canonical relation between units (`contain`, `call`, `import`, `inherit`, `ref`) with `support_sources`, `support_ids`, `resolution_method`
- `IRUnitEmbedding` — semantic sidecar for each canonical unit (model-versioned, not inline in the unit object)

**Key principle: SCIP anchors, it does not win.**

SCIP is not a separate pipeline. Tree-sitter builds the unit skeleton (always), SCIP attaches precision anchors (when available). One pipeline, not two rival pipelines. The merge uses **scoped alignment scoring**, not hard overwrite rules:

```
Alignment stages: file → container → callable → residuals
Score = span_overlap + name_match + kind_match + container_match
Strong match (≥ threshold) → SCIP becomes primary anchor
Medium match → candidate anchor
No match → SCIP symbol stays as synthetic unit
```

**What was removed:**
- Confidence enum (`precise`/`resolved`/`heuristic`) from canonical facts — use deterministic provenance (`source_set`) plus resolution metadata instead
- Separate SCIP pipeline — SCIP is now an anchor source within the single extraction pipeline
- Inline embeddings in canonical objects — embeddings are model-versioned sidecars keyed by `unit_id`
- Rule-based merge ("SCIP wins on overlap") — replaced by scoped alignment algorithm

**Files rewritten:** `semantic_ir.py`, `ast_to_ir.py`, `scip_to_ir.py`, `ir_merge.py`, `ir_validators.py`, `ir_graph_builder.py`, `snapshot_symbol_index.py`, `main.py`

### 2. Doc-Integrated Scouting with Traceability Projection

**Key principle: docs are a separate evidence collection that projects into code-space, never enters the graph.**

The scouting loop was refactored into explicit stages:

```
1. Query processing → intent, keywords, pseudocode, collection prior π_code/π_doc
2. Retrieval (parallel):
   - Code channel: semantic (pgvector HNSW) + keyword (BM25 GIN) → intra-collection fusion
   - Doc channel: semantic + keyword → intra-collection fusion
3. Cross-collection weighted RRF → mixed evidence slate (rank-space)
4. Doc→code projection (score-space, before graph expansion):
   - D(v|q) = 1 - Π(1 - Ŝ_doc(d|q)·P̂(v|d))   [noisy-or, bounded [0,1]]
   - seed(v|q) = (1-β(q))·Ŝ_code(v|q) + β(q)·D(v|q)
   - β(q) = β_max · π_doc(q)
5. Graph expansion — code-only seeds, docs never enter as graph nodes
6. Return — mixed evidence slate with provenance channels
```

**Why projection must happen before graph expansion, in score-space:**

The user identified a critical order issue: if you do cross-collection RRF first (rank-space), you lose score magnitudes that the projection formula needs. The fix: projection uses raw within-collection fused scores (Ŝ_doc), not post-RRF ranks. RRF is only used for the final mixed evidence slate.

**The formula is bounded:**
- `Ŝ_doc(d|q)` normalized to [0,1] from within-doc-collection fused scores
- `P̂(v|d)` grounded traceability weight from doc mentions (deterministic, no LLM)
- `D(v|q)` via noisy-or, naturally bounded [0,1]
- `β(q) = β_max · π_doc(q)` — query controls doc influence, β_max default 0.3

**Files changed:** `retriever.py`, `doc_ingester.py`, `main.py`

## Why This Matters

**Unit-grounded IR** ensures that extraction produces a single coherent model of the codebase, not two competing pipelines that overwrite each other. The alignment algorithm handles ambiguity gracefully (multiple match strengths) instead of relying on hard rules that break on edge cases (overloads, nested classes, synthetic symbols).

**Doc-integrated scouting** ensures that architectural documents and design decisions actually influence code retrieval. Without projection, docs are just noise in the ranked list. With projection, a doc mentioning `AuthMiddleware` gives a bounded prior boost to the actual `AuthMiddleware` code units — which then seed graph expansion. The noisy-or formulation correctly handles multiple docs mentioning the same symbol (diminishing returns, not linear inflation).

## When to Apply

- When adding new extraction sources (e.g., LSP, language-server) — they should attach as anchor sources to existing units, not create a separate pipeline
- When modifying the merge algorithm — always use scoped alignment, not overwrite rules
- When adding new evidence channels to the scouting loop — they must project into code-space before graph expansion, never enter the graph directly
- When modifying the seed formula — keep it bounded and use score-space inputs, not rank-space

## Examples

**Before (old model):**
```python
# Two separate pipelines
ast_snapshot = build_ir_from_ast(repo)   # IRDocument + IRSymbol + IREdge
scip_snapshot = build_ir_from_scip(repo) # separate set of IRSymbols

# Hard overwrite merge
if scip_symbol matches ast_symbol:
    ast_symbol.confidence = "precise"  # overwrite
    ast_symbol.external_id = scip_symbol.id  # overwrite
```

**After (unit-grounded model):**
```python
# One pipeline: tree-sitter skeleton, SCIP anchors onto it
ast_units = build_ir_from_ast(repo)      # IRCodeUnit tree (always)
scip_data = build_ir_from_scip(repo)     # precision anchors (when available)

# Scoped alignment: file → container → callable → residuals
merged = align_and_merge(ast_units, scip_data)
# Strong match → primary anchor on existing unit
# Medium match → candidate anchor
# No match → synthetic unit with scip source_set
```

**Before (flat doc fusion):**
```python
# Docs mixed into same retrieval list as code
all_elements = code_elements + doc_elements
results = hybrid_retrieve(query, all_elements)  # flat BM25 + vector
# Docs compete with code in the same ranked list, no projection
```

**After (doc projection):**
```python
# Separate channels
code_hits = code_channel.retrieve(query)
doc_hits = doc_channel.retrieve(query)

# Score-space projection: docs → code priors
doc_prior = noisy_or_projection(doc_hits, trace_links)  # D(v|q) bounded [0,1]
code_seeds = bounded_mixture(code_hits, doc_prior, beta=query.doc_mass)

# Graph expansion on code seeds only
expanded = graph_expansion(code_seeds)  # docs never enter graph
```

## Related

- **Codex session `019db273`** (Apr 21-22) — Produced both refactorings. User directives: "DIRECT rewrite not just patch" and "clean architecture, better algorithm matters."
- Commit `f22a52c` — Unit-grounded IR model with precision-anchored SCIP alignment
- Commits `01f4ee5`, `3e2cc5b` — Test updates for the new model
- Commits `57a1706`, `127f042`, `c1c90ca` — Trivial cleanup bugs exposed by the refactoring
- `docs/audit-branch-index-pipeline.md` — Pre-refactoring audit, Sections 2.2 and 6.1 now stale
- HKUDS paper arxiv:2603.01012v2 — Sections 3.1 (semantic-structural representation) and 3.2 (scouting loop)
