## Goal

Continue building FastCode's semantic resolver layer — an adaptive knowledge graph with git features for AI agents, based on the HKUDS scouting-first framework. The resolver system provides language-specific semantic upgrades to a canonical IR that merges Tree-sitter structural extraction with SCIP precise symbol identity.

The original Codex plan (`codex-session-019ddf34-plan.md`) specified a full rewrite with 12 language adapters, Clang-backed C/C++, typed schemas, and query-time escalation budgets. Two Codex sessions (Apr 28-30 and May 1) implemented the resolver framework, broadened language coverage, and fixed boundary issues. The working tree now has **19 uncommitted files** with additional hardening and a shared HTTP schema extraction that still needs cleanup and testing.

## Instructions

- **Read `CLAUDE.md` first** — it defines the architecture (dual-source extraction: SCIP + Tree-sitter/LLM, three-layer storage: PostgreSQL + TerminusDB + retrieval indices), design principles (canonical facts are truth, graph is derived view), and the index pipeline flow.
- **Read `IMPLEMENTATION_TODOS.md`** — this is the authoritative task list written by Codex at the end of the last session, organized into four categories: semantic resolver, SCIP/indexing, template-rule refactor, and verification.
- **Resolver upgrade layer sits AFTER `merge_ir()`** — never before it. The useful identity state is created by the merge. Resolvers emit `ResolutionPatch` upgrades through one bus.
- **No `model_dump()` at the API boundary** — use explicit field-by-field schema translation. The current uncommitted changes moved shared HTTP request/response models into `fastcode.schemas.api` but this is untested.
- **New languages without compiler backends use graph-backed structural evidence + diagnostics** — they emit `ToolDiagnostic` when tools are missing, never silently claiming precision.
- **Follow the template layered architecture**: `api → infrastructure → core → schemas` dependency direction. `core/` must be pure algorithms with no Pydantic, DB, filesystem, subprocess, or LSP imports.
- **Architecture tests in `tests/core/test_boundary.py`** enforce layer dependencies. Expand to cover `main`, `api`, `infrastructure`, and store modules.

## Discoveries

### What Was Built (Codex sessions Apr 28–30, May 1)

- **Resolver package** (`semantic_resolvers/`): `base.py` (ResolverSpec, ToolDiagnostic, typed contract), `registry.py` (12 languages), `language_graph.py` (generic graph-backed fallback for JS/TS/Java/Go/Rust/C#/Zig/Fortran/Julia), `c_family.py` (C/C++ structural: includes + inheritance), `python.py` (refactored from legacy graph), `graph_backed.py` (reusable base), `patching.py` (patch application + precedence).
- **Pipeline wiring** in `main.py`: resolver pass after incremental merge, SCIP multi-language, old `call_graph_bridge` removed.
- **Template refined**: `/home/dev/repo_template/python_v1.0_single/AGENTS.md` — fixed `uv` commands, `schemas` naming, dependency rule, Schemathesis command.
- **API boundary**: Removed two `model_dump()` leaks from `api.py` and `web_app.py`.
- **Architecture test**: Updated `test_boundary.py` from obsolete `effects/db.py` to current `infrastructure/db.py`.
- **C/C++ fixes**: Header classification (`.h`/`.hpp`), relation source ranking, merge key normalization (`base` vs `base_name`).
- **SCIP**: Widened from Python-only to multi-language detection. Fixed multi-SCIP merge bug.
- **Extension/language detection**: Added Zig, Fortran, Julia extensions.

### Key Unresolved Design Decisions

- **Clang AST dump works locally** (`clang -Xclang -ast-dump=json`) but `scip-clang` and `scip-python` are NOT installed. C/C++ resolver is still structural only, not compiler-backed.
- **`pending_capabilities` and `resolution_rank`** were added to IR but not yet consumed by the resolver pipeline for capability-gated execution.
- **Query-time semantic escalation** (index-time vs query-time budget) was proposed but not implemented — currently all upgrades are index-time only.
- **Structural fallback vs compiler-confirmed relations** are not distinguished in `resolution_state` — both go through the same patch path.

### Current Working Tree State

**19 uncommitted files** with changes since last commit (`ee60506`):
- `fastcode/src/fastcode/api.py` — shared HTTP schema extraction (150 deletions)
- `fastcode/src/fastcode/schemas/__init__.py` — new api schema exports (58 additions)
- `fastcode/src/fastcode/schemas/api.py` — **new untracked file** — shared request/response models
- `fastcode/src/fastcode/main.py` — pipeline wiring updates (77 additions)
- `fastcode/src/fastcode/scip_indexers.py` — multi-language SCIP
- `fastcode/src/fastcode/semantic_resolvers/` — 8 files with hardening
- `fastcode/src/fastcode/utils/_compat.py` — compat helpers
- `fastcode/src/fastcode/utils/paths.py` — path utilities
- `fastcode/src/fastcode/web_app.py` — shared schema rewire (84 deletions)
- Tests: `test_api.py` (133 changes), `test_semantic_resolvers.py` (86 changes), `test_scip_indexers.py` (30 changes), `test_boundary.py` (8 changes), `test_e2e_indexing.py` (17), `test_e2e_semantic_pipeline.py` (17)

## Accomplished

### Completed (committed)
- [x] Resolver framework with typed contract (`ResolverSpec`, `ToolDiagnostic`)
- [x] Default registry for 12 languages (Python, JS, TS, Java, Go, Rust, C#, C, C++, Zig, Fortran, Julia)
- [x] Graph-backed fallback resolvers for all non-Python/C-family languages
- [x] C/C++ structural resolver (includes + inheritance)
- [x] Template `AGENTS.md` refinement
- [x] API boundary leaks removed
- [x] Architecture test updated
- [x] SCIP multi-language orchestration
- [x] Multi-SCIP merge bug fix
- [x] Extension/language detection expansion
- [x] `pending_capabilities` and `resolution_rank` added to IR
- [x] Linear scan elimination for doc_id lookup
- [x] Frozen=True removed from ResolutionPatch (allowing mutation)
- [x] Regression tests for resolver audit findings
- [x] Source preference hierarchy for C/C++ resolvers
- [x] Removed hardcoded language filter in target_paths fallback

### In Progress (uncommitted working tree)
- [ ] Shared HTTP schema extraction into `fastcode.schemas.api` — code written but **not tested** (`test_api.py` rewritten but not run)
- [ ] `api.py` and `web_app.py` rewire to use shared schema — code written but **not tested**
- [x] 19 files changed, 576 insertions, 309 deletions — needs commit + test verification

### Not Started (from `IMPLEMENTATION_TODOS.md`)

#### Semantic Resolver
- [ ] Replace graph-backed fallback resolvers with true frontend-backed adapters:
  - JavaScript/TypeScript via Compiler API helper
  - Java via JDT or language-server-backed binding resolution
  - Go via `go/packages` or gopls-backed resolver
  - Rust via rust-analyzer semantic queries
  - C# via Roslyn-backed helper
  - Zig via ZLS or zig AST/semantic tooling
  - Fortran via fortls or fparser-backed semantic resolver
  - Julia via LanguageServer.jl-backed resolver
- [ ] Resolver-specific capability gating (don't run every language resolver indiscriminately)
- [ ] Query-time semantic escalation budget (index-time-only → index + query time)
- [ ] Distinguish structural fallback from compiler-confirmed relations in `resolution_state`

#### SCIP/Indexing
- [ ] Replace placeholder Zig/Fortran/Julia SCIP command wiring with verified contracts
- [ ] Preserve all generated SCIP artifacts for multi-language indexing (not just first path)
- [ ] Tests for multi-language SCIP merge path in `FastCode.index_repository`
- [ ] E2E configs/fixtures for new supported extensions

#### Template-Rule Refactor
- [ ] Move mutable dict-heavy core functions toward typed dataclass inputs/outputs
- [ ] Convert store/runtime methods returning raw `dict[str, Any]` to frozen dataclass records
- [ ] Split orchestration from logic in `main.py` (currently mixed: loading, indexing, graphing, SCIP, persistence)
- [ ] Tighten architecture tests for `main`, `api`, `infrastructure`, `store` modules

#### Verification
- [ ] Focused `pyright` on semantic resolver and indexing modules
- [ ] Keep regression suite green: `test_api`, `test_semantic_resolvers`, `test_scip_indexers`, `core/test_boundary`
- [ ] Regression tests for header-language classification (`.h` as C vs C++)
- [ ] API tests for explicit source mapping on `index-multiple`
- [ ] Integration tests for Zig/Fortran/Julia entering indexing pipeline

## Relevant files / directories

### Must read first
- `CLAUDE.md` — architecture, design principles, quick start
- `IMPLEMENTATION_TODOS.md` — authoritative task list with completed/remaining items
- `codex-session-019ddf34-plan.md` — the full original plan (8703 chars, raw)

### Resolver package (core of recent work)
- `fastcode/src/fastcode/semantic_resolvers/__init__.py` — registry exports
- `fastcode/src/fastcode/semantic_resolvers/base.py` — `ResolverSpec`, `ToolDiagnostic`, `SemanticResolver`, `SemanticResolutionRequest`
- `fastcode/src/fastcode/semantic_resolvers/registry.py` — `default_resolver_registry()`, all 12 languages
- `fastcode/src/fastcode/semantic_resolvers/language_graph.py` — `GraphBackedResolver` for non-Python/C-family
- `fastcode/src/fastcode/semantic_resolvers/c_family.py` — C/C++ structural resolver
- `fastcode/src/fastcode/semantic_resolvers/python.py` — Python resolver
- `fastcode/src/fastcode/semantic_resolvers/graph_backed.py` — reusable base class
- `fastcode/src/fastcode/semantic_resolvers/patching.py` — `apply_resolution_patches`, patch precedence

### Pipeline integration
- `fastcode/src/fastcode/main.py` — `_apply_semantic_resolvers()` after incremental merge, SCIP multi-language
- `fastcode/src/fastcode/ir_merge.py` — `merge_ir()`, the seam resolvers plug into
- `fastcode/src/fastcode/semantic_ir.py` — canonical IR types (IRSnapshot, IRUnit, IREdge, IRSupport, etc.)
- `fastcode/src/fastcode/scip_indexers.py` — multi-language SCIP detection and running

### API layer (uncommitted changes need testing)
- `fastcode/src/fastcode/schemas/api.py` — **new untracked** — shared request/response models
- `fastcode/src/fastcode/schemas/__init__.py` — updated exports
- `fastcode/src/fastcode/api.py` — rewritten to use shared schema (not tested)
- `fastcode/src/fastcode/web_app.py` — rewritten to use shared schema (not tested)

### Core modules (template refactor targets)
- `fastcode/src/fastcode/core/` — pure algorithms
- `fastcode/src/fastcode/adapters/` — AST-to-IR, SCIP-to-IR
- `fastcode/src/fastcode/infrastructure/` — DB, filesystem, external tooling

### Tests
- `fastcode/tests/test_semantic_resolvers.py` — 715+ lines, registry/diagnostics/language detection/C/C++ tests
- `fastcode/tests/test_scip_indexers.py` — multi-language SCIP detection
- `fastcode/tests/test_api.py` — rewritten (needs run to verify)
- `fastcode/tests/core/test_boundary.py` — architecture dependency enforcement
- `fastcode/tests/test_e2e_semantic_pipeline.py` — end-to-end semantic pipeline
- `fastcode/tests/test_e2e_indexing.py` — end-to-end indexing
- `fastcode/tests/test_main.py` — main pipeline integration
- `fastcode/tests/core/` — core logic tests
