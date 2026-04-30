## Goal

Audit and expand FastCode's semantic resolver layer:
1. Check the template `AGENTS.md` at `/home/dev/repo_template/python_v1.0_single/AGENTS.md` for correctness, refine if needed.
2. Audit FastCode source code against the refined template rules.
3. Assess a proposal to ad-hoc integrate language-specific AST as a selective semantic upgrade layer.
4. Implement: review/audit the existing C/C++ resolver, fix issues, implement resolvers for all core languages (JS, TS, Java, Go, Rust, C#, C, C++, Zig, Fortran, Julia), and fully refactor the codebase to follow the template architecture.

## Instructions

- Audit the template `AGENTS.md` first, fix it, then audit FastCode against the refined rules.
- The semantic resolver upgrade layer must sit **after** `merge_ir()` in the pipeline, not before it — the useful identity state is created by the merge.
- Language-specific frontends should be a selective semantic upgrade layer, not a third rival extractor pipeline. Tree-sitter skeleton + SCIP anchoring + resolver upgrades through one bus.
- New languages that lack compiler backends should use graph-backed structural evidence and emit diagnostics when required external tools are unavailable, rather than silently claiming precision.
- Follow the template layered architecture: `api -> infrastructure -> core -> schemas` dependency direction.
- No `model_dump()` boundary leaks at the API layer — use schema-bound translation instead.

## Discoveries

- **Template `AGENTS.md` issues found and fixed**: inconsistent `schema` vs `schemas` naming, pyright commands not run through `uv`, confusing dependency direction around `api`/`infrastructure`, security/tooling commands named but not executable.
- **FastCode audit findings**:
  - Two `model_dump()` API boundary leaks in `api.py` and `web_app.py` — removed.
  - Architecture test (`test_boundary.py`) was pointed at obsolete `effects/db.py` path and was skipping — updated to check current `infrastructure/db.py`.
  - C/C++ resolver was shallow: only resolves includes and inheritance heuristically, no compiler-backed call/type/macro resolution, treats `.h` as C in SCIP detection, and has patch-merge gaps (inheritance keys use `base` while emitted metadata uses `base_name`).
  - `main.py` still hard-coded `run_scip_python_index()` even though `scip_indexers.py` already had a dormant multi-language map including `scip-clang`.
  - Multiple SCIP outputs needed combining before merge, not passing `None` through AST merge path — fixed.
- **C/C++ Clang probe**: `clang -Xclang -ast-dump=json` works locally, but `scip-clang` and `scip-python` are not installed.
- **Resolver patch precedence issue**: AST and resolver edges can share the same semantic slot but have different targets with the same `resolution_state` — fixed so resolver-emitted relations override same-slot AST heuristics.
- **Relation source ranking**: Multi-source relations were accidentally ranked by alphabetically first support rather than evidence quality — fixed.

## Accomplished

### Completed
- Refined `/home/dev/repo_template/python_v1.0_single/AGENTS.md`: fixed `uv` commands, corrected `schemas` naming, made `api -> infrastructure -> core -> schemas` dependency rule explicit, tightened request lifecycle wording, fixed Schemathesis command.
- Removed two `model_dump()` API boundary leaks from `api.py` and `web_app.py`.
- Updated architecture test to check current `infrastructure/db.py`.
- Added `ResolverSpec` and `ToolDiagnostic` to `semantic_resolvers/base.py`.
- Broadened default resolver registry (`registry.py`) to cover Python, JS, TS, Java, Go, Rust, C#, C, C++, Zig, Fortran, Julia via `language_graph.py`.
- Fixed C/C++ resolver: header classification (`.h` treated as C), relation source ranking, merge key normalization (`base` vs `base_name`).
- Widened SCIP orchestration from Python-only to multi-language detected indexes.
- Fixed multi-SCIP merge bug (outputs need combining before AST merge, not `None` passthrough).
- Added new language extension detection for Zig, Fortran, Julia.
- Broadened default indexed extensions so new languages enter the indexing pipeline.
- All tests and ruff passing.

### In Progress / Left
- C/C++ resolver is still **structural only** (no compiler-backed call/type/macro resolution) — the Clang AST probe was done but not wired as a production adapter.
- Zig, Fortran, Julia resolvers are graph-backed fallbacks with diagnostics — no language-specific adapters implemented yet.
- The full template-rule refactor of the rest of the FastCode codebase was requested but only the resolver/boundary slice was implemented.
- TS/JS, Java, Go, Rust, C# resolvers registered but rely on generic graph-backed evidence — no language-specific logic.

## Relevant files / directories

### Template
- `/home/dev/repo_template/python_v1.0_single/AGENTS.md` — refined template rules

### FastCode resolver package (new/modified)
- `fastcode/src/fastcode/semantic_resolvers/` — entire package
- `fastcode/src/fastcode/semantic_resolvers/__init__.py` — registry exports
- `fastcode/src/fastcode/semantic_resolvers/base.py` — `ResolverSpec`, `ToolDiagnostic`, typed contract
- `fastcode/src/fastcode/semantic_resolvers/registry.py` — default resolver registry (all 12 languages)
- `fastcode/src/fastcode/semantic_resolvers/language_graph.py` — generic graph-backed resolver for new languages
- `fastcode/src/fastcode/semantic_resolvers/c_family.py` — C/C++ resolver (structural, includes + inheritance)
- `fastcode/src/fastcode/semantic_resolvers/python.py` — Python resolver (refactored)
- `fastcode/src/fastcode/semantic_resolvers/graph_backed.py` — reusable graph-backed base
- `fastcode/src/fastcode/semantic_resolvers/patching.py` — patch application and precedence logic

### FastCode core (modified)
- `fastcode/src/fastcode/main.py` — pipeline wiring: resolver pass after incremental merge, SCIP multi-language, removed `call_graph_bridge`
- `fastcode/src/fastcode/api.py` — removed `model_dump()` boundary leak
- `fastcode/src/fastcode/web_app.py` — removed `model_dump()` boundary leak
- `fastcode/src/fastcode/scip_indexers.py` — widened from Python-only to multi-language
- `fastcode/src/fastcode/parser.py` — broadened default extensions, fixed `.h`/`.hpp` detection

### Tests (modified/created)
- `fastcode/tests/test_semantic_resolvers.py` — registry, diagnostics, language detection, C/C++ patch hardening
- `fastcode/tests/test_scip_indexers.py` — multi-language SCIP detection
- `fastcode/tests/core/test_boundary.py` — updated to check `infrastructure/db.py`
