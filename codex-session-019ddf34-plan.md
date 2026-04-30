<proposed_plan>
# FastCode Semantic Resolver Expansion And Template Rewrite

## Summary

Refactor FastCode around the template architecture and implement the proposal as a real semantic upgrade pipeline: Tree-sitter builds structural candidates, SCIP anchors symbols where available, and language-specific compiler/AST frontends emit `ResolutionPatch` upgrades through one resolver bus.

Current C/C++ audit outcome: the existing C/C++ resolver is useful but shallow. It only resolves includes and inheritance heuristically from existing metadata, has no compiler-backed call/type/macro resolution, treats `.h` as C in SCIP detection, and has patch-merge gaps such as inheritance keys using `base` while emitted metadata uses `base_name`.

Target language set: Python, JavaScript, TypeScript, Java, Go, Rust, C#, C, C++, Zig, Fortran, and Julia.

Compatibility choice: breaking changes are allowed. Old snapshots/caches may be invalidated rather than migrated if that keeps the architecture clean.

## Implementation Changes

- Rewrite package boundaries to match the template:
  - `schemas/`: Pydantic API models plus frozen dataclass DTOs for IR, resolver specs, DB records, indexing requests, and snapshots.
  - `core/`: pure algorithms only: IR merge, resolution patch merge, capability gating, graph scoring, retrieval fusion, query planning. No Pydantic, DB, filesystem, subprocess, LSP, network, or model calls.
  - `infrastructure/`: DB stores, filesystem, SCIP runners, compiler/LSP frontends, language helper subprocesses, external API clients.
  - `api/` and `main/`: route/CLI/composition wiring only, with explicit field-by-field translation into dataclasses.

- Replace the current resolver implementation with a capability-driven resolver bus:
  - Keep one canonical `ResolutionPatch` contract.
  - Add `ResolverSpec` with `language`, `capabilities`, `cost_class`, `requires_tools`, `source_name`, and `frontend_kind`.
  - Add `SemanticResolutionRequest` with snapshot id, target paths, budget, candidate units, candidate relations, SCIP anchors, repo root, and tool context.
  - Resolvers must return patches only. They must not mutate snapshots, stores, graph builders, or global state.

- Harden C/C++ first:
  - Replace the current C/C++ heuristic resolver with a Clang-backed resolver using `compile_commands.json` when present and inferred fallback args when absent.
  - Resolve includes, calls, inheritance, namespaces, templates where Clang exposes stable targets, macro-origin spans, and overloaded function targets.
  - Keep heuristic include matching only as fallback and mark fallback edges as `candidate` or `structural`, never `semantically_resolved`.
  - Fix `.h` handling by classifying headers as C or C++ based on nearby source files, compile database entries, and include ownership.
  - Fix patch merge keys so inheritance uses `base_name`, calls use `call_name`, imports use `module`, and source preference uses the highest-ranked support source rather than a single sorted source.

- Add compiler/AST-backed resolvers for the full language set:
  - Python: LibCST metadata plus Jedi for qualified names, imports, scopes, inheritance, and local call targets.
  - JS/TS: TypeScript Compiler API helper for symbols, imports, type-directed calls, class inheritance, JSX/TSX support.
  - Java: Eclipse JDT helper for bindings, imports, method calls, inheritance, interfaces, and overload targets.
  - Go: `go/packages` helper for imports, defs/refs, calls, receiver methods, interface implementation candidates.
  - Rust: rust-analyzer LSP plus SCIP anchors for modules, defs/refs, calls, impls, traits, and macro-aware locations where available.
  - C#: Roslyn helper for symbols, calls, inheritance, interfaces, extension methods, and using aliases.
  - Zig: Tree-sitter Zig plus ZLS/Zig compiler-backed resolver for imports, containers, functions, call candidates, and comptime-limited metadata.
  - Fortran: fparser plus fortls for modules, `use` associations, procedures, derived types, calls, and type extension.
  - Julia: JuliaSyntax/CSTParser plus LanguageServer.jl for modules, imports, functions, methods, types, and dynamic-dispatch call candidates.

- Add a deterministic escalation policy:
  - Index-time default: run structural Tree-sitter and SCIP for detected languages.
  - Run compiler-backed resolvers for changed files, unresolved candidate edges, and hotspot files.
  - Query-time upgrade: refine only the induced subgraph for path, impact, inheritance, call-chain, or “what breaks” queries.
  - Store unresolved capabilities on relations as `pending_capabilities`; never pretend unsupported dynamic or macro-heavy edges are precise.

- Normalize persistence around typed records:
  - Store modules must return frozen dataclass records, not `dict[str, Any]`.
  - Cache/snapshot formats move to versioned dataclass schemas.
  - On first run after refactor, old cache artifacts are ignored and regenerated.

## Public Interfaces And Types

- Add stable dataclass APIs:
  - `SemanticCapability`: enum-like string constants such as `resolve_calls`, `resolve_imports`, `resolve_types`, `resolve_inheritance`, `expand_macros`, `recover_qualified_names`.
  - `ResolverSpec`: resolver metadata and required tool declarations.
  - `SemanticResolutionRequest`: immutable resolver input.
  - `ResolutionPatch`: supports, relations, unit metadata updates, diagnostics, stats.
  - `ToolDiagnostic`: missing tool, invalid project config, partial analysis, timeout, parse failure.

- Add configuration knobs:
  - `semantic_resolution.enabled = true`
  - `semantic_resolution.default_budget = "changed_files"`
  - `semantic_resolution.query_time_budget = "path_critical"`
  - `semantic_resolution.required_tools = true`
  - `semantic_resolution.timeout_seconds_per_language = 30`
  - `semantic_resolution.invalidate_legacy_cache = true`

- Keep CLI/API behavior conceptually the same where practical, but allow request/response schema cleanup:
  - API request models live under `schemas/`.
  - Routes explicitly map request fields into dataclass requests.
  - No `model_dump()` handoff into application logic.

## Test Plan

- Architecture tests:
  - `core/` imports no Pydantic, subprocess, DB, filesystem, LSP, HTTP, or model SDK modules.
  - `infrastructure/` does not import `api/`.
  - DB/store public methods return dataclass records, not raw dicts.
  - API routes use explicit field mapping, not `**request.model_dump()`.

- Resolver contract tests:
  - Registry discovers all target languages.
  - Each resolver advertises capabilities and required tools.
  - Resolver failures degrade into diagnostics and do not crash indexing.
  - Patch application preserves provenance, merges support ids, and upgrades resolution state monotonically.

- C/C++ tests:
  - `.h` ownership classification for C and C++ repos.
  - Compile database path resolution.
  - Include resolution with include directories.
  - Namespaced inheritance, templated base classes, overloaded calls, macro-origin diagnostics.
  - Heuristic fallback stays lower confidence than Clang-backed results.

- Language fixture tests:
  - One small fixture repo per language with import/module, inheritance/type relation, direct call, unresolved/dynamic case, and cross-file symbol.
  - Verify expected `support_sources`, `pending_capabilities`, `resolution_state`, and canonical target ids.
  - Zig, Fortran, and Julia tests must include dynamic/partial-resolution cases so the system proves it can be honest about uncertainty.

- End-to-end tests:
  - Multi-language repo indexing produces one canonical IR.
  - SCIP unavailable path still indexes structurally with diagnostics.
  - Query-time semantic upgrade improves path/impact answers without rewriting stored snapshots unexpectedly.
  - Full quality gate: ruff check, pyright, pytest, architecture tests, and selected e2e fixtures.

## Assumptions And Defaults

- External language tooling may be required. The implementation should document install commands and fail with actionable diagnostics when tools are missing.
- Breaking internal snapshot/cache changes are acceptable. Regenerate rather than migrate old artifacts unless a migration is trivial.
- Compiler-backed facts outrank SCIP only when they resolve a relation SCIP does not express; SCIP remains the preferred symbol identity source when available.
- Dynamic dispatch-heavy languages may emit `candidate` relations rather than precise edges. This is correct behavior, not a failure.
- The first complete implementation should prioritize correctness and provenance over speed, then add caching and query-time budgets once all languages share the same contract.
</proposed_plan>