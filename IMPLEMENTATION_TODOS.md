# FastCode Full TODOs

## Completed in current slice

- Expanded semantic resolver contract with `ResolverSpec` and `ToolDiagnostic`.
- Registered broad language resolver set: Python, JavaScript, TypeScript, Java, Go, Rust, C#, C, C++, Zig, Fortran, Julia.
- Added graph-backed fallback resolver implementations for non-Python/C-family languages.
- Hardened C/C++ resolver diagnostics and relation source preference logic.
- Removed direct `model_dump()` handoff in API and web multi-repo endpoints.
- Moved shared HTTP request/response models into `fastcode.schemas.api` and rewired `api.py` / `web_app.py` to import them.
- Expanded default extension and language detection for Zig, Fortran, and Julia.
- Added focused tests for resolver registry, diagnostics, and new language detection.

## Remaining semantic resolver work

- Replace graph-backed fallback resolvers with true frontend-backed adapters:
  - JavaScript/TypeScript via Compiler API helper.
  - Java via JDT or language-server-backed binding resolution.
  - Go via `go/packages` or gopls-backed resolver.
  - Rust via rust-analyzer semantic queries.
  - C# via Roslyn-backed helper.
  - Zig via ZLS or zig AST/semantic tooling.
  - Fortran via fortls or fparser-backed semantic resolver.
  - Julia via LanguageServer.jl-backed resolver.
- Add resolver-specific capability gating so unresolved graphs do not run every language resolver indiscriminately.
- Add query-time semantic escalation budget instead of index-time-only upgrades.
- Distinguish structural fallback relations from true compiler-confirmed relations in `resolution_state` and metadata.

## Remaining SCIP/indexing work

- Replace the placeholder Zig/Fortran/Julia SCIP command wiring with verified command contracts or feature-gated optional integrations.
- Preserve all generated SCIP artifacts for multi-language indexing instead of copying only the first artifact path.
- Add tests for multi-language SCIP merge path in `FastCode.index_repository`.
- Expand e2e configs and fixtures to include the new supported extensions.

## Remaining template-rule refactor work

- Move mutable dict-heavy core functions toward typed dataclass inputs/outputs.
- Convert store/runtime methods that still return raw `dict[str, Any]` into frozen dataclass records.
- Split orchestration from logic in `main.py`; reduce mixed responsibilities across loading, indexing, graphing, SCIP, and persistence.
- Tighten architecture tests to cover `main`, `api`, `infrastructure`, and store modules consistently.

## Remaining verification work

- Run focused `pyright` on touched semantic resolver and indexing modules.
- Keep the focused regression suite green: `test_api`, `test_semantic_resolvers`, `test_scip_indexers`, and `core/test_boundary`.
- Add regression tests for header-language classification (`.h` as C vs C++).
- Add API tests that assert explicit source mapping for `index-multiple`.
- Add integration tests that verify Zig/Fortran/Julia files enter the indexing pipeline when present.
