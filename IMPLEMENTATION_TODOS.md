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
- Added shared helper-backed semantic resolver infrastructure for compiler/LSP upgrade layers.
- Added structured helper paths for JavaScript/TypeScript, Go, Java, Rust, C#, Zig, Fortran, and Julia semantic upgrades.
- Added focused mapping tests for helper-backed semantic relation emission across core and added languages.

## Remaining semantic resolver work

- Deepen helper-backed resolvers toward richer semantics:
  - Expand JavaScript/TypeScript beyond import/call facts to type and inheritance upgrades.
  - Expand Java beyond import/call facts to type and inheritance upgrades.
  - Expand Go beyond import/call facts to type and interface/implementation upgrades.
  - Replace heuristic Rust/C#/Zig/Fortran/Julia helper parsing with stronger frontend-native semantic facts where available.
- Add resolver-specific capability gating so unresolved graphs do not run every language resolver indiscriminately.
- Add query-time semantic escalation budget instead of index-time-only upgrades.
- Preserve stronger distinction between heuristic helper facts and true frontend-native semantic facts where available.

## Remaining SCIP/indexing work

- Replace the placeholder Zig/Fortran/Julia SCIP command wiring with verified command contracts or feature-gated optional integrations.
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
