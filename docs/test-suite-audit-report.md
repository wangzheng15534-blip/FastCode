# FastCode Test Suite Quality Audit Report

**Date:** 2026-04-29
**Auditor:** Murat (bmad-tea + bmad-testarch-test-review)
**Scope:** Suite-wide (59 files, 41,130 lines, 1,445 tests)

---

## Executive Summary

The FastCode test suite is **genuinely above-average**. ~60% of tests provide real confidence through meaningful assertions against real production code. The remaining ~40% falls into three camps: weak assertions on real code (fixable), mock-testing-mock theater (needs rewrite), and modern frameworks applied to trivial code (theater).

| Verdict     | Files | Approx Tests |
|-------------|-------|--------------|
| SUBSTANTIAL | 20    | ~750         |
| MIXED       | 16    | ~550         |
| SUPERFICIAL |  3    | ~30          |

**Runtime:** 1,432 passed, 13 skipped, 86s, 146 warnings.

---

## Methodology

Applied the bmad-tea Test Quality Definition of Done checklist against every test file:

- No hard waits / conditionals / try-catch for flow control
- < 300 lines per test (flagged violations)
- Explicit assertions (no hidden asserts in helpers)
- Self-cleaning / parallel-safe
- Tests mirror usage patterns
- Prefer lower test levels (unit > integration > E2E)

Evaluated each file for:
1. **Tautological tests** — construct object, assert constructor worked
2. **Mock-testing-mock** — fake returns hardcoded data, test asserts hardcoded data
3. **Range-only assertions** — directional (`<`, `>`) instead of exact values for deterministic formulas
4. **Missing negative/edge case coverage**
5. **Hypothesis theater** — property-based testing on trivially branchless code

---

## File-by-File Verdicts

### Schema / Model Tests

| File | Tests | Lines | Verdict | Notes |
|------|-------|-------|---------|-------|
| `ir/test_ir_contracts.py` | 7 | 330 | **SUBSTANTIAL** | Tests merge rules, validators, graph building. Zero tautologies. |
| `retrieval/test_contracts.py` | 17 | 188 | **MIXED** | 6/17 tautological (field readback, constant uniqueness checks). Good negative/edge tests for HitFromRetrievalRow. |
| `test_semantic_ir.py` | 18 | 299 | **MIXED** | TestSourcePriority, TestConfidenceMapping, TestOccurrenceDedup are solid. TestLegacyConversion is field-existence only. |
| `test_scip_models.py` | 13 | 194 | **MIXED** | Reserved-field separation and non-dict filtering tests are real. Nested conversion tests are tautological. |
| `test_projection_models.py` | 6 | 130 | **SUPERFICIAL** | 5/6 tests are Hypothesis on trivial `to_dict()` accessors. Only `utc_now_iso` format check has value. |

### Adapter / Converter Tests

| File | Tests | Lines | Verdict | Notes |
|------|-------|-------|---------|-------|
| `adapters/test_ast_to_ir.py` | ~40 | 1,129 | **MIXED** | Good: type filtering, dedup, edge creation, line clamping. Bad: constant readbacks, pass-through assertions. |
| `adapters/test_scip_to_ir.py` | ~55 | 1,447 | **MIXED** | Good: role-based ref filtering, range normalization, language fallback. **Gap: `_normalize_kind` completely untested.** |
| `test_ir_merge.py` | ~27 | 779 | **SUBSTANTIAL** | Tests real alignment algorithm with same-name-different-class disambiguation. Caveat: legacy API mismatch, weak idempotency (counts only). |
| `test_ir_validators.py` | ~21 | 600 | **MIXED** | ~8/16 validator rules tested. **Untested: duplicate support/relation/embedding IDs, parent refs, anchor uniqueness, relation type/source.** |

### Core Algorithm Tests

| File | Tests | Lines | Verdict | Notes |
|------|-------|-------|---------|-------|
| `core/test_scoring.py` | 39 | 298 | **SUBSTANTIAL** | Exact math: `sigmoid(0)==0.5`, symmetry invariant, parametrized lookup tables with `pytest.approx`. |
| `core/test_fusion.py` | 14 | 412 | **MIXED** | Directional/monotonicity tests are good. **Zero exact-value tests** for alpha/k_code/k_doc. No RRF score verification. |
| `core/test_filtering.py` | 20 | 209 | **MIXED** | Rerank tests have exact values. Diversity penalty only asserts `< 0.8` instead of `== 0.4`. |
| `core/test_combination.py` | 9 | 65 | **MIXED** | Two tests have exact arithmetic. BM25 normalization only checks ordering. |
| `core/test_graph_build.py` | 11 | 139 | **MIXED** | Confidence mapping tests are exact. Main graph construction tests are shallow (counts only). |
| `core/test_boundary.py` | 8 | 251 | **SUBSTANTIAL** | **AST-based architectural invariant tests** — parses source and asserts no forbidden imports, no `**kwargs`. |
| `core/test_iteration.py` | 33 | 274 | **MIXED** | Simple functions have exact values. `calculate_repo_factor` and `initialize_adaptive_parameters` only have range assertions. |
| `core/test_snapshot.py` | 8 | 70 | **SUBSTANTIAL** | Determinism + collision resistance + format verification for hash functions. |

### Infrastructure / Integration Tests

| File | Tests | Lines | Verdict | Notes |
|------|-------|-------|---------|-------|
| `test_db_runtime.py` | 73 | 809 | **SUBSTANTIAL** | Real SQLite PRAGMA verification, transaction rollback, constraint enforcement. |
| `test_snapshot_store.py` | ~85 | 861 | **MIXED** | Excellent hypothesis round-trips. ~12 SQLite stub tests (lock, redo, staging) test constants. |
| `test_snapshot_store_extended.py` | ~105 | 1,538 | **SUBSTANTIAL** | **Hypothesis stateful machine** with 4 invariants. Unicode, large metadata, isolation tests. |
| `test_terminus_publisher.py` | ~55 | 873 | **SUBSTANTIAL** | Real payload construction logic. Conditional node creation, edge mapping, cross-subsystem merging. |
| `test_e2e_indexing.py` | 2 | 536 | **SUBSTANTIAL** | Real git repo → real Ollama → real SQLite/PostgreSQL. Full pipeline. |
| `infrastructure/test_db.py` | 4 | 69 | **SUBSTANTIAL** | Real SQL, frozen dataclass enforcement, round-trip. |
| `infrastructure/test_fs.py` | 3 | 30 | **SUBSTANTIAL** | Real filesystem. Trivial code under test. |
| `infrastructure/test_llm.py` | 5 | 94 | **SUPERFICIAL** | Mock returns hardcoded value, test asserts hardcoded value. Error tests verify Python built-in behavior. |

### API / Pipeline / Tool Tests

| File | Tests | Lines | Verdict | Notes |
|------|-------|-------|---------|-------|
| `test_api.py` | 3 | 150 | **SUPERFICIAL** | All 3 tests inject fake returning hardcoded dicts, then assert hardcoded values. One assertion is `x == x`. |
| `test_mcp_graph_tools.py` | 59 | 897 | **SUBSTANTIAL** | Real NetworkX graphs, BFS, Steiner tree, shortest path. Zero mocks. |
| `test_projection_transform.py` | 33 | 1,087 | **SUBSTANTIAL** | Hypothesis property tests + real pipeline. Supporting docs feature well-tested. |
| `test_redo_worker.py` | 25 | 384 | **MIXED** | Good property tests with real fakes. Half use MagicMock call assertions. |
| `test_outbox_pattern.py` | 24 | 389 | **MIXED** | Real SQLite tests and deterministic ID tests. Publisher tests are mock-heavy. |
| `test_retriever.py` | 6 | 247 | **SUBSTANTIAL** | Tests real fusion math and doc projection logic with noisy-or priors. |
| `test_snapshot_pipeline.py` | 11 | 183 | **SUBSTANTIAL** | Real SQLite round-trip persistence. Manifest HEAD, index run idempotency. |
| `test_main.py` | 17 | 453 | **MIXED** | Good gating/prefix logic tests. API/MCP endpoint tests are mock-theater. |
| `test_incremental_update.py` | 19 | 580 | **SUBSTANTIAL** | Full incremental cycle with structural assertions. |
| `test_nanobot_fastcode_tools.py` | 27 | 763 | **SUBSTANTIAL** | HTTP-level mocking with exact request body verification. |

### Utility / Benchmark Tests

| File | Tests | Lines | Verdict | Notes |
|------|-------|-------|---------|-------|
| `conftest.py` | 0 (infra) | 538 | **MIXED** | High-quality Hypothesis strategies. Violates CLAUDE.md "no conftest" convention. |
| `utils/test_utils.py` | 77 | 481 | **SUBSTANTIAL** | Hypothesis + thorough edge cases for 16 functions. |
| `utils/test_hashing.py` | 5 | 34 | **MIXED** | Tests determinism + uniqueness. Minimal coverage, no edge cases. |
| `utils/test_json.py` | 20 | 115 | **SUBSTANTIAL** | Real parsing: trailing commas, embedded JSON, comment preservation. |
| `test_path_utils.py` | 32 | 303 | **SUBSTANTIAL** | Real filesystem + Hypothesis. **`is_safe_path` has no path-traversal tests.** |
| `bench_graph_projection.py` | 2 | 103 | **SUBSTANTIAL** | Pre-validates correctness. Narrow coverage. |
| `bench_ir_merge.py` | 3 | 115 | **SUBSTANTIAL** | Full pipeline benchmarks. |
| `bench_validation.py` | 1 | 93 | **MIXED** | Single benchmark function. |

### Files Exceeding 300-Line Guideline

| File | Lines |
|------|-------|
| `adapters/test_scip_to_ir.py` | 1,447 |
| `test_snapshot_store_extended.py` | 1,538 |
| `adapters/test_ast_to_ir.py` | 1,129 |
| `test_projection_transform.py` | 1,087 |
| `test_mcp_graph_tools.py` | 897 |
| `test_terminus_publisher.py` | 873 |
| `test_snapshot_store.py` | 861 |
| `test_db_runtime.py` | 809 |
| `test_ir_merge.py` | 779 |
| `test_nanobot_fastcode_tools.py` | 763 |

---

## Systemic Issues

### 1. Mock-Testing-Mock Pattern

Files: `test_api.py`, `infrastructure/test_llm.py`, `test_main.py` (API/MCP tests)

Pattern: Create fake → inject via patch → assert fake's hardcoded values appear in output. Tests FastAPI routing / Python attribute access, not business logic. These tests cannot catch real bugs.

### 2. Range-Only Assertions on Deterministic Formulas

Files: `test_fusion.py`, `test_filtering.py`, `test_combination.py`, `test_iteration.py`

Pattern: Test mathematical functions with `<`, `>`, or `[lo, hi]` instead of hand-computed exact values. A regression that applies a penalty twice or shifts a formula by a constant can pass these tests.

### 3. Hypothesis Theater

Files: `test_projection_models.py`, parts of `test_ast_to_ir.py`, `test_scip_to_ir.py`

Pattern: `@given` decorator on functions with no branch logic. Hypothesis fuzzes inputs but the assertion is "pass value in, get same value out" — adding randomized inputs to a tautology doesn't make it meaningful.

### 4. Legacy vs Canonical API Mismatch

Files: `test_ir_merge.py`, `test_semantic_ir.py`, adapter tests

Pattern: Tests use legacy fields (`symbols`, `occurrences`, `edges`, `documents`, `attachments`) while source code operates on canonical fields (`units`, `supports`, `relations`, `embeddings`). `IRSnapshot.__init__` bridges this, potentially masking canonical-path bugs.

### 5. Convention Violation

`CLAUDE.md` states *"No conftest.py — each test file is self-contained."* A 538-line `conftest.py` exists with strategies, fixtures, and factories used by ~16 files.

---

## Critical Missing Tests

| Priority | Gap | File | Risk |
|----------|-----|------|------|
| **P0** | `is_safe_path` has no path-traversal tests | `test_path_utils.py` | Security: path traversal is the primary threat |
| **P0** | Zero exact-value tests for fusion alpha/k | `test_fusion.py` | Core algorithm: directional assertions miss formula regressions |
| **P1** | 8 validator rules completely untested | `test_ir_validators.py` | Integrity: validator is the gatekeeper for all IR data |
| **P1** | `_normalize_kind` mapping untested | `test_scip_to_ir.py` | Adapter: kind mapping bugs corrupt symbol types silently |
| **P1** | No determinism tests for pure functions | All core/ files | Reliability: basic invariant for pure scoring/fusion functions |
| **P2** | `test_api.py` has zero real behavior tests | `test_api.py` | Signal: 3 tests that cannot fail provide false confidence |
| **P2** | Diversity penalty exact arithmetic untested | `test_filtering.py` | Precision: double-penalty bug would pass current test |
| **P2** | `test_llm.py` tests mock wiring, not behavior | `infrastructure/test_llm.py` | Waste: tests Python attribute access, not LLM integration |
| **P3** | `test_projection_models.py` Hypothesis is theater | `test_projection_models.py` | Hygiene: `@given` on `to_dict()` is overhead without value |
| **P3** | No mutation testing infrastructure | Suite-wide | Gap: no mechanism to measure test suite mutation score |

---

## Recommendations (Prioritized)

### P0 — Security & Algorithm Correctness

1. **Add path-traversal tests to `is_safe_path`** — `../../etc/passwd`, symlink attacks, null byte injection, URL-encoded traversal
2. **Add exact-value tests for fusion parameters** — hand-compute alpha/k_code/k_doc for 3 known query profiles, assert exact values with `pytest.approx`
3. **Add Schemathesis API fuzz testing** — generate API tests from FastAPI schema, test `/query`, `/repos`, `/symbols/find`, `/graph/*` endpoints with fuzzed inputs

### P1 — Coverage Gaps

4. **Fill validator coverage gaps** — add tests for 8 untested rules (duplicate support/relation/embedding IDs, parent refs, anchor uniqueness, relation type/source, embedding ref/source)
5. **Add `_normalize_kind` tests** — parametrize across all known kinds (`documentation→doc`, `module→file`, `type→class`) plus unknown fallback
6. **Add Mutmut mutation testing** — measure mutation score on `core/scoring.py`, `core/fusion.py`, `adapters/scip_to_ir.py`, `ir_validators.py`
7. **Migrate tests from legacy to canonical API** — replace `symbols`/`occurrences`/`edges` with `units`/`supports`/`relations` in test construction

### P2 — Test Quality Improvements

8. **Rewrite `test_api.py`** — test actual business logic, not FastAPI routing. Use real SnapshotStore with SQLite, exercise handler logic
9. **Add determinism tests** — call pure functions twice with same inputs, assert structural equality
10. **Replace range-only assertions with exact values** — `test_fusion.py`, `test_filtering.py`, `test_combination.py`, `test_iteration.py`
11. **Rewrite or remove `test_llm.py`** — test error handling, retry logic, rate limiting, or delete if the 6-line wrapper isn't worth testing

### P3 — Hygiene

12. **Remove or strengthen `test_projection_models.py`** — delete Hypothesis theater, add tests for actual projection model logic
13. **Update CLAUDE.md** — document the conftest.py or split it into `strategies.py` + `factories.py`
14. **Split oversized test files** — 10 files exceed the 300-line guideline; split by test concern
