# FastCode Status Tracker

This file is the maintained source of truth for implementation status.
The ad hoc audit report was removed on May 3, 2026.

## Current state

The core architecture now works as a three-layer pipeline:

1. `plain_ast_embedding`
   - upstream-style tree-sitter structure + embeddings
2. `unified_ir_scip_merge`
   - canonical IR plus SCIP merge and persisted SCIP lineage
3. `language_specific_semantic_upgrade`
   - helper-backed or graph-backed resolver upgrades

The pipeline now exposes explicit layer status, metrics, warnings, and non-silent fallback behavior.

## Landed hardening

- Resolver contract expanded with:
  - `ResolverSpec`
  - `SemanticCapability`
  - `ToolDiagnostic`
  - `ResolutionPatch`
- Registered resolver coverage for:
  - Python
  - JavaScript / TypeScript
  - Java
  - Go
  - Rust
  - C#
  - C / C++
  - Zig
  - Fortran
  - Julia
- Added graph-backed fallback resolvers for non-C-family languages.
- Added helper-backed upgrade infrastructure and packaged helper assets into builds.
- Helper-backed resolvers now:
  - degrade to structural fallback on helper failure
  - preserve explicit diagnostics and helper failure stats
  - honor the indexed repository root instead of shell `cwd`
- Multi-language SCIP lineage now persists all artifact refs instead of collapsing to one row.
- Experimental SCIP languages are now explicit:
  - Zig
  - Fortran
  - Julia
  - pipeline layer 2 records warnings and metrics for them
- Reuse-path and legacy snapshot metadata now backfill and persist:
  - `pipeline_layers`
  - `pipeline_metrics`
  - SCIP artifact lineage metadata
- Query-time semantic escalation now exists for snapshot-scoped path/impact-style queries.
- Store boundary typing improved with frozen records for:
  - `ManifestRecord`
  - `SnapshotRefRecord`
- Typed default factories were added to semantic resolver base models and API schema defaults.
- Core helper semantic fidelity improved:
  - TypeScript helper emits inheritance facts
  - Go helper emits embedded-relationship inheritance facts
  - Java helper emits `extends` / `implements` inheritance facts
- Context-aware C/C++ header classification now upgrades ambiguous `.h` files:
  - parser-time language detection uses C++ content markers and sibling-source layout
  - C/C++ resolver dispatch applies the same heuristics during snapshot-scoped upgrades
- Store-boundary typing hardened:
  - `SnapshotRecord` frozen dataclass with `to_dict()`/`from_dict()`
  - `get_snapshot_record()` and `save_snapshot()` return typed records instead of raw dicts
  - All call sites migrated from bracket access to attribute access
- Snapshot query concurrency safety:
  - `query_snapshot` serializes load+query with a `threading.Lock`
  - Thread-barrier regression tests (2 and 3 concurrent callers) verify artifact isolation

## Critical verification currently green

- Full smoke gate:
  - `1663 passed, 7 skipped`
- Focused regression areas repeatedly verified:
  - `test_api`
  - `test_manifest_store`
  - `test_query_handler`
  - `test_scip_indexers`
  - `test_semantic_resolvers`
  - `test_snapshot_pipeline`
  - `test_snapshot_store`

## Remaining highest-value work

### 1. Deepen semantic fidelity in core helpers

- JavaScript / TypeScript:
  - export alias facts
  - stronger type facts
  - richer interface / implementation facts
- Go:
  - stronger interface implementation facts
  - richer type facts beyond embedded relationships
- Java:
  - stronger type/member binding facts beyond regex/source heuristics

### 2. Replace narrow helper heuristics in secondary languages

- Rust
- C#
- Zig
- Fortran
- Julia

These currently emit useful structured facts, but they are still narrower than frontend-native semantic APIs.

### 3. Tighten query-time semantic escalation

- Current hook is intentionally conservative and snapshot-scoped.
- Remaining work:
  - use better terminal selection than top retrieved file paths
  - support induced-subgraph upgrades for path queries
  - add deeper query-time tests around real graph expansion effects

### 4. Keep experimental SCIP integrations honest

- Zig / Fortran / Julia are now marked experimental in metadata.
- Remaining work:
  - verify actual command contracts in more environments
  - add availability-gated integration tests for artifact loadability

### 5. Continue store-boundary typing

Typed records now exist for manifests, snapshot refs, and snapshot records.

Still raw-dict-heavy at boundaries:
- some API-facing manifest / artifact payload adapters
- index run records

### 6. Add a few remaining critical regressions

### 6b. Query-time semantic escalation changing IR graph expansion behavior end to end

**Status:** Test guards added.

**Audit verdict:** FAIL (risk score 6/9 — P2×I3)

**Problem:** Existing tests verified mechanism (callback called, retrieve count) but not behavior (retrieval results actually change after IR graph expansion).

**Test coverage added** (`fastcode/tests/test_query_handler.py`):
- `test_semantic_escalation_changes_retrieval_results_end_to_end` — verifies answer generator receives expanded second-retrieval results (3 files vs 1)
- `test_local_budget_triggers_escalation_for_find_intent` — budget="local" path with find intent
- `test_escalate_query_semantics_returns_skipped_when_snapshot_not_found` — early return when snapshot missing
- `test_escalate_query_semantics_returns_skipped_when_no_target_paths` — early return when no paths extractable
- `test_escalate_query_semantics_returns_degraded_when_warnings_present` — degraded status with resolver warnings
- `test_escalation_does_not_rerun_when_callback_returns_none` — no rerun when callback returns None

### 6c. Experimental SCIP warning propagation in full index flow

**Status:** Already covered by existing test.

**Audit verdict:** CONCERNS/borderline PASS (risk score 2/9 — P1×I2)

**Existing test coverage:**
- `test_pipeline_layer2_records_experimental_scip_languages_non_silently` (`test_snapshot_pipeline.py:582`) — comprehensive single-language chain: detection → warning string → `layer2["warnings"]` → `result["warnings"]` + metrics
- `test_experimental_scip_profiles_are_marked_explicitly` (`test_scip_indexers.py:93`) — profile classification guard
- `test_experimental_scip_languages_set` (`test_semantic_resolvers.py:2156`) — canonical set guard

**Minor gaps (low priority):**
- Multi-language warning string (sorted join for Zig+Fortran) untested
- Warning persistence to snapshot store metadata untested

### 6d. Helper asset execution from packaged installs

**Status:** Test guards added.

**Audit verdict:** CONCERNS (risk score 6/9 — P2×I3)

**Problem:** `_helper_path()` at `helper_backed.py:248` uses `Path(__file__).with_name(self.helper_filename)`. All 12 existing helper tests bypass this method via patches. Zero tests verify helper assets exist on disk.

**Test coverage added** (`fastcode/tests/test_semantic_resolvers.py`):
- `test_helper_path_returns_existing_file_for_each_resolver` — parametrized across 9 resolvers; asserts `_helper_path().exists()` and correct filename
- `test_helper_command_includes_existing_helper_path` — full chain `_helper_path()` → `_helper_command()` → file exists
- `test_helper_backed_resolver_degrades_gracefully_when_helper_file_missing` — missing asset simulation with `_DummyFallbackResolver`; asserts `structural_fallback` tier and `tool_invocation_failed` diagnostic
- `test_helper_path_co_located_with_resolver_module` — co-location invariant: helper parent directory matches `helper_backed.py` module directory

## Rules for updating this file

- Update this file whenever a hardening slice lands.
- Remove completed items from the remaining-work sections or move them into landed status.
- Do not create separate audit/status markdown files unless the user explicitly asks for one.
