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

## Critical verification currently green

- Full smoke gate:
  - `1613 passed, 7 skipped`
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

Typed records now exist for manifests and snapshot refs.

Still raw-dict-heavy at boundaries:
- some API-facing manifest / artifact payload adapters
- index run records
- parts of snapshot metadata payloads

### 6. Add a few remaining critical regressions

- header-language classification for `.h` as C vs C++
- query-time semantic escalation changing IR graph expansion behavior end to end
- experimental SCIP warning propagation in full index flow
- helper asset execution from packaged installs

## Rules for updating this file

- Update this file whenever a hardening slice lands.
- Remove completed items from the remaining-work sections or move them into landed status.
- Do not create separate audit/status markdown files unless the user explicitly asks for one.
