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

## Release readiness verdict

FastCode is currently in a hardened pre-release state, not a real stable release.

The core algorithmic path is much stronger than the original prototype: snapshot indexing, IR/SCIP merge, semantic resolver fallback, query-time escalation, artifact isolation, and async endpoint offloading all have regression coverage and green smoke gates.

The remaining gap to a stable release is no longer mainly "does the core pipeline work?" The gap is production evidence and operational hardening:
- install/package reproducibility from a clean environment
- real external-tool integration evidence across supported languages
- real PostgreSQL/backend semantics beyond local fakes
- API/file-upload security hardening
- service-wide state isolation under concurrent load/query/mutation traffic
- release documentation, compatibility policy, and deployment runbooks

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
- Query/agency hardening:
  - agency-mode `query()` and `query_stream()` preserve cheap detected intent
  - iterative-agent standard retrieval applies caller filters after rerank
  - regression test proves semantic escalation can enable real IR graph expansion, not just a mocked second retrieval
- Async endpoint hardening:
  - REST API repository load/index/cache-load/multi-index/upload paths offload blocking work with `asyncio.to_thread`
  - web UI upload and upload+index paths offload ZIP extraction, repository load, and indexing
  - REST API and web UI delete/cache/refresh maintenance endpoints offload blocking mutations
- Lock fencing hardening:
  - PostgreSQL same-owner lock refresh preserves the current fencing token instead of invalidating in-flight work

## Critical verification currently green

- Full smoke gate:
  - `1677 passed, 7 skipped`
- Focused regression areas repeatedly verified:
  - `test_api`
  - `test_manifest_store`
  - `test_query_handler`
  - `test_scip_indexers`
  - `test_semantic_resolvers`
  - `test_snapshot_pipeline`
  - `test_snapshot_store`

## Stable Release Gap

### P0: Must Close Before Stable Release

These are release blockers. Do not tag a stable release while any P0 item is open.

### P0.1 Install and packaging reproducibility

**Gap:** Tests run in the repository checkout, but stable users install packages, optional extras, and helper assets from built artifacts.

**Required work:**
- Build and test both wheel and sdist from a clean checkout.
- Run import/CLI/API smoke tests from the installed wheel, not from editable source.
- Verify helper assets are included in built artifacts and executable from installed package paths.
- Define supported Python versions and run the release gate against each supported version.
- Split optional extras clearly: API server, PostgreSQL, SCIP/tooling, docs ingestion, dev/test.
- Pin or constrain high-risk runtime dependencies enough to avoid resolver drift.

**Exit criteria:**
- `pip install dist/*.whl` in a fresh virtualenv can run CLI, API import, helper path checks, and a tiny index/query smoke.
- Release docs list exact install commands for local-only and production/PostgreSQL modes.

### P0.2 Real external-tool integration matrix

**Gap:** Resolver and SCIP contracts are represented in metadata and unit tests, but several languages still rely on helper heuristics or experimental tooling. Stable release claims must match real command behavior.

**Required work:**
- Add availability-gated integration tests for actual SCIP/indexer commands where supported.
- Verify command contracts for Python, JavaScript, TypeScript, Java, Go, Rust, C#, C, and C++.
- Keep Zig, Fortran, and Julia explicitly experimental unless real command contracts are validated in CI-like environments.
- Persist and test multi-language experimental warning strings and warning metadata.
- Document exact system dependencies per language.

**Exit criteria:**
- Stable language list has real command smoke coverage or is downgraded in docs to structural/experimental support.
- Experimental languages never appear as stable in API metadata, docs, or release notes.

### P0.3 Production storage semantics

**Gap:** SQLite local mode is well covered, and Postgres-oriented code exists, but stable production semantics require real backend tests for locks, migrations, outbox, redo, and relational fact persistence.

**Required work:**
- Run a real PostgreSQL integration suite for snapshot save/load, manifests, locks, fencing, staging, outbox, redo tasks, SCIP refs, and graph facts.
- Verify migration/idempotency behavior on an existing database, not only fresh schema creation.
- Document SQLite as local/single-process only where lock/fact APIs are no-ops.
- Add stale-lock and ownership tests against real PostgreSQL transactions.
- Add backup/restore and schema version compatibility notes.

**Exit criteria:**
- Postgres integration tests are part of the release gate or explicitly required before tagging.
- SQLite limitations are documented in user-facing release docs.

### P0.4 API and file-upload security hardening

**Gap:** Blocking upload/index work is now offloaded, but upload extraction is still not release-grade security.

**Required work:**
- Replace raw `ZipFile.extractall()` with safe member validation.
- Reject absolute paths, `..` traversal, symlinks, device files, and zip bombs.
- Enforce expanded-size, file-count, and per-file limits, not only compressed upload size.
- Make CORS origins configurable; do not ship production defaults with unrestricted `allow_origins=["*"]`.
- Define authentication/authorization expectations for mutation endpoints or clearly mark the server as trusted-local only.
- Add tests for malicious ZIP entries and oversize expansion.

**Exit criteria:**
- Upload path traversal and zip-bomb regressions are tested.
- Production deployment docs include CORS/auth guidance.

### P0.5 Service-wide state isolation under concurrent traffic

**Gap:** `query_snapshot` now serializes artifact load+query, but other global-state mutations can still race with query and load flows on the singleton service instance.

**Required work:**
- Audit every endpoint that mutates vector store, retriever, graph builder, cache, repository state, or loaded artifacts.
- Add a service-level read/write lock or move toward request-local artifact handles for query paths.
- Add concurrency tests for query vs load, query vs index, query vs delete, upload vs query, and refresh/unload vs query.
- Decide whether API/web singleton mode is supported for production or only for trusted local use.

**Exit criteria:**
- Concurrent mutation/query behavior is either serialized by design or explicitly rejected with clear errors.
- No query can observe half-loaded artifacts or deleted repository state.

### P0.6 Release gate and compatibility policy

**Gap:** Smoke tests are green, but stable release gates need a documented contract beyond "current tests pass."

**Required work:**
- Define supported OS/Python/backend matrix.
- Define semantic versioning policy and compatibility promises for snapshots, manifests, and projection artifacts.
- Add release checklist: tests, packaging, install smoke, migration smoke, docs update, changelog.
- Add a minimal benchmark/performance envelope for medium repositories to catch major regressions.

**Exit criteria:**
- A maintainer can tag a release by following one checklist with reproducible commands and expected outputs.

### P1: Strongly Recommended Before Stable Release

These can be deferred only if the release notes explicitly scope them out.

### P1.1 Deepen semantic fidelity in core helpers

- JavaScript / TypeScript:
  - export alias facts
  - stronger type facts
  - richer interface / implementation facts
- Go:
  - stronger interface implementation facts
  - richer type facts beyond embedded relationships
- Java:
  - stronger type/member binding facts beyond regex/source heuristics

### P1.2 Replace narrow helper heuristics in secondary languages

- Rust
- C#
- Zig
- Fortran
- Julia

These currently emit useful structured facts, but they are still narrower than frontend-native semantic APIs.

### P1.3 Tighten query-time semantic escalation

- Current hook is intentionally conservative and snapshot-scoped.
- Remaining work:
  - use better terminal selection than top retrieved file paths
  - support induced-subgraph upgrades for path queries
  - add ranking tests proving graph-expanded results improve answer context, not only appear in retrieved rows

### P1.4 Continue store-boundary typing

Typed records now exist for manifests, snapshot refs, and snapshot records.

Still raw-dict-heavy at boundaries:
- API-facing manifest / artifact payload adapters
- index run records
- SCIP artifact record payloads
- projection/cache session payloads

### P1.5 Observability and debuggability

- Promote pipeline layer metrics and resolver diagnostics into stable API fields.
- Add structured logs for artifact swaps, semantic escalation, resolver fallback, and lock acquisition/release.
- Add a documented diagnostic bundle for support: config summary, storage backend, dependency availability, latest run metadata.

### P2: Post-Stable Quality Improvements

- Improve UI polish and workflow around snapshot selection and failure diagnostics.
- Add larger benchmark corpora and performance regression dashboards.
- Expand projection quality tests for architecture summaries and L0/L1/L2 stability.
- Add more real-world repository fixtures for polyglot monorepos.

## Closed Audit Items

### Query-time semantic escalation changing IR graph expansion behavior end to end

**Status:** Test guards added.

**Audit verdict:** PASS after hardening.

**Problem fixed:** Earlier tests verified mechanism (callback called, retrieve count) but not behavior (retrieval results actually change after IR graph expansion).

**Test coverage added** (`fastcode/tests/test_query_handler.py`):
- `test_semantic_escalation_changes_retrieval_results_end_to_end` — verifies answer generator receives expanded second-retrieval results (3 files vs 1)
- `test_semantic_escalation_enables_real_ir_graph_expansion` — uses real `HybridRetriever._expand_with_graph()` and an installed IR call graph to prove escalation changes retrieval behavior
- `test_local_budget_triggers_escalation_for_find_intent` — budget="local" path with find intent
- `test_escalate_query_semantics_returns_skipped_when_snapshot_not_found` — early return when snapshot missing
- `test_escalate_query_semantics_returns_skipped_when_no_target_paths` — early return when no paths extractable
- `test_escalate_query_semantics_returns_degraded_when_warnings_present` — degraded status with resolver warnings
- `test_escalation_does_not_rerun_when_callback_returns_none` — no rerun when callback returns None

### Experimental SCIP warning propagation in full index flow

**Status:** Already covered by existing test.

**Audit verdict:** CONCERNS/borderline PASS (risk score 2/9 — P1×I2)

**Existing test coverage:**
- `test_pipeline_layer2_records_experimental_scip_languages_non_silently` (`test_snapshot_pipeline.py:582`) — comprehensive single-language chain: detection → warning string → `layer2["warnings"]` → `result["warnings"]` + metrics
- `test_experimental_scip_profiles_are_marked_explicitly` (`test_scip_indexers.py:93`) — profile classification guard
- `test_experimental_scip_languages_set` (`test_semantic_resolvers.py:2156`) — canonical set guard

**Minor gaps (low priority):**
- Multi-language warning string (sorted join for Zig+Fortran) untested
- Warning persistence to snapshot store metadata untested

### Helper asset execution from packaged installs

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
