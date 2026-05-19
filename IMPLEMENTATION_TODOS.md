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
- release-grade evidence and compatibility policy for the now-implemented
  incremental, path-sharded artifact flow under real repositories, backend
  services, and upgrade scenarios
- first-class embedding/model fingerprint reuse across snapshots, vector
  artifacts, repository overviews, and query embeddings; active PG retrieval
  rows now carry embedding refs/fingerprints
- strict FP/FCIS dataflow enforcement across API -> schema -> core -> store boundaries
- completion of typed store/query/projection records so persistence and API shells stop leaking raw dict payloads
- package/install reproducibility for the active interpreter is now covered by
  a built-artifact release gate; service extras are split from the core path,
  but multi-Python matrix coverage and dependency drift policy remain open
- real external-tool integration evidence across supported languages
- real PostgreSQL/backend semantics beyond local fakes
- production deployment/auth runbooks for API/file-upload use beyond the current trusted-local/proxy-auth contract
- request-local artifact serving and query read sharing now exist, with
  concurrency benchmarks; stable release still needs endpoint/operator evidence
  under realistic mutation and streaming workloads
- release gate documentation and a first compatibility policy now exist;
  backend/tool gate automation, dependency drift review, supported-version
  matrix, and operator runbooks remain open

Stable release should mean all of the following are true at once:
- core indexing/query paths are incrementally efficient, not only correct
- cache reuse is deterministic and invalidates on schema/model/tool changes
- package/import boundaries preserve a thin shell around a test-heavy functional core
- storage backends, migrations, and artifact compatibility are tested against real upgrade scenarios
- install, deploy, operate, and recover workflows are documented and reproducible
- release claims about language support, backend support, and compatibility are narrower than the proven evidence, not broader

## Source-level audit - May 10, 2026

This audit checked current source and regression tests directly, not git history.

**Verdict:** the tracker should say "hardened pre-release with several partial passes," not "stable-ready." Several earlier TODO entries were stale in both directions: some work marked future now has a v0 implementation, while some work marked closed only satisfies a trusted-local or unit-tested contract rather than the full non-functional release goal.

**Evidence anchors checked:**
- architecture gates: `fastcode/tests/architecture/`
- incremental planning/materialization: `fastcode/src/fastcode/indexing/pipeline.py`,
  `fastcode/src/fastcode/scip/ast_adapter.py`
- embedding cache: `fastcode/src/fastcode/indexing/embedder.py`
- snapshot query handles and service locking:
  `fastcode/src/fastcode/indexing/pipeline.py`,
  `fastcode/src/fastcode/query/handler.py`,
  `fastcode/src/fastcode/main/fastcode.py`
- API upload/CORS/offload hardening: `fastcode/src/fastcode/utils/archive.py`,
  `fastcode/src/fastcode/api/cors.py`, `fastcode/src/fastcode/api/routes.py`
- agent turn/context v0: `fastcode/src/fastcode/retrieval/core/agent_context.py`,
  `fastcode/src/fastcode/retrieval/core/context_compiler.py`,
  `fastcode/src/fastcode/store/records.py`, `fastcode/src/fastcode/store/cache.py`

**Really implemented:**
- Architecture gates are active and green for import layers, no Pydantic in core,
  package-root purity, explicit translation, settings flow, and materialization
  boundaries. Verified with `uv run pytest fastcode/tests/architecture -q`
  (`16 passed`).
- Incremental indexing has moved beyond the May 4 audit: `build_ir_from_ast()` now persists file-level `content_hash`/`blob_oid` metadata, incremental planning rebuilds AST IR from changed elements, changed-unit embeddings can be reused when `embedding_text_hash` matches, and package/path repair-frontier logic can run scoped SCIP/semantic refresh.
- Embedding cache reuse is active for repeated texts and writes `float32` buffer payloads. Focused cache/materialization tests pass.
- `LoadedSnapshotArtifacts` plus an artifact-key LRU exist in `IndexPipeline`, and `QueryPipeline.query_snapshot()` can use request-local retriever/graph handles without its own snapshot-query lock.
- API upload/CORS hardening is closed for the current trusted-local/proxy-auth contract: active ZIP paths use safe extraction, defaults are local CORS origins and localhost binding, and blocking endpoint work is offloaded with `asyncio.to_thread`.
- A turn-journal/context-compiler v0 exists: typed `EvidenceRef`,
  `ToolObservation`, `RiskState`, `AcceptanceContract`, `TurnIntent`,
  `TurnPlan`, `WorkingMemoryArtifact`, `TurnJournal`, `ContextBundle`,
  `DistillationRecord`, `ActivationRecord`, and `HandoffArtifact`; typed cache
  records; bundle invalidation fingerprints; source-ref-preserving distillation
  reuse; and REST/web/MCP facades for turn context, bundle render/ref expansion,
  activation, and handoff. Verified with focused context bundle/cache/query/API
  tests plus touched-file ruff and pyright checks.

**Still only partially implemented against the non-functional goals:**
- Incremental update is not file-shard-native end to end. The pipeline still
  rehydrates a full combined element list and builds temporary whole-snapshot
  artifacts, but persisted vector/BM25 shards, conservative legacy graph
  shards, and safe PostgreSQL relational fact publication now reuse prior
  snapshot work for unchanged paths. Incremental plans now also expose explicit
  per-file interface digests, dependency-frontier metadata, and deterministic
  degraded-mode reasons. Scoped SCIP still widens or falls back to repo/tool
  scope in important cases.
- Embedding identity is closer to a uniform serving contract: snapshot/vector
  artifacts persist fingerprints, repository overviews carry fingerprints,
  `CodeEmbedder` now implements an explicit `EmbeddingService` boundary, and
  query-time semantic search passes the active embedding fingerprint into local
  vector search, repository-overview vector search, and PostgreSQL semantic
  search so same-dimension stale vectors are filtered instead of silently
  reused. The remaining gap is extending backend/release evidence beyond the
  active retrieval and indexing surfaces.
- Snapshot artifact handles are closer to serving isolation: public
  `FastCode.query()`, `FastCode.query_snapshot()`, and `FastCode.query_stream()`
  now share the read side of the service state lock while mutations keep the
  write side. Snapshot-scoped streams now capture immutable artifact handles
  before releasing the service read lock, and the `/query` API route has
  endpoint-level concurrency coverage proving independent snapshot-scoped reads
  can overlap after offloading. Endpoint/operator benchmark evidence is still
  open.
- FP/FCIS dataflow is much better but not complete. Store hot paths have many typed-record adapters and regression guards, while query/retrieval/API compatibility surfaces still pass raw dict-shaped payloads in several places.

**Still open:**
- multi-Python wheel/sdist install smoke from built artifacts
- real external SCIP/tool command matrix for stable language claims
- real PostgreSQL migration/locking/outbox/redo/fact semantics under integration load
- backend/tool/performance release gates, full artifact migration policy,
  dependency/supply-chain review, and deployment/operator runbooks

## Source-level audit - May 18, 2026

This pass reconciles the implementation tracker after completing
[PERFORMANCE_TODOS.md](./PERFORMANCE_TODOS.md).

**Verdict:** the performance checklist is closed, but this tracker still has
six legitimate stable-release blockers in packaging, dependency, release-matrix,
and documentation areas. They should remain unchecked until the repo has direct
release evidence, not just implementation code.

**Performance/incremental status now updated:**
- Incremental artifact publication now has concrete vector, BM25, legacy graph,
  snapshot shard, relational fact, and IR graph delta paths, with fallback
  reasons recorded in metrics.
- Snapshot persistence is manifest/shard based for units, supports, relations,
  embeddings, and embedding vectors; lazy readers cover metadata, path/unit
  subsets, relations, supports, and embeddings.
- Query-time lexical retrieval can load shard-native BM25 metadata and score
  only needed shards/postings instead of rebuilding a full `BM25Okapi` object
  on the primary sharded path.
- Projection building uses a projection-native graph representation and
  `igraph` for hot algorithms; NetworkX is retained for compatibility/export
  surfaces.
- MCP graph tools route compact paths and cluster projection rebuilds through
  saved graph handles and sidecar symbol maps when available, with explicit
  compatibility fallback metadata.
- Helper-backed semantic resolver runs now have artifact-native cache keys and
  cache hit/miss metadata under `.fastcode/semantic_helper_cache`.
- Benchmarks now cover edit classes, concurrent query/mutation scenarios,
  materialization surfaces, and graph-engine decisions.

**Open after this audit:**
- supported OS/Python/backend matrix and multi-Python release-gate execution
- dependency drift/security review and high-risk runtime constraints
- optional extras/dependency release review
- a single tag checklist with expected outputs across every release gate
- release-grade install/deployment/operator runbooks

## Source-level audit - May 11, 2026

This pass checked current source against the non-functional goals and excludes
the agent-integration design track. New performance-specific items are tracked
in [PERFORMANCE_TODOS.md](./PERFORMANCE_TODOS.md); this section records the
release-level implications.

**Newly recorded non-functional gaps:**
- The materialization enforcement story is incomplete. The guard test covers
  selected hot files, semantic patching, and the main graph helper compact
  traversal regression. MCP graph tools now have focused compact-path
  regressions, but projection transforms, snapshot persistence, and query-time
  compact symbol-index registration can still add materialization without
  failing an architecture-level guard.
- Embedder laziness is no longer overclaimed for compatibility fingerprinting
  and all-cache-hit validation: those paths avoid provider startup when
  dimension is not configured. Ollama provider-native batching is landed with
  bounded per-text fallback; provider timing separation and benchmark evidence
  remain open performance work.
- Semantic resolver patching remains copy-heavy: applying a resolver patch
  still copies whole snapshot collections, but the old generic
  `to_dict() -> from_dict()` clone path and generic JSON cleanup have been
  replaced with explicit field copies and patch-local serializers.
- Shell graph tools are partially aligned with compact graph design. MCP
  directed path, impact, caller, and Steiner tools now use saved IR graph
  handles plus sidecar symbol maps when available; legacy snapshots, missing
  compact artifacts, and projection rebuild fallback still load full snapshots
  and may rebuild NetworkX graphs.
- Query-time symbol-index registration now has a compact snapshot sidecar and
  avoids full `IRSnapshot` loads for current snapshots. Single-symbol
  `find_symbol()` lookups can also return sidecar symbol records without
  scanning a full snapshot. Legacy snapshots without the sidecar still need a
  backfill or relational-fact read path.
- Local repository indexing now defaults to read-only in-place loading for
  local paths and only copies a whole working tree when explicit
  workspace-copy mode is requested. Explicit `local_source_mode: "hardlink"`
  can isolate the workspace without duplicating file bytes on compatible
  filesystems. The remaining copy-minimization gap is content-addressed reuse
  for byte-copy fallback and explicit copy mode.
- PostgreSQL relational fact persistence now has a safe changed-path delta for
  incremental plans with a prior snapshot and no semantic frontier widening:
  unchanged rows are copied from the previous snapshot and changed-path rows are
  upserted. Full rebuild remains for widened/repair flows, missing previous
  snapshots, and non-Postgres paths, and real backend load evidence is still
  required.

**Release implication:** the project remains a hardened pre-release. The code
has many real partial passes, but stable language around "incremental",
"zero-copy", "native graph", and "query serving concurrency" must stay scoped
until the new TODOs have implementation, enforcement, and benchmark evidence.

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
  - `store.records` now also defines typed `SCIPArtifactRecord`, `RedoTaskRecord`, and `OutboxEventRecord` boundaries for active snapshot-store queue/artifact flows
  - `store.records` now also defines typed `ProjectionBuildRecord` and `ProjectionDirtyScopeRecord` boundaries for active projection-store lookup/dirty-scope flows
- Snapshot query concurrency safety:
  - `query_snapshot` serializes load+query with a `threading.Lock`
  - Thread-barrier regression tests (2 and 3 concurrent callers) verify artifact isolation
- Query/agency hardening:
  - agency-mode `query()` and `query_stream()` preserve cheap detected intent
  - iterative-agent standard retrieval applies caller filters after rerank
  - regression test proves semantic escalation can enable real IR graph expansion, not just a mocked second retrieval
- Incremental prefilter hardening:
  - `IndexPipeline._plan_incremental_elements()` reuses unchanged element metadata from the previous artifact manifest
  - compatible snapshots call `CodeIndexer.index_files()` only for added/modified files instead of falling back to full `extract_elements()`
  - integration coverage proves unchanged files bypass full extraction when compatibility checks match
  - a pipeline run now shares one loader inventory across snapshot identity, manifest diffing, and full AST extraction
  - `build_ir_from_ast()` now persists file-level `content_hash`/`blob_oid` metadata on file units
  - incremental plans now build AST IR from changed elements and merge with the previous snapshot rather than rebuilding AST IR from unchanged files
  - changed-unit embeddings can be reused when stable unit identity and `embedding_text_hash` match
  - package/path repair-frontier logic can scope semantic refresh and SCIP reruns when API or edge surfaces change
  - manifest-first incremental planning now refuses reuse when prior or current file entries lack content fingerprints instead of trusting size/mtime fallback
  - persisted vector and BM25 path shards now reuse compatible previous snapshot artifact shards for unchanged paths under the new artifact key
  - legacy graph path shards reuse compatible previous snapshot artifact shards only for conservative implementation-local changes, avoiding reuse when removals or edge/API/signature changes could stale graph edges
  - incremental result metadata now reports `artifact_shard_reuse` counts for vector/BM25/graph publication
- Embedding cache hardening:
  - `CodeEmbedder` now uses cache-aware batch embedding in the active indexing path
  - identical embedding texts are deduplicated before provider calls
  - cache keys are model-aware across provider/model/dimension/normalization/sequence-length settings
  - active cache hits and changed-unit incremental embedding reuse now validate
    both prepared-text hash and embedding fingerprint before reusing vectors
  - embedding provider metrics now expose startup, request, batch, cache-hit,
    cache-miss, and cache-write counters, and Ollama per-text requests support
    bounded concurrency
  - embedding cache payloads now store `float32` byte buffers plus explicit shape/dtype metadata instead of Python `list[float]` vectors
  - Ollama embedding now uses the provider-level `/api/embed` batch endpoint
    when available, with bounded per-text fallback for older servers; a real
    `all-minilm:l6-v2` smoke returned three 384-dimensional embeddings with one
    provider request
  - `store/cache.py` now persists hot dialogue/session/query payloads as explicit JSON envelopes and embedding entries as buffer-aware cache envelopes instead of generic object storage on the active cache paths
  - PostgreSQL retrieval persistence now strips raw embedding arrays from JSON metadata and stores vectors only in vector-specific columns
  - PostgreSQL retrieval upserts now batch vector/search-document writes and
    expose row count, batch count, and vector adapter path metrics
  - PostgreSQL retrieval metadata now carries embedding artifact refs and
    fingerprints for code elements and documentation chunks while rejecting
    embedding/vector-shaped numeric metadata payloads, so raw embeddings stay
    confined to vector columns/artifacts
  - Real PostgreSQL/Ollama evidence on May 14, 2026:
    `postgresql://postgres:postgres@127.0.0.1:5432` plus
    `http://10.0.0.203:11434` indexed a docs-enabled temp repo with
    `all-minilm:l6-v2`, queried through `gpt-oss:20b-cloud`, and produced 10
    PG vector rows with 384-dimensional pgvector values, zero legacy
    `embedding_arr` rows, no JSON embedding leaks, and no missing
    refs/fingerprints. The run degraded only because Terminus was intentionally
    unconfigured.
  - `store/pg_retrieval.py` now builds JSON row payloads from an explicit stable element field set instead of recursively normalizing arbitrary whole-element dicts during hot inserts
- Publishing boundary hardening:
  - Terminus lineage publishing now has a typed `IRSnapshot` boundary that builds payloads from snapshot fields and IR units/relations without full `IRSnapshot.to_dict()` expansion
  - `PublishingService` and the active index pipeline call the typed lineage publisher; the dict-based publisher API remains as a compatibility shim
- API shell serialization hardening:
  - REST and web query/session endpoints now use explicit source and dialogue serializers from `api/serialization.py` instead of generic recursive `safe_jsonable()` conversion
- Manifest/index-run boundary hardening:
  - `store/manifest.py` now reconstructs and serializes manifest payloads through explicit field mappings instead of `row_to_dict() -> ManifestRecord.from_dict()/to_dict()` round-trips on hot paths
  - `store/index_run.py` now materializes run/publish-task payloads through explicit row field extraction, and claimed publish tasks return the post-claim `running`/incremented-attempt state
  - `store/index_run.py` now also exposes typed `IndexRunRecord` / `PublishTaskRecord` accessors, and active publishing/indexing callers prefer those record APIs instead of dict payloads
- Unit-artifact boundary hardening:
  - `store/unit_artifacts.py` now reconstructs listed unit payloads through explicit row field extraction instead of `row_to_dict()`, and metadata JSON serialization is bounded to the metadata subtree with opaque values falling back to stable `repr(...)` strings
- Snapshot-record boundary hardening:
  - `store/snapshot.py` now reconstructs snapshot and snapshot-ref records through explicit row field extraction and uses explicit compatibility serializers for `find_by_*` / `resolve_snapshot_for_ref` dict-return helpers instead of `row_to_dict() -> from_dict()/to_dict()` round-trips on those active paths
  - `store/snapshot.py` now owns explicit snapshot-file serializers/deserializers for `save_snapshot()` / `load_snapshot()` instead of delegating the persistence boundary to `IRSnapshot.to_dict()` / `IRSnapshot.from_dict()`
  - active SCIP artifact read/write paths now use explicit typed artifact records instead of `SCIPArtifactRef.to_dict()` and `row_to_dict()` round-trips, and lineage metadata JSON serialization is bounded to the metadata subtree with opaque values falling back to stable `repr(...)` strings
  - redo-task and publish-outbox claim/failure paths now use explicit typed row extraction, and claimed payloads return the post-claim `running` / `in_progress` state with current timestamps and counters
- Projection boundary hardening:
  - `store/projection.py` now reconstructs dirty-scope and build payloads through explicit typed row adapters and explicit compatibility serializers instead of tuple-shaped row unpacking directly into ad hoc dict payloads
  - `indexing/projection.py` now hashes scope parameters and materializes built projection payloads through explicit field serializers instead of `ProjectionScope.to_dict()` / `ProjectionBuildResult.to_dict()`
- Cache boundary hardening:
  - `store/cache.py` now persists dialogue turns and session indexes through typed `DialogueTurnRecord` / `DialogueSessionRecord` payload serializers and exposes typed record accessors alongside compatibility dict-return helpers
  - active query/session callers now prefer typed cache session records where they only need turn counts or `multi_turn` flags
- Indexer boundary hardening:
  - `indexing/indexer.py` now serializes import payloads explicitly and reuses the serialized import list for file hash/metadata derivation instead of repeated `ImportInfo.to_dict()` expansion
  - active extract/incremental embedding handoff now builds a minimal explicit element payload for the embedder instead of deep-copying full `CodeElement.to_dict()` payloads on the hot path
- Index pipeline boundary hardening:
  - `indexing/pipeline.py` now materializes vector-store and PG-upsert element payloads through explicit serializers on the active snapshot indexing path instead of `CodeElement.to_dict()` expansion plus in-place metadata mutation
- Retrieval boundary hardening:
  - `ir/element.py` now exposes explicit `serialize_code_element(...)` and `deserialize_code_element(...)` adapters so retrieval shells can materialize and rehydrate stable element payloads without recursive dataclass expansion or `CodeElement(**payload)` mass-assignment
  - `retrieval/hybrid.py` and `retrieval/core/fusion.py` now build keyword/file/type/graph-expansion/doc-projection result payloads and BM25 persistence payloads through explicit `CodeElement` serialization instead of repeated `CodeElement.to_dict()` calls on active paths
  - `retrieval/iterative.py` now serializes agent-selected class/function hits and file-level indexed lookups through explicit element payloads instead of `CodeElement.to_dict()` expansion on active agency-mode selection paths
  - `retrieval/hybrid.py` and `main/fastcode.py` now reload persisted BM25 element payloads through the explicit deserializer instead of `CodeElement(**payload)` expansion on active cache/load paths
- Legacy graph boundary hardening:
  - `graph/build.py` now persists compatibility graph element indices through explicit element serializers instead of `CodeElement.to_dict()` calls
  - `graph/build.py` now reloads and merges persisted graph element payloads through the explicit element deserializer instead of `CodeElement(**payload)` expansion, while preserving the existing duplicate-name recovery behavior via `element_by_id`
- Object materialization hardening:
  - PostgreSQL semantic fallback now keeps candidate vectors in NumPy arrays and only JSON-decodes metadata for ranked results returned across the Python boundary
  - repository overview persistence now writes an explicit JSON manifest plus NumPy embedding archive, lets metadata-only callers skip the embedding archive entirely, and only JSON-decodes ranked overview metadata for returned rows
  - `utils/vectors.py` now centralizes float32 vector/matrix ownership policy so view-oriented call sites do not mutate caller-owned arrays while sanitizing non-finite values
  - PostgreSQL relational fact persistence uses explicit typed payload serializers instead of repeated record `to_dict()` expansion
  - snapshot persistence writes a compact symbol-index sidecar, and query-time
    snapshot serving can register symbol aliases or return single-symbol
    records from that sidecar instead of full-loading the `IRSnapshot`
- Async endpoint hardening:
  - REST API repository load/index/cache-load/multi-index/upload paths offload blocking work with `asyncio.to_thread`
  - web UI upload and upload+index paths offload ZIP extraction, repository load, and indexing
  - REST API and web UI delete/cache/refresh maintenance endpoints offload blocking mutations
- API upload/CORS hardening:
  - REST and web ZIP upload paths now validate archive members before extraction, rejecting traversal, absolute paths, symlinks/special files, oversized members, excessive file counts, and suspicious expansion ratios
  - repository ZIP loading in `RepositoryLoader` uses the same safe extraction helper instead of raw `ZipFile.extractall()`
  - CORS defaults are no longer wildcard; allowed origins and credentials are controlled by explicit environment settings
  - REST API and web examples now bind localhost by default, with public binding documented as an explicit trusted-operator decision
- Service singleton lock hardening:
  - `FastCode` now exposes a service-level reentrant read/write state lock: repository load/index, snapshot pipeline, projection build, multi-repo cache load, delete, cleanup, and shutdown flows use the write side, while query/query-stream use the read side
  - REST and web load+index/upload+index helpers now execute the combined mutation under one service critical section, preventing interleaving between repository replacement and indexing
  - Regression tests cover query serialization against load, index, delete, refresh, and cleanup/unload-style mutations, plus concurrent query read sharing
- Snapshot artifact-handle hardening:
  - `IndexPipeline` now exposes `LoadedSnapshotArtifacts` handles and an artifact-key LRU cache
  - `QueryPipeline.query_snapshot()` can use request-local retriever/graph handles instead of mutating the singleton retriever state
  - regression coverage proves the handle path can isolate concurrent snapshot queries without `QueryPipeline._snapshot_query_lock`; public `FastCode` still serializes query entrypoints with the service lock, so this is not yet a read-scalability closure
- Agent-context v0 hardening:
  - pure typed turn/context records now live in `retrieval/core/agent_context.py`
  - `retrieval/core/context_compiler.py` renders stable/turn/observation FCX working-memory artifacts and typed handoff artifacts
  - `store.records` and `store/cache.py` persist typed working-memory, turn-journal, and handoff records through explicit payload serializers
  - REST, web, MCP, and `FastCode` facades expose latest/specific turn context, evidence-ref expansion, and handoff creation/fetch
- Lock fencing hardening:
  - PostgreSQL same-owner lock refresh preserves the current fencing token instead of invalidating in-flight work
- Package-root import-boundary hardening:
  - `fastcode/__init__.py` no longer mutates process environment at import time
  - `fastcode/__init__.py` keeps compatibility re-exports lazy instead of importing shell-heavy modules eagerly
  - `fastcode.main.__init__` now exposes `FastCode` lazily so `import fastcode.main` does not load the composition root
  - internal API/MCP modules import `FastCode` from `fastcode.main.fastcode` instead of the root compatibility surface
  - architecture tests now enforce thin package roots and ban internal `from fastcode import ...` re-export usage
- Release package/install gate hardening:
  - `scripts/release_gate.py` builds all workspace wheel/sdist artifacts with `uv build --all-packages --clear`
  - the gate verifies required semantic helper assets are present in FastCode wheel and sdist artifacts
  - the gate installs built sdists and wheels into fresh virtualenvs with `pip`, then smokes installed imports and console entrypoints
  - the wheel smoke indexes and queries a tiny repository using fake Ollama embedding and OpenAI chat-completion endpoints, proving configured installed runtime paths instead of editable-source imports
  - BM25 indexing now treats empty element and repository-overview corpora as a no-op instead of constructing invalid `rank_bm25` indexes, and it clears stale repository-overview state before rebuilding

## Critical verification currently green

- Workspace-root smoke gate:
  - `1721 passed, 53 skipped`
- Package-root smoke gate:
  - `1694 passed, 13 skipped`
- Package/install release gate:
  - `python scripts/release_gate.py` passed on May 13, 2026 with Python 3.13.13
- Focused regression areas repeatedly verified:
  - `test_api`
  - `test_manifest_store`
  - `test_query_handler`
  - `test_scip_indexers`
  - `test_semantic_resolvers`
  - `test_snapshot_pipeline`
  - `test_snapshot_store`

## Incremental Update and Cache Audit

**Audit date:** May 4, 2026; source-level status refreshed May 10, 2026

**Verdict:** partial PASS. FastCode now has real early reuse for unchanged files, but it does not yet implement graceful end-to-end incremental update for compiler/tool-backed stages.

**What is already true in code:**
- `fastcode.indexing.pipeline._plan_incremental_elements()` reuses unchanged element metadata from the previous artifact manifest and only reindexes added/modified files.
- `fastcode.tests.integration.test_snapshot_pipeline.test_pipeline_incremental_prefilter_only_indexes_changed_files` proves unchanged files bypass `extract_elements()` and only changed files are passed to `index_files()`.
- `run_index_pipeline()` now shares a precomputed loader inventory through snapshot identity, incremental planning, and full AST extraction.
- `fastcode.indexing.embedder.CodeEmbedder` uses cache-aware `embed_code_elements()`, deduplicates identical texts, and keys cache entries by provider/model/dimension/normalization/max-sequence-length identity.
- `build_ir_from_ast()` persists file-level `content_hash` / `blob_oid` metadata on file units.
- incremental plans can build AST IR from changed elements and merge it with the previous snapshot instead of rebuilding AST IR from unchanged files.
- changed-unit embeddings can be reused when stable unit identity and `embedding_text_hash` match.
- helper-backed semantic resolvers already scope helper input to changed `target_paths`, so helper work is narrower than full-repo in the happy path.
- package/path repair-frontier logic can scope semantic refresh and SCIP reruns when changed API or edge surfaces are known.
- manifest-first incremental planning now disables reuse when prior or current file entries lack required fingerprints.

**What is not yet true:**
- the pipeline still rehydrates a full combined `elements` list and builds temporary downstream vector/BM25/legacy graph objects from that set, but persisted path shards now avoid rewriting unchanged vector/BM25 and conservative graph artifacts.
- optional SCIP now has scoped paths, but language detection and unsupported/widened cases can still pay near full compiler/indexer cost.
- file-unit `content_hash` / `blob_oid` metadata exists, but file identity is not yet a single canonical planner anchor shared by inventory, incremental planning, SCIP, file-artifact reuse, and downstream publication.
- helper-backed runtimes still pay per-run process startup cost; the Go helper currently uses `go run`, which also implies repeated compile/build overhead.

**Release implication:** current behavior materially reduces repeated parse and embedding work for unchanged files, but stable-release claims must still say compiler/tooling cost is only partially incremental.

## Architecture contract in force

The current release bar is aligned to the repo-template functional-core guidance:

1. Pydantic stops at the boundary:
   - `fastcode.schemas.api` owns request/response validation.
   - core packages (`ir`, `graph`, `retrieval`) stay Pydantic-free.
2. Persistence trusts typed records:
   - store/infrastructure code should return frozen dataclasses or typed records, not ad hoc dict payloads.
3. Translation stays explicit:
   - shell code maps field-by-field between API payloads, core dataclasses, and storage records.
   - no `**model_dump()`, `**__dict__`, or hidden mass-assignment at boundaries.
4. FCIS test split:
   - functional core gets the majority of regression effort.
   - imperative shell remains thin and covered mostly by boundary/integration tests.
5. Package roots stay thin:
   - Python always executes `fastcode/__init__.py` before importing any `fastcode.*` submodule.
   - root and subpackage `__init__.py` files must not perform runtime setup, environment mutation, or eager shell imports.
   - root re-exports are compatibility-only and lazy; internal modules import concrete domain/composition modules directly.

This contract is only partially complete today. The architecture tests now enforce parts of it, including package-root purity, but large store and shell surfaces are still dict-heavy.

## Next Research Track: Agent Context Integration

**Problem statement:** FastCode should not stop at "API plus CLI/MCP tools for code
search." The next design task is to make FastCode useful as an agent context
engineering substrate: cacheable, budget-aware, provenance-preserving code
evidence that agents can retrieve, compress, reactivate, and cite without
re-reading or re-summarizing the same repository facts.

Detailed design reference:
[AGENT_INTEGRATION_PATTERNS.md](./AGENT_INTEGRATION_PATTERNS.md)

**Why this is now the right research target:** The original March design notes
framed FastCode as the code-intelligence provider beside a dynamic memory/core
agent system. Most of the needed FastCode-side foundations now exist:
- canonical IR/SCIP merge and semantic fallback
- snapshot and manifest lineage
- L0/L1/L2 projection generation
- typed cache/session records
- MCP graph tools for paths, impact, callers, and clusters
- query-time semantic escalation
- incremental/cache hardening that can eventually make context refresh cheap

**Reference pattern:** [OpenCode DCP](https://github.com/Opencode-DCP/opencode-dynamic-context-pruning)
is a useful example of session-local context pruning: it injects compression
guidance, lets the model choose compression ranges/messages, protects important
tool outputs, deduplicates repeated tool calls, and replaces stale messages with
summary placeholders instead of modifying the original session history. FastCode
should borrow the *context lifecycle idea*, not its exact implementation. DCP is
AGPL-licensed, so any direct code reuse needs explicit license review. FastCode
can do more because it owns structured repository facts, symbol/edge provenance,
embeddings, projections, and snapshot lineage.

**Core design thesis:** Keep authoritative code facts in FastCode and expose
agent-facing context as derived artifacts:
- `EvidenceRef`: immutable pointer to code facts, projection chunks, tool
  traces, or prior session turns
- `ContextBundle`: ranked, token-budgeted set of evidence refs with text
  renderings, summaries, citations, freshness, and invalidation metadata
- `DistillationRecord`: lossy summary of one or more evidence refs, with
  coverage, omitted refs, source snapshot, model/prompt fingerprint, and
  token-cost metadata
- `ActivationRecord`: record of which bundle/distillation an agent actually
  consumed for a task, so future retrieval can reuse proven context rather than
  recomputing from raw snippets

**Two-surface memory model:** The context system should explicitly separate:
- historical truth:
  append-only, complete, external, and restorable
- cognitive working memory:
  small, recent, relevant, and cache-friendly

The rewrite target is the second surface. FastCode should preserve the
historical evidence/observation journal and continuously recompile the
prompt-facing working-memory view from typed state.

**What FastCode should do beyond DCP:**
- code-aware compression:
  - summarize by symbols, files, call paths, dependency frontiers, clusters, and
    changed shards instead of only by message ranges
  - never summarize away citations; every summary must remain expandable to
    source files/symbols/ranges/snapshots
- cacheable context bundles:
  - key bundle reuse by query/task fingerprint, snapshot/artifact key,
    projection algorithm version, embedding fingerprint, distillation prompt
    fingerprint, and token budget
  - invalidate bundles when source snapshot, projection method, model/prep
    schema, or cited evidence changes
- progressive disclosure:
  - L0 for orientation, L1 for navigation and relationships, L2 chunks for
    evidence, raw source only when needed
  - include explicit "expand handles" so agents can ask for the next level
    without a fresh broad search
- agent feedback loop:
  - record accepted/rejected evidence, files actually edited, tests run, and
    whether retrieved context was sufficient
  - use those activation records as ranking signals for future similar tasks
- protected evidence:
  - preserve high-value outputs such as subagent findings, architecture
    decisions, failing test logs, user constraints, and write-set summaries
  - prune or distill duplicate/error-heavy tool output after it is superseded
- budget policy:
  - distinguish prompt-cache-sensitive stable prefixes from late dynamic context
  - prefer stable IDs/placeholders plus expandable summaries over repeatedly
    injecting large raw snippets

**Current state:** v0 already defines `EvidenceRef` and typed
turn/working-memory/handoff records outside Pydantic-heavy shells. The
bundle/distillation/activation layer remains future work.

**Proposed staged roadmap:**
1. Define frozen core records for `ContextBundle`, `DistillationRecord`, and
   `ActivationRecord`, extending the existing `EvidenceRef` contract only where
   bundle expansion requires it.
2. Add explicit serializers and cache keys for those records in `store/cache.py`
   or a dedicated context store.
3. Add a read-only context-bundle builder that consumes current retrieval
   results and projection artifacts, enforces a token budget, and emits
   provenance-preserving L0/L1/L2 sections.
4. Expose bundle tools through MCP/API:
   - build bundle for task/query/snapshot
   - expand evidence ref
   - compress/distill old bundle
   - list/reactivate prior task bundles
   - record activation feedback
5. Add DCP-style pruning policies for FastCode session/tool history, but make
   code evidence protected by default and replace only the prompt-facing
   working-memory view with stable expandable refs; do not destructively rewrite
   the historical journal.
6. Add evaluation fixtures for agent tasks:
   - context tokens consumed
   - answer/edit correctness
   - citation coverage
   - repeated-query reuse
   - source-change invalidation correctness

**Research questions to resolve before implementation:**
- What is the minimal task fingerprint that reuses useful context without
  leaking stale or irrelevant evidence?
- Which context should be prefix-stable for provider prompt caching, and which
  should stay late/dynamic?
- How much lossy distillation is acceptable before code-edit accuracy drops?
- Should bundle ranking be learned from activation records or remain
  deterministic for v1?
- How should external agent memories such as MemOS link to FastCode bundles:
  copied summaries, refs only, or hybrid refs plus selected summaries?

**Acceptance bar for this track:**
- A repeated agent task can reuse an existing context bundle without rerunning
  broad retrieval or re-summarizing unchanged evidence.
- A source edit invalidates only bundles that cite changed/affected evidence.
- Agents can expand any summarized item back to file/symbol/range/snapshot
  evidence.
- Metrics report bundle cache hit/miss, distillation reuse, context token
  savings, citation coverage, and stale-evidence invalidations.
- Regression tests fail if a bundle summary loses all source refs or silently
  survives an incompatible snapshot/projection/model fingerprint change.

**v1 implementation slice, when this moves from research to code:**
- Keep it read-only with respect to repository facts:
  - no new indexing behavior
  - no mutation of source files
  - no external memory-store dependency
- Add pure context records and helpers in an inner package:
  - evidence normalization from retrieval rows and projection chunks
  - deterministic token-budget allocation
  - bundle rendering with stable expansion handles
  - explicit source-ref coverage checks
- Add a typed context cache/store boundary:
  - bundle metadata
  - distillation metadata
  - activation metadata
  - explicit serializers keyed by snapshot/artifact/projection/model
    fingerprints
- Add MCP/API shell tools only after the pure bundle builder has focused tests:
  - `build_context_bundle`
  - `expand_context_ref`
  - `list_context_bundles`
  - `record_context_activation`
- Add regressions before adding agent-facing defaults:
  - stale snapshot rejects cached bundle reuse
  - changed projection algorithm rejects cached text reuse
  - summary without source refs is rejected
  - bundle token budget is enforced deterministically
  - repeated identical request reuses cached distillation

**Layering guardrails:**
- `retrieval/core/` can own pure evidence scoring and budget logic.
- `query/` can orchestrate retrieval plus bundle assembly.
- `store/` can persist typed bundle/distillation/activation records.
- `api/` and `mcp/` should remain transport adapters.
- Do not introduce Pydantic, direct env reads, generic `safe_jsonable()` walks,
  or recursive `to_dict()` round-trips in the new inner bundle path.

**Turn-centric rewrite target:**
- Rewrite agent integration around typed turn artifacts instead of raw prompt
  carry-over plus saved summaries.
- Split turn memory into two layers:
  - append-only turn/evidence journal
  - replaceable compiled working-memory artifact
- Introduce a turn journal model:
  - `TurnIntent`
  - `TurnPlan`
  - `ToolObservation`
  - `BeliefState`
  - `WorkingSet`
  - `TurnOutcome`
- Treat each tool call as an observation update:
  - normalize the result
  - extract evidence refs
  - update active hypotheses
  - append the observation to the journal
  - rewrite the next prompt/context section
- Keep the stable prompt prefix byte-stable when snapshot/task invariants match:
  - `L0`
  - selected `L1`
  - accepted facts
  - protected constraints
- Keep the dynamic tail small and recent:
  - active hypotheses
  - unresolved questions
  - recent observations
  - next actions
- Keep human-readable tool output as a rendering layer, not the authoritative
  agent state.

**Concrete v1 rewrite path against current code:**
1. Keep `query/handler.py` as the outer turn orchestrator, but stop treating
   session history plus `get_recent_summaries()` as the main carry-forward
   state.
2. Refactor `retrieval/iterative.py` round state into explicit typed
   plan/observation/history records so confidence, keep-files, and tool-call
   history stop living mainly in prompt-local dicts.
3. Wrap `retrieval/agent_tools.py` outputs in normalized observation adapters
   with evidence refs, cost, warnings, and freshness metadata.
4. Extend `store/cache.py` or a sibling store with typed persistence for the
   turn journal and working-set artifacts, not only dialogue turns.
5. Add a context compiler that renders the next-turn prompt from:
   accepted facts, active hypotheses, unresolved uncertainty, protected
   evidence, and reusable bundle refs.
6. Add verifier-aware turn transitions:
   retrieval -> inspect -> verify -> commit
   retrieval -> inspect -> uncertainty persists -> branch or ask
   retrieval/tool failure -> degraded evidence -> abstain or widen search

**Acceptance criteria for the rewrite:**
- A tool result can update turn state without forcing the next model call to
  reread the full raw transcript.
- The next-turn context is reproducible from typed state plus bundle refs.
- Accepted and rejected hypotheses are explicit in the carried-forward state.
- Verification failures remove or downgrade conflicting hypotheses in the next
  prompt rewrite.
- Replaying the same turn against the same snapshot/artifact inputs yields the
  same working set and bundle handles, modulo model nondeterminism in optional
  summary text.

**Model-facing DSL:**
- Use the line-oriented FCX DSL defined in
  [AGENT_CONTEXT_DSL.md](./AGENT_CONTEXT_DSL.md) for compact prompt rendering.
- Treat FCX as a rendering/parsing layer over typed records, not as storage
  truth.
- Compare FCX token counts against pretty JSON, minified JSON, and prose before
  making it the default agent prompt format.

## Stable Release Gap

### P0: Must Close Before Stable Release

These are release blockers. Do not tag a stable release while any P0 item is open.

### P0 Acceptance Bar

Every P0 item needs all three forms of closure:
- implementation: the code path exists and is wired into the default runtime path
- enforcement: tests, architecture checks, or release-gate automation fail when it regresses
- evidence: docs, metrics, or benchmark output show the stable-release claim is true in practice

### P0.1 Install and packaging reproducibility

**Gap:** Tests run in the repository checkout, but stable users install packages, optional extras, and helper assets from built artifacts.

**Current status:** partially closed. `scripts/release_gate.py` now builds all
workspace wheel/sdist artifacts, verifies FastCode helper assets inside both
artifact types, installs built sdists and wheels into fresh virtualenvs with
`pip`, smokes installed imports and entrypoints, and runs an installed-wheel
index/query flow against fake Ollama/OpenAI-compatible endpoints. The gate also
tests service extras separately so API/MCP/PostgreSQL/Redis boundaries are
explicit without forcing the local embedding or Nanobot stacks into the default
install path. The latest checked run passed on May 13, 2026 with Python 3.13.13.

**Remaining stable-release gap:** the gate still runs only on the active host
interpreter, and the supported-version matrix is not yet encoded in automation.
Audit on May 18, 2026 found these release blockers still open; the current
package metadata declares Python `>=3.11`, but only the active Python 3.13 host
has recent gate evidence.

**Required work:**
- [x] Build and test both wheel and sdist from a clean checkout.
- [x] Run import/CLI/API smoke tests from installed artifacts, not from editable source.
- [x] Verify helper assets are included in built artifacts.
- Define supported Python versions and run the release gate against each supported version.
- [x] Split optional extras clearly: API server, PostgreSQL, SCIP/tooling, docs ingestion, dev/test.
- [x] Pin or constrain the local embedding stack out of the core runtime path.
- [ ] Pin or constrain remaining high-risk runtime dependencies enough to avoid resolver drift.

**Exit criteria:**
- [x] `pip install dist/*.whl` in a fresh virtualenv can run CLI, API import, helper path checks, and a tiny index/query smoke.
- [x] Release docs list exact install commands for local-only mode and document that production/PostgreSQL gate evidence is still open.
- [ ] Supported Python matrix gate passes for every supported version.
- [ ] Optional extras and dependency constraints are release-reviewed.

### P0.1a Workspace and layout stability

**Gap:** The package split is now mostly landed, but stable release also requires the merged workspace to behave correctly from the repository root, not only when commands are run from `fastcode/`.

The template philosophy remains correct for Python: use one importable package with descriptive domain modules instead of pretending uv workspace members provide native import isolation. The stable-release risk is not the nested domain layout itself; it is accidental boundary collapse through package roots, eager re-exports, and imports that route pure modules through shell-heavy compatibility surfaces.

**Current state:**
- root `pytest` collection now honors workspace-level `testpaths`, recursion excludes, import paths, and marker registration
- stale renamed-module references were cleaned up in active tests and comments
- root smoke no longer collects `repo_backup/` or dies on import-file mismatch

**Remaining work:**
- remove or clean stale `__pycache__` trees left behind by the pre-merge layout so they stop polluting diagnostics
- keep root and package pytest policy in sync when markers or plugins change
- audit public docs/examples for pre-merge module paths
- keep architecture tests green for package-root purity whenever new public exports or subpackage `__init__.py` files are added

**Exit criteria:**
- `uv run pytest` from repository root is a supported release command
- workspace-level QA commands no longer depend on implicit `cd fastcode`

### P0.2 True incremental source/index caching

**Gap:** The current release path is partially incremental, not gracefully incremental end to end. The manifest-driven prefilter can skip unchanged-file parse and embedding, changed AST IR can be merged with the previous snapshot, persisted vector/BM25/selected graph shards can reuse unchanged path artifacts, and some semantic/SCIP work can be scoped. The pipeline still materializes combined in-memory repo-level artifacts, and widened SCIP/helper/tool startup costs can still scale close to a full reindex.

**Why this is core-level:** This is the largest remaining architecture gap between a hardened prototype and a stable release. On medium and large repos, important parts of “incremental” behavior can still scale like full reindex even when parse and embedding reuse are working.

**Current failure modes:**
- repository inventory and file hashing are still not represented by one canonical planner object across snapshot identity, incremental planning, SCIP language detection, file-artifact reuse, and publication.
- `run_index_pipeline()` still rehydrates unchanged metadata into one full `elements` list and builds temporary whole-snapshot vector/BM25/legacy graph objects before persistence.
- persisted vector/BM25 path shards reuse compatible previous artifact files
  for unchanged paths, and legacy graph shard reuse is intentionally
  conservative; PostgreSQL relational facts have a safe changed-path delta, but
  IR graph publication and snapshot JSON are still whole-snapshot.
- optional SCIP has scoped paths, but `detect_scip_languages()` still walks the repository and unsupported/widened tool paths still invoke repo/package-scale indexers even when only one file changed.
- file units now carry `content_hash` / `blob_oid` metadata, and the manifest-first prefilter now disables reuse when required fingerprints are missing; file identity is not yet the single shared planner anchor across inventory, snapshot diffing, SCIP scope, file-artifact stores, and publication.
- unchanged files are skipped before parse/embedding, before changed-file AST
  merge, before persisted vector/BM25 artifact-shard rewrites, and before safe
  PostgreSQL relational fact upserts. They are not yet skipped before every IR
  graph, compiler/indexer, repair/widened relational fact, or temporary
  artifact-build step.

**Required work:**
- Promote the existing manifest-first prefilter into the canonical `plan_changes` stage:
  - inventory repository once
  - compute file fingerprints once
  - diff against prior snapshot before parse/embed/SCIP/helper execution
- Share one canonical inventory pass across:
  - repository identity
  - loader-visible files
  - incremental planner
  - SCIP language detection
- Make file fingerprints first-class and mandatory:
  - prefer git `blob_oid` when available
  - otherwise persist `content_hash`
  - do not trust size/mtime for reuse unless a future degraded mode makes that
    risk explicit
  - use the same identity contract in inventory, file units, manifest planning,
    SCIP scoping, and artifact publication
- Add a `FileArtifactStore` keyed by `(repo, rel_path, blob_oid/content_hash)` for:
  - parsed file output
  - file-level AST/IR shards
  - per-file semantic facts
  - per-file embedding payloads or embedding references
- Refactor the pipeline into:
  - `plan_changes`
  - `extract_changed`
  - `merge_snapshot`
  - `persist`
- Promote existing `CodeIndexer.index_files()` and file-manifest logic into the snapshot pipeline instead of keeping them as side paths.
- Reuse unchanged per-file artifacts before embedding and before SCIP where supported.
- Split compiler/tooling invalidation by scope:
  - file-local helper execution when exported interface is unchanged
  - dependent-neighborhood rebuild when exported interface changes
  - package/workspace or repo-wide rebuild when toolchain/build graph changes
- Remove avoidable repeated tool startup:
  - replace `go run` helper execution with packaged binaries, cached build artifacts, or daemonized helper mode
- Keep treating “missing file fingerprints” as a reason to disable incremental
  mode, not to silently claim success; the manifest-first prefilter now does
  this, but the rule still needs to hold after the planner is centralized.

**Exit criteria:**
- A one-file edit does not trigger full-repo parsing, full-repo embedding, or whole-repo downstream artifact rebuild when file-shard reuse is valid.
- A one-file implementation-only edit does not trigger whole-repo SCIP/helper recomputation unless interface or build-graph rules require widening.
- Incremental publish reports correct `added/modified/removed/unchanged` counts for both git and non-git repos.
- There are integration tests proving unchanged files are not reparsed, re-embedded, or republished through full vector/BM25/graph rebuilds when reuse is valid.
- There is a benchmark/regression test showing one-file update cost is materially below full reindex.

### P0.2a Graceful update under code change

**Research framing:** A graceful updater should widen work in proportion to semantic impact, not in proportion to raw textual churn.

**Current status:** partially implemented. Incremental plans now classify
change kinds, reuse changed-unit embeddings where the prepared text is
unchanged, detect API-frontier changes, compute package-scope roots, persist
per-file interface digests, emit dependency-frontier metadata, record
deterministic degraded-mode reasons, and feed scoped semantic/SCIP refresh
paths in the happy path. The remaining release-grade gap is broader
file-shard-native end-to-end behavior plus backend/toolchain benchmark evidence
by edit class.

**Core idea:** Separate change detection into four classes and bind each class to an invalidation radius:
- identity-stable: no rebuild
- implementation-local: changed file body, stable exported interface
- interface-affecting: changed exports, signatures, inheritance, imports, or generated symbol set
- toolchain/global: changed build config, dependency lockfile, generator output, parser fingerprint, or model/tool fingerprint

**Recommended update calculus:**
1. inventory once:
   - collect `blob_oid` or `content_hash`
   - language
   - package/workspace membership
   - toolchain slice (`go.mod`, `package.json`, `tsconfig`, `pyproject.toml`, `Cargo.toml`, etc.)
2. derive two digests per file:
   - content digest for exact file identity
   - interface digest for exported semantic surface
3. choose invalidation radius:
   - content changed, interface stable:
     - reparse changed file
     - recompute embeddings only for changed file elements
     - rerun file-local semantic helper
     - preserve reverse dependencies
   - interface changed:
     - invalidate reverse imports/callers/inheritors reachable from the changed file
     - rerun affected helper/SCIP/package slices
   - toolchain/global changed:
     - widen to package, workspace, or full repository explicitly
4. store file-native artifacts:
   - `ParseShard`
   - `IRShard`
   - `EmbeddingShard`
   - `SemanticFactShard`
   - `InterfaceDigest`
   - dependency frontier metadata
5. publish atomically:
   - stage a new snapshot
   - merge unchanged shards plus recomputed shards
   - promote only after validation succeeds

**Graceful degradation rules:**
- Missing or inconsistent fingerprints must widen scope, never narrow it.
- Helper/SCIP/tool failure must downgrade semantic precision with warnings, not silently preserve stale changed-file facts.
- Unknown dependency frontier must invalidate the larger safe neighborhood.
- A degraded incremental publish should carry explicit metadata about which reuse assumptions were disabled.

**Evidence requirements:**
- A function-body edit stays file-local.
- A signature or inheritance change invalidates dependents but not unrelated subtrees.
- A build-config change widens scope deterministically.
- Benchmarks are reported by edit class, not only by “full vs incremental”.

### P0.3 Embedding/model cache correctness and reuse

**Gap:** There is now a real text-level embedding cache in the active indexing path, but its correctness envelope is still narrower than a stable release requires.

**Why this is core-level:** Stable release users will judge the system by repeated indexing/update cost. Without a correct embedding cache, model inference dominates runtime and cost even when source changes are small.

**Current status:** partially implemented. The active embedder path now has an explicit `EmbeddingService` boundary with `prepare_text(...)`, `fingerprint(...)`, `embed_many(...)`, and cache-aware `embed_elements(..., reuse_index=...)`; the active indexer uses that service for element and repository-overview embeddings. The path deduplicates identical prepared texts, stores cached vectors as `float32` buffer payloads, treats stale list-format cache entries as misses, validates active cache and changed-unit reuse with a typed embedding fingerprint, carries embedding refs/fingerprints into active PG retrieval rows for both code and documentation chunks, persists fingerprints on vector artifacts and repository overviews, and carries the active query embedding fingerprint into local vector search, repository-overview vector search, and PostgreSQL semantic search. Same-dimension stale vectors are filtered on those serving paths. The remaining gap is extending the same release-grade evidence across every backend/reuse surface that can serve vectors.

**Current failure modes:**
- embedding cache identity is now exposed through an `EmbeddingService`
  boundary on the active `CodeEmbedder`, and snapshot manifests, vector
  artifacts, repo-overview artifacts, and query semantic search share the
  persisted fingerprint payload.
- PG retrieval rows now carry embedding refs/fingerprints, but backend evidence still needs to expand beyond retrieval rows into locks, migrations, outbox/redo, and relational facts.
- the incremental manifest compatibility hash checks major embedding settings, but the same fingerprint discipline is not yet propagated uniformly across all reuse surfaces.
- Ollama embedding uses the true `/api/embed` batch endpoint where supported and
  falls back to bounded per-text concurrency on older servers.

**Required work:**
- [x] Formalize the current embedder path behind an `EmbeddingService` or equivalent core boundary with:
  - `prepare_text(element)`
  - `fingerprint()`
  - `embed_many(texts)`
  - cache-aware `embed_elements(elements, reuse_index)`
- Define and persist an `embedding_fingerprint` including:
  - provider
  - model name/version identity
  - normalization flag
  - max sequence length
  - embedding text preparation schema version
  - embedding dimension
- Key embedding cache entries by:
  - prepared text hash
  - embedding fingerprint
- Store fingerprint on:
  - snapshot metadata
  - vector-store artifacts
  - repo overviews
  - query/embedding cache keys
  - PG/vector persistence rows that serve embeddings; active PG retrieval rows are done
- Refuse artifact reuse or force rebuild when embedding fingerprint mismatches.
- Batch cache hits/misses correctly: fetch cached vectors first, embed only misses, then write misses.
- Make embedder/model initialization lazy for non-embedding code paths.
- Remove avoidable quadratic work in batch embedding assignment.

**Exit criteria:**
- Repeated indexing of unchanged files makes zero provider/model calls.
- Switching embedding model or preparation schema invalidates old vectors deterministically.
- There is a test proving two identical embedding texts trigger one provider computation.
- There is a test proving same-dimension model changes do not silently reuse stale vectors.

### P0.4 Snapshot artifact handle caching for query serving

**Status:** partially implemented. `LoadedSnapshotArtifacts` handles and an artifact-key LRU now exist in `IndexPipeline`, and `QueryPipeline.query_snapshot()` can consume request-local retriever/graph handles without swapping singleton query state. Public `FastCode` query entrypoints now share a read lock instead of serializing independent reads behind the mutation lock. Snapshot-scoped `FastCode.query_stream()` now captures immutable snapshot handles under the read lock and releases that lock before streaming generation, while non-snapshot streams keep the singleton-state fence. `LoadedSnapshotArtifacts` now documents the read-only serving-handle contract, and the `/query` API endpoint has a concurrency regression proving two snapshot-scoped requests can overlap after `asyncio.to_thread` offload.

**Remaining gap:** this is not yet a full serving isolation/read-scalability closure. Non-snapshot streaming queries still hold a read lock for the generator duration to fence singleton state, and endpoint/operator benchmarks are still needed.

**Required work:**
- [x] Narrow the public `FastCode` service lock so immutable snapshot queries can run concurrently while mutations remain serialized.
- Keep artifact-handle load/cache mutation internally locked without serializing the full query.
- [x] Treat `LoadedSnapshotArtifacts` contents as read-only serving handles and document which contained stores/retrievers are safe to share across concurrent reads.
- Keep serving-time artifact load separate from repair/rebuild behavior.
- [x] Add endpoint-level concurrency coverage above the lower `QueryPipeline` handle tests.
- [x] Let snapshot-scoped streaming queries release the service read lock after immutable handle capture.

**Exit criteria:**
- Repeated queries against the same snapshot avoid disk reload.
- Concurrent snapshot queries on different snapshots do not cross-contaminate results.
- Eviction behavior is bounded and tested.
- Public API/`FastCode` query entrypoints can serve independent immutable snapshot reads concurrently, or the singleton serialization contract is explicitly documented as a non-goal.

### P0.5 Real external-tool integration matrix

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

### P0.6 Production storage semantics

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

### P0.6a FP/FCIS data schema flow completion

**Gap:** The codebase now has the right package layout for `api`, `schemas`, `retrieval`, `store`, and `main`, but the dataflow contract is still inconsistent. Core paths use frozen dataclasses in several places, while other store/query/projection flows still pass raw `dict[str, Any]` payloads across boundaries.

**Why this is release-critical:** Stable release needs predictable schema flow:
- API shell validates with Pydantic
- core logic operates on explicit frozen types
- persistence returns typed records
- translation between these layers is explicit and testable

Without that, layout cleanup is cosmetic; runtime contracts remain implicit.

**Current gaps visible in code:**
- `store/manifest.py` and `store/snapshot.py` still expose mixed record/dict APIs instead of converging on typed records
- `store/projection.py`, `store/cache.py`, `store/vector.py`, and `store/pg_retrieval.py` remain heavily dict-oriented at public compatibility boundaries even where active row materialization is now typed
- query/retrieval paths still use dict-shaped rows as a primary interchange format instead of a small typed boundary model set
- `schemas/__init__.py` and `schemas/api.py` correctly isolate Pydantic, but the rest of the system has not fully converged on explicit translation adapters

**Required work:**
- define the canonical schema flow explicitly in code/docs:
  - `schemas.api` / `schemas.config` for Pydantic boundary types
  - `schemas.core_types` and `store.records` for frozen dataclass interchange
  - adapters that translate field-by-field between boundary, core, and persistence forms
- eliminate mixed dict/record public APIs where possible:
  - promote typed snapshot artifact/redo/outbox records into public `*_record()` accessors where those surfaces are stable
  - add typed projection/build/session/cache metadata records where those payloads are stable enough
- make store modules prefer `*_record()` style APIs and treat dict-returning helpers as compatibility shims
- keep retrieval-core purity strict:
  - no Pydantic imports
  - no hidden boundary serialization logic in pure ranking/fusion/graph functions
- add architecture fitness tests that assert:
  - shell packages do not mass-assign Pydantic payloads into core/store records
  - store modules do not regress from typed records back to dict returns in stabilized surfaces
  - imports continue to point inward after the package split

**Exit criteria:**
- a maintainer can trace request data as:
  - `api schema -> explicit translator -> core dataclass -> explicit persistence record`
- the hot release paths no longer rely on undocumented dict shapes as their primary cross-layer contract
- architecture tests fail fast when new code bypasses explicit translation

### P0.6b Copy-boundary and serialization discipline

**Gap:** Boundary handling is still copy-heavy. Several hot paths materialize Python `dict`/`list` payloads repeatedly before persistence, cache storage, or native-library handoff. That undermines both throughput and the intended thin-shell contract.

**Current audit findings:**
- generic JSON normalization is now narrower, but still present in remaining shell/storage edges:
  - `utils/json.py:safe_jsonable()` recursively converts dict/list/set/object trees
  - `api/routes.py` and `api/web.py` now use explicit source/dialogue serializers, but other API payloads still pass through dict-shaped records
- store persistence still serializes JSON payloads at DB boundaries, but some hot rows now avoid full object expansion:
- `store/snapshot.py` relational facts now use explicit field serializers
  instead of `json.dumps(obj.to_dict())`, and safe PostgreSQL incremental
  plans can copy unchanged previous rows plus upsert only changed-path rows
- `store/snapshot.py` snapshot/snapshot-ref lookup helpers now use explicit typed row reconstruction and explicit compatibility serializers on active lookup paths
- `store/snapshot.py` snapshot-file save/load paths now use store-owned explicit IR payload serializers instead of `IRSnapshot.to_dict()` / `IRSnapshot.from_dict()` round-trips, and non-`networkx` graph persistence now uses a narrow JSON walker instead of generic object normalization
- active `store/snapshot.py` SCIP artifact, redo-task, and publish-outbox paths now materialize rows through explicit typed record adapters instead of `row_to_dict()`
- `store/projection.py` now reconstructs active dirty-scope/build rows through typed record adapters and explicit compatibility serializers
- `store/index_run.py` now exposes typed run/publish-task records and active publishing/indexing callers prefer those record APIs, but compatibility dict-return helpers still remain
- `store/cache.py` now reconstructs active dialogue/session cache payloads through typed records and explicit compatibility serializers instead of mutating ad hoc dict payloads in place
- `store/pg_retrieval.py` now emits explicit typed element payloads for `metadata_json`, and `store/manifest.py` now uses explicit row-field serializers on active paths, but compatibility dict-return APIs still expose dict-shaped boundaries in places
- vector/cache paths copy embeddings through Python containers:
  - `indexing/embedder.py` now stores cached embeddings as native `float32` byte buffers
  - `store/cache.py` now writes typed dialogue/session/query payloads as explicit JSON envelopes and embedding payloads as length-prefixed buffer envelopes; generic `get()/set()` remains for legacy or untyped callers
  - `store/vector.py` now persists repository overview metadata as an explicit JSON manifest plus NumPy embedding archive and lets metadata-only overview callers avoid loading embedding payloads
  - `indexing/pipeline.py` now stages embedded elements as a single `np.ndarray[np.float32]` matrix instead of a Python list of arrays before vector-store insertion, and vector-store search/repository-overview ranking uses explicit float32 boundary helpers
  - `DBRuntime` registers the pgvector Psycopg adapter when available, and `store/pg_retrieval.py` now passes native float32 vectors at the active PG boundary instead of duplicating new rows into `list[float]` array payloads; SQL vector literals remain only as an adapter-missing fallback
  - `store/pg_retrieval.py` still keeps legacy `embedding_arr` read fallback support, but ranks fallback candidates as a NumPy matrix before metadata inflation
- unit artifact persistence is narrower, but still dict-shaped:
  - `store/unit_artifacts.py` no longer uses generic row normalization on list/load paths and now serializes only the metadata subtree explicitly
  - it still persists unit metadata as JSON text plus a dict-shaped compatibility payload for callers
- publishing/integration edges are partially hardened:
  - Terminus lineage publishing now accepts typed `IRSnapshot` objects and avoids full snapshot `to_dict()` expansion in `PublishingService` and the active index pipeline
  - `PublishingService` now also returns compatibility manifest payloads through an explicit field serializer instead of `manifest.to_dict()`
  - remaining copy points include JSON outbox payload strings and the legacy dict publisher API kept for compatibility
- indexing hot paths are narrower:
  - `indexing/indexer.py` now serializes imports explicitly and reuses the serialized import list for file-level hashes/metadata instead of repeated `ImportInfo.to_dict()` calls
  - active embedder handoff now passes a minimal explicit element payload instead of deep-copying full `CodeElement.to_dict()` payloads before embedding
  - `indexing/pipeline.py` now materializes unit-artifact refresh/replace rows through explicit `CodeElement`/mapping serializers instead of a generic `to_dict()` fallback, and it no longer mutates input metadata while enriching those rows
- snapshot indexing materialization is narrower:
  - `indexing/pipeline.py` now stages vector-store and PG retrieval payloads from explicit serialized element views instead of expanding each `CodeElement` with `to_dict()` and mutating live element metadata for transport-only fields
  - PostgreSQL relational fact publication now avoids whole-table rewrite for
    safe incremental plans by copying unchanged rows from the previous snapshot
    and upserting only changed-path rows
- retrieval result assembly is narrower:
  - `retrieval/hybrid.py` and `retrieval/core/fusion.py` now materialize result elements through explicit `serialize_code_element(...)` payloads instead of repeated `CodeElement.to_dict()` calls for keyword hits, file/type helpers, graph expansion, projected-only doc backfill, and BM25 persistence
  - `retrieval/iterative.py` now uses the same explicit serializer for agent-found file/class/function result rows instead of expanding full `CodeElement.to_dict()` payloads during selection-to-result conversion
  - persisted BM25 reload paths in `retrieval/hybrid.py` and `main/fastcode.py` now rehydrate `CodeElement` objects through an explicit adapter instead of `CodeElement(**payload)` mass-assignment
- graph artifact boundaries are narrower:
  - `graph/build.py` now writes compatibility graph element indices through explicit `serialize_code_element(...)` payloads and rehydrates persisted `element_by_name` / `element_by_id` entries through `deserialize_code_element(...)` on load/merge instead of `CodeElement.to_dict()` / `CodeElement(**payload)` round-trips
  - IR graph snapshots now persist and reload compact `IRGraphView` payloads backed by `python-igraph`; retrieval graph expansion uses the compact reachability API instead of materializing a NetworkX union graph when compact views are available

**Direction:** "Zero-copy" is only realistic where data remains in native buffers or immutable typed views end to end. In pure Python object graphs, the release target should be:
- zero extra copies across native/vector boundaries where possible
- one explicit materialization at the shell/storage boundary where JSON is actually required
- no repeated dict/list reconstruction between adjacent internal layers

**Required work:**
- Define boundary classes by transport type:
  - Python-core typed dataclasses for internal flow
  - native-buffer carriers for embeddings/vectors
  - JSON/bytes serializers only at persistence/API/native-process edges
- For embedding/vector paths:
  - keep embeddings as `np.ndarray[np.float32]` through compute, cache lookup, batch add, and DB/vector handoff
  - avoid `tolist()` except where the storage driver strictly requires JSON/array literals
  - prefer binary/native array transfer where backend supports it
- For JSON/document paths:
  - serialize once at the final boundary instead of `to_dict()` followed by recursive cleanup followed by `json.dumps()`
  - evaluate direct JSON emission in the native/storage layer when supported by the backend/driver
- For store/query shell:
  - replace generic `safe_jsonable()` usage on known typed payloads with explicit serializers
  - eliminate dict round-trips like `record -> dict -> row/json -> dict -> record` on hot paths
- Add architecture/perf tests that fail when:
  - embeddings are converted to Python lists in active hot paths without a backend requirement
  - boundary code performs repeated `to_dict()` / `safe_jsonable()` / `dict()` normalization on already-typed objects
  - cache/storage adapters regress from typed/buffer-aware APIs to generic pickle/dict payloads

**Exit criteria:**
- hot embedding/vector flows stay in native array form until the final backend boundary
- typed internal payloads are materialized to JSON at most once per boundary crossing
- a benchmark proves lower copy/allocation overhead for repeated index/update/query workloads

### P0.6c Object materialization minimization

**Gap:** Even where FastCode already uses native-backed libraries (`FAISS`, `NumPy`, `pgvector`), the runtime frequently materializes results back into Python `dict`, `list`, dataclass, or `networkx` object graphs too early. That destroys the benefit of keeping work in C/C++/Rust-backed layers.

**Current audit findings:**
- embedding path:
  - `indexing/embedder.py` now writes cached vectors as `float32` byte buffers and reconstructs cache hits with `np.frombuffer(...)`
  - stale pre-buffer list payloads are treated as cache misses and overwritten with the current binary format
  - `indexing/indexer.py` now hands the embedder a minimal explicit payload instead of materializing full `CodeElement.to_dict()` copies for every indexed element
- retrieval result path:
  - `retrieval/hybrid.py` and `retrieval/core/fusion.py` now defer `CodeElement` materialization to explicit final result serializers instead of expanding full dataclass payloads through `to_dict()` in keyword/graph/doc-projection helper paths
- PostgreSQL retrieval path:
  - `store/pg_retrieval.py` now strips raw embedding arrays from `metadata_json` before insert so vector payloads are not duplicated in JSON storage
  - the active insert/search path passes native float32 arrays through the pgvector Psycopg adapter; vector literals are retained only for the degraded case where the Python adapter is unavailable
- vector retrieval path:
  - `store/vector.py` keeps shard-backed vector rows hot after load, rebuilds FAISS lazily only when a compatibility operation needs it, persists repository overviews as an explicit JSON manifest plus NumPy embedding archive, and vectorizes repository-overview ranking before result metadata assembly
  - `utils/vectors.py` gives vector hot paths explicit view/contiguous/mutable ownership policies and avoids mutating caller-owned arrays during non-finite sanitization
  - full repository-overview consumers still decode JSON metadata into Python dicts because selector/BM25 flows currently operate on Python text/metadata payloads
- PostgreSQL retrieval result path:
  - semantic fallback now delays JSON metadata inflation until after vectorized NumPy ranking
  - direct pgvector/keyword result rows still materialize JSON payloads at the retrieval boundary
- graph path:
  - IR graph expansion now uses compact `IRGraphView` reachability on the active retrieval path when compact graph payloads are available
  - main composition-root callees/callers/dependencies now use compact bounded graph traversal when graph artifacts are available
  - MCP directed path, impact, caller, and Steiner tools now use compact saved graph handles and sidecar symbol maps on current snapshots
  - architecture tests now freeze NetworkX imports to explicit compatibility
    boundaries, so new graph hot paths cannot add NetworkX without updating the
    allowlist deliberately
  - projection graph algorithms still use `networkx`, which is a named compatibility/materialization boundary until a projection-native rewrite
  - compatibility `graph/build.py` loads now retain compact shard adjacency payloads and only materialize `networkx` lazily for save/merge/path/stats compatibility paths
  - snapshot IR graph storage/load no longer has to reconstruct full `networkx` graph objects for active retrieval; legacy node-link graph payloads remain supported
- IR/store path:
  - `IRSnapshot.to_dict()/from_dict()` and record `to_dict()/from_dict()` patterns are still common interchange mechanisms
  - DB rows are routinely converted with `row_to_dict()` before typed reconstruction

**Design rule:** keep data in C/C++/Rust-backed representation as long as possible, and only materialize into Python objects when:
- Python business logic must inspect or mutate it
- an API boundary requires JSON/text output
- a backend driver lacks a native/binary transfer path

**Required work:**
- Introduce explicit "materialization boundaries" in architecture docs/code:
  - native buffer / native index boundary
  - typed Python domain boundary
  - JSON/text/API boundary
- For vectors/embeddings:
  - prefer ndarray/buffer-preserving cache payloads
  - avoid list-of-float round-trips for pgvector/FAISS handoff if binary/native driver support is available
  - keep batch search/rerank operations vectorized in NumPy/FAISS rather than re-looping through Python objects prematurely
- For graph processing:
  - migrate hot graph algorithms toward `igraph` or another compact native-backed engine
  - isolate Python object materialization to debug/export/API layers
- For store/DB paths:
  - move from `row -> dict -> typed record` toward `row -> typed record` adapters
  - avoid full JSON object inflation when only a few fields are needed
- For IR interchange:
  - stop using `to_dict()/from_dict()` as the default internal interchange format in hot paths
  - keep typed instances or compact columnar/native forms until a real boundary requires expansion
- Add measurement around:
  - Python allocation counts
  - peak RSS during index/query/update workloads
  - materialization cost share in profiling for medium repositories

**Exit criteria:**
- profiling shows hot paths spend most time in native/vector/graph backends, not Python object assembly
- repeated query/index/update workloads avoid unnecessary `list`/`dict`/JSON inflation in the inner loop
- architecture tests and profiling notes identify any intentional materialization boundary explicitly

### P0.7 API and file-upload security hardening

**Status:** Closed for the current trusted-local/proxy-auth contract. Direct unauthenticated production exposure remains explicitly unsupported.

**Recently landed:**
- Raw `ZipFile.extractall()` was replaced on active upload and repository ZIP load paths with safe member validation.
- Archive validation rejects absolute paths, traversal, symlinks/special files, excessive member counts, oversized extracted payloads, and suspicious compression ratios.
- CORS defaults are local-origin only and configurable through explicit environment settings.
- REST API now binds `127.0.0.1` by default; public binding requires an explicit `--host 0.0.0.0` operator decision.
- Deployment notes define the auth expectation: FastCode has no built-in user auth, so shared/remote deployments require a proxy or gateway with TLS and authz for mutation endpoints.
- Regression tests cover utility-level malicious archive entries and expansion limits, endpoint-level path traversal uploads, and non-wildcard CORS defaults.

**Remaining follow-up:**
- Add full production deployment examples for a concrete reverse proxy/auth gateway.
- Add API-level zip-bomb tests if upload limits become configurable enough to test without large fixtures.

**Exit criteria:**
- Upload path traversal and zip-bomb regressions are tested.
- Production deployment docs include CORS/auth guidance.

### P0.8 Service-wide state isolation under concurrent traffic

**Status:** closed for correctness in the singleton trusted-local mode, but not closed for read scalability. Service-wide locking prevents mixed query/mutation races; artifact handles now exist below the composition root, but public query entrypoints still serialize behind the same lock.

**Recently landed:**
- Repository load/index, snapshot pipeline, projection build, query/query-stream, multi-repo cache load, delete, cleanup, and shutdown paths now acquire a shared reentrant service lock.
- REST and web load+index/upload+index helpers hold that lock across the entire combined mutation, including scan-cache invalidation.
- Regression coverage verifies the combined helpers use one critical section.
- Regression coverage verifies query serving does not overlap load, index, delete, refresh, or cleanup/unload-style mutations.
- Lower-level snapshot query serving can use request-local `LoadedSnapshotArtifacts` handles without `QueryPipeline._snapshot_query_lock`.

**Remaining follow-up:**
- Add endpoint-level concurrency tests for upload vs query with real ASGI request scheduling.
- Add benchmark evidence for concurrent snapshot queries with and without
  background mutations.

**Exit criteria:**
- Concurrent mutation/query behavior is either serialized by design or explicitly rejected with clear errors.
- No query can observe half-loaded artifacts or deleted repository state.

### P0.9 Release gate and compatibility policy

**Gap:** Smoke tests are green, but stable release gates need a documented contract beyond "current tests pass."

**Current status:** partially closed. [docs/release.md](./docs/release.md)
now defines blocking gate tiers, package/install gate commands, patch/minor/major
blocking rules, acceptable degraded behavior, and a first pre-stable artifact
compatibility policy. `scripts/release_gate.py` implements the package/install
tier for built artifacts.

**Remaining stable-release gap:** backend and external-tool gates are documented
but not implemented as release automation. Performance benchmarks now exist for
the main hot paths, but stable release still needs a release-threshold policy,
captured expected outputs, and artifact-family compatibility tests.

**Required work:**
- [ ] Define supported OS/Python/backend matrix.
- [x] Define initial semantic versioning and compatibility promises for snapshots, manifests, and projection artifacts.
- [x] Add release checklist: tests, packaging, install smoke, migration smoke, docs update, changelog.
- [x] Add a minimal benchmark/performance envelope for medium repositories to catch major regressions.
- [x] Separate release gates into explicit tiers:
  - architecture gate
  - package/install gate
  - backend integration gate
  - external-tool gate
  - performance gate
- [x] Define what can block a patch release vs minor release vs major release.
- [x] Define which warnings/errors are acceptable degraded behavior and which invalidate a stable release claim.

**Exit criteria:**
- [ ] A maintainer can tag a release by following one checklist with reproducible commands and expected outputs across architecture, package/install, backend, external-tool, performance, and docs gates.

### P0.10 Documentation, deployment, and operator runbooks

**Gap:** The codebase has more runtime modes than the docs currently prove safe to install and operate.

**Current status:** partially closed. `DEPLOYMENT.md` now documents local
checkout install, built-artifact install smoke, trusted-local API/web exposure,
CORS/upload assumptions, and the release gate. [docs/release.md](./docs/release.md)
documents the release gate matrix and explicitly marks PostgreSQL production
semantics as gate-open.

**Remaining stable-release gap:** full production runbooks are still missing
for backup/restore, migration/rollback, cache invalidation, lock/fencing
incident recovery, and optional language-tool installation.
May 18, 2026 audit: `DEPLOYMENT.md` and `docs/release.md` cover release-gate
validation, local install, built-artifact smoke, trusted-local API/web exposure,
CORS, upload safety, and service extras. They do not yet meet the release-grade
guide/runbook bar below, so the install-guide checkbox remains open.

**Required work:**
- [ ] Write a release-grade install guide for:
  - local single-user mode
  - API/web service mode
  - PostgreSQL-backed mode
  - optional language-tooling/SCIP mode
- Write deployment runbooks covering:
  - required environment variables
  - filesystem expectations for cache/artifact roots
  - startup/shutdown sequencing
  - backup/restore of snapshots, manifests, and projection artifacts
  - log locations and diagnostic collection
- Write operator playbooks for:
  - cache invalidation
  - artifact rebuild after model/tool change
  - database migration/rollback handling
  - lock/fencing incident recovery
  - failed upload/index remediation
- Document trusted-local-only vs production-service assumptions explicitly for the API/web entrypoints.
- Document supported language tiers:
  - stable
  - degraded/structural
  - experimental

**Exit criteria:**
- A new maintainer can install, start, index, query, upgrade, and recover the service using only checked-in docs.
- Public docs do not over-claim support that is only partially tested.

### P0.11 Artifact and migration compatibility policy

**Gap:** Snapshot/manifests/projection/vector artifacts now carry more metadata, but there is not yet a strict compatibility contract for upgrades and invalidation.

**Required work:**
- Version every persisted artifact family explicitly:
  - snapshot metadata
  - manifest records
  - projection artifacts
  - cache payloads
  - vector/index artifacts
- Define upgrade rules for:
  - forward-incompatible schema changes
  - additive metadata fields
  - tool-output schema changes
  - embedding fingerprint changes
  - helper/scip artifact lineage changes
- Add startup validation that detects incompatible artifacts early and reports rebuild requirements clearly.
- Add migration/compatibility tests covering:
  - load old artifact -> accept
  - load old artifact -> rebuild required
  - load incompatible artifact -> explicit hard failure

**Exit criteria:**
- Upgrading versions cannot silently load semantically incompatible artifacts.
- Release notes can state exactly which prior artifacts remain readable and which require rebuild.

### P0.12 Dependency and supply-chain hardening

**Gap:** Stable release needs tighter control over dependency drift and packaging trust than development smoke tests currently prove.

**Current status:** partially closed. The package/install release gate verifies
wheel/sdist content for required helper assets and installs from built artifacts
in fresh virtualenvs. The core runtime no longer imports the local embedding
stack eagerly, and service extras are split out of the default install path.
This catches missing package data and editable-install drift, but it is not yet
a dependency security or resolver-drift gate.

**Required work:**
- Add a dependency review pass for runtime packages with:
  - pinned or constrained high-risk libraries
  - explicit extras boundaries
  - known-native/tooling dependencies documented
- Add security/reproducibility checks:
  - `pip-audit` or equivalent
  - [x] wheel/sdist content verification for required helper assets
  - [x] helper asset inclusion verification
  - lockfile or documented resolver strategy for repeatable installs
- Review subprocess/tool invocation surfaces for:
  - path resolution
  - environment inheritance
  - timeout policy
  - stderr/stdout diagnostics
- Decide which external tools are mandatory, optional, or auto-detected and encode that in install metadata/docs.

**Exit criteria:**
- A stable release can be rebuilt and installed reproducibly from a clean environment with documented dependency expectations.
- Dependency/tooling changes that would alter runtime behavior are visible in release review, not discovered after release.

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

### P1.4 Repository inventory and parse-cache efficiency

- Unify repository inventory across loader, repo info, indexing, non-git identity, and SCIP language detection.
- Add per-content-hash parse cache keyed by `(rel_path, content_hash, parser_version)`.
- Make loader ignore rules and SCIP visible-file selection share one canonical inventory.
- Add explicit degraded-mode metrics when parse cache is bypassed or invalidated.

### P1.5 Continue store-boundary typing

Typed records now exist for manifests, snapshot refs, snapshot records, index runs, publish tasks, SCIP artifact refs, redo tasks, outbox events, dialogue turns, dialogue sessions, and active projection build/dirty-scope rows.

Still raw-dict-heavy at boundaries:
- API-facing manifest / artifact payload adapters
- snapshot-store and projection-store compatibility payload adapters
- cache/query result payloads
- vector-store search/repo-overview payloads
- pg-retrieval row/result payloads

This item should be treated as the implementation slice of the broader P0.6a schema-flow requirement above, not as isolated cleanup.

### P1.6 Observability and debuggability

- Promote pipeline layer metrics and resolver diagnostics into stable API fields.
- Add structured logs for artifact swaps, semantic escalation, resolver fallback, and lock acquisition/release.
- Add stable cache/update metrics:
  - repository inventory reuse hit/miss
  - parse cache hit/miss
  - embedding cache hit/miss
  - artifact-handle cache hit/miss
  - incremental planner changed-file counts
- Add a documented diagnostic bundle for support: config summary, storage backend, dependency availability, latest run metadata.

### P1.7 Release-fixture and benchmark realism

- Add one or more medium-sized real-world fixtures that exercise:
  - multi-language indexing
  - repeated incremental updates
  - cache reuse after restart
  - query-time snapshot switching
- Record baseline envelopes for:
  - cold full index
  - warm full index
  - one-file incremental update
  - repeated identical query against cached snapshot artifacts
- Track peak memory and artifact size growth, not just wall-clock time.

### P1.8 Runtime dependency optimization audit

**Gap:** Several hot-path dependencies are selected for convenience or compatibility rather than measured runtime efficiency. Stable release should not carry pure-Python bottlenecks in the core path if there is already a viable compiled/runtime-efficient alternative.

**Current audit findings:**
- May 18, 2026 update: the graph-engine decision is now recorded in
  [GRAPH_ENGINE_DECISION.md](./GRAPH_ENGINE_DECISION.md).
- The canonical hot path is `IRGraphView` / `igraph`:
  - `ir/graph.py` now provides compact native traversal, shortest path,
    component stats, degree, neighbor iteration, and undirected views.
  - `indexing/projection_transform.py` uses a projection-native graph
    representation and `igraph` algorithms instead of constructing a primary
    NetworkX graph.
  - MCP directed path, impact, caller, Steiner, and compact cluster projection
    paths use saved graph handles when available.
- `networkx` remains for compatibility/export/debug and legacy graph surfaces:
  - `graph/build.py`
  - `mcp/graph_tools.py` compatibility fallbacks
  - explicit `IRGraphView.to_networkx()` / legacy JSON graph loading
- residual dependency optimization now belongs mostly to P1.9 unless a new hot
  path starts converting native graph handles back to NetworkX.

**Required work:**
- [x] Benchmark graph-heavy operations under `networkx` vs `igraph` on representative repository snapshots:
  - graph build
  - path/impact analysis
  - community clustering
  - retrieval graph expansion
  - snapshot graph load/save conversion
- [x] Decide the graph contract intentionally:
  - `igraph` as primary runtime graph engine for hot paths
  - `networkx` retained only as compatibility/test adapter if needed
- [x] If `igraph` wins materially, refactor toward one canonical internal graph representation for hot paths and isolate conversions at boundaries.
- [x] Avoid dual-maintenance where one module uses `igraph` and the next immediately converts back to `networkx`.
- [x] Add correctness regression tests proving graph engine substitution preserves:
  - directed/undirected semantics
  - edge attributes
  - multi-edge expectations or explicit non-support
  - path and clustering output invariants

**Exit criteria:**
- The graph engine used in hot paths is chosen by benchmark evidence, not convenience.
- Stable release does not pay repeated conversion overhead between graph libraries on critical query/index flows.

### P1.9 Dependency-level performance hardening

**Gap:** Beyond graph libraries, the runtime stack still includes some heavyweight or slower-than-necessary choices that need measurement and possible narrowing before stable release.

**Audit targets:**
- serialization and API payloads:
  - evaluate whether `orjson` should replace standard JSON on hot API/cache paths
- tabular/dataframe dependency surface:
  - audit `pandas` usage frequency and hot-path necessity; remove or isolate if it is not on the release-critical path
- cache backend behavior:
  - compare `diskcache` local latency against actual workload patterns
  - verify whether Redis is only optional scale-out infrastructure or part of the default performance story
- embedding and vector flow:
  - remove avoidable Python loops around NumPy/FAISS batch operations
  - confirm no per-row conversion churn between Python lists and `ndarray`
- Pydantic boundary cost:
  - keep validation at the shell only; avoid repeated model materialization in internal loops
- text/token prep:
  - audit repeated tokenization/normalization work for cacheable preprocessing

**Required work:**
- Produce a dependency audit table with:
  - dependency
  - active hot-path usage
  - measured cost
  - replacement/isolation candidate
  - decision
- Split dependencies into:
  - core hot-path mandatory
  - optional feature dependency
  - dev/test only
  - compatibility shim pending removal
- Remove or demote dependencies that are not justified by measured production value.
- Add benchmark tests around any substitution that changes runtime behavior or output formatting.

**Exit criteria:**
- The stable dependency set is intentionally minimal for the hot path.
- Performance-sensitive libraries are selected based on measured workload evidence and documented tradeoffs.

### P1.10 Agent context bundle v0

**Status:** bundle layer not implemented. This is not a stable-release P0 blocker
for the current API/CLI/MCP product shape, but it is required before FastCode
should claim to be an agent-native context engineering system. The source-level
audit found a real turn-journal/context-compiler v0, including typed
`EvidenceRef`, but no durable bundle/distillation/activation layer.

**Gap:** Current agent integration can compile and persist typed working-memory,
turn-journal, and handoff artifacts. Agents do not yet get durable, cacheable,
expandable `ContextBundle` artifacts with source-ref preserving distillation,
activation feedback, and reuse metadata.

**Required work:**
- Add frozen context records for:
  - context bundles
  - distillation records
  - activation records
- Extend the existing `EvidenceRef` model only where bundle expansion requires
  additional source handles.
- Build a read-only context-bundle path over existing retrieval/projection
  outputs.
- Make bundle cache keys include snapshot/artifact, projection, embedding,
  retrieval-policy, distillation-prompt, and budget fingerprints.
- Add an expansion path from any summary item back to source evidence.
- Add a basic activation API so agent adapters can record which evidence was
  actually useful.
- Keep DCP-style pruning as an adapter/policy layer, not the source of truth for
  code facts.

**Exit criteria:**
- A repeated task can reuse an existing bundle/distillation without broad
  retrieval or re-summarization when fingerprints match.
- Cached bundles are rejected when cited source snapshots, projection versions,
  or distillation fingerprints are incompatible.
- Bundle rendering enforces a token budget deterministically.
- Every bundle item has at least one source ref or an explicit non-code
  provenance reason.
- MCP/API tools expose build, expand, list, and activation-record operations
  without importing Pydantic into inner packages.

### P1.11 Turn journal and context compiler

**Status:** v0 partially implemented.

**Already landed:**
- Pure typed records exist for `EvidenceRef`, `ToolObservation`, `RiskState`,
  `AcceptanceContract`, `TurnIntent`, `TurnPlan`, `WorkingMemoryArtifact`,
  `TurnJournal`, and `HandoffArtifact`.
- `context_compiler.py` renders stable/turn/observation FCX sections from typed
  state and builds typed handoff artifacts.
- `store.records` and `store/cache.py` persist working-memory, turn-journal, and
  handoff records through explicit payload serializers.
- `query/handler.py` feeds prior compiled context into subsequent query turns and
  saves typed working-memory plus journal records.
- REST, web, MCP, and `FastCode` facades expose latest/specific turn context,
  evidence-ref expansion, and handoff creation/fetch.

**Gap:** The v0 gives FastCode a real typed turn/context substrate, but it is not
yet the full deterministic agent-control layer described here. Verifier
transitions, rejected-hypothesis lifecycle, strict FCX parse-back, reset,
branch, ask, and abstain policy gates, and replay/cache-stability guarantees
remain open.

**Required work:**
- Normalize tool outputs from `agent_tools` and future verifier tools into
  observation records with evidence refs and freshness metadata.
- Keep raw tool output and deep context external/restorable by reference rather
  than inline in the compiled prompt.
- Add a strict FCX parser/round-trip contract for compact model-facing prompt
  state.
- Add possibility-management fields:
  - hypotheses
  - supporting refs
  - conflicting refs
  - verifier status
  - unresolved questions
- Add explicit decision-control objects:
  - `RejectedHypothesisLedger`
- Add a versioned promotion/rejection policy:
  - observation -> cited note
  - cited note -> hypothesis support/conflict
  - favored hypothesis -> accepted fact
  - accepted fact -> protected constraint
  - failed or contradicted hypothesis -> rejected ledger entry
  - rejected hypothesis -> reopened only with new evidence, snapshot change, or
    contract change
- Add deterministic action policy gates for:
  - answer/edit
  - retrieve/expand
  - verify
  - ask user
  - branch
  - reset/handoff
  - abstain
- Add execution-regime switching:
  - short-horizon direct mode
  - long-horizon managed mode
  - documented trigger thresholds for promotion/demotion between them
- Add tail-recitation rendering for current goal, current plan, and stop
  condition without replaying old narration.
- Integrate verifier outcomes into next-turn state transitions.
- Add rejected-hypothesis persistence so rewrites do not reintroduce killed
  paths.
- Add handoff-artifact generation for clean-context reset, reviewer delegation,
  and rollback.
- Add replay tests for same-turn deterministic working-set reconstruction.
- Add cache-stability tests proving the stable FCX prefix remains byte-stable
  when snapshot, task, and protected constraints are unchanged.
- Add tests proving model-proposed rewrites cannot directly promote facts,
  constraints, or risk/contract state.
- Add tests for reset triggers, handoff reconstruction, and reopened-hypothesis
  rules.

**Exit criteria:**
- The active agent loop can be inspected as a turn journal, not only as prompt
  text.
- Tool observations can be cached and reused across turns when snapshot inputs
  are unchanged.
- Context rewrite after each tool observation is visible and testable.
- The compiled working-memory view can be replaced without mutating historical
  observations.
- FCX output is generated from typed state and can be parsed back into refs,
  hypotheses, observations, contracts, risk state, rejected hypotheses, and
  working-set directives.
- The loop can branch, verify, downgrade, or abstain based on explicit typed
  uncertainty signals rather than prompt text alone.
- The runtime can explain why it answered, edited, asked, branched, reset, or
  abstained from typed `RiskState` plus `AcceptanceContract`, not only from
  prose summaries.
- Killed hypotheses remain suppressed across rewrites until the reopen rule is
  satisfied.
- Clean-context reset and delegation both run from a typed handoff artifact
  rather than transcript replay.

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

### Layout merge and root test entrypoint

**Status:** Fixed.

**Audit verdict:** PASS after hardening.

**What changed:**
- workspace-root pytest now has explicit discovery config:
  - `testpaths = ["tests", "fastcode/tests"]`
  - `norecursedirs` excludes `repo_backup/` and other non-project trees
  - root `pythonpath` includes workspace root, `fastcode/src`, and `nanobot`
  - root marker registration covers `edge`, `negative`, `test_double`, `integration`, `e2e`, `regression`, `property`, `benchmark`, and `snapshot`
  - root addopts now mirrors package behavior for schemathesis opt-out
- stale renamed-module test references were updated for the new package layout

**Verified outcome:**
- `uv run pytest -q -rs` from repository root now completes successfully instead of failing during collection on `repo_backup/`

## Rules for updating this file

- Update this file whenever a hardening slice lands.
- Remove completed items from the remaining-work sections or move them into landed status.
- Do not create separate audit/status markdown files unless the user explicitly asks for one.
- When adding a new stable-release claim, also add the matching enforcement test and evidence requirement here.
