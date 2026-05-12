# FastCode Performance TODOs

This tracker is separate from `IMPLEMENTATION_TODOS.md` and focuses only on
non-functional performance goals:

- efficient incremental update
- zero-copy or copy-minimal native data flow
- materialization minimization

It intentionally excludes cross-project design work. It also does not track
broad cleanup, style issues, packaging, API hardening, or release process items
unless they directly affect these three goals.

## Audit Verdict - May 10, 2026

FastCode has real partial passes, but the implementation is not yet
performance-native end to end.

Implementation update, May 11, 2026:

- The active snapshot pipeline now scans a fingerprinted file inventory once
  and reuses it for repository info, incremental diffing, AST file units, and
  file manifest publication.
- `CodeEmbedder` now loads/probes providers lazily and exposes a shared
  embedding fingerprint method used by incremental compatibility checks.
- Legacy `np.array(vectors)` insertion paths in `FastCode` now use the explicit
  vector boundary helper.
- Vector shard load/write paths avoid the old `.tolist()` plus row-by-row
  `vstack` pattern in the primary sharded artifact path.
- `IRGraphView.reachable_within()` now uses cutoff neighborhood traversal
  instead of all-node distance calculation.
- PG retrieval metadata JSON now rejects leaked NumPy arrays instead of
  silently list-materializing them; embeddings remain bound through vector
  columns.
- Scoped SCIP reruns now use repo-root filtered execution by default, cache
  filtered SCIP artifacts by tool/profile plus target-file and package-marker
  fingerprints, and report cache hits/misses and scope-copy counts.
- Embedding identity now has a typed `EmbeddingFingerprint` owned by
  `CodeEmbedder`; cache keys, cache payloads, embedded element metadata, and
  incremental compatibility payloads are derived from the same fingerprint
  payload.
- Architecture guards now fail new hot-path uses of generic JSON conversion,
  unapproved `.tolist()` materialization, raw `np.array(vector-list)`
  insertion, and generic row/object dict round trips unless the call site is
  documented as an explicit compatibility or storage boundary.
- Index runs now publish scoped runtime counters for explicit materialization
  boundaries, including JSON encode/decode, pickle load/dump, NetworkX
  conversion, vector list conversion, full snapshot load, and full graph load.
- Active-path regressions now patch generic conversion shortcuts to raise across
  indexing, retrieval, and persistence coverage, so the optimized adapters stay
  on the exercised paths.
- Embedding fingerprint lookup and cache-hit validation no longer start
  providers merely to discover an unconfigured dimension, and the prepared-text
  schema version is part of the fingerprint payload.
- `VectorStore` in-memory row append and the generic vector sequence helper no
  longer grow homogeneous matrices through repeated `np.vstack()` calls.
- Semantic resolver patching no longer clones IR objects through generic
  `to_dict() -> from_dict()` round trips, and the materialization guard now
  covers that patching path.
- PostgreSQL relational fact full-rebuild persistence now batches inserts per
  fact table; true delta fact persistence is still open.
- Snapshot persistence now writes a compact symbol-index sidecar, and
  `QueryPipeline.query_snapshot()` uses it to register symbol aliases without a
  full `IRSnapshot` load when the sidecar is available.
- The same compact sidecar now carries symbol records for `FastCode.find_symbol()`,
  so current snapshots can answer single-symbol lookups without materializing
  all snapshot units and symbols.
- `FastCode` composition-root graph helpers now traverse compact
  `IRGraphView` handles with bounded native/generic adjacency expansion for
  callees, callers, and dependencies instead of calling NetworkX traversal.
- Local repository indexing now defaults to read-only in-place loading for
  local paths, records whether a workspace copy was made, reports copied
  bytes/files when copy mode is requested, and refuses ref checkout against
  in-place working trees.

- Incremental indexing can skip unchanged-file parse and embedding work, reuse
  changed-unit embeddings, merge changed AST IR with a previous snapshot, and
  reuse persisted vector/BM25 and limited graph shards.
- The active indexing path still rehydrates a full combined element list, builds
  full temporary vector/BM25/legacy graph artifacts, loads the previous full IR
  snapshot for merge, writes full snapshot JSON, rebuilds whole IR graph
  materializations, and rewrites relational fact rows for the whole snapshot.
- Vector flow uses NumPy/FAISS/pgvector in important places, but old list-vector
  paths, JSON/list embedding payloads, compression-vs-mmap tradeoffs, and
  fallback materialization paths still exist.
- `IRGraphView` uses `python-igraph`, but NetworkX remains in active hot paths:
  legacy graph building, IR merge matching, projection transforms, MCP graph
  helpers, and compatibility persistence.
- Existing perf benchmarks are useful baselines, but they do not yet enforce
  budgeted update cost, peak RSS, materialization count, bytes read/written, or
  graph-engine selection criteria.

Stable performance claims should wait until the exit criteria in this file are
met with benchmark output from representative repositories.

## Follow-Up Source Audit - May 11, 2026

This audit checked current source paths for gaps not clearly recorded in the
May 10 tracker. It excludes the agent-integration design track and focuses only
on efficient update, copy-minimal native flow, and materialization.

New or sharpened findings:

- Materialization guard coverage is still narrower than the implementation
  summary target. `fastcode/tests/architecture/test_materialization_boundaries.py`
  now covers semantic patch application in addition to the earlier indexing,
  vector, PG retrieval, retrieval, selected vector insertion paths, and the
  main graph helper compact traversal regression. It still does not guard MCP
  graph tools, projection graph transforms, snapshot persistence, or the
  compact query-time symbol-index registration path.
- Provider-starting embedding fingerprint construction and cache-hit validation
  were found in this audit and are now fixed for the compatibility and
  all-cache-hit paths that previously touched `embedding_dim` before actual
  embedding work.
- Semantic resolver patching remains a full-collection copy path, but the
  generic `to_dict() -> from_dict()` object clones and generic `safe_jsonable()`
  calls found in this audit have been replaced with explicit field copies and
  patch-local serializers.
- MCP graph tools still bypass compact snapshot handles. `mcp/server.py` loads
  a full `IRSnapshot` for graph tools, and `mcp/graph_tools.py` rebuilds
  NetworkX graphs from that snapshot per request while resolving symbols
  through repeated linear scans.
- Query snapshot serving now uses compact snapshot symbol-index sidecars for
  symbol-index registration when available. Legacy snapshots without the
  sidecar still fall back to full snapshot load.
- `FastCode.resolve_snapshot_symbol()` and `FastCode.find_symbol()` also use the
  compact sidecar for alias maps and single-symbol records on current snapshots.
- Local repository loading is copy-minimal by default for local paths:
  `RepositoryLoader.load_from_path()` now indexes the caller-provided tree
  in place unless `repository.local_source_mode: "copy"` is requested.
  Content-addressed or hardlinking copy mode remains open for explicit
  workspace-copy use cases.
- Vector preallocation is still partial. Shard load/write paths, in-memory
  vector-row append, and homogeneous `as_float32_matrix()` sequence conversion
  avoid the old row-list/`vstack` pattern, but memory-mapped shard handles and
  lazy unchanged-shard publication are still open.
- PostgreSQL relational fact persistence remains whole-snapshot. The row-at-a-time
  insert baseline found in this audit has been replaced with per-table batched
  inserts on the full-rebuild path, but delta delete/upsert by changed path is
  still open.

## P0 - Make Incremental Update The Execution Model

### P0.1 Canonical File Inventory And Fingerprint Planner

**Gap:** file identity is recomputed by multiple stages instead of being carried
as one canonical planner object.

Evidence:

- `RepositoryLoader.get_repository_info()` calls `scan_files()` after the
  pipeline already calls `scan_files()` (`fastcode/src/fastcode/indexing/loader.py:350`,
  `fastcode/src/fastcode/indexing/pipeline.py:2402`).
- `RepositoryLoader.scan_files()` returns path, relative path, size, and
  extension, but not content hash or git blob identity
  (`fastcode/src/fastcode/indexing/loader.py:250`).
- `IndexPipeline._file_fingerprint()` hashes files for incremental manifests
  (`fastcode/src/fastcode/indexing/pipeline.py:1138`), and
  `build_ir_from_ast()` hashes the same files again for file units
  (`fastcode/src/fastcode/scip/ast_adapter.py:84`).

TODO:

- [ ] Introduce a typed `FileInventory` / `FileFingerprint` planner record that
  carries normalized path, size, mtime, content hash, git blob oid when
  available, language, package root, and supported-tool eligibility.
- [x] Build it once per index run and pass it through snapshot identity,
  incremental planning, AST extraction, SCIP scope, artifact reuse, IR file
  units, and publication.
- [ ] Prefer git tree/blob identities when available, with content hashing as a
  fallback for untracked or non-git inputs.
- [x] Remove duplicate scan/hash calls from repository info, manifest diffing,
  AST IR construction, and artifact manifest publication.
- [x] Add a regression that fails when a one-file update hashes unchanged files
  more than once per run.

Exit criteria:

- one file inventory is visible in pipeline metrics
- unchanged files do not perform repeated content reads or hashes in a one-file
  body edit benchmark
- file counts and total sizes are derived from the shared inventory, not a
  second scan

### P0.2 Replace Full Temporary Artifacts With Delta-First Builders

**Gap:** shard reuse currently happens at publication time after full temporary
objects have already been built.

Evidence:

- `_plan_incremental_elements()` reconstructs unchanged `CodeElement` objects
  and returns `all_elements = unchanged_elements + new_elements`
  (`fastcode/src/fastcode/indexing/pipeline.py:1633`).
- `run_index_pipeline()` still materializes all elements into a temporary
  vector store, builds a full legacy graph, and builds full BM25 over `elements`
  (`fastcode/src/fastcode/indexing/pipeline.py:2528`).
- Incremental vector/BM25/graph shard reuse happens later during artifact
  persistence (`fastcode/src/fastcode/indexing/pipeline.py:3119`).

TODO:

- [ ] Split indexing into unchanged artifact handles plus changed-file deltas,
  so unchanged paths are not reconstructed into full `CodeElement` objects for
  vector/BM25/legacy graph staging.
- [ ] Add vector-store APIs that publish a new snapshot from previous shard
  handles plus changed matrix rows, without first building a full temporary
  matrix.
- [ ] Add lexical index APIs that publish from previous shard handles plus
  changed token rows, without constructing a full `BM25Okapi` corpus in memory.
- [ ] Add graph publication APIs that update adjacency shards for affected
  paths and affected cross-file edges only.
- [ ] Keep a compatibility path for full rebuilds, but record degraded/full
  fallback reasons in metrics.

Exit criteria:

- body-only one-file update does not instantiate full vector, BM25, or legacy
  graph builders for unchanged files
- artifact metrics include changed rows, reused rows, copied/linked shards,
  written shards, deleted shards, and fallback reason
- benchmark proves one-file update cost is materially below full reindex for
  parse, embedding, vector, BM25, graph, persistence, and peak RSS

### P0.3 Incremental IR Snapshot And Relational Fact Persistence

**Gap:** changed AST IR is merged with the previous snapshot, but persistence
still rewrites whole-snapshot surfaces.

Evidence:

- Incremental AST IR narrows to changed paths before `build_ir_from_ast()`
  (`fastcode/src/fastcode/indexing/pipeline.py:2580`).
- The previous full snapshot is loaded and merged in memory
  (`fastcode/src/fastcode/indexing/pipeline.py:2920`).
- `apply_incremental_update()` merges list-shaped units, supports, relations,
  and embeddings (`fastcode/src/fastcode/indexing/incremental.py:344`).
- `save_snapshot()` writes one full `ir_snapshot.json`
  (`fastcode/src/fastcode/store/snapshot.py:963`).
- `save_relational_facts()` deletes and reinserts all documents, symbols,
  occurrences, edges, and attachments for the snapshot
  (`fastcode/src/fastcode/store/snapshot.py:1694`).
- `IRGraphBuilder.build_graphs()` and `save_ir_graphs()` rebuild and persist
  whole IR graph payloads during indexing
  (`fastcode/src/fastcode/indexing/pipeline.py:3248`).

TODO:

- [ ] Introduce per-file or per-unit IR artifact shards with a snapshot manifest
  that maps paths and stable unit ids to shard references.
- [ ] Save changed IR shards and metadata deltas instead of rewriting a whole
  `ir_snapshot.json` on every incremental update.
- [ ] Add relational fact delta operations: delete removed/modified-path facts,
  upsert changed facts, preserve unchanged rows, and update snapshot-level
  manifest tables atomically.
- [x] Batch PostgreSQL fact writes with `executemany`, `COPY`, or a backend
  equivalent on full rebuild paths; row-at-a-time fact insertion should not be
  the release-grade baseline.
- [ ] Build IR graph deltas from changed relations and only rebuild global
  derived views when edge changes cross declared invalidation thresholds.
- [ ] Preserve source-owned evidence with explicit per-source invalidation
  contracts, not only path-level tombstone/relink heuristics.

Exit criteria:

- one-file update writes only changed IR/fact/graph shards plus compact
  manifests
- full relational delete/insert is reserved for forced rebuilds and schema
  migrations
- benchmark captures changed row counts and database write amplification

### P0.4 Tool And SCIP Scope Must Be Artifact-Native

**Gap:** scoped SCIP is useful, but it still materializes package copies and can
fall back to package or repo-scale tool work.

Evidence:

- `_incremental_scip_scope()` can skip or return package scope, but not a true
  per-file artifact reuse plan (`fastcode/src/fastcode/indexing/pipeline.py:1800`).
- `_run_scoped_scip_frontier()` copies scope roots with `shutil.copytree()`,
  runs language indexers per scope/language, then filters resulting artifacts
  to target paths (`fastcode/src/fastcode/indexing/pipeline.py:2113`).
- Full fallback language detection and indexer execution still exist in the
  active pipeline (`fastcode/src/fastcode/indexing/pipeline.py:2710`).
- Helper-backed semantic resolvers narrow helper target files, but still start
  external helper processes per resolver run
  (`fastcode/src/fastcode/semantic/resolvers/helper_backed.py:138`).

TODO:

- [x] Cache scoped SCIP output by language, package root, tool profile,
  dependency/package-marker fingerprint, and target file fingerprints.
- [ ] Extend the same artifact-native cache contract to helper-backed semantic
  tools and unsupported/widened SCIP tool surfaces.
- [x] Reuse previous scoped SCIP facts for unchanged file/package scopes instead
  of rerunning the scoped indexer.
- [x] Replace temporary copied package roots with repo-root filtered execution
  by default; keep copied roots only as an explicit compatibility mode.
- [ ] Persist explicit degraded metadata when unsupported, widened, or
  dependency-frontier changes require a full tool rerun.
- [ ] Add edit-class benchmarks: body-only, signature/API, import/dependency,
  package manifest, file delete, and rename.

Exit criteria:

- scoped tool work reports artifact cache hits/misses by language and package
  root
- unchanged packages are not copied or re-indexed in package-local body edits
- full repo tool rerun is visible as an explicit degraded mode with a reason

## P0 - Make Vector Flow Copy-Minimal

### P0.5 First-Class Embedding Fingerprint Contract

**Gap:** embedding cache keys are model-aware, but embedding identity is still
mostly local to `CodeEmbedder`.

Evidence:

- `_embedding_cache_key()` includes provider, model, dimension, sequence length,
  normalize flag, Ollama URL, and cache version
  (`fastcode/src/fastcode/indexing/embedder.py:152`).
- `_incremental_compatibility_payload()` repeats a subset of embedding identity
  fields inside the pipeline (`fastcode/src/fastcode/indexing/pipeline.py:1156`).
- Embedding vectors are persisted in multiple surfaces: vector store shards,
  IR snapshot embeddings, PG rows, repository overviews, query embeddings, and
  cache payloads.

TODO:

- [x] Add a typed `EmbeddingFingerprint` value owned by the embedding boundary.
- [ ] Persist the same fingerprint in file manifests, vector manifests, IR
  embeddings, PG metadata, repository overview artifacts, query embedding cache
  entries, and incremental compatibility checks.
- [x] Make fingerprint lookup non-starting: compatibility planning and cache-hit
  validation must not load a sentence-transformer model or probe Ollama merely
  to learn an embedding dimension.
- [ ] Make embedding reuse depend on fingerprint plus prepared-text hash, not
  ad hoc local key construction.
- [x] Add an explicit prepared-text schema version to the fingerprint so
  `_prepare_code_text()` changes invalidate cache entries without relying on
  operator-managed `cache_version`.
- [ ] Add tests that patch stale fingerprint surfaces and prove reuse is refused
  consistently.

Exit criteria:

- all embedding-bearing artifacts expose the same fingerprint fields
- model/provider/config changes invalidate every embedding reuse path
- body-only updates reuse unchanged embeddings without revalidating through
  unrelated serializers

### P0.6 Lazy Embedder Startup And Provider Batching

**Gap:** embedding setup is lazy at construction, and the compatibility/cache-hit
startup gaps found in the follow-up audit have been fixed. Provider batching is
still incomplete: the Ollama path embeds one text per HTTP request.

Evidence:

- Historical fixed evidence: `IndexPipeline._incremental_compatibility_payload()`,
  `CodeEmbedder.embedding_fingerprint_record()`, and
  `CodeEmbedder._get_cached_embedding()` previously touched `embedding_dim`
  early enough to start providers before embedding work.
- `_embed_batch_uncached()` calls `_embed_text_ollama()` once per text for the
  Ollama provider (`fastcode/src/fastcode/indexing/embedder.py:116`).

TODO:

- [x] Defer model load and Ollama dimension probing until the first operation
  that actually needs embeddings.
- [x] Persist configured embedding dimension when available so non-embedding
  paths do not require provider startup.
- [x] Stop `_incremental_compatibility_payload()` and cache-hit validation from
  touching `embedding_dim` when no configured or persisted dimension is
  available.
- [ ] Add provider-level batch APIs where supported, and bounded concurrency
  where only per-text APIs exist.
- [ ] Expose provider startup time, request count, batch count, and cache hit
  count in pipeline metrics.

Exit criteria:

- metadata-only and cache-load flows do not load/probe embedding providers
- Ollama indexing reports fewer provider calls than texts when batch/concurrent
  mode is configured
- benchmarks separate provider time from local pipeline materialization time

### P0.7 Remove Legacy List-Vector Paths

**Gap:** active compatibility paths still collect embeddings into Python lists
and convert with `np.array()`.

Evidence:

- `FastCode.index_repository()` builds `vectors: list[Any]` and converts with
  `np.array(vectors)` (`fastcode/src/fastcode/main/fastcode.py:420`).
- multi-repository indexing repeats the list-to-array path
  (`fastcode/src/fastcode/main/fastcode.py:1564`).
- `as_float32_matrix()` is available as the explicit vector boundary helper
  (`fastcode/src/fastcode/utils/vectors.py:41`).

TODO:

- [x] Route every vector-store insertion through `as_float32_matrix()` with an
  explicit copy policy.
- [ ] Remove or quarantine old direct index paths that bypass the snapshot
  pipeline and duplicate vector/BM25/graph staging.
- [ ] Add tests that fail when hot vector insertion paths use raw
  `np.array(vectors)` or list materialization.

Exit criteria:

- no active indexing path converts embedding lists with raw `np.array()`
- copy policy is visible at every vector-store, pgvector, and cache boundary

### P0.8 Vector Shards: Preallocate, Memory Map, Avoid Row Loops

**Gap:** vector storage uses native arrays, and the original row-list/`vstack`
load/write/append hot spots are largely fixed. Large shards are still not
memory-map-native, and unchanged-shard publication still lacks lazy handles.

Evidence:

- Historical fixed evidence: `_append_vector_rows()`,
  `_vector_matrix_ordered_by_sequences()`, `_write_vector_bundle()`,
  `_write_vector_bundle_with_sequences()`, `load_vector_payload()`, and the
  homogeneous sequence path in `as_float32_matrix()` previously grew matrices
  through row lists, `np.vstack()`, or `.tolist()`.
- shard bytes are written with `np.savez_compressed()`, which favors compact
  artifacts over direct memory mapping (`fastcode/src/fastcode/store/vector.py:1206`).

TODO:

- [x] Preallocate destination matrices from sequence plans and fill by slice or
  index arrays instead of row-list plus `vstack`.
- [x] Replace `VectorStore._append_vector_rows()` growth-by-`vstack` with a
  planned append buffer or direct backend insertion path.
- [x] Give `as_float32_matrix()` a shape-aware preallocation path for
  homogeneous vector sequences so approved helper usage does not hide a row-list
  materialization cost.
- [ ] Store shard vectors in a format that supports memory mapping for large
  shards, or make compression an explicit tradeoff controlled by config.
- [x] Keep sequence numbers as arrays and avoid `.tolist()` during hot loads.
- [ ] Support lazy shard handles for search and publication so unchanged shards
  are not loaded merely to publish a new snapshot.
- [ ] Add allocation benchmarks around vector append, incremental save,
  incremental load, and search on small and medium repositories.

Exit criteria:

- unchanged vector shards can be linked into a new snapshot without loading
  vectors into Python
- changed vector shards write through contiguous matrix slices
- peak allocation for vector load/save scales with changed rows, not full rows,
  in incremental update benchmarks

### P0.9 Enforce Native pgvector Boundaries

**Gap:** PG retrieval stores vectors through vector-specific columns, but JSON
fallbacks can still list-materialize arrays if embeddings leak into metadata.

Evidence:

- `_json_safe_payload()` recursively converts NumPy arrays with `.tolist()`
  (`fastcode/src/fastcode/store/pg_retrieval.py:216`).
- `upsert_elements()` serializes metadata JSON per element and binds vector
  parameters separately (`fastcode/src/fastcode/store/pg_retrieval.py:258`).

TODO:

- [x] Make array-in-JSON a hard error on hot PG upsert paths, except for
  explicitly marked compatibility exports.
- [ ] Use batched insert/update APIs for PG vector and search-document rows
  instead of one `execute()` pair per element.
- [ ] Preserve embeddings only in vector columns or vector artifacts; metadata
  should carry embedding refs and fingerprints, not numeric arrays.

Exit criteria:

- tests fail if an embedding array reaches metadata JSON serialization during
  active PG upsert
- PG upsert metrics report row count, batch count, and vector adapter path

## P0 - Minimize Graph And Object Materialization

### P0.10 Replace NetworkX In Hot Graph Paths With `igraph`/`IRGraphView`

**Gap:** compact graph persistence exists, but active algorithms still build or
convert to NetworkX in multiple hot paths.

Evidence:

- `IRGraphView` is backed by `python-igraph`, but `copy()` and
  `to_undirected()` materialize NetworkX graphs
  (`fastcode/src/fastcode/ir/graph.py:20`).
- `IRGraphView.reachable_within()` and `distances_within()` now use bounded
  traversal, but shortest path, component stats, and undirected view helpers
  are still incomplete without NetworkX conversion.
- `graph/build.py` keeps lazy adjacency payloads but materializes NetworkX for
  path/stats/merge compatibility (`fastcode/src/fastcode/graph/build.py:220`).
- `retrieval/hybrid.py` uses `IRGraphView.union()` when compact views are
  available, but falls back to NetworkX expansion otherwise
  (`fastcode/src/fastcode/retrieval/hybrid.py:278`).
- `mcp/graph_tools.py` still calls NetworkX graph algorithms directly. Main
  composition-root callees/callers/dependencies now use compact bounded graph
  traversal when graph artifacts are available.

TODO:

- [x] Add bounded `IRGraphView` reachability/distance traversal for frontier
  graph questions.
- [ ] Add `IRGraphView` methods for shortest path, component stats, degree,
  neighbor iteration, and undirected views without NetworkX conversion.
- [x] Change active retrieval graph expansion to use compact reachability when
  compact graph handles exist.
- [ ] Change MCP graph helpers and retrieval compatibility fallback paths to
  avoid direct `nx.*` calls where compact graph handles exist.
- [x] Change main composition-root callees/callers/dependencies helpers to use
  bounded compact graph traversal instead of direct NetworkX traversal.
- [ ] Keep NetworkX only for explicit compatibility/export/debug surfaces.
- [ ] Add an architecture/perf guard that fails when new hot-path graph code
  imports NetworkX outside approved modules.

Exit criteria:

- query-time graph expansion on compact IR graphs does not materialize a
  NetworkX union graph
- cutoff reachability cost scales with the requested frontier, not all nodes
- NetworkX import locations are documented as compatibility boundaries

### P0.11 Rewrite Projection Graph Algorithms Around Native Graph Handles

**Gap:** projection building constructs NetworkX graphs even when IR graphs are
already compact, then converts to `igraph` only for part of clustering.

Evidence:

- `ProjectionTransformer.build()` constructs weighted undirected and directed
  NetworkX graphs (`fastcode/src/fastcode/indexing/projection_transform.py:94`).
- `_build_weighted_graph()` and `_build_directed_weighted_graph()` add snapshot
  and IR graph edges into NetworkX (`fastcode/src/fastcode/indexing/projection_transform.py:279`,
  `fastcode/src/fastcode/indexing/projection_transform.py:358`).
- scope BFS, Steiner tree fallback, greedy modularity, PageRank, and centrality
  use NetworkX (`fastcode/src/fastcode/indexing/projection_transform.py:404`).
- Leiden clustering converts the NetworkX graph to `igraph` only after the
  NetworkX graph has already been built (`fastcode/src/fastcode/indexing/projection_transform.py:532`).

TODO:

- [ ] Define a projection-native graph representation backed by `igraph` or
  `IRGraphView` plus compact side tables for node attributes and edge weights.
- [ ] Implement scope BFS, hub compression, PageRank/centrality, and Leiden
  directly on the native representation.
- [ ] Replace NetworkX Steiner usage with a bounded native approximation or
  explicitly mark query-scope Steiner as a compatibility fallback.
- [ ] Benchmark NetworkX vs `igraph` on representative projection scopes before
  locking the implementation.

Exit criteria:

- projection build does not construct a full NetworkX graph on the primary path
- graph-engine choice is backed by benchmark output
- projection memory use is measured for snapshot, entity, and query scopes

### P0.12 Replace All-Pairs IR Merge Matching

**Gap:** AST/SCIP alignment does all-pairs candidate scoring and then uses
NetworkX max-weight matching.

Evidence:

- `_select_matches()` loops over every AST unit and every SCIP unit, then builds
  a NetworkX graph for matching (`fastcode/src/fastcode/ir/merge.py:221`).
- `merge_ir()` clones list-shaped units, supports, relations, and embeddings for
  whole snapshots (`fastcode/src/fastcode/ir/merge.py:338`).
- Existing IR merge benchmarks are baselines, not budget gates
  (`fastcode/tests/benchmarks/bench_ir_merge.py`).

TODO:

- [ ] Bucket candidates by path, kind, normalized name, stable unit id, and span
  before scoring.
- [ ] Cap candidate fanout per unit and record when candidates are dropped or
  widened.
- [ ] Replace NetworkX matching with a smaller native or specialized matching
  implementation, or prove NetworkX is not the bottleneck after candidate
  pruning.
- [ ] Avoid whole-snapshot clone work when merging changed-path deltas.

Exit criteria:

- merge candidate count is near-linear in units for normal repositories
- benchmark reports candidate pairs, selected pairs, match time, clone time, and
  peak allocation
- one-file updates do not rerun all-pairs matching for unchanged files

### P0.13 Snapshot, IR Graph, And Embedding Persistence Should Be Sharded

**Gap:** persistence still serializes large whole-snapshot JSON payloads and
list-shaped embedding vectors.

Evidence:

- `_snapshot_file_payload()` builds full JSON lists for units, supports,
  relations, embeddings, and metadata (`fastcode/src/fastcode/store/snapshot.py:342`).
- `_embedding_payload()` stores vectors as JSON lists
  (`fastcode/src/fastcode/store/snapshot.py:329`).
- `save_ir_graphs()` writes graph JSON payloads
  (`fastcode/src/fastcode/store/snapshot.py:1069`).
- `load_snapshot()` loads one whole snapshot payload
  (`fastcode/src/fastcode/store/snapshot.py:1154`).

TODO:

- [ ] Split snapshot persistence into manifests plus unit/relation/support and
  embedding shards.
- [ ] Store embedding vectors in NumPy or vector-store artifacts referenced from
  IR embedding records instead of JSON lists.
- [ ] Store IR graph edges as compact typed arrays or adjacency shards, with
  JSON only for small metadata manifests.
- [ ] Add lazy snapshot readers for metadata, path/unit subsets, relations, and
  embeddings.

Exit criteria:

- loading snapshot metadata does not load unit/relation/embedding arrays
- changed-path update rewrites only affected snapshot shards
- IR snapshot embedding JSON lists are removed from active persistence

### P0.14 Lexical Retrieval Must Stop Rebuilding Full BM25 On Load

**Gap:** BM25 artifacts can be sharded, but query load reconstructs a full
`BM25Okapi` object from materialized corpus and element lists.

Evidence:

- `index_for_bm25()` builds a full corpus and `BM25Okapi`
  (`fastcode/src/fastcode/retrieval/hybrid.py:220`).
- `load_bm25()` reloads all corpus and element payloads and rebuilds
  `BM25Okapi` (`fastcode/src/fastcode/retrieval/hybrid.py:1853`).
- `_load_bm25_payload()` materializes ordered corpus and element lists from
  shards (`fastcode/src/fastcode/retrieval/hybrid.py:2253`).

TODO:

- [ ] Choose and implement a shard-native lexical index strategy: incremental
  BM25 statistics, a compact inverted index, or an embedded search engine.
- [ ] Make query-time lexical retrieval read only needed postings/statistics,
  not every element payload.
- [ ] Preserve current BM25 output semantics with golden ranking tests during
  migration.

Exit criteria:

- loading lexical retrieval for a snapshot does not rebuild a full corpus object
- one-file update changes only lexical shards for affected paths and global
  statistics
- query benchmark reports lexical load time separately from ranking time

### P0.15 Public Query Paths Must Not Serialize All Reads Behind The State Lock

**Gap:** request-local artifact handles exist, but public query entrypoints still
serialize reads behind the service state lock.

Evidence:

- `FastCode.query_snapshot()` and `FastCode.query()` hold `_state_lock()` around
  the full query call (`fastcode/src/fastcode/main/fastcode.py:1000`).
- `FastCode.query_stream()` holds `_state_lock()` for the generator duration
  (`fastcode/src/fastcode/main/fastcode.py:1113`).

TODO:

- [ ] Split mutation locks from read locks and serve immutable loaded artifact
  handles without serializing independent queries.
- [ ] Keep load/index/delete/publish operations fenced, but let queries share a
  read snapshot handle.
- [x] Build query-time symbol indexes from compact symbol-index sidecars instead
  of full-loading `IRSnapshot` solely to register aliases when current snapshot
  artifacts are available.
- [x] Serve `FastCode.find_symbol()` from compact sidecar symbol records when
  available instead of full-loading the snapshot and scanning all symbols.
- [ ] Backfill or derive compact symbol indexes for legacy snapshots that lack
  the sidecar, preferably from relational facts when available.
- [ ] Add concurrent query benchmarks with and without background mutations.

Exit criteria:

- N concurrent snapshot queries scale better than one-at-a-time serialized
  execution
- streaming query lock duration does not include the full response generation
  path unless a mutation is active

### P0.16 Make Shell Graph Tools Artifact-Handle Native

**Gap:** MCP graph helper paths still load full snapshots and materialize
NetworkX graphs for per-request graph questions. Main composition-root
callees/callers/dependencies now use compact graph handles on the primary path.

Evidence:

- MCP graph tools call `fc.snapshot_store.load_snapshot(snapshot_id)` before
  directed path, impact, cluster, Steiner, and caller computations
  (`fastcode/src/fastcode/mcp/server.py:846`).
- `mcp/graph_tools.py` rebuilds selected graphs with
  `IRGraphBuilder().build_graphs(snapshot)` and composes NetworkX graphs per
  request (`fastcode/src/fastcode/mcp/graph_tools.py:107`).
- `resolve_unit_id()` and `format_path_node()` repeatedly scan `snapshot.units`
  linearly (`fastcode/src/fastcode/mcp/graph_tools.py:54`).
- `FastCode.get_graph_callees()`, `get_graph_callers()`, and
  `get_graph_dependencies()` now use bounded compact graph traversal when graph
  artifacts are available, with compatibility fallback only when compact graph
  handles are missing.

TODO:

- [ ] Route shell graph tools through `LoadedSnapshotArtifacts` and compact
  `IRGraphView`/fact indexes when those artifacts exist.
- [ ] Add per-snapshot symbol lookup maps for MCP graph tools instead of
  repeated linear scans over full snapshot units.
- [x] Route main composition-root callees/callers/dependencies helpers through
  compact graph handles.
- [ ] Keep full `IRSnapshot` + NetworkX rebuild only as an explicit
  compatibility fallback with materialization metrics and degraded reason.
- [ ] Add MCP graph-tool regression tests that patch `load_snapshot()` or
  `IRGraphBuilder.build_graphs()` to fail when compact handles are available.

Exit criteria:

- directed path, impact, caller, and dependency queries do not full-load
  snapshots on the primary path
- graph tool cost scales with selected frontier/path size rather than full
  snapshot node count

### P0.17 Avoid Whole-Tree Copies Before Incremental Planning

**Gap:** explicit local workspace-copy mode still uses a whole-tree copy before
the incremental planner can decide what changed. Default local path indexing is
now read-only and in-place.

Evidence:

- `RepositoryLoader.load_from_path()` now defaults to
  `repository.local_source_mode: "in_place"` for local paths and only calls
  `shutil.copytree(...)` when explicit copy mode is requested
  (`fastcode/src/fastcode/indexing/loader.py:157`).

TODO:

- [x] Add a read-only source checkout mode that scans and fingerprints the
  caller-provided repository in place when mutation isolation is not required.
- [x] Report local load mode, workspace-copy status, copied bytes, and copied
  file counts from repository loading.
- [ ] When a workspace copy is required, use a content-addressed or hardlinking
  copy strategy and report linked bytes in pipeline metrics.
- [ ] Feed the canonical file inventory directly from the source checkout into
  incremental planning before any full-tree copy when explicit copy mode is
  requested. The default in-place mode already scans the source tree directly.

Exit criteria:

- repeated local indexing of a large unchanged repository does not copy the
  whole working tree merely to discover that no indexed files changed
- pipeline metrics distinguish scanned bytes, hashed bytes, copied bytes, and
  linked bytes

### P0.18 Make Semantic Patch Application Delta-Native

**Gap:** semantic resolver patches still copy whole snapshot collections before
applying a usually small patch, although the generic dict round trips have been
removed.

Evidence:

- `apply_resolution_patch()` still clones all units, supports, embeddings, and
  relations through explicit helper copies
  (`fastcode/src/fastcode/semantic/resolvers/patching.py:41`,
  `fastcode/src/fastcode/semantic/resolvers/patching.py:196`).
- Patch metadata and unit metadata updates now pass through a patch-local
  serializer, but this is still recursive metadata normalization during the
  resolver hot path
  (`fastcode/src/fastcode/semantic/resolvers/patching.py:37`).

TODO:

- [ ] Replace full snapshot cloning with structural sharing or path/unit-scoped
  copy-on-write updates for units, supports, relations, and embeddings.
- [x] Avoid generic IR object clones and generic JSON conversion in resolver
  patch application.
- [ ] Replace patch-local recursive metadata normalization with explicit
  metadata serializers for resolver patch payloads.
- [ ] Add runtime materialization counters around semantic patch application.
- [ ] Benchmark helper-backed semantic upgrade on unchanged, body-only,
  signature/API, and inheritance-change edit classes.

Exit criteria:

- applying a small resolver patch does not copy every unchanged IR unit and
  relation
- materialization metrics report changed IR objects separately from preserved
  objects

## P1 - Enforcement And Benchmark Evidence

### P1.1 Performance Envelope By Edit Class

TODO:

- [ ] Add benchmark fixtures for body-only edit, signature/API edit,
  import/dependency edit, package manifest edit, delete, rename, and new file.
- [ ] For each fixture, record full reindex cost and incremental update cost.
- [ ] Track wall time, provider calls, files scanned, files hashed, bytes read,
  bytes written, changed vectors, reused vectors, BM25 shards, graph shards,
  database rows, peak RSS, and Python allocation peaks.
- [ ] Store benchmark reports in a repeatable artifact format that can be
  compared across commits.

Exit criteria:

- performance claims are tied to fixture output, not source inspection alone
- one-file edit budgets are visible and fail CI when a regression exceeds the
  accepted threshold

### P1.2 Materialization Boundary Guards

TODO:

- [x] Extend architecture tests to guard hot paths against `safe_jsonable()`,
  generic `to_dict()` / `from_dict()` round trips, `row_to_dict()`, `.tolist()`,
  and raw `np.array(vectors)` unless the call site is annotated as an allowed
  boundary.
- [x] Expand the guard allowlist/scope to cover semantic patching.
- [ ] Expand the guard allowlist/scope to cover MCP graph helpers, projection
  transforms, snapshot persistence, and query-time compact symbol-index
  registration; the current guard only covers a subset of hot materialization
  paths.
- [x] Add runtime counters for explicit materialization boundaries: JSON encode,
  JSON decode, pickle load/dump, NetworkX conversion, vector list conversion,
  snapshot full load, and graph full load.
- [x] Add tests that patch old generic conversion helpers to raise in active
  indexing, retrieval, and persistence paths.

Exit criteria:

- new materialization points require an explicit boundary annotation
- hot-path tests fail if old generic conversion shortcuts return

### P1.3 Graph Engine Decision Record

TODO:

- [ ] Benchmark NetworkX and `igraph` for IR graph build, cutoff reachability,
  shortest path, projection scope, clustering, PageRank/centrality, and merge
  matching workloads.
- [ ] Record memory use and wall time for small, medium, and large synthetic
  snapshots plus at least one real representative repository.
- [ ] Decide the canonical hot-path graph engine based on data.
- [ ] Keep compatibility exporters isolated after the decision.

Exit criteria:

- the NetworkX-to-`igraph` migration is justified by measured data
- modules do not immediately convert native graph handles back to NetworkX on
  the primary path

### P1.4 Allocation And IO Profiling For Index Pipeline

TODO:

- [ ] Add opt-in pipeline profiling that reports allocation peaks by stage.
- [ ] Track temporary directories, copied bytes, hard-linked bytes, and deleted
  bytes for scoped tool runs and artifact publication.
- [ ] Track snapshot store bytes written for IR, graph, vector, lexical, unit
  artifact, PG, and relational fact surfaces.

Exit criteria:

- incremental update reports enough data to explain whether cost is CPU,
  provider, graph, serialization, database, or filesystem dominated
- regressions can be attributed to a stage without manual profiling

## Do Not Count These As Done

- Shard reuse after full temporary artifact construction does not satisfy
  delta-first execution.
- Compact graph persistence does not satisfy graph materialization minimization
  if a later hot path converts back to NetworkX.
- Buffer-backed embedding cache entries do not satisfy zero-copy vector flow if
  active paths still create list vectors, JSON vector lists, or full stacked
  matrices for unchanged data.
- Baseline benchmark tests do not satisfy performance gates unless they enforce
  budgets or produce comparable reports used in release decisions.
