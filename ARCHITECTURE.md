# FastCode Architecture

This document describes the architecture currently implemented in `develop`.
It is intentionally narrower than older README or marketing language.

## Status

FastCode is in a hardened pre-release state.

The core indexing, IR merge, retrieval, semantic-upgrade, and shell boundaries
are materially stronger than the original prototype, but the stable-release bar
is not met yet. The main remaining gap is operational and dataflow hardening,
not basic pipeline existence.

The maintained implementation tracker is
[IMPLEMENTATION_TODOS.md](./IMPLEMENTATION_TODOS.md).

## Design goals

FastCode is a repository-understanding system for coding agents with four core
goals:

1. Build canonical code facts once and reuse them across retrieval paths.
2. Keep the functional core test-heavy and free from shell concerns.
3. Make settings and environment access explicit at boundaries.
4. Support a package layout that can keep growing without collapsing imports.

## Package layout

The importable package is `fastcode/src/fastcode/`.

Current top-level modules:

- `api/` HTTP and web transport shells
- `graph/` graph construction primitives
- `indexing/` repository loading, extraction orchestration, projections
- `ir/` canonical frozen IR types and helpers
- `main/` composition root and CLI wiring
- `mcp/` MCP transport shell
- `query/` query understanding and answer generation orchestration
- `retrieval/` retrieval orchestration and iterative agent behavior
- `retrieval/core/` pure retrieval logic and scoring helpers
- `schemas/` boundary models and typed config
- `scip/` SCIP adapters and loaders
- `semantic/` semantic resolver registry and helper-backed upgrades
- `store/` persistence orchestration
- `store/infrastructure/` lower-level storage adapters
- `utils/` compatibility helpers and shared utilities

## Enforced layer DAG

The import graph is enforced by
`fastcode/tests/architecture/test_import_graph.py`.

Layers:

- Layer 0:
  - `schemas`
  - `ir`
  - `utils`
  - `retrieval.core`
  - `store.infrastructure`
- Layer 1:
  - `graph`
  - `indexing`
  - `query`
  - `retrieval`
  - `scip`
  - `semantic`
  - `store`
- Layer 2:
  - `api`
  - `mcp`
  - `main`

Rule: lower layers must not import upward, and cross-layer cycles are forbidden.

## Architecture contract

The repo currently enforces these rules in code, tests, and module-local
`ruff.toml` files:

1. Pydantic stops at the boundary.
   - `schemas/` owns Pydantic validation and settings parsing.
   - `ir/`, `graph/`, and `retrieval/core/` remain Pydantic-free.

2. Pure packages stay shell-free.
   - `ir/`, `graph/`, and `retrieval/` may not pull in `pydantic`,
     `sqlite3`, `subprocess`, or `urllib`.

3. Package roots stay thin.
   - `fastcode/__init__.py` and subpackage `__init__.py` are compatibility
     surfaces, not composition roots.
   - Internal packages must not rely on `from fastcode import ...` re-exports.

4. Translation stays explicit.
   - Shell packages must not mass-assign with `**model_dump()` or `**__dict__`.
   - Mapping between API payloads, runtime config, IR records, and store records
     must be visible in code.
   - Active hot paths should prefer typed records or explicit field serializers
     over generic `to_dict()`, `from_dict()`, `row_to_dict()`, or recursive
     `safe_jsonable()` cleanup.

5. Inner packages do not read environment directly.
   - `indexing`, `query`, `retrieval`, `store`, `mcp`, `schemas`, and related
     inner packages must not call `os.getenv`, `os.environ[...]`, or
     `load_dotenv()` directly.
   - Environment loading is centralized in config preparation.

## Runtime configuration flow

The current config boundary is:

1. Raw YAML and `.env` input enter through `fastcode.utils._compat`.
2. `prepare_runtime_config_mapping(...)` resolves paths and overlays env-backed
   runtime settings.
3. `fastcode.schemas.config` validates that mapping with Pydantic boundary
   models and returns frozen dataclass runtime config as `FastCodeConfig`.
4. `fastcode.main.fastcode.FastCode` owns the runtime config instance and
   exposes explicit mutation points for runtime-only overrides.
5. Legacy dict consumers still receive a compatibility dict view, but the
   canonical config is typed and frozen.

Important current rule:

- direct environment reads should happen only in config preparation code, not in
  query, indexing, store, or MCP leaf modules.

## Data model

### Canonical IR

The deepest internal truth for code structure is the IR layer under `ir/`.
Higher-level graph, retrieval, and storage representations derive from it.

The IR path is designed to support:

- repository snapshots
- file and document identity
- symbol identity and relationships
- provenance across extraction paths
- merge of AST-derived and SCIP-derived facts

### Typed boundaries

The codebase is moving away from dict-heavy shell payloads toward typed records.
Already-landed areas include:

- frozen runtime config in `schemas/config.py`
- typed snapshot and manifest records
- typed store-facing record flows in parts of persistence
- explicit `CodeElement` serializers for retrieval, graph persistence, and
  index-storage boundaries
- explicit snapshot-file serializers in `store/snapshot.py` instead of routing
  persistence through `IRSnapshot.to_dict()` / `IRSnapshot.from_dict()`

Still incomplete:

- store, query, and projection records are not fully typed end to end
- several shell and persistence paths still expose raw dict payloads

## Indexing pipeline

The implemented indexing path is a layered pipeline:

1. repository load and file scan
2. structural extraction
3. optional SCIP extraction
4. canonical IR merge and validation
5. semantic upgrade by language-specific resolvers
6. graph, index, and projection persistence

Current hardened properties:

- explicit pipeline layer status, metrics, and warnings
- non-silent fallback behavior
- persisted SCIP lineage metadata
- helper-backed semantic upgrades with structural fallback
- snapshot-level regression coverage around reuse and concurrency
- explicit storage materialization for vector-store, PG retrieval, and
  unit-artifact boundaries instead of generic `CodeElement.to_dict()` expansion

### Incrementality

Current verdict: partially incremental, not fully incremental.

What already works:

- unchanged files can bypass repeated parse and embedding work via manifest-first
  planning in the active path
- embeddings are deduplicated and cached with model-aware keys
- helper-backed semantic resolvers can scope work to changed paths

What is still missing before stable-release claims:

- file-native IR shard reuse as the primary execution model
- truly incremental SCIP and tool-backed extraction
- end-to-end canonical file fingerprinting as the single planner anchor
- deterministic cache invalidation across schema, model, and tool changes

## Retrieval and query flow

The query stack is split between:

- `query/` for intent detection, rewriting, answer generation, and selection
- `retrieval/` for orchestration
- `retrieval/core/` for pure scoring, combination, graph-boundary logic, and
  context assembly

Current design:

1. query understanding extracts intent and search cues
2. retrieval combines semantic, keyword, and graph-aware evidence
3. iterative retrieval can escalate when direct retrieval is insufficient
4. answer generation consumes retrieved evidence plus provenance

Current hardened properties:

- agency-mode preserves cheap detected intent
- caller-filter handling is fixed after rerank
- semantic escalation can drive real IR-graph expansion

## Semantic upgrade flow

Semantic upgrade is a separate stage from basic parsing.

It currently supports:

- helper-backed resolvers where external language helpers exist
- graph-backed fallback resolvers for non-C-family languages
- explicit diagnostics and metrics on helper degradation
- repository-root aware helper execution instead of shell-cwd coupling

Registered resolver coverage includes:

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

Experimental SCIP language support is explicit and warning-bearing rather than
silently treated as production-complete.

## Storage architecture

Storage is split into orchestration and infrastructure:

- `store/` coordinates persistence behavior
- `store/infrastructure/` holds lower-level storage-facing code

Current backends in active use:

- SQLite and PostgreSQL-oriented runtime paths
- vector and cache stores
- projection and manifest persistence

Current design rule:

- infrastructure returns typed records and primitives where possible
- higher-level store code should not depend on API schemas or transport shells
- compatibility dict-return helpers may remain for callers, but active internal
  persistence and queue paths should reconstruct rows through explicit typed
  adapters

## Shells and entrypoints

The outer shells live in:

- `main/` CLI and composition root
- `api/` FastAPI and web entrypoints
- `mcp/` MCP server entrypoint

These packages may depend inward on the rest of the system. The inverse is not
allowed.

Current entrypoints from `fastcode/pyproject.toml`:

- `fastcode`
- `fastcode-api`
- `fastcode-mcp`
- `fastcode-web`

## Concurrency and state

The codebase has some landed hardening here, but this remains a release-risk
area.

Landed:

- `query_snapshot` serializes load-plus-query with a `threading.Lock`
- regression tests cover concurrent callers and artifact isolation
- several blocking API and web paths are offloaded with `asyncio.to_thread`

Open concerns:

- wider service-state isolation under mixed query, indexing, and mutation load
- real backend behavior under concurrent traffic

## Release blockers still open

The main remaining stable-release blockers are:

- install and packaging reproducibility from built artifacts
- workspace-root and layout stability in merged form
- true end-to-end incremental source and index caching
- broader FP/FCIS dataflow completion across store, query, and projection boundaries
- real backend and toolchain evidence across supported languages
- API and file-upload security hardening
- deployment and operations documentation

See [IMPLEMENTATION_TODOS.md](./IMPLEMENTATION_TODOS.md) for the maintained
release-gap list and acceptance criteria.

## Source of truth

Use this order of trust:

1. architecture tests and module-local lint rules
2. current code under `fastcode/src/fastcode/`
3. [IMPLEMENTATION_TODOS.md](./IMPLEMENTATION_TODOS.md)
4. this document
5. historical README or marketing claims

If this document and the tests disagree, the tests win.
