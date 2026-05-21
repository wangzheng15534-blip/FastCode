# Layer DAG Enforcement Design

Date: 2026-05-21

## Goal

Enforce a 5-layer dependency DAG in FastCode using `pytest-archon` deny rules,
following the pattern proven in the ETL project. Dependencies flow downward
only. Domain modules stay pure. Shell stages call downward into domain logic.

## Layer Model

```
Facade (4)      api, mcp, main
                    |
Application (3) indexing, query, store
                    |
Domain (2)      retrieval, graph, scip, semantic
                    |
Shared Kernel(1)    ir, schemas (small)
                    |
Foundation (0)      utils (zero-dep)
```

Rule: any layer may import from layers below. No upward imports. Same-layer
imports allowed.

### Layer Assignments

| Module | Layer | Role |
|--------|-------|------|
| `api` | Facade (4) | HTTP/web transport shell |
| `mcp` | Facade (4) | MCP transport shell |
| `main` | Facade (4) | Composition root, CLI wiring |
| `indexing` | Application (3) | Pipeline orchestration, I/O stages |
| `query` | Application (3) | Query orchestration |
| `store` | Application (3) | Persistence orchestration |
| `retrieval` | Domain (2) | Retrieval logic and scoring |
| `graph` | Domain (2) | Graph construction primitives |
| `scip` | Domain (2) | SCIP translation logic |
| `semantic` | Domain (2) | Semantic resolution |
| `ir` | Shared Kernel (1) | Canonical IR types |
| `schemas` | Shared Kernel (1) | Config, cross-cutting identity (small) |
| `utils` | Foundation (0) | Pure stdlib helpers, zero fastcode.* imports |

## Distributed Type Ownership

Each domain module owns its local types. No global schema dumping ground.

| Module | Owns |
|--------|------|
| `retrieval/contracts.py` | Scoring types, agent context records, ranking contracts |
| `graph/contracts.py` | Graph construction contracts |
| `scip/contracts.py` | SCIP model types, symbol resolution contracts |
| `semantic/contracts.py` | Resolver registry types, patch contracts |
| `store/contracts.py` | Persistence record types, serializer contracts |
| `query/contracts.py` | Query intent types, answer contracts |
| `ir/` (shared kernel) | Canonical IR: snapshot, element, symbol, relation |
| `schemas/` (shared kernel) | Only `FastCodeConfig`, runtime config, API-level shared contracts |

Domain `contracts.py` files use frozen dataclasses. No Pydantic in domain
contracts.

## Violation Fixes

Six upward imports must be broken. Every fix follows the same ETL principle:
code lives in the domain that owns it, shell calls downward, domain never
reaches up.

### Fix 1: `graph -> indexing`

**Current:** `graph` imports `indexing.call_extractor`
**Fix:** Move `call_extractor` into `graph/`. The shell (`indexing`) calls it
downward from `graph`.

### Fix 2: `scip -> indexing`

**Current:** `scip` imports `indexing.global_builder`
**Fix:** Move `global_builder` into `scip/`. The shell (`indexing`) calls it
downward from `scip`.

### Fix 3: `retrieval -> indexing`

**Current:** `retrieval` imports `indexing.embedder`
**Fix:** Move `embedder` into `retrieval/`. The shell (`indexing`) calls it
downward from `retrieval`.

### Fix 4: `retrieval <-> query` cycle

**Current:** `retrieval` imports `query.processor`, `query.selector`; `query`
imports `retrieval`
**Fix:** Extract shared types (ranking contracts, selection results) into
`retrieval/core/contracts.py` (shared kernel tier within retrieval). `query`
imports from `retrieval.core` (downward, Application -> Shared Kernel).
`retrieval` no longer imports `query`.

### Fix 5: `retrieval -> store`

**Current:** `retrieval` imports `store.pg_retrieval`, `store.vector`
**Fix:** `retrieval` stays pure. Shell stages in `indexing`/`query` persist
retrieval results via `store`. `retrieval` never imports `store`.

### Fix 6: `utils -> schemas`

**Current:** `utils` imports `schemas.config`
**Fix:** Config preparation (`prepare_runtime_config_mapping`) moves to
`schemas/_compat.py`. `utils` becomes truly zero-dep: no `fastcode.*` imports.

## Schema Distribution

The current monolithic `schemas` module gets decomposed:

- `schemas.config` stays in shared kernel (runtime config is cross-cutting)
- `schemas.core_types` domain-specific types move to their owning modules
- `schemas.api` moves to `api/` as API-layer contracts

## Enforcement: pytest-archon Deny Rules

New file: `fastcode/tests/architecture/test_layer_dag.py`

### Layer Patterns

```python
FACADE = ("fastcode.api.*", "fastcode.mcp.*", "fastcode.main.*")
APPLICATION = ("fastcode.indexing.*", "fastcode.query.*", "fastcode.store.*")
DOMAIN = ("fastcode.retrieval.*", "fastcode.graph.*", "fastcode.scip.*", "fastcode.semantic.*")
SHARED_KERNEL = ("fastcode.schemas.*", "fastcode.ir.*")
FOUNDATION = ("fastcode.utils.*",)
```

### Deny Rules

1. **domain_must_not_import_application_or_facade**
   `DOMAIN` should_not_import `APPLICATION + FACADE`

2. **domain_must_not_import_store**
   Domain modules should_not_import `fastcode.store.*`

3. **shared_kernel_must_not_import_upper_layers**
   `SHARED_KERNEL` should_not_import `DOMAIN + APPLICATION + FACADE`

4. **foundation_must_not_import_any_fastcode**
   `FOUNDATION` should_not_import `fastcode.*`

5. **application_must_not_import_facade**
   `APPLICATION` should_not_import `FACADE`

### Additional Contract Tests

- Each domain module's `contracts.py` uses frozen dataclasses, not Pydantic
- Domain modules contain no I/O imports (`sqlite3`, `subprocess`, `urllib`)

### Existing Architecture Tests

Kept (not replaced):

- `test_no_pydantic_in_core.py`
- `test_purity_gates.py`
- `test_explicit_translation.py`
- `test_settings_flow.py`
- `test_package_roots.py`
- `test_materialization_boundaries.py`

Replaced:

- `test_import_graph.py` -> `test_layer_dag.py`

## Dependency Addition

Add `pytest-archon>=0.0.7` to `fastcode/pyproject.toml` under dev
dependencies.

## What Stays the Same

- Module names (no renaming)
- Package location (`fastcode/src/fastcode/`)
- All existing non-architecture tests
- All existing architecture tests except `test_import_graph.py`

## Out of Scope

- Module renaming to match ETL naming conventions
- Protocol-based DI patterns
- Changes to public API surface
- Performance optimizations
