# Layer DAG Enforcement Design

Date: 2026-05-21

## Goal

Enforce a 7-tier dependency DAG in FastCode using `pytest-archon` deny rules,
following the pattern proven in the ETL project:

```
Facade → Shell → Events → Interface → Domain → Common → Base
```

Dependencies flow downward. Domain modules stay pure. Interface defines
Pydantic boundary models. Shell stages call downward through Interface
translations into Domain logic.

## Layer Model

```
Facade (6)      api, mcp, main
                    ↓
Shell (5)       indexing, query, store
                    ↓
Events (4)      (pipeline lifecycle events, agent context events)
                    ↓
Interface (3)   schemas (Pydantic boundary models, config, translation shapes)
                    ↓
Domain (2)      retrieval, graph, scip, semantic
                    ↓
Common (1)      ir (canonical types, shared value objects)
                    ↓
Base (0)        utils (zero-dep stdlib helpers)
```

### Layer Assignments

| Module | Layer | Role |
|--------|-------|------|
| `api` | Facade (6) | HTTP/web transport |
| `mcp` | Facade (6) | MCP transport |
| `main` | Facade (6) | Composition root, CLI wiring |
| `indexing` | Shell (5) | Pipeline orchestration, I/O stages |
| `query` | Shell (5) | Query orchestration |
| `store` | Shell (5) | Persistence I/O |
| (events) | Events (4) | Pipeline lifecycle, agent context events (extraction TBD) |
| `schemas` | Interface (3) | Pydantic config, boundary models, translation shapes |
| `retrieval` | Domain (2) | Retrieval logic and scoring |
| `graph` | Domain (2) | Graph construction primitives |
| `scip` | Domain (2) | SCIP translation logic |
| `semantic` | Domain (2) | Semantic resolution |
| `ir` | Common (1) | Canonical IR types, shared value objects |
| `utils` | Base (0) | Zero-dep stdlib helpers |

### What Each Layer Means (from ETL)

- **Facade:** composition root / presentation. Wires everything together.
- **Shell:** I/O infrastructure. Runs stages, reads/writes storage. Calls
  downward into Events, Interface, Domain, Common, Base.
- **Events:** event sourcing contracts, pipeline lifecycle events, agent
  context event envelopes. May import Interface, Common, Base.
- **Interface:** Pydantic config, typed column/record definitions, boundary
  models. The translation layer between Shell I/O and Domain types. May
  import Common, Base. Must NOT import Domain.
- **Domain:** pure business logic. No I/O, no Pydantic. Each domain module has
  its own `contracts.py` with frozen dataclasses. May import Common, Base.
- **Common:** shared value objects, cross-cutting contracts. May import Base.
- **Base:** zero-dep. `utils` may not import any `fastcode.*` package.

### Key Constraint: Interface and Domain Are Siblings

Interface and Domain do NOT import each other. They are parallel layers that
both depend on Common and Base:

```
Shell → Events → Interface ──→ Common → Base
                  Domain   ──→ Common → Base
```

Shell stages translate between Interface (Pydantic) and Domain (frozen
dataclass) types at the boundary. This is the explicit translation pattern
already enforced by `test_explicit_translation.py`.

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
| `ir/` (common) | Canonical IR: snapshot, element, symbol, relation |
| `schemas/` (interface) | `FastCodeConfig`, runtime config, API-level boundary models |

Domain `contracts.py` files use frozen dataclasses. No Pydantic in domain
contracts. Interface (`schemas/`) uses Pydantic. Shell stages explicitly
translate between the two.

## Events Layer

FastCode currently has event-like records scattered across modules:
- Pipeline status/metrics/warnings in `indexing/`
- Agent context events (turn journals, observations) in `retrieval/core/`

The Events layer extracts these into a dedicated module. Events are
cross-cutting lifecycle contracts that Shell stages emit and consume. They use
frozen dataclasses (no Pydantic), following ETL's pattern where Events sits
above Interface.

Initial Events extraction candidates:
- Pipeline stage lifecycle events (started, completed, failed, skipped)
- Indexing run diagnostics and metrics events
- Agent context lifecycle (turn started, observation appended, turn completed)

This can start small and grow. The deny rules still enforce the boundary even
if Events is initially empty or minimal.

## Violation Fixes

Upward imports must be broken. Every fix follows the same ETL principle:
code lives in the domain that owns it, shell calls downward, domain never
reaches up.

### Fix 1: `graph -> indexing` (Domain → Shell)

**Current:** `graph` imports `indexing.call_extractor`
**Fix:** Move `call_extractor` into `graph/`. Shell (`indexing`) calls it
downward.

### Fix 2: `scip -> indexing` (Domain → Shell)

**Current:** `scip` imports `indexing.global_builder`
**Fix:** Move `global_builder` into `scip/`. Shell (`indexing`) calls it
downward.

### Fix 3: `retrieval -> indexing` (Domain → Shell)

**Current:** `retrieval` imports `indexing.embedder`
**Fix:** Move `embedder` into `retrieval/`. Shell (`indexing`) calls it
downward.

### Fix 4: `retrieval <-> query` cycle (Domain ↔ Shell)

**Current:** `retrieval` imports `query.processor`, `query.selector`; `query`
imports `retrieval`
**Fix:** Extract shared types (ranking contracts, selection results) into
`retrieval/core/contracts.py` (Common tier). `query` imports from
`retrieval.core` (downward, Shell → Common). `retrieval` no longer imports
`query`.

### Fix 5: `retrieval -> store` (Domain → Shell)

**Current:** `retrieval` imports `store.pg_retrieval`, `store.vector`
**Fix:** `retrieval` stays pure. Shell stages in `indexing`/`query` persist
retrieval results via `store`. `retrieval` never imports `store`.

### Fix 6: `utils -> schemas` (Base → Interface)

**Current:** `utils` imports `schemas.config`
**Fix:** Config preparation (`prepare_runtime_config_mapping`) moves to
`schemas/_compat.py`. `utils` becomes truly zero-dep: no `fastcode.*` imports.

## Schema Distribution

The current `schemas` module becomes the Interface layer (Pydantic boundary
models only):

- `schemas.config` stays (runtime config is cross-cutting)
- `schemas.core_types` domain-specific types move to their owning domain
  modules as frozen dataclass contracts
- `schemas.api` moves to `api/` as API-layer contracts

## Enforcement: pytest-archon Deny Rules

New file: `fastcode/tests/architecture/test_layer_dag.py`

### Layer Patterns

```python
FACADE = ("fastcode.api.*", "fastcode.mcp.*", "fastcode.main.*")
SHELL = ("fastcode.indexing.*", "fastcode.query.*", "fastcode.store.*")
EVENTS = ("fastcode.events.*",)
INTERFACE = ("fastcode.schemas.*",)
DOMAIN = ("fastcode.retrieval.*", "fastcode.graph.*", "fastcode.scip.*", "fastcode.semantic.*")
COMMON = ("fastcode.ir.*",)
BASE = ("fastcode.utils.*",)
```

### Deny Rules (one test per boundary, matching ETL pattern)

1. **domain_must_not_import_facade_shell_or_interface**
   Domain should_not_import `FACADE + SHELL + INTERFACE`

2. **interface_must_not_import_domain_facade_or_shell**
   Interface should_not_import `DOMAIN + FACADE + SHELL`

3. **events_must_not_import_domain_facade_or_shell**
   Events should_not_import `DOMAIN + FACADE + SHELL`

4. **common_must_not_import_events_interface_domain_facade_or_shell**
   Common should_not_import `EVENTS + INTERFACE + DOMAIN + FACADE + SHELL`

5. **base_must_not_import_any_fastcode**
   Base should_not_import `fastcode.*`

6. **shell_must_not_import_facade**
   Shell should_not_import `FACADE`

### Additional Contract Tests

- Each domain module's `contracts.py` uses frozen dataclasses, not Pydantic
  (AST-based check, following ETL's `test_stage_domain_contracts_use_frozen_dataclasses_not_pydantic`)
- Domain modules contain no I/O imports (`sqlite3`, `subprocess`, `urllib`)
- Foundation (`utils`) imports no third-party heavy deps (no pydantic, numpy,
  etc.) — following ETL's `test_foundation_is_pure_constants_no_external_deps`
- Explicit translation: Shell stages that receive Interface (Pydantic) models
  must explicitly convert to Domain (frozen dataclass) contracts, not use
  `**model_dump()` or `**__dict__` mass-assignment

### Existing Architecture Tests

Kept (not replaced):

- `test_no_pydantic_in_core.py`
- `test_purity_gates.py`
- `test_explicit_translation.py`
- `test_settings_flow.py`
- `test_package_roots.py`
- `test_materialization_boundaries.py`

Replaced:

- `test_import_graph.py` → `test_layer_dag.py`

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
