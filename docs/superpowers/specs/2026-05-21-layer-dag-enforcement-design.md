# Layer DAG Enforcement Design

Date: 2026-05-21

## Goal

Enforce a 7-tier dependency DAG in FastCode with repo-local architecture tests,
following the pattern proven in the ETL project:

```
Facade → Shell → Events → Interface → Domain → Common → Base
```

Dependencies flow downward. Domain modules stay pure. Interface defines
Pydantic boundary models. Shell stages call downward through Interface
translations into Domain logic.

Implementation refinement: the current gate uses AST-based tests in
`fastcode/tests/architecture/test_layer_dag.py` instead of adding
`pytest-archon`. This keeps the same deny-rule semantics while also enforcing
that every runtime module is classified.

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
| `events` | Events (4) | Pipeline lifecycle and agent context events |
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
| `ir/` (common) | Canonical IR: snapshot, element, symbol, relation |
| `api/contracts.py` | API request/response boundary contracts |
| `events/__init__.py` | Pipeline and agent lifecycle event contracts |
| `schemas/` (interface) | `FastCodeConfig` and Pydantic boundary models |

Domain `contracts.py` files use frozen dataclasses. No Pydantic in domain
contracts. Interface (`schemas/`) uses Pydantic. Shell stages explicitly
translate between the two.

## Events Layer

FastCode currently has event-like records scattered across modules:
- Pipeline status/metrics/warnings in `indexing/`
- Agent context events (turn journals, observations) in `retrieval/`

The Events layer extracts these into a dedicated module. Events are
cross-cutting lifecycle contracts that Shell stages emit and consume. They use
frozen dataclasses (no Pydantic), following ETL's pattern where Events sits
above Interface.

Initial Events extraction candidates:
- Pipeline stage lifecycle events (started, completed, failed, skipped)
- Indexing run diagnostics and metrics events
- Agent context lifecycle (turn started, observation appended, turn completed)

This starts small with `PipelineStageEvent` and `AgentContextEvent`. The deny
rules still enforce the boundary even while event coverage grows.

## Violation Fixes

Upward imports must be broken. Every fix follows the same ETL principle:
code lives in the domain that owns it, shell calls downward, domain never
reaches up.

### Fix 1: `graph -> indexing` (Domain → Shell)

**Old violation:** `graph` imported parser helpers from `indexing`.
**Fix:** move parser-owned graph helpers into the graph domain:
`graph/call_extractor.py` and `graph/tree_sitter.py`.

### Fix 2: `scip -> indexing` and root helpers (Domain → Shell/Facade)

**Old violation:** SCIP translation imported shell/global helpers.
**Fix:** move SCIP-owned translation helpers into the SCIP domain:
`scip/global_builder.py` and `scip/module_resolver.py`.

### Fix 3: retrieval orchestration in the wrong layer

**Old violation:** orchestration-heavy retrieval modules lived under the
retrieval domain while importing query, store, and other shell concerns.
**Fix:** move shell orchestration into the query shell:
`query/retriever.py`, `query/iterative_agent.py`, and `query/agent_tools.py`.
The retrieval package now owns pure contracts and core scoring/context logic.

### Fix 4: shell-side native/runtime adapters

**Old violation:** root runtime helpers and native-library adapters sat outside
the 7-tier package map.
**Fix:** move runtime and native-adapter code to the owning shell package:
`store/infrastructure/runtime.py`, `store/infrastructure/graph_runtime.py`,
and `store/vector_math.py`.

### Fix 5: subprocess execution in domain packages

**Old violation:** SCIP and semantic domain modules directly executed external
toolchains.
**Fix:** domain modules expose profiles and injected contracts; shell runners
own process execution in `indexing/scip_runner.py` and
`indexing/semantic_helper_runner.py`.

### Fix 6: `utils -> schemas` (Base → Interface)

**Old violation:** `utils` handled config/env compatibility and imported
interface-layer types.
**Fix:** config preparation moved to `main/config.py`, token helpers moved to
`query/tokens.py`, ignore handling moved to `indexing/ignore.py`, path helpers
live in `utils/path_utils.py`, and `utils` is now stdlib-only.

## Schema Distribution

The current `schemas` module becomes the Interface layer:

- `schemas.config` stays for Pydantic config validation and frozen runtime
  config construction
- `schemas.ir` stays as the Pydantic boundary adapter for IR payloads
- `schemas.core_types` was removed; domain-specific types moved to owning
  contract modules such as `retrieval/contracts.py`, `scip/contracts.py`, and
  `graph/contracts.py`
- API request/response contracts moved to `api/contracts.py`

## Enforcement: AST Deny Rules

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

### Deny Rules

1. **domain_must_not_import_facade_shell_or_interface**
   Domain must not import `FACADE + SHELL + INTERFACE`

2. **interface_must_not_import_domain_facade_or_shell**
   Interface must not import `DOMAIN + FACADE + SHELL`

3. **events_must_not_import_domain_facade_or_shell**
   Events must not import `DOMAIN + FACADE + SHELL`

4. **common_must_not_import_events_interface_domain_facade_or_shell**
   Common must not import `EVENTS + INTERFACE + DOMAIN + FACADE + SHELL`

5. **base_must_not_import_any_fastcode**
   Base must not import `fastcode.*`

6. **shell_must_not_import_facade**
   Shell must not import `FACADE`

### Additional Contract Tests

- Every runtime module must be classified in exactly one top-level layer.
- Each domain module's `contracts.py` uses frozen dataclasses, not Pydantic
  (AST-based check, following ETL's `test_stage_domain_contracts_use_frozen_dataclasses_not_pydantic`)
- Domain modules contain no I/O imports (`sqlite3`, `subprocess`, `urllib`)
- Foundation (`utils`) imports only stdlib modules — a stricter form of ETL's
  `test_foundation_is_pure_constants_no_external_deps`
- Explicit translation: Shell stages that receive Interface (Pydantic) models
  must explicitly convert to Domain (frozen dataclass) contracts, not use
  `**model_dump()` or `**__dict__` mass-assignment
- Package roots are markers or explicit contract modules, not compatibility
  import APIs.
- Deleted pre-split modules remain deleted, and callers import their owning
  split modules directly.

### Existing Architecture Tests

Kept (not replaced):

- `test_no_pydantic_in_domain.py`
- `test_purity_gates.py`
- `test_explicit_translation.py`
- `test_settings_flow.py`
- `test_package_roots.py`
- `test_materialization_boundaries.py`

Replaced:

- `test_import_graph.py` → `test_layer_dag.py`

## Dependency Decision

No new dependency is required for this gate. The repo-local AST test enforces
the same layer deny rules and additionally checks module classification,
domain contracts, domain I/O purity, and Base-layer purity.

## What Stays the Same

- Package location (`fastcode/src/fastcode/`)
- All existing non-architecture tests
- All existing architecture tests except `test_import_graph.py`
- Runtime API endpoint behavior

## Out of Scope

- Module renaming to match ETL naming conventions
- Protocol-based DI patterns
- Backward-compatible Python import shims for moved modules
- Performance optimizations
