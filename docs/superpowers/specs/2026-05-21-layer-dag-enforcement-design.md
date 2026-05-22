# Layer DAG Enforcement Design

Date: 2026-05-21

## Goal

Enforce a dependency DAG in FastCode with repo-local architecture tests,
following the pattern proven in the ETL project:

```
Facade/Shell → Inbound Mapper → Schemas DTO
             → Runtime Contracts
             → Domain → Common → Base
```

Dependencies flow downward. Domain modules stay pure. `schemas/` defines
Pydantic boundary DTOs, `inbound/` performs schema-to-contract translation, and
`runtime/` owns frozen runtime contracts.

Implementation refinement: the current gate uses AST-based tests in
`fastcode/tests/architecture/test_layer_dag.py` instead of adding
`pytest-archon`. This keeps the same deny-rule semantics while also enforcing
that every runtime module is classified.

## Layer Model

The current AST gate is a partial order rather than one strict vertical stack:

- Facade: `api`, `mcp`, `main`
- Shell: `indexing`, `query`, `store`
- Inbound mapper: `inbound`
- Inbound schema: `schemas`
- Runtime contracts: `runtime`
- Domain: `retrieval`, `graph`, `scip`, `semantic`
- Common: `ir`
- Base: `utils`

### Layer Assignments

| Module | Layer | Role |
|--------|-------|------|
| `api` | Facade (6) | HTTP/web transport |
| `mcp` | Facade (6) | MCP transport |
| `main` | Facade (6) | Composition root, CLI wiring |
| `indexing` | Shell (5) | Pipeline orchestration, I/O stages |
| `query` | Shell (5) | Query orchestration |
| `store` | Shell (5) | Persistence I/O |
| `inbound` | Inbound (3) | Explicit inbound schema/DTO to frozen contract mappers |
| `schemas` | Interface (2) | Pydantic config DTOs and boundary models |
| `runtime` | Runtime contract (2) | Frozen runtime config and lifecycle event contracts |
| `retrieval` | Domain (2) | Retrieval logic and scoring |
| `graph` | Domain (2) | Graph construction primitives |
| `scip` | Domain (2) | SCIP translation logic |
| `semantic` | Domain (2) | Semantic resolution |
| `ir` | Common (1) | Canonical IR types, shared value objects |
| `utils` | Base (0) | Zero-dep stdlib helpers |

### What Each Layer Means (from ETL)

- **Facade:** composition root / presentation. Wires everything together.
- **Shell:** I/O infrastructure. Runs stages, reads/writes storage. Calls
  downward into runtime contracts, inbound mappers, Interface, Domain, Common,
  Base.
- **Runtime:** frozen runtime config and lifecycle event contracts. No
  Pydantic, shell, facade, inbound, schema, or domain imports.
- **Inbound:** explicit translation from validated external DTOs into frozen
  internal contracts. May import `schemas` DTOs and `runtime` contracts.
- **Interface:** Pydantic config DTOs and boundary models. Owns external shape,
  aliases, defaults, coercion, and validation. Must NOT import runtime,
  inbound, shell, facade, or domain packages.
- **Domain:** pure business logic. No I/O, no Pydantic. Each domain module has
  its own `contracts.py` with frozen dataclasses. May import Common, Base.
- **Common:** shared value objects, cross-cutting contracts. May import Base.
- **Base:** zero-dep. `utils` may not import any `fastcode.*` package.

### Key Constraint: Boundaries Are Directional

External config input moves through a one-way boundary:

```
raw mapping → schemas DTO → inbound mapper → runtime contract
```

Schemas do not construct runtime contracts, and runtime contracts do not import
schemas. Inbound mappers perform field-by-field translation instead of
`model_dump()` / `**kwargs` construction.

Closed vocabulary fields are represented with `StrEnum` on both sides of the
boundary. Schema enums describe accepted external values; runtime/domain enums
describe trusted internal states. The inbound mapper owns value translation
between those enum sets.

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
| `runtime/events.py` | Runtime lifecycle event contracts |
| `runtime/config.py` | Frozen runtime config contracts |
| `schemas/` (inbound schema) | `FastCodeConfigDTO` and Pydantic boundary models |
| `inbound/config_mapper.py` | Config DTO to frozen runtime config translation |

Domain `contracts.py` files use frozen dataclasses. No Pydantic in domain
contracts. Inbound schema (`schemas/`) uses Pydantic DTOs only. Inbound mappers
explicitly translate between external DTO meaning and internal frozen runtime
contracts.

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
the classified package map.
**Fix:** move runtime and native-adapter code to the owning shell package:
`store/infrastructure/runtime.py`, `store/infrastructure/graph_runtime.py`,
and `store/vector_math.py`.

### Fix 5: subprocess execution in domain packages

**Old violation:** SCIP and semantic domain modules directly executed external
toolchains.
**Fix:** domain modules expose profiles and injected contracts; shell runners
own process execution in `indexing/scip_runner.py` and
`indexing/semantic_helper_runner.py`.

### Fix 6: `utils -> schemas` (Base → Inbound Schema)

**Old violation:** `utils` handled config/env compatibility and imported
inbound-schema types.
**Fix:** config preparation moved to `main/config.py`, token helpers moved to
`query/tokens.py`, ignore handling moved to `indexing/ignore.py`, path helpers
live in `utils/path_utils.py`, and `utils` is now stdlib-only.

## Schema Distribution

The current config boundary is split by responsibility:

- `schemas.config` owns Pydantic config DTO validation only
- `inbound.config_mapper` owns field-explicit DTO to runtime contract mapping
- `runtime.config` owns frozen runtime config contracts
- runtime/domain contracts own semantic invariants and command/config validity,
  not Pydantic models
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
RUNTIME = ("fastcode.runtime.*",)
INBOUND = ("fastcode.inbound.*",)
INBOUND_SCHEMA = ("fastcode.schemas.*",)
DOMAIN = ("fastcode.retrieval.*", "fastcode.graph.*", "fastcode.scip.*", "fastcode.semantic.*")
COMMON = ("fastcode.ir.*",)
BASE = ("fastcode.utils.*",)
```

### Deny Rules

1. **domain_must_not_import_facade_shell_or_inbound_schema**
   Domain must not import `FACADE + SHELL + INBOUND_SCHEMA`

2. **inbound_schema_must_not_import_runtime_inbound_domain_facade_or_shell**
   Inbound schema must not import `RUNTIME + INBOUND + DOMAIN + FACADE + SHELL`

3. **runtime_must_not_import_schemas_inbound_domain_facade_or_shell**
   Runtime contracts must stay Pydantic-free and independent of boundary
   schemas, inbound mappers, domain packages, facades, and shells.

4. **inbound_must_not_import_domain_facade_or_shell**
   Inbound mappers may import schemas and runtime contracts, but not domain,
   facade, or shell packages.

5. **common_must_not_import_runtime_inbound_schema_domain_facade_or_shell**
   Common must not import `RUNTIME + INBOUND + INBOUND_SCHEMA + DOMAIN + FACADE + SHELL`

6. **base_must_not_import_any_fastcode**
   Base must not import `fastcode.*`

7. **shell_must_not_import_facade**
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
