# Package Layout Tightening Plan

**Date:** 2026-05-25  
**Source of truth:** `uv run python scripts/check_deps.py --group layout-assessment --report-gaps`

## Goal

Tighten the FastCode package layout before flat shell/domain packages grow into
hard-to-police compatibility surfaces.

This is a structural plan, not a behavior rewrite:

- keep the FCIS layer split intact
- reduce top-level module sprawl inside large packages
- group contracts with their owning bounded context
- preserve thin package roots and direct concrete imports
- avoid compatibility re-export shims while moving files

## Current Gaps

The strict layout assessment currently fails these package areas:

1. `store`
   - 24 root modules
   - 1 immediate subpackage
   - 9 root-level `*_contracts.py` modules
   - Gap: too many bounded contexts share one flat app-runtime shell namespace.

2. `indexing`
   - 18 root modules
   - 0 immediate subpackages
   - Gap: extraction, orchestration, projection, and publishing all sit at one level.

3. `query`
   - 12 root modules
   - 0 immediate subpackages
   - Gap: turn orchestration, agent flow, selection, and LLM glue are mixed together.

4. `retrieval`
   - 17 root modules
   - 0 immediate subpackages
   - Gap: pure-domain ranking/context/graph helpers accumulate in one namespace.

5. `semantic/resolvers`
   - 23 root modules
   - 0 immediate subpackages
   - Gap: language resolvers, helper-backed runners, and registry support are flat.

## Target Layout

### `store/`

Target:

```text
store/
  __init__.py
  infrastructure/
    __init__.py
    db.py
    execution.py
    fs.py
    graph_runtime.py
    llm.py
    runtime.py
  cache/
    __init__.py
    service.py
    contracts.py
    records.py
  artifacts/
    __init__.py
    file.py
    file_contracts.py
    graph.py
    unit.py
    unit_contracts.py
  snapshots/
    __init__.py
    manifest.py
    manifest_contracts.py
    projection.py
    projection_contracts.py
    snapshot.py
    snapshot_contracts.py
  vectors/
    __init__.py
    pg_retrieval.py
    pg_retrieval_contracts.py
    vector.py
    vector_contracts.py
    vector_math.py
  runs/
    __init__.py
    index_run.py
    index_run_contracts.py
```

Rules:

- `store/infrastructure/` remains the concrete effect boundary.
- Contract files move with the owning bounded context instead of staying flat.
- Private payload helpers stay private inside the owning subpackage.

### `indexing/`

Target:

```text
indexing/
  __init__.py
  extractors/
    __init__.py
    definition.py
    imports.py
    parser.py
  pipeline/
    __init__.py
    incremental.py
    indexer.py
    pipeline.py
    redo_worker.py
  projection/
    __init__.py
    build.py
    transform.py
  doc_ingester.py
  embedder.py
  file_inventory.py
  ignore.py
  loader.py
  overview.py
  publishing.py
  scip_runner.py
  terminus.py
```

Rules:

- Keep external runners/adapters visible as concrete modules.
- Group extraction and pipeline orchestration separately.
- Keep projection code in its own bounded context.

### `query/`

Target:

```text
query/
  __init__.py
  agent/
    __init__.py
    iterative.py
    tools.py
  orchestration/
    __init__.py
    answer.py
    handler.py
    processor.py
  selection/
    __init__.py
    retriever.py
    selector.py
  boundary.py
  context_payloads.py
  contracts.py
  llm.py
  tokens.py
```

Rules:

- Turn/session orchestration belongs together.
- Agent flow is separate from plain query orchestration.
- Selection/retriever glue becomes its own bounded context.

### `retrieval/`

Target:

```text
retrieval/
  __init__.py
  contracts.py
  ranking/
    __init__.py
    combination.py
    fcx.py
    filtering.py
    fusion.py
  context/
    __init__.py
    agent_context.py
    context.py
    context_compiler.py
    prompts.py
  graph/
    __init__.py
    graph_build.py
    iteration.py
    traversal.py
```

Rules:

- `retrieval/contracts.py` stays the public domain contract surface.
- Keep pure-domain helpers grouped by concern, not by historical file age.
- Do not add shell/inbound code while moving files.

### `semantic/resolvers/`

Target:

```text
semantic/resolvers/
  __init__.py
  core/
    __init__.py
    graph_backed.py
    helper_backed.py
    registry.py
    support.py
  helpers/
    __init__.py
    csharp_semantic_helper.py
    fortran_semantic_helper.py
    go_semantic_helper.py
    java_semantic_helper.py
    julia_semantic_helper.py
    rust_semantic_helper.py
    zig_semantic_helper.py
  languages/
    __init__.py
    c_family.py
    csharp.py
    fortran.py
    go.py
    java.py
    javascript.py
    julia.py
    python.py
    rust.py
    typescript.py
    zig.py
```

Rules:

- Separate language-specific resolvers from shared runtime/helper glue.
- Shared support and registry logic stay in `core/`.
- Helper-backed assets and wrappers stay together.

## Migration Order

1. `store`
   - Highest payoff because it already exposes the strongest bounded-context signals.

2. `query`
   - Next highest because user-facing orchestration complexity is concentrated here.

3. `indexing`
   - Important once `store` and `query` are less noisy.

4. `semantic/resolvers`
   - Good candidate for package split after shell boundaries are stable.

5. `retrieval`
   - Keep last because it is pure domain code and lower risk structurally.

## Change Discipline

For each package split:

1. Move one bounded context at a time.
2. Update imports directly; do not add compatibility re-export shims.
3. Keep `__init__.py` files marker-only unless the root metadata rule explicitly allows otherwise.
4. Update `import-linter` protected modules when private files move.
5. Keep `scripts/check_deps.py --group layout-assessment --report-gaps` improving monotonically.
6. Only enable `--strict-layout` in normal quality gates once the package passes.
