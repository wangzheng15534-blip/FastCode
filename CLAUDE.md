# AGENTS.md

Repository-level instructions for contributors and coding agents. Read the
nearest module-local `AGENTS.md` for package-specific rules.

## Truth Order

Use this order when docs disagree:

1. Architecture tests and repo-wide lint rules.
2. Current code under `fastcode/fastcode/`.
3. Nearest module-local `AGENTS.md`.
4. [IMPLEMENTATION_TODOS.md](./IMPLEMENTATION_TODOS.md).
5. [ARCHITECTURE.md](./ARCHITECTURE.md).
6. Historical README or older branch docs.

FastCode has moved significantly. Verify current paths, imports, and runtime
behavior before trusting older examples.

## Package Map

The runtime package is a layered monolith rooted at `fastcode/fastcode/`.
The current branch is a hardened pre-release, not a stable release.

- `utils/`: generic primitives and stdlib-only helper APIs.
- `common/`: shared identity vocabulary and frozen config contracts.
- `runtime_support/`: generic retry, health, observability, and lifecycle event helpers.
- `ports/`: narrow shared capability contract surfaces.
- `ir/`: canonical frozen IR dataclasses, graph views, merge, validation.
- `graph/`: graph-domain construction, tree-sitter helpers, call extraction.
- `retrieval/`: pure retrieval scoring, fusion, context, iteration logic.
- `semantic/`: semantic resolver contracts and helper-backed upgrades.
- `scip/`: SCIP models, loaders, indexers, symbol resolution, IR adapters.
- `app/indexing/`: repository loading, parsing, indexing, projection, publishing.
- `app/query/`: query orchestration, retriever shell, agent tools, LLM answering.
- `app/store/`: persistence, snapshots, vectors, manifests, cache, records.
- `infrastructure/storage/`: DB, filesystem persistence adapters.
- `infrastructure/execution/`: execution runners.
- `infrastructure/llm/`: LLM SDK wrappers.
- `infrastructure/graph_runtime/`: graph runtime adapter.
- `main/`: config preparation, CLI wiring, config DTO shaping, `FastCode` composition root.
- `api/`: HTTP API shell, CORS, web entrypoint, response serialization.
- `mcp/`: MCP transport shell and graph/query tool adapters.

## Architecture Rules

Dependencies flow downward through the layer DAG. Do not introduce upward imports
or cross-layer cycles. `ir/`, `graph/`, and `retrieval/` stay Pydantic-free and
shell-free.

Shell code follows the FCIS split:

- app-runtime/use-case shell: coordinates workflows and owns mutable runtime
  use, currently `app/indexing/`, `app/query/`, and most of `app/store/`;
- capability ports: shared external capability contracts under
  `fastcode/ports/`.
  Ports are compile-time capability contracts, not runtime wiring modules:
  app-runtime and infrastructure may both import them, but ports must not import
  either side or construct adapters;
- infrastructure: concrete network, DB, filesystem, subprocess, native-library,
  and SDK wrappers, currently `infrastructure/` plus owner-local runners
  such as `app/indexing/scip_runner.py`.

Do not add package-local `ports.py` modules for DB, network, filesystem,
subprocess, event, queue, storage, or other external capabilities. Keep those
capability contracts in `fastcode.ports` and keep domain contracts limited to
pure domain types or domain polymorphism.

Inner packages do not read env directly. Config flows through
`prepare_runtime_config_mapping(...)`,
`fastcode.main.config_schema.FastCodeConfigDTO`, explicit mappers in
`fastcode.main.config_mapper`, `fastcode.common.config.FastCodeConfig`,
then `FastCode`.

Events and config are not miscellaneous hub layers. Runtime events live
under `runtime_support/`; frozen config contracts live under `common/`;
config DTO shaping and loading live in `main/`; future domain events/config
should live near their owning domain.

Keep package roots thin. Avoid compatibility exports, `__getattr__` shims, and
runtime imports in `__init__.py`.

Prefer explicit field serializers/deserializers at API, persistence, cache, and
native-library boundaries. Avoid hot-path `row_to_dict() -> from_dict()/to_dict()`
round trips when typed records or explicit adapters are available.

Keep embeddings and ranked vector candidates native or NumPy-backed until a
backend boundary requires JSON or Python lists.

## Commands

Run commands from the repository root unless a tool explicitly says otherwise.

- Install workspace deps: `uv sync --extra dev`
- Format check: `uv run ruff format --check .`
- Lint check: `uv run ruff check .`
- Type check: `uv run pyright`
- Architecture tests: `uv run pytest -n auto fastcode/tests/architecture`
- Full tests: `uv run pytest -n auto`
- Build artifacts: `uv build`

## Testing

Prefer focused tests while iterating, then validate at root level. Use fixtures
from `fastcode/tests/conftest.py`. `FastCode.__new__(FastCode)` is acceptable
for unit tests that bypass heavy initialization.

For boundary-hardening changes, add regressions that fail if old generic
conversion paths are reused, for example patched `to_dict()`, `from_dict()`, or
`row_to_dict()` calls that raise.
