# AGENTS.md

Repository-level instructions for contributors and coding agents. Read the
nearest module-local `AGENTS.md` for package-specific rules.

## Truth Order

Use this order when docs disagree:

1. Architecture tests and repo-wide lint rules.
2. Current code under `fastcode/src/fastcode/`.
3. Nearest module-local `AGENTS.md`.
4. [IMPLEMENTATION_TODOS.md](./IMPLEMENTATION_TODOS.md).
5. [ARCHITECTURE.md](./ARCHITECTURE.md).
6. Historical README or older branch docs.

FastCode has moved significantly. Verify current paths, imports, and runtime
behavior before trusting older examples.

## Package Map

The runtime package is a layered monolith rooted at `fastcode/src/fastcode/`.
The current branch is a hardened pre-release, not a stable release.

- `api/`: HTTP API shell, CORS, web entrypoint, response serialization.
- `graph/`: graph-domain construction, tree-sitter helpers, call extraction.
- `inbound/`: explicit inbound DTO/schema to frozen contract mappers.
- `indexing/`: repository loading, parsing, indexing, projection, publishing.
- `ir/`: canonical frozen IR dataclasses, graph views, merge, validation.
- `main/`: config preparation, CLI wiring, `FastCode` composition root.
- `mcp/`: MCP transport shell and graph/query tool adapters.
- `query/`: query orchestration, retriever shell, agent tools, LLM answering.
- `retrieval/`: pure retrieval scoring, fusion, context, iteration logic.
- `runtime/`: frozen runtime config and runtime lifecycle event contracts.
- `schemas/`: Pydantic inbound validation schemas and DTOs.
- `scip/`: SCIP models, loaders, indexers, symbol resolution, IR adapters.
- `semantic/`: semantic resolver contracts and helper-backed upgrades.
- `store/`: persistence, snapshots, vectors, manifests, cache, records.
- `store/infrastructure/`: lower-level DB, filesystem, graph, LLM runtime glue.
- `utils/`: stdlib-only shared helpers.

## Architecture Rules

Dependencies flow downward through the layer DAG. Do not introduce upward imports
or cross-layer cycles. `ir/`, `graph/`, and `retrieval/` stay Pydantic-free and
shell-free.

Inner packages do not read env directly. Config flows through
`prepare_runtime_config_mapping(...)`,
`fastcode.schemas.config.FastCodeConfigDTO`, explicit inbound mappers in
`fastcode.inbound.config_mapper`, `fastcode.runtime.config.FastCodeConfig`,
then `FastCode`.

Events and config are not miscellaneous hub layers. Runtime events/config live
under `runtime/`; external config schemas live under `schemas/`; schema-to-
contract translation lives under `inbound/`; config loading stays in `main/`;
future domain events/config should live near their owning domain.

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
