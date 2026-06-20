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


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->
