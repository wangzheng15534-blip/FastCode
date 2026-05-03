# AGENTS.md

This file provides contributor guidance for work in this repository.

## What to trust

Use this order of truth when docs disagree:

1. architecture tests and module-local lint rules
2. current code under `fastcode/src/fastcode/`
3. [IMPLEMENTATION_TODOS.md](./IMPLEMENTATION_TODOS.md)
4. [ARCHITECTURE.md](./ARCHITECTURE.md)
5. historical README or older branch docs

The codebase has moved significantly. Do not assume older module paths or
runtime behavior are still correct without checking the current package layout.

## Quick Start

```bash
# Install workspace deps
uv sync --extra dev

# Fast checks
uv run ruff format --check .
uv run ruff check .
uv run pyright

# Run all tests from repository root
uv run pytest -n auto

# Build package artifacts
uv build
```

## Current project shape

FastCode is a repository-understanding system for coding agents. The current
`develop` branch is a hardened pre-release, not a finished stable release.

The implementation is a layered monolithic package rooted at
`fastcode/src/fastcode/`.

Primary modules:

- `api/`
- `graph/`
- `indexing/`
- `ir/`
- `main/`
- `mcp/`
- `query/`
- `retrieval/`
- `retrieval/core/`
- `schemas/`
- `scip/`
- `semantic/`
- `store/`
- `store/infrastructure/`
- `utils/`

Read [ARCHITECTURE.md](./ARCHITECTURE.md) for the current package DAG and
boundary rationale.

## Enforced architecture rules

These rules are active in tests and lint:

1. No upward imports across layers.
2. No cross-layer cycles.
3. No Pydantic in `ir/`, `graph/`, or `retrieval/core/`.
4. No direct env reads or `load_dotenv()` in inner packages.
5. Package roots stay thin and lazy.
6. No `**model_dump()` or `**__dict__` mass-assignment in shell packages.

Important architecture tests:

- `fastcode/tests/architecture/test_import_graph.py`
- `fastcode/tests/architecture/test_no_pydantic_in_core.py`
- `fastcode/tests/architecture/test_purity_gates.py`
- `fastcode/tests/architecture/test_explicit_translation.py`
- `fastcode/tests/architecture/test_settings_flow.py`

Important module-local guards:

- `fastcode/src/fastcode/graph/ruff.toml`
- `fastcode/src/fastcode/ir/ruff.toml`
- `fastcode/src/fastcode/retrieval/core/ruff.toml`
- `fastcode/src/fastcode/schemas/ruff.toml`
- `fastcode/src/fastcode/store/infrastructure/ruff.toml`

## Runtime configuration

The active config flow is:

1. raw YAML and `.env` input
2. `fastcode.utils._compat.prepare_runtime_config_mapping(...)`
3. `fastcode.schemas.config.FastCodeConfig`
4. `fastcode.main.fastcode.FastCode`

`FastCodeConfig` is the canonical frozen runtime config.

Do not add new direct `os.getenv`, `os.environ[...]`, or `load_dotenv()` calls
inside `indexing/`, `query/`, `retrieval/`, `store/`, `mcp/`, or `schemas/`.

## Testing guidance

- Run tests from repository root. The merged workspace layout matters.
- Prefer focused runs while iterating, then validate at root-level.
- Use existing fixture factories in `fastcode/tests/conftest.py`.
- Partial construction via `FastCode.__new__(FastCode)` is normal in unit tests
  that need to bypass heavy initialization.

Common focused commands:

```bash
uv run pytest -n auto -k "architecture or settings_flow"
uv run pytest -n auto fastcode/tests/architecture
uv run pytest -n auto fastcode/tests/main/test_main.py
uv run pytest -n auto fastcode/tests/mcp/test_mcp_graph_tools.py
```

## Packaging and runtime notes

- Package metadata currently requires Python `>=3.11`.
- Do not trust stale README claims that say `3.12+` without checking current
  package metadata and dependency support.
- Entry points are defined in `fastcode/pyproject.toml`:
  - `fastcode`
  - `fastcode-api`
  - `fastcode-mcp`
  - `fastcode-web`

## Current release reality

The core pipeline is much harder to break than the prototype, but several
stable-release blockers remain open. The maintained list is in
[IMPLEMENTATION_TODOS.md](./IMPLEMENTATION_TODOS.md), especially:

- install and packaging reproducibility
- true end-to-end incremental caching
- deeper FP/FCIS boundary typing
- real backend and toolchain evidence
- API and upload hardening
- deploy and operations docs
