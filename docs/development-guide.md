# FastCode Development Guide

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | 3.12+ for Docker; LadybugDB max 3.11 |
| uv | 0.11+ | Package manager (preferred) |
| pip | Latest | Fallback if uv unavailable |
| Ollama | Latest | Local embedding model server |
| PostgreSQL | 15+ | Optional — production storage backend |
| TerminusDB | Latest | Optional — graph storage backend |
| Docker + Compose | Latest | Container deployment |
| git | Latest | Repository cloning |

## Installation

```bash
# Clone and enter
git clone <repo-url> && cd FastCode

# Install (editable with dev deps)
uv pip install -e ".[dev]"

# Optional: SCIP protobuf support
uv pip install -e ".[scip]"

# Optional: LadybugDB graph backend (Python 3.11 only)
uv pip install -e ".[ladybug]"
```

## Environment Setup

Copy `env.example` to `.env` and configure:

```bash
# Required for LLM features
OPENAI_API_KEY=your_key
MODEL=your_model          # e.g. google/gemini-3-flash-preview
BASE_URL=your_base_url    # e.g. https://openrouter.ai/api/v1

# Ollama (alternative for embeddings)
# OPENAI_API_KEY=ollama
# MODEL=qwen3-coder-30b_fastcode
# BASE_URL=http://localhost:11434/v1

# Optional: PostgreSQL backend
# FASTCODE_STORAGE_BACKEND=postgres
# FASTCODE_POSTGRES_DSN=postgresql://user:pass@127.0.0.1:5432/fastcode

# Optional: Projection store (separate PG instance)
# FASTCODE_PROJECTION_POSTGRES_DSN=postgresql://user:pass@127.0.0.1:5432/fastcode
```

## Running

```bash
# CLI — index and query repositories
uv run python main.py load-repo /path/to/repo
uv run python main.py query "how does authentication work?"

# FastAPI REST server
uv run python api.py --host 0.0.0.0 --port 8001

# Flask web UI
uv run python web_app.py

# MCP server (for Claude Code, Cursor, Windsurf)
uv run mcp run mcp_server.py

# Nanobot gateway (multi-channel AI agent)
cd nanobot && uv run python -m nanobot gateway
```

## Docker Deployment

```bash
# Start both services (FastCode + Nanobot)
docker compose up

# Background mode
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

**Services:**
- `fastcode` — Python app on port 8001 (API + indexing + retrieval)
- `nanobot` — Gateway on port 18791, connects to fastcode internally

## Configuration

Default config in `config/config.yaml`. Key sections:

| Section | Purpose |
|---------|---------|
| `storage` | Backend selection (sqlite/postgres), connection pool |
| `embedding` | Model, provider (ollama/sentence_transformers), device |
| `retrieval` | Hybrid search weights, agency mode, adaptive fusion |
| `query` | Query expansion, intent detection, LLM enhancement |
| `generation` | LLM answer synthesis settings |
| `graph` | Graph building, LadybugDB settings |
| `projection` | Leiden clustering, edge weights, LLM labeling |
| `terminus` | TerminusDB endpoint and API key |
| `docs_integration` | Key document ingestion paths and chunking |

## Testing

```bash
# Run all tests (parallel via pytest-xdist)
uv run pytest tests/ -v

# Single test file
uv run pytest tests/test_ir_core.py -v

# Filter by name
uv run pytest tests/ -v -k "fusion"

# With coverage
uv run pytest tests/ --cov=fastcode

# Property-based tests only
uv run pytest tests/property/ -v

# Benchmarks
uv run pytest tests/bench_*.py -v --benchmark-only

# E2E tests (requires Ollama + PostgreSQL)
uv run pytest tests/test_e2e_indexing.py -v

# Snapshot contract tests
uv run pytest tests/snapshots/ -v
```

**Test categories (markers):**
- `@pytest.mark.slow` — Long-running tests (deselect with `-m "not slow"`)
- `@pytest.mark.property` — Hypothesis property-based tests
- `@pytest.mark.snapshot` — Syrupy snapshot contract tests
- `@pytest.mark.happy` — Normal-path tests
- `@pytest.mark.edge` — Edge-case tests
- `@pytest.mark.mutation` — Mutation-killing tests
- `@pytest.mark.integration` — Tests requiring real DB/external service

## Code Quality

```bash
# Linting (ruff)
ruff check fastcode/ tests/

# Type checking (pyright — strict mode)
pyright fastcode/ tests/

# Git hooks (lefthook — runs ruff + pyright on commit)
lefthook run pre-commit
```

**Ruff config** (pyproject.toml):
- Line length: 88
- Target: Python 3.11
- Rules: pycodestyle, pyflakes, isort, flake8-bugbear, comprehensions, pyupgrade, annotations, bandit, pytest-style, simplify, return, raise, pylint, ruff-specific

**Pyright config** (pyproject.toml):
- Mode: strict
- Reports: unnecessary comparison, unreachable code, unused variables

## Common Development Tasks

### Adding a new SCIP language indexer
1. Add tree-sitter grammar to `requirements.txt`
2. Implement indexer in `fastcode/scip_indexers.py`
3. Add property tests in `tests/property/test_scip_indexers_properties.py`
4. Update `config.yaml` supported_extensions if needed

### Adding a new API endpoint
1. Add route handler in `api.py`
2. Update MCP server in `mcp_server.py` if needed for AI agent access
3. Add test in appropriate test file
4. Update nanobot bridge tool in `nanobot/nanobot/agent/tools/fastcode.py` if needed

### Adding a new graph algorithm
1. Implement in `fastcode/projection_transform.py`
2. Add to projection pipeline steps
3. Add property tests in `tests/property/`
4. Benchmark with `tests/bench_graph_projection.py`
