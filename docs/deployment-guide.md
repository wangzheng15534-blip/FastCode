# FastCode Deployment Guide

## Infrastructure Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| Python | 3.11+ | 3.12+ for Docker; LadybugDB max 3.11 |
| RAM | 4GB+ minimum | Embedding models + FAISS indexing |
| Disk | 2GB+ base | Vector stores grow with indexed repos |
| Ollama | Optional | Local embedding model (`all-minilm:l6-v2`) |
| PostgreSQL | 15+ | Optional — production storage backend |
| TerminusDB | Latest | Optional — graph storage backend |
| Docker | Latest | Container deployment |

## Storage Modes

SQLite is the default local mode. Treat it as single-user and single-process
storage: lock/fencing calls are compatibility no-ops, redo/outbox claim paths do
not provide durable multi-worker queue semantics, and PostgreSQL relational fact
tables are not populated. SQLite is appropriate for local development, demos,
and package/install smokes.

Use PostgreSQL for production-style storage semantics:

```bash
export FASTCODE_STORAGE_BACKEND=postgres
export FASTCODE_POSTGRES_DSN='postgresql://user:pass@postgres:5432/fastcode'
```

Before promoting a PostgreSQL-backed release candidate or deployment, run the
real backend storage gate from the repository root:

```bash
PG_E2E_DSN="$FASTCODE_POSTGRES_DSN" \
uv run pytest -n auto fastcode/tests/e2e/test_e2e_pg_storage_semantics.py -q
```

This gate exercises snapshot save/load, manifests, idempotent schema checks,
staging, lock ownership and stale fencing tokens, redo/outbox transitions, SCIP
refs, and relational graph facts against a real PostgreSQL database.

Backups must pair the PostgreSQL database with filesystem artifact roots from
the same point in time. PostgreSQL stores snapshots, refs, manifests, locks,
redo/outbox state, SCIP refs, design documents, and relational facts. The
configured data/cache/vector directories store snapshot shards, graph shards,
vector indexes, BM25 artifacts, and cache payloads. After restore, start
FastCode once so idempotent schema checks run, then rerun the storage gate.
Release notes must state whether existing schema/artifact families are readable,
migrated, or require rebuild.

## Docker Deployment

### Two-Service Architecture

```
┌──────────────────────┐      ┌──────────────────────┐
│     FastCode API     │      │   Nanobot Gateway    │
│   Port 8001 (HTTP)   │◄─────│   Port 18791 (HTTP)  │
│                      │      │                      │
│  - Indexing pipeline │      │  - Multi-channel AI  │
│  - Query engine      │      │  - FastCode bridge   │
│  - Projection layer  │      │  - LLM provider      │
│  - Graph API         │      │                      │
└──────────────────────┘      └──────────────────────┘
         │                              │
    ┌────┴────┐                   ┌────┴────┐
    │ Postgres│ (optional)        │  LLM    │
    │TerminusDB│(optional)        │Provider │
    └─────────┘                   └─────────┘
```

### docker-compose.yml

```yaml
services:
  fastcode:
    build: .
    ports: ["8001:8001"]
    volumes:
      - ./.env:/app/.env:ro
      - ./config:/app/config:ro
      - ./data:/app/data          # Indexes, cache
      - ./repos:/app/repos        # Repository storage
      - ./logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
      - TOKENIZERS_PARALLELISM=false

  nanobot:
    build: ./nanobot
    command: ["gateway"]
    ports: ["18791:18790"]
    volumes:
      - ./nanobot_config.json:/root/.nanobot/config.json:ro
      - nanobot-workspace:/root/.nanobot/workspace
      - nanobot-sessions:/root/.nanobot/sessions
      - ./repos:/app/repos:ro
    environment:
      - NANOBOT_ENV=docker
      - FASTCODE_API_URL=http://fastcode:8001
    depends_on: [fastcode]
```

### Build & Run

```bash
docker compose up              # Foreground
docker compose up -d           # Background
docker compose logs -f         # Logs
docker compose down            # Stop
docker compose build --no-cache  # Rebuild
```

### Dockerfile Details

- Base: `python:3.12-slim-bookworm`
- Pre-downloads embedding model (~470MB) in separate layer for caching
- Copies: `fastcode/`, `api.py`, `config/`
- Entry: `python api.py --host 0.0.0.0 --port 8001`

## Environment Configuration

### Required (for LLM features)

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | LLM API key | `sk-...` or `ollama` |
| `MODEL` | Model identifier | `google/gemini-3-flash-preview` |
| `BASE_URL` | LLM API base URL | `https://openrouter.ai/api/v1` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `FASTCODE_STORAGE_BACKEND` | `sqlite` | `postgres` for production |
| `FASTCODE_POSTGRES_DSN` | (none) | PostgreSQL connection string |
| `FASTCODE_PROJECTION_POSTGRES_DSN` | (none) | Separate projection store |
| `FASTCODE_EXCLUDE_SITE_PACKAGES` | (none) | Set to `1` to exclude during indexing |
| `NANOBOT_MODEL` | (none) | Model for nanobot agent reasoning |

## Nanobot Configuration

`nanobot_config.json` configures:
- **Agent defaults:** model, max tokens, temperature, system prompt
- **Channels:** Feishu (enabled), Telegram, Discord, WhatsApp, Slack, QQ, DingTalk, Email
- **LLM providers:** Anthropic, OpenAI, OpenRouter, DeepSeek, Groq, Zhipu, Dashscope, vLLM, Gemini, Moonshot
- **Gateway:** host/port
- **Tools:** web search, exec timeout

## Production Considerations

- **Storage backend:** Use PostgreSQL (`FASTCODE_STORAGE_BACKEND=postgres`) for multi-user production
- **Graph backend:** TerminusDB for branch-aware graph (configure `terminus.endpoint` in `config.yaml`)
- **Caching:** Redis for distributed cache (`cache.backend: redis`)
- **Embedding model:** Docker image pre-downloads `paraphrase-multilingual-MiniLM-L12-v2`
- **Repository storage:** `./repos/` should be on persistent volume
- **No auth configured:** Add authentication middleware before public deployment
