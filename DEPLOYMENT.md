# FastCode Deployment Notes

FastCode's API and web entrypoints are trusted-local by default. They can clone,
read, upload, delete, and index repository contents, so do not expose them
directly to an untrusted network.

For release-tag validation, use the checked-in release gate:

```bash
python scripts/release_gate.py
```

The release gate builds wheel/sdist artifacts, verifies packaged helper assets,
installs the artifacts into fresh virtualenvs with `pip`, smokes installed
entrypoints, and runs a tiny installed-wheel index/query flow. See
[docs/release.md](./docs/release.md) for the full gate matrix and the open
stable-release blockers.

## Install Modes

Local checkout for contributors:

```bash
uv sync --extra dev
uv run fastcode --help
```

Built artifact install for release validation:

```bash
uv build --all-packages --clear
python -m venv /tmp/fastcode-install-smoke
/tmp/fastcode-install-smoke/bin/python -m pip install dist/fastcode-*.whl
/tmp/fastcode-install-smoke/bin/fastcode --help
```

Service extras are installable explicitly when API, web, MCP, Postgres, or Redis
entrypoints are needed:

```bash
/tmp/fastcode-install-smoke/bin/python -m pip install \
  'dist/fastcode-*.whl[api,mcp,postgres,redis]'
```

PostgreSQL-backed storage semantics are guarded by the real PostgreSQL backend
gate in [docs/release.md](./docs/release.md). Run it before any release or
deployment claim that depends on durable locks, redo/outbox queues, manifests,
SCIP refs, or relational graph facts.

## Local Mode

Use localhost binding for single-user development:

```bash
fastcode-api --host 127.0.0.1 --port 8000
fastcode-web --host 127.0.0.1 --port 5777
```

The REST API default host is `127.0.0.1`. Passing `--host 0.0.0.0` is an
explicit operator decision and requires the production controls below.

SQLite is local and single-process only. In SQLite mode, lock/fencing APIs are
compatibility no-ops, redo/outbox claim paths do not provide durable
multi-worker queue semantics, and PostgreSQL relational fact tables are not
populated. Use SQLite for local development, single-user demos, and package
smokes; use PostgreSQL for production-style storage semantics.

## PostgreSQL Storage

Install the service extras and configure the storage backend explicitly:

```bash
python -m pip install 'fastcode[postgres]'
export FASTCODE_STORAGE_BACKEND=postgres
export FASTCODE_POSTGRES_DSN='postgresql://user:pass@host:5432/fastcode'
```

Before operating a PostgreSQL-backed release candidate, run:

```bash
PG_E2E_DSN="$FASTCODE_POSTGRES_DSN" \
uv run pytest -n auto fastcode/tests/e2e/test_e2e_pg_storage_semantics.py -q
```

That gate exercises snapshot save/load, manifest heads, schema idempotency,
staging, lock ownership and stale fencing tokens, redo/outbox state transitions,
SCIP refs, and relational graph facts against a real PostgreSQL database.

Back up PostgreSQL and filesystem artifacts together. PostgreSQL owns snapshots,
refs, manifests, locks, redo/outbox state, SCIP refs, design documents, and
relational facts. The configured persist/cache/vector directories own snapshot
shards, graph shards, vector indexes, BM25 artifacts, and cache payloads. Restore
both from the same point in time, start FastCode once so idempotent schema checks
run, then rerun the PostgreSQL storage gate. Schema compatibility is tracked by
`schema_migrations`; release notes must say whether existing artifact families
are readable, migrated, or require rebuild.

## Production Exposure

FastCode does not currently implement built-in user authentication or
authorization. For any shared or remote deployment, put the service behind a
reverse proxy or gateway that provides:

- TLS termination.
- Authentication and authorization for every mutation endpoint.
- Request size limits that are at least as strict as FastCode upload limits.
- Access logs and audit retention appropriate for source-code access.

Mutation endpoints include repository load/index/upload/delete/cache operations,
index publishing/retry/recovery endpoints, and repository unload/refresh actions.

## CORS

CORS defaults are local-only and do not allow credentials:

```bash
FASTCODE_CORS_ALLOW_ORIGINS=http://localhost:5777,http://127.0.0.1:5777
FASTCODE_CORS_ALLOW_CREDENTIALS=false
```

Use exact origins in production. If `FASTCODE_CORS_ALLOW_ORIGINS=*` is set,
FastCode forces `allow_credentials=false` to avoid wildcard credential exposure.

## Upload Safety

ZIP uploads are validated before extraction. FastCode rejects unsafe archive
members including absolute paths, `..` traversal, symlinks, special files,
excessive member counts, oversized expanded files, and suspicious compression
ratios.

Operators should still enforce proxy-level upload body limits and monitor failed
upload/index attempts. A failed upload may leave no indexed repository, but logs
under `./logs/` should be retained for diagnosis.

## Diagnostic Collection

The REST API exposes `GET /diagnostics` for support-safe runtime diagnostics.
The bundle includes:

- a redacted config summary with secret-bearing values reported only as
  configured/not configured
- storage backend and artifact/cache locations
- Python package and external tool availability
- latest index-run metadata, warnings, pipeline layers, and pipeline metrics

Collect this endpoint output together with relevant logs under `./logs/` when
triaging indexing, cache reuse, dependency, or storage-backend issues. The
endpoint does not probe external services or execute tool commands; it reports
local availability only.
