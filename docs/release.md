# FastCode Release Gate

FastCode is still a hardened pre-release. A stable tag requires every blocking
gate in this file to pass, and any unsupported mode must be explicitly scoped
out in the release notes.

## Supported Matrix

Current package metadata declares Python `>=3.11`. CI runs the default test gate
on Python 3.11, 3.12, and 3.13. The manual `release-gate` workflow runs the
artifact/package smoke on Python 3.13 and can include heavy optional extras via
the workflow input.

The default package gate now keeps the local embedding stack and Nanobot stack
out of the core install path. Service extras are installed separately, and the
heavy docs/local-embedding/Nanobot extras are behind `--include-heavy-extras`
when the gate needs them.

Supported deployment modes for the current pre-release:

- local single-user SQLite mode
- trusted-local API/web service mode behind localhost binding
- PostgreSQL-backed retrieval and metadata storage mode, gated by the real
  PostgreSQL integration commands below

The API and web services do not provide built-in user authentication. Shared or
remote deployments require a proxy/gateway with TLS and authz.

## Blocking Gates

Run these from the repository root.

### Architecture Gate

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -n auto fastcode/tests/architecture
```

This gate protects package layering, Pydantic boundaries, settings flow,
package-root laziness, explicit translation, and materialization allowlists.

### Test Gate

```bash
uv run pytest -n auto
```

This is the default workspace-root regression gate. Focused package-root runs
are useful during iteration, but they do not replace the workspace-root gate.

### Package And Install Gate

```bash
python scripts/release_gate.py
```

This gate intentionally avoids editable installs. It performs all of the
following in temporary directories:

- `uv build --all-packages --clear`
- verify required helper assets are present in the FastCode wheel and sdist
- install built sdists into a fresh virtualenv with `pip install`
- smoke installed imports and installed console entrypoints
- install built wheels into a fresh virtualenv with `pip install`
- run installed CLI index/query smoke against a tiny repository
- install service extras separately so API/MCP/PostgreSQL/Redis imports and
  entrypoints are exercised without forcing the heavy embedding/docs stack
- serve fake Ollama embeddings and OpenAI chat-completions endpoints so the
  installed wheel exercises configured embedding and answer generation paths

Use this variant when the built artifacts need to be inspected:

```bash
python scripts/release_gate.py --keep-artifacts
```

Use this variant when you want the heavy optional extras in the smoke path:

```bash
python scripts/release_gate.py --include-heavy-extras
```

Current evidence: this gate passed on May 13, 2026 with Python 3.13.13.

### Backend Integration Gate

Status: required before tagging any release that claims PostgreSQL-backed
storage semantics.

Current real-service evidence from May 14, 2026:

```bash
FASTCODE_E2E_OLLAMA_URL=http://10.0.0.203:11434/api/embeddings \
PG_E2E_DSN=postgresql://postgres:postgres@127.0.0.1:5432 \
uv run pytest -n 0 --timeout=300 \
  fastcode/tests/e2e/test_e2e_indexing.py::test_e2e_indexing_pg_real_embeddings -q
```

That gate passed against PostgreSQL 17.9 with pgvector and Ollama
`all-minilm:l6-v2` embeddings. A separate docs-enabled smoke using
`gpt-oss:20b-cloud` through Ollama's OpenAI-compatible endpoint indexed and
queried a temp repo, producing 10 PG vector rows, 384-dimensional pgvector
values, zero legacy `embedding_arr` rows, no raw embedding leaks in
`metadata_json`, and no missing embedding refs/fingerprints across code and doc
chunk rows. The smoke degraded only because Terminus publishing was intentionally
unconfigured.

Storage semantics gate:

```bash
PG_E2E_DSN=postgresql://postgres:postgres@127.0.0.1:5432/fastcode_e2e \
uv run pytest -n auto \
  fastcode/tests/e2e/test_e2e_pg_storage_semantics.py -q
```

That gate covers real PostgreSQL snapshot save/load, manifest head chaining,
schema migration idempotency on an existing database, snapshot staging and
promotion, lock ownership and stale-fencing-token invalidation, redo task
claim/retry/done transitions, outbox duplicate insert detection and retry/done
transitions, multi-artifact SCIP refs, and relational graph fact persistence
including delta copy/upsert behavior. It is intentionally separate from the
Ollama/pgvector indexing smoke so storage semantics can be validated without an
embedding service.

SQLite is local/single-process storage. PostgreSQL is required for production
backend semantics. In SQLite mode, lock/fencing APIs are no-ops that return
local success values, redo/outbox claim paths do not provide durable queue
semantics, and PostgreSQL relational fact tables are not populated.

For PostgreSQL backup/restore validation, take database dumps and filesystem
artifact backups together. The database owns snapshots, refs, manifests,
resource locks, redo/outbox state, SCIP refs, design documents, and relational
facts; the configured persist/cache/vector directories own snapshot shards,
graph shards, vector indexes, BM25 artifacts, and cache payloads. Restore both
sets from the same point in time, instantiate `SnapshotStore` and
`ManifestStore` once to apply idempotent schema checks, then rerun the storage
semantics gate above. Current schema compatibility is tracked through
`schema_migrations` rows for `core_metadata`, `pg_full_spec_alignment`, and
`manifest_store`; release notes must state whether existing artifact families
are readable or require rebuild.

### External Tool Gate

Status: availability-gated command smokes are automated; full per-language
indexing evidence is still required before stable release claims.

CI includes `fastcode/tests/scip/test_scip_tool_smoke.py`, which runs
`--version` command smokes for installed stable SCIP tool binaries and skips
tools that are unavailable on the runner. Set
`FASTCODE_SCIP_SMOKE_LANGUAGES=python,rust` to restrict a local run:

```bash
uv run pytest fastcode/tests/scip/test_scip_tool_smoke.py -q
```

Before stable release, run availability-gated command smokes for every language
listed as stable in release notes. Languages without command evidence must be
documented as structural, degraded, or experimental.

### Performance Gate

Status: open.

Current narrow evidence from May 14, 2026: Ollama embedding uses the
provider-native `/api/embed` batch endpoint when available. A real
`all-minilm:l6-v2` smoke returned three 384-dimensional embeddings with one
provider request and `provider_true_batch_count=1`. This proves the provider
batch boundary, but it is not a replacement for the full performance gate.

Before stable release, run a medium-repository benchmark suite that records:

- full index time
- one-file implementation-only update time
- one-file interface-affecting update time
- query latency against a warm snapshot
- peak RSS or allocation evidence for index/update/query hot paths

The gate should fail on major regressions or on any claim that incremental
updates avoided work when metadata shows a full rebuild or degraded widening.

### Documentation Gate

Before tagging, update:

- `IMPLEMENTATION_TODOS.md` release status
- `ARCHITECTURE.md` release blockers
- deployment notes
- changelog or release notes
- compatibility and artifact migration notes

Release notes must list exact gate commands, host OS, Python versions, backend
versions, external tool versions, and any scoped-out limitations.

## Compatibility Policy

Stable compatibility promises are not active yet. Until a stable tag exists:

- persisted snapshots, manifests, caches, vector stores, projection artifacts,
  and helper/SCIP artifacts may require rebuild after upgrade
- incompatible artifacts must fail clearly or be rebuilt, not silently reused
- release notes must say which artifact families are readable and which require
  rebuild

For a future stable line:

- patch releases may add metadata fields and bug fixes, but must not silently
  reinterpret existing artifacts
- minor releases may add artifact versions and migrations when old artifacts
  remain readable or fail with clear rebuild instructions
- major releases may break artifact compatibility, but must document rebuild
  steps and any migration tooling

## Patch, Minor, And Major Blocking Rules

Block a patch release for:

- security regressions
- data loss or stale-artifact reuse
- package/install gate failures
- migration failures for artifacts promised compatible in that patch line
- dependency resolver changes that alter runtime behavior without review

Block a minor release for:

- any failing blocking gate above
- unsupported language/backend claims in docs or API metadata
- missing migration or rebuild notes for changed artifact families
- performance regressions outside the documented envelope

Block a major release for:

- unreproducible package builds
- unclear artifact breakage or missing upgrade path
- missing deployment/auth assumptions for exposed services

## Acceptable Degraded Behavior

Acceptable degraded behavior must be explicit in metadata and logs:

- unavailable optional SCIP/helper tooling can fall back to structural facts
- unknown dependency frontiers can widen incremental work
- missing fingerprints can disable reuse and force rebuild
- unsupported languages can remain structural/experimental

Not acceptable for stable claims:

- silently reusing stale embeddings, graph facts, or snapshots
- claiming incremental success after a full rebuild without metadata
- exposing API/web services directly to untrusted networks
- listing experimental languages as stable
