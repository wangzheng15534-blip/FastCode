# FastCode Release Gate

FastCode is still a hardened pre-release. A stable tag requires every blocking
gate in this file to pass, and any unsupported mode must be explicitly scoped
out in the release notes.

## Supported Matrix

Current package metadata declares Python `>=3.11`.

The package/install gate currently proves the active release-host interpreter
only. The latest checked run was on Python 3.13.13. Before a stable release, run
the same gates across every Python version that will be listed as supported.

Supported deployment modes for the current pre-release:

- local single-user SQLite mode
- trusted-local API/web service mode behind localhost binding
- PostgreSQL-backed mode as an implementation target, pending real backend gate
  evidence

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
- serve fake Ollama embeddings and OpenAI chat-completions endpoints so the
  installed wheel exercises configured embedding and answer generation paths

Use this variant when the built artifacts need to be inspected:

```bash
python scripts/release_gate.py --keep-artifacts
```

Current evidence: this gate passed on May 13, 2026 with Python 3.13.13.

### Backend Integration Gate

Status: open.

Before stable release, run a real PostgreSQL suite that covers snapshot
save/load, manifests, locks, fencing, staging, outbox, redo tasks, SCIP refs,
graph facts, migration idempotency, and stale-lock ownership behavior.

SQLite is local/single-process storage. PostgreSQL is required for production
backend semantics.

### External Tool Gate

Status: open.

Before stable release, run availability-gated command smokes for every language
listed as stable in release notes. Languages without command evidence must be
documented as structural, degraded, or experimental.

### Performance Gate

Status: open.

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
