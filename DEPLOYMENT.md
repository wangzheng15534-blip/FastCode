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
/tmp/fastcode-install-smoke/bin/python -m pip install dist/fastcode-*.whl dist/nanobot_ai-*.whl
/tmp/fastcode-install-smoke/bin/fastcode --help
```

PostgreSQL-backed production semantics are still release-gate open. Do not claim
production backend support until the real PostgreSQL integration gate in
[docs/release.md](./docs/release.md) has passed for the release candidate.

## Local Mode

Use localhost binding for single-user development:

```bash
fastcode-api --host 127.0.0.1 --port 8000
fastcode-web --host 127.0.0.1 --port 5777
```

The REST API default host is `127.0.0.1`. Passing `--host 0.0.0.0` is an
explicit operator decision and requires the production controls below.

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
