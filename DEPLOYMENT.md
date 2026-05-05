# FastCode Deployment Notes

FastCode's API and web entrypoints are trusted-local by default. They can clone,
read, upload, delete, and index repository contents, so do not expose them
directly to an untrusted network.

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
