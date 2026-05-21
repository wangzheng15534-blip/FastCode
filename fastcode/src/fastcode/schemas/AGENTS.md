# schemas

Boundary schemas and runtime config validation.

- Owns Pydantic boundary models and conversion into frozen runtime config
  dataclasses.
- `FastCodeConfig` is the canonical runtime config shape after validation.
- Do not import API route implementations, MCP server code, main runtime objects,
  query orchestration, retrieval logic, graph, scip, semantic, or store.
- Do not read env directly. Env overlays are prepared in `main/config.py` before
  schema validation.
- Keep compatibility exports thin and explicit; IR remains canonical for code
  structure types.
- Focused tests live under `fastcode/tests/schemas/`.
