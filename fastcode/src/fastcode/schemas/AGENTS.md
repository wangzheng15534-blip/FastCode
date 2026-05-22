# schemas

Inbound validation schemas and DTOs.

- Own Pydantic boundary models for external/config shapes: aliases, defaults,
  coercion, and field validation.
- Do not build frozen runtime/app contracts here. Explicit schema-to-contract
  translation belongs in `fastcode.inbound`.
- Do not import API route implementations, MCP server code, main runtime objects,
  runtime contracts, inbound mappers, query orchestration, retrieval logic,
  graph, scip, semantic, or store.
- Do not read env directly. Env overlays are prepared in `main/config.py` before
  schema validation.
- Keep compatibility exports thin and explicit; IR remains canonical for code
  structure types.
- Focused tests live under `fastcode/tests/schemas/`.
