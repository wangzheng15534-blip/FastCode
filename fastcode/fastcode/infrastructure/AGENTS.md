# store/infrastructure

Lower-level storage and runtime adapters.

- Owns DB runtime wrappers, filesystem storage helpers, graph runtime glue, and
  low-level LLM adapter plumbing used by store code.
- Keep this package independent of API, MCP, main, query, retrieval, indexing,
  graph, scip, semantic, and schemas.
- Do not import Pydantic. Operate on primitives, typed records, and frozen
  dataclasses supplied by outer layers.
- Keep SQL and row mapping explicit; avoid generic dict round trips on active
  paths.
- Focused tests live under `fastcode/tests/store/infrastructure/`.
