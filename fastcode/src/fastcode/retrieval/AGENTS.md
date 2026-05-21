# retrieval

Pure retrieval domain logic.

- Owns scoring, fusion, filtering, combination, context compilation, iteration
  decisions, prompts, summaries, and retrieval contracts.
- Keep this package Pydantic-free and independent of API, MCP, main, query,
  indexing, store, schemas, graph orchestration, scip loaders, and semantic
  adapters.
- Prefer frozen dataclasses and explicit contract types over boundary schemas.
- Do not perform database, subprocess, env, or network I/O here.
- Keep vector and candidate data in native carriers until the caller reaches a
  real boundary.
- Focused tests live under `fastcode/tests/retrieval/` and architecture tests.
