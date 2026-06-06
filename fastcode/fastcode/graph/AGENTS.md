# graph

Graph-domain construction and source-structure helpers.

- Owns graph building, call extraction, and tree-sitter helper integration.
- Depends on IR and low-level helpers, not on query, retrieval orchestration,
  schemas, store, API, MCP, or main.
- Keep this layer Pydantic-free and independent of database, subprocess, and
  network I/O.
- Graph facts should be represented as IR-level or graph-domain contracts before
  shell packages materialize them.
- Focused tests live under `fastcode/tests/graph/` and architecture purity tests.
