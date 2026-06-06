# ir

Canonical internal representation.

- Owns frozen dataclasses for documents, symbols, occurrences, relations,
  supports, embeddings, snapshots, graph views, merge, and validation.
- This is common/core code. It must not import graph, indexing, query,
  retrieval, schemas, scip, semantic, store, API, MCP, or main.
- Keep IR Pydantic-free, env-free, DB-free, subprocess-free, and network-free.
- Add explicit `to_dict()` or `from_dict()` fields only at intentional artifact
  boundaries; do not add generic compatibility shims.
- Focused tests live under `fastcode/tests/ir/` and architecture purity tests.
