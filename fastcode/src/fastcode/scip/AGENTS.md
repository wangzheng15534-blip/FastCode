# scip

SCIP model, loading, indexing, and IR adapter layer.

- Owns SCIP dataclasses, JSON loading, indexer selection, module/symbol
  resolution, and translation between SCIP/AST facts and IR.
- Prefer explicit adapters such as `scip_adapter.py`, `ast_adapter.py`, and
  `resolution_bridge.py` over generic payload plumbing.
- Do not import query, store, API, MCP, or main orchestration.
- Keep generated protobuf concerns isolated in `pb2.py` and loader/transform
  boundaries.
- Focused tests live under `fastcode/tests/scip/`.
