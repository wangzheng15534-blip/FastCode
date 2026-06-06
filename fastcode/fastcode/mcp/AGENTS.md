# mcp

MCP transport shell.

- Owns MCP server wiring and tool handlers that adapt requests to FastCode
  services.
- Keep tool output JSON-serializable with explicit response shaping.
- Do not read env directly; receive runtime config through the composition root.
- Do not implement retrieval, graph traversal, or indexing algorithms here.
  Delegate to query, graph, retrieval, store, or main as appropriate.
- Keep startup light with lazy runtime initialization where needed.
- Focused tests live under `fastcode/tests/mcp/`.
