# client

Default user-facing entry frames and command integrations.

- Owns launchable client commands and entry-frame dispatch for CLI, API server,
  web server, and MCP server modes.
- Parses command-line or process-launch input, then delegates to transport
  shells or the composition root.
- Do not load persistent config or read env here except for narrow process
  startup guards already required by the launched entry mode.
- Do not implement retrieval, graph, indexing, storage, or transport algorithms
  here.
- Focused tests live under `fastcode/tests/client/`.
