# events

Lightweight lifecycle event contracts.

- Keep this package dependency-light so shell and store code can import events
  without side effects.
- Use small immutable or simple typed structures for event payloads.
- Do not import API, MCP, main, query, store, or runtime adapters from here.
- Avoid registration side effects at import time.
- Prefer adding event fields explicitly over carrying untyped payload blobs.
