# semantic

Semantic resolver registry and helper-backed upgrades.

- Owns resolver contracts, symbol indexing, language resolver registry, and
  patch objects that enrich IR facts.
- Keep helper process integration inside resolver/helper modules; callers should
  receive typed diagnostics and patches.
- Do not import query, store, API, MCP, or main orchestration from semantic
  domain code.
- Prefer graph-backed or language-specific resolver contracts over ad hoc dict
  payloads.
- Focused tests live under `fastcode/tests/semantic/`.
