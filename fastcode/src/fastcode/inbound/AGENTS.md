# inbound

Inbound boundary mappers.

- Own explicit translation from validated external schemas/DTOs into frozen
  internal runtime/app contracts.
- May import `fastcode.schemas` DTOs and `fastcode.runtime` contracts.
- Do not import API route implementations, MCP server code, main runtime
  objects, query/indexing/store orchestration, domain packages, or outbound
  infrastructure.
- Do not read env or load config files here. Raw input preparation belongs to
  facade/composition packages; validation belongs to `schemas`.
- Prefer field-explicit mapper functions over `model_dump() -> **kwargs`
  construction.
