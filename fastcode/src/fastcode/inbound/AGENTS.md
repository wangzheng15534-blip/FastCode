# inbound

Inbound boundary schemas and mappers.

- Own Pydantic DTOs for inbound config shapes and explicit translation into
  frozen internal runtime/app contracts.
- Keep Pydantic in schema modules such as `config_schema.py`; mapper modules
  consume validated DTOs and `fastcode.runtime` contracts.
- Do not import API route implementations, MCP server code, main runtime
  objects, query/indexing/store orchestration, domain packages, or outbound
  infrastructure.
- Do not read env or load config files here. Raw input preparation belongs to
  facade/composition packages.
- Prefer field-explicit mapper functions over `model_dump() -> **kwargs`
  construction.
