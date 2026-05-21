# api

HTTP-facing shell for FastCode.

- Owns FastAPI/Flask route wiring, CORS policy, API contracts, and response
  serialization.
- May use Pydantic boundary models and transport-specific error handling.
- Do not put retrieval, graph, indexing, or storage algorithms here; call the
  `FastCode` composition root or explicit service functions.
- Keep request/response mapping explicit. Avoid `**model_dump()` and `**__dict__`
  mass assignment.
- Environment values should arrive through config, not direct env reads in route
  code.
- Focused tests live under `fastcode/tests/api/`.
