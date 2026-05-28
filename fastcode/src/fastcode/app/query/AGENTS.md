# query

Query orchestration shell.

- Owns query processing, retrieval orchestration, selector flow, agent tools,
  iterative agent behavior, token handling, and LLM answer generation.
- May call retrieval domain logic, store backends, and LLM adapters, but should
  keep pure scoring and fusion rules in `retrieval/`.
- Direct env reads are forbidden. Provider credentials and runtime knobs arrive
  through config or explicit adapters.
- Keep persisted context, journals, working memory, and distillation payloads
  mapped field by field.
- Avoid adding new generic `from_dict()` or `to_dict()` round trips on hot
  retrieval paths.
- Focused tests live under `fastcode/tests/query/`.
