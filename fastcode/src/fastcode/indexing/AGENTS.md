# indexing

Repository ingestion and indexing shell.

- Owns repository loading, file inventory, parsing, embedding, SCIP runs,
  projection builds, publishing, and redo orchestration.
- This package may orchestrate IR, graph, scip, semantic, store, and query
  components, but should not hide domain logic in shell glue.
- Direct env reads and `load_dotenv()` are forbidden; receive settings through
  `FastCodeConfig` or explicit config adapters.
- Keep subprocess/tool invocation in dedicated runner modules such as
  `scip_runner.py` and `semantic_helper_runner.py`.
- Preserve typed records and native vector carriers on hot paths until a real
  persistence or API boundary requires conversion.
- Focused tests live under `fastcode/tests/indexing/`.
