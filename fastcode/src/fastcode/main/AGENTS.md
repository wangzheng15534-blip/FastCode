# main

Composition root and user entrypoints.

- Owns CLI wiring, config loading/preparation, and the `FastCode` runtime object.
- Raw YAML and `.env` input enters here, then flows through
  `fastcode.schemas.config.FastCodeConfigDTO` and
  `fastcode.inbound.config_mapper.config_from_mapping(...)` into
  `fastcode.runtime.config.FastCodeConfig`.
- It is acceptable for `config.py` to use env and dotenv APIs; do not copy those
  reads into inner packages.
- Keep import-time side effects low. Heavy runtime construction belongs behind
  CLI commands or explicit `FastCode` initialization.
- Do not bury pure scoring, parsing, graph, or storage algorithms in
  `fastcode.py`; delegate to owning packages.
- Focused tests live under `fastcode/tests/main/`.
