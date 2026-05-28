# main

Composition root, user entrypoints, and config shaping.

- Owns CLI wiring, config loading/preparation, config DTO validation, and the
  `FastCode` runtime object.
- Raw YAML and `.env` input enters here, then flows through
  `fastcode.main.config_schema.FastCodeConfigDTO` and
  `fastcode.main.config_mapper.config_from_mapping(...)` into
  `fastcode.kernel.config.FastCodeConfig`.
- It is acceptable for `config.py` to use env and dotenv APIs; do not copy those
  reads into inner packages.
- Keep import-time side effects low. Heavy runtime construction belongs behind
  CLI commands or explicit `FastCode` initialization.
- Do not bury pure scoring, parsing, graph, or storage algorithms in
  `fastcode.py`; delegate to owning packages.
- Focused tests live under `fastcode/tests/main/`.
