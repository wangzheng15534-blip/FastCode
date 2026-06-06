# main

Composition root, user entrypoints, and persistent config ingress.

- Owns CLI wiring, persistent config loading, merge, validation, mapping, and
  the `FastCode` runtime object.
- Raw YAML and `.env` input enters only here, then flows through
  `fastcode.main.config.prepare_runtime_config_mapping(...)` and
  `fastcode.main.config.config_from_mapping(...)` into
  `fastcode.kernel.config.FastCodeConfig`.
- After composition-root mapping, pass typed config to entry frames and let
  internal modules apply their local config and propagate smaller typed config
  or capability handles to the next layer.
- Client/command roots parse arguments, build typed command state, and then
  dispatch to an entry facade or composition-root runtime factory; pure entry
  frames must not call config loaders or read persistent config.
- It is acceptable for `config.py` to use env and dotenv APIs; do not copy those
  reads into inner packages.
- Keep import-time side effects low. Heavy runtime construction belongs behind
  CLI commands or explicit `FastCode` initialization.
- Do not bury pure scoring, parsing, graph, or storage algorithms in
  `fastcode.py`; delegate to owning packages.
- Focused tests live under `fastcode/tests/main/`.
