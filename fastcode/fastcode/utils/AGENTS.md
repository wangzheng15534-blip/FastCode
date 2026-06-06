# utils

Shared stdlib-only helpers.

- Owns archive safety, clocks, filesystem helpers, hashing, JSON repair,
  materialization counters, path helpers, and text utilities.
- Keep utilities stdlib-only unless an existing helper already establishes a
  narrow exception.
- Do not import `fastcode.*` from here; utilities are below all runtime layers.
- Do not read env, open network connections, spawn processes, or depend on
  Pydantic.
- Prefer small deterministic functions with explicit inputs and outputs.
- Focused tests live under `fastcode/tests/utils/`.
