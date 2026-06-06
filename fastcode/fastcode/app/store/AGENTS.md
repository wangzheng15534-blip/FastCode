# store

Persistence, snapshots, vectors, cache, and records.

- Owns snapshot storage, manifests, vector stores, projection state, index-run
  state, cache, file/unit artifacts, and typed persistence records.
- Map database rows through explicit adapters and record types. Do not add
  generic `row_to_dict()` hot-path conversions.
- Keep vector insertion/search paths in NumPy or backend-native form until JSON,
  pickle, SQL, or API boundaries require conversion.
- Store orchestration may use `store/infrastructure/`; lower-level runtime code
  follows the stricter rules in `store/infrastructure/AGENTS.md`.
- Shared external capability contracts live in `fastcode.ports`.
  Prefer ports such as `StoreDatabaseRuntime`, `FileArtifactStore`, and
  `UnitArtifactStore` over importing concrete infrastructure runtimes or
  storage adapters from clients.
- Do not import API, MCP, main, or query orchestration from store modules.
- Focused tests live under `fastcode/tests/store/`.
