# runtime

Runtime contracts shared by shell and boundary adapter code.

- Own frozen runtime config dataclasses and runtime lifecycle event dataclasses.
- Do not import Pydantic, dotenv, API/MCP/main entrypoints, shell orchestration,
  storage orchestration, domain packages, or external SDK clients.
- Do not read env or load config files here. Runtime config values must arrive
  already resolved by `fastcode.main.config` and validated by inbound DTOs.
- Keep `__init__.py` marker-only; import concrete modules such as
  `fastcode.runtime.config` and `fastcode.runtime.events`.
