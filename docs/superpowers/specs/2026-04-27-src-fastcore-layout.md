# Cargo-Style Workspace Layout & Test Infrastructure Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the FP core into a `fastcode-core` workspace member with proper schema/core/utils/effects separation, archive original `fastcode/` as `_fastcode/`, restructure tests to mirror src inside workspace members, and add pytest-timeout + pytest-subprocess for reliable test infrastructure.

**Approach:** Cargo-style uv workspace monorepo. Archive original `fastcode/` to `_fastcode/` (prefix `_` excludes from Python). Create `libs/core/` workspace member (`fastcode-core` package). Root `pyproject.toml` is workspace root AND the `fastcode` app. No `pkgutil.extend_path` — package names differ (`fastcode` vs `fastcode_core`).

**Constraint:** Do NOT run original `tests/` suite during implementation (reduce test time). Only run `libs/core/tests/` to verify.

---

## Cargo ↔ uv Workspace Analogy

| Feature | Rust (Cargo) | Modern Python (uv) |
|---------|-------------|-------------------|
| Root Config | `Cargo.toml` (`[workspace]`) | `pyproject.toml` (`[tool.uv.workspace]`) |
| Lockfile | Single `Cargo.lock` | Single `uv.lock` |
| Package Config | `Cargo.toml` in every crate | `pyproject.toml` in every package |
| Local Dep | `{ path = "../libs/core" }` | `dependencies = ["fastcode-core"]` (uv resolves locally) |

---

## 1. Package Structure

```
libs/core/                         # Workspace member: fastcode-core
  pyproject.toml                   # name = "fastcode-core"
  src/
    fastcode_core/
      __init__.py
      schema/                      # All frozen dataclasses + IR types
        __init__.py
        core_types.py              # Hit, FusionConfig, IterationState, SnapshotRecord, etc.
        ir.py                      # IRSnapshot, IRSymbol, IREdge, IROccurrence, IRDocument
      core/                        # Domain-specific pure functions
        __init__.py
        scoring.py                 # Score calculation, sigmoid, adaptive params
        fusion.py                  # RRF fusion, cross-collection merge
        filtering.py               # apply_filters, diversify, rerank
        combination.py             # combine_results (semantic+keyword+pseudocode)
        iteration.py               # should_continue_iteration, AdaptiveParams
        prompts.py                 # format_elements_with_metadata, format_tool_call_history
        context.py                 # prepare_context, parse_response_with_summary
        summary.py                 # generate_fallback_summary, format_answer_with_sources
        graph_build.py             # build_code_graph_payload
        snapshot.py                # extract_sources_from_elements
        repo_analysis.py           # is_key_file, infer_project_type, generate_structure_based_overview
        scip_transform.py          # symbol_role_to_str, scip_kind_to_str
        boundary.py                # CoreQueryInput, query_request_to_core, hit_to_response
      utils/                       # Domain-independent (copy-paste test)
        __init__.py
        json.py                    # sanitize_json_string, remove_json_comments, extract_json_from_response, robust_json_parse, safe_jsonable
        hashing.py                 # projection_params_hash, deterministic_event_id
        paths.py                   # get_language_from_extension, projection_scope_key
      effects/                     # I/O boundary
        __init__.py
        db.py                      # load_snapshot_record, save_snapshot_record
        llm.py                     # chat_completion
        fs.py                      # read_file, write_file, file_exists
  tests/                           # Tests mirror src/ inside the workspace member
    schema/
      test_core_types.py
      test_ir.py
    core/
      test_scoring.py
      test_fusion.py
      ... (mirrors core/ exactly)
    utils/
      test_json.py
      test_hashing.py
      test_paths.py
    effects/
      test_db.py
      test_llm.py
      test_fs.py
```

### Copy-paste test for utils vs core

A function belongs in `utils/` if it would make sense copy-pasted into a totally different app (Food Delivery, Video Streaming, etc.). Functions that encode domain knowledge about code intelligence stay in `core/`.

| Function | Location | Reason |
|----------|----------|--------|
| `sanitize_json_string` | `utils/json.py` | String sanitization — no domain |
| `extract_json_from_response` | `utils/json.py` | JSON extraction — no domain |
| `robust_json_parse` | `utils/json.py` | JSON parsing — no domain |
| `remove_json_comments` | `utils/json.py` | Comment stripping — no domain |
| `safe_jsonable` | `utils/json.py` | Recursive serialization — no domain |
| `projection_params_hash` | `utils/hashing.py` | Deterministic hashing — no domain |
| `deterministic_event_id` | `utils/hashing.py` | UUID generation — no domain |
| `get_language_from_extension` | `utils/paths.py` | Extension lookup table — no domain |
| `projection_scope_key` | `utils/paths.py` | String key builder — no domain |
| `combine_results` | `core/combination.py` | Domain: merges retrieval channels |
| `build_code_graph_payload` | `core/graph_build.py` | Domain: code graph construction |
| `generate_fallback_summary` | `core/summary.py` | Domain: answer generation |

---

## 2. Workspace Configuration

### Root `pyproject.toml`

```toml
[tool.uv.workspace]
members = ["nanobot", "libs/core"]

[tool.uv.sources]
nanobot-ai = { workspace = true }
fastcode-core = { workspace = true }

# dependencies = [..., "fastcode-core"]

[tool.setuptools.packages.find]
include = ["fastcode*"]
# Finds root fastcode/ (the app). _fastcode/ is excluded by _ prefix.
```

### `libs/core/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fastcode-core"
version = "0.1.0"
description = "Pure functional core for code intelligence"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]
include = ["fastcode_core*"]
```

---

## 3. Pytest Infrastructure

### New dev dependencies

```toml
[project.optional-dependencies]
dev = [
    # ... existing ...
    "pytest-timeout",       # Per-test timeout enforcement
    "pytest-subprocess",    # fp fixture for subprocess test-doubles
]
```

### Configuration

```toml
[tool.pytest.ini_options]
timeout = 30               # Default: 30s per test
timeout_method = "thread"  # Thread-based (compatible with xdist)
```

### Usage in effects tests

Effects-layer tests use `pytest-subprocess`'s `fp` fixture to simulate infrastructure failures:
- Database connection timeout → subprocess double returns delayed response
- CLI tool crash → subprocess double returns non-zero exit code
- Network hang → subprocess double blocks indefinitely (caught by pytest-timeout)

Core-layer tests do NOT use subprocess doubles — they test pure data transformations only.

---

## 4. Migration: fastcode/ Import Updates

After moving code to `fastcode-core`, the root `fastcode/` app files need import path updates:

| Old import | New import |
|------------|------------|
| `from .core.types import Hit` | `from fastcode_core.schema.core_types import Hit` |
| `from .core.scoring import ...` | `from fastcode_core.core.scoring import ...` |
| `from .core.fusion import ...` | `from fastcode_core.core.fusion import ...` |
| `from .core.filtering import ...` | `from fastcode_core.core.filtering import ...` |
| `from .core.combination import ...` | `from fastcode_core.core.combination import ...` |
| `from .core.iteration import ...` | `from fastcode_core.core.iteration import ...` |
| `from .core.parsing import ...` | `from fastcode_core.utils.json import ...` |
| `from .core.prompts import ...` | `from fastcode_core.core.prompts import ...` |
| `from .core.context import ...` | `from fastcode_core.core.context import ...` |
| `from .core.summary import ...` | `from fastcode_core.core.summary import ...` |
| `from .core.graph_build import ...` | `from fastcode_core.core.graph_build import ...` |
| `from .core.snapshot import ...` | `from fastcode_core.core.snapshot import ...` |
| `from .core.repo_analysis import ...` | `from fastcode_core.core.repo_analysis import ...` |
| `from .core.scip_transform import ...` | `from fastcode_core.core.scip_transform import ...` |
| `from .core.boundary import ...` | `from fastcode_core.core.boundary import ...` |

Files that need import updates in `fastcode/`:
- `retriever.py` — delegates to core/filtering, core/combination
- `iterative_agent.py` — delegates to core/iteration, core/prompts, core/parsing
- `answer_generator.py` — delegates to core/context, core/summary
- `terminus_publisher.py` — delegates to core/graph_build
- `main.py` — delegates to core/snapshot
- `repo_overview.py` — delegates to core/repo_analysis
- `scip_loader.py` — delegates to core/scip_transform
