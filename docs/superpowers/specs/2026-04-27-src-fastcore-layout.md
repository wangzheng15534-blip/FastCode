# src/fastcore Layout & Test Infrastructure Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the FP core into an independent `src/fastcore` package with proper schema/core/utils/effects separation, restructure tests to mirror src, and add pytest-timeout + pytest-subprocess for reliable test infrastructure.

**Approach:** Move-and-rewrite-imports — physically move `fastcode/core/` and `fastcode/effects/` into `src/fastcore/`, extract types into `schema/`, extract domain-independent functions into `utils/`, update import paths in `fastcode/`.

**Constraint:** Do NOT run original `fastcode/` tests during implementation (reduce test time). Only run the new `tests/fastcore/` tests.

---

## 1. Package Structure

```
src/
  fastcore/
    __init__.py
    schema/                    # All frozen dataclasses + IR types
      __init__.py
      core_types.py            # Hit, FusionConfig, IterationState, SnapshotRecord, etc.
      ir.py                    # IRSnapshot, IRSymbol, IREdge, IROccurrence, IRDocument
    core/                      # Domain-specific pure functions
      __init__.py
      scoring.py               # Score calculation, sigmoid, adaptive params
      fusion.py                # RRF fusion, cross-collection merge
      filtering.py             # apply_filters, diversify, rerank
      combination.py           # combine_results (semantic+keyword+pseudocode)
      iteration.py             # should_continue_iteration, AdaptiveParams
      prompts.py               # format_elements_with_metadata, format_tool_call_history
      context.py               # prepare_context, parse_response_with_summary
      summary.py               # generate_fallback_summary, format_answer_with_sources
      graph_build.py           # build_code_graph_payload
      snapshot.py              # extract_sources_from_elements
      repo_analysis.py         # is_key_file, infer_project_type, generate_structure_based_overview
      scip_transform.py        # symbol_role_to_str, scip_kind_to_str
      boundary.py              # CoreQueryInput, query_request_to_core, hit_to_response
    utils/                     # Domain-independent (copy-paste test)
      __init__.py
      json.py                  # sanitize_json_string, remove_json_comments, extract_json_from_response, robust_json_parse
      hashing.py               # projection_params_hash, deterministic_event_id
      paths.py                 # get_language_from_extension, projection_scope_key
    effects/                   # I/O boundary
      __init__.py
      db.py                    # load_snapshot_record, save_snapshot_record
      llm.py                   # chat_completion
      fs.py                    # read_file, write_file, file_exists
```

### Copy-paste test for utils vs core

A function belongs in `utils/` if it would make sense copy-pasted into a totally different app (Food Delivery, Video Streaming, etc.). Functions that encode domain knowledge about code intelligence stay in `core/`.

| Function | Location | Reason |
|----------|----------|--------|
| `sanitize_json_string` | `utils/json.py` | String sanitization — no domain |
| `extract_json_from_response` | `utils/json.py` | JSON extraction — no domain |
| `robust_json_parse` | `utils/json.py` | JSON parsing — no domain |
| `remove_json_comments` | `utils/json.py` | Comment stripping — no domain |
| `projection_params_hash` | `utils/hashing.py` | Deterministic hashing — no domain |
| `deterministic_event_id` | `utils/hashing.py` | UUID generation — no domain |
| `get_language_from_extension` | `utils/paths.py` | Extension lookup table — no domain |
| `projection_scope_key` | `utils/paths.py` | String key builder — no domain |
| `combine_results` | `core/combination.py` | Domain: merges retrieval channels |
| `build_code_graph_payload` | `core/graph_build.py` | Domain: code graph construction |
| `generate_fallback_summary` | `core/summary.py` | Domain: answer generation |

---

## 2. Schema Extraction

All frozen dataclasses move into `schema/`. Two files:

- **`schema/core_types.py`** — Types from `fastcode/core/types.py`: Hit, FusionConfig, IterationState, SnapshotRecord, AdaptiveParams, CoreQueryInput, GenerateResult, RetrievalResult, etc.
- **`schema/ir.py`** — Types from `fastcode/semantic_ir.py`: IRSnapshot, IRDocument, IRSymbol, IROccurrence, IREdge. These are the canonical IR dataclasses with `to_dict()`/`from_dict()`.

`schema/__init__.py` re-exports all types for convenient importing:
```python
from fastcore.schema.core_types import Hit, FusionConfig, IterationState, ...
from fastcore.schema.ir import IRSnapshot, IRDocument, IRSymbol, IROccurrence, IREdge
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

## 4. Test Layout (Mirrors src)

```
tests/
  fastcore/                  # Mirrors src/fastcore/
    test_schema_types.py     # Schema: core_types dataclasses
    test_schema_ir.py        # Schema: IR dataclasses
    test_core_scoring.py
    test_core_fusion.py
    test_core_filtering.py
    test_core_combination.py
    test_core_iteration.py
    test_core_prompts.py
    test_core_context.py
    test_core_summary.py
    test_core_graph_build.py
    test_core_snapshot.py
    test_core_repo_analysis.py
    test_core_scip_transform.py
    test_core_boundary.py
    test_utils_json.py
    test_utils_hashing.py
    test_utils_paths.py
    test_effects_db.py       # Uses pytest-subprocess for DB doubles
    test_effects_llm.py      # Uses pytest-subprocess for LLM doubles
    test_effects_fs.py       # Uses pytest-subprocess for FS doubles
  property/                  # Existing property tests — unchanged
  snapshots/                 # Existing snapshot tests — unchanged
  bench_*.py                 # Existing benchmarks — unchanged
  # Original fastcode tests (test_ir_core.py, etc.) stay at tests/ root — untouched
```

---

## 5. Migration: fastcode/ Import Updates

After moving code to `src/fastcore/`, the `fastcode/` delegation modules need import path updates:

| Old import | New import |
|------------|------------|
| `from fastcode.core.scoring import X` | `from fastcore.core.scoring import X` |
| `from fastcode.core.types import Hit` | `from fastcore.schema.core_types import Hit` |
| `from fastcode.core.fusion import X` | `from fastcore.core.fusion import X` |
| `from fastcode.core.filtering import X` | `from fastcore.core.filtering import X` |
| `from fastcode.core.combination import X` | `from fastcore.core.combination import X` |
| `from fastcode.core.iteration import X` | `from fastcore.core.iteration import X` |
| `from fastcode.core.parsing import X` | `from fastcore.utils.json import X` |
| `from fastcode.core.prompts import X` | `from fastcore.core.prompts import X` |
| `from fastcode.core.context import X` | `from fastcore.core.context import X` |
| `from fastcode.core.summary import X` | `from fastcore.core.summary import X` |
| `from fastcode.core.graph_build import X` | `from fastcore.core.graph_build import X` |
| `from fastcode.core.snapshot import X` | `from fastcore.core.snapshot import X` (domain parts) / `from fastcore.utils.hashing import X` / `from fastcore.utils.paths import X` |
| `from fastcode.core.repo_analysis import X` | `from fastcore.core.repo_analysis import X` / `from fastcore.utils.paths import get_language_from_extension` |
| `from fastcode.core.scip_transform import X` | `from fastcore.core.scip_transform import X` |
| `from fastcode.core.boundary import X` | `from fastcore.core.boundary import X` |
| `from fastcode.effects.db import X` | `from fastcore.effects.db import X` |
| `from fastcode.effects.llm import X` | `from fastcore.effects.llm import X` |
| `from fastcode.effects.fs import X` | `from fastcore.effects.fs import X` |

Files that need import updates in `fastcode/`:
- `retriever.py` — delegates to core/filtering, core/combination
- `iterative_agent.py` — delegates to core/iteration, core/prompts, core/parsing
- `answer_generator.py` — delegates to core/context, core/summary
- `terminus_publisher.py` — delegates to core/graph_build
- `main.py` — delegates to core/snapshot
- `repo_overview.py` — delegates to core/repo_analysis
- `scip_loader.py` — delegates to core/scip_transform

---

## 6. Workspace Configuration

`src/fastcore` becomes a uv workspace member. Add to root `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["nanobot", "src/fastcore"]
```

Create `src/fastcore/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fastcore"
version = "0.1.0"
description = "Pure functional core for code intelligence"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["."]
include = ["fastcore*"]
```

Root `pyproject.toml` adds dependency on fastcore:

```toml
[tool.uv.sources]
nanobot-ai = { workspace = true }
fastcore = { workspace = true }

dependencies = [
    # ... existing ...
    "fastcore",
]
```
