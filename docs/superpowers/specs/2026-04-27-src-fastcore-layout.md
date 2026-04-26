# fastcode Internal Reorganization & Test Infrastructure

**Goal:** Reorganize `fastcode/` internally — extract `schema/` and `utils/` from `core/`, archive original as `_fastcode/`. Single package, single `pyproject.toml`. Add pytest-timeout + pytest-subprocess.

**Approach:** Archive `fastcode/` → `_fastcode/`, create new `fastcode/` with reorganized internal layout. All imports stay `from fastcode.xxx`. No package rename, no workspace splitting. `fastcode` is the main workspace. `nanobot` is a vendored upstream dep, unrelated.

**Constraint:** Do NOT run original `tests/` suite during implementation. Only run `tests/fastcore/`.

---

## 1. Package Structure

```
fastcode/
  schema/                     # NEW: all frozen dataclasses + IR types
    __init__.py
    core_types.py             # Hit, FusionConfig, IterationState, SnapshotRecord, etc.
    ir.py                     # IRSnapshot, IRSymbol, IREdge, IROccurrence, IRDocument
  core/                       # Domain-specific pure functions (stays)
    scoring.py                # Score calculation, sigmoid, adaptive params
    fusion.py                 # RRF fusion, cross-collection merge
    filtering.py              # apply_filters, diversify, rerank
    combination.py            # combine_results
    iteration.py              # should_continue_iteration, AdaptiveParams
    prompts.py                # format_elements_with_metadata, format_tool_call_history
    context.py                # prepare_context, parse_response_with_summary
    summary.py                # generate_fallback_summary, format_answer_with_sources
    graph_build.py            # build_code_graph_payload
    snapshot.py               # extract_sources_from_elements (utils fns extracted)
    repo_analysis.py          # is_key_file, infer_project_type, etc. (utils fns extracted)
    scip_transform.py         # symbol_role_to_str, scip_kind_to_str
    boundary.py               # CoreQueryInput, query_request_to_core, hit_to_response
  utils/                      # NEW: domain-independent (copy-paste test)
    json.py                   # safe_jsonable + extract_json_from_response, sanitize_json_string, remove_json_comments, robust_json_parse
    hashing.py                # projection_params_hash, deterministic_event_id
    paths.py                  # get_language_from_extension, projection_scope_key
  effects/                    # I/O boundary (stays)
    db.py                     # load_snapshot_record, save_snapshot_record
    llm.py                    # chat_completion
    fs.py                     # read_file, write_file, file_exists
  main.py, retriever.py, ... # App modules (unchanged)
```

### Copy-paste test for utils vs core

A function belongs in `utils/` if it would make sense copy-pasted into a totally different app. Functions encoding domain knowledge about code intelligence stay in `core/`.

| Function | Location | Reason |
|----------|----------|--------|
| `safe_jsonable` | `utils/json.py` | Recursive serialization — no domain |
| `sanitize_json_string` | `utils/json.py` | String sanitization — no domain |
| `extract_json_from_response` | `utils/json.py` | JSON extraction — no domain |
| `robust_json_parse` | `utils/json.py` | JSON parsing — no domain |
| `remove_json_comments` | `utils/json.py` | Comment stripping — no domain |
| `projection_params_hash` | `utils/hashing.py` | Deterministic hashing — no domain |
| `deterministic_event_id` | `utils/hashing.py` | UUID generation — no domain |
| `get_language_from_extension` | `utils/paths.py` | Extension lookup — no domain |
| `projection_scope_key` | `utils/paths.py` | String key builder — no domain |
| `combine_results` | `core/combination.py` | Domain: merges retrieval channels |
| `build_code_graph_payload` | `core/graph_build.py` | Domain: code graph construction |

---

## 2. Pytest Infrastructure

### New dev dependencies

```toml
"pytest-timeout",       # Per-test timeout enforcement
"pytest-subprocess",    # fp fixture for subprocess test-doubles
```

### Configuration

```toml
[tool.pytest.ini_options]
timeout = 30
timeout_method = "thread"
```

---

## 3. Test Layout (Mirrors fastcode/)

```
tests/
  fastcore/
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
  ...existing app tests...
```

---

## 4. Migration: Import Updates

| Old import | New import |
|------------|------------|
| `from fastcode.core.types import X` | `from fastcode.schema.core_types import X` |
| `from fastcode.core.parsing import X` | `from fastcode.utils.json import X` |
| `from fastcode.core.snapshot import projection_params_hash` | `from fastcode.utils.hashing import projection_params_hash` |
| `from fastcode.core.snapshot import projection_scope_key` | `from fastcode.utils.paths import projection_scope_key` |
| `from fastcode.core.repo_analysis import get_language_from_extension` | `from fastcode.utils.paths import get_language_from_extension` |
| `from fastcode.core.graph_build import deterministic_event_id` | `from fastcode.utils.hashing import deterministic_event_id` |
| `from ..semantic_ir import _resolution_to_confidence` | `from fastcode.schema.ir import _resolution_to_confidence` |

All other `from fastcode.core.X import Y` stay unchanged — modules remain in `core/`.
