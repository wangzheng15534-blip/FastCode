# fastcode Internal Reorganization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `fastcode/` internally — extract schema/, utils/ from core/, archive original as `_fastcode/` to avoid interpreter ambiguity. Single package, single `pyproject.toml`. Add pytest-timeout + pytest-subprocess.

**Architecture:** Archive `fastcode/` → `_fastcode/`, create new `fastcode/` with reorganized internal layout. All imports stay `from fastcode.xxx` — no package rename, no workspace splitting. `fastcode` IS the main workspace. `nanobot` is a vendored upstream dep (already a workspace member), unrelated to this reorganization.

**Tech Stack:** Python 3.11+, uv, setuptools, pytest-timeout, pytest-subprocess

**Constraint:** Do NOT run the full `tests/` suite during implementation. Only run `libs/core/tests/` → `tests/fastcore/` to verify.

---

## Final Directory Structure

```
pyproject.toml                    # Single package config (fastcode)
uv.lock                           # Single lockfile
_fastcode/                        # Archived original (prefix _ excludes from Python)
fastcode/                         # Reorganized package
  __init__.py
  schema/                         # NEW: all frozen dataclasses + IR types
    __init__.py
    core_types.py                 # Hit, FusionConfig, IterationState, etc.
    ir.py                         # IRSnapshot, IRSymbol, IREdge, etc.
  core/                           # Domain-specific pure functions (stays)
    __init__.py
    scoring.py
    fusion.py
    filtering.py
    combination.py
    iteration.py
    prompts.py
    context.py
    summary.py
    graph_build.py
    snapshot.py                   # (utils fns removed to utils/)
    repo_analysis.py              # (utils fns removed to utils/)
    scip_transform.py
    boundary.py
  utils/                          # NEW: domain-independent (copy-paste test)
    __init__.py
    json.py                       # safe_jsonable + LLM JSON parsing
    hashing.py                    # projection_params_hash, deterministic_event_id
    paths.py                      # get_language_from_extension, projection_scope_key
  effects/                        # I/O boundary (stays)
    __init__.py
    db.py
    llm.py
    fs.py
  main.py                         # App modules (unchanged)
  retriever.py                    # (imports updated)
  iterative_agent.py              # (imports updated)
  answer_generator.py             # (imports updated)
  terminus_publisher.py           # (imports updated)
  ...all other app modules...
nanobot/                          # Vendored upstream dep (unrelated)
tests/
  fastcore/                       # NEW: tests for schema/core/utils/effects
    schema/
      test_core_types.py
      test_ir.py
    core/
      test_scoring.py
      test_fusion.py
      ...
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

## Import Changes

All imports stay `from fastcode.xxx` — only the internal paths change:

| Old import | New import |
|------------|------------|
| `from fastcode.core.types import Hit` | `from fastcode.schema.core_types import Hit` |
| `from fastcode.core.types import FusionConfig` | `from fastcode.schema.core_types import FusionConfig` |
| `from fastcode.core.types import IterationConfig` | `from fastcode.schema.core_types import IterationConfig` |
| `from fastcode.core.types import SnapshotRecord` | `from fastcode.schema.core_types import SnapshotRecord` |
| `from fastcode.core.parsing import ...` | `from fastcode.utils.json import ...` |
| `from fastcode.core.snapshot import projection_params_hash` | `from fastcode.utils.hashing import projection_params_hash` |
| `from fastcode.core.snapshot import projection_scope_key` | `from fastcode.utils.paths import projection_scope_key` |
| `from fastcode.core.repo_analysis import get_language_from_extension` | `from fastcode.utils.paths import get_language_from_extension` |
| `from fastcode.core.graph_build import deterministic_event_id` | `from fastcode.utils.hashing import deterministic_event_id` |
| `from ..semantic_ir import _resolution_to_confidence` | `from fastcode.schema.ir import _resolution_to_confidence` |

All other `from fastcode.core.X import Y` imports stay the same — the modules stay in `fastcode/core/`.

---

## Task 1: Archive & Scaffold

**Files:**
- Create: `fastcode/schema/`, `fastcode/utils/` directories
- Modify: `pyproject.toml`

- [ ] **Step 1: Archive original `fastcode/` as `_fastcode/`**

```bash
git mv fastcode/ _fastcode/
```

- [ ] **Step 2: Create new `fastcode/` with all app files minus core/ and effects/ dirs**

```bash
mkdir -p fastcode/schema fastcode/utils
# Copy all root-level .py files
for f in _fastcode/*.py; do
  cp "$f" "fastcode/$(basename "$f")"
done
# Copy core/ and effects/ as-is (imports updated later)
cp -r _fastcode/core fastcode/core
cp -r _fastcode/effects fastcode/effects
# Remove __pycache__ from copies
find fastcode/ -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
```

- [ ] **Step 3: Create new subpackage `__init__.py` files**

`fastcode/schema/__init__.py`:
```python
"""fastcode.schema — all frozen dataclasses and IR types."""
```

`fastcode/utils/__init__.py`:
```python
"""fastcode.utils — domain-independent utilities (copy-paste test)."""
```

- [ ] **Step 4: Update `pyproject.toml` — add pytest plugins and timeout config**

Add to dev deps:
```toml
[project.optional-dependencies]
dev = [
    # ... existing entries ...
    "pytest-timeout",
    "pytest-subprocess",
]
```

Add to pytest config:
```toml
[tool.pytest.ini_options]
timeout = 30
timeout_method = "thread"
# ... keep existing testpaths, asyncio_mode, addopts, markers ...
```

No workspace changes needed — `fastcode` is already the main package.

- [ ] **Step 5: Run `uv sync` and verify**

```bash
uv sync
uv run python -c "import fastcode; print(fastcode.__file__)"
```

Expected: prints path to new `fastcode/__init__.py`

- [ ] **Step 6: Commit**

```bash
git add _fastcode/ fastcode/ pyproject.toml uv.lock
git commit -m "chore: archive fastcode/ to _fastcode/, scaffold new fastcode/ with schema/ and utils/ subpackages"
```

---

## Task 2: Schema — Core Types

**Files:**
- Create: `fastcode/schema/core_types.py`
- Modify: `fastcode/schema/__init__.py`

- [ ] **Step 1: Copy `_fastcode/core/types.py` → `fastcode/schema/core_types.py`**

```bash
cp _fastcode/core/types.py fastcode/schema/core_types.py
```

No import changes needed — only uses `dataclasses` and `typing` (stdlib).

- [ ] **Step 2: Delete old `fastcode/core/types.py`**

```bash
rm fastcode/core/types.py
```

- [ ] **Step 3: Update `fastcode/schema/__init__.py` with re-exports**

```python
"""fastcode.schema — all frozen dataclasses and IR types."""

from fastcode.schema.core_types import (
    ElementFilter,
    FileAnalysis,
    FusionConfig,
    FusionWeights,
    GenerationInput,
    GenerationResult,
    Hit,
    IterationConfig,
    IterationHistoryEntry,
    IterationMetrics,
    IterationState,
    RepoStructure,
    RetrievalChannelOutput,
    RoundResult,
    ScipKind,
    ScipRole,
    SnapshotRecord,
    SourceRef,
    ToolCall,
)

__all__ = [
    "ElementFilter",
    "FileAnalysis",
    "FusionConfig",
    "FusionWeights",
    "GenerationInput",
    "GenerationResult",
    "Hit",
    "IterationConfig",
    "IterationHistoryEntry",
    "IterationMetrics",
    "IterationState",
    "RepoStructure",
    "RetrievalChannelOutput",
    "RoundResult",
    "ScipKind",
    "ScipRole",
    "SnapshotRecord",
    "SourceRef",
    "ToolCall",
]
```

- [ ] **Step 4: Update all `from .types import ...` → `from fastcode.schema.core_types import ...` in core/ modules**

Files that import from `.types`:
- `fastcode/core/fusion.py`: `from .types import FusionConfig` → `from fastcode.schema.core_types import FusionConfig`
- `fastcode/core/iteration.py`: `from .types import IterationConfig` → `from fastcode.schema.core_types import IterationConfig`
- `fastcode/core/boundary.py`: `from fastcode.core.types import Hit` → `from fastcode.schema.core_types import Hit`
- `fastcode/effects/db.py`: `from fastcode.core.types import SnapshotRecord` → `from fastcode.schema.core_types import SnapshotRecord`

- [ ] **Step 5: Verify**

```bash
uv run python -c "from fastcode.schema.core_types import Hit; print(Hit)"
uv run python -c "from fastcode.core.fusion import adaptive_fuse_channels; print('fusion OK')"
uv run python -c "from fastcode.core.boundary import hit_to_response; print('boundary OK')"
```

- [ ] **Step 6: Commit**

```bash
git add fastcode/schema/ fastcode/core/ fastcode/effects/
git commit -m "feat: extract core/types.py → schema/core_types.py, update imports"
```

---

## Task 3: Schema — IR Types

**Files:**
- Create: `fastcode/schema/ir.py`
- Modify: `fastcode/schema/__init__.py`

- [ ] **Step 1: Create `fastcode/schema/ir.py`**

Copy from `_fastcode/semantic_ir.py`:
- Helper functions: `_sorted_set`, `_normalize_set`, `_resolution_to_confidence`, `_confidence_to_resolution`, `_unit_kind_to_symbol_kind`, `_symbol_kind_to_unit_kind`
- Dataclasses: `IRDocument`, `IRSymbol`, `IROccurrence`, `IREdge`, `IRAttachment`, `IRSnapshot`

Change the import at the top:
```python
# OLD: from .utils import safe_jsonable
# NEW: from fastcode.utils.json import safe_jsonable
```
(will be created in Task 4 — verify after Task 4)

Do NOT copy: `IRCodeUnit`, `IRUnitSupport`, `IRRelation`, `IRUnitEmbedding` — these stay in `fastcode/semantic_ir.py`.

- [ ] **Step 2: Update `fastcode/schema/__init__.py` — add IR re-exports**

Append:
```python
from fastcode.schema.ir import (
    IRAttachment,
    IRDocument,
    IREdge,
    IROccurrence,
    IRSnapshot,
    IRSymbol,
)

__all__ += [
    "IRAttachment",
    "IRDocument",
    "IREdge",
    "IROccurrence",
    "IRSnapshot",
    "IRSymbol",
]
```

- [ ] **Step 3: Update `from ..semantic_ir import _resolution_to_confidence` in `fastcode/core/graph_build.py`**

```python
# OLD:
from ..semantic_ir import _resolution_to_confidence
# NEW:
from fastcode.schema.ir import _resolution_to_confidence
```

- [ ] **Step 4: Commit**

```bash
git add fastcode/schema/ fastcode/core/graph_build.py
git commit -m "feat: extract IR dataclasses → schema/ir.py, update graph_build import"
```

---

## Task 4: Utils Extraction

**Files:**
- Create: `fastcode/utils/json.py`, `fastcode/utils/hashing.py`, `fastcode/utils/paths.py`
- Modify: `fastcode/utils/__init__.py`
- Modify: `fastcode/core/snapshot.py`, `fastcode/core/repo_analysis.py`, `fastcode/core/graph_build.py`

- [ ] **Step 1: Create `fastcode/utils/json.py`**

Copy `safe_jsonable` from `_fastcode/utils.py` (lines 288-323) and all four functions from `_fastcode/core/parsing.py`:
- `extract_json_from_response`
- `sanitize_json_string`
- `remove_json_comments`
- `robust_json_parse`

All are pure stdlib (ast, json, re, typing) — no external dependencies.

- [ ] **Step 2: Create `fastcode/utils/hashing.py`**

Copy from `_fastcode/core/snapshot.py` and `_fastcode/core/graph_build.py`:
- `projection_params_hash` (from snapshot.py)
- `deterministic_event_id` (from graph_build.py)

```python
"""Domain-independent hashing utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def projection_params_hash(scope_dict: dict[str, Any], version: str = "v1") -> str:
    payload = json.dumps(
        {"scope": scope_dict, "projection_algo_version": version},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def deterministic_event_id(snapshot_id: str, payload: str) -> str:
    h = hashlib.sha256(f"{snapshot_id}:{payload}".encode()).hexdigest()[:32]
    return f"outbox:{snapshot_id}:{h}"
```

- [ ] **Step 3: Create `fastcode/utils/paths.py`**

Copy from `_fastcode/core/snapshot.py` and `_fastcode/core/repo_analysis.py`:
- `projection_scope_key` (from snapshot.py)
- `get_language_from_extension` (from repo_analysis.py)

- [ ] **Step 4: Update `fastcode/utils/__init__.py`**

```python
"""fastcode.utils — domain-independent utilities (copy-paste test)."""

from fastcode.utils.json import (
    extract_json_from_response,
    robust_json_parse,
    safe_jsonable,
    sanitize_json_string,
)
from fastcode.utils.hashing import deterministic_event_id, projection_params_hash
from fastcode.utils.paths import get_language_from_extension, projection_scope_key

__all__ = [
    "deterministic_event_id",
    "extract_json_from_response",
    "get_language_from_extension",
    "projection_params_hash",
    "projection_scope_key",
    "robust_json_parse",
    "safe_jsonable",
    "sanitize_json_string",
]
```

- [ ] **Step 5: Remove extracted functions from their original locations**

In `fastcode/core/snapshot.py` — remove `projection_scope_key` and `projection_params_hash`. Keep only `extract_sources_from_elements`. New file:
```python
"""Pure snapshot logic — domain-specific parts."""

from __future__ import annotations

from typing import Any


def extract_sources_from_elements(
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for elem_data in elements:
        elem = elem_data.get("element", {})
        sources.append(
            {
                "file": elem.get("relative_path", ""),
                "repo": elem.get("repo_name", ""),
                "type": elem.get("type", ""),
                "name": elem.get("name", ""),
                "start_line": elem.get("start_line", 0),
                "end_line": elem.get("end_line", 0),
            }
        )
    return sources
```

In `fastcode/core/repo_analysis.py` — remove `get_language_from_extension`. Add `from fastcode.utils.paths import get_language_from_extension` at top. Keep `is_key_file`, `infer_project_type`, `generate_structure_based_overview`, `format_file_structure`.

In `fastcode/core/graph_build.py` — remove `deterministic_event_id`. Add `from fastcode.utils.hashing import deterministic_event_id` at top. Keep `build_code_graph_payload`.

Delete `fastcode/core/parsing.py` — all functions moved to `fastcode/utils/json.py`.

- [ ] **Step 6: Verify all imports**

```bash
uv run python -c "from fastcode.utils.json import safe_jsonable; from fastcode.utils.hashing import deterministic_event_id; from fastcode.utils.paths import get_language_from_extension; print('OK')"
uv run python -c "from fastcode.schema.ir import IRSnapshot; print('IR OK')"
uv run python -c "from fastcode.core.snapshot import extract_sources_from_elements; from fastcode.core.graph_build import build_code_graph_payload; print('core OK')"
```

- [ ] **Step 7: Commit**

```bash
git add fastcode/utils/ fastcode/core/snapshot.py fastcode/core/repo_analysis.py fastcode/core/graph_build.py
git rm fastcode/core/parsing.py 2>/dev/null || true
git commit -m "feat: extract utils/ (json, hashing, paths) from core/, delete parsing.py"
```

---

## Task 5: App Import Updates

**Files:**
- Modify: `fastcode/retriever.py`, `fastcode/iterative_agent.py`, `fastcode/answer_generator.py`
- Modify: `fastcode/terminus_publisher.py`, `fastcode/main.py`, `fastcode/repo_overview.py`, `fastcode/scip_loader.py`

- [ ] **Step 1: Update imports in all 7 files**

**`fastcode/retriever.py`** (lines 16-20):
```python
# OLD:
from .core import combination as _combination
from .core import filtering as _filtering
from .core import fusion as _fusion
from .core import scoring as _scoring
from .core.types import FusionConfig
# NEW:
from .core import combination as _combination
from .core import filtering as _filtering
from .core import fusion as _fusion
from .core import scoring as _scoring
from .schema.core_types import FusionConfig
```

**`fastcode/iterative_agent.py`** (lines 16-19):
```python
# OLD:
from .core import iteration as _iteration
from .core import parsing as _parsing
from .core import prompts as _prompts
from .core.types import IterationConfig
# NEW:
from .core import iteration as _iteration
from .utils import json as _json_utils
from .core import prompts as _prompts
from .schema.core_types import IterationConfig
```
Find all `_parsing.` usages and replace with `_json_utils.`.

**`fastcode/answer_generator.py`** (lines 15-16): no change needed — `from .core import context/summary` stays.

**`fastcode/terminus_publisher.py`** (line 13): no change needed — `from .core import graph_build` stays.

**`fastcode/main.py`** (line 25): no change needed — `from .core import snapshot` stays.

**`fastcode/repo_overview.py`** (line 14): no change needed — `from .core import repo_analysis` stays.

**`fastcode/scip_loader.py`** (line 13): no change needed — `from fastcode.core import scip_transform` stays.

- [ ] **Step 2: Verify fastcode imports**

```bash
uv run python -c "from fastcode.retriever import HybridRetriever; print('retriever OK')"
uv run python -c "from fastcode.iterative_agent import IterativeAgent; print('iterative_agent OK')"
```

- [ ] **Step 3: Commit**

```bash
git add fastcode/retriever.py fastcode/iterative_agent.py
git commit -m "refactor: update app imports — types → schema, parsing → utils.json"
```

---

## Task 6: Tests Restructure

**Files:**
- Create: `tests/fastcore/` with test files mirroring new layout

- [ ] **Step 1: Create test directory structure**

```bash
mkdir -p tests/fastcore/{schema,core,utils,effects}
```

- [ ] **Step 2: Copy and update test files**

```bash
# Schema tests
cp tests/test_core_types.py tests/fastcore/schema/test_core_types.py

# Core tests
for f in scoring fusion filtering combination iteration prompts \
         context summary graph_build snapshot repo_analysis \
         scip_transform boundary; do
    cp "tests/test_core_${f}.py" "tests/fastcore/core/test_${f}.py"
done

# Effects tests
for f in db llm fs; do
    cp "tests/test_effects_${f}.py" "tests/fastcore/effects/test_${f}.py"
done

# Utils tests
cp tests/test_core_parsing.py tests/fastcore/utils/test_json.py
```

- [ ] **Step 3: Update imports in all copied test files**

Import mapping for test files:

| Test file | Old | New |
|-----------|-----|-----|
| All core tests | `from fastcode.core.X` | `from fastcode.core.X` (unchanged) |
| All core tests | `from fastcode.core.types import X` | `from fastcode.schema.core_types import X` |
| `test_json.py` | `from fastcode.core.parsing import X` | `from fastcode.utils.json import X` |
| All effects tests | `from fastcode.core.types import X` | `from fastcode.schema.core_types import X` |
| `test_snapshot.py` | `from fastcode.core.snapshot import projection_params_hash` | `from fastcode.utils.hashing import projection_params_hash` |
| `test_snapshot.py` | `from fastcode.core.snapshot import projection_scope_key` | `from fastcode.utils.paths import projection_scope_key` |

**Special: `test_boundary.py`** — update `CORE_DIR` path:
```python
# OLD: ... / "fastcode" / "core"
# NEW: ... / "fastcode" / "core"  (same — core/ didn't move)
```
Actually, core/ stayed in `fastcode/core/`, so paths are unchanged. Only the types import changed.

- [ ] **Step 4: Create `tests/fastcore/utils/test_hashing.py`**

```python
"""Tests for fastcode.utils.hashing."""

from __future__ import annotations

from fastcode.utils.hashing import deterministic_event_id, projection_params_hash


class TestProjectionParamsHash:
    def test_deterministic(self) -> None:
        scope = {"snapshot_id": "snap:repo:abc", "kind": "architecture"}
        h1 = projection_params_hash(scope)
        h2 = projection_params_hash(scope)
        assert h1 == h2

    def test_different_scopes_differ(self) -> None:
        h1 = projection_params_hash({"snapshot_id": "a"})
        h2 = projection_params_hash({"snapshot_id": "b"})
        assert h1 != h2


class TestDeterministicEventId:
    def test_deterministic(self) -> None:
        eid1 = deterministic_event_id("snap:repo:abc", "payload")
        eid2 = deterministic_event_id("snap:repo:abc", "payload")
        assert eid1 == eid2

    def test_different_payloads_differ(self) -> None:
        eid1 = deterministic_event_id("snap:repo:abc", "payload1")
        eid2 = deterministic_event_id("snap:repo:abc", "payload2")
        assert eid1 != eid2

    def test_format_starts_with_outbox(self) -> None:
        eid = deterministic_event_id("snap:repo:abc", "payload")
        assert eid.startswith("outbox:")
```

- [ ] **Step 5: Run the new tests**

```bash
uv run pytest tests/fastcore/ -v --timeout=30
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/fastcore/
git commit -m "test: restructure tests into tests/fastcore/ mirroring package layout"
```

---

## Task 7: Cleanup Old Test Files

- [ ] **Step 1: Delete old test files**

```bash
rm tests/test_core_types.py tests/test_core_scoring.py tests/test_core_fusion.py \
   tests/test_core_boundary.py tests/test_core_filtering.py tests/test_core_combination.py \
   tests/test_core_iteration.py tests/test_core_prompts.py tests/test_core_parsing.py \
   tests/test_core_context.py tests/test_core_summary.py tests/test_core_graph_build.py \
   tests/test_core_snapshot.py tests/test_core_repo_analysis.py tests/test_core_scip_transform.py \
   tests/test_effects_db.py tests/test_effects_llm.py tests/test_effects_fs.py
```

- [ ] **Step 2: Run tests to confirm**

```bash
uv run pytest tests/fastcore/ -v --timeout=30
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove old test files (migrated to tests/fastcore/)"
```

---

## Task 8: Final Verification

- [ ] **Step 1: Run ruff on changed files**

```bash
uv run ruff check fastcode/ tests/fastcore/ --fix
uv run ruff format fastcode/ tests/fastcore/
```

- [ ] **Step 2: Verify pytest config**

```bash
uv run pytest tests/fastcore/ -v --timeout=30
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: format and final verification"
```
