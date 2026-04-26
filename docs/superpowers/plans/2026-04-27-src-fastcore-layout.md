# src/fastcore Layout Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the FP core into an independent `src/fastcore` package with schema/core/utils/effects separation, move tests to mirror src, add pytest-timeout + pytest-subprocess.

**Architecture:** Move-and-rewrite-imports. Code from `fastcode/core/` and `fastcode/effects/` is copied to `src/fastcore/` with import path adjustments. Utility functions extracted by the copy-paste test. Original `fastcode/` delegation files get import path updates. Old `fastcode/core/` and `fastcode/effects/` directories are deleted.

**Tech Stack:** Python 3.11+, uv workspace, pytest-timeout, pytest-subprocess

**Constraint:** Do NOT run the full `tests/` suite. Only run `tests/fastcore/` to verify.

---

## File Structure Map

### New files to create

| File | Source | Responsibility |
|------|--------|---------------|
| `src/fastcore/__init__.py` | new | Package root |
| `src/fastcore/pyproject.toml` | new | Package metadata |
| `src/fastcore/schema/__init__.py` | new | Re-exports all types |
| `src/fastcore/schema/core_types.py` | copy from `fastcode/core/types.py` | Frozen dataclasses (Hit, FusionConfig, etc.) |
| `src/fastcore/schema/ir.py` | copy from `fastcode/semantic_ir.py` | IR dataclasses (IRSnapshot, IRSymbol, etc.) |
| `src/fastcore/core/__init__.py` | copy from `fastcode/core/__init__.py` | Package marker |
| `src/fastcore/core/scoring.py` | copy from `fastcode/core/scoring.py` | Scoring functions |
| `src/fastcore/core/fusion.py` | copy from `fastcode/core/fusion.py` | Fusion functions |
| `src/fastcore/core/filtering.py` | copy from `fastcode/core/filtering.py` | Filtering functions |
| `src/fastcore/core/combination.py` | copy from `fastcode/core/combination.py` | Combination functions |
| `src/fastcore/core/iteration.py` | copy from `fastcode/core/iteration.py` | Iteration functions |
| `src/fastcore/core/prompts.py` | copy from `fastcode/core/prompts.py` | Prompt formatting |
| `src/fastcore/core/context.py` | copy from `fastcode/core/context.py` | Context preparation |
| `src/fastcore/core/summary.py` | copy from `fastcode/core/summary.py` | Summary generation |
| `src/fastcore/core/graph_build.py` | copy from `fastcode/core/graph_build.py` | Graph payload building |
| `src/fastcore/core/snapshot.py` | copy from `fastcode/core/snapshot.py` | Source extraction (utils functions removed) |
| `src/fastcore/core/repo_analysis.py` | copy from `fastcode/core/repo_analysis.py` | Repo analysis (utils functions removed) |
| `src/fastcore/core/scip_transform.py` | copy from `fastcode/core/scip_transform.py` | SCIP transforms |
| `src/fastcore/core/boundary.py` | copy from `fastcode/core/boundary.py` | Explicit translation |
| `src/fastcore/utils/__init__.py` | new | Package marker |
| `src/fastcore/utils/json.py` | extract from `fastcode/core/parsing.py` + `fastcode/utils.py` | JSON utilities + safe_jsonable |
| `src/fastcore/utils/hashing.py` | extract from `fastcode/core/snapshot.py` + `fastcode/core/graph_build.py` | Hashing utilities |
| `src/fastcore/utils/paths.py` | extract from `fastcode/core/snapshot.py` + `fastcode/core/repo_analysis.py` | Path/language utilities |
| `src/fastcore/effects/__init__.py` | copy from `fastcode/effects/__init__.py` | Package marker |
| `src/fastcore/effects/db.py` | copy from `fastcode/effects/db.py` | DB effects |
| `src/fastcore/effects/llm.py` | copy from `fastcode/effects/llm.py` | LLM effects |
| `src/fastcore/effects/fs.py` | copy from `fastcode/effects/fs.py` | FS effects |
| `tests/fastcore/__init__.py` | new | Test package marker |
| `tests/fastcore/test_schema_types.py` | copy from `tests/test_core_types.py` | Core types tests |
| `tests/fastcore/test_schema_ir.py` | new | IR types tests |
| `tests/fastcore/test_core_scoring.py` | copy from `tests/test_core_scoring.py` | Scoring tests |
| `tests/fastcore/test_core_fusion.py` | copy from `tests/test_core_fusion.py` | Fusion tests |
| `tests/fastcore/test_core_boundary.py` | copy from `tests/test_core_boundary.py` | Boundary tests |
| `tests/fastcore/test_core_filtering.py` | copy from `tests/test_core_filtering.py` | Filtering tests |
| `tests/fastcore/test_core_combination.py` | copy from `tests/test_core_combination.py` | Combination tests |
| `tests/fastcore/test_core_iteration.py` | copy from `tests/test_core_iteration.py` | Iteration tests |
| `tests/fastcore/test_core_prompts.py` | copy from `tests/test_core_prompts.py` | Prompts tests |
| `tests/fastcore/test_core_context.py` | copy from `tests/test_core_context.py` | Context tests |
| `tests/fastcore/test_core_summary.py` | copy from `tests/test_core_summary.py` | Summary tests |
| `tests/fastcore/test_core_graph_build.py` | copy from `tests/test_core_graph_build.py` | Graph build tests |
| `tests/fastcore/test_core_snapshot.py` | copy from `tests/test_core_snapshot.py` | Snapshot tests |
| `tests/fastcore/test_core_repo_analysis.py` | copy from `tests/test_core_repo_analysis.py` | Repo analysis tests |
| `tests/fastcore/test_core_scip_transform.py` | copy from `tests/test_core_scip_transform.py` | SCIP transform tests |
| `tests/fastcore/test_utils_json.py` | copy from `tests/test_core_parsing.py` | JSON utils tests |
| `tests/fastcore/test_utils_hashing.py` | new | Hashing utils tests |
| `tests/fastcore/test_utils_paths.py` | extract from `tests/test_core_snapshot.py` + `tests/test_core_repo_analysis.py` | Path utils tests |
| `tests/fastcore/test_effects_db.py` | copy from `tests/test_effects_db.py` | DB effects tests |
| `tests/fastcore/test_effects_llm.py` | copy from `tests/test_effects_llm.py` | LLM effects tests |
| `tests/fastcore/test_effects_fs.py` | copy from `tests/test_effects_fs.py` | FS effects tests |

### Existing files to modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add workspace member, add fastcore dependency, add pytest-timeout/subprocess, add timeout config |
| `fastcode/retriever.py` | Update imports: `fastcode.core.*` → `fastcore.*` |
| `fastcode/iterative_agent.py` | Update imports: `fastcode.core.*` → `fastcore.*` |
| `fastcode/answer_generator.py` | Update imports: `fastcode.core.*` → `fastcore.*` |
| `fastcode/terminus_publisher.py` | Update imports: `fastcode.core.*` → `fastcore.*` |
| `fastcode/main.py` | Update imports: `fastcode.core.*` → `fastcore.*` |
| `fastcode/repo_overview.py` | Update imports: `fastcode.core.*` → `fastcore.*` |
| `fastcode/scip_loader.py` | Update imports: `fastcode.core.*` → `fastcore.*` |

### Files/directories to delete after migration

| Path | Reason |
|------|--------|
| `fastcode/core/` | Moved to `src/fastcore/core/` |
| `fastcode/effects/` | Moved to `src/fastcore/effects/` |
| `tests/test_core_*.py` (15 files) | Moved to `tests/fastcore/` |
| `tests/test_effects_*.py` (3 files) | Moved to `tests/fastcore/` |

---

## Task 1: Package Scaffolding & Workspace Config

**Files:**
- Create: `src/fastcore/pyproject.toml`
- Create: `src/fastcore/__init__.py`
- Create: `src/fastcore/schema/__init__.py`
- Create: `src/fastcore/core/__init__.py`
- Create: `src/fastcore/utils/__init__.py`
- Create: `src/fastcore/effects/__init__.py`
- Create: `tests/fastcore/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/fastcore/schema src/fastcore/core src/fastcore/utils src/fastcore/effects tests/fastcore
```

- [ ] **Step 2: Create `src/fastcore/pyproject.toml`**

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

- [ ] **Step 3: Create package `__init__.py` files**

`src/fastcore/__init__.py`:
```python
"""fastcore — pure functional core for code intelligence."""
```

`src/fastcore/schema/__init__.py`:
```python
"""Schema — all frozen dataclasses and IR types."""
```

`src/fastcore/core/__init__.py`:
```python
"""Core — domain-specific pure functions."""
```

`src/fastcore/utils/__init__.py`:
```python
"""Utils — domain-independent utilities (copy-paste test)."""
```

`src/fastcore/effects/__init__.py`:
```python
"""Effects — thin I/O boundary."""
```

`tests/fastcore/__init__.py`:
```python
```

- [ ] **Step 4: Update root `pyproject.toml`**

In `[tool.uv.workspace]`, add `src/fastcore`:
```toml
[tool.uv.workspace]
members = ["nanobot", "src/fastcore"]
```

In `[tool.uv.sources]`, add fastcore:
```toml
[tool.uv.sources]
nanobot-ai = { workspace = true }
fastcore = { workspace = true }
```

In `dependencies`, add fastcore:
```toml
dependencies = [
    # ... existing entries ...
    "fastcore",
]
```

In `[project.optional-dependencies]` dev, add pytest plugins:
```toml
[project.optional-dependencies]
dev = [
    # ... existing entries ...
    "pytest-timeout",
    "pytest-subprocess",
]
```

In `[tool.pytest.ini_options]`, add timeout config:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "strict"
addopts = "-n auto"
timeout = 30
timeout_method = "thread"
```

- [ ] **Step 5: Run `uv sync` and verify workspace**

```bash
uv sync
uv run python -c "import fastcore; print(fastcore.__file__)"
```

Expected: prints path to `src/fastcore/__init__.py`

- [ ] **Step 6: Commit**

```bash
git add src/fastcore/ tests/fastcore/ pyproject.toml uv.lock
git commit -m "chore: scaffold src/fastcore package and workspace config"
```

---

## Task 2: Schema — Core Types

**Files:**
- Create: `src/fastcore/schema/core_types.py`
- Modify: `src/fastcore/schema/__init__.py`

- [ ] **Step 1: Copy `fastcode/core/types.py` to `src/fastcore/schema/core_types.py`**

```bash
cp fastcode/core/types.py src/fastcore/schema/core_types.py
```

No changes needed — the file has no relative imports that need updating (it only imports from `dataclasses` and `typing`).

- [ ] **Step 2: Update `src/fastcore/schema/__init__.py` with re-exports**

```python
"""Schema — all frozen dataclasses and IR types."""

from fastcore.schema.core_types import (
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

- [ ] **Step 3: Verify import works**

```bash
uv run python -c "from fastcore.schema.core_types import Hit; print(Hit)"
```

Expected: `<class 'fastcore.schema.core_types.Hit'>`

- [ ] **Step 4: Commit**

```bash
git add src/fastcore/schema/
git commit -m "feat: add schema/core_types.py — frozen dataclasses"
```

---

## Task 3: Schema — IR Types

**Files:**
- Create: `src/fastcore/schema/ir.py`
- Modify: `src/fastcore/schema/__init__.py`

- [ ] **Step 1: Create `src/fastcore/schema/ir.py`**

Read `fastcode/semantic_ir.py` and copy the following items. Adjust the import: replace `from .utils import safe_jsonable` with `from fastcore.utils.json import safe_jsonable` (will be created in Task 4).

Items to copy verbatim:
- Constants: `_MAX_SAFE_JSONABLE_DEPTH` is not here — it's in utils.py. The `safe_jsonable` function will be in `fastcore/utils/json.py`.
- Helper functions: `_sorted_set`, `_normalize_set`, `_resolution_to_confidence`, `_confidence_to_resolution`, `_unit_kind_to_symbol_kind`, `_symbol_kind_to_unit_kind`
- Dataclasses: `IRDocument`, `IRSymbol`, `IROccurrence`, `IREdge`, `IRAttachment`
- The `IRSnapshot` class with its properties and methods

Top of file should be:
```python
"""Canonical IR dataclasses — extracted from fastcode.semantic_ir."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from fastcore.utils.json import safe_jsonable
```

Do NOT copy: `IRCodeUnit`, `IRUnitSupport`, `IRRelation`, `IRUnitEmbedding` — these are advanced types not used by the core modules. They stay in `fastcode.semantic_ir` for now.

- [ ] **Step 2: Update `src/fastcore/schema/__init__.py` — add IR re-exports**

Append to the existing file:
```python
from fastcore.schema.ir import (
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

- [ ] **Step 3: Verify import works**

```bash
uv run python -c "from fastcore.schema.ir import IRSnapshot, IRSymbol; print(IRSnapshot, IRSymbol)"
```

Expected: both classes print without error

- [ ] **Step 4: Commit**

```bash
git add src/fastcore/schema/ir.py src/fastcore/schema/__init__.py
git commit -m "feat: add schema/ir.py — canonical IR dataclasses"
```

---

## Task 4: Utils Extraction

**Files:**
- Create: `src/fastcore/utils/json.py`
- Create: `src/fastcore/utils/hashing.py`
- Create: `src/fastcore/utils/paths.py`
- Modify: `src/fastcore/utils/__init__.py`

- [ ] **Step 1: Create `src/fastcore/utils/json.py`**

Copy `safe_jsonable` from `fastcode/utils.py` (lines 288-323) and all four parsing functions from `fastcode/core/parsing.py`:

```python
"""Domain-independent JSON utilities."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

_MAX_SAFE_JSONABLE_DEPTH = 12


def safe_jsonable(obj: Any, *, _depth: int = 0) -> Any:
    """Recursively convert objects to JSON-serializable structures."""
    if _depth > _MAX_SAFE_JSONABLE_DEPTH:
        return repr(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        safe_dict = {}
        for k, v in obj.items():
            try:
                safe_dict[str(k)] = safe_jsonable(v, _depth=_depth + 1)
            except Exception:
                safe_dict[str(k)] = repr(v)
        return safe_dict
    if isinstance(obj, (list, tuple, set)):
        return [safe_jsonable(v, _depth=_depth + 1) for v in obj]
    if hasattr(obj, "to_dict"):
        try:
            return safe_jsonable(obj.to_dict(), _depth=_depth + 1)
        except Exception:
            return {"repr": repr(obj)}
    if hasattr(obj, "__dict__"):
        try:
            return safe_jsonable(vars(obj), _depth=_depth + 1)
        except Exception:
            return {"repr": repr(obj)}
    return repr(obj)


# Copy all four function bodies verbatim from fastcode/core/parsing.py:
#   extract_json_from_response, sanitize_json_string,
#   remove_json_comments, robust_json_parse
# These functions have no external dependencies beyond stdlib (ast, json, re).
```

Copy the function bodies verbatim from `fastcode/core/parsing.py`. The functions have no external dependencies beyond `ast`, `json`, `re`, and `typing` — all stdlib.

- [ ] **Step 2: Create `src/fastcore/utils/hashing.py`**

Copy `projection_params_hash` from `fastcode/core/snapshot.py` and `deterministic_event_id` from `fastcode/core/graph_build.py`:

```python
"""Domain-independent hashing utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def projection_params_hash(scope_dict: dict[str, Any], version: str = "v1") -> str:
    """Hash projection parameters for cache key."""
    payload = json.dumps(
        {"scope": scope_dict, "projection_algo_version": version},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def deterministic_event_id(snapshot_id: str, payload: str) -> str:
    """Generate a deterministic event ID from snapshot_id + payload hash."""
    h = hashlib.sha256(f"{snapshot_id}:{payload}".encode()).hexdigest()[:32]
    return f"outbox:{snapshot_id}:{h}"
```

- [ ] **Step 3: Create `src/fastcore/utils/paths.py`**

Copy `get_language_from_extension` from `fastcode/core/repo_analysis.py` and `projection_scope_key` from `fastcode/core/snapshot.py`:

```python
"""Domain-independent path and extension utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


# Copy get_language_from_extension function body verbatim from
# fastcode/core/repo_analysis.py (pure extension lookup table, no dependencies)


def projection_scope_key(
    scope_kind: str,
    snapshot_id: str,
    query: str | None,
    target_id: str | None,
    filters: dict[str, Any] | None,
) -> str:
    """Compute a deterministic hash key for a projection scope."""
    base = {
        "scope_kind": scope_kind,
        "snapshot_id": snapshot_id,
        "query": query or "",
        "target_id": target_id or "",
        "filters": filters or {},
    }
    payload = json.dumps(base, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]
```

- [ ] **Step 4: Update `src/fastcore/utils/__init__.py`**

```python
"""Utils — domain-independent utilities (copy-paste test)."""

from fastcore.utils.json import (
    extract_json_from_response,
    robust_json_parse,
    safe_jsonable,
    sanitize_json_string,
)
from fastcore.utils.hashing import deterministic_event_id, projection_params_hash
from fastcore.utils.paths import get_language_from_extension, projection_scope_key

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

- [ ] **Step 5: Verify imports work**

```bash
uv run python -c "from fastcore.utils.json import safe_jsonable; from fastcore.utils.hashing import deterministic_event_id; from fastcore.utils.paths import get_language_from_extension; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/fastcore/utils/
git commit -m "feat: add utils/ — domain-independent json, hashing, paths utilities"
```

---

## Task 5: Core Modules Move

**Files:**
- Create: 14 files in `src/fastcode/core/` (copies with import adjustments)

- [ ] **Step 1: Copy all core modules**

```bash
for f in scoring.py fusion.py filtering.py combination.py iteration.py \
         prompts.py context.py summary.py graph_build.py snapshot.py \
         repo_analysis.py scip_transform.py boundary.py; do
    cp "fastcode/core/$f" "src/fastcore/core/$f"
done
```

- [ ] **Step 2: Fix imports in each file**

Every file that has relative imports like `from .types import ...` or `from .scoring import ...` needs adjustment. Here is the complete mapping:

**`src/fastcore/core/scoring.py`** — no changes (only stdlib imports)

**`src/fastcore/core/fusion.py`** — change:
```python
# OLD:
from .scoring import clone_result_row, normalized_query_entropy, normalized_totals, sigmoid, tokenize_signal, trace_confidence_weight, weighted_keyword_affinity
from .types import FusionConfig
# NEW:
from fastcore.core.scoring import clone_result_row, normalized_query_entropy, normalized_totals, sigmoid, tokenize_signal, trace_confidence_weight, weighted_keyword_affinity
from fastcore.schema.core_types import FusionConfig
```

**`src/fastcore/core/filtering.py`** — no changes (only stdlib imports)

**`src/fastcore/core/combination.py`** — no changes (only stdlib imports)

**`src/fastcore/core/iteration.py`** — change:
```python
# OLD:
from .types import IterationConfig
# NEW:
from fastcore.schema.core_types import IterationConfig
```

**`src/fastcore/core/prompts.py`** — no changes (only stdlib imports)

**`src/fastcore/core/context.py`** — no changes (only stdlib imports)

**`src/fastcore/core/summary.py`** — no changes (only stdlib imports)

**`src/fastcore/core/graph_build.py`** — change:
```python
# OLD:
from ..semantic_ir import _resolution_to_confidence
# NEW:
from fastcore.schema.ir import _resolution_to_confidence
```

**`src/fastcore/core/snapshot.py`** — this file needs splitting:
- Remove `projection_scope_key` and `projection_params_hash` (moved to `fastcore/utils/`)
- Keep `extract_sources_from_elements` (domain-specific)
- Update any remaining imports

New `src/fastcore/core/snapshot.py`:
```python
"""Pure snapshot logic — domain-specific parts."""

from __future__ import annotations

from typing import Any


def extract_sources_from_elements(
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract source information from retrieved elements."""
    # Copy verbatim from fastcode/core/snapshot.py
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

**`src/fastcore/core/repo_analysis.py`** — remove `get_language_from_extension` (moved to `fastcore/utils/paths.py`). Keep: `is_key_file`, `infer_project_type`, `generate_structure_based_overview`, `format_file_structure`.

Change:
```python
# Add at top:
from fastcore.utils.paths import get_language_from_extension
# Remove the get_language_from_extension function definition from this file
```

**`src/fastcore/core/scip_transform.py`** — no changes (no imports)

**`src/fastcore/core/boundary.py`** — change:
```python
# OLD:
from fastcode.core.types import Hit
# NEW:
from fastcore.schema.core_types import Hit
```

- [ ] **Step 3: Verify all core modules import cleanly**

```bash
uv run python -c "
from fastcore.core.scoring import sigmoid
from fastcore.core.fusion import adaptive_fuse_channels
from fastcore.core.filtering import apply_filters
from fastcore.core.combination import combine_results
from fastcore.core.iteration import should_continue_iteration
from fastcore.core.prompts import format_elements_with_metadata
from fastcore.core.context import prepare_context
from fastcore.core.summary import generate_fallback_summary
from fastcore.core.graph_build import build_code_graph_payload
from fastcore.core.snapshot import extract_sources_from_elements
from fastcore.core.repo_analysis import infer_project_type
from fastcore.core.scip_transform import symbol_role_to_str
from fastcore.core.boundary import hit_to_response
print('All core modules OK')
"
```

Expected: `All core modules OK`

- [ ] **Step 4: Commit**

```bash
git add src/fastcore/core/
git commit -m "feat: move core modules to src/fastcore/core/ with adjusted imports"
```

---

## Task 6: Effects Modules Move

**Files:**
- Create: `src/fastcore/effects/db.py`, `src/fastcore/effects/llm.py`, `src/fastcore/effects/fs.py`

- [ ] **Step 1: Copy effects modules**

```bash
cp fastcode/effects/db.py src/fastcore/effects/db.py
cp fastcode/effects/llm.py src/fastcore/effects/llm.py
cp fastcode/effects/fs.py src/fastcore/effects/fs.py
```

- [ ] **Step 2: Fix imports**

**`src/fastcore/effects/db.py`** — change:
```python
# OLD:
from fastcode.core.types import SnapshotRecord
# NEW:
from fastcore.schema.core_types import SnapshotRecord
```

**`src/fastcore/effects/llm.py`** — no changes

**`src/fastcore/effects/fs.py`** — no changes

- [ ] **Step 3: Verify imports**

```bash
uv run python -c "from fastcore.effects.db import load_snapshot_record; from fastcore.effects.llm import chat_completion; from fastcore.effects.fs import read_file; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/fastcore/effects/
git commit -m "feat: move effects modules to src/fastcore/effects/"
```

---

## Task 7: Tests Restructure

**Files:**
- Create: 21 test files in `tests/fastcore/`

- [ ] **Step 1: Copy test files**

```bash
# Schema tests
cp tests/test_core_types.py tests/fastcore/test_schema_types.py

# Core tests
for f in scoring fusion filtering combination iteration prompts \
         context summary graph_build snapshot repo_analysis \
         scip_transform boundary; do
    cp "tests/test_core_${f}.py" "tests/fastcore/test_core_${f}.py"
done

# Effects tests
for f in db llm fs; do
    cp "tests/test_effects_${f}.py" "tests/fastcore/test_effects_${f}.py"
done

# Utils tests (parsing → json)
cp tests/test_core_parsing.py tests/fastcore/test_utils_json.py
```

- [ ] **Step 2: Update imports in all test files**

Every `from fastcode.core.X import ...` must become `from fastcore.core.X import ...`.
Every `from fastcode.core.types import ...` must become `from fastcore.schema.core_types import ...`.
Every `from fastcode.effects.X import ...` must become `from fastcore.effects.X import ...`.

For `test_core_parsing.py` → `test_utils_json.py`: change `from fastcode.core.parsing import ...` to `from fastcore.utils.json import ...`.

Complete mapping for each test file:

| Test file | Old import prefix | New import prefix |
|-----------|------------------|-------------------|
| `test_schema_types.py` | `from fastcode.core.types` | `from fastcore.schema.core_types` |
| `test_core_scoring.py` | `from fastcode.core.scoring` | `from fastcore.core.scoring` |
| `test_core_fusion.py` | `from fastcode.core.fusion` / `from fastcode.core.types` | `from fastcore.core.fusion` / `from fastcore.schema.core_types` |
| `test_core_boundary.py` | `from fastcode.core.boundary` / `from fastcode.core.types` | `from fastcore.core.boundary` / `from fastcore.schema.core_types` |
| `test_core_filtering.py` | `from fastcode.core.filtering` | `from fastcore.core.filtering` |
| `test_core_combination.py` | `from fastcode.core.combination` | `from fastcore.core.combination` |
| `test_core_iteration.py` | `from fastcode.core.iteration` / `from fastcode.core.types` | `from fastcode.core.iteration` / `from fastcore.schema.core_types` |
| `test_core_prompts.py` | `from fastcode.core.prompts` | `from fastcore.core.prompts` |
| `test_core_context.py` | `from fastcode.core.context` | `from fastcore.core.context` |
| `test_core_summary.py` | `from fastcode.core.summary` | `from fastcore.core.summary` |
| `test_core_graph_build.py` | `from fastcode.core.graph_build` | `from fastcore.core.graph_build` |
| `test_core_snapshot.py` | `from fastcode.core.snapshot` | `from fastcore.core.snapshot` + `from fastcore.utils.hashing` + `from fastcore.utils.paths` |
| `test_core_repo_analysis.py` | `from fastcode.core.repo_analysis` | `from fastcore.core.repo_analysis` |
| `test_core_scip_transform.py` | `from fastcode.core.scip_transform` | `from fastcore.core.scip_transform` |
| `test_utils_json.py` | `from fastcode.core.parsing` | `from fastcore.utils.json` |
| `test_effects_db.py` | `from fastcode.core.types` / `from fastcode.effects.db` | `from fastcore.schema.core_types` / `from fastcore.effects.db` |
| `test_effects_llm.py` | `from fastcode.effects.llm` | `from fastcore.effects.llm` |
| `test_effects_fs.py` | `from fastcode.effects.fs` | `from fastcore.effects.fs` |

**Special handling for `test_core_snapshot.py`:** The original imports `projection_params_hash` and `projection_scope_key` from `fastcode.core.snapshot`. These functions moved to `fastcore.utils.hashing` and `fastcore.utils.paths` respectively. Update:

```python
# OLD:
from fastcode.core.snapshot import (
    extract_sources_from_elements, projection_params_hash, projection_scope_key
)
# NEW:
from fastcore.core.snapshot import extract_sources_from_elements
from fastcore.utils.hashing import projection_params_hash
from fastcore.utils.paths import projection_scope_key
```

**Special handling for `test_core_boundary.py`:** The `TestCoreImportGuard` test scans `fastcode/core/` for I/O imports. Update it to scan `src/fastcore/core/` instead. Change:

```python
# OLD:
CORE_DIR = pathlib.Path(__file__).resolve().parent.parent / "fastcode" / "core"
# NEW:
CORE_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "fastcore" / "core"
```

Also update the `TestDbEffectsReturnDataclasses` test to point to `src/fastcore/effects/db.py`:

```python
# OLD:
db_effects = (
    pathlib.Path(__file__).resolve().parent.parent
    / "fastcode"
    / "effects"
    / "db.py"
)
# NEW:
db_effects = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src"
    / "fastcore"
    / "effects"
    / "db.py"
)
```

Also update `TestBoundaryExplicitTranslation` to point to `fastcore.core.boundary`:

```python
# OLD:
source = pathlib.Path(
    __import__("fastcode.core.boundary", fromlist=[""]).__file__
).read_text(encoding="utf-8")
# NEW:
source = pathlib.Path(
    __import__("fastcore.core.boundary", fromlist=[""]).__file__
).read_text(encoding="utf-8")
```

- [ ] **Step 3: Create `tests/fastcore/test_utils_hashing.py`**

Extract hashing-related tests from `test_core_snapshot.py` and `test_core_graph_build.py`:

```python
"""Tests for fastcore.utils.hashing."""

from __future__ import annotations

from fastcore.utils.hashing import deterministic_event_id, projection_params_hash


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

- [ ] **Step 4: Run the new tests**

```bash
uv run pytest tests/fastcore/ -v --timeout=30
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/fastcore/
git commit -m "test: restructure tests into tests/fastcore/ mirroring src layout"
```

---

## Task 8: fastcode/ Import Updates

**Files:**
- Modify: `fastcode/retriever.py`
- Modify: `fastcode/iterative_agent.py`
- Modify: `fastcode/answer_generator.py`
- Modify: `fastcode/terminus_publisher.py`
- Modify: `fastcode/main.py`
- Modify: `fastcode/repo_overview.py`
- Modify: `fastcode/scip_loader.py`

- [ ] **Step 1: Update imports in all 7 files**

Apply these exact replacements:

**`fastcode/retriever.py`** (lines 16-20):
```python
# OLD:
from .core import combination as _combination
from .core import filtering as _filtering
from .core import fusion as _fusion
from .core import scoring as _scoring
from .core.types import FusionConfig
# NEW:
from fastcore.core import combination as _combination
from fastcore.core import filtering as _filtering
from fastcore.core import fusion as _fusion
from fastcore.core import scoring as _scoring
from fastcore.schema.core_types import FusionConfig
```

**`fastcode/iterative_agent.py`** (lines 16-19):
```python
# OLD:
from .core import iteration as _iteration
from .core import parsing as _parsing
from .core import prompts as _prompts
from .core.types import IterationConfig
# NEW:
from fastcore.core import iteration as _iteration
from fastcore.utils import json as _json_utils
from fastcore.core import prompts as _prompts
from fastcore.schema.core_types import IterationConfig
```

Note: `parsing` module moved to `fastcore.utils.json`. Find all usages of `_parsing.` in this file and replace with `_json_utils.`.

**`fastcode/answer_generator.py`** (lines 15-16):
```python
# OLD:
from .core import context as _context
from .core import summary as _summary
# NEW:
from fastcore.core import context as _context
from fastcore.core import summary as _summary
```

**`fastcode/terminus_publisher.py`** (line 13):
```python
# OLD:
from .core import graph_build as _graph_build
# NEW:
from fastcore.core import graph_build as _graph_build
```

**`fastcode/main.py`** (line 25):
```python
# OLD:
from .core import snapshot as _snapshot
# NEW:
from fastcore.core import snapshot as _snapshot
```

**`fastcode/repo_overview.py`** (line 14):
```python
# OLD:
from .core import repo_analysis as _repo_analysis
# NEW:
from fastcore.core import repo_analysis as _repo_analysis
```

**`fastcode/scip_loader.py`** (line 13):
```python
# OLD:
from fastcode.core import scip_transform as _scip_transform
# NEW:
from fastcore.core import scip_transform as _scip_transform
```

- [ ] **Step 2: Verify fastcode still imports correctly**

```bash
uv run python -c "from fastcode.retriever import HybridRetriever; print('retriever OK')"
uv run python -c "from fastcode.iterative_agent import IterativeAgent; print('iterative_agent OK')"
uv run python -c "from fastcode.answer_generator import AnswerGenerator; print('answer_generator OK')"
```

- [ ] **Step 3: Commit**

```bash
git add fastcode/retriever.py fastcode/iterative_agent.py fastcode/answer_generator.py \
       fastcode/terminus_publisher.py fastcode/main.py fastcode/repo_overview.py \
       fastcode/scip_loader.py
git commit -m "refactor: update fastcode/ imports to use fastcore package"
```

---

## Task 9: Cleanup Old Directories

**Files:**
- Delete: `fastcode/core/` directory (14 files)
- Delete: `fastcode/effects/` directory (3 files)
- Delete: `tests/test_core_*.py` (15 files)
- Delete: `tests/test_effects_*.py` (3 files)
- Delete: `tests/test_core_parsing.py` (moved to test_utils_json.py)

- [ ] **Step 1: Delete old core and effects directories**

```bash
rm -rf fastcode/core/ fastcode/effects/
```

- [ ] **Step 2: Delete old test files**

```bash
rm tests/test_core_types.py tests/test_core_scoring.py tests/test_core_fusion.py \
   tests/test_core_boundary.py tests/test_core_filtering.py tests/test_core_combination.py \
   tests/test_core_iteration.py tests/test_core_prompts.py tests/test_core_parsing.py \
   tests/test_core_context.py tests/test_core_summary.py tests/test_core_graph_build.py \
   tests/test_core_snapshot.py tests/test_core_repo_analysis.py tests/test_core_scip_transform.py \
   tests/test_effects_db.py tests/test_effects_llm.py tests/test_effects_fs.py
```

- [ ] **Step 3: Run fastcore tests to confirm nothing broke**

```bash
uv run pytest tests/fastcore/ -v --timeout=30
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove old fastcode/core/ and fastcode/effects/ (migrated to src/fastcore/)"
```

---

## Task 10: Pytest Config & Final Verification

**Files:**
- Modify: `pyproject.toml` (pytest-timeout config already added in Task 1)
- Create: `tests/fastcore/conftest.py` (optional per-test timeout overrides)

- [ ] **Step 1: Verify pytest config is correct**

Check `pyproject.toml` has:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "strict"
addopts = "-n auto"
timeout = 30
timeout_method = "thread"
```

And dev dependencies include `pytest-timeout` and `pytest-subprocess`.

- [ ] **Step 2: Create `tests/fastcore/conftest.py`** (optional timeout overrides)

```python
"""Shared fixtures for fastcore tests."""
```

This file can grow over time with shared fixtures for effects testing using `pytest-subprocess`.

- [ ] **Step 3: Run all fastcore tests with timeout enforcement**

```bash
uv run pytest tests/fastcore/ -v --timeout=30
```

Expected: All tests pass, no timeouts triggered.

- [ ] **Step 4: Run ruff check on new code**

```bash
uv run ruff check src/fastcore/ tests/fastcore/ --fix
uv run ruff format src/fastcore/ tests/fastcore/
```

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: add pytest-timeout and pytest-subprocess, format new code"
```
