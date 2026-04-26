# Cargo-Style Workspace Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the FP core into a `fastcode-core` workspace member (Cargo-style monorepo), archive the original `fastcode/` to `_fastcode/`, restructure tests to mirror src inside each workspace member, add pytest-timeout + pytest-subprocess.

**Architecture:** Cargo-style uv workspace monorepo. Root `pyproject.toml` is the workspace root AND the `fastcode` app. `libs/core/` is a workspace member (`fastcode-core`) with schema/core/utils/effects separation. Tests live inside each workspace member, mirroring its `src/` layout. No `pkgutil.extend_path` — package names are different (`fastcode` vs `fastcode_core`), so no ambiguity.

**Tech Stack:** Python 3.11+, uv workspace, setuptools, pytest-timeout, pytest-subprocess

**Constraint:** Do NOT run the full `tests/` suite during implementation. Only run `libs/core/tests/` to verify the extracted core.

---

## Cargo ↔ uv Workspace Analogy

| Feature | Rust (Cargo) | Modern Python (uv) |
|---------|-------------|-------------------|
| Root Config | `Cargo.toml` (`[workspace]`) | `pyproject.toml` (`[tool.uv.workspace]`) |
| Lockfile | Single `Cargo.lock` | Single `uv.lock` |
| Package Config | `Cargo.toml` in every crate | `pyproject.toml` in every package |
| Local Dep | `{ path = "../libs/core" }` | `dependencies = ["fastcode-core"]` (uv resolves locally) |

## Final Directory Structure

```
pyproject.toml                    # Workspace root + fastcode app
uv.lock                           # Single lockfile
_fastcode/                        # Archived original (prefixed _ to exclude from Python)
  core/                           # (original, kept for reference)
  effects/                        # (original, kept for reference)
  ...all original app files...
fastcode/                         # NEW root app package (copied from _fastcode/ minus core/ + effects/)
  __init__.py
  main.py
  retriever.py                    # imports from fastcode_core
  iterative_agent.py              # imports from fastcode_core
  ...all other app modules...
nanobot/                          # Existing workspace member
libs/
  core/                           # Workspace member: fastcode-core
    pyproject.toml                # name = "fastcode-core"
    src/
      fastcode_core/
        __init__.py
        schema/
          __init__.py
          core_types.py           # Hit, FusionConfig, IterationState, etc.
          ir.py                   # IRSnapshot, IRSymbol, IREdge, etc.
        core/
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
          snapshot.py
          repo_analysis.py
          scip_transform.py
          boundary.py
        utils/
          __init__.py
          json.py                 # safe_jsonable + LLM JSON parsing
          hashing.py              # projection_params_hash, deterministic_event_id
          paths.py                # get_language_from_extension, projection_scope_key
        effects/
          __init__.py
          db.py
          llm.py
          fs.py
    tests/                        # Tests mirror src/ inside the workspace member
      __init__.py
      schema/
        __init__.py
        test_core_types.py
        test_ir.py
      core/
        __init__.py
        test_scoring.py
        test_fusion.py
        test_filtering.py
        test_combination.py
        test_iteration.py
        test_prompts.py
        test_context.py
        test_summary.py
        test_graph_build.py
        test_snapshot.py
        test_repo_analysis.py
        test_scip_transform.py
        test_boundary.py
      utils/
        __init__.py
        test_json.py
        test_hashing.py
        test_paths.py
      effects/
        __init__.py
        test_db.py
        test_llm.py
        test_fs.py
tests/                            # Root-level tests (original app tests, unchanged)
  ...
```

---

## File Structure Map

### New files to create

| File | Source | Responsibility |
|------|--------|---------------|
| `libs/core/pyproject.toml` | new | Workspace member package config |
| `libs/core/src/fastcode_core/__init__.py` | new | Package root |
| `libs/core/src/fastcode_core/schema/__init__.py` | new | Re-exports all types |
| `libs/core/src/fastcode_core/schema/core_types.py` | copy from `fastcode/core/types.py` | Frozen dataclasses |
| `libs/core/src/fastcode_core/schema/ir.py` | copy from `fastcode/semantic_ir.py` | IR dataclasses |
| `libs/core/src/fastcode_core/core/__init__.py` | new | Package marker |
| `libs/core/src/fastcode_core/core/scoring.py` | copy from `fastcode/core/scoring.py` | Scoring functions |
| `libs/core/src/fastcode_core/core/fusion.py` | copy from `fastcode/core/fusion.py` | Fusion functions |
| `libs/core/src/fastcode_core/core/filtering.py` | copy from `fastcode/core/filtering.py` | Filtering functions |
| `libs/core/src/fastcode_core/core/combination.py` | copy from `fastcode/core/combination.py` | Combination functions |
| `libs/core/src/fastcode_core/core/iteration.py` | copy from `fastcode/core/iteration.py` | Iteration functions |
| `libs/core/src/fastcode_core/core/prompts.py` | copy from `fastcode/core/prompts.py` | Prompt formatting |
| `libs/core/src/fastcode_core/core/context.py` | copy from `fastcode/core/context.py` | Context preparation |
| `libs/core/src/fastcode_core/core/summary.py` | copy from `fastcode/core/summary.py` | Summary generation |
| `libs/core/src/fastcode_core/core/graph_build.py` | copy from `fastcode/core/graph_build.py` | Graph payload building |
| `libs/core/src/fastcode_core/core/snapshot.py` | copy from `fastcode/core/snapshot.py` | Snapshot logic (utils fns removed) |
| `libs/core/src/fastcode_core/core/repo_analysis.py` | copy from `fastcode/core/repo_analysis.py` | Repo analysis (utils fns removed) |
| `libs/core/src/fastcode_core/core/scip_transform.py` | copy from `fastcode/core/scip_transform.py` | SCIP transforms |
| `libs/core/src/fastcode_core/core/boundary.py` | copy from `fastcode/core/boundary.py` | Explicit translation |
| `libs/core/src/fastcode_core/utils/__init__.py` | new | Re-exports all utils |
| `libs/core/src/fastcode_core/utils/json.py` | extract from `fastcode/core/parsing.py` + `fastcode/utils.py` | JSON utilities |
| `libs/core/src/fastcode_core/utils/hashing.py` | extract from `fastcode/core/snapshot.py` + `fastcode/core/graph_build.py` | Hashing utilities |
| `libs/core/src/fastcode_core/utils/paths.py` | extract from `fastcode/core/snapshot.py` + `fastcode/core/repo_analysis.py` | Path utilities |
| `libs/core/src/fastcode_core/effects/__init__.py` | new | Package marker |
| `libs/core/src/fastcode_core/effects/db.py` | copy from `fastcode/effects/db.py` | DB effects |
| `libs/core/src/fastcode_core/effects/llm.py` | copy from `fastcode/effects/llm.py` | LLM effects |
| `libs/core/src/fastcode_core/effects/fs.py` | copy from `fastcode/effects/fs.py` | FS effects |
| `libs/core/tests/**` (21 test files) | copy from `tests/test_core_*.py` + `tests/test_effects_*.py` | Tests mirroring src |

### Existing files to modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add workspace member, add fastcode-core dependency, add pytest plugins, add timeout config |
| `fastcode/retriever.py` | Update imports: `from .core.*` → `from fastcode_core.*` |
| `fastcode/iterative_agent.py` | Update imports: `from .core.*` → `from fastcode_core.*` |
| `fastcode/answer_generator.py` | Update imports: `from .core.*` → `from fastcode_core.*` |
| `fastcode/terminus_publisher.py` | Update imports: `from .core.*` → `from fastcode_core.*` |
| `fastcode/main.py` | Update imports: `from .core.*` → `from fastcode_core.*` |
| `fastcode/repo_overview.py` | Update imports: `from .core.*` → `from fastcode_core.*` |
| `fastcode/scip_loader.py` | Update imports: `from fastcode.core.*` → `from fastcode_core.*` |

### Import Mapping

| Old import | New import |
|------------|------------|
| `from fastcode.core.types import Hit` | `from fastcode_core.schema.core_types import Hit` |
| `from fastcode.core.types import FusionConfig` | `from fastcode_core.schema.core_types import FusionConfig` |
| `from fastcode.core.types import IterationConfig` | `from fastcode_core.schema.core_types import IterationConfig` |
| `from fastcode.core.types import SnapshotRecord` | `from fastcode_core.schema.core_types import SnapshotRecord` |
| `from .core.scoring import ...` | `from fastcode_core.core.scoring import ...` |
| `from .core.fusion import ...` | `from fastcode_core.core.fusion import ...` |
| `from .core.filtering import ...` | `from fastcode_core.core.filtering import ...` |
| `from .core.combination import ...` | `from fastcode_core.core.combination import ...` |
| `from .core.iteration import ...` | `from fastcode_core.core.iteration import ...` |
| `from .core.prompts import ...` | `from fastcode_core.core.prompts import ...` |
| `from .core.context import ...` | `from fastcode_core.core.context import ...` |
| `from .core.summary import ...` | `from fastcode_core.core.summary import ...` |
| `from .core.graph_build import ...` | `from fastcode_core.core.graph_build import ...` |
| `from .core.snapshot import ...` | `from fastcode_core.core.snapshot import ...` |
| `from .core.repo_analysis import ...` | `from fastcode_core.core.repo_analysis import ...` |
| `from .core.scip_transform import ...` | `from fastcode_core.core.scip_transform import ...` |
| `from .core.boundary import ...` | `from fastcode_core.core.boundary import ...` |
| `from .core.parsing import ...` | `from fastcode_core.utils.json import ...` |
| `from ..semantic_ir import _resolution_to_confidence` | `from fastcode_core.schema.ir import _resolution_to_confidence` |
| `from fastcode.core import scip_transform` | `from fastcode_core.core import scip_transform` |
| `from .effects.db import ...` | `from fastcode_core.effects.db import ...` |

---

## Task 1: Archive Original & Scaffold Workspace

**Files:**
- Create: `libs/core/pyproject.toml`
- Create: `libs/core/src/fastcode_core/__init__.py`
- Create: `libs/core/src/fastcode_core/schema/__init__.py`
- Create: `libs/core/src/fastcode_core/core/__init__.py`
- Create: `libs/core/src/fastcode_core/utils/__init__.py`
- Create: `libs/core/src/fastcode_core/effects/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Archive original `fastcode/` as `_fastcode/`**

```bash
git mv fastcode/ _fastcode/
```

This renames the directory with the `_` prefix. Python and setuptools ignore directories starting with `_`, eliminating ambiguity.

- [ ] **Step 2: Create new root `fastcode/` with app-level files only**

Copy all files from `_fastcode/` to `fastcode/` EXCEPT `core/` and `effects/` subdirectories:

```bash
mkdir -p fastcode
# Copy all root-level .py files (excluding __init__.py which we'll write fresh)
for f in _fastcode/*.py; do
  cp "$f" "fastcode/$(basename "$f")"
done
```

Then create `fastcode/__init__.py` (copy verbatim from `_fastcode/__init__.py`):

```python
"""
FastCode 2.0 - Repository-Level Code Understanding System
With Multi-Repository Support
"""

import os
import platform

if platform.system() == 'Darwin':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

from .main import FastCode
from .loader import RepositoryLoader
from .parser import CodeParser
from .indexer import CodeIndexer
from .retriever import HybridRetriever
from .answer_generator import AnswerGenerator
from .repo_overview import RepositoryOverviewGenerator
from .repo_selector import RepositorySelector
from .iterative_agent import IterativeAgent
from .agent_tools import AgentTools
from .semantic_ir import (
    IRCodeUnit,
    IRDocument,
    IREdge,
    IRRelation,
    IRSnapshot,
    IRSymbol,
    IRUnitEmbedding,
    IRUnitSupport,
    IROccurrence,
)

__version__ = "2.0.0"
FastCode = FastCode

__all__ = [
    "FastCode",
    "FastCode",
    "RepositoryLoader",
    "CodeParser",
    "CodeIndexer",
    "HybridRetriever",
    "AnswerGenerator",
    "RepositoryOverviewGenerator",
    "RepositorySelector",
    "IterativeAgent",
    "AgentTools",
    "IRSnapshot",
    "IRCodeUnit",
    "IRUnitSupport",
    "IRRelation",
    "IRUnitEmbedding",
    "IRDocument",
    "IRSymbol",
    "IROccurrence",
    "IREdge",
]
```

Note: The imports in `retriever.py`, `iterative_agent.py`, etc. still reference `from .core import ...` — these will be updated in Task 8.

- [ ] **Step 3: Create workspace member directory structure**

```bash
mkdir -p libs/core/src/fastcode_core/{schema,core,utils,effects}
mkdir -p libs/core/tests/{schema,core,utils,effects}
```

- [ ] **Step 4: Create `libs/core/pyproject.toml`**

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

- [ ] **Step 5: Create package `__init__.py` files**

`libs/core/src/fastcode_core/__init__.py`:
```python
"""fastcode_core — pure functional core for code intelligence."""
```

`libs/core/src/fastcode_core/schema/__init__.py`:
```python
"""Schema — all frozen dataclasses and IR types."""
```

`libs/core/src/fastcode_core/core/__init__.py`:
```python
"""Core — domain-specific pure functions."""
```

`libs/core/src/fastcode_core/utils/__init__.py`:
```python
"""Utils — domain-independent utilities (copy-paste test)."""
```

`libs/core/src/fastcode_core/effects/__init__.py`:
```python
"""Effects — thin I/O boundary."""
```

`libs/core/tests/__init__.py`:
```python
```

`libs/core/tests/schema/__init__.py`:
```python
```

`libs/core/tests/core/__init__.py`:
```python
```

`libs/core/tests/utils/__init__.py`:
```python
```

`libs/core/tests/effects/__init__.py`:
```python
```

- [ ] **Step 6: Update root `pyproject.toml`**

Add `libs/core` to workspace members:

```toml
[tool.uv.workspace]
members = ["nanobot", "libs/core"]
```

Add workspace source for fastcode-core:

```toml
[tool.uv.sources]
nanobot-ai = { workspace = true }
fastcode-core = { workspace = true }
```

Add `fastcode-core` to dependencies:

```toml
dependencies = [
    # ... existing entries ...
    "fastcode-core",
]
```

Add pytest plugins to dev deps:

```toml
[project.optional-dependencies]
dev = [
    # ... existing entries ...
    "pytest-timeout",
    "pytest-subprocess",
]
```

Add timeout config to pytest:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "strict"
addopts = "-n auto"
timeout = 30
timeout_method = "thread"
```

Keep existing `[tool.setuptools.packages.find]` — it finds root `fastcode/` (the app):
```toml
[tool.setuptools.packages.find]
include = ["fastcode*"]
```

- [ ] **Step 7: Run `uv sync` and verify workspace**

```bash
uv sync
uv run python -c "import fastcode_core; print(fastcode_core.__file__)"
```

Expected: prints path to `libs/core/src/fastcode_core/__init__.py`

- [ ] **Step 8: Commit**

```bash
git add _fastcode/ fastcode/ libs/core/ pyproject.toml uv.lock
git commit -m "chore: archive fastcode/ to _fastcode/, scaffold libs/core workspace member

- Rename fastcode/ to _fastcode/ (archive original)
- Create new fastcode/ with app-level files only
- Add libs/core/ workspace member (fastcode-core)
- Add pytest-timeout and pytest-subprocess to dev deps
- Add Cargo-style workspace config to root pyproject.toml"
```

---

## Task 2: Schema — Core Types

**Files:**
- Create: `libs/core/src/fastcode_core/schema/core_types.py`
- Modify: `libs/core/src/fastcode_core/schema/__init__.py`

- [ ] **Step 1: Copy `_fastcode/core/types.py` to `libs/core/src/fastcode_core/schema/core_types.py`**

```bash
cp _fastcode/core/types.py libs/core/src/fastcode_core/schema/core_types.py
```

No changes needed — the file only imports from `dataclasses` and `typing` (stdlib).

- [ ] **Step 2: Update `libs/core/src/fastcode_core/schema/__init__.py` with re-exports**

```python
"""Schema — all frozen dataclasses and IR types."""

from fastcode_core.schema.core_types import (
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
uv run python -c "from fastcode_core.schema.core_types import Hit; print(Hit)"
```

Expected: `<class 'fastcode_core.schema.core_types.Hit'>`

- [ ] **Step 4: Commit**

```bash
git add libs/core/src/fastcode_core/schema/
git commit -m "feat: add fastcode-core schema/core_types.py — frozen dataclasses"
```

---

## Task 3: Schema — IR Types

**Files:**
- Create: `libs/core/src/fastcode_core/schema/ir.py`
- Modify: `libs/core/src/fastcode_core/schema/__init__.py`

- [ ] **Step 1: Create `libs/core/src/fastcode_core/schema/ir.py`**

Read `_fastcode/semantic_ir.py` and copy the following items. Adjust the import: replace `from .utils import safe_jsonable` with `from fastcode_core.utils.json import safe_jsonable` (will be created in Task 4).

Items to copy verbatim from `_fastcode/semantic_ir.py`:
- Helper functions: `_sorted_set`, `_normalize_set`, `_resolution_to_confidence`, `_confidence_to_resolution`, `_unit_kind_to_symbol_kind`, `_symbol_kind_to_unit_kind`
- Dataclasses: `IRDocument`, `IRSymbol`, `IROccurrence`, `IREdge`, `IRAttachment`
- The `IRSnapshot` class with its properties and methods

Top of file:
```python
"""Canonical IR dataclasses — extracted from fastcode.semantic_ir."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from fastcode_core.utils.json import safe_jsonable
```

Do NOT copy: `IRCodeUnit`, `IRUnitSupport`, `IRRelation`, `IRUnitEmbedding` — these are advanced types not used by the core modules. They stay in root `fastcode/semantic_ir.py`.

- [ ] **Step 2: Update `libs/core/src/fastcode_core/schema/__init__.py` — add IR re-exports**

Append to the existing file:
```python
from fastcode_core.schema.ir import (
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

Note: This will fail until Task 4 creates `fastcode_core.utils.json`. Verify after Task 4.

```bash
uv run python -c "from fastcode_core.schema.ir import IRSnapshot, IRSymbol; print(IRSnapshot, IRSymbol)"
```

Expected: both classes print without error

- [ ] **Step 4: Commit**

```bash
git add libs/core/src/fastcode_core/schema/ir.py libs/core/src/fastcode_core/schema/__init__.py
git commit -m "feat: add fastcode-core schema/ir.py — canonical IR dataclasses"
```

---

## Task 4: Utils Extraction

**Files:**
- Create: `libs/core/src/fastcode_core/utils/json.py`
- Create: `libs/core/src/fastcode_core/utils/hashing.py`
- Create: `libs/core/src/fastcode_core/utils/paths.py`
- Modify: `libs/core/src/fastcode_core/utils/__init__.py`

- [ ] **Step 1: Create `libs/core/src/fastcode_core/utils/json.py`**

Copy `safe_jsonable` from `_fastcode/utils.py` (lines 288-323) and all four parsing functions from `_fastcode/core/parsing.py`:

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


def extract_json_from_response(response: str) -> str:
    """Extract JSON string from LLM response, handling markdown blocks and reasoning text."""
    # Copy verbatim from _fastcode/core/parsing.py
    ...


def sanitize_json_string(json_str: str) -> str:
    """Sanitize JSON string to fix common issues from small models."""
    # Copy verbatim from _fastcode/core/parsing.py
    ...


def remove_json_comments(json_str: str) -> str:
    """Remove inline comments from JSON string (# or // style)."""
    # Copy verbatim from _fastcode/core/parsing.py
    ...


def robust_json_parse(json_str: str) -> Any:
    """Robustly parse JSON with multiple fallback strategies."""
    # Copy verbatim from _fastcode/core/parsing.py
    ...
```

Copy the four function bodies verbatim from `_fastcode/core/parsing.py`. They have no external dependencies beyond `ast`, `json`, `re`, and `typing` — all stdlib.

- [ ] **Step 2: Create `libs/core/src/fastcode_core/utils/hashing.py`**

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

- [ ] **Step 3: Create `libs/core/src/fastcode_core/utils/paths.py`**

Copy `get_language_from_extension` from `_fastcode/core/repo_analysis.py` and `projection_scope_key` from `_fastcode/core/snapshot.py`:

```python
"""Domain-independent path and extension utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def get_language_from_extension(ext: str) -> str:
    """Get programming language from extension."""
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".cpp": "cpp",
        ".c": "c",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
    }
    return language_map.get(ext.lower(), "unknown")


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

- [ ] **Step 4: Update `libs/core/src/fastcode_core/utils/__init__.py`**

```python
"""Utils — domain-independent utilities (copy-paste test)."""

from fastcode_core.utils.json import (
    extract_json_from_response,
    robust_json_parse,
    safe_jsonable,
    sanitize_json_string,
)
from fastcode_core.utils.hashing import deterministic_event_id, projection_params_hash
from fastcode_core.utils.paths import get_language_from_extension, projection_scope_key

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

- [ ] **Step 5: Verify all imports work**

```bash
uv run python -c "from fastcode_core.utils.json import safe_jsonable; from fastcode_core.utils.hashing import deterministic_event_id; from fastcode_core.utils.paths import get_language_from_extension; print('OK')"
uv run python -c "from fastcode_core.schema.ir import IRSnapshot; print('IR OK')"
```

Expected: `OK` and `IR OK`

- [ ] **Step 6: Commit**

```bash
git add libs/core/src/fastcode_core/utils/
git commit -m "feat: add fastcode-core utils/ — domain-independent json, hashing, paths"
```

---

## Task 5: Core Modules Move

**Files:**
- Create: 14 files in `libs/core/src/fastcode_core/core/` (copies with import adjustments)

- [ ] **Step 1: Copy all core modules from `_fastcode/core/`**

```bash
for f in scoring.py fusion.py filtering.py combination.py iteration.py \
         prompts.py context.py summary.py graph_build.py snapshot.py \
         repo_analysis.py scip_transform.py boundary.py; do
    cp "_fastcode/core/$f" "libs/core/src/fastcode_core/core/$f"
done
```

Note: `parsing.py` is NOT copied here — it was split into `utils/json.py` in Task 4.

- [ ] **Step 2: Fix imports in each file**

**`scoring.py`** — no changes (only stdlib imports)

**`fusion.py`** — change:
```python
# OLD:
from .scoring import (
    clone_result_row, normalized_query_entropy, normalized_totals,
    sigmoid, tokenize_signal, trace_confidence_weight, weighted_keyword_affinity,
)
# NEW:
from fastcode_core.core.scoring import (
    clone_result_row, normalized_query_entropy, normalized_totals,
    sigmoid, tokenize_signal, trace_confidence_weight, weighted_keyword_affinity,
)
```

**`filtering.py`** — no changes (only stdlib imports)

**`combination.py`** — no changes (only stdlib imports)

**`iteration.py`** — change:
```python
# OLD:
from .types import IterationConfig
# NEW:
from fastcode_core.schema.core_types import IterationConfig
```

**`prompts.py`** — no changes (only stdlib imports)

**`context.py`** — no changes (only stdlib imports)

**`summary.py`** — no changes (only stdlib imports)

**`graph_build.py`** — change:
```python
# OLD:
from ..semantic_ir import _resolution_to_confidence
# NEW:
from fastcode_core.schema.ir import _resolution_to_confidence
```
Also remove `deterministic_event_id` from this file — it moved to `utils/hashing.py`. Replace any usage with `from fastcode_core.utils.hashing import deterministic_event_id`.

**`snapshot.py`** — this file needs splitting:
- Remove `projection_scope_key` (moved to `fastcode_core/utils/paths.py`)
- Remove `projection_params_hash` (moved to `fastcode_core/utils/hashing.py`)
- Keep `extract_sources_from_elements` (domain-specific)

New `libs/core/src/fastcode_core/core/snapshot.py`:
```python
"""Pure snapshot logic — domain-specific parts."""

from __future__ import annotations

from typing import Any


def extract_sources_from_elements(
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract source information from retrieved elements."""
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

**`repo_analysis.py`** — remove `get_language_from_extension` (moved to `fastcode_core/utils/paths.py`). Add import:
```python
from fastcode_core.utils.paths import get_language_from_extension
```
Keep: `is_key_file`, `infer_project_type`, `generate_structure_based_overview`, `format_file_structure`.

**`scip_transform.py`** — no changes (no relative imports)

**`boundary.py`** — change:
```python
# OLD:
from fastcode.core.types import Hit
# NEW:
from fastcode_core.schema.core_types import Hit
```

- [ ] **Step 3: Verify all core modules import cleanly**

```bash
uv run python -c "
from fastcode_core.core.scoring import sigmoid
from fastcode_core.core.fusion import adaptive_fuse_channels
from fastcode_core.core.filtering import apply_filters
from fastcode_core.core.combination import combine_results
from fastcode_core.core.iteration import should_continue_iteration
from fastcode_core.core.prompts import format_elements_with_metadata
from fastcode_core.core.context import prepare_context
from fastcode_core.core.summary import generate_fallback_summary
from fastcode_core.core.graph_build import build_code_graph_payload
from fastcode_core.core.snapshot import extract_sources_from_elements
from fastcode_core.core.repo_analysis import infer_project_type
from fastcode_core.core.scip_transform import symbol_role_to_str
from fastcode_core.core.boundary import hit_to_response
print('All core modules OK')
"
```

Expected: `All core modules OK`

- [ ] **Step 4: Commit**

```bash
git add libs/core/src/fastcode_core/core/
git commit -m "feat: move core modules to fastcode-core with adjusted imports"
```

---

## Task 6: Effects Modules Move

**Files:**
- Create: `libs/core/src/fastcode_core/effects/db.py`, `llm.py`, `fs.py`

- [ ] **Step 1: Copy effects modules**

```bash
cp _fastcode/effects/db.py libs/core/src/fastcode_core/effects/db.py
cp _fastcode/effects/llm.py libs/core/src/fastcode_core/effects/llm.py
cp _fastcode/effects/fs.py libs/core/src/fastcode_core/effects/fs.py
```

- [ ] **Step 2: Fix imports**

**`db.py`** — change:
```python
# OLD:
from fastcode.core.types import SnapshotRecord
# NEW:
from fastcode_core.schema.core_types import SnapshotRecord
```

**`llm.py`** — no changes

**`fs.py`** — no changes

- [ ] **Step 3: Verify imports**

```bash
uv run python -c "from fastcode_core.effects.db import load_snapshot_record; from fastcode_core.effects.llm import chat_completion; from fastcode_core.effects.fs import read_file; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add libs/core/src/fastcode_core/effects/
git commit -m "feat: move effects modules to fastcode-core"
```

---

## Task 7: Tests Restructure

**Files:**
- Create: 21 test files in `libs/core/tests/` mirroring `libs/core/src/`

- [ ] **Step 1: Copy test files**

```bash
# Schema tests
cp tests/test_core_types.py libs/core/tests/schema/test_core_types.py

# Core tests
for f in scoring fusion filtering combination iteration prompts \
         context summary graph_build snapshot repo_analysis \
         scip_transform boundary; do
    cp "tests/test_core_${f}.py" "libs/core/tests/core/test_${f}.py"
done

# Effects tests
for f in db llm fs; do
    cp "tests/test_effects_${f}.py" "libs/core/tests/effects/test_${f}.py"
done

# Utils tests (parsing → json)
cp tests/test_core_parsing.py libs/core/tests/utils/test_json.py
```

- [ ] **Step 2: Update imports in all test files**

Every `from fastcode.core.X import ...` must become `from fastcode_core.core.X import ...`.
Every `from fastcode.core.types import ...` must become `from fastcode_core.schema.core_types import ...`.
Every `from fastcode.effects.X import ...` must become `from fastcode_core.effects.X import ...`.

For `test_json.py`: change `from fastcode.core.parsing import ...` to `from fastcode_core.utils.json import ...`.

Complete mapping:

| Test file | Old import | New import |
|-----------|-----------|------------|
| `tests/schema/test_core_types.py` | `from fastcode.core.types` | `from fastcode_core.schema.core_types` |
| `tests/core/test_scoring.py` | `from fastcode.core.scoring` | `from fastcode_core.core.scoring` |
| `tests/core/test_fusion.py` | `from fastcode.core.fusion` / `from fastcode.core.types` | `from fastcode_core.core.fusion` / `from fastcode_core.schema.core_types` |
| `tests/core/test_boundary.py` | `from fastcode.core.boundary` / `from fastcode.core.types` | `from fastcode_core.core.boundary` / `from fastcode_core.schema.core_types` |
| `tests/core/test_filtering.py` | `from fastcode.core.filtering` | `from fastcode_core.core.filtering` |
| `tests/core/test_combination.py` | `from fastcode.core.combination` | `from fastcode_core.core.combination` |
| `tests/core/test_iteration.py` | `from fastcode.core.iteration` / `from fastcode.core.types` | `from fastcode_core.core.iteration` / `from fastcode_core.schema.core_types` |
| `tests/core/test_prompts.py` | `from fastcode.core.prompts` | `from fastcode_core.core.prompts` |
| `tests/core/test_context.py` | `from fastcode.core.context` | `from fastcode_core.core.context` |
| `tests/core/test_summary.py` | `from fastcode.core.summary` | `from fastcode_core.core.summary` |
| `tests/core/test_graph_build.py` | `from fastcode.core.graph_build` | `from fastcode_core.core.graph_build` |
| `tests/core/test_snapshot.py` | `from fastcode.core.snapshot` | `from fastcode_core.core.snapshot` + `from fastcode_core.utils.hashing` + `from fastcode_core.utils.paths` |
| `tests/core/test_repo_analysis.py` | `from fastcode.core.repo_analysis` | `from fastcode_core.core.repo_analysis` |
| `tests/core/test_scip_transform.py` | `from fastcode.core.scip_transform` | `from fastcode_core.core.scip_transform` |
| `tests/utils/test_json.py` | `from fastcode.core.parsing` | `from fastcode_core.utils.json` |
| `tests/effects/test_db.py` | `from fastcode.core.types` / `from fastcode.effects.db` | `from fastcode_core.schema.core_types` / `from fastcode_core.effects.db` |
| `tests/effects/test_llm.py` | `from fastcode.effects.llm` | `from fastcode_core.effects.llm` |
| `tests/effects/test_fs.py` | `from fastcode.effects.fs` | `from fastcode_core.effects.fs` |

**Special: `test_snapshot.py`** — `projection_params_hash` and `projection_scope_key` moved:
```python
# OLD:
from fastcode.core.snapshot import (
    extract_sources_from_elements, projection_params_hash, projection_scope_key
)
# NEW:
from fastcode_core.core.snapshot import extract_sources_from_elements
from fastcode_core.utils.hashing import projection_params_hash
from fastcode_core.utils.paths import projection_scope_key
```

**Special: `test_boundary.py`** — update path references:
```python
# OLD: CORE_DIR = pathlib.Path(...) / "fastcode" / "core"
# NEW: CORE_DIR = pathlib.Path(...) / "libs" / "core" / "src" / "fastcode_core" / "core"

# OLD: db_effects = ... / "fastcode" / "effects" / "db.py"
# NEW: db_effects = ... / "libs" / "core" / "src" / "fastcode_core" / "effects" / "db.py"
```

- [ ] **Step 3: Create `libs/core/tests/utils/test_hashing.py`**

```python
"""Tests for fastcode_core.utils.hashing."""

from __future__ import annotations

from fastcode_core.utils.hashing import deterministic_event_id, projection_params_hash


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
uv run pytest libs/core/tests/ -v --timeout=30
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add libs/core/tests/
git commit -m "test: restructure tests into libs/core/tests/ mirroring src layout"
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

**`fastcode/retriever.py`** (lines 16-20):
```python
# OLD:
from .core import combination as _combination
from .core import filtering as _filtering
from .core import fusion as _fusion
from .core import scoring as _scoring
from .core.types import FusionConfig
# NEW:
from fastcode_core.core import combination as _combination
from fastcode_core.core import filtering as _filtering
from fastcode_core.core import fusion as _fusion
from fastcode_core.core import scoring as _scoring
from fastcode_core.schema.core_types import FusionConfig
```

**`fastcode/iterative_agent.py`** (lines 16-19):
```python
# OLD:
from .core import iteration as _iteration
from .core import parsing as _parsing
from .core import prompts as _prompts
from .core.types import IterationConfig
# NEW:
from fastcode_core.core import iteration as _iteration
from fastcode_core.utils import json as _json_utils
from fastcode_core.core import prompts as _prompts
from fastcode_core.schema.core_types import IterationConfig
```

Note: `parsing` module moved to `fastcode_core.utils.json`. Find all usages of `_parsing.` in this file and replace with `_json_utils.`.

**`fastcode/answer_generator.py`** (lines 15-16):
```python
# OLD:
from .core import context as _context
from .core import summary as _summary
# NEW:
from fastcode_core.core import context as _context
from fastcode_core.core import summary as _summary
```

**`fastcode/terminus_publisher.py`** (line 13):
```python
# OLD:
from .core import graph_build as _graph_build
# NEW:
from fastcode_core.core import graph_build as _graph_build
```

**`fastcode/main.py`** (line 25):
```python
# OLD:
from .core import snapshot as _snapshot
# NEW:
from fastcode_core.core import snapshot as _snapshot
```

**`fastcode/repo_overview.py`** (line 14):
```python
# OLD:
from .core import repo_analysis as _repo_analysis
# NEW:
from fastcode_core.core import repo_analysis as _repo_analysis
```

**`fastcode/scip_loader.py`** (line 13):
```python
# OLD:
from fastcode.core import scip_transform as _scip_transform
# NEW:
from fastcode_core.core import scip_transform as _scip_transform
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
git commit -m "refactor: update fastcode/ imports to use fastcode_core package"
```

---

## Task 9: Cleanup Old Test Files

**Files:**
- Delete: `tests/test_core_*.py` (15 files)
- Delete: `tests/test_effects_*.py` (3 files)

The original test files at `tests/` root are no longer needed — they've been copied to `libs/core/tests/`.

- [ ] **Step 1: Delete old core and effects test files**

```bash
rm tests/test_core_types.py tests/test_core_scoring.py tests/test_core_fusion.py \
   tests/test_core_boundary.py tests/test_core_filtering.py tests/test_core_combination.py \
   tests/test_core_iteration.py tests/test_core_prompts.py tests/test_core_parsing.py \
   tests/test_core_context.py tests/test_core_summary.py tests/test_core_graph_build.py \
   tests/test_core_snapshot.py tests/test_core_repo_analysis.py tests/test_core_scip_transform.py \
   tests/test_effects_db.py tests/test_effects_llm.py tests/test_effects_fs.py
```

- [ ] **Step 2: Run fastcode-core tests to confirm nothing broke**

```bash
uv run pytest libs/core/tests/ -v --timeout=30
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove old test files (migrated to libs/core/tests/)"
```

---

## Task 10: Pytest Config & Final Verification

**Files:**
- Modify: `pyproject.toml` (verify pytest config from Task 1)
- Create: `libs/core/tests/conftest.py`

- [ ] **Step 1: Verify pytest config in root `pyproject.toml`**

Check it has:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "strict"
addopts = "-n auto"
timeout = 30
timeout_method = "thread"
```

And dev dependencies include `pytest-timeout` and `pytest-subprocess`.

- [ ] **Step 2: Create `libs/core/tests/conftest.py`**

```python
"""Shared fixtures for fastcode_core tests."""
```

- [ ] **Step 3: Run all fastcode-core tests with timeout enforcement**

```bash
uv run pytest libs/core/tests/ -v --timeout=30
```

Expected: All tests pass, no timeouts triggered.

- [ ] **Step 4: Run ruff on new code**

```bash
uv run ruff check libs/core/ --fix
uv run ruff format libs/core/
```

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: add pytest-timeout/pytest-subprocess config, format new code"
```
