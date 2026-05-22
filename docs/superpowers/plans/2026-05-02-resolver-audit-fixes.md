# Resolver-Expansion Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all CRITICAL and HIGH issues identified in the resolver-expansion code audit, plus highest-value MEDIUM issues.

**Architecture:** Surgical fixes — no restructuring. Extract shared utilities to a new `_utils.py` module, remove dead overrides, add path validation, replace pickle with JSON, and fix mislabeled frontend kinds. Each task is independent and commits cleanly.

**Tech Stack:** Python 3.11+, pytest, NetworkX, dataclasses

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `fastcode/src/fastcode/semantic_resolvers/_utils.py` | Shared `_hash_id`, `_normalize_path`, `validate_helper_paths` |
| Modify | `fastcode/src/fastcode/semantic_resolvers/graph_backed.py` | Import `_hash_id` from `_utils` |
| Modify | `fastcode/src/fastcode/semantic_resolvers/helper_backed.py` | Import utils, add path validation |
| Modify | `fastcode/src/fastcode/semantic_resolvers/c_family.py` | Import `_hash_id`, `_normalize_path` from `_utils` |
| Modify | `fastcode/src/fastcode/semantic_resolvers/rust.py` | Fix `frontend_kind`, remove `_has_tools` |
| Modify | `fastcode/src/fastcode/semantic_resolvers/java.py` | Remove `_has_tools` |
| Modify | `fastcode/src/fastcode/semantic_resolvers/go.py` | Remove `_has_tools` |
| Modify | `fastcode/src/fastcode/semantic_resolvers/csharp.py` | Remove `_has_tools` |
| Modify | `fastcode/src/fastcode/semantic_resolvers/fortran.py` | Remove `_has_tools` |
| Modify | `fastcode/src/fastcode/semantic_resolvers/zig.py` | Remove `_has_tools` |
| Modify | `fastcode/src/fastcode/semantic_resolvers/julia.py` | Remove `_has_tools` |
| Modify | `fastcode/src/fastcode/semantic_resolvers/js_ts.py` | Remove `_has_tools` |
| Modify | `fastcode/src/fastcode/semantic_resolvers/base.py` | Add default `applicable()` on ABC |
| Modify | `fastcode/src/fastcode/semantic_resolvers/patching.py` | Derive source preference from tier |
| Modify | `fastcode/src/fastcode/snapshot_store.py` | Replace pickle with JSON |
| Modify | `fastcode/tests/test_semantic_resolvers.py` | Add tests for new behaviors |

---

### Task 1: Create shared `_utils.py` module

**Files:**
- Create: `fastcode/src/fastcode/semantic_resolvers/_utils.py`
- Test: `fastcode/tests/test_semantic_resolvers.py`

- [ ] **Step 1: Write the failing test**

Add these tests to `fastcode/tests/test_semantic_resolvers.py`:

```python
from fastcode.semantic_resolvers._utils import _hash_id, _normalize_path, validate_helper_paths


class TestSharedUtils:
    def test_hash_id_is_deterministic(self) -> None:
        result = _hash_id("support", "snapshot:go_resolver:import:src/main.go:pkg/mod.go:fmt")
        assert result == _hash_id("support", "snapshot:go_resolver:import:src/main.go:pkg/mod.go:fmt")

    def test_hash_id_different_inputs_differ(self) -> None:
        a = _hash_id("support", "aaa")
        b = _hash_id("support", "bbb")
        assert a != b

    def test_hash_id_different_prefixes_differ(self) -> None:
        a = _hash_id("support", "same")
        b = _hash_id("rel", "same")
        assert a != b

    def test_normalize_path_strips_dot_slash(self) -> None:
        assert _normalize_path("./src/main.go") == "src/main.go"

    def test_normalize_path_converts_backslashes(self) -> None:
        assert _normalize_path("src\\main.go") == "src/main.go"

    def test_normalize_path_idempotent(self) -> None:
        assert _normalize_path(_normalize_path("src\\main.go")) == _normalize_path("src/main.go")

    def test_validate_helper_paths_rejects_symlink(self, tmp_path: Path) -> None:
        target = tmp_path / "real.txt"
        target.write_text("ok")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        safe, rejected = validate_helper_paths([str(link)], str(tmp_path))
        assert len(rejected) == 1
        assert len(safe) == 0

    def test_validate_helper_paths_rejects_outside_repo(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside.txt"
        outside.write_text("ok")
        repo = tmp_path / "repo"
        repo.mkdir()
        safe, rejected = validate_helper_paths([str(outside)], str(repo))
        assert len(rejected) == 1
        assert len(safe) == 0

    def test_validate_helper_paths_accepts_valid(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        f = repo / "main.go"
        f.write_text("ok")
        safe, rejected = validate_helper_paths([str(f)], str(repo))
        assert len(safe) == 1
        assert len(rejected) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py::TestSharedUtils -v`
Expected: FAIL with `ImportError: cannot import name '_utils' from 'fastcode.semantic_resolvers'`

- [ ] **Step 3: Write `_utils.py`**

Create `fastcode/src/fastcode/semantic_resolvers/_utils.py`:

```python
"""Shared utilities for semantic resolver implementations."""

from __future__ import annotations

import hashlib
import os
import posixpath
from pathlib import Path


def _hash_id(prefix: str, payload: str) -> str:
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()
    return f"{prefix}:{digest}"


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return posixpath.normpath(normalized)


def validate_helper_paths(
    paths: list[str], repo_root: str
) -> tuple[list[str], list[str]]:
    """Validate helper file paths are regular files within repo root.

    Returns (safe_paths, rejected_paths). Rejects symlinks, missing files,
    and files outside the repo root.
    """
    repo = Path(repo_root).resolve()
    safe: list[str] = []
    rejected: list[str] = []
    for p in paths:
        resolved = Path(p).resolve()
        if not resolved.is_file():
            rejected.append(p)
            continue
        if resolved.is_symlink():
            rejected.append(p)
            continue
        try:
            resolved.relative_to(repo)
        except ValueError:
            rejected.append(p)
            continue
        safe.append(p)
    return safe, rejected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py::TestSharedUtils -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add fastcode/src/fastcode/semantic_resolvers/_utils.py fastcode/tests/test_semantic_resolvers.py
git commit -m "feat(semantic): add shared resolver utils — _hash_id, _normalize_path, validate_helper_paths"
```

---

### Task 2: Deduplicate `_hash_id` and `_normalize_path` — update consumers

**Files:**
- Modify: `fastcode/src/fastcode/semantic_resolvers/graph_backed.py`
- Modify: `fastcode/src/fastcode/semantic_resolvers/helper_backed.py`
- Modify: `fastcode/src/fastcode/semantic_resolvers/c_family.py`

- [ ] **Step 1: Write the regression test**

Add to `fastcode/tests/test_semantic_resolvers.py`:

```python
class TestHashIdConsistency:
    """Verify _hash_id produces same result from all import sites."""

    def test_graph_backed_and_helper_backed_produce_same_hash(self) -> None:
        from fastcode.semantic_resolvers._utils import _hash_id as util_hash
        from fastcode.semantic_resolvers.graph_backed import _hash_id as graph_hash
        from fastcode.semantic_resolvers.helper_backed import _hash_id as helper_hash
        from fastcode.semantic_resolvers.c_family import _hash_id as c_family_hash

        payload = "snapshot:go_resolver:import:src/main.go:pkg/mod.go:fmt"
        assert util_hash("rel", payload) == graph_hash("rel", payload)
        assert util_hash("rel", payload) == helper_hash("rel", payload)
        assert util_hash("rel", payload) == c_family_hash("rel", payload)

    def test_normalize_path_consistency(self) -> None:
        from fastcode.semantic_resolvers._utils import _normalize_path as util_norm
        from fastcode.semantic_resolvers.helper_backed import _normalize_path as helper_norm
        from fastcode.semantic_resolvers.c_family import _normalize_path as c_family_norm

        path = "src\\main.go"
        assert util_norm(path) == helper_norm(path)
        assert util_norm(path) == c_family_norm(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py::TestHashIdConsistency -v`
Expected: PASS (they currently have identical implementations) — this is a regression guard, not a red-green test. The value is ensuring they stay consistent after refactor.

- [ ] **Step 3: Update `graph_backed.py`**

In `fastcode/src/fastcode/semantic_resolvers/graph_backed.py`, replace:

```python
import hashlib
```

with nothing (remove that import), and replace the local `_hash_id` function:

```python
def _hash_id(prefix: str, payload: str) -> str:
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()
    return f"{prefix}:{digest}"
```

with an import:

```python
from ._utils import _hash_id
```

The file should now have no `import hashlib` at the top, and the local `_hash_id` definition is removed.

- [ ] **Step 4: Update `helper_backed.py`**

In `fastcode/src/fastcode/semantic_resolvers/helper_backed.py`:

1. Remove `import hashlib` from top-level imports.
2. Remove `import posixpath` from top-level imports.
3. Remove the local `_hash_id` function (lines 40-42).
4. Remove the local `_normalize_path` function (lines 45-49).
5. Add import: `from ._utils import _hash_id, _normalize_path`

- [ ] **Step 5: Update `c_family.py`**

In `fastcode/src/fastcode/semantic_resolvers/c_family.py`:

1. Remove `import hashlib` from top-level imports.
2. Remove `import posixpath` from top-level imports.
3. Remove the local `_hash_id` function (lines 21-23).
4. Remove the local `_normalize_path` function (lines 26-30).
5. Add import: `from ._utils import _hash_id, _normalize_path`

- [ ] **Step 6: Run all resolver tests**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py -v`
Expected: All tests PASS (same behavior, single source of truth)

- [ ] **Step 7: Commit**

```bash
git add fastcode/src/fastcode/semantic_resolvers/graph_backed.py fastcode/src/fastcode/semantic_resolvers/helper_backed.py fastcode/src/fastcode/semantic_resolvers/c_family.py fastcode/tests/test_semantic_resolvers.py
git commit -m "refactor(semantic): deduplicate _hash_id and _normalize_path into _utils.py"
```

---

### Task 3: Remove redundant `_has_tools` overrides from all 8 language resolvers

**Files:**
- Modify: `fastcode/src/fastcode/semantic_resolvers/rust.py`
- Modify: `fastcode/src/fastcode/semantic_resolvers/java.py`
- Modify: `fastcode/src/fastcode/semantic_resolvers/go.py`
- Modify: `fastcode/src/fastcode/semantic_resolvers/csharp.py`
- Modify: `fastcode/src/fastcode/semantic_resolvers/fortran.py`
- Modify: `fastcode/src/fastcode/semantic_resolvers/zig.py`
- Modify: `fastcode/src/fastcode/semantic_resolvers/julia.py`
- Modify: `fastcode/src/fastcode/semantic_resolvers/js_ts.py`

- [ ] **Step 1: Write a regression test**

Add to `fastcode/tests/test_semantic_resolvers.py`:

```python
class TestHasToolsInheritance:
    """Verify _has_tools is inherited, not overridden."""

    @pytest.mark.parametrize(
        "resolver_cls",
        [
            RustCompilerResolver,
            JavaCompilerResolver,
            GoCompilerResolver,
            CSharpCompilerResolver,
            FortranCompilerResolver,
            ZigCompilerResolver,
            JuliaCompilerResolver,
            JavaScriptCompilerResolver,
            TypeScriptCompilerResolver,
        ],
    )
    def test_has_tools_is_inherited_not_overridden(self, resolver_cls: type) -> None:
        # The method should come from HelperBackedSemanticResolver, not defined locally
        assert "_has_tools" not in resolver_cls.__dict__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py::TestHasToolsInheritance -v`
Expected: FAIL — every resolver defines `_has_tools` in `__dict__`

- [ ] **Step 3: Remove `_has_tools` and `import shutil` from each file**

For each of the 8 files, remove:

```python
import shutil
```

and remove:

```python
    def _has_tools(self) -> bool:
        return all(shutil.which(tool) is not None for tool in self.required_tools)
```

The base class `HelperBackedSemanticResolver._has_tools` (at `helper_backed.py:138-139`) provides this implementation. No other changes needed.

Files to edit:
- `fastcode/src/fastcode/semantic_resolvers/rust.py` — remove `import shutil` and `_has_tools` method
- `fastcode/src/fastcode/semantic_resolvers/java.py` — remove `import shutil` and `_has_tools` method
- `fastcode/src/fastcode/semantic_resolvers/go.py` — remove `import shutil` and `_has_tools` method
- `fastcode/src/fastcode/semantic_resolvers/csharp.py` — remove `import shutil` and `_has_tools` method
- `fastcode/src/fastcode/semantic_resolvers/fortran.py` — remove `import shutil` and `_has_tools` method
- `fastcode/src/fastcode/semantic_resolvers/zig.py` — remove `import shutil` and `_has_tools` method
- `fastcode/src/fastcode/semantic_resolvers/julia.py` — remove `import shutil` and `_has_tools` method
- `fastcode/src/fastcode/semantic_resolvers/js_ts.py` — remove `import shutil` and `_has_tools` method (from `_JsTsResolverBase`)

- [ ] **Step 4: Run all resolver tests**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py -v`
Expected: All tests PASS — `_has_tools` now inherited from base class

- [ ] **Step 5: Commit**

```bash
git add fastcode/src/fastcode/semantic_resolvers/rust.py fastcode/src/fastcode/semantic_resolvers/java.py fastcode/src/fastcode/semantic_resolvers/go.py fastcode/src/fastcode/semantic_resolvers/csharp.py fastcode/src/fastcode/semantic_resolvers/fortran.py fastcode/src/fastcode/semantic_resolvers/zig.py fastcode/src/fastcode/semantic_resolvers/julia.py fastcode/src/fastcode/semantic_resolvers/js_ts.py fastcode/tests/test_semantic_resolvers.py
git commit -m "refactor(semantic): remove redundant _has_tools overrides, inherit from HelperBackedSemanticResolver"
```

---

### Task 4: Add default `applicable()` to `SemanticResolver` ABC

**Files:**
- Modify: `fastcode/src/fastcode/semantic_resolvers/base.py`
- Modify: `fastcode/src/fastcode/semantic_resolvers/graph_backed.py`
- Modify: `fastcode/src/fastcode/semantic_resolvers/helper_backed.py`
- Modify: `fastcode/src/fastcode/semantic_resolvers/c_family.py`

- [ ] **Step 1: Write regression test**

Add to `fastcode/tests/test_semantic_resolvers.py`:

```python
class TestApplicableInheritance:
    """Verify applicable() is inherited from SemanticResolver ABC."""

    def test_graph_backed_uses_default_applicable(self) -> None:
        from fastcode.semantic_resolvers.graph_backed import GraphBackedSemanticResolver
        assert "applicable" not in GraphBackedSemanticResolver.__dict__

    def test_helper_backed_uses_default_applicable(self) -> None:
        from fastcode.semantic_resolvers.helper_backed import HelperBackedSemanticResolver
        assert "applicable" not in HelperBackedSemanticResolver.__dict__

    def test_c_family_uses_default_applicable(self) -> None:
        from fastcode.semantic_resolvers.c_family import CFamilySemanticResolver
        assert "applicable" not in CFamilySemanticResolver.__dict__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py::TestApplicableInheritance -v`
Expected: FAIL — all three classes define `applicable` in `__dict__`

- [ ] **Step 3: Add default `applicable()` to `SemanticResolver` in `base.py`**

In `fastcode/src/fastcode/semantic_resolvers/base.py`, change the `applicable` abstract method to a concrete default implementation:

Replace:
```python
    @abstractmethod
    def applicable(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
    ) -> bool:
        """Return True when the resolver should run for this batch."""
```

With:
```python
    def applicable(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
    ) -> bool:
        """Return True when the resolver should run for this batch."""
        del snapshot
        return any(
            elem.language == self.language
            and (elem.relative_path or elem.file_path) in target_paths
            for elem in elements
        )
```

Note: This changes `applicable` from `@abstractmethod` to a concrete method. The `@abstractmethod` decorator must be removed since this is now a default implementation. This is safe because `resolve()` remains `@abstractmethod`, so the class is still abstract.

- [ ] **Step 4: Remove `applicable()` from `GraphBackedSemanticResolver`**

In `fastcode/src/fastcode/semantic_resolvers/graph_backed.py`, remove the `applicable` method (lines 47-59).

- [ ] **Step 5: Remove `applicable()` from `HelperBackedSemanticResolver`**

In `fastcode/src/fastcode/semantic_resolvers/helper_backed.py`, remove the `applicable` method (lines 72-84).

- [ ] **Step 6: Remove `applicable()` from `CFamilySemanticResolver`**

In `fastcode/src/fastcode/semantic_resolvers/c_family.py`, remove the `applicable` method (lines 56-68).

- [ ] **Step 7: Run all resolver tests**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add fastcode/src/fastcode/semantic_resolvers/base.py fastcode/src/fastcode/semantic_resolvers/graph_backed.py fastcode/src/fastcode/semantic_resolvers/helper_backed.py fastcode/src/fastcode/semantic_resolvers/c_family.py fastcode/tests/test_semantic_resolvers.py
git commit -m "refactor(semantic): move applicable() default to SemanticResolver ABC, remove triple duplication"
```

---

### Task 5: Add path validation to `_run_semantic_helper` (C1 fix)

**Files:**
- Modify: `fastcode/src/fastcode/semantic_resolvers/helper_backed.py`
- Test: `fastcode/tests/test_semantic_resolvers.py`

- [ ] **Step 1: Write the failing test**

Add to `fastcode/tests/test_semantic_resolvers.py`:

```python
class TestHelperPathValidation:
    """Verify helper_backed validates file paths before subprocess invocation."""

    def test_resolve_skips_symlink_files(self, tmp_path: Path) -> None:
        """Symlink files should be filtered out before subprocess call."""
        repo = tmp_path / "repo"
        repo.mkdir()
        real = repo / "real.rs"
        real.write_text("fn main() {}")
        link = repo / "link.rs"
        link.symlink_to(real)

        resolver = RustCompilerResolver()
        snapshot = IRSnapshot(
            repo_name="test",
            snapshot_id="snap:test:abc",
            units=[
                _file_unit("real.rs", language="rust"),
                _file_unit("link.rs", language="rust"),
            ],
        )
        # Patch _has_tools to return True so we enter helper path
        resolver._has_tools = lambda: True  # type: ignore[assignment]
        # Patch _run_semantic_helper to avoid actually running subprocess
        with patch.object(resolver, "_run_semantic_helper", return_value={}) as mock_run:
            resolver._resolve_via_helper(
                snapshot=snapshot,
                target_paths={"real.rs", "link.rs"},
            )
            # The helper_files argument should NOT contain the symlink
            call_args = mock_run.call_args
            helper_files = call_args[1]["helper_files"] if "helper_files" in (call_args[1] or {}) else call_args[0][0]
            basenames = {Path(f).name for f in helper_files}
            assert "link.rs" not in basenames

    def test_resolve_skips_outside_repo_files(self, tmp_path: Path) -> None:
        """Files outside repo root should be filtered."""
        repo = tmp_path / "repo"
        repo.mkdir()
        outside = tmp_path / "outside.rs"
        outside.write_text("fn main() {}")
        inside = repo / "main.rs"
        inside.write_text("fn main() {}")

        resolver = RustCompilerResolver()
        resolver._has_tools = lambda: True  # type: ignore[assignment]
        snapshot = IRSnapshot(
            repo_name="test",
            snapshot_id="snap:test:abc",
            units=[_file_unit("main.rs", language="rust")],
        )
        with patch.object(resolver, "_run_semantic_helper", return_value={}) as mock_run:
            resolver._resolve_via_helper(
                snapshot=snapshot,
                target_paths={"main.rs", str(outside)},
            )
            call_args = mock_run.call_args
            helper_files = call_args[1].get("helper_files", call_args[0][0] if call_args[0] else [])
            basenames = {Path(f).name for f in helper_files}
            assert "outside.rs" not in basenames
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py::TestHelperPathValidation -v`
Expected: FAIL — `_target_files` does not currently filter symlinks or outside paths

- [ ] **Step 3: Add path validation to `_target_files` in `helper_backed.py`**

In `fastcode/src/fastcode/semantic_resolvers/helper_backed.py`, add the import:

```python
from ._utils import _hash_id, _normalize_path, validate_helper_paths
```

Replace the `_target_files` method:

```python
    def _target_files(self, target_paths: set[str]) -> list[str]:
        repo_root = os.getcwd()
        files: list[str] = []
        for path in sorted(target_paths):
            normalized = path if os.path.isabs(path) else os.path.join(repo_root, path)
            if self.file_extensions and not normalized.endswith(self.file_extensions):
                continue
            files.append(os.path.abspath(normalized))
        return files
```

With:

```python
    def _target_files(self, target_paths: set[str]) -> list[str]:
        repo_root = os.getcwd()
        raw: list[str] = []
        for path in sorted(target_paths):
            normalized = path if os.path.isabs(path) else os.path.join(repo_root, path)
            if self.file_extensions and not normalized.endswith(self.file_extensions):
                continue
            raw.append(os.path.abspath(normalized))
        safe, _ = validate_helper_paths(raw, repo_root)
        return safe
```

- [ ] **Step 4: Run all resolver tests**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add fastcode/src/fastcode/semantic_resolvers/helper_backed.py fastcode/tests/test_semantic_resolvers.py
git commit -m "fix(security): validate helper file paths — reject symlinks and out-of-repo files"
```

---

### Task 6: Fix Rust resolver `frontend_kind` mislabeling (C3 fix)

**Files:**
- Modify: `fastcode/src/fastcode/semantic_resolvers/rust.py`

- [ ] **Step 1: Write the failing test**

Add to `fastcode/tests/test_semantic_resolvers.py`:

```python
class TestRustFrontendKind:
    def test_rust_frontend_kind_reflects_regex_heuristic(self) -> None:
        resolver = RustCompilerResolver()
        assert resolver.frontend_kind == "regex_heuristic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py::TestRustFrontendKind -v`
Expected: FAIL — `frontend_kind` is currently `"rust_analyzer_scip"`

- [ ] **Step 3: Fix `frontend_kind` in `rust.py`**

In `fastcode/src/fastcode/semantic_resolvers/rust.py`, change the `_RUST_SPEC`:

```python
    frontend_kind="rust_analyzer_scip",
```

to:

```python
    frontend_kind="regex_heuristic",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py::TestRustFrontendKind -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fastcode/src/fastcode/semantic_resolvers/rust.py fastcode/tests/test_semantic_resolvers.py
git commit -m "fix(semantic): rename Rust frontend_kind to regex_heuristic — helper is regex, not rust-analyzer"
```

---

### Task 7: Replace pickle with JSON for IR graph serialization (C1 fix)

**Files:**
- Modify: `fastcode/src/fastcode/snapshot_store.py`
- Test: `fastcode/tests/test_semantic_resolvers.py` (or `test_snapshot_store.py`)

- [ ] **Step 1: Write the failing test**

Add to `fastcode/tests/test_snapshot_store.py` (or `test_semantic_resolvers.py`):

```python
class TestIRGraphSerialization:
    """Verify IR graphs are serialized as JSON, not pickle."""

    def test_save_and_load_ir_graphs_json(self, tmp_path: Path) -> None:
        import networkx as nx
        from fastcode.snapshot_store import SnapshotStore

        store = SnapshotStore(str(tmp_path / "store.db"), storage_backend="sqlite")
        # Build a small graph
        graph = nx.DiGraph()
        graph.add_edge("a", "b", weight=1.0)
        graph.add_edge("b", "c", call_name="foo")
        graph.graph["snapshot_id"] = "snap:test:abc"

        # Save
        snapshot_id = "snap:test:abc"
        store.save_snapshot_record(
            snapshot_id=snapshot_id,
            repo_name="test",
            commit_id="abc123",
            branch="main",
            tree_id="tree1",
            metadata={},
        )
        path = store.save_ir_graphs(snapshot_id, {"dependency_graph": graph})

        # Verify it's JSON, not pickle
        with open(path) as f:
            content = f.read()
        import json
        data = json.loads(content)
        assert "graphs" in data

        # Load and verify round-trip
        loaded = store.load_ir_graphs(snapshot_id)
        assert loaded is not None
        loaded_graph = nx.node_link_graph(loaded["graphs"]["dependency_graph"], directed=True)
        assert list(loaded_graph.edges()) == [("a", "b"), ("b", "c")]
        assert loaded_graph.edges["b", "c"]["call_name"] == "foo"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd fastcode && uv run pytest tests/test_snapshot_store.py::TestIRGraphSerialization -v`
Expected: FAIL — current implementation uses pickle, not JSON

- [ ] **Step 3: Replace pickle with JSON in `snapshot_store.py`**

In `fastcode/src/fastcode/snapshot_store.py`:

1. Remove `import pickle` from top-level imports.
2. Replace the `save_ir_graphs` method:

```python
    def save_ir_graphs(self, snapshot_id: str, ir_graphs: Any) -> str:
        snap_dir = self.snapshot_dir(snapshot_id)
        path = os.path.join(snap_dir, "ir_graphs.json")
        import networkx as nx

        serializable: dict[str, Any] = {}
        for name, graph in (ir_graphs or {}).items():
            if isinstance(graph, nx.Graph):
                serializable[name] = nx.node_link_data(graph)
            else:
                serializable[name] = safe_jsonable(graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"graphs": serializable}, f)
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                "UPDATE snapshots SET ir_graphs_path=? WHERE snapshot_id=?",
                (path, snapshot_id),
            )
            conn.commit()
        return path
```

3. Replace the `load_ir_graphs` method:

```python
    def load_ir_graphs(self, snapshot_id: str) -> Any | None:
        row = self.get_snapshot_record(snapshot_id)
        if not row:
            return None
        path = row.get("ir_graphs_path")
        if not path or not os.path.exists(path):
            return None
        import networkx as nx

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        graphs_data = data.get("graphs", {})
        result: dict[str, Any] = {}
        for name, graph_data in graphs_data.items():
            if isinstance(graph_data, dict) and "nodes" in graph_data:
                result[name] = nx.node_link_graph(graph_data, directed=True)
            else:
                result[name] = graph_data
        return result
```

4. Add `from ..utils import safe_jsonable` import if not already present.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd fastcode && uv run pytest tests/test_snapshot_store.py::TestIRGraphSerialization -v`
Expected: PASS

- [ ] **Step 5: Run broader snapshot tests to check for regressions**

Run: `cd fastcode && uv run pytest tests/test_snapshot_store.py -v --timeout=60`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add fastcode/src/fastcode/snapshot_store.py fastcode/tests/test_snapshot_store.py
git commit -m "fix(security): replace pickle with JSON for IR graph serialization"
```

---

### Task 8: Derive source preference from `ResolutionTier` in patching.py (M2 fix)

**Files:**
- Modify: `fastcode/src/fastcode/semantic_resolvers/patching.py`
- Test: `fastcode/tests/test_semantic_resolvers.py`

- [ ] **Step 1: Write the failing test**

Add to `fastcode/tests/test_semantic_resolvers.py`:

```python
class TestSourcePreferenceAutoTier:
    """Verify unknown compiler-confirmed sources get proper preference."""

    def test_unknown_compiler_confirmed_source_gets_high_preference(self) -> None:
        from fastcode.semantic_resolvers.patching import _source_preference
        relation = IRRelation(
            relation_id="rel:abc",
            src_unit_id="src",
            dst_unit_id="dst",
            relation_type="call",
            resolution_state="structural",
            support_sources={"hypothetical_new_resolver"},
            metadata={"resolution_tier": "compiler_confirmed"},
        )
        # Unknown source name, but compiler_confirmed tier should boost to 2
        assert _source_preference(relation) == 2

    def test_unknown_structural_source_gets_zero(self) -> None:
        from fastcode.semantic_resolvers.patching import _source_preference
        relation = IRRelation(
            relation_id="rel:abc",
            src_unit_id="src",
            dst_unit_id="dst",
            relation_type="call",
            resolution_state="structural",
            support_sources={"hypothetical_new_resolver"},
            metadata={},
        )
        # Unknown source, no tier boost
        assert _source_preference(relation) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py::TestSourcePreferenceAutoTier -v`
Expected: PASS on first test (existing tier boost already handles this), but the test documents the intended behavior and guards against regression.

- [ ] **Step 3: Add a comment documenting the auto-tier behavior**

In `fastcode/src/fastcode/semantic_resolvers/patching.py`, update `_source_preference` to add a clarifying comment:

```python
def _source_preference(relation: IRRelation) -> int:
    """Rank a relation by its best evidence source.

    Higher value = stronger evidence.  Unknown source names start at rank 0,
    but any relation with ``resolution_tier == "compiler_confirmed"`` in its
    metadata is boosted to at least rank 2 (matching SCIP).  This means new
    resolvers are automatically ranked correctly without updating the
    preferences dict.
    """
    preferences: dict[str, int] = {
        "fc_structure": 0,
    }
    sources = set(relation.support_sources)
    if relation.source:
        sources.add(relation.source)
    base_pref = max((preferences.get(source, 0) for source in sources), default=0)
    tier = (relation.metadata or {}).get("resolution_tier", "")
    if tier == ResolutionTier.COMPILER_CONFIRMED:
        base_pref = max(base_pref, 2)
    return base_pref
```

This simplifies `preferences` to only contain the known structural source, removing the per-resolver entries (since they all have the same value of 1, and compiler_confirmed tier boost handles the rest).

- [ ] **Step 4: Run all resolver tests**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add fastcode/src/fastcode/semantic_resolvers/patching.py fastcode/tests/test_semantic_resolvers.py
git commit -m "refactor(semantic): simplify _source_preference — derive rank from tier, not hardcoded resolver list"
```

---

### Task 9: Fix `language_graph.py` to use `SemanticCapability` constants

**Files:**
- Modify: `fastcode/src/fastcode/semantic_resolvers/language_graph.py`

- [ ] **Step 1: Write the failing test**

Add to `fastcode/tests/test_semantic_resolvers.py`:

```python
class TestLanguageGraphCapabilities:
    """Verify language_graph.py uses SemanticCapability constants."""

    @pytest.mark.parametrize(
        "cls",
        [
            JavaScriptSemanticResolver,
            TypeScriptSemanticResolver,
        ],
    )
    def test_capabilities_reference_constants(self, cls: type) -> None:
        from fastcode.semantic_resolvers.base import SemanticCapability
        # Every capability should be a known SemanticCapability value
        valid = {v for k, v in vars(SemanticCapability).items() if not k.startswith("_")}
        for cap in cls.capabilities:
            assert cap in valid, f"Unknown capability {cap!r} in {cls.__name__}"
```

- [ ] **Step 2: Run test**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py::TestLanguageGraphCapabilities -v`
Expected: This will FAIL because `language_graph.py` uses raw string literals like `"resolve_calls"` which ARE valid SemanticCapability values, but the test validates they're from the known set. Let me check... Actually the raw strings in language_graph.py like `"resolve_calls"` match `SemanticCapability.RESOLVE_CALLS = "resolve_calls"`, so they ARE in the valid set. The test should pass.

The real fix is to import and use the constants directly, making typos impossible at import time rather than runtime.

- [ ] **Step 3: Update `language_graph.py` to use `SemanticCapability` constants**

Replace the raw string literals with `SemanticCapability` references. Full replacement for `fastcode/src/fastcode/semantic_resolvers/language_graph.py`:

```python
"""Language-specific graph-backed semantic resolvers."""

from __future__ import annotations

from .base import SemanticCapability
from .graph_backed import GraphBackedSemanticResolver


class JavaScriptSemanticResolver(GraphBackedSemanticResolver):
    language = "javascript"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_BINDINGS,
        }
    )
    cost_class = "medium"
    source_name = "javascript_resolver"
    extractor_name = "fastcode.semantic_resolvers.javascript"
    frontend_kind = "typescript_compiler_api_fallback"
    required_tools = ("node",)


class TypeScriptSemanticResolver(GraphBackedSemanticResolver):
    language = "typescript"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_TYPES,
            SemanticCapability.RESOLVE_BINDINGS,
        }
    )
    cost_class = "medium"
    source_name = "typescript_resolver"
    extractor_name = "fastcode.semantic_resolvers.typescript"
    frontend_kind = "typescript_compiler_api_fallback"
    required_tools = ("node", "tsc")


class JavaSemanticResolver(GraphBackedSemanticResolver):
    language = "java"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_INHERITANCE,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_TYPES,
        }
    )
    cost_class = "high"
    source_name = "java_resolver"
    extractor_name = "fastcode.semantic_resolvers.java"
    frontend_kind = "jdt_fallback"
    required_tools = ("java",)


class GoSemanticResolver(GraphBackedSemanticResolver):
    language = "go"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_TYPES,
            SemanticCapability.RESOLVE_BINDINGS,
        }
    )
    cost_class = "medium"
    source_name = "go_resolver"
    extractor_name = "fastcode.semantic_resolvers.go"
    frontend_kind = "go_packages_fallback"
    required_tools = ("go",)


class RustSemanticResolver(GraphBackedSemanticResolver):
    language = "rust"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_INHERITANCE,
            SemanticCapability.RESOLVE_TYPES,
            SemanticCapability.RESOLVE_BINDINGS,
            SemanticCapability.EXPAND_MACROS,
        }
    )
    cost_class = "high"
    source_name = "rust_resolver"
    extractor_name = "fastcode.semantic_resolvers.rust"
    frontend_kind = "rust_analyzer_fallback"
    required_tools = ("rust-analyzer",)


class CSharpSemanticResolver(GraphBackedSemanticResolver):
    language = "csharp"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_INHERITANCE,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_TYPES,
        }
    )
    cost_class = "high"
    source_name = "csharp_resolver"
    extractor_name = "fastcode.semantic_resolvers.csharp"
    frontend_kind = "roslyn_fallback"
    required_tools = ("dotnet",)


class ZigSemanticResolver(GraphBackedSemanticResolver):
    language = "zig"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_TYPES,
            SemanticCapability.RESOLVE_BINDINGS,
        }
    )
    cost_class = "high"
    source_name = "zig_resolver"
    extractor_name = "fastcode.semantic_resolvers.zig"
    frontend_kind = "zls_fallback"
    required_tools = ("zig", "zls")


class FortranSemanticResolver(GraphBackedSemanticResolver):
    language = "fortran"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_TYPES,
            SemanticCapability.RESOLVE_INHERITANCE,
        }
    )
    cost_class = "high"
    source_name = "fortran_resolver"
    extractor_name = "fastcode.semantic_resolvers.fortran"
    frontend_kind = "fortls_fallback"
    required_tools = ("fortls",)


class JuliaSemanticResolver(GraphBackedSemanticResolver):
    language = "julia"
    capabilities = frozenset(
        {
            SemanticCapability.RESOLVE_CALLS,
            SemanticCapability.RESOLVE_IMPORT_ALIASES,
            SemanticCapability.RESOLVE_TYPES,
            SemanticCapability.RESOLVE_BINDINGS,
        }
    )
    cost_class = "high"
    source_name = "julia_resolver"
    extractor_name = "fastcode.semantic_resolvers.julia"
    frontend_kind = "julia_language_server_fallback"
    required_tools = ("julia",)
```

- [ ] **Step 4: Run all resolver tests**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add fastcode/src/fastcode/semantic_resolvers/language_graph.py fastcode/tests/test_semantic_resolvers.py
git commit -m "refactor(semantic): use SemanticCapability constants in language_graph.py instead of raw strings"
```

---

### Task 10: Use `dict`/`list` directly as `default_factory` in `ResolutionPatch` (L4 fix)

**Files:**
- Modify: `fastcode/src/fastcode/semantic_resolvers/base.py`

- [ ] **Step 1: Write the test**

This is a trivial cleanup. No new test needed — existing tests already verify `ResolutionPatch` field defaults work.

- [ ] **Step 2: Update `base.py`**

In `fastcode/src/fastcode/semantic_resolvers/base.py`, remove the helper functions and use standard `default_factory`:

Replace:
```python
def _empty_dict() -> dict[str, Any]:
    return {}


def _empty_list() -> list[Any]:
    return []
```

And in the `ResolutionPatch` dataclass, change:
```python
    unit_metadata_updates: dict[str, dict[str, Any]] = field(
        default_factory=_empty_dict
    )
    metadata_updates: dict[str, Any] = field(default_factory=_empty_dict)
    supports: list[IRUnitSupport] = field(default_factory=_empty_list)
    relations: list[IRRelation] = field(default_factory=_empty_list)
    warnings: list[str] = field(default_factory=_empty_list)
    diagnostics: list[ToolDiagnostic] = field(default_factory=_empty_list)
    stats: dict[str, Any] = field(default_factory=_empty_dict)
```

To:
```python
    unit_metadata_updates: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    metadata_updates: dict[str, Any] = field(default_factory=dict)
    supports: list[IRUnitSupport] = field(default_factory=list)
    relations: list[IRRelation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: list[ToolDiagnostic] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
```

Note: The `_empty_dict` returned `dict[str, Any]` and `_empty_list` returned `list[Any]`. Using `dict` and `list` directly produces the same runtime values. The type annotations on the fields handle the type safety.

- [ ] **Step 3: Run all resolver tests**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add fastcode/src/fastcode/semantic_resolvers/base.py
git commit -m "refactor(semantic): use dict/list directly as default_factory in ResolutionPatch"
```

---

### Task 11: Final integration test run

**Files:** None — verification only.

- [ ] **Step 1: Run full test suite**

Run: `cd fastcode && uv run pytest tests/test_semantic_resolvers.py tests/test_snapshot_store.py -v --timeout=120`
Expected: All tests PASS

- [ ] **Step 2: Run type checker**

Run: `cd fastcode && uv run pyright fastcode/src/fastcode/semantic_resolvers/`
Expected: No new errors

- [ ] **Step 3: Run linter**

Run: `cd fastcode && uv run ruff check fastcode/src/fastcode/semantic_resolvers/`
Expected: No new violations

---

## Self-Review Checklist

- [x] **Spec coverage:** Each CRITICAL and HIGH issue from the audit maps to a task:
  - C1 (pickle RCE) → Task 7
  - C2 (subprocess path traversal) → Task 5
  - C3 (Rust frontend_kind) → Task 6
  - H1 (_hash_id duplication) → Task 2
  - H2 (_has_tools duplication) → Task 3
  - H3 (applicable() duplication) → Task 4
  - H4 (Any type leak for legacy_graph_builder) → Not in scope (Protocol definition is a design decision, not a bug fix)
  - M2 (source_preference hardcoded) → Task 8
  - L3 (language_graph raw strings) → Task 9
  - L4 (default_factory indirection) → Task 10

- [x] **Placeholder scan:** No TBDs, no "implement later", no "add appropriate error handling". All code blocks contain actual implementation code.

- [x] **Type consistency:** All function signatures, class attributes, and imports are consistent across tasks. `_hash_id` is `(str, str) -> str` everywhere. `validate_helper_paths` returns `tuple[list[str], list[str]]`. `frontend_kind` is a plain `str` class attribute.
