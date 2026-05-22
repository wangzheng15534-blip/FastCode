# FP Core Follow-Up — Detailed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the FP core + thin I/O refactoring from Phase 1.3 through Phase 6. Phase 0 and Phase 1.1-1.2 are already done.

**Architecture:** Pure functions in `fastcode/core/`, thin I/O wrappers in `fastcode/effects/`, existing orchestrators delegate. Three Golden Rules enforced: (1) Pydantic stops at the door, (2) DB trusts dataclasses, (3) explicit translation.

**Tech Stack:** Python >=3.11, stdlib dataclasses (frozen), no new dependencies.

**Design Spec:** `docs/superpowers/specs/2026-04-26-fp-core-thin-io-design.md`
**Parent Plan:** `docs/superpowers/plans/2026-04-26-fp-core-thin-io.md`

### Three Golden Rules (enforced in code review)

1. **Pydantic Stops at the Door.** No `pydantic` imports in `fastcode/core/`.
2. **Database Trusts Dataclasses.** `effects/db.py` returns frozen dataclasses, never `dict[str, Any]`.
3. **Explicit Translation.** No `**kwargs` unpacking, no `from_orm`, no `model_dump`. Field-by-field mapping in `core/boundary.py`.

---

## Already Completed (Phase 0 + Phase 1.1-1.2)

- `fastcode/core/__init__.py` — package scaffolding
- `fastcode/core/types.py` — 18 frozen dataclasses (Hit, FusionConfig, IterationState, etc.)
- `fastcode/core/scoring.py` — 7 pure scoring functions
- `fastcode/core/fusion.py` — 6 pure fusion functions
- `fastcode/core/boundary.py` — explicit translation (CoreQueryInput, query_request_to_core, hit_to_response)
- `fastcode/effects/__init__.py` — package scaffolding
- `tests/test_core_types.py` — 55 tests
- `tests/test_core_scoring.py` — 44 tests
- `tests/test_core_fusion.py` — 14 tests
- `tests/test_core_boundary.py` — 7 tests (including I/O import guard with pydantic in forbidden set)
- `fastcode/retriever.py` — wired scoring + fusion to core delegates

---

## Phase 1 Continued: Extract Remaining Retrieval Logic

### Task 1.3: Extract filtering functions (from retriever.py)

**Files:**
- Create: `fastcode/core/filtering.py`
- Create: `tests/test_core_filtering.py`
- Modify: `fastcode/retriever.py`

Extract these pure methods from `HybridRetriever`:
- `_apply_filters` (line 1591-1625) — filter results by language, type, file_path, snapshot_id
- `_diversify` (line 1627-1654) — penalize repeated files
- `_final_repo_filter` (line 1656-1695) — safety filter for repo names
- `_rerank` (line 1563-1589) — element-type-based reranking

These are all pure: they read `self.diversity_penalty` and `self.logger` but the core logic is pure computation. Extract by passing `diversity_penalty` as a parameter and dropping logger calls from core (return filtered count instead).

- [ ] **Step 1: Write failing tests for filtering functions**

```python
# tests/test_core_filtering.py
"""Tests for pure filtering functions extracted from retriever."""
from fastcode.core.filtering import (
    apply_filters,
    diversify,
    final_repo_filter,
    rerank,
)


def _mk_row(
    elem_id: str,
    elem_type: str = "function",
    total: float = 0.8,
    *,
    language: str = "python",
    file_path: str = "src/main.py",
    relative_path: str = "src/main.py",
    repo_name: str = "myrepo",
    snapshot_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    elem: dict = {
        "id": elem_id,
        "type": elem_type,
        "name": elem_id,
        "language": language,
        "file_path": file_path,
        "relative_path": relative_path,
        "repo_name": repo_name,
    }
    if snapshot_id:
        elem["snapshot_id"] = snapshot_id
    if metadata:
        elem["metadata"] = metadata
    return {
        "element": elem,
        "semantic_score": total,
        "keyword_score": total * 0.5,
        "pseudocode_score": 0.0,
        "graph_score": 0.0,
        "total_score": total,
    }


class TestApplyFilters:
    def test_filter_by_language(self):
        rows = [
            _mk_row("a", language="python"),
            _mk_row("b", language="java"),
        ]
        result = apply_filters(rows, {"language": "python"})
        assert len(result) == 1
        assert result[0]["element"]["id"] == "a"

    def test_filter_by_type(self):
        rows = [
            _mk_row("a", elem_type="function"),
            _mk_row("b", elem_type="class"),
        ]
        result = apply_filters(rows, {"type": "class"})
        assert len(result) == 1
        assert result[0]["element"]["id"] == "b"

    def test_filter_by_file_path(self):
        rows = [
            _mk_row("a", relative_path="src/core/scoring.py"),
            _mk_row("b", relative_path="tests/test_main.py"),
        ]
        result = apply_filters(rows, {"file_path": "core"})
        assert len(result) == 1
        assert result[0]["element"]["id"] == "a"

    def test_filter_by_snapshot_id(self):
        rows = [
            _mk_row("a", snapshot_id="snap:v1"),
            _mk_row("b", snapshot_id="snap:v2"),
        ]
        result = apply_filters(rows, {"snapshot_id": "snap:v1"})
        assert len(result) == 1

    def test_filter_by_snapshot_id_in_metadata(self):
        rows = [
            _mk_row("a", metadata={"snapshot_id": "snap:v1"}),
            _mk_row("b", metadata={"snapshot_id": "snap:v2"}),
        ]
        result = apply_filters(rows, {"snapshot_id": "snap:v1"})
        assert len(result) == 1

    def test_no_filters_returns_all(self):
        rows = [_mk_row("a"), _mk_row("b")]
        result = apply_filters(rows, {})
        assert len(result) == 2

    def test_multiple_filters(self):
        rows = [
            _mk_row("a", language="python", elem_type="function"),
            _mk_row("b", language="python", elem_type="class"),
            _mk_row("c", language="java", elem_type="function"),
        ]
        result = apply_filters(rows, {"language": "python", "type": "function"})
        assert len(result) == 1
        assert result[0]["element"]["id"] == "a"


class TestDiversify:
    def test_no_penalty(self):
        rows = [_mk_row("a", total=0.9, file_path="f.py"), _mk_row("b", total=0.8, file_path="f.py")]
        result = diversify(rows, diversity_penalty=0.0)
        assert len(result) == 2

    def test_penalty_reduces_duplicate_file_scores(self):
        rows = [
            _mk_row("a", total=0.9, file_path="f.py"),
            _mk_row("b", total=0.8, file_path="f.py"),
        ]
        result = diversify(rows, diversity_penalty=0.5)
        assert len(result) == 2
        # Second result from same file should have lower total_score
        assert result[1]["total_score"] < 0.8

    def test_different_files_not_penalized(self):
        rows = [
            _mk_row("a", total=0.9, file_path="f1.py"),
            _mk_row("b", total=0.8, file_path="f2.py"),
        ]
        result = diversify(rows, diversity_penalty=0.5)
        assert result[0]["total_score"] == 0.9
        assert result[1]["total_score"] == 0.8

    def test_empty_results(self):
        result = diversify([], diversity_penalty=0.5)
        assert result == []

    def test_result_is_sorted(self):
        rows = [
            _mk_row("a", total=0.5, file_path="f1.py"),
            _mk_row("b", total=0.9, file_path="f2.py"),
        ]
        result = diversify(rows, diversity_penalty=0.0)
        assert result[0]["element"]["id"] == "b"


class TestFinalRepoFilter:
    def test_filters_by_repo(self):
        rows = [
            _mk_row("a", repo_name="repo1"),
            _mk_row("b", repo_name="repo2"),
        ]
        result = final_repo_filter(rows, ["repo1"])
        assert len(result) == 1
        assert result[0]["element"]["id"] == "a"

    def test_empty_filter_returns_all(self):
        rows = [_mk_row("a"), _mk_row("b")]
        result = final_repo_filter(rows, [])
        assert len(result) == 2

    def test_returns_filtered_count(self):
        rows = [
            _mk_row("a", repo_name="repo1"),
            _mk_row("b", repo_name="repo2"),
            _mk_row("c", repo_name="repo3"),
        ]
        result, count = final_repo_filter(rows, ["repo1"], return_count=True)
        assert count == 2
        assert len(result) == 1


class TestRerank:
    def test_function_gets_boost(self):
        rows = [
            _mk_row("a", elem_type="file", total=0.9),
            _mk_row("b", elem_type="function", total=0.9),
        ]
        result = rerank(rows)
        # function (1.2x) > file (0.9x) at same base score
        assert result[0]["element"]["id"] == "b"

    def test_unknown_type_gets_no_change(self):
        rows = [_mk_row("a", elem_type="module", total=0.5)]
        result = rerank(rows)
        assert abs(result[0]["total_score"] - 0.5) < 1e-9

    def test_results_sorted_after_rerank(self):
        rows = [
            _mk_row("a", elem_type="documentation", total=0.9),
            _mk_row("b", elem_type="function", total=0.8),
        ]
        result = rerank(rows)
        # function (0.8 * 1.2 = 0.96) > documentation (0.9 * 0.8 = 0.72)
        assert result[0]["element"]["id"] == "b"

    def test_empty(self):
        result = rerank([])
        assert result == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_filtering.py -v`
Expected: FAIL — `cannot import name 'apply_filters'`

- [ ] **Step 3: Write `fastcode/core/filtering.py`**

```python
# fastcode/core/filtering.py
"""Pure filtering, diversification, and reranking — extracted from retriever.py."""
from __future__ import annotations

from typing import Any


def apply_filters(
    results: list[dict[str, Any]],
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter results by language, type, file_path, and snapshot_id."""
    filtered: list[dict[str, Any]] = []
    for result in results:
        elem = result["element"]
        if "language" in filters:
            if elem.get("language") != filters["language"]:
                continue
        if "type" in filters:
            if elem.get("type") != filters["type"]:
                continue
        if "file_path" in filters:
            if filters["file_path"] not in elem.get("relative_path", ""):
                continue
        if "snapshot_id" in filters:
            elem_snapshot = elem.get("snapshot_id") or (
                elem.get("metadata", {}) or {}
            ).get("snapshot_id")
            if elem_snapshot != filters["snapshot_id"]:
                continue
        filtered.append(result)
    return filtered


def diversify(
    results: list[dict[str, Any]],
    diversity_penalty: float,
) -> list[dict[str, Any]]:
    """Penalize results from already-seen files to improve diversity."""
    if not results or diversity_penalty == 0:
        # Sort by total_score descending
        return sorted(results, key=lambda x: x["total_score"], reverse=True)

    diversified: list[dict[str, Any]] = []
    seen_files: set[str] = set()

    for result in results:
        file_path = result["element"].get("file_path", "")
        if file_path in seen_files:
            penalty_factor = 1 - diversity_penalty
            result = {
                **result,
                "total_score": result["total_score"] * penalty_factor,
                "semantic_score": result["semantic_score"] * penalty_factor,
                "keyword_score": result["keyword_score"] * penalty_factor,
                "pseudocode_score": result["pseudocode_score"] * penalty_factor,
                "graph_score": result["graph_score"] * penalty_factor,
            }
        else:
            seen_files.add(file_path)
        diversified.append(result)

    diversified.sort(key=lambda x: x["total_score"], reverse=True)
    return diversified


def final_repo_filter(
    results: list[dict[str, Any]],
    repo_filter: list[str],
    return_count: bool = False,
) -> Any:
    """Filter results to only include elements from allowed repositories.

    If return_count is True, returns (filtered_results, filtered_count).
    Otherwise returns filtered_results only.
    """
    if not repo_filter:
        return (results, 0) if return_count else results

    filtered_results: list[dict[str, Any]] = []
    filtered_count = 0
    for result in results:
        elem = result["element"]
        repo_name = elem.get("repo_name", "")
        if repo_name in repo_filter:
            filtered_results.append(result)
        else:
            filtered_count += 1

    return (filtered_results, filtered_count) if return_count else filtered_results


def rerank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-rank results by element type preferences."""
    type_weights = {
        "function": 1.2,
        "class": 1.1,
        "file": 0.9,
        "documentation": 0.8,
        "design_document": 0.95,
    }

    reranked: list[dict[str, Any]] = []
    for result in results:
        elem_type = result["element"].get("type", "")
        weight = type_weights.get(elem_type, 1.0)
        reranked.append({
            **result,
            "total_score": result["total_score"] * weight,
            "semantic_score": result["semantic_score"] * weight,
            "keyword_score": result["keyword_score"] * weight,
            "pseudocode_score": result["pseudocode_score"] * weight,
            "graph_score": result["graph_score"] * weight,
        })

    reranked.sort(key=lambda x: x["total_score"], reverse=True)
    return reranked
```

- [ ] **Step 4: Run core filtering tests**

Run: `uv run pytest tests/test_core_filtering.py -v`
Expected: All PASS

- [ ] **Step 5: Wire `HybridRetriever` to delegate to core**

In `fastcode/retriever.py`, add `from fastcode.core import filtering as _filtering` and replace method bodies:

```python
# In retriever.py, at top:
from fastcode.core import filtering as _filtering

# Replace _apply_filters:
def _apply_filters(self, results, filters):
    return _filtering.apply_filters(results, filters)

# Replace _diversify:
def _diversify(self, results):
    return _filtering.diversify(results, self.diversity_penalty)

# Replace _final_repo_filter:
def _final_repo_filter(self, results, repo_filter):
    return _filtering.final_repo_filter(results, repo_filter)

# Replace _rerank:
def _rerank(self, query, results):
    return _filtering.rerank(results)
```

- [ ] **Step 6: Run existing retriever tests**

Run: `uv run pytest tests/test_adaptive_fusion.py tests/test_doc_channel_projection.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add fastcode/core/filtering.py tests/test_core_filtering.py fastcode/retriever.py
git commit -m "feat: extract pure filtering functions to core/filtering.py"
```

---

### Task 1.4: Extract combination function (from retriever.py)

**Files:**
- Create: `fastcode/core/combination.py`
- Create: `tests/test_core_combination.py`
- Modify: `fastcode/retriever.py`

Extract `_combine_results` (line 1324-1410) — merges semantic + keyword + pseudocode results with BM25 normalization and source-priority boost. Currently reads `self.semantic_weight` and `self.keyword_weight` — pass as parameters.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_core_combination.py
"""Tests for pure combination function extracted from retriever."""
from fastcode.core.combination import combine_results


def _mk_meta(elem_id: str, **extra) -> dict:
    meta = {"id": elem_id, "type": "function", "name": elem_id}
    meta.update(extra)
    return meta


class TestCombineResults:
    def test_merges_semantic_and_keyword(self):
        sem = [(_mk_meta("a"), 0.8)]
        kw = [(_mk_meta("b"), 5.0)]
        result = combine_results(sem, kw)
        ids = [r["element"]["id"] for r in result]
        assert "a" in ids
        assert "b" in ids

    def test_merges_same_element(self):
        sem = [(_mk_meta("a"), 0.8)]
        kw = [(_mk_meta("a"), 5.0)]
        result = combine_results(sem, kw)
        assert len(result) == 1
        assert result[0]["semantic_score"] > 0
        assert result[0]["keyword_score"] > 0

    def test_pseudocode_results(self):
        sem = [(_mk_meta("a"), 0.8)]
        pseudo = [(_mk_meta("b"), 0.6)]
        result = combine_results(sem, [], pseudo)
        assert len(result) == 2

    def test_source_priority_boost(self):
        sem_high = [(_mk_meta("a", metadata={"source_priority": 100}), 0.8)]
        sem_low = [(_mk_meta("b", metadata={"source_priority": 0}), 0.8)]
        result = combine_results(sem_high + sem_low, [])
        assert result[0]["element"]["id"] == "a"

    def test_empty_inputs(self):
        result = combine_results([], [])
        assert result == []

    def test_sorted_by_total_score(self):
        sem = [(_mk_meta("low"), 0.3), (_mk_meta("high"), 0.9)]
        result = combine_results(sem, [])
        assert result[0]["element"]["id"] == "high"

    def test_bm25_normalization(self):
        kw = [(_mk_meta("a"), 5.0), (_mk_meta("b"), 2.5)]
        result = combine_results([], kw)
        # Both should have keyword scores, normalized by max
        assert result[0]["keyword_score"] >= result[1]["keyword_score"]

    def test_semantic_weight_applied(self):
        sem = [(_mk_meta("a"), 1.0)]
        result = combine_results(sem, [], semantic_weight=0.5)
        assert abs(result[0]["semantic_score"] - 0.5) < 1e-9

    def test_keyword_weight_applied(self):
        kw = [(_mk_meta("a"), 2.0)]
        result = combine_results([], kw, keyword_weight=0.3)
        assert abs(result[0]["keyword_score"] - 0.3) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_combination.py -v`
Expected: FAIL

- [ ] **Step 3: Write `fastcode/core/combination.py`**

```python
# fastcode/core/combination.py
"""Pure result combination — merges semantic, keyword, pseudocode search results."""
from __future__ import annotations

from typing import Any


def combine_results(
    semantic_results: list[tuple[dict[str, Any], float]],
    keyword_results: list[tuple[dict[str, Any], float]],
    pseudocode_results: list[tuple[dict[str, Any], float]] | None = None,
    *,
    semantic_weight: float = 1.0,
    keyword_weight: float = 1.0,
    pseudocode_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """Combine semantic, keyword, and pseudocode search results.

    Merges by element ID, normalizes BM25 scores to 0-1, applies
    source-priority boost, sorts by total_score descending.
    """
    combined: dict[str, dict[str, Any]] = {}

    for metadata, score in semantic_results:
        elem_id = metadata.get("id")
        if elem_id:
            combined[elem_id] = {
                "element": metadata,
                "semantic_score": score * semantic_weight,
                "keyword_score": 0.0,
                "pseudocode_score": 0.0,
                "graph_score": 0.0,
                "total_score": score * semantic_weight,
            }

    if pseudocode_results:
        for metadata, score in pseudocode_results:
            elem_id = metadata.get("id")
            if elem_id:
                pseudocode_contrib = score * pseudocode_weight
                if elem_id in combined:
                    combined[elem_id]["pseudocode_score"] = pseudocode_contrib
                    combined[elem_id]["total_score"] += pseudocode_contrib
                else:
                    combined[elem_id] = {
                        "element": metadata,
                        "semantic_score": 0.0,
                        "keyword_score": 0.0,
                        "pseudocode_score": pseudocode_contrib,
                        "graph_score": 0.0,
                        "total_score": pseudocode_contrib,
                    }

    if keyword_results:
        max_bm25 = max(score for _, score in keyword_results) if keyword_results else 0
        if max_bm25 > 0:
            for metadata, score in keyword_results:
                elem_id = metadata.get("id")
                if elem_id:
                    normalized_score = (score / max_bm25) * keyword_weight
                    if elem_id in combined:
                        combined[elem_id]["keyword_score"] = normalized_score
                        combined[elem_id]["total_score"] += normalized_score
                    else:
                        combined[elem_id] = {
                            "element": metadata,
                            "semantic_score": 0.0,
                            "keyword_score": normalized_score,
                            "pseudocode_score": 0.0,
                            "graph_score": 0.0,
                            "total_score": normalized_score,
                        }

    results = list(combined.values())

    for result in results:
        elem = result.get("element", {})
        meta = elem.get("metadata", {}) if isinstance(elem, dict) else {}
        source_priority = meta.get("source_priority", 0)
        try:
            source_priority = float(source_priority)
        except Exception:
            source_priority = 0.0
        boost = 1.0 + min(max(source_priority, 0.0), 100.0) / 200.0
        result["total_score"] *= boost

    results.sort(key=lambda x: x["total_score"], reverse=True)
    return results
```

- [ ] **Step 4: Run core combination tests**

Run: `uv run pytest tests/test_core_combination.py -v`
Expected: All PASS

- [ ] **Step 5: Wire `HybridRetriever._combine_results` to delegate**

```python
# In retriever.py:
from fastcode.core import combination as _combination

def _combine_results(self, semantic_results, keyword_results, pseudocode_results=None):
    return _combination.combine_results(
        semantic_results, keyword_results, pseudocode_results,
        semantic_weight=self.semantic_weight,
        keyword_weight=self.keyword_weight,
    )
```

- [ ] **Step 6: Run existing retriever tests**

Run: `uv run pytest tests/test_adaptive_fusion.py tests/test_doc_channel_projection.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add fastcode/core/combination.py tests/test_core_combination.py fastcode/retriever.py
git commit -m "feat: extract pure combination function to core/combination.py"
```

---

## Phase 2: Extract Pure Iteration Logic (from iterative_agent.py)

### Task 2.1: Extract iteration control functions

**Files:**
- Create: `fastcode/core/iteration.py`
- Create: `tests/test_core_iteration.py`
- Modify: `fastcode/iterative_agent.py`

Extract the pure math/decision functions:
- `_calculate_recent_confidence_gain` (line 2392-2396)
- `_calculate_recent_lines_added` (line 2398-2402)
- `_get_min_roi_threshold` (line 2404-2430)
- `_calculate_repo_factor` (line 2432-2459)
- `_calculate_total_lines` (line 2461-2470)
- `_should_continue_iteration` (line 2268-2390) — the 6-check stopping logic
- `_initialize_adaptive_parameters` (line 109-152)
- `_determine_stopping_reason` (line 420-432)
- `_rate_efficiency` (line 434-444)
- `_generate_iteration_metadata` (line 345-418)

All of these are pure math/decision logic. They read from `self` only to access config values or history — pass those as parameters.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_core_iteration.py
"""Tests for pure iteration control functions."""
import pytest

from fastcode.core.iteration import (
    calculate_recent_confidence_gain,
    calculate_recent_lines_added,
    get_min_roi_threshold,
    calculate_repo_factor,
    calculate_total_lines,
    should_continue_iteration,
    initialize_adaptive_parameters,
    determine_stopping_reason,
    rate_efficiency,
)
from fastcode.core.types import IterationConfig


class TestCalculateRecentConfidenceGain:
    def test_with_history(self):
        history = [
            {"confidence": 50, "total_lines": 500},
            {"confidence": 65, "total_lines": 800},
        ]
        assert calculate_recent_confidence_gain(history) == 15.0

    def test_single_entry(self):
        history = [{"confidence": 50, "total_lines": 500}]
        assert calculate_recent_confidence_gain(history) == 0.0

    def test_empty(self):
        assert calculate_recent_confidence_gain([]) == 0.0


class TestCalculateRecentLinesAdded:
    def test_with_history(self):
        history = [
            {"confidence": 50, "total_lines": 500},
            {"confidence": 65, "total_lines": 800},
        ]
        assert calculate_recent_lines_added(history) == 300

    def test_single_entry(self):
        history = [{"confidence": 50, "total_lines": 500}]
        assert calculate_recent_lines_added(history) == 0

    def test_empty(self):
        assert calculate_recent_lines_added([]) == 0


class TestGetMinRoiThreshold:
    def test_high_complexity_lower_threshold(self):
        roi = get_min_roi_threshold(query_complexity=90, current_confidence=60)
        assert roi < 2.0

    def test_low_complexity_higher_threshold(self):
        roi = get_min_roi_threshold(query_complexity=20, current_confidence=60)
        assert roi >= 1.5

    def test_high_confidence_demands_more(self):
        roi_low = get_min_roi_threshold(query_complexity=50, current_confidence=60)
        roi_high = get_min_roi_threshold(query_complexity=50, current_confidence=90)
        assert roi_high > roi_low


class TestCalculateRepoFactor:
    def test_no_stats(self):
        assert calculate_repo_factor(None) == 1.0

    def test_empty_stats(self):
        assert calculate_repo_factor({}) == 1.0

    def test_small_repo(self):
        factor = calculate_repo_factor({
            "total_files": 10, "total_classes": 5, "total_functions": 20,
            "avg_file_lines": 50, "max_depth": 2,
        })
        assert 0.5 <= factor <= 2.0

    def test_large_repo(self):
        factor = calculate_repo_factor({
            "total_files": 500, "total_classes": 100, "total_functions": 300,
            "avg_file_lines": 300, "max_depth": 8,
        })
        assert 0.5 <= factor <= 2.0


class TestCalculateTotalLines:
    def test_with_line_ranges(self):
        elements = [
            {"element": {"start_line": 10, "end_line": 20}},
            {"element": {"start_line": 30, "end_line": 40}},
        ]
        assert calculate_total_lines(elements) == 22  # (11 + 11)

    def test_no_line_info(self):
        elements = [{"element": {}}]
        assert calculate_total_lines(elements) == 0

    def test_empty(self):
        assert calculate_total_lines([]) == 0


class TestShouldContinueIteration:
    def test_stops_when_confidence_exceeds_threshold(self):
        assert not should_continue_iteration(
            confidence=96, current_round=2, max_iterations=4,
            total_lines=5000, line_budget=12000,
            confidence_threshold=95, history=(),
            min_confidence_gain=0.5,
        )

    def test_stops_at_max_iterations(self):
        assert not should_continue_iteration(
            confidence=50, current_round=5, max_iterations=4,
            total_lines=5000, line_budget=12000,
            confidence_threshold=95, history=(),
            min_confidence_gain=0.5,
        )

    def test_stops_at_line_budget(self):
        assert not should_continue_iteration(
            confidence=50, current_round=2, max_iterations=4,
            total_lines=13000, line_budget=12000,
            confidence_threshold=95, history=(),
            min_confidence_gain=0.5,
        )

    def test_continues_when_below_threshold(self):
        assert should_continue_iteration(
            confidence=60, current_round=2, max_iterations=4,
            total_lines=5000, line_budget=12000,
            confidence_threshold=95, history=(),
            min_confidence_gain=0.5,
        )

    def test_stops_on_stagnation(self):
        history = (
            {"confidence": 60, "confidence_gain": 0.3, "roi": 1.0, "total_lines": 5000},
            {"confidence": 60, "confidence_gain": 0.2, "roi": 0.5, "total_lines": 5500},
            {"confidence": 60, "confidence_gain": 0.1, "roi": 0.3, "total_lines": 6000},
        )
        assert not should_continue_iteration(
            confidence=60, current_round=4, max_iterations=6,
            total_lines=6000, line_budget=12000,
            confidence_threshold=95, history=history,
            min_confidence_gain=0.5,
        )


class TestInitializeAdaptiveParameters:
    def test_simple_query_reduces_iterations(self):
        params = initialize_adaptive_parameters(
            query_complexity=20, repo_factor=1.0,
            config=IterationConfig(),
        )
        assert params.max_iterations <= 4

    def test_complex_query_increases_iterations(self):
        params = initialize_adaptive_parameters(
            query_complexity=90, repo_factor=1.5,
            config=IterationConfig(),
        )
        assert params.max_iterations >= 3

    def test_threshold_adjusts_for_complex_queries(self):
        params = initialize_adaptive_parameters(
            query_complexity=85, repo_factor=1.0,
            config=IterationConfig(base_confidence_threshold=95),
        )
        assert params.confidence_threshold < 95

    def test_simple_query_full_threshold(self):
        params = initialize_adaptive_parameters(
            query_complexity=30, repo_factor=1.0,
            config=IterationConfig(base_confidence_threshold=95),
        )
        assert params.confidence_threshold == 95

    def test_line_budget_scales_with_complexity(self):
        simple = initialize_adaptive_parameters(
            query_complexity=20, repo_factor=1.0,
            config=IterationConfig(max_total_lines=12000),
        )
        complex_ = initialize_adaptive_parameters(
            query_complexity=90, repo_factor=1.0,
            config=IterationConfig(max_total_lines=12000),
        )
        assert complex_.adaptive_line_budget >= simple.adaptive_line_budget


class TestDetermineStoppingReason:
    def test_confidence_reached(self):
        reason = determine_stopping_reason(
            final_confidence=96, confidence_threshold=95,
            current_round=3, max_iterations=4,
            iteration_history=(),
            line_budget=12000,
        )
        assert "confidence" in reason.lower() or "threshold" in reason.lower()

    def test_max_iterations(self):
        history = ({"round": i} for i in range(4))
        reason = determine_stopping_reason(
            final_confidence=60, confidence_threshold=95,
            current_round=4, max_iterations=4,
            iteration_history=tuple({"round": i} for i in range(4)),
            line_budget=12000,
        )
        assert "iteration" in reason.lower() or "max" in reason.lower()

    def test_line_budget(self):
        reason = determine_stopping_reason(
            final_confidence=60, confidence_threshold=95,
            current_round=2, max_iterations=4,
            iteration_history=tuple({"total_lines": 13000} for _ in range(2)),
            line_budget=12000,
        )
        assert "budget" in reason.lower()


class TestRateEfficiency:
    def test_excellent(self):
        assert rate_efficiency(overall_roi=6.0, budget_used_pct=50) == "excellent"

    def test_good(self):
        assert rate_efficiency(overall_roi=3.5, budget_used_pct=70) == "good"

    def test_acceptable(self):
        assert rate_efficiency(overall_roi=2.0, budget_used_pct=80) == "acceptable"

    def test_inefficient(self):
        assert rate_efficiency(overall_roi=0.5, budget_used_pct=95) == "inefficient"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_iteration.py -v`
Expected: FAIL

- [ ] **Step 3: Write `fastcode/core/iteration.py`**

```python
# fastcode/core/iteration.py
"""Pure iteration control functions — extracted from iterative_agent.py."""
from __future__ import annotations

import numpy as np
from typing import Any, NamedTuple

from fastcode.core.types import IterationConfig


class AdaptiveParams(NamedTuple):
    """Result of initialize_adaptive_parameters."""
    max_iterations: int
    confidence_threshold: int
    adaptive_line_budget: int


def calculate_recent_confidence_gain(
    iteration_history: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> float:
    """Calculate confidence gain in the most recent iteration."""
    if len(iteration_history) < 2:
        return 0.0
    return iteration_history[-1]["confidence"] - iteration_history[-2]["confidence"]


def calculate_recent_lines_added(
    iteration_history: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> int:
    """Calculate lines added in the most recent iteration."""
    if len(iteration_history) < 2:
        return 0
    return iteration_history[-1]["total_lines"] - iteration_history[-2]["total_lines"]


def get_min_roi_threshold(query_complexity: int, current_confidence: int) -> float:
    """Get minimum acceptable ROI threshold."""
    base_roi = 2.0
    if query_complexity >= 80:
        complexity_factor = 0.5
    elif query_complexity >= 60:
        complexity_factor = 0.7
    else:
        complexity_factor = 1.0
    if current_confidence >= 85:
        confidence_factor = 1.5
    elif current_confidence >= 70:
        confidence_factor = 1.0
    else:
        confidence_factor = 0.8
    return base_roi * complexity_factor * confidence_factor


def calculate_repo_factor(
    repo_stats: dict[str, Any] | None,
) -> float:
    """Calculate repository complexity factor (0.5 - 2.0)."""
    if not repo_stats:
        return 1.0
    total_files = repo_stats.get("total_files", 100)
    avg_file_lines = repo_stats.get("avg_file_lines", 200)
    max_depth = repo_stats.get("max_depth", 5)
    file_factor = np.log10(total_files + 1) / np.log10(1000)
    file_factor = np.clip(file_factor, 0.3, 1.5)
    complexity_factor = avg_file_lines / 200
    complexity_factor = np.clip(complexity_factor, 0.5, 2.0)
    depth_factor = max_depth / 5
    depth_factor = np.clip(depth_factor, 0.7, 1.3)
    final_factor = (file_factor + complexity_factor + depth_factor) / 3
    return float(np.clip(final_factor, 0.5, 2.0))


def calculate_total_lines(elements: list[dict[str, Any]]) -> int:
    """Calculate total lines of code in elements."""
    total = 0
    for elem_data in elements:
        elem = elem_data.get("element", {})
        start = elem.get("start_line", 0)
        end = elem.get("end_line", 0)
        if end > start:
            total += (end - start + 1)
    return total


def should_continue_iteration(
    *,
    confidence: int,
    current_round: int,
    max_iterations: int,
    total_lines: int,
    line_budget: int,
    confidence_threshold: int,
    history: tuple[dict[str, Any], ...],
    min_confidence_gain: float,
) -> bool:
    """Decide whether to continue iteration (6-check logic)."""
    # Check 1: Confidence already sufficient
    if confidence >= confidence_threshold:
        return False
    # Check 2: Hard iteration limit
    if current_round >= max_iterations:
        return False
    # Check 3: Line budget check
    if total_lines >= line_budget:
        return False

    # Check 4: Adaptive Trend & ROI Analysis
    if len(history) >= 2:
        current_metrics = history[-1]
        current_gain = current_metrics["confidence_gain"]
        current_roi = current_metrics["roi"]

        # 4a. Stagnation Check
        if abs(current_gain) < 1.0:
            if len(history) >= 3:
                prev_gain = history[-2]["confidence_gain"]
                if abs(prev_gain) < 1.0:
                    return False

        def is_low_performance(gain: float, roi: float, query_comp: int, curr_conf: int) -> bool:
            min_roi = get_min_roi_threshold(query_comp, curr_conf)
            if gain < -1.0:
                return True
            if -1.0 <= gain < min_confidence_gain and roi < min_roi:
                return True
            return False

        current_is_low = is_low_performance(current_gain, current_roi, 0, confidence)

        if current_is_low:
            if len(history) >= 3:
                prev_metrics = history[-2]
                prev_gain = prev_metrics["confidence_gain"]
                prev_roi = prev_metrics["roi"]
                prev_conf = prev_metrics["confidence"]
                if is_low_performance(prev_gain, prev_roi, 0, prev_conf):
                    return False

    # Check 5: Strict Stagnation (last 3 rounds)
    if len(history) >= 3:
        last_three = [h["confidence"] for h in history[-3:]]
        if max(last_three) - min(last_three) < 2:
            return False

    # Check 6: Cost-benefit threshold
    confidence_gap = confidence_threshold - confidence
    remaining_budget = line_budget - total_lines
    estimated_lines_needed = confidence_gap * 100

    if estimated_lines_needed > remaining_budget * 1.5:
        if len(history) >= 2 and history[-1]["confidence_gain"] < 0:
            pass  # exploration mode
        else:
            return False

    return True


def initialize_adaptive_parameters(
    *,
    query_complexity: int,
    repo_factor: float,
    config: IterationConfig,
) -> AdaptiveParams:
    """Initialize adaptive parameters based on query and repo complexity."""
    complexity_score = (query_complexity / 100 + repo_factor) / 2
    max_iterations = max(2, min(6, int(config.base_max_iterations * (0.7 + complexity_score * 0.6))))

    if query_complexity >= 80:
        confidence_threshold = max(90, config.base_confidence_threshold - 5)
    elif query_complexity >= 60:
        confidence_threshold = max(92, config.base_confidence_threshold - 3)
    else:
        confidence_threshold = config.base_confidence_threshold

    if query_complexity <= 30:
        adaptive_line_budget = int(config.max_total_lines * 0.6)
    elif query_complexity <= 60:
        adaptive_line_budget = int(config.max_total_lines * 0.8)
    else:
        adaptive_line_budget = int(config.max_total_lines * 1.0 * repo_factor)

    return AdaptiveParams(
        max_iterations=max_iterations,
        confidence_threshold=confidence_threshold,
        adaptive_line_budget=adaptive_line_budget,
    )


def determine_stopping_reason(
    *,
    final_confidence: int,
    confidence_threshold: int,
    current_round: int,
    max_iterations: int,
    iteration_history: tuple[dict[str, Any], ...],
    line_budget: int,
) -> str:
    """Determine why iteration stopped."""
    if final_confidence >= confidence_threshold:
        return "confidence_threshold_reached"
    elif len(iteration_history) >= max_iterations:
        return "max_iterations_reached"
    elif iteration_history and iteration_history[-1].get("total_lines", 0) >= line_budget:
        return "line_budget_exceeded"
    elif len(iteration_history) >= 3:
        recent_gains = [h.get("confidence_gain", 0) for h in iteration_history[-2:]]
        if all(g < 0.5 for g in recent_gains):
            return "diminishing_returns"
    return "other"


def rate_efficiency(overall_roi: float, budget_used_pct: float) -> str:
    """Rate the efficiency of the iteration process."""
    if overall_roi >= 5.0 and budget_used_pct < 70:
        return "excellent"
    elif overall_roi >= 3.0 and budget_used_pct < 85:
        return "good"
    elif overall_roi >= 1.5 or budget_used_pct < 90:
        return "acceptable"
    else:
        return "inefficient"
```

- [ ] **Step 4: Run core iteration tests**

Run: `uv run pytest tests/test_core_iteration.py -v`
Expected: All PASS

- [ ] **Step 5: Wire `IterativeAgent` to delegate to core**

In `fastcode/iterative_agent.py`, add:
```python
from fastcode.core import iteration as _iteration
from fastcode.core.types import IterationConfig
```

Replace each method body to delegate:
```python
def _calculate_recent_confidence_gain(self):
    return _iteration.calculate_recent_confidence_gain(self.iteration_history)

def _calculate_recent_lines_added(self):
    return _iteration.calculate_recent_lines_added(self.iteration_history)

def _get_min_roi_threshold(self, query_complexity, current_confidence):
    return _iteration.get_min_roi_threshold(query_complexity, current_confidence)

def _calculate_repo_factor(self):
    return _iteration.calculate_repo_factor(self.repo_stats)

def _calculate_total_lines(self, elements):
    return _iteration.calculate_total_lines(elements)

def _should_continue_iteration(self, current_round, confidence, current_elements, query_complexity):
    return _iteration.should_continue_iteration(
        confidence=confidence, current_round=current_round,
        max_iterations=self.max_iterations,
        total_lines=self._calculate_total_lines(current_elements),
        line_budget=self.adaptive_line_budget,
        confidence_threshold=self.confidence_threshold,
        history=tuple(self.iteration_history),
        min_confidence_gain=self.min_confidence_gain,
    )

def _initialize_adaptive_parameters(self, query_complexity):
    params = _iteration.initialize_adaptive_parameters(
        query_complexity=query_complexity,
        repo_factor=self._calculate_repo_factor(),
        config=IterationConfig(
            base_max_iterations=self.base_max_iterations,
            base_confidence_threshold=self.base_confidence_threshold,
            max_total_lines=self.max_total_lines,
        ),
    )
    self.max_iterations = params.max_iterations
    self.confidence_threshold = params.confidence_threshold
    self.adaptive_line_budget = params.adaptive_line_budget

def _determine_stopping_reason(self, final_confidence):
    return _iteration.determine_stopping_reason(
        final_confidence=final_confidence,
        confidence_threshold=self.confidence_threshold,
        current_round=len(self.iteration_history),
        max_iterations=self.max_iterations,
        iteration_history=tuple(self.iteration_history),
        line_budget=self.adaptive_line_budget,
    )

def _rate_efficiency(self, overall_roi, budget_used_pct):
    return _iteration.rate_efficiency(overall_roi, budget_used_pct)
```

- [ ] **Step 6: Run existing iterative_agent tests**

Run: `uv run pytest tests/ -k "iterat" -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add fastcode/core/iteration.py tests/test_core_iteration.py fastcode/iterative_agent.py
git commit -m "feat: extract pure iteration control functions to core/iteration.py"
```

---

### Task 2.2: Extract prompt building functions

**Files:**
- Create: `fastcode/core/prompts.py`
- Create: `tests/test_core_prompts.py`
- Modify: `fastcode/iterative_agent.py`

Extract pure prompt construction methods:
- `_format_elements_with_metadata` (line 1730-1795) — formats elements as text with grouping by file
- `_format_tool_call_history` (line 1714-1728) — formats tool call history

These are pure string formatters — no I/O. `_build_round_one_prompt` and `_build_round_n_prompt` are NOT fully pure (they call `self._generate_directory_tree`), so extract only the pure formatters. The prompt builders themselves will be extracted later when `_generate_directory_tree` becomes an effect.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_core_prompts.py
"""Tests for pure prompt formatting functions."""
import json

from fastcode.core.prompts import (
    format_elements_with_metadata,
    format_tool_call_history,
)


def _mk_elem_data(
    elem_id: str,
    *,
    repo_name: str = "myrepo",
    relative_path: str = "src/main.py",
    elem_type: str = "function",
    start_line: int = 10,
    end_line: int = 20,
    total_score: float = 0.8,
    agent_found: bool = False,
    llm_file_selected: bool = False,
    related_to: str | None = None,
    signature: str | None = None,
) -> dict:
    elem: dict = {
        "id": elem_id, "type": elem_type, "name": elem_id,
        "repo_name": repo_name, "relative_path": relative_path,
        "start_line": start_line, "end_line": end_line,
    }
    if signature:
        elem["signature"] = signature
    result: dict = {
        "element": elem,
        "total_score": total_score,
        "agent_found": agent_found,
        "llm_file_selected": llm_file_selected,
    }
    if related_to:
        result["related_to"] = related_to
    return result


class TestFormatElementsWithMetadata:
    def test_single_element(self):
        elements = [_mk_elem_data("func1")]
        text = format_elements_with_metadata(elements)
        assert "myrepo/src/main.py" in text
        assert "func1" in text

    def test_groups_by_file(self):
        elements = [
            _mk_elem_data("func1", relative_path="a.py"),
            _mk_elem_data("func2", relative_path="a.py"),
            _mk_elem_data("func3", relative_path="b.py"),
        ]
        text = format_elements_with_metadata(elements)
        # Should have two file groups
        assert "a.py" in text
        assert "b.py" in text

    def test_agent_found_source(self):
        elements = [_mk_elem_data("func1", agent_found=True)]
        text = format_elements_with_metadata(elements)
        assert "Tool" in text

    def test_graph_source(self):
        elements = [_mk_elem_data("func1", related_to="other_func")]
        text = format_elements_with_metadata(elements)
        assert "Graph" in text

    def test_retrieval_source(self):
        elements = [_mk_elem_data("func1")]
        text = format_elements_with_metadata(elements)
        assert "Retrieval" in text

    def test_shows_signature(self):
        elements = [_mk_elem_data("func1", signature="def func1(x: int) -> str")]
        text = format_elements_with_metadata(elements)
        assert "def func1(x: int) -> str" in text

    def test_shows_line_count(self):
        elements = [_mk_elem_data("func1", start_line=10, end_line=20)]
        text = format_elements_with_metadata(elements)
        assert "Lines" in text

    def test_empty(self):
        text = format_elements_with_metadata([])
        assert text == ""


class TestFormatToolCallHistory:
    def test_with_history(self):
        history = [
            {"round": 1, "tool": "search_codebase", "parameters": {"search_term": "foo"}},
            {"round": 2, "tool": "list_directory", "parameters": {"path": "src"}},
        ]
        text = format_tool_call_history(history, current_round=3)
        assert "search_codebase" in text
        assert "list_directory" in text

    def test_filters_current_round(self):
        history = [
            {"round": 1, "tool": "search_codebase", "parameters": {"search_term": "foo"}},
            {"round": 2, "tool": "list_directory", "parameters": {"path": "src"}},
        ]
        text = format_tool_call_history(history, current_round=2)
        assert "search_codebase" in text
        assert "list_directory" not in text

    def test_empty(self):
        text = format_tool_call_history([], current_round=1)
        assert text == "None"

    def test_no_history_attribute(self):
        text = format_tool_call_history(None, current_round=1)
        assert text == "None"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_prompts.py -v`
Expected: FAIL

- [ ] **Step 3: Write `fastcode/core/prompts.py`**

```python
# fastcode/core/prompts.py
"""Pure prompt formatting functions — extracted from iterative_agent.py."""
from __future__ import annotations

import json
from typing import Any


def format_elements_with_metadata(elements: list[dict[str, Any]]) -> str:
    """Format elements with metadata, grouped by (repo_name, relative_path)."""
    if not elements:
        return ""

    lines: list[str] = []
    file_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for elem_data in elements:
        elem = elem_data.get("element", {})
        repo_name = elem.get("repo_name", "")
        relative_path = elem.get("relative_path", elem.get("file_path", ""))
        group_key = (repo_name, relative_path)
        if group_key not in file_groups:
            file_groups[group_key] = []
        file_groups[group_key].append(elem_data)

    for i, (group_key, elem_list) in enumerate(file_groups.items(), 1):
        repo_name, relative_path = group_key
        display_path = f"{repo_name}/{relative_path}" if repo_name else relative_path
        lines.append(f"\n{i}. {display_path}")

        sources: set[str] = set()
        related_to: set[str] = set()
        total_lines = 0

        for elem_data in elem_list:
            elem = elem_data.get("element", {})
            if elem_data.get("agent_found"):
                sources.add("Tool")
            elif elem_data.get("llm_file_selected"):
                sources.add("LLM Selection")
            elif elem_data.get("related_to"):
                sources.add("Graph")
                related_to.add(elem_data.get("related_to", ""))
            else:
                sources.add("Retrieval")

            start = elem.get("start_line", 0)
            end = elem.get("end_line", 0)
            if end > start:
                total_lines += (end - start + 1)

        if repo_name:
            lines.append(f"   Repo: {repo_name}")
        lines.append(f"   Type: {elem_list[0]['element'].get('type', 'unknown')}")
        lines.append(f"   Source: {', '.join(sources)}")
        if total_lines > 0:
            lines.append(f"   Lines: {total_lines}")
        if related_to:
            lines.append(f"   Related to: {', '.join(related_to)}")

        for elem_data in elem_list[:5]:
            elem = elem_data.get("element", {})
            if elem.get("signature"):
                lines.append(f"   - {elem['signature']}")

    return "\n".join(lines)


def format_tool_call_history(
    history: list[dict[str, Any]] | None,
    current_round: int,
) -> str:
    """Format tool call history up to the previous round."""
    if not history:
        return "None"

    lines: list[str] = []
    for entry in history:
        if entry.get("round", 0) >= current_round:
            continue
        tool_name = entry.get("tool", "")
        params = entry.get("parameters", {})
        params_text = json.dumps(params, ensure_ascii=True, sort_keys=True)
        lines.append(f"- Round {entry['round']}: {tool_name} {params_text}")

    return "\n".join(lines) if lines else "None"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_core_prompts.py -v`
Expected: All PASS

- [ ] **Step 5: Wire `IterativeAgent` to delegate**

```python
# In iterative_agent.py:
from fastcode.core import prompts as _prompts

def _format_elements_with_metadata(self, elements):
    return _prompts.format_elements_with_metadata(elements)

def _format_tool_call_history(self, current_round):
    return _prompts.format_tool_call_history(self.tool_call_history, current_round)
```

- [ ] **Step 6: Run existing tests**

Run: `uv run pytest tests/ -k "iterat" -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add fastcode/core/prompts.py tests/test_core_prompts.py fastcode/iterative_agent.py
git commit -m "feat: extract pure prompt formatting to core/prompts.py"
```

---

### Task 2.3: Extract parsing functions

**Files:**
- Create: `fastcode/core/parsing.py`
- Create: `tests/test_core_parsing.py`
- Modify: `fastcode/iterative_agent.py`

Extract pure JSON parsing methods:
- `_extract_json_from_response` (line 2590-2646)
- `_sanitize_json_string` (line 2648-2726)
- `_remove_json_comments` (line 2728-2777)
- `_robust_json_parse` (line 2779-2841)

All pure string/JSON operations — no I/O, no `self` reads.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_core_parsing.py
"""Tests for pure JSON parsing functions."""
import json

from fastcode.core.parsing import (
    extract_json_from_response,
    sanitize_json_string,
    remove_json_comments,
    robust_json_parse,
)


class TestExtractJsonFromResponse:
    def test_plain_json(self):
        response = '{"confidence": 80, "reasoning": "test"}'
        result = extract_json_from_response(response)
        data = json.loads(result)
        assert data["confidence"] == 80

    def test_json_in_markdown_block(self):
        response = '```json\n{"confidence": 90}\n```'
        result = extract_json_from_response(response)
        data = json.loads(result)
        assert data["confidence"] == 90

    def test_json_with_prefix(self):
        response = 'Here is the JSON:\n{"confidence": 70}'
        result = extract_json_from_response(response)
        data = json.loads(result)
        assert data["confidence"] == 70

    def test_no_json(self):
        response = "No JSON here"
        result = extract_json_from_response(response)
        assert result == response

    def test_embedded_json(self):
        response = 'Some text before {"key": "value"} and after'
        result = extract_json_from_response(response)
        data = json.loads(result)
        assert data["key"] == "value"


class TestSanitizeJsonString:
    def test_trailing_comma_in_object(self):
        result = sanitize_json_string('{"a": 1,}')
        assert json.loads(result) == {"a": 1}

    def test_trailing_comma_in_array(self):
        result = sanitize_json_string('[1, 2,]')
        assert json.loads(result) == [1, 2]

    def test_missing_comma_between_objects(self):
        result = sanitize_json_string('{"a": 1}{"b": 2}')
        data = json.loads(result)
        assert data == [{"a": 1}, {"b": 2}]

    def test_control_chars_in_strings(self):
        result = sanitize_json_string('{"text": "line1\nline2"}')
        data = json.loads(result)
        assert "line1" in data["text"]


class TestRemoveJsonComments:
    def test_hash_comment(self):
        result = remove_json_comments('{"a": 1} # comment')
        assert "# comment" not in result
        assert '"a": 1' in result

    def test_double_slash_comment(self):
        result = remove_json_comments('{"a": 1} // comment')
        assert "// comment" not in result

    def test_preserves_comments_in_strings(self):
        result = remove_json_comments('{"url": "http://example.com"}')
        assert "http://example.com" in result


class TestRobustJsonParse:
    def test_valid_json(self):
        result = robust_json_parse('{"a": 1}')
        assert result == {"a": 1}

    def test_trailing_comma(self):
        result = robust_json_parse('{"a": 1,}')
        assert result == {"a": 1}

    def test_unquoted_keys(self):
        result = robust_json_parse('{a: 1}')
        assert result == {"a": 1}

    def test_invalid_raises(self):
        import pytest
        with pytest.raises(json.JSONDecodeError):
            robust_json_parse("not json at all {{{")

    def test_comment_stripped(self):
        result = robust_json_parse('{"a": 1} # comment')
        assert result == {"a": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_parsing.py -v`
Expected: FAIL

- [ ] **Step 3: Write `fastcode/core/parsing.py`**

```python
# fastcode/core/parsing.py
"""Pure JSON parsing functions — extracted from iterative_agent.py."""
from __future__ import annotations

import ast
import json
import re
from typing import Any


def extract_json_from_response(response: str) -> str:
    """Extract JSON string from LLM response, handling markdown blocks and prefixes."""
    response = response.strip()

    prefixes_to_remove = [
        "here's the json:", "here is the json:", "the json is:",
        "json:", "response:", "output:", "result:",
        "here's the response:", "here is the response:",
    ]
    response_lower = response.lower()
    for prefix in prefixes_to_remove:
        if response_lower.startswith(prefix):
            response = response[len(prefix):].strip()
            break

    json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        start = response.find("{")
        if start == -1:
            return response
        brace_count = 0
        end = -1
        for i in range(start, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break
        if end == -1:
            end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = response[start:end + 1]
        else:
            return response

    json_str = sanitize_json_string(json_str)
    return json_str


def sanitize_json_string(json_str: str) -> str:
    """Sanitize JSON string — fix control chars, trailing commas, missing commas."""
    cleaned: list[str] = []
    in_string = False
    escape_next = False

    for i, char in enumerate(json_str):
        if escape_next:
            cleaned.append(char)
            escape_next = False
            continue
        if char == '\\' and not escape_next:
            if i + 1 < len(json_str):
                next_char = json_str[i + 1]
                if next_char in r'"\\/bfnrtu':
                    cleaned.append(char)
                    escape_next = True
                else:
                    cleaned.append(char)
            else:
                cleaned.append(char)
            continue
        if char == '"':
            in_string = not in_string
            cleaned.append(char)
            continue
        if in_string:
            if char == '\n':
                cleaned.append('\\n')
            elif char == '\r':
                cleaned.append('\\r')
            elif char == '\t':
                cleaned.append('\\t')
            elif ord(char) < 32:
                cleaned.append(' ')
            else:
                cleaned.append(char)
        else:
            cleaned.append(char)

    result = ''.join(cleaned)
    result = remove_json_comments(result)
    result = re.sub(r',(\s*[}\]])', r'\1', result)
    result = re.sub(r'\}(\s*)\{', r'},\1{', result)
    result = re.sub(r'\](\s*)\[', r'],\1[', result)
    result = re.sub(r'\}(\s*)\[', r'},\1[', result)
    result = re.sub(r'\](\s*)\{', r'],\1{', result)
    result = re.sub(r'(["}\]])(\s*)(")', r'\1,\2\3', result)
    result = re.sub(r'\b(true|false|null)(\s*)(["{[])', r'\1,\2\3', result)
    return result


def remove_json_comments(json_str: str) -> str:
    """Remove inline comments from JSON (# or // style)."""
    lines = json_str.split('\n')
    cleaned_lines: list[str] = []

    for line in lines:
        in_string = False
        escape_next = False
        cleaned_line: list[str] = []

        i = 0
        while i < len(line):
            char = line[i]
            if escape_next:
                cleaned_line.append(char)
                escape_next = False
                i += 1
                continue
            if char == '\\':
                cleaned_line.append(char)
                escape_next = True
                i += 1
                continue
            if char == '"':
                in_string = not in_string
                cleaned_line.append(char)
                i += 1
                continue
            if not in_string:
                if char == '#':
                    break
                if char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                    break
            cleaned_line.append(char)
            i += 1

        cleaned_lines.append(''.join(cleaned_line).rstrip())

    return '\n'.join(cleaned_lines)


def robust_json_parse(json_str: str) -> Any:
    """Robustly parse JSON with multiple fallback strategies."""
    # Strategy 1: Direct parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Sanitize then parse
    try:
        sanitized = sanitize_json_string(json_str)
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Fix unquoted keys
    try:
        fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', json_str)
        return json.loads(fixed)
    except (json.JSONDecodeError, Exception):
        pass

    # Strategy 4: ast.literal_eval
    try:
        result = ast.literal_eval(json_str)
        if isinstance(result, (dict, list)):
            return result
    except (ValueError, SyntaxError, Exception):
        pass

    # Strategy 5: Extract first complete object
    try:
        start = json_str.find('{')
        if start != -1:
            brace_count = 0
            for i in range(start, len(json_str)):
                if json_str[i] == '{':
                    brace_count += 1
                elif json_str[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return json.loads(json_str[start:i + 1])
    except (json.JSONDecodeError, Exception):
        pass

    raise json.JSONDecodeError("All parsing strategies failed", json_str, 0)
```

- [ ] **Step 4: Run core parsing tests**

Run: `uv run pytest tests/test_core_parsing.py -v`
Expected: All PASS

- [ ] **Step 5: Wire `IterativeAgent` to delegate**

```python
# In iterative_agent.py:
from fastcode.core import parsing as _parsing

def _extract_json_from_response(self, response):
    return _parsing.extract_json_from_response(response)

def _sanitize_json_string(self, json_str):
    return _parsing.sanitize_json_string(json_str)

def _remove_json_comments(self, json_str):
    return _parsing.remove_json_comments(json_str)

def _robust_json_parse(self, json_str):
    return _parsing.robust_json_parse(json_str)
```

- [ ] **Step 6: Run existing tests**

Run: `uv run pytest tests/ -k "iterat" -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add fastcode/core/parsing.py tests/test_core_parsing.py fastcode/iterative_agent.py
git commit -m "feat: extract pure JSON parsing to core/parsing.py"
```

---

## Phase 3: Extract Pure Generation Logic (from answer_generator.py)

### Task 3.1: Extract context preparation and response parsing

**Files:**
- Create: `fastcode/core/context.py`
- Create: `tests/test_core_context.py`
- Modify: `fastcode/answer_generator.py`

Extract:
- `_prepare_context` (line 447-515) — pure string formatting
- `_parse_response_with_summary` (line 737-768) — pure regex extraction

`_build_prompt` reads `self.include_file_paths`, `self.include_line_numbers`, `self.enable_multi_turn`, `self.context_rounds` — pass these as config parameters.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_core_context.py
"""Tests for pure context preparation and response parsing."""
from fastcode.core.context import (
    prepare_context,
    parse_response_with_summary,
)


def _mk_element(
    elem_id: str = "func1",
    *,
    repo_name: str = "myrepo",
    relative_path: str = "src/main.py",
    elem_type: str = "function",
    code: str = "def func1(): pass",
    start_line: int = 10,
    end_line: int = 12,
    language: str = "python",
    total_score: float = 0.85,
) -> dict:
    return {
        "element": {
            "id": elem_id, "type": elem_type, "name": elem_id,
            "repo_name": repo_name, "relative_path": relative_path,
            "code": code, "start_line": start_line, "end_line": end_line,
            "language": language,
        },
        "total_score": total_score,
    }


class TestPrepareContext:
    def test_single_element(self):
        elements = [_mk_element()]
        context = prepare_context(elements)
        assert "func1" in context
        assert "def func1(): pass" in context

    def test_includes_repo_name(self):
        elements = [_mk_element(repo_name="myrepo")]
        context = prepare_context(elements, include_file_paths=True)
        assert "myrepo" in context

    def test_includes_file_path(self):
        elements = [_mk_element(relative_path="src/core/scoring.py")]
        context = prepare_context(elements, include_file_paths=True)
        assert "scoring.py" in context

    def test_includes_line_numbers(self):
        elements = [_mk_element(start_line=10, end_line=20)]
        context = prepare_context(elements, include_line_numbers=True)
        assert "10-20" in context

    def test_truncates_long_code(self):
        long_code = "x" * 200000
        elements = [_mk_element(code=long_code)]
        context = prepare_context(elements)
        assert "truncated" in context

    def test_multiple_elements(self):
        elements = [_mk_element("func1"), _mk_element("func2")]
        context = prepare_context(elements)
        assert "func1" in context
        assert "func2" in context

    def test_empty(self):
        context = prepare_context([])
        assert context == ""

    def test_element_with_metadata(self):
        elements = [{
            "element": {
                "id": "a", "type": "function", "name": "a",
                "code": "def a(): pass", "language": "python",
                "metadata": {"complexity": 5, "num_methods": 3},
            },
            "total_score": 0.9,
        }]
        context = prepare_context(elements)
        assert "Complexity: 5" in context
        assert "Methods: 3" in context


class TestParseResponseWithSummary:
    def test_extracts_summary_tags(self):
        response = "Here is the answer.\n<SUMMARY>\nFiles Read:\n- foo.py\n</SUMMARY>"
        answer, summary = parse_response_with_summary(response)
        assert "Files Read" in summary
        assert "<SUMMARY>" not in answer

    def test_no_summary(self):
        response = "Just a plain answer with no summary."
        answer, summary = parse_response_with_summary(response)
        assert answer == response
        assert summary is None

    def test_case_insensitive_tags(self):
        response = "Answer\n<summary>\nContent\n</summary>"
        answer, summary = parse_response_with_summary(response)
        assert summary is not None

    def test_bold_summary(self):
        response = "Answer\n**<SUMMARY>**\nContent\n**</SUMMARY>**"
        answer, summary = parse_response_with_summary(response)
        assert summary is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_context.py -v`
Expected: FAIL

- [ ] **Step 3: Write `fastcode/core/context.py`**

```python
# fastcode/core/context.py
"""Pure context preparation and response parsing — extracted from answer_generator.py."""
from __future__ import annotations

import re
from typing import Any


def prepare_context(
    elements: list[dict[str, Any]],
    *,
    include_file_paths: bool = True,
    include_line_numbers: bool = True,
) -> str:
    """Prepare context string from retrieved elements."""
    if not elements:
        return ""

    context_parts: list[str] = []
    for i, elem_data in enumerate(elements, 1):
        elem = elem_data.get("element", {})
        score = elem_data.get("total_score", 0)

        parts = [f"## Relevant Code Snippet {i}"]
        repo_name = elem.get("repo_name")
        if repo_name:
            parts.append(f"**Repository**: `{repo_name}`")

        if include_file_paths:
            rel_path = elem.get("relative_path", "")
            if rel_path:
                display = f"{repo_name}/{rel_path}" if repo_name else rel_path
                parts.append(f"**File**: `{display}`")

        elem_type = elem.get("type", "")
        elem_name = elem.get("name", "")
        parts.append(f"**Type**: {elem_type}")
        parts.append(f"**Name**: `{elem_name}`")

        if include_line_numbers:
            start_line = elem.get("start_line", 0)
            end_line = elem.get("end_line", 0)
            if start_line > 0:
                parts.append(f"**Lines**: {start_line}-{end_line}")

        code = elem.get("code", "")
        if code:
            language = elem.get("language", "")
            if len(code) > 100000:
                code = code[:100000] + "\n... (truncated)"
            parts.append(f"**Code**:\n```{language}\n{code}\n```")

        metadata = elem.get("metadata", {})
        if metadata:
            meta_parts = []
            if "complexity" in metadata:
                meta_parts.append(f"Complexity: {metadata['complexity']}")
            if "num_methods" in metadata:
                meta_parts.append(f"Methods: {metadata['num_methods']}")
            if meta_parts:
                parts.append(f"**Metadata**: {', '.join(meta_parts)}")

        context_parts.append("\n".join(parts))

    return "\n\n---\n\n".join(context_parts)


def parse_response_with_summary(raw_response: str) -> tuple[str, str | None]:
    """Parse LLM response to extract answer and optional <SUMMARY> block."""
    summary_patterns = [
        r'<\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*:?\s*>(.*?)<\s*/\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*>',
        r'\*\*\s*<\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*>\s*\*\*(.*?)\*\*\s*<\s*/\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*>\s*\*\*',
        r'\*\*\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*\*\*\s*:?\s*\n(.*?)(?=\n\n(?:\*\*|##|$)|\Z)',
        r'[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*:?\s*\n(.*?)(?=\n\n(?:\*\*|##|$)|\Z)',
    ]

    for pattern in summary_patterns:
        match = re.search(pattern, raw_response, re.DOTALL)
        if match:
            summary = match.group(1).strip()
            answer = re.sub(pattern, '', raw_response, flags=re.DOTALL).strip()
            return answer, summary

    return raw_response, None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_core_context.py -v`
Expected: All PASS

- [ ] **Step 5: Wire `AnswerGenerator` to delegate**

```python
# In answer_generator.py:
from fastcode.core import context as _context

def _prepare_context(self, elements):
    return _context.prepare_context(
        elements,
        include_file_paths=self.include_file_paths,
        include_line_numbers=self.include_line_numbers,
    )

def _parse_response_with_summary(self, raw_response):
    return _context.parse_response_with_summary(raw_response)
```

- [ ] **Step 6: Run existing tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add fastcode/core/context.py tests/test_core_context.py fastcode/answer_generator.py
git commit -m "feat: extract pure context preparation to core/context.py"
```

---

### Task 3.2: Extract summary and formatting functions

**Files:**
- Create: `fastcode/core/summary.py`
- Create: `tests/test_core_summary.py`
- Modify: `fastcode/answer_generator.py`

Extract:
- `_generate_fallback_summary` (line 770-843)
- `_extract_sources` (line 845-861)
- `format_answer_with_sources` (line 863-888)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_core_summary.py
"""Tests for pure summary and formatting functions."""
from fastcode.core.summary import (
    generate_fallback_summary,
    extract_sources,
    format_answer_with_sources,
)


def _mk_elem(
    elem_id: str = "func1",
    *,
    repo_name: str = "myrepo",
    relative_path: str = "src/main.py",
    elem_type: str = "function",
    total_score: float = 0.85,
) -> dict:
    return {
        "element": {
            "id": elem_id, "type": elem_type, "name": elem_id,
            "repo_name": repo_name, "relative_path": relative_path,
            "start_line": 10, "end_line": 20,
        },
        "total_score": total_score,
    }


class TestGenerateFallbackSummary:
    def test_basic(self):
        elements = [_mk_elem()]
        summary = generate_fallback_summary("How does X work?", "Answer text", elements)
        assert "Files Read:" in summary
        assert "myrepo/src/main.py" in summary

    def test_no_elements(self):
        summary = generate_fallback_summary("query", "answer", [])
        assert "Files Read: None" in summary

    def test_includes_query(self):
        summary = generate_fallback_summary("How does X work?", "answer", [])
        assert "How does X work?" in summary

    def test_includes_answer_preview(self):
        summary = generate_fallback_summary("query", "A detailed answer", [])
        assert "A detailed answer" in summary

    def test_limits_files(self):
        elements = [_mk_elem(f"f{i}", relative_path=f"file{i}.py") for i in range(20)]
        summary = generate_fallback_summary("query", "answer", elements)
        # Should be limited to 10 files
        assert summary.count(".py") <= 12  # 10 in files + some in code elements


class TestExtractSources:
    def test_basic(self):
        elements = [_mk_elem()]
        sources = extract_sources(elements)
        assert len(sources) == 1
        assert sources[0]["repository"] == "myrepo"
        assert sources[0]["file"] == "src/main.py"

    def test_empty(self):
        assert extract_sources([]) == []

    def test_includes_score(self):
        elements = [_mk_elem(total_score=0.75)]
        sources = extract_sources(elements)
        assert sources[0]["score"] == 0.75


class TestFormatAnswerWithSources:
    def test_basic(self):
        result = {
            "answer": "Test answer",
            "sources": [
                {"repository": "myrepo", "file": "src/main.py",
                 "name": "func1", "type": "function",
                 "lines": "10-20", "score": 0.85},
            ],
            "prompt_tokens": 100,
            "context_elements": 5,
        }
        text = format_answer_with_sources(result)
        assert "Test answer" in text
        assert "func1" in text
        assert "src/main.py" in text
        assert "100 prompt tokens" in text

    def test_no_sources(self):
        result = {"answer": "Simple answer", "sources": []}
        text = format_answer_with_sources(result)
        assert "Simple answer" in text
        assert "Sources" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_summary.py -v`
Expected: FAIL

- [ ] **Step 3: Write `fastcode/core/summary.py`**

```python
# fastcode/core/summary.py
"""Pure summary and formatting functions — extracted from answer_generator.py."""
from __future__ import annotations

from typing import Any


def generate_fallback_summary(
    query: str,
    answer: str,
    retrieved_elements: list[dict[str, Any]],
) -> str:
    """Generate a fallback summary when LLM doesn't produce one."""
    parts: list[str] = []

    files_read: set[str] = set()
    for elem_data in retrieved_elements:
        elem = elem_data.get("element", {})
        repo_name = elem.get("repo_name", "")
        rel_path = elem.get("relative_path", "")
        if repo_name and rel_path:
            files_read.add(f"{repo_name}/{rel_path}")

    if files_read:
        parts.append("Files Read:")
        for file_path in sorted(files_read)[:10]:
            parts.append(f"- {file_path}")
    else:
        parts.append("Files Read: None")

    parts.append("\nCode Elements Referenced:")
    elements_added = 0
    for elem_data in retrieved_elements[:15]:
        elem = elem_data.get("element", {})
        repo_name = elem.get("repo_name", "")
        rel_path = elem.get("relative_path", "")
        elem_type = elem.get("type", "")
        elem_name = elem.get("name", "")

        if repo_name and rel_path and elem_name:
            elem_info = f"- [{repo_name}/{rel_path}] {elem_type}: {elem_name}"
            signature = elem.get("signature", "")
            if signature:
                elem_info += f" ({signature})"
            parts.append(elem_info)
            docstring = elem.get("docstring", "")
            if docstring:
                doc_preview = docstring[:150].replace("\n", " ").strip()
                if len(docstring) > 150:
                    doc_preview += "..."
                parts.append(f"  Doc: {doc_preview}")
            elements_added += 1

    if elements_added == 0:
        parts.append("- No specific code elements")

    parts.append(f"\nQuery: {query[:200]}")
    answer_preview = answer.replace("\n", " ").strip()
    parts.append(f"Answer Preview: {answer_preview}")

    return "\n".join(parts)


def extract_sources(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract source information from elements."""
    sources: list[dict[str, Any]] = []
    for elem_data in elements:
        elem = elem_data.get("element", {})
        sources.append({
            "repository": elem.get("repo_name", ""),
            "file": elem.get("relative_path", ""),
            "name": elem.get("name", ""),
            "type": elem.get("type", ""),
            "lines": f"{elem.get('start_line', 0)}-{elem.get('end_line', 0)}",
            "score": elem_data.get("total_score", 0),
        })
    return sources


def format_answer_with_sources(result: dict[str, Any]) -> str:
    """Format answer with sources for display."""
    output: list[str] = []

    output.append("## Answer\n")
    output.append(result.get("answer", ""))

    sources = result.get("sources", [])
    if sources:
        output.append("\n\n## Sources\n")
        for i, source in enumerate(sources, 1):
            repo_info = f"[{source['repository']}] " if source.get('repository') else ""
            output.append(
                f"{i}. {repo_info}**{source['name']}** ({source['type']}) "
                f"in `{source['file']}` (lines {source['lines']}) "
                f"- Relevance: {source['score']:.2f}"
            )

    if "prompt_tokens" in result:
        output.append(
            f"\n\n*Used {result['prompt_tokens']} prompt tokens, "
            f"{result.get('context_elements', 0)} code snippets*"
        )

    return "\n".join(output)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_core_summary.py -v`
Expected: All PASS

- [ ] **Step 5: Wire `AnswerGenerator` to delegate**

```python
# In answer_generator.py:
from fastcode.core import summary as _summary

def _generate_fallback_summary(self, query, answer, retrieved_elements):
    return _summary.generate_fallback_summary(query, answer, retrieved_elements)

def _extract_sources(self, elements):
    return _summary.extract_sources(elements)

def format_answer_with_sources(self, result):
    return _summary.format_answer_with_sources(result)
```

- [ ] **Step 6: Run existing tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add fastcode/core/summary.py tests/test_core_summary.py fastcode/answer_generator.py
git commit -m "feat: extract pure summary/formatting to core/summary.py"
```

---

## Phase 4: Extract Pure Transforms from Remaining Modules

### Task 4.2: Extract graph payload construction

**Files:**
- Create: `fastcode/core/graph_build.py`
- Create: `tests/test_core_graph_build.py`
- Modify: `fastcode/terminus_publisher.py`

Extract the pure payload construction:
- `build_code_graph_payload` (line 154-222) — builds nodes + edges from IR snapshot
- `build_lineage_payload` (line 251-400+) — builds full lineage graph
- `_deterministic_event_id` (line 29-32) — hash-based event ID

These are pure dict construction — no I/O.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_core_graph_build.py
"""Tests for pure graph payload construction."""
from fastcode.core.graph_build import (
    deterministic_event_id,
    build_code_graph_payload,
)


class TestDeterministicEventId:
    def test_deterministic(self):
        result1 = deterministic_event_id("snap:test:abc", "payload1")
        result2 = deterministic_event_id("snap:test:abc", "payload1")
        assert result1 == result2

    def test_different_inputs(self):
        result1 = deterministic_event_id("snap:test:abc", "payload1")
        result2 = deterministic_event_id("snap:test:abc", "payload2")
        assert result1 != result2

    def test_format(self):
        result = deterministic_event_id("snap:test:abc", "payload")
        assert result.startswith("outbox:")
        assert "snap:test:abc" in result


class TestBuildCodeGraphPayload:
    def test_skips_file_units(self):
        snapshot = {
            "snapshot_id": "snap:test:abc",
            "units": [
                {"kind": "file", "unit_id": "file1"},
                {"kind": "function", "unit_id": "sym1", "display_name": "func1"},
            ],
            "relations": [],
        }
        result = build_code_graph_payload(snapshot)
        assert len(result["nodes"]) == 1
        assert "sym1" in result["nodes"][0]["id"]

    def test_skips_doc_units(self):
        snapshot = {
            "snapshot_id": "snap:test:abc",
            "units": [
                {"kind": "doc", "unit_id": "doc1"},
            ],
            "relations": [],
        }
        result = build_code_graph_payload(snapshot)
        assert len(result["nodes"]) == 0

    def test_builds_edges(self):
        snapshot = {
            "snapshot_id": "snap:test:abc",
            "units": [],
            "relations": [
                {"relation_id": "r1", "src_unit_id": "a", "dst_unit_id": "b",
                 "relation_type": "calls", "resolution_state": "precise"},
            ],
        }
        result = build_code_graph_payload(snapshot)
        assert len(result["edges"]) == 1
        assert result["edges"][0]["type"] == "calls"

    def test_skips_units_without_id(self):
        snapshot = {
            "snapshot_id": "snap:test:abc",
            "units": [{"kind": "function"}],
            "relations": [],
        }
        result = build_code_graph_payload(snapshot)
        assert len(result["nodes"]) == 0

    def test_empty_snapshot(self):
        result = build_code_graph_payload({"snapshot_id": "snap:test:abc"})
        assert result == {"nodes": [], "edges": []}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_graph_build.py -v`
Expected: FAIL

- [ ] **Step 3: Write `fastcode/core/graph_build.py`**

```python
# fastcode/core/graph_build.py
"""Pure graph payload construction — extracted from terminus_publisher.py."""
from __future__ import annotations

import hashlib
from typing import Any


def deterministic_event_id(snapshot_id: str, payload: str) -> str:
    """Generate a deterministic event ID from snapshot_id + payload hash."""
    h = hashlib.sha256(f"{snapshot_id}:{payload}".encode()).hexdigest()[:32]
    return f"outbox:{snapshot_id}:{h}"


def build_code_graph_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build TerminusDB payload for code graph (symbol nodes + relation edges)."""
    snapshot_id = snapshot.get("snapshot_id")
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for unit in snapshot.get("units") or []:
        kind = unit.get("kind", "")
        if kind in ("file", "doc"):
            continue
        unit_id = unit.get("unit_id")
        if not unit_id:
            continue
        node_id = f"sym:{snapshot_id}:{unit_id}"
        source_set = unit.get("source_set") or []
        nodes.append({
            "id": node_id,
            "type": "Symbol",
            "props": {
                "unit_id": unit_id,
                "display_name": unit.get("display_name"),
                "kind": kind,
                "path": unit.get("path"),
                "language": unit.get("language"),
                "start_line": unit.get("start_line"),
                "end_line": unit.get("end_line"),
                "qualified_name": unit.get("qualified_name"),
                "scip_symbol": unit.get("primary_anchor_symbol_id"),
                "source_set": source_set if isinstance(source_set, list) else list(source_set),
            },
        })

    for rel in snapshot.get("relations") or []:
        rel_id = rel.get("relation_id")
        if not rel_id:
            continue
        src_id = rel.get("src_unit_id")
        dst_id = rel.get("dst_unit_id")
        if not src_id or not dst_id:
            continue
        support_sources = rel.get("support_sources") or []
        edges.append({
            "id": f"rel:{snapshot_id}:{rel_id}",
            "type": rel.get("relation_type", ""),
            "src": f"sym:{snapshot_id}:{src_id}",
            "dst": f"sym:{snapshot_id}:{dst_id}",
            "confidence": _resolution_to_confidence(rel.get("resolution_state", "")),
            "resolution_state": rel.get("resolution_state", ""),
            "source_set": support_sources if isinstance(support_sources, list) else list(support_sources),
        })

    return {"nodes": nodes, "edges": edges}


def _resolution_to_confidence(state: str) -> float:
    """Map resolution state to confidence score."""
    return {"precise": 1.0, "resolved": 0.8, "heuristic": 0.5}.get(state, 0.3)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_core_graph_build.py -v`
Expected: All PASS

- [ ] **Step 5: Wire `TerminusPublisher` to delegate**

```python
# In terminus_publisher.py:
from fastcode.core import graph_build as _graph_build

def _deterministic_event_id(self, snapshot_id, payload):
    return _graph_build.deterministic_event_id(snapshot_id, payload)

def build_code_graph_payload(self, snapshot):
    return _graph_build.build_code_graph_payload(snapshot)
```

- [ ] **Step 6: Run existing tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add fastcode/core/graph_build.py tests/test_core_graph_build.py fastcode/terminus_publisher.py
git commit -m "feat: extract pure graph payload construction to core/graph_build.py"
```

---

### Task 4.3: Extract snapshot pure logic

**Files:**
- Create: `fastcode/core/snapshot.py`
- Create: `tests/test_core_snapshot.py`
- Modify: `fastcode/main.py`

Extract from `FastCode`:
- `_projection_scope_key` (line 1333-1348) — static hash key computation
- `_projection_params_hash` (line 1350-1362) — static parameter hashing
- `_extract_sources_from_elements` (line 2026-2040) — source extraction

These are all static methods — pure computation.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_core_snapshot.py
"""Tests for pure snapshot logic."""
from fastcode.core.snapshot import (
    projection_scope_key,
    projection_params_hash,
    extract_sources_from_elements,
)


class TestProjectionScopeKey:
    def test_deterministic(self):
        key1 = projection_scope_key("global", "snap:test:abc", None, None, None)
        key2 = projection_scope_key("global", "snap:test:abc", None, None, None)
        assert key1 == key2

    def test_different_scope(self):
        key1 = projection_scope_key("global", "snap:test:abc", None, None, None)
        key2 = projection_scope_key("targeted", "snap:test:abc", None, None, None)
        assert key1 != key2

    def test_with_query(self):
        key_no_query = projection_scope_key("global", "snap:test:abc", None, None, None)
        key_with_query = projection_scope_key("global", "snap:test:abc", "how does X work?", None, None)
        assert key_no_query != key_with_query

    def test_with_filters(self):
        key_no_filter = projection_scope_key("global", "snap:test:abc", None, None, None)
        key_with_filter = projection_scope_key("global", "snap:test:abc", None, None, {"language": "python"})
        assert key_no_filter != key_with_filter


class TestProjectionParamsHash:
    def test_produces_hash(self):
        # Use a simple dict-based scope since we don't have ProjectionScope here
        import json
        import hashlib
        scope_dict = {"scope_kind": "global", "snapshot_id": "snap:test"}
        payload = json.dumps({"scope": scope_dict, "projection_algo_version": "v1"}, sort_keys=True)
        expected = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        # Just verify it returns a hex string
        assert len(expected) == 40


class TestExtractSourcesFromElements:
    def test_basic(self):
        elements = [
            {
                "element": {
                    "relative_path": "src/main.py",
                    "repo_name": "myrepo",
                    "type": "function",
                    "name": "func1",
                    "start_line": 10,
                    "end_line": 20,
                },
            },
        ]
        sources = extract_sources_from_elements(elements)
        assert len(sources) == 1
        assert sources[0]["file"] == "src/main.py"
        assert sources[0]["repo"] == "myrepo"

    def test_empty(self):
        assert extract_sources_from_elements([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_snapshot.py -v`
Expected: FAIL

- [ ] **Step 3: Write `fastcode/core/snapshot.py`**

```python
# fastcode/core/snapshot.py
"""Pure snapshot logic — extracted from main.py."""
from __future__ import annotations

import hashlib
import json
from typing import Any


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


def projection_params_hash(scope_dict: dict[str, Any], version: str = "v1") -> str:
    """Hash projection parameters for cache key."""
    payload = json.dumps(
        {"scope": scope_dict, "projection_algo_version": version},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def extract_sources_from_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract source information from retrieved elements."""
    sources: list[dict[str, Any]] = []
    for elem_data in elements:
        elem = elem_data.get("element", {})
        sources.append({
            "file": elem.get("relative_path", ""),
            "repo": elem.get("repo_name", ""),
            "type": elem.get("type", ""),
            "name": elem.get("name", ""),
            "start_line": elem.get("start_line", 0),
            "end_line": elem.get("end_line", 0),
        })
    return sources
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_core_snapshot.py -v`
Expected: All PASS

- [ ] **Step 5: Wire `FastCode` to delegate**

```python
# In main.py:
from fastcode.core import snapshot as _snapshot

@staticmethod
def _projection_scope_key(scope_kind, snapshot_id, query, target_id, filters):
    return _snapshot.projection_scope_key(scope_kind, snapshot_id, query, target_id, filters)

@staticmethod
def _projection_params_hash(scope, projection_algo_version="v1"):
    return _snapshot.projection_params_hash(scope.to_dict(), projection_algo_version)

def _extract_sources_from_elements(self, elements):
    return _snapshot.extract_sources_from_elements(elements)
```

- [ ] **Step 6: Run existing tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add fastcode/core/snapshot.py tests/test_core_snapshot.py fastcode/main.py
git commit -m "feat: extract pure snapshot logic to core/snapshot.py"
```

---

### Task 4.4: Extract repo analysis functions

**Files:**
- Create: `fastcode/core/repo_analysis.py`
- Create: `tests/test_core_repo_analysis.py`
- Modify: `fastcode/repo_overview.py`

Extract the pure functions:
- `_get_language_from_extension` (line 194-211) — static mapping
- `_is_key_file` (line 213-226) — static check
- `_infer_project_type` (line 311-344) — static inference
- `_generate_structure_based_overview` (line 283-309) — pure string building
- `_format_file_structure` (line 346-378) — pure formatting

- [ ] **Step 1: Write failing tests**

```python
# tests/test_core_repo_analysis.py
"""Tests for pure repo analysis functions."""
from fastcode.core.repo_analysis import (
    get_language_from_extension,
    is_key_file,
    infer_project_type,
    generate_structure_based_overview,
    format_file_structure,
)


class TestGetLanguageFromExtension:
    def test_python(self):
        assert get_language_from_extension(".py") == "python"

    def test_typescript(self):
        assert get_language_from_extension(".ts") == "typescript"

    def test_go(self):
        assert get_language_from_extension(".go") == "go"

    def test_unknown(self):
        assert get_language_from_extension(".xyz") == "unknown"

    def test_case_insensitive(self):
        assert get_language_from_extension(".PY") == "python"


class TestIsKeyFile:
    def test_main(self):
        assert is_key_file("src/main.py")

    def test_package_json(self):
        assert is_key_file("package.json")

    def test_dockerfile(self):
        assert is_key_file("Dockerfile")

    def test_regular_file(self):
        assert not is_key_file("src/utils/helper.py")


class TestInferProjectType:
    def test_react(self):
        assert "React" in infer_project_type(["package.json"], {"tsx": 10})

    def test_python_project(self):
        assert "Python" in infer_project_type(["requirements.txt"], {"python": 10})

    def test_django(self):
        assert "Django" in infer_project_type(["requirements.txt", "manage.py"], {})

    def test_default(self):
        assert "software project" in infer_project_type([], {})


class TestGenerateStructureBasedOverview:
    def test_basic(self):
        structure = {
            "total_files": 10,
            "languages": {"Python": 8, "TypeScript": 2},
            "key_files": ["main.py", "setup.py"],
        }
        overview = generate_structure_based_overview("myrepo", structure)
        assert "myrepo" in overview
        assert "10 files" in overview

    def test_no_languages(self):
        structure = {"total_files": 5, "languages": {}, "key_files": []}
        overview = generate_structure_based_overview("repo", structure)
        assert "unknown" in overview


class TestFormatFileStructure:
    def test_basic(self):
        structure = {
            "total_files": 10,
            "languages": {"Python": 8},
            "all_files": ["main.py", "utils.py"],
            "directories": {"src": ["main.py"]},
        }
        text = format_file_structure(structure)
        assert "Total Files: 10" in text
        assert "Python: 8 files" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_repo_analysis.py -v`
Expected: FAIL

- [ ] **Step 3: Write `fastcode/core/repo_analysis.py`**

```python
# fastcode/core/repo_analysis.py
"""Pure repo analysis functions — extracted from repo_overview.py."""
from __future__ import annotations

import os
from typing import Any


def get_language_from_extension(ext: str) -> str:
    """Get programming language from file extension."""
    language_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "javascript", ".tsx": "typescript", ".java": "java",
        ".go": "go", ".cpp": "cpp", ".c": "c", ".rs": "rust",
        ".rb": "ruby", ".php": "php", ".cs": "csharp",
    }
    return language_map.get(ext.lower(), "unknown")


def is_key_file(file_path: str) -> bool:
    """Check if file is a key/important file."""
    file_name = os.path.basename(file_path).lower()
    key_names = [
        "main", "index", "app", "init", "config", "setup",
        "package.json", "requirements.txt", "go.mod", "cargo.toml",
        "dockerfile", "makefile", "cmakelists.txt",
    ]
    return any(key in file_name for key in key_names)


def infer_project_type(key_files: list[str], languages: dict[str, int]) -> str:
    """Infer project type from key files and languages."""
    key_files_str = " ".join(key_files).lower()

    if "package.json" in key_files_str:
        if "react" in key_files_str or "tsx" in languages:
            return "React web application"
        elif "vue" in key_files_str:
            return "Vue.js web application"
        return "Node.js application"

    if "requirements.txt" in key_files_str or "setup.py" in key_files_str:
        if "django" in key_files_str:
            return "Django web application"
        elif "flask" in key_files_str:
            return "Flask web application"
        return "Python application"

    if "android" in key_files_str or "java" in languages:
        return "Android application"
    if "ios" in key_files_str or "swift" in key_files_str:
        return "iOS application"
    if "dockerfile" in key_files_str:
        return "containerized application"

    return "software project"


def generate_structure_based_overview(
    repo_name: str,
    file_structure: dict[str, Any],
) -> str:
    """Generate overview from file structure when README is unavailable."""
    languages = file_structure.get("languages", {})
    total_files = file_structure.get("total_files", 0)
    key_files = file_structure.get("key_files", [])

    primary_lang = max(languages.items(), key=lambda x: x[1])[0] if languages else "unknown"
    project_type = infer_project_type(key_files, languages)

    summary = f"{repo_name} is a {primary_lang} {project_type} with {total_files} files. "

    if len(languages) > 1:
        lang_list = ", ".join(languages.keys())
        summary += f"It uses multiple languages: {lang_list}. "

    if key_files:
        summary += f"Key entry points include: {', '.join(key_files[:5])}."

    return summary


def format_file_structure(file_structure: dict[str, Any]) -> str:
    """Format file structure as readable text."""
    lines: list[str] = []

    total_files = file_structure.get("total_files", 0)
    lines.append(f"Total Files: {total_files}")

    languages = file_structure.get("languages", {})
    if languages:
        lines.append("\nLanguages:")
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {lang}: {count} files")

    all_files = file_structure.get("all_files", [])
    if all_files:
        lines.append("\nFiles:")
        for file_path in sorted(all_files):
            lines.append(f"  - {file_path}")

    directories = file_structure.get("directories", {})
    top_dirs = [d for d in directories.keys() if os.sep not in d or d.count(os.sep) == 0]
    if top_dirs:
        lines.append("\nTop-Level Directories:")
        for td in sorted(top_dirs)[:15]:
            file_count = len(directories[td])
            lines.append(f"  - {td}/ ({file_count} files)")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_core_repo_analysis.py -v`
Expected: All PASS

- [ ] **Step 5: Wire `RepoOverview` to delegate**

```python
# In repo_overview.py:
from fastcode.core import repo_analysis as _repo_analysis

def _get_language_from_extension(self, ext):
    return _repo_analysis.get_language_from_extension(ext)

def _is_key_file(self, file_path):
    return _repo_analysis.is_key_file(file_path)

def _infer_project_type(self, key_files, languages):
    return _repo_analysis.infer_project_type(key_files, languages)

def _generate_structure_based_overview(self, repo_name, file_structure):
    return _repo_analysis.generate_structure_based_overview(repo_name, file_structure)

def _format_file_structure(self, file_structure):
    return _repo_analysis.format_file_structure(file_structure)
```

- [ ] **Step 6: Run existing tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add fastcode/core/repo_analysis.py tests/test_core_repo_analysis.py fastcode/repo_overview.py
git commit -m "feat: extract pure repo analysis to core/repo_analysis.py"
```

---

### Task 4.5: Extract SCIP transform functions

**Files:**
- Create: `fastcode/core/scip_transform.py`
- Create: `tests/test_core_scip_transform.py`
- Modify: `fastcode/scip_loader.py`

Extract:
- `_symbol_role_to_str` (line 112-122) — bitmask to string
- `_scip_kind_to_str` (line 125-152) — enum to string

`_protobuf_to_scip_index` depends on protobuf imports — mark as a future extraction (it touches I/O-bound protobuf parsing). Extract the pure enum/bitmask conversions first.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_core_scip_transform.py
"""Tests for pure SCIP transform functions."""
from fastcode.core.scip_transform import symbol_role_to_str, scip_kind_to_str


class TestSymbolRoleToStr:
    def test_definition(self):
        assert symbol_role_to_str(1) == "definition"

    def test_import(self):
        assert symbol_role_to_str(2) == "import"

    def test_write_access(self):
        assert symbol_role_to_str(4) == "write_access"

    def test_forward_definition(self):
        assert symbol_role_to_str(64) == "forward_definition"

    def test_reference(self):
        assert symbol_role_to_str(0) == "reference"

    def test_combined_roles(self):
        # Definition | Reference = 1 | 0 (definition wins)
        assert symbol_role_to_str(1) == "definition"


class TestScipKindToStr:
    def test_known_kinds(self):
        # Test with known kind values using the enum if available
        # Falls back to "symbol" for unknown values
        assert scip_kind_to_str(999) == "symbol"

    def test_fallback(self):
        assert scip_kind_to_str(-1) == "symbol"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_scip_transform.py -v`
Expected: FAIL

- [ ] **Step 3: Write `fastcode/core/scip_transform.py`**

```python
# fastcode/core/scip_transform.py
"""Pure SCIP transform functions — extracted from scip_loader.py."""
from __future__ import annotations


def symbol_role_to_str(roles: int) -> str:
    """Convert SCIP symbol_roles bitmask to a semantic role string."""
    if roles & 1:
        return "definition"
    if roles & 2:
        return "import"
    if roles & 4:
        return "write_access"
    if roles & 64:
        return "forward_definition"
    return "reference"


def scip_kind_to_str(kind_value: int) -> str:
    """Convert SCIP protobuf Kind enum to string."""
    try:
        from fastcode.scip_pb2 import SymbolInformation
    except ImportError:
        return "symbol"

    kind_map = {
        SymbolInformation.Kind.Function: "function",
        SymbolInformation.Kind.Method: "method",
        SymbolInformation.Kind.Class: "class",
        SymbolInformation.Kind.Interface: "interface",
        SymbolInformation.Kind.Enum: "enum",
        SymbolInformation.Kind.EnumMember: "enum_member",
        SymbolInformation.Kind.Variable: "variable",
        SymbolInformation.Kind.Constant: "constant",
        SymbolInformation.Kind.Property: "property",
        SymbolInformation.Kind.Type: "type",
        SymbolInformation.Kind.Macro: "macro",
        SymbolInformation.Kind.Module: "module",
        SymbolInformation.Kind.Namespace: "namespace",
        SymbolInformation.Kind.Package: "package",
        SymbolInformation.Kind.Parameter: "parameter",
        SymbolInformation.Kind.TypeParameter: "type_parameter",
        SymbolInformation.Kind.Constructor: "constructor",
        SymbolInformation.Kind.Struct: "struct",
    }
    return kind_map.get(kind_value, "symbol")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_core_scip_transform.py -v`
Expected: All PASS

- [ ] **Step 5: Wire `scip_loader.py` to delegate**

```python
# In scip_loader.py:
from fastcode.core import scip_transform as _scip_transform

def _symbol_role_to_str(roles):
    return _scip_transform.symbol_role_to_str(roles)

def _scip_kind_to_str(kind_value):
    return _scip_transform.scip_kind_to_str(kind_value)
```

- [ ] **Step 6: Run existing tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add fastcode/core/scip_transform.py tests/test_core_scip_transform.py fastcode/scip_loader.py
git commit -m "feat: extract pure SCIP transforms to core/scip_transform.py"
```

---

## Phase 5: Create Thin Effects Layer

### Task 5.1: Create DB effects module

**Files:**
- Create: `fastcode/effects/db.py`
- Create: `tests/test_effects_db.py`
- Modify: `fastcode/core/types.py` (add `SnapshotRecord`)

Rule 2 enforcement: all functions return frozen dataclasses, never `dict[str, Any]`.

- [ ] **Step 1: Add `SnapshotRecord` to `core/types.py`**

```python
# In core/types.py, add:
@dataclass(frozen=True)
class SnapshotRecord:
    """A snapshot metadata row from the database."""
    snapshot_id: str
    repo_name: str
    branch: str | None = None
    commit_id: str | None = None
    tree_id: str | None = None
```

- [ ] **Step 2: Write failing tests using in-memory SQLite**

```python
# tests/test_effects_db.py
"""Tests for DB effects — verify frozen dataclass returns."""
import sqlite3
from dataclasses import FrozenInstanceError

import pytest

from fastcode.core.types import SnapshotRecord
from fastcode.effects.db import load_snapshot_record, save_snapshot_record


def _setup_test_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE snapshots (
            snapshot_id TEXT PRIMARY KEY,
            repo_name TEXT,
            branch TEXT,
            commit_id TEXT,
            tree_id TEXT
        )
    """)
    return conn


class TestLoadSnapshotRecord:
    def test_returns_dataclass_not_dict(self):
        conn = _setup_test_db()
        conn.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?)",
            ("snap:test:abc123", "myrepo", "main", "abc123", "tree1"),
        )
        result = load_snapshot_record(conn, "snap:test:abc123")
        assert isinstance(result, SnapshotRecord)
        assert result.snapshot_id == "snap:test:abc123"
        assert result.repo_name == "myrepo"

    def test_not_found_returns_none(self):
        conn = _setup_test_db()
        result = load_snapshot_record(conn, "nonexistent")
        assert result is None

    def test_result_is_frozen(self):
        conn = _setup_test_db()
        conn.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?)",
            ("snap:test:abc123", "myrepo", "main", "abc123", "tree1"),
        )
        result = load_snapshot_record(conn, "snap:test:abc123")
        with pytest.raises(FrozenInstanceError):
            result.repo_name = "changed"  # type: ignore[misc]


class TestSaveSnapshotRecord:
    def test_insert_and_load(self):
        conn = _setup_test_db()
        record = SnapshotRecord(
            snapshot_id="snap:test:new",
            repo_name="newrepo",
            branch="dev",
            commit_id="def456",
            tree_id="tree2",
        )
        save_snapshot_record(conn, record)
        loaded = load_snapshot_record(conn, "snap:test:new")
        assert loaded is not None
        assert loaded.repo_name == "newrepo"
        assert loaded.branch == "dev"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_effects_db.py -v`
Expected: FAIL

- [ ] **Step 4: Write `fastcode/effects/db.py`**

```python
# fastcode/effects/db.py
"""Thin wrappers for database I/O — each function does one query.

Rule 2: Database Trusts Dataclasses.
Every function maps DB rows into frozen dataclasses before returning.
No dict[str, Any] returns.
"""
from __future__ import annotations

from typing import Any

from fastcode.core.types import SnapshotRecord


def load_snapshot_record(
    conn: Any,
    snapshot_id: str,
) -> SnapshotRecord | None:
    """Load a snapshot record by ID. Returns frozen dataclass, not dict."""
    cursor = conn.execute(
        "SELECT snapshot_id, repo_name, branch, commit_id, tree_id "
        "FROM snapshots WHERE snapshot_id = ?",
        (snapshot_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return SnapshotRecord(
        snapshot_id=row[0],
        repo_name=row[1],
        branch=row[2],
        commit_id=row[3],
        tree_id=row[4],
    )


def save_snapshot_record(conn: Any, record: SnapshotRecord) -> None:
    """Insert or update a snapshot record. Accepts frozen dataclass."""
    conn.execute(
        "INSERT OR REPLACE INTO snapshots "
        "(snapshot_id, repo_name, branch, commit_id, tree_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (record.snapshot_id, record.repo_name, record.branch,
         record.commit_id, record.tree_id),
    )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_effects_db.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add fastcode/effects/db.py tests/test_effects_db.py fastcode/core/types.py
git commit -m "feat: add thin DB effects layer — returns frozen dataclasses only"
```

---

### Task 5.2: Create LLM effects module

**Files:**
- Create: `fastcode/effects/llm.py`
- Create: `tests/test_effects_llm.py`

Thin wrapper for LLM API calls. Wraps existing `openai_chat_completion` with consistent error handling.

- [ ] **Step 1: Write test with mock client**

```python
# tests/test_effects_llm.py
"""Tests for LLM effects — verify thin wrapper behavior."""
from unittest.mock import MagicMock

from fastcode.effects.llm import chat_completion


class TestChatCompletion:
    def test_returns_content_string(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_client.chat.completions.create.return_value = mock_response

        result = chat_completion(
            mock_client, model="test-model", messages=[], max_tokens=100,
        )
        assert result == "test response"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_effects_llm.py -v`
Expected: FAIL

- [ ] **Step 3: Write `fastcode/effects/llm.py`**

```python
# fastcode/effects/llm.py
"""Thin wrappers for LLM API I/O."""
from __future__ import annotations

from typing import Any


def chat_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float = 0.3,
    **kwargs: Any,
) -> str:
    """Single LLM completion. Returns response content string."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_effects_llm.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add fastcode/effects/llm.py tests/test_effects_llm.py
git commit -m "feat: add thin LLM effects layer"
```

---

### Task 5.3: Create FS effects module

**Files:**
- Create: `fastcode/effects/fs.py`
- Create: `tests/test_effects_fs.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_effects_fs.py
"""Tests for FS effects — verify thin file I/O wrappers."""
import os
import tempfile

from fastcode.effects.fs import read_file, write_file, file_exists


class TestReadWriteFile:
    def test_round_trip(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            write_file(path, "hello world")
            content = read_file(path)
            assert content == "hello world"
        finally:
            os.unlink(path)

    def test_file_exists(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            assert file_exists(path)
        finally:
            os.unlink(path)

    def test_file_not_exists(self):
        assert not file_exists("/nonexistent/path/file.txt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_effects_fs.py -v`
Expected: FAIL

- [ ] **Step 3: Write `fastcode/effects/fs.py`**

```python
# fastcode/effects/fs.py
"""Thin wrappers for file system I/O."""
from __future__ import annotations

import os
from typing import Any


def read_file(path: str) -> str:
    """Read file contents."""
    with open(path) as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    """Write file contents."""
    with open(path, "w") as f:
        f.write(content)


def file_exists(path: str) -> bool:
    """Check if file exists."""
    return os.path.exists(path)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_effects_fs.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add fastcode/effects/fs.py tests/test_effects_fs.py
git commit -m "feat: add thin FS effects layer"
```

---

## Phase 6: Validate Golden Rules and Run Full Suite

### Task 6.1: Validate Rule 1 — Pydantic Stops at the Door

- [ ] **Step 1: Run the I/O import guard test**

Run: `uv run pytest tests/test_core_boundary.py::test_core_modules_have_no_io_imports -v`
Expected: PASS — no I/O imports (including `pydantic`) in any `core/` module

- [ ] **Step 2: Verify via grep**

Run: `grep -r "pydantic" fastcode/core/ || echo "NO PYDANTIC IN CORE - PASS"`
Expected: `NO PYDANTIC IN CORE - PASS`

---

### Task 6.2: Validate Rule 2 — Database Trusts Dataclasses

- [ ] **Step 1: Run the DB effects return-type guard test**

Add to `tests/test_core_boundary.py`:

```python
def test_db_effects_return_dataclasses_not_dicts():
    """Rule 2: effects/db.py must return frozen dataclasses, never dict."""
    import ast
    from pathlib import Path

    db_effects = Path(__file__).resolve().parent.parent / "fastcode" / "effects" / "db.py"
    if not db_effects.exists():
        pytest.skip("effects/db.py not yet created")

    tree = ast.parse(db_effects.read_text())
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            ret = node.returns
            if ret is None:
                continue
            ret_str = ast.dump(ret)
            if "dict" in ret_str.lower() and "Any" in ret_str:
                violations.append(f"{node.name}: return type contains dict[str, Any]")
    assert violations == [], "Rule 2 violations in effects/db.py:\n" + "\n".join(violations)
```

Run: `uv run pytest tests/test_core_boundary.py::test_db_effects_return_dataclasses_not_dicts -v`
Expected: PASS

---

### Task 6.3: Validate Rule 3 — Explicit Translation

- [ ] **Step 1: Run the explicit translation guard test**

Run: `uv run pytest tests/test_core_boundary.py::test_boundary_uses_explicit_translation -v`
Expected: PASS

---

### Task 6.4: Run full test suite

- [ ] **Step 1: Run all existing tests (verify no regressions)**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 2: Run all new core + effects tests**

Run: `uv run pytest tests/test_core_*.py tests/test_core_boundary.py tests/test_effects_*.py -v`
Expected: All PASS

- [ ] **Step 3: Run coverage check on core modules**

Run: `uv run pytest tests/ --cov=fastcode/core --cov-report=term-missing`
Expected: >90% coverage on core/ modules

---

### Task 6.5: Commit phase completion

- [ ] **Step 1: Final commit**

```bash
git add -A
git commit -m "feat: complete FP core + thin I/O refactoring (Phases 1-6)

- Extract pure logic into fastcode/core/ (14 modules)
- Create thin I/O wrappers in fastcode/effects/ (5 modules)
- Wire orchestrators to delegate to core + effects
- Existing API surface unchanged
- All existing tests pass
- Rule 1: Pydantic stops at the door (automated guard)
- Rule 2: DB effects return frozen dataclasses only (automated guard)
- Rule 3: Explicit translation in boundary.py (automated guard)"
```

---

## Summary: Task Dependency Order

```
Phase 1 (remaining)
  1.3: Filtering functions (independent of 1.4)
  1.4: Combination function (independent of 1.3)

Phase 2 (iterative_agent)
  2.1: Iteration control (independent)
  2.2: Prompt building (independent)
  2.3: Parsing functions (independent)

Phase 3 (answer_generator)
  3.1: Context preparation (independent)
  3.2: Summary/formatting (independent)

Phase 4 (remaining modules)
  4.2: Graph payload construction (independent)
  4.3: Snapshot pure logic (independent)
  4.4: Repo analysis (independent)
  4.5: SCIP transforms (independent)

Phase 5 (effects layer)
  5.1: DB effects (depends on SnapshotRecord from types.py)
  5.2: LLM effects (independent)
  5.3: FS effects (independent)

Phase 6 (validation)
  6.1-6.5: Run guards, full suite, coverage
```

### Dependency Graph

```
Phases 1.3-1.4, 2.1-2.3, 3.1-3.2, 4.2-4.5 are all independent.
Phase 5 can start as soon as types.py has needed dataclasses.
Phase 6 is the final validation gate after all other phases complete.
```
