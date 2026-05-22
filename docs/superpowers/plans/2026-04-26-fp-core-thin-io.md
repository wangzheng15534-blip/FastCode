# FP Core with Thin I/O Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract pure business logic from I/O-touching modules into `fastcode/core/` (frozen dataclasses + pure functions) and thin I/O wrappers into `fastcode/effects/`, using the strangler fig pattern.

**Architecture:** New `core/` package holds zero-I/O pure functions and frozen dataclasses. New `effects/` package holds thin wrappers for DB, LLM, FS, and graph DB I/O. Existing orchestrator classes (`HybridRetriever`, `FastCode`, `IterativeAgent`, etc.) are modified to call `core/` for logic and `effects/` for I/O. Existing API surface stays unchanged.

**Tech Stack:** Python >=3.11, stdlib dataclasses (frozen), no new dependencies.

**Design Spec:** `docs/superpowers/specs/2026-04-26-fp-core-thin-io-design.md`

### Three Golden Rules (enforced in code review)

These rules are non-negotiable. Every commit must pass them:

**1. Pydantic Stops at the Door.** No file inside `fastcode/core/` may import `pydantic`. If a pure function needs a value from a user request, the orchestrator (`main.py`) must unpack it and pass it as a standard `str`/`int` or a frozen dataclass. Automated guard: `test_core_modules_have_no_io_imports` includes `pydantic` in the forbidden set.

**2. Database Trusts Dataclasses.** `effects/db.py` functions must never return `dict[str, Any]` or Pydantic models. They must immediately map database rows into frozen dataclasses before returning. If a row doesn't fit an existing dataclass, create a new one in `core/types.py`.

**3. Explicit Translation.** Bridging between Pydantic and dataclasses requires explicit field-by-field mapping: `MyDataclass(field=pydantic_obj.field)`. No `**pydantic_obj.dict()`, no `from_orm`, no auto-conversion. This explicit mapping is the mechanism that prevents mass-assignment vulnerabilities and keeps the core fast. The `core/boundary.py` module is the only place this translation happens.

---

## File Structure

### New Files to Create

```
fastcode/core/
  __init__.py
  types.py              # Frozen dataclasses: Hit, FusionWeights, IterationState, etc.
  scoring.py            # Scoring, normalization, sigmoid, entropy, affinity
  fusion.py             # Adaptive fusion, doc projection, channel fusion
  filtering.py          # Result filtering, diversification, repo filtering
  combination.py        # Result combination, reranking
  iteration.py          # Iteration control, stopping conditions, adaptive params
  prompts.py            # Prompt construction for iterative agent
  parsing.py            # JSON parsing, LLM output parsing, robust_json_parse
  snapshot.py           # Snapshot key generation, hash computation
  projection.py         # Projection transforms (pure graph algorithms)
  graph_build.py        # Graph payload construction for TerminusDB
  context.py            # Context preparation, truncation for answer generation
  summary.py            # Summary generation, source extraction, formatting
  repo_analysis.py      # File structure analysis, project type detection, language mapping
  scip_transform.py     # SCIP protobuf/enum to native type conversion

fastcode/effects/
  __init__.py
  db.py                 # PostgreSQL/SQLite queries (thin wrappers)
  llm.py                # OpenAI/Anthropic/Ollama API calls
  fs.py                 # File system and git operations
  graph_db.py           # TerminusDB/LadybugDB operations
  embedding.py          # Embedding generation

tests/
  test_core_types.py
  test_core_scoring.py
  test_core_fusion.py
  test_core_filtering.py
  test_core_combination.py
  test_core_iteration.py
  test_core_prompts.py
  test_core_parsing.py
  test_core_snapshot.py
  test_core_projection.py
  test_core_graph_build.py
  test_core_context.py
  test_core_summary.py
  test_core_repo_analysis.py
  test_core_scip_transform.py
```

### Existing Files to Modify (per phase)

```
fastcode/retriever.py            # Wire to core/ functions
fastcode/main.py                 # Wire query path to core/ + effects/
fastcode/iterative_agent.py      # Wire iteration to core/ + effects/
fastcode/answer_generator.py     # Wire generation to core/ + effects/
fastcode/projection_transform.py # Wire to core/projection.py
fastcode/terminus_publisher.py   # Wire to core/graph_build.py + effects/graph_db.py
fastcode/embedder.py             # Wire to effects/embedding.py
fastcode/scip_loader.py          # Wire to core/scip_transform.py + effects/fs.py
fastcode/repo_overview.py        # Wire to core/repo_analysis.py + effects/llm.py
fastcode/index_run.py            # Wire to effects/db.py
fastcode/snapshot_store.py       # Wire to core/snapshot.py + effects/db.py
fastcode/pg_retrieval.py         # Wire to effects/db.py
```

---

## Phase 0: Package Structure and Core Types

**Goal:** Create the `core/` and `effects/` packages with shared frozen dataclasses that replace `Dict[str, Any]` throughout.

### Task 0.1: Create package scaffolding

**Files:**
- Create: `fastcode/core/__init__.py`
- Create: `fastcode/effects/__init__.py`

- [ ] **Step 1: Create empty `__init__.py` files**

```python
# fastcode/core/__init__.py
"""Pure business logic — zero I/O imports."""
```

```python
# fastcode/effects/__init__.py
"""Thin I/O wrappers — each function does exactly one I/O operation."""
```

- [ ] **Step 2: Verify imports work**

Run: `uv run python -c "from fastcode.core import types; print('ok')"`
Expected: may fail (types.py not yet created) — that's fine, just check package is importable after types.py is created

- [ ] **Step 3: Commit**

```bash
git add fastcode/core/__init__.py fastcode/effects/__init__.py
git commit -m "feat: scaffold core/ and effects/ packages for FP refactoring"
```

---

### Task 0.2: Define core frozen dataclasses

**Files:**
- Create: `fastcode/core/types.py`
- Create: `tests/test_core_types.py`

These frozen dataclasses replace the `Dict[str, Any]` patterns used throughout retriever, iterative_agent, and answer_generator. They are the canonical internal types — no Pydantic, no validation overhead.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core_types.py
"""Tests for core frozen dataclasses — construction, immutability, serialization."""
from dataclasses import FrozenInstanceError

import pytest

from fastcode.core.types import (
    Hit,
    FusionConfig,
    FusionWeights,
    RetrievalChannelOutput,
    ElementFilter,
    IterationState,
    IterationConfig,
    IterationMetrics,
    RoundResult,
    ToolCall,
    IterationHistoryEntry,
    GenerationInput,
    GenerationResult,
    SourceRef,
    FileAnalysis,
    RepoStructure,
    ScipKind,
    ScipRole,
)


class TestHit:
    def test_construction(self):
        h = Hit(element_id="sym:repo:func", score=0.85, source="semantic")
        assert h.element_id == "sym:repo:func"
        assert h.score == 0.85
        assert h.source == "semantic"

    def test_frozen(self):
        h = Hit(element_id="a", score=0.5, source="keyword")
        with pytest.raises(FrozenInstanceError):
            h.score = 0.9  # type: ignore[misc]

    def test_from_dict(self):
        row = {
            "element": {"id": "sym:repo:func", "type": "function", "name": "func"},
            "semantic_score": 0.8,
            "keyword_score": 0.5,
            "pseudocode_score": 0.0,
            "graph_score": 0.0,
            "total_score": 0.8,
        }
        hit = Hit.from_retrieval_row(row)
        assert hit.element_id == "sym:repo:func"
        assert hit.semantic_score == 0.8
        assert hit.total_score == 0.8

    def test_from_dict_with_metadata(self):
        row = {
            "element": {
                "id": "sym:repo:func",
                "type": "function",
                "name": "func",
                "metadata": {"ir_symbol_id": "ir:123"},
            },
            "semantic_score": 0.8,
            "keyword_score": 0.5,
            "pseudocode_score": 0.0,
            "graph_score": 0.0,
            "total_score": 0.8,
        }
        hit = Hit.from_retrieval_row(row)
        assert hit.metadata.get("ir_symbol_id") == "ir:123"


class TestFusionConfig:
    def test_from_dict(self):
        cfg = FusionConfig.from_dict({
            "alpha_base": 0.8,
            "alpha_min": 0.25,
            "alpha_max": 0.9,
            "rrf_k_base": 60,
            "rrf_k_min": 20,
            "rrf_k_max": 100,
        })
        assert cfg.alpha_base == 0.8
        assert cfg.rrf_k_base == 60

    def test_defaults(self):
        cfg = FusionConfig.from_dict({})
        assert cfg.alpha_base == 0.8
        assert cfg.rrf_k_base == 60

    def test_frozen(self):
        cfg = FusionConfig.from_dict({})
        with pytest.raises(FrozenInstanceError):
            cfg.alpha_base = 0.5  # type: ignore[misc]


class TestIterationState:
    def test_initial_state(self):
        state = IterationState(
            round_num=1,
            elements=(),
            history=(),
            tool_call_history=(),
        )
        assert state.round_num == 1
        assert len(state.elements) == 0

    def test_with_new_elements(self):
        state = IterationState(round_num=1, elements=(), history=(), tool_call_history=())
        hits = (Hit(element_id="a", score=0.9, source="semantic"),)
        updated = state.with_elements(hits)
        assert len(updated.elements) == 1
        assert len(state.elements) == 0  # original unchanged

    def test_with_history_entry(self):
        state = IterationState(round_num=1, elements=(), history=(), tool_call_history=())
        entry = IterationHistoryEntry(
            round=1, confidence=50, query_complexity=40,
            elements_count=5, total_lines=500,
            confidence_gain=0.0, lines_added=500, roi=0.0,
            budget_usage_pct=4.0,
        )
        updated = state.with_history_entry(entry)
        assert len(updated.history) == 1
        assert len(state.history) == 0


class TestRoundResult:
    def test_construction(self):
        r = RoundResult(
            confidence=75,
            tool_calls=(),
            keep_files=(),
            reasoning="test",
        )
        assert r.confidence == 75
        assert len(r.tool_calls) == 0


class TestToolCall:
    def test_construction(self):
        tc = ToolCall(tool="search_codebase", parameters={"search_term": "foo"})
        assert tc.tool == "search_codebase"

    def test_frozen(self):
        tc = ToolCall(tool="search_codebase", parameters={})
        with pytest.raises(FrozenInstanceError):
            tc.tool = "other"  # type: ignore[misc]


class TestGenerationResult:
    def test_construction(self):
        r = GenerationResult(
            answer="test answer",
            sources=(SourceRef(path="foo.py", name="func", line=10),),
            prompt_tokens=100,
        )
        assert r.answer == "test answer"
        assert len(r.sources) == 1


class TestFileAnalysis:
    def test_construction(self):
        fa = FileAnalysis(
            total_files=10,
            languages={"Python": 8, "TypeScript": 2},
            file_types={".py": 8, ".ts": 2},
            key_files=("setup.py", "tsconfig.json"),
        )
        assert fa.total_files == 10
        assert fa.languages["Python"] == 8


class TestRepoStructure:
    def test_construction(self):
        rs = RepoStructure(
            repo_name="my-repo",
            summary="A test repo",
            analysis=FileAnalysis(
                total_files=5,
                languages={"Python": 5},
                file_types={".py": 5},
                key_files=(),
            ),
        )
        assert rs.repo_name == "my-repo"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_types.py -v`
Expected: FAIL — `cannot import name 'Hit' from 'fastcode.core.types'`

- [ ] **Step 3: Write `fastcode/core/types.py`**

```python
"""Frozen dataclasses for core logic — no I/O imports, no Pydantic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Retrieval types (replaces Dict[str, Any] from retriever.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Hit:
    """A single retrieval result with scores and provenance."""
    element_id: str
    element_type: str
    element_name: str
    score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    pseudocode_score: float = 0.0
    graph_score: float = 0.0
    total_score: float = 0.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    projected_only: bool = False
    llm_selected: bool = False
    agent_found: bool = False

    @classmethod
    def from_retrieval_row(cls, row: dict[str, Any]) -> Hit:
        elem = row.get("element", {})
        return cls(
            element_id=elem.get("id", ""),
            element_type=elem.get("type", ""),
            element_name=elem.get("name", ""),
            score=row.get("total_score", 0.0),
            semantic_score=row.get("semantic_score", 0.0),
            keyword_score=row.get("keyword_score", 0.0),
            pseudocode_score=row.get("pseudocode_score", 0.0),
            graph_score=row.get("graph_score", 0.0),
            total_score=row.get("total_score", 0.0),
            metadata=elem.get("metadata", {}),
            projected_only=row.get("projected_only", False),
            llm_selected=row.get("llm_file_selected", False),
            agent_found=row.get("agent_found", False),
        )

    def to_retrieval_row(self, element_extra: dict[str, Any] | None = None) -> dict[str, Any]:
        elem: dict[str, Any] = {
            "id": self.element_id,
            "type": self.element_type,
            "name": self.element_name,
        }
        if self.metadata:
            elem["metadata"] = dict(self.metadata)
        if element_extra:
            elem.update(element_extra)
        return {
            "element": elem,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "pseudocode_score": self.pseudocode_score,
            "graph_score": self.graph_score,
            "total_score": self.total_score,
            "projected_only": self.projected_only,
            "llm_file_selected": self.llm_selected,
            "agent_found": self.agent_found,
        }


@dataclass(frozen=True)
class FusionConfig:
    """Adaptive fusion parameters — extracted from retriever config dict."""
    alpha_base: float = 0.8
    alpha_min: float = 0.25
    alpha_max: float = 0.9
    rrf_k_base: int = 60
    rrf_k_min: int = 20
    rrf_k_max: int = 100

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FusionConfig:
        return cls(
            alpha_base=d.get("alpha_base", 0.8),
            alpha_min=d.get("alpha_min", 0.25),
            alpha_max=d.get("alpha_max", 0.9),
            rrf_k_base=d.get("rrf_k_base", 60),
            rrf_k_min=d.get("rrf_k_min", 20),
            rrf_k_max=d.get("rrf_k_max", 100),
        )


@dataclass(frozen=True)
class FusionWeights:
    """Weights for cross-collection fusion."""
    code_weight: float = 0.7
    doc_weight: float = 0.3
    alpha: float = 0.8
    beta: float = 0.35
    rrf_k_code: float = 60.0
    rrf_k_doc: float = 60.0


@dataclass(frozen=True)
class RetrievalChannelOutput:
    """Output from a single retrieval channel (code or doc)."""
    collection: str
    semantic_results: tuple[tuple[dict[str, Any], float], ...]
    keyword_results: tuple[tuple[dict[str, Any], float], ...]
    pseudocode_results: tuple[tuple[dict[str, Any], float], ...] = ()
    ranked_results: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ElementFilter:
    """Filter criteria for retrieval results."""
    language: str | None = None
    element_type: str | None = None
    file_path: str | None = None
    snapshot_id: str | None = None


# ---------------------------------------------------------------------------
# Iteration types (replaces Dict[str, Any] from iterative_agent.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IterationConfig:
    """Adaptive iteration parameters — computed once from query complexity."""
    base_max_iterations: int = 4
    base_confidence_threshold: int = 95
    min_confidence_gain: float = 0.5
    max_total_lines: int = 12000
    max_iterations: int = 4
    confidence_threshold: int = 95
    adaptive_line_budget: int = 12000
    max_elements: int = 30
    max_candidates_display: int = 40
    temperature: float = 0.3
    max_tokens: int = 4096


@dataclass(frozen=True)
class IterationHistoryEntry:
    """One round of iteration history."""
    round: int
    confidence: int
    query_complexity: int
    elements_count: int
    total_lines: int
    confidence_gain: float
    lines_added: int
    roi: float
    budget_usage_pct: float


@dataclass(frozen=True)
class ToolCall:
    """A tool call from the LLM during iteration."""
    tool: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundResult:
    """Parsed result from one LLM round."""
    confidence: int
    tool_calls: tuple[ToolCall, ...]
    keep_files: tuple[str, ...]
    reasoning: str
    query_complexity: int | None = None
    should_answer_directly: bool = False


@dataclass(frozen=True)
class IterationMetrics:
    """Final metrics from completed iteration."""
    rounds: int
    answered_directly: bool
    query_complexity: int
    initial_confidence: int
    final_confidence: int
    confidence_gain: int
    total_elements: int
    total_lines: int
    budget_used_pct: float
    iterations_used_pct: float
    overall_roi: float
    round_efficiencies: tuple[dict[str, Any], ...]
    adaptive_params: dict[str, Any]
    stopping_reason: str
    efficiency_rating: str


@dataclass(frozen=True)
class IterationState:
    """Immutable state threaded through iteration rounds."""
    round_num: int
    elements: tuple[Hit, ...]
    history: tuple[IterationHistoryEntry, ...]
    tool_call_history: tuple[ToolCall, ...]
    retained_elements: tuple[Hit, ...] = ()
    pending_elements: tuple[Hit, ...] = ()
    confidence: int = 0
    dialogue_history: tuple[dict[str, Any], ...] = ()

    def with_elements(self, new_elements: tuple[Hit, ...]) -> IterationState:
        return IterationState(
            round_num=self.round_num,
            elements=new_elements,
            history=self.history,
            tool_call_history=self.tool_call_history,
            retained_elements=self.retained_elements,
            pending_elements=self.pending_elements,
            confidence=self.confidence,
            dialogue_history=self.dialogue_history,
        )

    def with_history_entry(self, entry: IterationHistoryEntry) -> IterationState:
        return IterationState(
            round_num=self.round_num,
            elements=self.elements,
            history=self.history + (entry,),
            tool_call_history=self.tool_call_history,
            retained_elements=self.retained_elements,
            pending_elements=self.pending_elements,
            confidence=self.confidence,
            dialogue_history=self.dialogue_history,
        )

    def with_tool_calls(self, calls: tuple[ToolCall, ...]) -> IterationState:
        return IterationState(
            round_num=self.round_num,
            elements=self.elements,
            history=self.history,
            tool_call_history=self.tool_call_history + calls,
            retained_elements=self.retained_elements,
            pending_elements=self.pending_elements,
            confidence=self.confidence,
            dialogue_history=self.dialogue_history,
        )

    def next_round(self) -> IterationState:
        return IterationState(
            round_num=self.round_num + 1,
            elements=self.elements,
            history=self.history,
            tool_call_history=self.tool_call_history,
            retained_elements=self.retained_elements,
            pending_elements=self.pending_elements,
            confidence=self.confidence,
            dialogue_history=self.dialogue_history,
        )


# ---------------------------------------------------------------------------
# Generation types (replaces Dict[str, Any] from answer_generator.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationInput:
    """Structured input for answer generation."""
    query: str
    context: str
    prompt_tokens: int
    max_tokens: int
    dialogue_history: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SourceRef:
    """A source reference in a generated answer."""
    path: str
    name: str
    line: int = 0
    element_type: str = ""
    repo_name: str = ""


@dataclass(frozen=True)
class GenerationResult:
    """Structured output from answer generation."""
    answer: str
    sources: tuple[SourceRef, ...]
    prompt_tokens: int
    summary: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Repo analysis types (replaces Dict[str, Any] from repo_overview.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileAnalysis:
    """Analysis of repository file structure."""
    total_files: int
    languages: dict[str, int] = field(default_factory=dict)
    file_types: dict[str, int] = field(default_factory=dict)
    key_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepoStructure:
    """Complete repository structure overview."""
    repo_name: str
    summary: str
    analysis: FileAnalysis
    has_readme: bool = False
    readme_content: str | None = None
    structure_text: str | None = None


# ---------------------------------------------------------------------------
# SCIP transform types (from scip_loader.py)
# ---------------------------------------------------------------------------

class ScipKind:
    """SCIP symbol kind constants — replaces magic strings."""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    INTERFACE = "interface"
    ENUM = "enum"
    VARIABLE = "variable"
    CONSTANT = "constant"
    PROPERTY = "property"
    TYPE = "type"
    UNKNOWN = "unknown"


class ScipRole:
    """SCIP symbol role constants."""
    DEFINITION = "definition"
    REFERENCE = "reference"
    IMPORT = "import"
    WRITE_ACCESS = "write_access"
    FORWARD_DEFINITION = "forward_definition"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_core_types.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add fastcode/core/__init__.py fastcode/core/types.py tests/test_core_types.py
git commit -m "feat: add core types package with frozen dataclasses for FP refactoring"
```

---

### Task 0.3: Add core/ I/O import guard and Pydantic boundary helpers

**Files:**
- Modify: `fastcode/core/__init__.py`
- Create: `fastcode/core/boundary.py`
- Create: `tests/test_core_boundary.py`

- [ ] **Step 1: Write failing test for I/O import guard**

```python
# tests/test_core_boundary.py
"""Verify core/ has zero I/O imports and boundary conversion works."""
import ast
import importlib
from pathlib import Path

IO_MODULES = frozenset({
    "psycopg", "sqlite3", "openai", "anthropic", "ollama",
    "requests", "urllib", "http", "subprocess", "pathlib",
    "dotenv", "torch", "sentence_transformers", "tiktoken",
    "pydantic",  # Rule 1: Pydantic Stops at the Door
})


def test_core_modules_have_no_io_imports():
    """Every .py file in core/ must not import any I/O module."""
    core_dir = Path(__file__).resolve().parent.parent / "fastcode" / "core"
    violations = []
    for py_file in core_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    if root_module in IO_MODULES:
                        violations.append(f"{py_file.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_module = node.module.split(".")[0]
                    if root_module in IO_MODULES:
                        violations.append(f"{py_file.name}: from {node.module}")
    assert violations == [], f"I/O imports found in core/:\n" + "\n".join(violations)
```

- [ ] **Step 2: Run test to verify it passes (no core/ files import I/O yet)**

Run: `uv run pytest tests/test_core_boundary.py::test_core_modules_have_no_io_imports -v`
Expected: PASS (core/ is new, no I/O imports yet)

- [ ] **Step 3: Write failing test for Pydantic boundary conversion**

```python
# In tests/test_core_boundary.py

from fastcode.core.boundary import hit_to_response, query_request_to_core


def test_hit_to_response_converts_to_dict():
    from fastcode.core.types import Hit
    hit = Hit(element_id="sym:repo:func", element_type="function", element_name="func", score=0.85, source="semantic")
    response = hit_to_response(hit)
    assert response["id"] == "sym:repo:func"
    assert response["type"] == "function"
    assert response["score"] == 0.85


def test_query_request_to_core_extracts_fields():
    request = {"question": "How does parsing work?", "repo_name": "myrepo", "branch": "main"}
    core_input = query_request_to_core(request)
    assert core_input.question == "How does parsing work?"
    assert core_input.repo_name == "myrepo"
```

- [ ] **Step 4: Implement boundary conversion functions**

```python
# fastcode/core/boundary.py
"""Conversion between API boundary types and core types.

Pydantic models at API edges → frozen dataclasses for core logic.
These functions bridge the gap without requiring Pydantic in core/.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastcode.core.types import Hit


@dataclass(frozen=True)
class CoreQueryInput:
    """Internal representation of a query request."""
    question: str
    repo_name: str | None = None
    branch: str | None = None
    snapshot_id: str | None = None
    session_id: str | None = None


def query_request_to_core(request: dict[str, Any]) -> CoreQueryInput:
    """Convert API request dict to core query input."""
    return CoreQueryInput(
        question=request["question"],
        repo_name=request.get("repo_name"),
        branch=request.get("branch"),
        snapshot_id=request.get("snapshot_id"),
        session_id=request.get("session_id"),
    )


def hit_to_response(hit: Hit) -> dict[str, Any]:
    """Convert core Hit to API response dict."""
    return {
        "id": hit.element_id,
        "type": hit.element_type,
        "name": hit.element_name,
        "score": hit.score,
        "source": hit.source,
    }
```

- [ ] **Step 5: Run all boundary tests**

Run: `uv run pytest tests/test_core_boundary.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add fastcode/core/__init__.py fastcode/core/boundary.py tests/test_core_boundary.py
git commit -m "feat: add I/O import guard and boundary conversion helpers"
```

---

## Phase 1: Extract Pure Retrieval Logic (from retriever.py)

**Goal:** Extract the 27 pure methods from `HybridRetriever` into `core/` modules. Each extraction follows: write test against pure function → verify it passes → wire into existing class. The existing test suite (`test_adaptive_fusion.py`, `test_doc_channel_projection.py`) continues to pass because the class delegates to the same pure functions.

This is the highest-impact phase — it eliminates `HybridRetriever.__new__()` bypasses in tests and removes the need for `_mk_retriever()` fakes.

### Task 1.1: Extract scoring functions (static methods from retriever.py)

**Files:**
- Create: `fastcode/core/scoring.py`
- Create: `tests/test_core_scoring.py`
- Modify: `fastcode/retriever.py` (delegate static methods to core)

The following static methods from `HybridRetriever` are pure and can be extracted:
- `_normalized_totals` (line 233-247)
- `_sigmoid` (line 1158-1161)
- `_tokenize_signal` (line 1164-1165)
- `_normalized_query_entropy` (line 1168-1182)
- `_weighted_keyword_affinity` (line 1185-1199)
- `_trace_confidence_weight` (line 329-338)
- `_clone_result_row` (line 226-230)

- [ ] **Step 1: Write failing tests for each scoring function**

```python
# tests/test_core_scoring.py
"""Tests for pure scoring functions extracted from retriever."""
import math

from fastcode.core.scoring import (
    normalized_totals,
    sigmoid,
    tokenize_signal,
    normalized_query_entropy,
    weighted_keyword_affinity,
    trace_confidence_weight,
    clone_result_row,
)


def _mk_row(elem_id: str, total: float) -> dict:
    return {
        "element": {"id": elem_id, "type": "function", "name": elem_id},
        "semantic_score": total,
        "keyword_score": total * 0.5,
        "pseudocode_score": 0.0,
        "graph_score": 0.0,
        "total_score": total,
    }


class TestNormalizedTotals:
    def test_basic(self):
        rows = [_mk_row("a", 0.8), _mk_row("b", 0.5)]
        totals = normalized_totals(rows)
        assert abs(totals["semantic"] - 1.3) < 1e-9
        assert abs(totals["keyword"] - 0.65) < 1e-9

    def test_empty(self):
        totals = normalized_totals([])
        assert totals["semantic"] == 0.0
        assert totals["keyword"] == 0.0


class TestSigmoid:
    def test_midpoint(self):
        result = sigmoid(0.0)
        assert abs(result - 0.5) < 1e-9

    def test_high(self):
        result = sigmoid(10.0)
        assert result > 0.99

    def test_low(self):
        result = sigmoid(-10.0)
        assert result < 0.01


class TestTokenizeSignal:
    def test_basic(self):
        tokens = tokenize_signal("hello world foo")
        assert tokens == ["hello", "world", "foo"]

    def test_empty(self):
        assert tokenize_signal("") == []


class TestNormalizedQueryEntropy:
    def test_uniform_distribution(self):
        tokens = ["a", "b", "c", "d"]
        entropy = normalized_query_entropy(tokens)
        assert 0.0 <= entropy <= 1.0
        assert entropy > 0.5  # uniform is high entropy

    def test_single_token(self):
        entropy = normalized_query_entropy(["only"])
        assert entropy == 0.0  # no entropy with one token


class TestWeightedKeywordAffinity:
    def test_basic(self):
        tokens = ["function", "class", "unknown"]
        weights = {"function": 1.0, "class": 0.8}
        affinity = weighted_keyword_affinity(tokens, weights)
        assert affinity > 0.0

    def test_no_match(self):
        tokens = ["foo", "bar"]
        weights = {"function": 1.0}
        affinity = weighted_keyword_affinity(tokens, weights)
        assert affinity == 0.0


class TestTraceConfidenceWeight:
    def test_precise(self):
        assert trace_confidence_weight("precise") == 1.0

    def test_resolved(self):
        assert trace_confidence_weight("resolved") == 0.8

    def test_heuristic(self):
        assert trace_confidence_weight("heuristic") == 0.5

    def test_unknown(self):
        assert trace_confidence_weight("unknown") == 0.3

    def test_none(self):
        assert trace_confidence_weight(None) == 0.3


class TestCloneResultRow:
    def test_deep_copy(self):
        row = _mk_row("a", 0.8)
        cloned = clone_result_row(row)
        cloned["element"]["id"] = "changed"
        assert row["element"]["id"] == "a"  # original unchanged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_scoring.py -v`
Expected: FAIL — `cannot import name 'normalized_totals'`

- [ ] **Step 3: Write the scoring module**

Read the exact implementations from `fastcode/retriever.py` lines 226-247, 329-338, 1158-1199 and extract them as standalone functions into `fastcode/core/scoring.py`. Each function takes only its data arguments (no `self`).

```python
# fastcode/core/scoring.py
"""Pure scoring and normalization functions — extracted from retriever.py."""
from __future__ import annotations

import math
from typing import Any


def clone_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """Deep-clone a retrieval result row."""
    import copy
    return copy.deepcopy(row)


def normalized_totals(results: list[dict[str, Any]]) -> dict[str, float]:
    """Sum and normalize scores across retrieval results.

    Returns dict with 'semantic', 'keyword', 'pseudocode', 'graph', 'total' keys.
    """
    totals: dict[str, float] = {
        "semantic": 0.0,
        "keyword": 0.0,
        "pseudocode": 0.0,
        "graph": 0.0,
        "total": 0.0,
    }
    for r in results:
        totals["semantic"] += r.get("semantic_score", 0.0)
        totals["keyword"] += r.get("keyword_score", 0.0)
        totals["pseudocode"] += r.get("pseudocode_score", 0.0)
        totals["graph"] += r.get("graph_score", 0.0)
        totals["total"] += r.get("total_score", 0.0)
    return totals


def sigmoid(x: float) -> float:
    """Standard sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def tokenize_signal(text: str) -> list[str]:
    """Tokenize text into lowercase words for signal matching."""
    return text.lower().split() if text else []


def normalized_query_entropy(tokens: list[str]) -> float:
    """Compute normalized Shannon entropy of token distribution.

    Returns value in [0, 1] where 1 = uniform distribution, 0 = single token.
    """
    if len(tokens) <= 1:
        return 0.0
    from collections import Counter
    counts = Counter(tokens)
    total = len(tokens)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def weighted_keyword_affinity(
    tokens: list[str],
    weights: dict[str, float],
) -> float:
    """Compute weighted keyword affinity score.

    Sums weights for tokens that appear in the weights dict.
    """
    if not tokens or not weights:
        return 0.0
    return sum(weights.get(t, 0.0) for t in tokens)


def trace_confidence_weight(confidence: str | None) -> float:
    """Map a trace confidence label to a numeric weight."""
    return {
        "precise": 1.0,
        "resolved": 0.8,
        "heuristic": 0.5,
    }.get(confidence or "", 0.3)
```

- [ ] **Step 4: Run core scoring tests**

Run: `uv run pytest tests/test_core_scoring.py -v`
Expected: All PASS

- [ ] **Step 5: Wire `HybridRetriever` static methods to delegate to core**

In `fastcode/retriever.py`, change each static method to delegate to the core function. The method signatures stay the same, so all existing callers and tests continue to work.

```python
# In retriever.py, at the top add:
from fastcode.core import scoring as _scoring

# Then replace each static method body:
@staticmethod
def _normalized_totals(results):
    return _scoring.normalized_totals(results)

@staticmethod
def _sigmoid(x):
    return _scoring.sigmoid(x)

# ... etc for each static method
```

- [ ] **Step 6: Run existing tests to verify nothing broke**

Run: `uv run pytest tests/test_adaptive_fusion.py tests/test_doc_channel_projection.py -v`
Expected: All PASS (existing tests use class methods, which now delegate to core)

- [ ] **Step 7: Commit**

```bash
git add fastcode/core/scoring.py tests/test_core_scoring.py fastcode/retriever.py
git commit -m "feat: extract pure scoring functions to core/scoring.py"
```

---

### Task 1.2: Extract fusion functions (from retriever.py)

**Files:**
- Create: `fastcode/core/fusion.py`
- Create: `tests/test_core_fusion.py`
- Modify: `fastcode/retriever.py`

Extract these methods:
- `_compute_adaptive_fusion_params` (line 1201-1303) — pure math computing alpha, beta, rrf_k
- `_adaptive_fuse_channels` (line 1071-1131) — RRF fusion algorithm
- `_new_fused_entry` (line 1134-1143)
- `_ensure_fused_entry` (line 1145-1155)
- `_project_doc_priors` (line 364-411) — doc-to-code projection scoring
- `_apply_doc_projection_to_code` (line 431-501) — projection application

This is the highest-value extraction — the existing `test_adaptive_fusion.py` and `test_doc_channel_projection.py` both use `HybridRetriever.__new__()` to bypass `__init__` just to test these pure functions. After extraction, tests can call functions directly.

- [ ] **Step 1: Write failing tests for fusion functions**

```python
# tests/test_core_fusion.py
"""Tests for pure fusion functions — zero test doubles needed."""
from fastcode.core.types import FusionConfig
from fastcode.core.fusion import (
    compute_adaptive_fusion_params,
    adaptive_fuse_channels,
    project_doc_priors,
    apply_doc_projection_to_code,
)


def _mk_row(elem_id: str, elem_type: str, total: float, **extra) -> dict:
    row = {
        "element": {"id": elem_id, "type": elem_type, "name": elem_id},
        "semantic_score": total,
        "keyword_score": total * 0.5,
        "pseudocode_score": 0.0,
        "graph_score": 0.0,
        "total_score": total,
    }
    row.update(extra)
    return row


def _default_fusion_config() -> FusionConfig:
    return FusionConfig.from_dict({
        "alpha_base": 0.8,
        "alpha_min": 0.25,
        "alpha_max": 0.9,
        "rrf_k_base": 60,
        "rrf_k_min": 20,
        "rrf_k_max": 100,
    })


class TestComputeAdaptiveFusionParams:
    def test_design_intent_queries_lower_alpha(self):
        code_rows = [_mk_row("code:1", "function", 0.45)]
        doc_rows = [_mk_row("doc:1", "design_document", 0.82)]
        alpha, beta, rrf_k = compute_adaptive_fusion_params(
            query="What is the architecture rationale?",
            query_info={"intent": "design_rationale"},
            code_results=code_rows,
            doc_results=doc_rows,
            config=_default_fusion_config(),
        )
        assert alpha < 0.8  # doc-oriented queries get lower alpha

    def test_code_intent_queries_keep_alpha(self):
        code_rows = [_mk_row("code:1", "function", 0.9)]
        doc_rows = []
        alpha, _, _ = compute_adaptive_fusion_params(
            query="How does the parser work?",
            query_info={"intent": "code_navigation"},
            code_results=code_rows,
            doc_results=doc_rows,
            config=_default_fusion_config(),
        )
        assert alpha >= 0.7  # code queries keep high alpha


class TestAdaptiveFuseChannels:
    def test_fuses_code_and_doc_results(self):
        code_rows = [_mk_row("code:1", "function", 0.8)]
        doc_rows = [_mk_row("doc:1", "design_document", 0.9)]
        fused = adaptive_fuse_channels(
            query="test query",
            query_info={},
            code_results=code_rows,
            doc_results=doc_rows,
            config=_default_fusion_config(),
        )
        ids = [r["element"]["id"] for r in fused]
        assert "code:1" in ids
        assert "doc:1" in ids

    def test_returns_sorted_by_score(self):
        code_rows = [
            _mk_row("code:1", "function", 0.5),
            _mk_row("code:2", "function", 0.9),
        ]
        doc_rows = []
        fused = adaptive_fuse_channels(
            query="test",
            query_info={},
            code_results=code_rows,
            doc_results=doc_rows,
            config=_default_fusion_config(),
        )
        assert fused[0]["element"]["id"] == "code:2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_fusion.py -v`
Expected: FAIL — `cannot import name 'compute_adaptive_fusion_params'`

- [ ] **Step 3: Read the exact implementations from retriever.py**

Read `fastcode/retriever.py` lines 1071-1303 (`_adaptive_fuse_channels`, `_compute_adaptive_fusion_params`, helper methods). Extract them as standalone functions that take `FusionConfig` as a parameter instead of reading `self.adaptive_fusion_cfg`.

The key transformation: `self.adaptive_fusion_cfg["alpha_base"]` → `config.alpha_base`, `self._sigmoid(x)` → `sigmoid(x)` (from scoring module).

- [ ] **Step 4: Write the fusion module**

```python
# fastcode/core/fusion.py
"""Pure fusion functions — adaptive RRF, doc projection, channel fusion."""
from __future__ import annotations

from typing import Any

from fastcode.core.scoring import sigmoid
from fastcode.core.types import FusionConfig


def compute_adaptive_fusion_params(
    *,
    query: str,
    query_info: dict[str, Any],
    code_results: list[dict[str, Any]],
    doc_results: list[dict[str, Any]],
    config: FusionConfig,
) -> tuple[float, float, float]:
    """Compute adaptive alpha, beta, and rrf_k for fusion.

    Returns (alpha, beta, rrf_k).

    Read the exact logic from fastcode/retriever.py:1201-1303 and extract
    with self.adaptive_fusion_cfg replaced by config parameter.
    """
    # Implementation will be copied from retriever._compute_adaptive_fusion_params
    # replacing self.adaptive_fusion_cfg with config, self._sigmoid with sigmoid,
    # self._tokenize_signal with tokenize_signal, etc.
    raise NotImplementedError("Copy from retriever.py during implementation")


def adaptive_fuse_channels(
    *,
    query: str,
    query_info: dict[str, Any],
    code_results: list[dict[str, Any]],
    doc_results: list[dict[str, Any]],
    config: FusionConfig,
) -> list[dict[str, Any]]:
    """Fuse code and doc channels using adaptive RRF.

    Read from fastcode/retriever.py:1071-1131.
    """
    raise NotImplementedError("Copy from retriever.py during implementation")


def project_doc_priors(
    *,
    query: str,
    query_info: dict[str, Any],
    doc_results: list[dict[str, Any]],
    config: FusionConfig,
) -> dict[str, Any]:
    """Project doc scores to code symbols via grounded mentions.

    Read from fastcode/retriever.py:364-411.
    """
    raise NotImplementedError("Copy from retriever.py during implementation")


def apply_doc_projection_to_code(
    *,
    query: str,
    query_info: dict[str, Any],
    code_results: list[dict[str, Any]],
    doc_results: list[dict[str, Any]],
    config: FusionConfig,
    bm25_elements: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Apply doc projection priors to code results.

    Read from fastcode/retriever.py:431-501.
    """
    raise NotImplementedError("Copy from retriever.py during implementation")
```

Note: During implementation, copy the exact logic from retriever.py, replacing:
- `self.adaptive_fusion_cfg` → `config` parameter
- `self._sigmoid()` → `sigmoid()` from core.scoring
- `self._tokenize_signal()` → `tokenize_signal()` from core.scoring
- `self._normalized_query_entropy()` → `normalized_query_entropy()` from core.scoring
- `self._weighted_keyword_affinity()` → `weighted_keyword_affinity()` from core.scoring
- `self._normalized_totals()` → `normalized_totals()` from core.scoring
- `self._trace_confidence_weight()` → `trace_confidence_weight()` from core.scoring
- `self._clone_result_row()` → `clone_result_row()` from core.scoring
- `self._last_fusion_debug` → return value or drop (debug only)
- `self.filtered_bm25_elements` / `self.full_bm25_elements` → `bm25_elements` parameter

- [ ] **Step 5: Run core fusion tests**

Run: `uv run pytest tests/test_core_fusion.py -v`
Expected: All PASS

- [ ] **Step 6: Wire HybridRetriever methods to delegate to core fusion**

In `fastcode/retriever.py`, update `_compute_adaptive_fusion_params`, `_adaptive_fuse_channels`, `_project_doc_priors`, `_apply_doc_projection_to_code` to call the core functions, passing `self.adaptive_fusion_cfg` as `FusionConfig.from_dict(self.adaptive_fusion_cfg)`.

- [ ] **Step 7: Run existing retriever tests**

Run: `uv run pytest tests/test_adaptive_fusion.py tests/test_doc_channel_projection.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add fastcode/core/fusion.py tests/test_core_fusion.py fastcode/retriever.py
git commit -m "feat: extract pure fusion functions to core/fusion.py"
```

---

### Task 1.3: Extract filtering functions (from retriever.py)

**Files:**
- Create: `fastcode/core/filtering.py`
- Create: `tests/test_core_filtering.py`
- Modify: `fastcode/retriever.py`

Extract these pure methods:
- `_apply_filters` (line 1713-1744)
- `_diversify` (line 1746-1773)
- `_final_repo_filter` (line 1775-1812)
- `_rerank` (line 1687-1711)

Follow the same pattern: write tests, extract to core, wire class to delegate.

- [ ] **Step 1-7: Same TDD pattern as Tasks 1.1 and 1.2**

Write tests against pure functions, implement, wire retriever to delegate, verify existing tests pass.

- [ ] **Step 8: Commit**

```bash
git add fastcode/core/filtering.py tests/test_core_filtering.py fastcode/retriever.py
git commit -m "feat: extract pure filtering functions to core/filtering.py"
```

---

### Task 1.4: Extract combination functions (from retriever.py)

**Files:**
- Create: `fastcode/core/combination.py`
- Create: `tests/test_core_combination.py`
- Modify: `fastcode/retriever.py`

Extract:
- `_combine_results` (line 1464-1547) — merges semantic + keyword + pseudocode results

Follow same TDD pattern.

- [ ] **Step 1-7: TDD cycle**

- [ ] **Step 8: Commit**

```bash
git add fastcode/core/combination.py tests/test_core_combination.py fastcode/retriever.py
git commit -m "feat: extract pure combination functions to core/combination.py"
```

---

## Phase 2: Extract Pure Iteration Logic (from iterative_agent.py)

**Goal:** Extract ~50 pure methods from `IterativeAgent` into `core/iteration.py`, `core/prompts.py`, and `core/parsing.py`. Only 7 I/O methods remain in the class.

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
- `_initialize_adaptive_parameters` (line 109-152) — threshold computation
- `_determine_stopping_reason` (line 420-432)
- `_rate_efficiency` (line 434-444)
- `_generate_iteration_metadata` (line 345-418)

All of these are pure math/decision logic. They read from `self` only to access config values or history — pass those as parameters instead.

- [ ] **Step 1: Write tests for iteration functions**

```python
# tests/test_core_iteration.py
"""Tests for pure iteration control functions."""
from fastcode.core.iteration import (
    initialize_adaptive_parameters,
    should_continue_iteration,
    get_min_roi_threshold,
    calculate_repo_factor,
    calculate_total_lines,
    determine_stopping_reason,
    rate_efficiency,
)
from fastcode.core.types import IterationConfig, IterationHistoryEntry


class TestInitializeAdaptiveParameters:
    def test_simple_query_reduces_iterations(self):
        params = initialize_adaptive_parameters(
            query_complexity=20,
            repo_factor=1.0,
            config=IterationConfig(),
        )
        assert params.max_iterations <= 4

    def test_complex_query_increases_iterations(self):
        params = initialize_adaptive_parameters(
            query_complexity=90,
            repo_factor=1.5,
            config=IterationConfig(),
        )
        assert params.max_iterations >= 3

    def test_threshold_adjusts_for_complex_queries(self):
        params = initialize_adaptive_parameters(
            query_complexity=85,
            repo_factor=1.0,
            config=IterationConfig(base_confidence_threshold=95),
        )
        assert params.confidence_threshold < 95


class TestShouldContinueIteration:
    def test_stops_when_confidence_exceeds_threshold(self):
        assert not should_continue_iteration(
            confidence=96,
            current_round=2,
            max_iterations=4,
            total_lines=5000,
            line_budget=12000,
            confidence_threshold=95,
            history=(),
            min_confidence_gain=0.5,
        )

    def test_stops_at_max_iterations(self):
        assert not should_continue_iteration(
            confidence=50,
            current_round=5,
            max_iterations=4,
            total_lines=5000,
            line_budget=12000,
            confidence_threshold=95,
            history=(),
            min_confidence_gain=0.5,
        )

    def test_continues_when_below_threshold(self):
        assert should_continue_iteration(
            confidence=60,
            current_round=2,
            max_iterations=4,
            total_lines=5000,
            line_budget=12000,
            confidence_threshold=95,
            history=(),
            min_confidence_gain=0.5,
        )


class TestGetMinRoiThreshold:
    def test_high_complexity_lower_threshold(self):
        roi = get_min_roi_threshold(query_complexity=90, current_confidence=60)
        assert roi < 2.0

    def test_low_complexity_higher_threshold(self):
        roi = get_min_roi_threshold(query_complexity=20, current_confidence=60)
        assert roi >= 1.5


class TestCalculateRepoFactor:
    def test_small_repo(self):
        factor = calculate_repo_factor(total_files=10, avg_file_lines=50, max_depth=2)
        assert 0.5 <= factor <= 2.0

    def test_large_repo(self):
        factor = calculate_repo_factor(total_files=500, avg_file_lines=300, max_depth=8)
        assert factor > 0.5


class TestDetermineStoppingReason:
    def test_confidence_reached(self):
        reason = determine_stopping_reason(
            confidence=96, threshold=95, current_round=3, max_iterations=4
        )
        assert "confidence" in reason.lower() or "threshold" in reason.lower()

    def test_max_iterations(self):
        reason = determine_stopping_reason(
            confidence=60, threshold=95, current_round=4, max_iterations=4
        )
        assert "iteration" in reason.lower() or "max" in reason.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_iteration.py -v`
Expected: FAIL

- [ ] **Step 3: Read exact implementations from iterative_agent.py lines 109-152, 345-444, 2268-2470 and extract as standalone functions**

Each function takes config values as explicit parameters instead of reading `self.*`. The `IterationConfig` frozen dataclass bundles the parameters.

- [ ] **Step 4: Write `fastcode/core/iteration.py`**

Copy exact logic from iterative_agent.py, replacing `self.*` references with function parameters. Use `IterationConfig` and `IterationHistoryEntry` types from core.types.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_core_iteration.py -v`
Expected: All PASS

- [ ] **Step 6: Wire IterativeAgent to delegate to core**

In `iterative_agent.py`, update methods to call core functions, passing `self.*` values as arguments.

- [ ] **Step 7: Run existing iterative_agent tests**

Run: `uv run pytest tests/ -k "iterat" -v`
Expected: All PASS

- [ ] **Step 8: Commit**

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

Extract the pure prompt construction methods:
- `_build_round_one_prompt` (line 473-589)
- `_build_round_n_prompt` (line 1311-1472)
- `_build_element_selection_prompt` (line 973-1016)
- `_format_elements_with_metadata` (line 1730-1795)
- `_format_candidates_with_elements` (line 1018-1052)
- `_format_element_list` (line 2131-2147)
- `_format_tool_call_history` (line 1714-1728)

These are all pure string formatting — no I/O, no self mutation. Read exact code from iterative_agent.py, replace `self.max_candidates_display` etc. with config parameters.

- [ ] **Step 1-7: TDD cycle — write tests, extract, wire, verify**

- [ ] **Step 8: Commit**

```bash
git add fastcode/core/prompts.py tests/test_core_prompts.py fastcode/iterative_agent.py
git commit -m "feat: extract pure prompt construction to core/prompts.py"
```

---

### Task 2.3: Extract parsing functions

**Files:**
- Create: `fastcode/core/parsing.py`
- Create: `tests/test_core_parsing.py`
- Modify: `fastcode/iterative_agent.py`

Extract the pure parsing methods:
- `_parse_round_one_response` (line 591-657)
- `_parse_round_n_response` (line 1797-1841)
- `_parse_element_selection_response` (line 1054-1063)
- `_normalize_query_enhancement` (line 659-712)
- `_parse_query_enhancement_fallback` (line 714-783)
- `_extract_json_from_response` (line 2590-2646)
- `_sanitize_json_string` (line 2648-2726)
- `_remove_json_comments` (line 2728-2777)
- `_robust_json_parse` (line 2779-2841)

These are all pure string/JSON parsing — no I/O. Read exact code, extract as standalone functions.

- [ ] **Step 1-7: TDD cycle**

- [ ] **Step 8: Commit**

```bash
git add fastcode/core/parsing.py tests/test_core_parsing.py fastcode/iterative_agent.py
git commit -m "feat: extract pure JSON/parsing functions to core/parsing.py"
```

---

## Phase 3: Extract Pure Generation Logic (from answer_generator.py)

**Goal:** Extract the ~60% pure logic from `AnswerGenerator` into `core/context.py` and `core/summary.py`.

### Task 3.1: Extract context preparation

**Files:**
- Create: `fastcode/core/context.py`
- Create: `tests/test_core_context.py`
- Modify: `fastcode/answer_generator.py`

Extract:
- `_prepare_context` (line 447-515) — pure string formatting
- `_build_prompt` (line 517-629) — pure string construction
- `_truncate_context` (line 631-633) — pure truncation
- `_parse_response_with_summary` (line 737-768) — pure regex

- [ ] **Step 1-7: TDD cycle**

- [ ] **Step 8: Commit**

```bash
git add fastcode/core/context.py tests/test_core_context.py fastcode/answer_generator.py
git commit -m "feat: extract pure context/prompt functions to core/context.py"
```

---

### Task 3.2: Extract summary and formatting

**Files:**
- Create: `fastcode/core/summary.py`
- Create: `tests/test_core_summary.py`
- Modify: `fastcode/answer_generator.py`

Extract:
- `_generate_fallback_summary` (line 770-843)
- `_extract_sources` (line 845-861)
- `format_answer_with_sources` (line 863-888)

- [ ] **Step 1-7: TDD cycle**

- [ ] **Step 8: Commit**

```bash
git add fastcode/core/summary.py tests/test_core_summary.py fastcode/answer_generator.py
git commit -m "feat: extract pure summary/formatting functions to core/summary.py"
```

---

## Phase 4: Extract Pure Transforms from Remaining Modules

### Task 4.1: Extract projection transforms

**Files:**
- Create: `fastcode/core/projection.py`
- Create: `tests/test_core_projection.py`
- Modify: `fastcode/projection_transform.py`

`projection_transform.py` is already ~95% pure. Extract all pure methods (lines 277-1108) into core, leaving only `_llm_rewrite_summary` (line 798) as the I/O boundary. The pure methods include: `_build_weighted_graph`, `_scope_nodes`, `_prune_steiner_leaves`, `_cluster_hierarchy`, `_select_cluster_level`, `_pick_representatives`, `_build_backbone_arborescence`, `_cross_cluster_xrefs`, `_cluster_labels`, `_build_l0_summary`, `_build_l1_summary`, `_build_l2_chunks`, `_build_l2_index`, and ~10 more.

- [ ] **Step 1-7: TDD cycle**

- [ ] **Step 8: Commit**

```bash
git add fastcode/core/projection.py tests/test_core_projection.py fastcode/projection_transform.py
git commit -m "feat: extract pure projection transforms to core/projection.py"
```

---

### Task 4.2: Extract graph payload construction

**Files:**
- Create: `fastcode/core/graph_build.py`
- Create: `tests/test_core_graph_build.py`
- Modify: `fastcode/terminus_publisher.py`

Extract the pure payload construction:
- `build_code_graph_payload` (line 154)
- `build_lineage_payload` (line 251)
- `_deterministic_event_id` (line 29)

- [ ] **Step 1-7: TDD cycle**

- [ ] **Step 8: Commit**

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
- `_resolve_snapshot_ref` (line 412-458) — snapshot ID resolution
- `_build_git_meta` (line 460-473) — git metadata construction
- `_previous_snapshot_symbol_versions` (line 475-497) — symbol version mapping
- `_projection_scope_key` (line 1333-1348) — hash key computation
- `_projection_params_hash` (line 1350-1362) — parameter hashing
- `_extract_sources_from_elements` (line 2026-2040) — source extraction

- [ ] **Step 1-7: TDD cycle**

- [ ] **Step 8: Commit**

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
- `parse_file_structure` (line 128-192)
- `_get_language_from_extension` (line 194-211)
- `_is_key_file` (line 213-226)
- `_generate_structure_based_overview` (line 283-309)
- `_infer_project_type` (line 311-344)
- `_format_file_structure` (line 346-378)

- [ ] **Step 1-7: TDD cycle**

- [ ] **Step 8: Commit**

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
- `_protobuf_to_scip_index` (line 73-109)
- `_symbol_role_to_str` (line 112-122)
- `_scip_kind_to_str` (line 125-152)

- [ ] **Step 1-7: TDD cycle**

- [ ] **Step 8: Commit**

```bash
git add fastcode/core/scip_transform.py tests/test_core_scip_transform.py fastcode/scip_loader.py
git commit -m "feat: extract pure SCIP transforms to core/scip_transform.py"
```

---

## Phase 5: Create Thin Effects Layer

**Goal:** Create `effects/` modules that wrap each I/O boundary. Each function does exactly one I/O operation. **Rule 2 enforcement: all `effects/db.py` functions must return frozen dataclasses, never `dict[str, Any]`.**

### Task 5.1: Create DB effects module

**Files:**
- Create: `fastcode/effects/db.py`
- Create: `tests/test_effects_db.py`

Thin wrappers for the most common DB operations currently scattered across classes. **Critical: every function maps rows to frozen dataclasses before returning. No `dict[str, Any]` return types.**

```python
# fastcode/effects/db.py
"""Thin wrappers for database I/O — each function does one query.

Rule 2: Database Trusts Dataclasses.
Every function maps DB rows into frozen dataclasses before returning.
No dict[str, Any] returns.
"""

from typing import Any

from fastcode.core.types import Hit


def load_snapshot_record(
    conn: Any, snapshot_id: str,
) -> SnapshotRecord | None:
    """Load a snapshot record by ID. Returns frozen dataclass, not dict."""
    cursor = conn.execute(
        "SELECT snapshot_id, repo_name, branch, commit_id, tree_id FROM snapshots WHERE snapshot_id = ?",
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
        "INSERT OR REPLACE INTO snapshots (snapshot_id, repo_name, branch, commit_id, tree_id) VALUES (?, ?, ?, ?, ?)",
        (record.snapshot_id, record.repo_name, record.branch, record.commit_id, record.tree_id),
    )


def semantic_search(
    conn: Any, snapshot_id: str, query_vector: list[float], limit: int = 20,
) -> list[Hit]:
    """Run pgvector semantic search. Returns list of Hit dataclasses."""
    vector_literal = "[" + ",".join(str(v) for v in query_vector) + "]"
    cursor = conn.execute(
        """SELECT element_id, element_type, element_name,
                  1 - (embedding <=> %s::vector) as score
           FROM elements WHERE snapshot_id = %s
           ORDER BY embedding <=> %s::vector LIMIT %s""",
        (vector_literal, snapshot_id, vector_literal, limit),
    )
    return [
        Hit(
            element_id=row[0],
            element_type=row[1],
            element_name=row[2],
            score=float(row[3]),
            source="semantic",
        )
        for row in cursor.fetchall()
    ]


def keyword_search(
    conn: Any, snapshot_id: str, query_text: str, limit: int = 10,
) -> list[Hit]:
    """Run GIN full-text keyword search. Returns list of Hit dataclasses."""
    cursor = conn.execute(
        """SELECT element_id, element_type, element_name,
                  ts_rank(docs_vec, plainto_tsquery(%s)) as score
           FROM elements WHERE snapshot_id = %s
           ORDER BY ts_rank(docs_vec, plainto_tsquery(%s)) DESC LIMIT %s""",
        (query_text, snapshot_id, query_text, limit),
    )
    return [
        Hit(
            element_id=row[0],
            element_type=row[1],
            element_name=row[2],
            score=float(row[3]),
            source="keyword",
        )
        for row in cursor.fetchall()
    ]
```

Note: During implementation, you'll need to add `SnapshotRecord` and any other missing return types to `core/types.py`. The rule is: if a DB function returns data, create a frozen dataclass for it. If a DB function accepts data, accept a frozen dataclass, not a dict.

- [ ] **Step 1: Add missing dataclasses to `core/types.py`**

Add `SnapshotRecord` and any other DB-specific frozen dataclasses needed by effects/db.py:

```python
@dataclass(frozen=True)
class SnapshotRecord:
    """A snapshot metadata row from the database."""
    snapshot_id: str
    repo_name: str
    branch: str | None = None
    commit_id: str | None = None
    tree_id: str | None = None
```

- [ ] **Step 2: Write tests using in-memory SQLite**

Tests must verify that every function returns frozen dataclasses, not dicts.

```python
# tests/test_effects_db.py
"""Tests for DB effects — verify frozen dataclass returns."""
import sqlite3

from fastcode.core.types import SnapshotRecord, Hit
from fastcode.effects.db import load_snapshot_record, semantic_search


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
    conn.execute(
        "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?)",
        ("snap:test:abc123", "myrepo", "main", "abc123", "tree1"),
    )
    return conn


class TestLoadSnapshotRecord:
    def test_returns_dataclass_not_dict(self):
        conn = _setup_test_db()
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
        result = load_snapshot_record(conn, "snap:test:abc123")
        from dataclasses import FrozenInstanceError
        import pytest
        with pytest.raises(FrozenInstanceError):
            result.repo_name = "changed"  # type: ignore[misc]
```

- [ ] **Step 3: Implement effects/db.py (copy exact SQL from snapshot_store.py, pg_retrieval.py)**

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_effects_db.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add fastcode/effects/db.py tests/test_effects_db.py fastcode/core/types.py
git commit -m "feat: add thin DB effects layer — returns frozen dataclasses only"
```

---

### Task 5.2: Create LLM effects module

**Files:**
- Create: `fastcode/effects/llm.py`
- Create: `tests/test_effects_llm.py`

Thin wrappers for LLM API calls:

```python
# fastcode/effects/llm.py
"""Thin wrappers for LLM API I/O."""

from typing import Any, Iterator


def chat_completion(client: Any, *, max_tokens: int, **kwargs: Any) -> str:
    """Single LLM completion. Wraps openai_chat_completion with retry."""
    from fastcode.llm_utils import openai_chat_completion
    response = openai_chat_completion(client, max_tokens=max_tokens, **kwargs)
    return response.choices[0].message.content


def chat_completion_stream(client: Any, *, max_tokens: int, **kwargs: Any) -> Iterator[str]:
    """Streaming LLM completion. Yields text chunks."""
    # ... thin streaming wrapper
    pass
```

- [ ] **Step 1-4: TDD cycle and commit**

```bash
git add fastcode/effects/llm.py tests/test_effects_llm.py
git commit -m "feat: add thin LLM effects layer"
```

---

### Task 5.3: Create FS effects module

**Files:**
- Create: `fastcode/effects/fs.py`
- Create: `tests/test_effects_fs.py`

Thin wrappers for file I/O and git operations:

```python
# fastcode/effects/fs.py
"""Thin wrappers for file system and git I/O."""

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
    import os
    return os.path.exists(path)


def load_pickle(path: str) -> Any:
    """Load a pickle file."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str, obj: Any) -> None:
    """Save object to pickle file."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def run_scip_indexer(repo_path: str, output_path: str) -> str:
    """Run SCIP indexer subprocess."""
    from fastcode.scip_loader import run_scip_python_index
    return run_scip_python_index(repo_path, output_path)


def load_scip_artifact(path: str) -> Any:
    """Load SCIP artifact from file."""
    from fastcode.scip_loader import load_scip_artifact
    return load_scip_artifact(path)
```

- [ ] **Step 1-4: TDD cycle and commit**

```bash
git add fastcode/effects/fs.py tests/test_effects_fs.py
git commit -m "feat: add thin FS effects layer"
```

---

### Task 5.4: Create embedding effects module

**Files:**
- Create: `fastcode/effects/embedding.py`
- Create: `tests/test_effects_embedding.py`

```python
# fastcode/effects/embedding.py
"""Thin wrappers for embedding generation I/O."""

from typing import Any

import numpy as np


def embed_text(embedder: Any, text: str) -> np.ndarray:
    """Generate embedding for a single text."""
    return embedder.embed_text(text)


def embed_batch(embedder: Any, texts: list[str]) -> np.ndarray:
    """Generate embeddings for a batch of texts."""
    return embedder.embed_batch(texts)
```

- [ ] **Step 1-4: TDD cycle and commit**

```bash
git add fastcode/effects/embedding.py tests/test_effects_embedding.py
git commit -m "feat: add thin embedding effects layer"
```

---

### Task 5.5: Create graph DB effects module

**Files:**
- Create: `fastcode/effects/graph_db.py`
- Create: `tests/test_effects_graph_db.py`

Thin wrappers for TerminusDB HTTP operations:

```python
# fastcode/effects/graph_db.py
"""Thin wrappers for graph database I/O (TerminusDB, LadybugDB)."""

from typing import Any


def post_to_terminus(client: Any, payload: dict[str, Any], branch: str | None = None) -> None:
    """Post a payload to TerminusDB."""
    # Thin HTTP wrapper
    pass


def load_graph_nodes(client: Any, snapshot_id: str) -> list[dict[str, Any]]:
    """Load graph nodes from TerminusDB."""
    # Thin query wrapper
    pass


def load_graph_edges(client: Any, snapshot_id: str, edge_type: str | None = None) -> list[dict[str, Any]]:
    """Load graph edges from TerminusDB."""
    # Thin query wrapper
    pass
```

- [ ] **Step 1-4: TDD cycle and commit**

```bash
git add fastcode/effects/graph_db.py tests/test_effects_graph_db.py
git commit -m "feat: add thin graph DB effects layer"
```

---

## Phase 6: Wire Orchestrators and Validate

**Goal:** Verify that all orchestrator classes correctly delegate to `core/` for logic and `effects/` for I/O. Validate all three golden rules. Run the full test suite to ensure no regressions.

### Task 6.1: Validate Rule 1 — Pydantic Stops at the Door

- [ ] **Step 1: Run the I/O import guard test (includes pydantic check)**

Run: `uv run pytest tests/test_core_boundary.py::test_core_modules_have_no_io_imports -v`
Expected: PASS — no I/O imports (including `pydantic`) in any `core/` module

- [ ] **Step 2: Verify `core/boundary.py` is the only translation point**

Run: `grep -r "pydantic" fastcode/core/ || echo "NO PYDANTIC IN CORE - PASS"`
Expected: `NO PYDANTIC IN CORE - PASS`

---

### Task 6.2: Validate Rule 2 — Database Trusts Dataclasses

- [ ] **Step 1: Write automated check that effects/db.py never returns dict**

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
            # Check return annotation for dict[str, Any] or dict
            ret = node.returns
            if ret is None:
                continue
            ret_str = ast.dump(ret)
            if "dict" in ret_str.lower() and "Any" in ret_str:
                violations.append(f"{node.name}: return type contains dict[str, Any]")
    assert violations == [], f"Rule 2 violations in effects/db.py:\n" + "\n".join(violations)
```

Run: `uv run pytest tests/test_core_boundary.py::test_db_effects_return_dataclasses_not_dicts -v`
Expected: PASS

---

### Task 6.3: Validate Rule 3 — Explicit Translation

- [ ] **Step 1: Verify boundary.py uses explicit field mapping, no **kwargs unpacking**

Add to `tests/test_core_boundary.py`:

```python
def test_boundary_uses_explicit_translation():
    """Rule 3: boundary.py must use explicit field mapping, no ** unpacking."""
    import ast
    from pathlib import Path

    boundary = Path(__file__).resolve().parent.parent / "fastcode" / "core" / "boundary.py"
    if not boundary.exists():
        pytest.skip("core/boundary.py not yet created")

    source = boundary.read_text()
    # Reject patterns: **some_dict, **pydantic_obj.dict(), from_orm, model_dump
    forbidden = ["**kwargs", "**request", "**data", "**obj", "from_orm", "model_dump"]
    violations = []
    for pattern in forbidden:
        if pattern in source:
            violations.append(f"Found '{pattern}' — must use explicit field mapping")
    assert violations == [], f"Rule 3 violations in boundary.py:\n" + "\n".join(violations)
```

Run: `uv run pytest tests/test_core_boundary.py::test_boundary_uses_explicit_translation -v`
Expected: PASS

---

### Task 6.4: Run full test suite

- [ ] **Step 1: Run all existing tests (verify no regressions)**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All PASS (same results as before refactoring)

- [ ] **Step 2: Run all new core tests**

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
git commit -m "feat: complete FP core + thin I/O refactoring

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
Phase 0 (infrastructure)
  0.1: Package scaffolding
  0.2: Core types (frozen dataclasses)
  0.3: I/O import guard + boundary conversion

Phase 1 (retriever — highest test-double pain)
  1.1: Scoring functions
  1.2: Fusion functions
  1.3: Filtering functions
  1.4: Combination functions

Phase 2 (iterative_agent)
  2.1: Iteration control
  2.2: Prompt building
  2.3: Parsing functions

Phase 3 (answer_generator)
  3.1: Context preparation
  3.2: Summary/formatting

Phase 4 (remaining modules)
  4.1: Projection transforms
  4.2: Graph payload construction
  4.3: Snapshot pure logic
  4.4: Repo analysis
  4.5: SCIP transforms

Phase 5 (effects layer)
  5.1: DB effects
  5.2: LLM effects
  5.3: FS effects
  5.4: Embedding effects
  5.5: Graph DB effects

Phase 6 (validation)
  6.1: Validate Rule 1 — Pydantic Stops at the Door
  6.2: Validate Rule 2 — Database Trusts Dataclasses
  6.3: Validate Rule 3 — Explicit Translation
  6.4: Full test suite + coverage
  6.5: Final commit
```

### Dependency Graph

```
Phase 0 → Phase 1 (retriever) → Phase 6
         → Phase 2 (iterative_agent) → Phase 6
         → Phase 3 (answer_generator) → Phase 6
         → Phase 4 (remaining modules) → Phase 6
         → Phase 5 (effects layer, can start early)
```

Within Phase 1: Task 1.1 (scoring) must complete before 1.2 (fusion imports scoring). Tasks 1.3 and 1.4 can proceed once 1.1 is done.
Within Phase 2: Task 2.1 (iteration) is independent of 2.2 (prompts) and 2.3 (parsing).
Phases 1, 2, 3, 4 are independent of each other and can be parallelized across different worktrees.
Phase 5 (effects) can start as soon as Phase 0 is complete — it wraps existing I/O code, not core logic.
Phase 6 is the final validation gate after all other phases complete.

### Note on Migration Order

The spec lists `main.py` before `iterative_agent.py`, but the plan reverses this because `iterative_agent.py` has the highest density of extractable pure functions (50/63 methods) and its extraction reduces test-double pain in `retriever.py` (which wraps it). `main.py` query logic extraction is covered in Task 4.3 (snapshot pure logic).
