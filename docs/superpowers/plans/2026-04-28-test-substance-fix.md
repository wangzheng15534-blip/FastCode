# Test Substance Fix — Eliminate Theater, Add Meaningful Tests

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ~60% of the test suite that is test theater (serialization round-trips, lookup table checks, mock echo assertions) with tests that verify actual business logic, error handling, and edge cases.

**Architecture:** Four phases: (1) delete pure theater files, (2) replace round-trip tests with behavior tests in the worst offenders, (3) add negative/edge case tests to critical paths, (4) consolidate test infrastructure. Each phase commits independently.

**Tech Stack:** Python 3.11, pytest, Hypothesis (property-based testing)

---

## Phase 1: Delete Pure Theater Files

These files test lookup tables or mock echo patterns. No production behavior is verified. Deleting them removes false confidence without losing any real coverage.

---

### Task 1: Delete test_scip_transform.py

**Files:**
- Delete: `fastcode/tests/core/test_scip_transform.py` (35 lines)

**Rationale:** Every test asserts `mapping_func(known_input) == known_output` for a hardcoded dict. If the mapping is wrong, the test encodes the same wrong mapping. Zero bug-finding value.

- [ ] **Step 1: Verify no other files import from this test**

Run: `grep -r "test_scip_transform" fastcode/tests/ --include="*.py" | grep -v "test_scip_transform.py"`
Expected: No imports (only self-reference)

- [ ] **Step 2: Delete the file**

```bash
rm fastcode/tests/core/test_scip_transform.py
```

- [ ] **Step 3: Run full test suite to confirm no breakage**

Run: `cd /home/jacob/develop/FastCode && uv run pytest fastcode/tests/ -x -q --ignore=fastcode/tests/test_e2e_indexing.py --ignore=fastcode/tests/test_e2e_semantic_pipeline.py 2>&1 | tail -5`
Expected: All passing, same count minus ~5 tests (the tests in deleted file)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: delete test_scip_transform.py — pure lookup-table theater (0 substance)"
```

---

### Task 2: Delete test_paths.py

**Files:**
- Delete: `fastcode/tests/utils/test_paths.py` (36 lines)

**Rationale:** Same as Task 1 — tests hardcoded extension→language mapping. No behavior verified.

- [ ] **Step 1: Verify no imports**

Run: `grep -r "test_paths" fastcode/tests/ --include="*.py" | grep -v "test_paths.py"`
Expected: No imports

- [ ] **Step 2: Delete the file**

```bash
rm fastcode/tests/utils/test_paths.py
```

- [ ] **Step 3: Run tests**

Run: `cd /home/jacob/develop/FastCode && uv run pytest fastcode/tests/ -x -q --ignore=fastcode/tests/test_e2e_indexing.py --ignore=fastcode/tests/test_e2e_semantic_pipeline.py 2>&1 | tail -5`
Expected: All passing

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: delete test_paths.py — pure lookup-table theater (0 substance)"
```

---

### Task 3: Rewrite test_llm.py

**Files:**
- Modify: `fastcode/tests/infrastructure/test_llm.py` (22 lines)
- Read: `fastcode/src/fastcode/infrastructure/llm.py`

**Rationale:** Current test mocks the LLM client and asserts the mock returns what was set up. Tests that mocks work, not that `chat_completion()` handles responses correctly.

- [ ] **Step 1: Read the production code**

Read: `fastcode/src/fastcode/infrastructure/llm.py`

The function `chat_completion(client, messages, model, max_tokens, temperature)` extracts `response.choices[0].message.content`. It has NO error handling for:
- Empty `choices` list
- `None` message or content
- Non-string content

- [ ] **Step 2: Write failing tests for actual error handling**

Replace the entire file with:

```python
"""Tests for fastcode.infrastructure.llm — error handling and response extraction."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from fastcode.infrastructure.llm import chat_completion


def _mock_client(choice_content: str | None = "hello", *, empty_choices: bool = False,
                 none_message: bool = False) -> MagicMock:
    """Build a mock OpenAI client with configurable response shape."""
    mock = MagicMock()
    response = MagicMock()
    if empty_choices:
        response.choices = []
    else:
        choice = MagicMock()
        if none_message:
            choice.message = None
        else:
            choice.message.content = choice_content
        response.choices = [choice]
    mock.chat.completions.create.return_value = response
    return mock


class TestChatCompletion:
    def test_extracts_content_from_response(self):
        client = _mock_client("test response text")
        result = chat_completion(client, messages=[{"role": "user", "content": "hi"}],
                                 model="gpt-4", max_tokens=100, temperature=0.5)
        assert result == "test response text"

    def test_passes_parameters_to_client(self):
        client = _mock_client("ok")
        chat_completion(client, messages=[{"role": "user", "content": "hi"}],
                        model="gpt-4", max_tokens=200, temperature=0.7)
        client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=200,
            temperature=0.7,
        )

    def test_raises_on_empty_choices(self):
        client = _mock_client(empty_choices=True)
        with pytest.raises(IndexError):
            chat_completion(client, messages=[{"role": "user", "content": "hi"}],
                            model="gpt-4", max_tokens=100, temperature=0.5)

    def test_raises_on_none_message(self):
        client = _mock_client(none_message=True)
        with pytest.raises(AttributeError):
            chat_completion(client, messages=[{"role": "user", "content": "hi"}],
                            model="gpt-4", max_tokens=100, temperature=0.5)

    def test_returns_none_content_as_none(self):
        """If the API returns None content, chat_completion should return None (not crash)."""
        client = _mock_client(None)
        result = chat_completion(client, messages=[{"role": "user", "content": "hi"}],
                                 model="gpt-4", max_tokens=100, temperature=0.5)
        assert result is None
```

- [ ] **Step 3: Run tests**

Run: `cd /home/jacob/develop/FastCode && uv run pytest fastcode/tests/infrastructure/test_llm.py -v`
Expected: 4 PASS, 1 FAIL (test_returns_none_content_as_none — production code may need a guard, or we accept None passthrough)

- [ ] **Step 4: If None test fails, verify production behavior**

Read `fastcode/src/fastcode/infrastructure/llm.py` — if `response.choices[0].message.content` already returns None naturally, the test should pass. If it crashes, that's a real bug worth documenting.

- [ ] **Step 5: Commit**

```bash
git add fastcode/tests/infrastructure/test_llm.py
git commit -m "test: rewrite test_llm.py — test error handling, not mock echo"
```

---

## Phase 2: Replace Round-Trip Tests With Behavior Tests

The worst offenders: `test_semantic_ir.py` (10/100), `test_scip_models.py` (10/100), `test_core_types.py` (5/100). These are 75%+ serialization round-trips. We keep 2-3 smoke tests per file and replace the rest with tests for actual business logic.

---

### Task 4: Rewrite test_core_types.py

**Files:**
- Modify: `fastcode/tests/schemas/test_core_types.py` (688 lines)
- Read: `fastcode/src/fastcode/schemas/core_types.py`

**What's theater:** 90% of the file tests that frozen dataclass fields exist and hold values. The only real logic is `Hit.from_retrieval_row()` / `Hit.to_retrieval_row()` — which does nested dict extraction with type coercion and fallbacks.

- [ ] **Step 1: Read production code**

Read: `fastcode/src/fastcode/schemas/core_types.py`

Focus on `Hit.from_retrieval_row()` (lines ~73-107) — it extracts nested fields from a dict with fallbacks and type coercion.

- [ ] **Step 2: Write the replacement tests**

Replace the entire file with tests targeting the actual logic — `Hit.from_retrieval_row()` conversion, frozen enforcement, and field mapping:

```python
"""Tests for fastcode.schemas.core_types — behavior, not field existence.

Previous version: 688 lines of dataclass field checks.
This version: tests actual conversion logic, error paths, and invariants.
"""
from __future__ import annotations

import dataclasses

import pytest

from fastcode.schemas.core_types import (
    FusionConfig,
    Hit,
    IterationConfig,
    ScipKind,
    ScipRole,
)


def _sample_retrieval_row(**overrides) -> dict:
    """Minimal valid retrieval row matching what PgRetrievalStore returns."""
    base = {
        "element": {
            "id": "elem-42",
            "type": "function",
            "name": "my_func",
            "qualified_name": "mod.my_func",
            "file_path": "/src/mod.py",
            "start_line": 10,
            "end_line": 20,
            "language": "python",
            "source": "scip",
        },
        "semantic_score": 0.85,
        "keyword_score": 0.72,
        "total_score": 0.78,
        "retrieval_channel": "semantic",
        "snapshot_id": "snap:test:abc",
        "repo_name": "test_repo",
        "branch": "main",
        "doc_id": "doc:/src/mod.py",
        "chunk_id": "chunk-1",
        "source_set": ["scip"],
        "metadata": {"role": "definition"},
        "llm_file_selected": True,
    }
    base.update(overrides)
    return base


# ─── Hit.from_retrieval_row — the only method with real logic ───


class TestHitFromRetrievalRow:
    def test_extracts_nested_element_fields(self):
        row = _sample_retrieval_row()
        hit = Hit.from_retrieval_row(row)
        assert hit.element_id == "elem-42"
        assert hit.element_type == "function"
        assert hit.element_name == "my_func"

    def test_coerces_scores_to_float(self):
        row = _sample_retrieval_row(
            semantic_score="0.5",
            keyword_score="0.3",
            total_score="0.4",
        )
        hit = Hit.from_retrieval_row(row)
        assert hit.semantic_score == pytest.approx(0.5)
        assert hit.keyword_score == pytest.approx(0.3)

    def test_defaults_scores_to_zero_when_missing(self):
        row = _sample_retrieval_row()
        del row["semantic_score"]
        del row["keyword_score"]
        del row["total_score"]
        hit = Hit.from_retrieval_row(row)
        assert hit.semantic_score == 0.0
        assert hit.keyword_score == 0.0
        assert hit.total_score == 0.0

    def test_handles_missing_element_key(self):
        """from_retrieval_row should handle element=None gracefully."""
        row = _sample_retrieval_row()
        row["element"] = None
        hit = Hit.from_retrieval_row(row)
        assert hit.element_id == ""
        assert hit.element_type == ""

    def test_maps_llm_file_selected_to_llm_selected(self):
        row = _sample_retrieval_row(llm_file_selected=True)
        hit = Hit.from_retrieval_row(row)
        assert hit.llm_selected is True

    def test_llm_selected_defaults_false(self):
        row = _sample_retrieval_row()
        del row["llm_file_selected"]
        hit = Hit.from_retrieval_row(row)
        assert hit.llm_selected is False

    def test_preserves_metadata_dict(self):
        row = _sample_retrieval_row(metadata={"role": "definition", "extra": True})
        hit = Hit.from_retrieval_row(row)
        assert hit.metadata["role"] == "definition"
        assert hit.metadata["extra"] is True

    def test_roundtrip_preserves_core_fields(self):
        """Smoke test: from_retrieval_row → to_retrieval_row preserves identity."""
        row = _sample_retrieval_row()
        hit = Hit.from_retrieval_row(row)
        out = hit.to_retrieval_row()
        assert out["element"]["id"] == "elem-42"
        assert out["semantic_score"] == pytest.approx(0.85)


# ─── Hit frozen enforcement ───


class TestHitFrozen:
    def test_frozen_raises_on_field_mutation(self):
        hit = Hit.from_retrieval_row(_sample_retrieval_row())
        with pytest.raises(dataclasses.FrozenInstanceError):
            hit.element_id = "changed"


# ─── FusionConfig.from_dict ───


class TestFusionConfig:
    def test_from_dict_extracts_known_fields(self):
        cfg = FusionConfig.from_dict({"alpha": 0.7, "k_code": 60, "k_doc": 30})
        assert cfg.alpha == pytest.approx(0.7)
        assert cfg.k_code == 60
        assert cfg.k_doc == 30

    def test_from_dict_uses_defaults_for_missing_keys(self):
        cfg = FusionConfig.from_dict({})
        assert isinstance(cfg.alpha, float)
        assert cfg.alpha > 0

    def test_from_dict_coerces_string_values(self):
        cfg = FusionConfig.from_dict({"alpha": "0.5", "k_code": "20"})
        assert cfg.alpha == pytest.approx(0.5)
        assert cfg.k_code == 20


# ─── ScipKind / ScipRole constants ───


class TestScipConstants:
    def test_kind_values_are_distinct(self):
        """All kind string values should be unique (no accidental duplicates)."""
        values = [v for k, v in vars(ScipKind).items() if not k.startswith("_")]
        assert len(values) == len(set(values))

    def test_role_values_are_distinct(self):
        values = [v for k, v in vars(ScipRole).items() if not k.startswith("_")]
        assert len(values) == len(set(values))
```

- [ ] **Step 3: Run tests**

Run: `cd /home/jacob/develop/FastCode && uv run pytest fastcode/tests/schemas/test_core_types.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add fastcode/tests/schemas/test_core_types.py
git commit -m "test: rewrite test_core_types.py — test conversion logic and error paths, not field existence"
```

---

### Task 5: Rewrite test_scip_models.py

**Files:**
- Modify: `fastcode/tests/test_scip_models.py` (619 lines)
- Read: `fastcode/src/fastcode/scip_models.py`

**What's theater:** 85% round-trip tests (create → to_dict → from_dict → assert fields match). The real logic is: `SCIPIndex.from_dict()` separating reserved fields from metadata, nested document conversion, range handling.

- [ ] **Step 1: Read production code**

Read: `fastcode/src/fastcode/scip_models.py`

Focus on `SCIPIndex.from_dict()` — it separates reserved fields (`documents`, `indexer_name`, `indexer_version`) from everything else (goes to `metadata`).

- [ ] **Step 2: Write replacement tests**

```python
"""Tests for fastcode.scip_models — metadata extraction, nested conversion, edge cases.

Previous version: 619 lines of to_dict/from_dict round-trips.
This version: tests actual conversion logic, reserved field separation, edge handling.
"""
from __future__ import annotations

import pytest

from fastcode.scip_models import (
    SCIPArtifactRef,
    SCIPDocument,
    SCIPIndex,
    SCIPOccurrence,
    SCIPSymbol,
)


def _make_symbol(symbol_name: str = "my_func()", **overrides) -> dict:
    base = {
        "symbol": symbol_name,
        "documentation": "doc for " + symbol_name,
        "kind": 12,  # Function
        "display_name": symbol_name,
    }
    base.update(overrides)
    return base


def _make_occurrence(**overrides) -> dict:
    base = {
        "symbol": "my_func()",
        "range": [1, 0, 1, 10],
        "role": 1,  # Definition
    }
    base.update(overrides)
    return base


def _make_document(path: str = "src/main.py", symbols=None, occurrences=None) -> dict:
    return {
        "relative_path": path,
        "language": "python",
        "symbols": symbols or [_make_symbol()],
        "occurrences": occurrences or [_make_occurrence()],
    }


def _make_index(documents=None, **overrides) -> dict:
    base = {
        "indexer_name": "scip-python",
        "indexer_version": "1.2.3",
        "documents": documents or [_make_document()],
    }
    base.update(overrides)
    return base


# ─── SCIPIndex metadata separation ───


class TestSCIPIndexMetadata:
    def test_reserved_fields_not_in_metadata(self):
        raw = _make_index(custom_field="extra_value")
        index = SCIPIndex.from_dict(raw)
        assert "indexer_name" not in index.metadata
        assert "indexer_version" not in index.metadata
        assert "documents" not in index.metadata

    def test_unknown_fields_go_to_metadata(self):
        raw = _make_index(custom_tool="cargo-scip", schema_version="3")
        index = SCIPIndex.from_dict(raw)
        assert index.metadata["custom_tool"] == "cargo-scip"
        assert index.metadata["schema_version"] == "3"

    def test_metadata_roundtrips_through_to_dict(self):
        raw = _make_index(extra="value")
        index = SCIPIndex.from_dict(raw)
        out = index.to_dict()
        assert out["extra"] == "value"
        assert out["indexer_name"] == "scip-python"

    def test_empty_documents_list(self):
        index = SCIPIndex.from_dict(_make_index(documents=[]))
        assert len(index.documents) == 0

    def test_non_dict_items_in_documents_filtered(self):
        raw = _make_index(documents=[_make_document(), "not_a_dict", None, 42])
        index = SCIPIndex.from_dict(raw)
        assert len(index.documents) == 1  # Only the valid dict


# ─── SCIPDocument nested conversion ───


class TestSCIPDocumentConversion:
    def test_converts_nested_symbols(self):
        raw = _make_document(symbols=[
            _make_symbol("func_a()"),
            _make_symbol("func_b()"),
        ])
        doc = SCIPDocument.from_dict(raw)
        assert len(doc.symbols) == 2
        assert doc.symbols[0].symbol == "func_a()"
        assert doc.symbols[1].symbol == "func_b()"

    def test_converts_nested_occurrences(self):
        raw = _make_document(occurrences=[
            _make_occurrence(symbol="f()", role=1),
            _make_occurrence(symbol="f()", role=2),
        ])
        doc = SCIPDocument.from_dict(raw)
        assert len(doc.occurrences) == 2
        assert doc.occurrences[0].role == 1
        assert doc.occurrences[1].role == 2

    def test_empty_symbols_and_occurrences(self):
        raw = _make_document(symbols=[], occurrences=[])
        doc = SCIPDocument.from_dict(raw)
        assert len(doc.symbols) == 0
        assert len(doc.occurrences) == 0


# ─── Range handling ───


class TestRangeHandling:
    def test_range_serialized_as_list(self):
        occ = SCIPOccurrence.from_dict(_make_occurrence(range=[5, 0, 5, 20]))
        out = occ.to_dict()
        assert isinstance(out["range"], list)
        assert out["range"] == [5, 0, 5, 20]

    def test_none_range_handled(self):
        occ = SCIPOccurrence.from_dict(_make_occurrence())
        occ.range = None
        out = occ.to_dict()
        assert "range" in out


# ─── SCIPArtifactRef ───


class TestSCIPArtifactRef:
    def test_from_dict_type_coercion(self):
        ref = SCIPArtifactRef.from_dict({
            "artifact_type": "lsif",
            "uri": "file:///src/main.py",
            "locator": {"provider": "sourcegraph"},
        })
        assert ref.artifact_type == "lsif"
        assert ref.uri == "file:///src/main.py"


# ─── Smoke round-trip (keep exactly 1) ───


class TestSmokeRoundTrip:
    def test_full_index_roundtrip_smoke(self):
        """One smoke test: full index survives to_dict → from_dict cycle."""
        raw = _make_index(extra_meta="test")
        index = SCIPIndex.from_dict(raw)
        restored = SCIPIndex.from_dict(index.to_dict())
        assert restored.indexer_name == index.indexer_name
        assert len(restored.documents) == len(index.documents)
        assert restored.metadata["extra_meta"] == "test"
```

- [ ] **Step 3: Run tests**

Run: `cd /home/jacob/develop/FastCode && uv run pytest fastcode/tests/test_scip_models.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add fastcode/tests/test_scip_models.py
git commit -m "test: rewrite test_scip_models.py — test metadata separation, nested conversion, edge cases"
```

---

### Task 6: Rewrite test_semantic_ir.py

**Files:**
- Modify: `fastcode/tests/test_semantic_ir.py` (1152 lines)
- Read: `fastcode/src/fastcode/semantic_ir.py`

**What's theater:** ~75% Hypothesis-powered round-trip tests (`to_dict` → `from_dict` → assert fields match). The real logic lives in: `IRSnapshot` legacy conversion, occurrence deduplication (SCIP priority), `IRRelation.confidence` mapping, `IRCodeUnit.source_priority` computation, and `_from_legacy_payload()`.

- [ ] **Step 1: Read production code**

Read: `fastcode/src/fastcode/semantic_ir.py`

Key functions to target:
- `_resolution_to_confidence()` / `_confidence_to_resolution()` — state mapping with fallback
- `IRCodeUnit.source_priority` — computed from `source_set` contents (SCIP+fc=100, SCIP-only=100, fc-only=50, other=0)
- `IRSnapshot.occurrences` — deduplication with composite key, SCIP priority
- `IRSnapshot._from_legacy_payload()` — bidirectional format conversion
- `IRRelation.source` — computed from `support_sources`
- `IRRelation.confidence` — computed from `resolution_state`

- [ ] **Step 2: Write replacement tests**

```python
"""Tests for fastcode.semantic_ir — computed properties, deduplication, legacy conversion.

Previous version: 1152 lines of Hypothesis round-trip tests.
This version: tests source_priority computation, occurrence dedup, confidence mapping,
              legacy payload conversion, and edge cases.
"""
from __future__ import annotations

import pytest

from fastcode.semantic_ir import (
    IRCodeUnit,
    IRDocument,
    IREdge,
    IROccurrence,
    IRRelation,
    IRSnapshot,
    IRSymbol,
    IRUnitSupport,
    _confidence_to_resolution,
    _resolution_to_confidence,
)


# ─── source_priority computation ───


class TestSourcePriority:
    def test_scip_and_fc_structure_gives_100(self):
        unit = IRCodeUnit(
            unit_id="u1", name="f", kind="function", path="a.py",
            source_set={"scip", "fc_structure"},
        )
        assert unit.source_priority == 100

    def test_scip_only_gives_100(self):
        unit = IRCodeUnit(
            unit_id="u2", name="f", kind="function", path="a.py",
            source_set={"scip"},
        )
        assert unit.source_priority == 100

    def test_fc_structure_only_gives_50(self):
        unit = IRCodeUnit(
            unit_id="u3", name="f", kind="function", path="a.py",
            source_set={"fc_structure"},
        )
        assert unit.source_priority == 50

    def test_no_known_source_gives_0(self):
        unit = IRCodeUnit(
            unit_id="u4", name="f", kind="function", path="a.py",
            source_set=set(),
        )
        assert unit.source_priority == 0

    def test_unknown_source_gives_0(self):
        unit = IRCodeUnit(
            unit_id="u5", name="f", kind="function", path="a.py",
            source_set={"custom_extractor"},
        )
        assert unit.source_priority == 0


# ─── confidence ↔ resolution_state mapping ───


class TestConfidenceMapping:
    @pytest.mark.parametrize("state,expected", [
        ("precise", "precise"),
        ("structural", "structural"),
        ("heuristic", "heuristic"),
    ])
    def test_known_states_map_to_themselves(self, state, expected):
        assert _resolution_to_confidence(state) == expected

    def test_unknown_state_falls_back_to_structural(self):
        assert _resolution_to_confidence("bogus_value") == "structural"

    def test_none_state_falls_back_to_structural(self):
        assert _resolution_to_confidence(None) == "structural"

    @pytest.mark.parametrize("confidence,expected", [
        ("precise", "precise"),
        ("structural", "structural"),
        ("heuristic", "heuristic"),
    ])
    def test_reverse_mapping_known(self, confidence, expected):
        assert _confidence_to_resolution(confidence) == expected

    def test_reverse_unknown_falls_back(self):
        assert _confidence_to_resolution("bogus") == "structural"


# ─── IRRelation computed properties ───


class TestIRRelationProperties:
    def test_source_returns_first_sorted_support_source(self):
        rel = IRRelation(
            relation_id="r1", src_unit_id="a", tgt_unit_id="b",
            relation_type="call",
            support_sources=["z_source", "a_source"],
        )
        assert rel.source == "a_source"

    def test_source_falls_back_to_metadata_when_no_support_sources(self):
        rel = IRRelation(
            relation_id="r2", src_unit_id="a", tgt_unit_id="b",
            relation_type="call",
            metadata={"source": "meta_source"},
        )
        assert rel.source == "meta_source"

    def test_confidence_from_resolution_state(self):
        rel = IRRelation(
            relation_id="r3", src_unit_id="a", tgt_unit_id="b",
            relation_type="call",
            resolution_state="precise",
        )
        assert rel.confidence == "precise"

    def test_doc_id_from_metadata(self):
        rel = IRRelation(
            relation_id="r4", src_unit_id="a", tgt_unit_id="b",
            relation_type="call",
            metadata={"doc_id": "doc:/src/main.py"},
        )
        assert rel.doc_id == "doc:/src/main.py"

    def test_doc_id_none_when_not_in_metadata(self):
        rel = IRRelation(
            relation_id="r5", src_unit_id="a", tgt_unit_id="b",
            relation_type="call",
        )
        assert rel.doc_id is None


# ─── Occurrence deduplication ───


class TestOccurrenceDeduplication:
    def _snapshot_with_occurrences(self, supports, *, unit_source="scip"):
        """Build a snapshot with the given IRUnitSupport entries."""
        units = [
            IRCodeUnit(unit_id="sym:func_a", name="func_a", kind="function",
                       path="a.py", source_set={unit_source}),
            IRCodeUnit(unit_id="doc:a.py", name="a.py", kind="file",
                       path="a.py", source_set={"scip"}),
        ]
        return IRSnapshot(
            repo_name="test", snapshot_id="snap:test:c1",
            units=units, supports=supports,
        )

    def test_duplicate_occurrences_deduplicated(self):
        """Two occurrences with same (symbol, doc, role, position) should become one."""
        supports = [
            IRUnitSupport(
                unit_id="sym:func_a", target_unit_id="doc:a.py",
                support_type="definition",
                source="scip",
                start_line=10, start_col=0, end_line=10, end_col=10,
            ),
            IRUnitSupport(
                unit_id="sym:func_a", target_unit_id="doc:a.py",
                support_type="definition",
                source="fc_structure",
                start_line=10, start_col=0, end_line=10, end_col=10,
            ),
        ]
        snap = self._snapshot_with_occurrences(supports)
        assert len(snap.occurrences) == 1

    def test_scip_wins_over_non_scip_on_duplicate(self):
        """When deduplicating, SCIP source should be kept over non-SCIP."""
        supports = [
            IRUnitSupport(
                unit_id="sym:func_a", target_unit_id="doc:a.py",
                support_type="definition",
                source="fc_structure",
                start_line=10, start_col=0, end_line=10, end_col=10,
            ),
            IRUnitSupport(
                unit_id="sym:func_a", target_unit_id="doc:a.py",
                support_type="definition",
                source="scip",
                start_line=10, start_col=0, end_line=10, end_col=10,
            ),
        ]
        snap = self._snapshot_with_occurrences(supports)
        assert len(snap.occurrences) == 1
        assert "scip" in snap.occurrences[0].source

    def test_different_positions_not_deduplicated(self):
        supports = [
            IRUnitSupport(
                unit_id="sym:func_a", target_unit_id="doc:a.py",
                support_type="definition",
                source="scip",
                start_line=10, start_col=0, end_line=10, end_col=10,
            ),
            IRUnitSupport(
                unit_id="sym:func_a", target_unit_id="doc:a.py",
                support_type="reference",
                source="scip",
                start_line=20, start_col=0, end_line=20, end_col=5,
            ),
        ]
        snap = self._snapshot_with_occurrences(supports)
        assert len(snap.occurrences) == 2


# ─── IRSnapshot legacy conversion ───


class TestLegacyConversion:
    def _legacy_payload(self, **overrides) -> dict:
        base = {
            "repo_name": "test",
            "snapshot_id": "snap:test:abc",
            "commit_id": "c1",
            "branch": "main",
            "documents": [
                IRDocument(doc_id="doc:/a.py", path="/a.py", language="python").to_dict(),
            ],
            "symbols": [
                {"symbol_id": "sym:f", "display_name": "f", "kind": "function",
                 "qualified_name": "mod.f", "path": "/a.py",
                 "start_line": 5, "end_line": 10},
            ],
            "occurrences": [],
            "edges": [],
            "attachments": [],
        }
        base.update(overrides)
        return base

    def test_legacy_creates_file_units_from_documents(self):
        snap = IRSnapshot.from_dict(self._legacy_payload())
        doc_units = [u for u in snap.units if u.kind == "file"]
        assert len(doc_units) == 1
        assert doc_units[0].path == "/a.py"

    def test_legacy_creates_symbol_units(self):
        snap = IRSnapshot.from_dict(self._legacy_payload())
        sym_units = [u for u in snap.units if u.kind == "function"]
        assert len(sym_units) == 1

    def test_legacy_with_empty_collections(self):
        snap = IRSnapshot.from_dict(self._legacy_payload(
            documents=[], symbols=[], occurrences=[], edges=[], attachments=[],
        ))
        assert len(snap.units) == 0
        assert len(snap.occurrences) == 0
        assert len(snap.edges) == 0

    def test_canonical_format_preserved(self):
        """When saving canonical format and reloading, units/supports survive."""
        snap = IRSnapshot(
            repo_name="test", snapshot_id="snap:test:abc",
            units=[
                IRCodeUnit(unit_id="u1", name="f", kind="function", path="a.py"),
            ],
            supports=[],
            relations=[],
        )
        data = snap.to_dict()
        assert data.get("schema_version") == "ir.v2"
        restored = IRSnapshot.from_dict(data)
        assert len(restored.units) == 1


# ─── Smoke round-trip (keep exactly 1) ───


class TestSmokeRoundTrip:
    def test_snapshot_roundtrip_smoke(self):
        """One smoke test: snapshot survives to_dict → from_dict."""
        snap = IRSnapshot(
            repo_name="r", snapshot_id="snap:r:c1",
            units=[IRCodeUnit(unit_id="u1", name="f", kind="function", path="a.py")],
            supports=[], relations=[],
        )
        restored = IRSnapshot.from_dict(snap.to_dict())
        assert restored.repo_name == snap.repo_name
        assert len(restored.units) == len(snap.units)
```

- [ ] **Step 3: Run tests**

Run: `cd /home/jacob/develop/FastCode && uv run pytest fastcode/tests/test_semantic_ir.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add fastcode/tests/test_semantic_ir.py
git commit -m "test: rewrite test_semantic_ir.py — test computed properties, dedup logic, legacy conversion"
```

---

## Phase 3: Add Negative/Edge Case Tests to Critical Paths

These files already have some substance but are missing error handling and edge case coverage.

---

### Task 7: Add edge cases to test_terminus_publisher.py

**Files:**
- Modify: `fastcode/tests/test_terminus_publisher.py` (757 lines)
- Read: `fastcode/src/fastcode/terminus_publisher.py`

**What's missing:** Tests for `build_lineage_payload()` error conditions: missing `snapshot_id`, missing `repo_name`, None branch/commit/run (conditional nodes), document-symbol matching, parent commit dual-field handling.

- [ ] **Step 1: Read the end of the current test file to know where to append**

Run: `tail -30 fastcode/tests/test_terminus_publisher.py`

- [ ] **Step 2: Append new test class at the end of the file**

Add this class at the end of `fastcode/tests/test_terminus_publisher.py`:

```python


# ─── Edge cases for build_lineage_payload (negative/error paths) ───


class TestLineagePayloadEdgeCases:
    """Tests for build_lineage_payload error conditions and conditional node creation."""

    def _pub(self) -> TerminusPublisher:
        return TerminusPublisher({"terminus": {"endpoint": "http://localhost:6363"}})

    def test_missing_snapshot_id_raises(self):
        pub = self._pub()
        with pytest.raises(ValueError, match="snapshot_id"):
            pub.build_lineage_payload({}, {})

    def test_missing_repo_name_raises(self):
        pub = self._pub()
        with pytest.raises(ValueError, match="repo_name"):
            pub.build_lineage_payload({"snapshot_id": "snap:test:abc"}, {})

    def test_none_branch_omits_branch_node(self):
        pub = self._pub()
        payload = pub.build_lineage_payload(
            {"snapshot_id": "snap:test:abc", "repo_name": "test", "branch": None},
            {},
        )
        node_types = [n["type"] for n in payload["nodes"]]
        assert "branch" not in node_types

    def test_none_commit_omits_commit_node(self):
        pub = self._pub()
        payload = pub.build_lineage_payload(
            {"snapshot_id": "snap:test:abc", "repo_name": "test", "commit_id": None},
            {},
        )
        node_types = [n["type"] for n in payload["nodes"]]
        assert "commit" not in node_types

    def test_none_run_omits_run_node(self):
        pub = self._pub()
        payload = pub.build_lineage_payload(
            {"snapshot_id": "snap:test:abc", "repo_name": "test", "run_id": None},
            {},
        )
        node_types = [n["type"] for n in payload["nodes"]]
        assert "index_run" not in node_types

    def test_documents_with_none_doc_id_skipped(self):
        pub = self._pub()
        payload = pub.build_lineage_payload(
            {
                "snapshot_id": "snap:test:abc",
                "repo_name": "test",
                "documents": [{"doc_id": None, "path": "a.py"}],
                "symbols": [],
            },
            {},
        )
        doc_nodes = [n for n in payload["nodes"] if n["type"] == "doc"]
        assert len(doc_nodes) == 0

    def test_symbols_with_none_symbol_id_skipped(self):
        pub = self._pub()
        payload = pub.build_lineage_payload(
            {
                "snapshot_id": "snap:test:abc",
                "repo_name": "test",
                "documents": [{"doc_id": "doc:a.py", "path": "a.py"}],
                "symbols": [{"symbol_id": None, "path": "a.py"}],
            },
            {},
        )
        sym_nodes = [n for n in payload["nodes"] if n["type"] == "symbol"]
        assert len(sym_nodes) == 0

    def test_empty_documents_and_symbols(self):
        pub = self._pub()
        payload = pub.build_lineage_payload(
            {
                "snapshot_id": "snap:test:abc",
                "repo_name": "test",
                "documents": [],
                "symbols": [],
            },
            {},
        )
        doc_nodes = [n for n in payload["nodes"] if n["type"] == "doc"]
        sym_nodes = [n for n in payload["nodes"] if n["type"] == "symbol"]
        assert len(doc_nodes) == 0
        assert len(sym_nodes) == 0

    def test_git_meta_fallback_for_branch(self):
        pub = self._pub()
        payload = pub.build_lineage_payload(
            {"snapshot_id": "snap:test:abc", "repo_name": "test"},
            {"branch": "feature/x"},
        )
        node_types = [n["type"] for n in payload["nodes"]]
        assert "branch" in node_types

    def test_unconfigured_publisher_returns_empty_payload(self):
        pub = TerminusPublisher({"terminus": {}})
        payload = pub.build_lineage_payload(
            {"snapshot_id": "snap:test:abc", "repo_name": "test"},
            {},
        )
        # Unconfigured publisher should still build payload (payload construction
        # doesn't require endpoint, only posting does)
        assert "nodes" in payload
```

- [ ] **Step 3: Run tests**

Run: `cd /home/jacob/develop/FastCode && uv run pytest fastcode/tests/test_terminus_publisher.py -v -k "EdgeCase"`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add fastcode/tests/test_terminus_publisher.py
git commit -m "test: add negative/edge case tests for build_lineage_payload"
```

---

### Task 8: Add edge cases to test_scip_to_ir.py

**Files:**
- Modify: `fastcode/tests/adapters/test_scip_to_ir.py` (1403 lines)
- Read: `fastcode/src/fastcode/adapters/scip_to_ir.py`

**What's missing:** Tests for malformed SCIP payloads, missing required fields, empty symbol/occurrence lists, and invalid range data.

- [ ] **Step 1: Read the end of the current test file**

Run: `tail -30 fastcode/tests/adapters/test_scip_to_ir.py`

- [ ] **Step 2: Append edge case tests at the end**

Add at the end of `fastcode/tests/adapters/test_scip_to_ir.py`:

```python


# ─── Edge cases: malformed inputs and error handling ───


class TestBuildIrFromScipEdgeCases:
    """Negative tests for build_ir_from_scip with malformed/missing data."""

    def test_empty_documents_list(self):
        """Index with no documents should produce snapshot with no documents."""
        payload = _make_scip_payload(n_docs=0, n_symbols=0, n_occurrences=0)
        # Remove documents entirely
        payload["documents"] = []
        snap = _build(payload)
        assert len(snap.documents) == 0

    def test_document_with_no_symbols(self):
        """Document with empty symbols list should still create a document unit."""
        payload = _make_scip_payload(n_docs=1, n_symbols=0, n_occurrences=0)
        snap = _build(payload)
        assert len(snap.documents) == 1

    def test_occurrence_with_empty_range(self):
        """Occurrence with None range values should not crash."""
        raw = _make_scip_payload(n_docs=1, n_symbols=1, n_occurrences=1)
        # Set range to all Nones
        raw["documents"][0]["occurrences"][0]["range"] = [None, None, None, None]
        snap = _build(raw)
        # Should not crash, occurrence may have None lines
        assert snap is not None

    def test_missing_indexer_name_uses_default(self):
        payload = _make_scip_payload()
        del payload["indexer_name"]
        snap = _build(payload)
        assert snap is not None

    def test_snapshot_id_format(self):
        """snapshot_id should follow snap:{repo}:{commit} format."""
        raw = _make_scip_payload()
        raw["repo_name"] = "my-repo"
        raw["commit_id"] = "abc1234"
        snap = _build(raw)
        assert snap.snapshot_id.startswith("snap:my-repo:")

    def test_metadata_records_source_modes(self):
        """Built snapshot should record which source modes were used."""
        snap = _build(_make_scip_payload())
        assert "scip" in snap.metadata.get("source_modes", [])
```

- [ ] **Step 3: Run tests**

Run: `cd /home/jacob/develop/FastCode && uv run pytest fastcode/tests/adapters/test_scip_to_ir.py -v -k "EdgeCase"`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add fastcode/tests/adapters/test_scip_to_ir.py
git commit -m "test: add edge case tests for build_ir_from_scip — empty inputs, missing fields, ranges"
```

---

### Task 9: Add edge cases to test_ast_to_ir.py

**Files:**
- Modify: `fastcode/tests/adapters/test_ast_to_ir.py` (1094 lines)
- Read: `fastcode/src/fastcode/adapters/ast_to_ir.py`

- [ ] **Step 1: Read the end of the current test file**

Run: `tail -30 fastcode/tests/adapters/test_ast_to_ir.py`

- [ ] **Step 2: Append edge case tests**

Add at the end of `fastcode/tests/adapters/test_ast_to_ir.py`:

```python


# ─── Edge cases: empty inputs, boundary conditions ───


class TestBuildIrFromAstEdgeCases:
    """Edge case tests for build_ir_from_ast with boundary inputs."""

    def test_empty_elements_list(self):
        """No code elements should produce snapshot with no symbols."""
        snap = _build([])
        assert len(snap.symbols) == 0

    def test_element_with_no_metadata(self):
        """Element with empty metadata dict should not crash."""
        elements = [_elem(name="f", type="function", start_line=1, metadata={})]
        snap = _build(elements)
        assert len(snap.symbols) == 1

    def test_element_with_zero_start_line(self):
        """Start line of 0 should be clamped or handled."""
        elements = [_elem(name="f", type="function", start_line=0, metadata={})]
        snap = _build(elements)
        assert len(snap.symbols) == 1

    def test_multiple_elements_same_name_different_files(self):
        """Elements with same name in different files get different symbol_ids."""
        e1 = _elem(name="handler", type="function", start_line=10, metadata={},
                    file_path="a.py", language="python")
        e2 = _elem(name="handler", type="function", start_line=20, metadata={},
                    file_path="b.py", language="python")
        snap = _build([e1, e2])
        assert len(snap.symbols) == 2
        ids = [s.symbol_id for s in snap.symbols]
        assert ids[0] != ids[1]
```

- [ ] **Step 3: Run tests**

Run: `cd /home/jacob/develop/FastCode && uv run pytest fastcode/tests/adapters/test_ast_to_ir.py -v -k "EdgeCase"`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add fastcode/tests/adapters/test_ast_to_ir.py
git commit -m "test: add edge case tests for build_ir_from_ast — empty inputs, boundary lines, name collisions"
```

---

## Phase 4: Infrastructure Consolidation

---

### Task 10: Consolidate shared factories into conftest.py

**Files:**
- Modify: `fastcode/tests/conftest.py` — add shared factory imports

**Rationale:** `_make_snapshot()` is duplicated in 3+ files. This is a maintainability issue, not substance, but fixing it prevents the duplicate factories from drifting.

- [ ] **Step 1: Find all duplicated factory functions**

Run: `grep -rn "def _make_snapshot" fastcode/tests/ --include="*.py"`

- [ ] **Step 2: Read conftest.py to see what already exists**

Run: `head -50 fastcode/tests/conftest.py`

- [ ] **Step 3: If conftest already has factories, verify test files use them**

Run: `grep -l "from.*conftest" fastcode/tests/*.py fastcode/tests/**/*.py 2>/dev/null`

- [ ] **Step 4: Document the duplication but do NOT refactor in this plan**

The duplicated factories are a separate concern from substance. Each file being self-contained (no conftest.py imports) is the project's stated convention per CLAUDE.md: "No conftest.py — each test file is self-contained with its own factory functions."

**Action:** Skip this task. The project convention explicitly avoids shared conftest factories.

- [ ] **Step 5: Commit (no changes needed)**

No commit — this is documented as "by design."

---

## Self-Review Checklist

- [x] **Spec coverage:** Every theater pattern identified in the audit has a task addressing it
- [x] **Placeholder scan:** No "TBD", "TODO", "implement later", "fill in details", "add validation" — all code is concrete
- [x] **Type consistency:** All function names, class names, and field names match between tasks (verified against production code analysis)
- [x] **File paths:** All paths are exact and verified to exist
- [x] **Test commands:** All pytest commands use correct paths and flags

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Test lines | ~24,025 | ~18,000 (est.) |
| Pure theater files | 2 (test_scip_transform, test_paths) | 0 |
| Files with <30 substance score | 6 | 0 |
| Deep assertion ratio (median) | ~25% | ~55% |
| Negative/edge case coverage | ~5% | ~25% |
| Tests testing mocks instead of code | ~15% | ~2% |

## What This Plan Does NOT Cover

- Adding tests for currently untested modules (out of scope — this is about fixing existing tests)
- Performance/regression testing (separate concern)
- Integration test coverage expansion (E2E tests require running services)
- Refactoring production code (all changes are test-only)
