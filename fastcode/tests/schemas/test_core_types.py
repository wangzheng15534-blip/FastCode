"""Tests for fastcode.schemas.core_types -- behavior, not field existence.

Previous version: 688 lines of dataclass field checks.
This version: tests actual conversion logic, error paths, and invariants.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from fastcode.schemas.core_types import (
    FusionConfig,
    Hit,
    ScipKind,
    ScipRole,
)


def _sample_retrieval_row(**overrides: Any) -> dict[str, Any]:
    """Minimal valid retrieval row matching what PgRetrievalStore returns."""
    base = {
        "element": {
            "id": "elem-42",
            "type": "function",
            "name": "my_func",
            "metadata": {"role": "definition"},
        },
        "score": 0.80,
        "semantic_score": 0.85,
        "keyword_score": 0.72,
        "pseudocode_score": 0.0,
        "graph_score": 0.0,
        "total_score": 0.78,
        "source": "scip",
        "projected_only": False,
        "llm_file_selected": True,
        "agent_found": False,
    }
    base.update(overrides)
    return base


# --- Hit.from_retrieval_row -- the only method with real logic ---


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
        del row["score"]
        hit = Hit.from_retrieval_row(row)
        assert hit.semantic_score == 0.0
        assert hit.keyword_score == 0.0
        assert hit.total_score == 0.0
        assert hit.score == 0.0

    def test_handles_missing_element_key(self):
        """from_retrieval_row should handle element=None gracefully."""
        row = _sample_retrieval_row()
        row["element"] = None
        hit = Hit.from_retrieval_row(row)
        assert hit.element_id == ""
        assert hit.element_type == ""
        assert hit.element_name == ""

    def test_handles_empty_element_dict(self):
        """from_retrieval_row should handle element={} gracefully."""
        row = _sample_retrieval_row()
        row["element"] = {}
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

    def test_preserves_metadata_from_element(self):
        """metadata is read from element.metadata, not the top-level row."""
        row = _sample_retrieval_row(
            element={
                "id": "elem-42",
                "type": "function",
                "name": "my_func",
                "metadata": {"role": "definition", "extra": True},
            }
        )
        hit = Hit.from_retrieval_row(row)
        assert hit.metadata["role"] == "definition"
        assert hit.metadata["extra"] is True

    def test_metadata_defaults_to_empty_dict(self):
        """When element has no metadata key, metadata should be empty."""
        row = _sample_retrieval_row(element={"id": "e1", "type": "t", "name": "n"})
        hit = Hit.from_retrieval_row(row)
        assert hit.metadata == {}

    def test_roundtrip_preserves_core_fields(self):
        """Smoke test: from_retrieval_row -> to_retrieval_row preserves identity."""
        row = _sample_retrieval_row()
        hit = Hit.from_retrieval_row(row)
        out = hit.to_retrieval_row()
        assert out["element"]["id"] == "elem-42"
        assert out["element"]["name"] == "my_func"
        assert out["semantic_score"] == pytest.approx(0.85)
        assert out["score"] == pytest.approx(0.80)
        assert out["llm_file_selected"] is True


# --- Hit frozen enforcement ---


class TestHitFrozen:
    def test_frozen_raises_on_field_mutation(self):
        hit = Hit.from_retrieval_row(_sample_retrieval_row())
        with pytest.raises(dataclasses.FrozenInstanceError):
            hit.element_id = "changed"

    def test_frozen_raises_on_metadata_reassignment(self):
        hit = Hit.from_retrieval_row(_sample_retrieval_row())
        with pytest.raises(dataclasses.FrozenInstanceError):
            hit.metadata = {}


# --- FusionConfig.from_dict ---


class TestFusionConfig:
    def test_from_dict_extracts_known_fields(self):
        cfg = FusionConfig.from_dict(
            {"alpha_base": 0.7, "rrf_k_base": 40, "rrf_k_max": 200}
        )
        assert cfg.alpha_base == pytest.approx(0.7)
        assert cfg.rrf_k_base == 40
        assert cfg.rrf_k_max == 200

    def test_from_dict_uses_defaults_for_missing_keys(self):
        cfg = FusionConfig.from_dict({})
        assert cfg.alpha_base == pytest.approx(0.8)
        assert cfg.rrf_k_base == 60
        assert cfg.alpha_min == pytest.approx(0.25)

    def test_from_dict_coerces_string_values(self):
        cfg = FusionConfig.from_dict({"alpha_base": "0.5", "rrf_k_base": "20"})
        assert cfg.alpha_base == pytest.approx(0.5)
        assert cfg.rrf_k_base == 20


# --- ScipKind / ScipRole constants ---


class TestScipConstants:
    def test_kind_values_are_distinct(self):
        """All kind string values should be unique (no accidental duplicates)."""
        values = [v for k, v in vars(ScipKind).items() if not k.startswith("_")]
        assert len(values) == len(set(values))

    def test_role_values_are_distinct(self):
        values = [v for k, v in vars(ScipRole).items() if not k.startswith("_")]
        assert len(values) == len(set(values))
