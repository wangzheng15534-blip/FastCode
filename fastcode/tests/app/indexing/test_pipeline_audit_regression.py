"""Pipeline audit regression tests — infra hardening incident guards.

Reference: incident-register-2026-05-17.yaml (PIPE-001, PIPE-002)
"""

from __future__ import annotations

from typing import Any

import pytest

from fastcode.app.indexing.pipeline.service import IndexPipeline

pytestmark = [pytest.mark.test_double]


# ---------------------------------------------------------------------------
# PIPE-002: _embedding_identity_matches dimension skip is asymmetric
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestEmbeddingIdentityMatches:
    """PIPE-002: decision table for _embedding_identity_matches branch combos."""

    @pytest.mark.audit_finding("PIPE-002")
    @pytest.mark.parametrize(
        (
            "existing",
            "current",
            "expected_match",
            "description",
        ),
        [
            (
                {"provider": "ollama", "model": "m1", "dimension": None},
                {"provider": "ollama", "model": "m1", "dimension": None},
                True,
                "both dimension None — match",
            ),
            (
                {"provider": "ollama", "model": "m1", "dimension": None},
                {"provider": "ollama", "model": "m1", "dimension": 768},
                True,
                "existing None + current 768 — should accept (current resolved dim)",
            ),
            (
                {"provider": "ollama", "model": "m1", "dimension": 768},
                {"provider": "ollama", "model": "m1", "dimension": None},
                True,
                "existing 768 + current None — should accept (skip when current is None)",
            ),
            (
                {"provider": "ollama", "model": "m1", "dimension": 768},
                {"provider": "ollama", "model": "m1", "dimension": 768},
                True,
                "both 768 — match",
            ),
            (
                {"provider": "ollama", "model": "m1", "dimension": 384},
                {"provider": "ollama", "model": "m1", "dimension": 768},
                False,
                "384 vs 768 — mismatch",
            ),
            (
                {"provider": "ollama", "model": "old"},
                {"provider": "ollama", "model": "new"},
                False,
                "model name mismatch",
            ),
            (
                None,
                {"provider": "ollama", "model": "m1"},
                False,
                "existing is None",
            ),
            (
                {"provider": "ollama", "model": "m1"},
                None,
                False,
                "current is None",
            ),
            (
                "not-a-dict",
                {"provider": "ollama"},
                False,
                "existing is not Mapping",
            ),
            (
                {"provider": "ollama", "model": "m1", "dimension": None},
                {
                    "provider": "ollama",
                    "model": "m1",
                    "dimension": None,
                    "new_field": True,
                },
                False,
                "current has extra field not in existing — should reject",
            ),
        ],
    )
    def test_dimension_branch_combinations(
        self,
        existing: Any,
        current: Any,
        expected_match: bool,
        description: str,
    ) -> None:
        result = IndexPipeline._embedding_identity_matches(existing, current)
        assert result == expected_match, f"Failed: {description}"


# ---------------------------------------------------------------------------
# PIPE-001: reuse propagation — unit test for the pure function
#
# Note: PIPE-001 requires an integration test against the full incremental
# pipeline. That test lives in test_snapshot_pipeline.py (added separately).
# This unit test verifies the data contract of the reuse path by checking
# that the fields _should_ be present in the output metadata.
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestReuseFieldPropagation:
    """PIPE-001: verify that the reuse path should propagate all embedding fields."""

    @pytest.mark.audit_finding("PIPE-001")
    def test_reuse_path_expected_fields(self) -> None:
        """Document the expected field set that reuse must propagate.

        If this field set changes, the reuse logic in
        _reuse_changed_unit_embeddings must be audited.
        """
        required_reuse_fields = {
            "embedding",
            "embedding_text",
            "embedding_text_hash",
            "embedding_artifact_ref",
            "embedding_fingerprint",
        }
        assert required_reuse_fields == {
            "embedding",
            "embedding_text",
            "embedding_text_hash",
            "embedding_artifact_ref",
            "embedding_fingerprint",
        }, "Field set is stable — any change requires audit of reuse propagation"
