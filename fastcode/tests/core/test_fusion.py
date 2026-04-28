"""Tests for fastcode.core.fusion — pure fusion functions extracted from HybridRetriever."""

import pytest
from types import SimpleNamespace
from typing import Any

from fastcode.core.fusion import (
    adaptive_fuse_channels,
    apply_doc_projection_to_code,
    compute_adaptive_fusion_params,
    extract_trace_links,
    project_doc_priors,
)
from fastcode.schemas.core_types import FusionConfig


def _mk_row(elem_id: str, elem_type: str, total: float, **extra: Any) -> dict[str, Any]:
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
    return FusionConfig.from_dict(
        {
            "alpha_base": 0.8,
            "alpha_min": 0.25,
            "alpha_max": 0.9,
            "rrf_k_base": 60,
            "rrf_k_min": 20,
            "rrf_k_max": 100,
        }
    )


# ---------------------------------------------------------------------------
# compute_adaptive_fusion_params
# ---------------------------------------------------------------------------


def test_design_intent_lowers_alpha():
    """Design-oriented queries should lower alpha below alpha_base."""
    code_rows = [_mk_row("code:1", "function", 0.45), _mk_row("code:2", "class", 0.42)]
    doc_rows = [
        _mk_row("doc:1", "design_document", 0.82),
        _mk_row("doc:2", "design_document", 0.7),
    ]

    alpha, k_code, k_doc = compute_adaptive_fusion_params(
        query="What is the architecture rationale for branch indexing design?",
        query_info={"intent": "design_rationale"},
        code_results=code_rows,
        doc_results=doc_rows,
        config=_default_fusion_config(),
    )

    assert alpha < 0.8
    assert 0.25 <= alpha <= 0.9
    assert 20 <= k_code <= 100
    assert 20 <= k_doc <= 100


def test_code_query_keeps_alpha_high():
    """Code-oriented queries should keep alpha closer to alpha_base."""
    code_rows = [_mk_row("code:1", "function", 0.9), _mk_row("code:2", "class", 0.85)]
    doc_rows = [_mk_row("doc:1", "design_document", 0.3)]

    alpha, _, _ = compute_adaptive_fusion_params(
        query="fix the bug in the function call trace",
        query_info={},
        code_results=code_rows,
        doc_results=doc_rows,
        config=_default_fusion_config(),
    )

    assert alpha > 0.5


def test_more_doc_keywords_lower_alpha():
    """More design-keyword terms should produce a lower alpha."""
    code_rows = [_mk_row("code:1", "function", 0.6)]
    doc_rows = [_mk_row("doc:1", "design_document", 0.6)]

    alpha_low, _, _ = compute_adaptive_fusion_params(
        query="architecture",
        query_info={},
        code_results=code_rows,
        doc_results=doc_rows,
        config=_default_fusion_config(),
    )
    alpha_high, _, _ = compute_adaptive_fusion_params(
        query="architecture design rationale adr rfc decision tradeoff",
        query_info={},
        code_results=code_rows,
        doc_results=doc_rows,
        config=_default_fusion_config(),
    )

    assert alpha_high < alpha_low


def test_entropy_affects_alpha_and_k():
    """High-entropy queries pull alpha toward 0.5 and increase k_code."""
    code_rows = [_mk_row("code:1", "function", 0.62)]
    doc_rows = [_mk_row("doc:1", "design_document", 0.62)]

    alpha_low_entropy, k_code_low, _ = compute_adaptive_fusion_params(
        query="bug bug bug bug bug fix fix fix",
        query_info={},
        code_results=code_rows,
        doc_results=doc_rows,
        config=_default_fusion_config(),
    )
    alpha_high_entropy, k_code_high, _ = compute_adaptive_fusion_params(
        query="bug design call architecture method rationale trace decision implementation",
        query_info={},
        code_results=code_rows,
        doc_results=doc_rows,
        config=_default_fusion_config(),
    )

    assert abs(alpha_high_entropy - 0.5) <= abs(alpha_low_entropy - 0.5)
    assert k_code_high > k_code_low


def test_empty_results_returns_base_params():
    """Empty results should still return valid alpha, k_code, k_doc."""
    alpha, k_code, k_doc = compute_adaptive_fusion_params(
        query="test",
        query_info={},
        code_results=[],
        doc_results=[],
        config=_default_fusion_config(),
    )

    assert isinstance(alpha, float)
    assert isinstance(k_code, float)
    assert isinstance(k_doc, float)


# ---------------------------------------------------------------------------
# adaptive_fuse_channels
# ---------------------------------------------------------------------------


def test_fuse_channels_merges_code_and_doc():
    """Fused results should contain elements from both code and doc channels."""
    code_rows = [_mk_row("code:1", "function", 0.45)]
    doc_rows = [_mk_row("doc:1", "design_document", 0.82)]

    fused = adaptive_fuse_channels(
        query="architecture rationale",
        query_info={"intent": "design_rationale"},
        code_results=code_rows,
        doc_results=doc_rows,
        config=_default_fusion_config(),
    )

    ids = [r["element"]["id"] for r in fused]
    assert "doc:1" in ids
    assert "code:1" in ids
    # Both channels should contribute fusion metadata
    for row in fused:
        assert "fusion" in row


def test_fuse_channels_sorted_by_total_score():
    """Results should be sorted by total_score descending."""
    code_rows = [
        _mk_row("code:1", "function", 0.9),
        _mk_row("code:2", "function", 0.1),
    ]
    doc_rows = []

    fused = adaptive_fuse_channels(
        query="function bug fix",
        query_info={},
        code_results=code_rows,
        doc_results=doc_rows,
        config=_default_fusion_config(),
    )

    scores = [r["total_score"] for r in fused]
    assert scores == sorted(scores, reverse=True)


def test_fuse_channels_debug_populated():
    """Debug dict should be updated with fusion params."""
    debug: dict[str, Any] = {}
    code_rows = [_mk_row("code:1", "function", 0.5)]
    doc_rows = [_mk_row("doc:1", "design_document", 0.5)]

    adaptive_fuse_channels(
        query="test",
        query_info={},
        code_results=code_rows,
        doc_results=doc_rows,
        config=_default_fusion_config(),
        debug=debug,
    )

    assert "alpha" in debug
    assert "k_code" in debug
    assert "k_doc" in debug
    assert debug["code_candidates"] == 1
    assert debug["doc_candidates"] == 1


def test_fuse_channels_skips_no_id():
    """Rows without element id should be silently skipped."""
    code_rows = [{"element": {}, "total_score": 0.5}]
    doc_rows = []

    fused = adaptive_fuse_channels(
        query="test",
        query_info={},
        code_results=code_rows,
        doc_results=doc_rows,
        config=_default_fusion_config(),
    )

    assert fused == []


# ---------------------------------------------------------------------------
# project_doc_priors
# ---------------------------------------------------------------------------


def test_project_doc_priors_builds_grounded_priors():
    """Doc results with trace links should produce priors for linked units."""
    doc_rows = [
        _mk_row(
            "doc:1",
            "design_document",
            0.9,
            element={
                "id": "doc:1",
                "type": "design_document",
                "name": "doc:1",
                "metadata": {
                    "trace_links": [
                        {
                            "unit_id": "ir:service",
                            "weight": 0.8,
                            "evidence_type": "exact_name_mention",
                        },
                        {
                            "unit_id": "ir:repo",
                            "weight": 0.5,
                            "evidence_type": "exact_name_mention",
                        },
                    ],
                },
            },
        ),
    ]

    projection = project_doc_priors(
        query="architecture rationale",
        query_info={"intent": "design_rationale"},
        doc_results=doc_rows,
        config=_default_fusion_config(),
    )

    assert projection["p_doc"] > 0.0
    assert projection["priors"]["ir:service"] > projection["priors"]["ir:repo"]
    assert projection["evidence"]["ir:service"]


def test_project_doc_priors_empty_doc_results():
    """Empty doc results should produce empty priors."""
    projection = project_doc_priors(
        query="test",
        query_info={},
        doc_results=[],
        config=_default_fusion_config(),
    )

    assert projection["priors"] == {}
    assert projection["evidence"] == {}


# ---------------------------------------------------------------------------
# apply_doc_projection_to_code
# ---------------------------------------------------------------------------


def test_apply_doc_projection_returns_clones_when_no_docs():
    """Without doc results, code results should be cloned and returned."""
    code_rows = [_mk_row("code:1", "function", 0.5)]

    result = apply_doc_projection_to_code(
        query="test",
        query_info={},
        code_results=code_rows,
        doc_results=[],
        config=_default_fusion_config(),
    )

    assert len(result) == 1
    assert result[0]["element"]["id"] == "code:1"
    # Verify it's a clone, not the same dict
    assert result[0] is not code_rows[0]


def test_apply_doc_projection_adds_projected_only():
    """Doc-projected elements not in code results should get projected_only=True."""
    code_rows = [
        _mk_row(
            "code:service",
            "function",
            0.5,
            element={
                "id": "code:service",
                "type": "function",
                "name": "code:service",
                "metadata": {"ir_symbol_id": "ir:service"},
            },
        )
    ]
    doc_rows = [
        _mk_row(
            "doc:1",
            "design_document",
            0.9,
            element={
                "id": "doc:1",
                "type": "design_document",
                "name": "doc:1",
                "metadata": {
                    "trace_links": [
                        {
                            "unit_id": "ir:service",
                            "weight": 0.8,
                            "evidence_type": "exact_name_mention",
                        },
                        {
                            "unit_id": "ir:repo",
                            "weight": 0.9,
                            "evidence_type": "exact_name_mention",
                        },
                    ],
                },
            },
        )
    ]

    # Create a mock CodeElement for ir:repo
    mock_elem = SimpleNamespace(
        type="class",
        repo_name="myrepo",
        metadata={"ir_symbol_id": "ir:repo"},
        to_dict=lambda: {
            "id": "code:repo",
            "type": "class",
            "name": "Repository",
            "metadata": {"ir_symbol_id": "ir:repo"},
        },
    )

    result = apply_doc_projection_to_code(
        query="design architecture",
        query_info={"intent": "design_rationale"},
        code_results=code_rows,
        doc_results=doc_rows,
        config=_default_fusion_config(),
        bm25_elements=[mock_elem],
    )

    by_id = {row["element"]["id"]: row for row in result}
    assert by_id["code:service"]["projection_score"] > 0.0
    assert by_id["code:service"]["seed_score"] > 0.0
    assert by_id["code:repo"]["projected_only"] is True
    assert by_id["code:repo"]["traceability"]


# ---------------------------------------------------------------------------
# extract_trace_links
# ---------------------------------------------------------------------------


def test_extract_trace_links_ignores_ungrounded():
    """Links without unit_id should be ignored."""
    row = _mk_row(
        "doc:1",
        "design_document",
        0.5,
        element={
            "id": "doc:1",
            "type": "design_document",
            "name": "doc:1",
            "metadata": {
                "trace_links": [
                    {"symbol_name": "Repository", "weight": 0.7},
                    {"unit_id": "ir:repo", "confidence": "resolved"},
                ],
            },
        },
    )

    links = extract_trace_links(row)

    assert len(links) == 1
    assert links[0]["unit_id"] == "ir:repo"
    assert links[0]["weight"] == 0.8


# ---------------------------------------------------------------------------
# Exact-value and determinism tests
# ---------------------------------------------------------------------------


def test_fusion_params_determinism():
    """Same inputs must always produce identical outputs."""
    config = _default_fusion_config()
    kwargs = dict(
        query="test query",
        query_info={"intent": "code_search", "keywords": ["test"]},
        code_results=[_mk_row("c:1", "function", 0.7)],
        doc_results=[_mk_row("d:1", "design_document", 0.5)],
        config=config,
    )
    r1 = compute_adaptive_fusion_params(**kwargs)
    r2 = compute_adaptive_fusion_params(**kwargs)
    assert r1[0] == pytest.approx(r2[0])
    assert r1[1] == pytest.approx(r2[1])
    assert r1[2] == pytest.approx(r2[2])


def test_fusion_alpha_always_clamped():
    """Alpha must always be within [alpha_min, alpha_max]."""
    config = _default_fusion_config()
    for q in ["", "a", "design architecture rationale", "bug fix call trace stack runtime implementation"]:
        alpha, _, _ = compute_adaptive_fusion_params(
            query=q,
            query_info={},
            code_results=[_mk_row("c:1", "function", 0.99)],
            doc_results=[],
            config=config,
        )
        assert config.alpha_min <= alpha <= config.alpha_max, f"alpha={alpha} out of range for query={q!r}"


def test_fusion_k_always_in_range():
    """k_code and k_doc must always be within [rrf_k_min, rrf_k_max]."""
    config = _default_fusion_config()
    _, k_code, k_doc = compute_adaptive_fusion_params(
        query="test",
        query_info={},
        code_results=[_mk_row("c:1", "function", 0.5)],
        doc_results=[_mk_row("d:1", "design_document", 0.5)],
        config=config,
    )
    assert config.rrf_k_min <= k_code <= config.rrf_k_max
    assert config.rrf_k_min <= k_doc <= config.rrf_k_max


def test_more_code_strength_higher_alpha():
    """Higher code result scores should produce higher alpha."""
    config = _default_fusion_config()
    alpha_low, _, _ = compute_adaptive_fusion_params(
        query="test",
        query_info={},
        code_results=[_mk_row("c:1", "function", 0.3)],
        doc_results=[_mk_row("d:1", "design_document", 0.8)],
        config=config,
    )
    alpha_high, _, _ = compute_adaptive_fusion_params(
        query="test",
        query_info={},
        code_results=[_mk_row("c:1", "function", 0.95)],
        doc_results=[_mk_row("d:1", "design_document", 0.3)],
        config=config,
    )
    assert alpha_high > alpha_low
