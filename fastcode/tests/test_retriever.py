"""Tests for retriever module."""

from __future__ import annotations

from typing import Any

from fastcode.indexer import CodeElement
from fastcode.retriever import HybridRetriever


def _mk_row(
    elem_id: str,
    elem_type: str,
    total: float,
    *,
    ir_symbol_id: str | None = None,
    trace_links: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata = {}
    if ir_symbol_id:
        metadata["ir_symbol_id"] = ir_symbol_id
    if trace_links is not None:
        metadata["trace_links"] = trace_links
    return {
        "element": {
            "id": elem_id,
            "type": elem_type,
            "name": elem_id,
            "metadata": metadata,
        },
        "semantic_score": total,
        "keyword_score": total * 0.5,
        "pseudocode_score": 0.0,
        "graph_score": 0.0,
        "total_score": total,
    }


def _mk_retriever() -> HybridRetriever:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.config = {"docs_integration": {"enabled": True}}
    retriever.retrieval_config = {"doc_projection_beta_max": 0.35}
    retriever.adaptive_fusion_cfg = {
        "alpha_base": 0.8,
        "alpha_min": 0.25,
        "alpha_max": 0.9,
        "rrf_k_base": 60,
        "rrf_k_min": 20,
        "rrf_k_max": 100,
    }
    retriever._last_fusion_debug = None
    retriever.filtered_bm25_elements = []
    retriever.full_bm25_elements = []
    return retriever


# --- Adaptive fusion tests ---


def test_adaptive_fusion_prefers_docs_for_design_intent_queries():
    retriever = _mk_retriever()

    code_rows = [_mk_row("code:1", "function", 0.45), _mk_row("code:2", "class", 0.42)]
    doc_rows = [
        _mk_row("doc:1", "design_document", 0.82),
        _mk_row("doc:2", "design_document", 0.7),
    ]
    alpha, _, _ = retriever._compute_adaptive_fusion_params(
        query="What is the architecture rationale for branch indexing design?",
        query_info={"intent": "design_rationale"},
        code_results=code_rows,
        doc_results=doc_rows,
    )
    assert alpha < 0.8

    fused = retriever._adaptive_fuse_channels(
        query="What is the architecture rationale for branch indexing design?",
        query_info={"intent": "design_rationale"},
        code_results=code_rows,
        doc_results=doc_rows,
    )
    ids = [r["element"]["id"] for r in fused]
    assert "doc:1" in ids
    assert "code:1" in ids
    assert fused[0]["element"]["id"] == "doc:1"


def test_adaptive_fusion_continuous_affinity_changes_alpha_gradually():
    retriever = _mk_retriever()
    code_rows = [_mk_row("code:1", "function", 0.6)]
    doc_rows = [_mk_row("doc:1", "design_document", 0.6)]

    alpha_low, _, _ = retriever._compute_adaptive_fusion_params(
        query="architecture",
        query_info={},
        code_results=code_rows,
        doc_results=doc_rows,
    )
    alpha_high, _, _ = retriever._compute_adaptive_fusion_params(
        query="architecture design rationale adr rfc decision tradeoff",
        query_info={},
        code_results=code_rows,
        doc_results=doc_rows,
    )
    assert alpha_high < alpha_low


def test_adaptive_fusion_entropy_signal_affects_alpha_and_k():
    retriever = _mk_retriever()
    code_rows = [_mk_row("code:1", "function", 0.62)]
    doc_rows = [_mk_row("doc:1", "design_document", 0.62)]

    alpha_low_entropy, k_code_low, _ = retriever._compute_adaptive_fusion_params(
        query="bug bug bug bug bug fix fix fix",
        query_info={},
        code_results=code_rows,
        doc_results=doc_rows,
    )
    alpha_high_entropy, k_code_high, _ = retriever._compute_adaptive_fusion_params(
        query="bug design call architecture method rationale trace decision implementation",
        query_info={},
        code_results=code_rows,
        doc_results=doc_rows,
    )
    assert abs(alpha_high_entropy - 0.5) <= abs(alpha_low_entropy - 0.5)
    assert k_code_high > k_code_low


# --- Doc channel projection tests ---


def test_doc_projection_builds_grounded_priors_from_doc_scores():
    retriever = _mk_retriever()
    doc_rows = [
        _mk_row(
            "doc:1",
            "design_document",
            0.9,
            trace_links=[
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
        ),
        _mk_row(
            "doc:2",
            "design_document",
            0.6,
            trace_links=[
                {
                    "unit_id": "ir:service",
                    "weight": 0.7,
                    "evidence_type": "exact_name_mention",
                }
            ],
        ),
    ]

    projection = retriever._project_doc_priors(
        query="architecture rationale",
        query_info={"intent": "design_rationale"},
        doc_results=doc_rows,
    )

    assert projection["p_doc"] > 0.0
    assert projection["priors"]["ir:service"] > projection["priors"]["ir:repo"]
    assert projection["evidence"]["ir:service"]


def test_doc_projection_seeds_existing_code_hits_and_adds_projected_only_hits():
    retriever = _mk_retriever()
    retriever.full_bm25_elements = [
        CodeElement(
            id="code:repo",
            type="class",
            name="Repository",
            file_path="repo.py",
            relative_path="repo.py",
            language="python",
            start_line=1,
            end_line=10,
            code="",
            signature=None,
            docstring=None,
            summary=None,
            metadata={"ir_symbol_id": "ir:repo"},
        )
    ]
    code_rows = [_mk_row("code:service", "function", 0.5, ir_symbol_id="ir:service")]
    doc_rows = [
        _mk_row(
            "doc:1",
            "design_document",
            0.9,
            trace_links=[
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
        )
    ]

    seeded = retriever._apply_doc_projection_to_code(
        query="design architecture",
        query_info={"intent": "design_rationale"},
        code_results=code_rows,
        doc_results=doc_rows,
    )

    by_id = {row["element"]["id"]: row for row in seeded}
    assert by_id["code:service"]["projection_score"] > 0.0
    assert by_id["code:service"]["seed_score"] > 0.0
    assert by_id["code:repo"]["projected_only"] is True
    assert by_id["code:repo"]["traceability"]


def test_extract_trace_links_ignores_ungrounded_links():
    retriever = _mk_retriever()
    row = _mk_row(
        "doc:1",
        "design_document",
        0.5,
        trace_links=[
            {"symbol_name": "Repository", "weight": 0.7},
            {"unit_id": "ir:repo", "confidence": "resolved"},
        ],
    )

    links = retriever._extract_trace_links(row)

    assert len(links) == 1
    assert links[0]["unit_id"] == "ir:repo"
    assert links[0]["weight"] == 0.8
