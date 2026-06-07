"""Scenario contracts for retrieval intent from design notes."""

from __future__ import annotations

from typing import Any

from fastcode.ir.element import CodeElement
from fastcode.retrieval.contracts import FusionConfig, Hit
from fastcode.retrieval.ranking.fusion import (
    adaptive_fuse_channels,
    apply_doc_projection_to_code,
)


def _default_fusion_config() -> FusionConfig:
    return FusionConfig(
        alpha_base=0.8,
        alpha_min=0.25,
        alpha_max=0.9,
        rrf_k_base=60,
        rrf_k_min=20,
        rrf_k_max=100,
    )


def _hit(elem_id: str, elem_type: str, total: float, **extra: Any) -> Hit:
    row = {
        "element": {"id": elem_id, "type": elem_type, "name": elem_id},
        "semantic_score": total,
        "keyword_score": total * 0.5,
        "pseudocode_score": 0.0,
        "graph_score": 0.0,
        "total_score": total,
    }
    row.update(extra)
    return Hit.from_retrieval_row(row)


def _code_element(
    elem_id: str,
    *,
    unit_id: str,
    elem_type: str = "function",
) -> CodeElement:
    return CodeElement(
        id=elem_id,
        type=elem_type,
        name=elem_id,
        file_path=f"/repo/{elem_id}.py",
        relative_path=f"{elem_id}.py",
        language="python",
        start_line=1,
        end_line=10,
        code="pass\n",
        signature=None,
        docstring=None,
        summary=None,
        metadata={"ir_symbol_id": unit_id},
        repo_name="repo",
        repo_url=None,
    )


def _given_design_query_with_direct_code_and_design_docs() -> tuple[
    list[Hit], list[Hit]
]:
    return (
        [_hit("code:auth", "function", 0.45)],
        [
            _hit("doc:adr", "design_document", 0.82),
            _hit("doc:rfc", "design_document", 0.7),
        ],
    )


def _when_the_design_query_is_fused(
    code_hits: list[Hit], doc_hits: list[Hit]
) -> list[Hit]:
    return adaptive_fuse_channels(
        query="What is the architecture rationale for branch indexing design?",
        query_info={"intent": "design_rationale"},
        code_results=code_hits,
        doc_results=doc_hits,
        config=_default_fusion_config(),
    )


def _then_the_evidence_slate_contains_direct_code_and_design_support(
    fused: list[Hit],
) -> None:
    by_id = {hit.element_id: hit for hit in fused}

    assert set(by_id) == {"code:auth", "doc:adr", "doc:rfc"}
    assert by_id["code:auth"].extra["fusion"]["code_rrf"] > 0.0
    assert by_id["doc:adr"].extra["fusion"]["doc_rrf"] > 0.0
    assert fused[0].element_id == "doc:adr"


def test_design_intent_query_returns_mixed_evidence_slate() -> None:
    """Scenario: design intent returns code hits and design supporting docs."""
    code_hits, doc_hits = _given_design_query_with_direct_code_and_design_docs()

    fused = _when_the_design_query_is_fused(code_hits, doc_hits)

    _then_the_evidence_slate_contains_direct_code_and_design_support(fused)


def _given_design_docs_are_grounded_to_code_units() -> tuple[
    list[Hit],
    list[Hit],
    list[CodeElement],
]:
    code_hits = [
        _hit(
            "code:auth",
            "function",
            0.5,
            element={
                "id": "code:auth",
                "type": "function",
                "name": "authenticate",
                "metadata": {"ir_symbol_id": "ir:auth"},
            },
        )
    ]
    doc_hits = [
        _hit(
            "doc:adr-auth",
            "design_document",
            0.9,
            element={
                "id": "doc:adr-auth",
                "type": "design_document",
                "name": "Authentication ADR",
                "metadata": {
                    "trace_links": [
                        {
                            "unit_id": "ir:auth",
                            "weight": 0.8,
                            "evidence_type": "exact_name_mention",
                            "chunk_id": "adr-auth#auth",
                        },
                        {
                            "unit_id": "ir:token-store",
                            "weight": 0.9,
                            "evidence_type": "design_reference",
                            "chunk_id": "adr-auth#storage",
                        },
                    ],
                },
            },
        )
    ]
    searchable_code = [
        _code_element("code:token-store", unit_id="ir:token-store", elem_type="class"),
        CodeElement(
            id="doc:adr-auth",
            type="design_document",
            name="Authentication ADR",
            file_path="/repo/docs/auth.md",
            relative_path="docs/auth.md",
            language="markdown",
            start_line=1,
            end_line=10,
            code="",
            signature=None,
            docstring=None,
            summary=None,
            metadata={"ir_symbol_id": "ir:doc"},
            repo_name="repo",
            repo_url=None,
        ),
    ]
    return code_hits, doc_hits, searchable_code


def _when_doc_projection_seeds_the_code_space(
    code_hits: list[Hit],
    doc_hits: list[Hit],
    searchable_code: list[CodeElement],
) -> list[Hit]:
    return apply_doc_projection_to_code(
        query="how does authentication storage design work",
        query_info={"intent": "design_rationale"},
        code_results=code_hits,
        doc_results=doc_hits,
        config=_default_fusion_config(),
        bm25_elements=searchable_code,
    )


def _then_design_docs_support_code_without_becoming_code_nodes(
    projected: list[Hit],
) -> None:
    by_id = {hit.element_id: hit for hit in projected}

    assert set(by_id) == {"code:auth", "code:token-store"}
    assert "doc:adr-auth" not in by_id
    assert by_id["code:auth"].projection_score > 0.0
    assert by_id["code:token-store"].projected_only is True
    assert by_id["code:token-store"].traceability[0].doc_id == "doc:adr-auth"
    assert by_id["code:token-store"].traceability[0].link.chunk_id == (
        "adr-auth#storage"
    )


def test_design_docs_project_to_code_priors_not_graph_nodes() -> None:
    """Scenario: design docs influence graph expansion only through code priors."""
    (
        code_hits,
        doc_hits,
        searchable_code,
    ) = _given_design_docs_are_grounded_to_code_units()

    projected = _when_doc_projection_seeds_the_code_space(
        code_hits,
        doc_hits,
        searchable_code,
    )

    _then_design_docs_support_code_without_becoming_code_nodes(projected)
