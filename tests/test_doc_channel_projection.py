from types import SimpleNamespace

from fastcode.indexer import CodeElement
from fastcode.retriever import HybridRetriever


def _mk_row(
    elem_id: str,
    elem_type: str,
    total: float,
    *,
    ir_symbol_id: str | None = None,
    trace_links: list[dict] | None = None,
) -> dict:
    metadata = {}
    if ir_symbol_id:
        metadata["ir_symbol_id"] = ir_symbol_id
    if trace_links is not None:
        metadata["trace_links"] = trace_links
    return {
        "element": {"id": elem_id, "type": elem_type, "name": elem_id, "metadata": metadata},
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


def test_doc_projection_builds_grounded_priors_from_doc_scores():
    retriever = _mk_retriever()
    doc_rows = [
        _mk_row(
            "doc:1",
            "design_document",
            0.9,
            trace_links=[
                {"unit_id": "ir:service", "weight": 0.8, "evidence_type": "exact_name_mention"},
                {"unit_id": "ir:repo", "weight": 0.5, "evidence_type": "exact_name_mention"},
            ],
        ),
        _mk_row(
            "doc:2",
            "design_document",
            0.6,
            trace_links=[{"unit_id": "ir:service", "weight": 0.7, "evidence_type": "exact_name_mention"}],
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
                {"unit_id": "ir:service", "weight": 0.8, "evidence_type": "exact_name_mention"},
                {"unit_id": "ir:repo", "weight": 0.9, "evidence_type": "exact_name_mention"},
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

