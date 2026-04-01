from fastcode.retriever import HybridRetriever


def _mk_row(elem_id: str, elem_type: str, total: float) -> dict:
    return {
        "element": {"id": elem_id, "type": elem_type, "name": elem_id},
        "semantic_score": total,
        "keyword_score": total * 0.5,
        "pseudocode_score": 0.0,
        "graph_score": 0.0,
        "total_score": total,
    }


def _mk_retriever() -> HybridRetriever:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.adaptive_fusion_cfg = {
        "alpha_base": 0.8,
        "alpha_min": 0.25,
        "alpha_max": 0.9,
        "rrf_k_base": 60,
        "rrf_k_min": 20,
        "rrf_k_max": 100,
    }
    retriever._last_fusion_debug = None
    return retriever


def test_adaptive_fusion_prefers_docs_for_design_intent_queries():
    retriever = _mk_retriever()

    code_rows = [_mk_row("code:1", "function", 0.45), _mk_row("code:2", "class", 0.42)]
    doc_rows = [_mk_row("doc:1", "design_document", 0.82), _mk_row("doc:2", "design_document", 0.7)]
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
    assert "doc:1" in ids and "code:1" in ids
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
