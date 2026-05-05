"""Tests for retriever module."""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from fastcode.ir.element import CodeElement
from fastcode.retrieval.hybrid import HybridRetriever


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


def _element(
    relative_path: str,
    *,
    element_id: str,
    element_type: str = "file",
    repo_name: str = "repo",
    metadata: dict[str, Any] | None = None,
) -> CodeElement:
    return CodeElement(
        id=element_id,
        type=element_type,
        name=relative_path,
        file_path=f"/repo/{relative_path}",
        relative_path=relative_path,
        language="python",
        start_line=1,
        end_line=10,
        code="pass\n",
        signature=None,
        docstring=None,
        summary=None,
        metadata=metadata or {},
        repo_name=repo_name,
        repo_url=None,
    )


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
        _element(
            "repo.py",
            element_id="code:repo",
            element_type="class",
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

    with patch.object(
        CodeElement,
        "to_dict",
        autospec=True,
        side_effect=AssertionError(
            "doc projection must not call CodeElement.to_dict()"
        ),
    ):
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


def test_keyword_search_avoids_code_element_to_dict() -> None:
    retriever = _mk_retriever()
    retriever.retrieval_backend = "local"
    retriever.pg_retrieval_store = None
    retriever._active_snapshot_id = None
    retriever.logger = MagicMock()
    retriever.filtered_bm25 = MagicMock()
    retriever.filtered_bm25.get_scores.return_value = np.asarray([2.5], dtype=float)
    retriever.filtered_bm25_elements = [
        _element("module.py", element_id="file:module", metadata={"k": "v"})
    ]
    retriever.full_bm25 = None

    with patch.object(
        CodeElement,
        "to_dict",
        autospec=True,
        side_effect=AssertionError(
            "keyword search must not call CodeElement.to_dict()"
        ),
    ):
        results = retriever._keyword_search("module", top_k=1)

    assert len(results) == 1
    payload, score = results[0]
    assert payload["id"] == "file:module"
    assert payload["metadata"] == {"k": "v"}
    assert score == 2.5


def test_retrieval_helpers_avoid_code_element_to_dict() -> None:
    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.graph_weight = 1.0

    file_elem = _element("service.py", element_id="file:service")
    related_elem = _element(
        "service.py",
        element_id="func:helper",
        element_type="function",
    )
    retriever.full_bm25_elements = [file_elem, related_elem]
    retriever.graph_builder = MagicMock()
    retriever.graph_builder.element_by_id = {related_elem.id: related_elem}
    retriever._get_related_ids = MagicMock(return_value=[related_elem.id])

    with patch.object(
        CodeElement,
        "to_dict",
        autospec=True,
        side_effect=AssertionError(
            "retrieval helper must not call CodeElement.to_dict()"
        ),
    ):
        selected = retriever._retrieve_elements_from_files(
            [
                {
                    "repo_name": "repo",
                    "file_path": "service.py",
                    "reason": "selected by test",
                }
            ]
        )
        by_file = retriever.retrieve_by_file("service.py")
        by_type = retriever.retrieve_by_type("function")
        expanded = retriever._expand_with_graph(
            [
                {
                    "element": {"id": file_elem.id, "name": file_elem.name},
                    "semantic_score": 1.0,
                    "keyword_score": 0.0,
                    "graph_score": 0.0,
                    "total_score": 1.0,
                }
            ],
            max_hops=1,
        )

    assert selected[0]["element"]["id"] == file_elem.id
    assert selected[0]["llm_file_selected"] is True
    assert by_file[0]["element"]["id"] == file_elem.id
    assert by_type[0]["element"]["id"] == related_elem.id
    assert [row["element"]["id"] for row in expanded] == [
        file_elem.id,
        related_elem.id,
    ]


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


def test_load_bm25_uses_explicit_code_element_deserializer(tmp_path: Path) -> None:
    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)
    payload = {
        "id": "file:module",
        "type": "file",
        "name": "module.py",
        "file_path": "/repo/module.py",
        "relative_path": "module.py",
        "language": "python",
        "start_line": 1,
        "end_line": 3,
        "code": "pass\n",
        "signature": None,
        "docstring": None,
        "summary": None,
        "metadata": {"k": "v"},
        "repo_name": "repo",
        "repo_url": None,
    }
    with open(tmp_path / "index_bm25.pkl", "wb") as handle:
        pickle.dump({"bm25_corpus": [["module"]], "bm25_elements": [payload]}, handle)

    calls: list[dict[str, Any]] = []

    def _deserialize(element_payload: dict[str, Any]) -> CodeElement:
        calls.append(element_payload)
        return _element(
            "module.py",
            element_id=element_payload["id"],
            metadata=element_payload["metadata"],
        )

    with patch(
        "fastcode.retrieval.hybrid.deserialize_code_element",
        side_effect=_deserialize,
    ) as mock_deserialize:
        assert retriever.load_bm25("index") is True

    assert mock_deserialize.call_count == 1
    assert calls == [payload]
    assert retriever.full_bm25_elements[0].id == "file:module"
    assert retriever.full_bm25_corpus == [["module"]]


def test_reload_specific_repositories_uses_explicit_deserializer(
    tmp_path: Path,
) -> None:
    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)
    retriever.config = {}
    retriever.embedder = SimpleNamespace(embedding_dim=3)
    retriever.filtered_vector_store = SimpleNamespace(
        clear=lambda: None,
        merge_from_index=lambda _repo_name: True,
        get_count=lambda: 1,
    )
    retriever.iterative_agent = None
    payload = {
        "id": "file:service",
        "type": "file",
        "name": "service.py",
        "file_path": "/repo/service.py",
        "relative_path": "service.py",
        "language": "python",
        "start_line": 1,
        "end_line": 2,
        "code": "pass\n",
        "signature": None,
        "docstring": None,
        "summary": None,
        "metadata": {"stable_unit_id": "unit:file:service"},
        "repo_name": "repo",
        "repo_url": None,
    }
    with open(tmp_path / "repo_bm25.pkl", "wb") as handle:
        pickle.dump({"bm25_corpus": [["service"]], "bm25_elements": [payload]}, handle)

    calls: list[dict[str, Any]] = []

    def _deserialize(element_payload: dict[str, Any]) -> CodeElement:
        calls.append(element_payload)
        return _element(
            "service.py",
            element_id=element_payload["id"],
            metadata=element_payload["metadata"],
        )

    with patch(
        "fastcode.retrieval.hybrid.deserialize_code_element",
        side_effect=_deserialize,
    ) as mock_deserialize:
        assert retriever.reload_specific_repositories(["repo"]) is True

    assert mock_deserialize.call_count == 1
    assert calls == [payload]
    assert retriever.filtered_bm25_elements[0].id == "file:service"
    assert retriever.filtered_bm25_corpus == [["service"]]
