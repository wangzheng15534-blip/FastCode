"""Tests for retriever module."""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from fastcode.ir.element import CodeElement
from fastcode.ir.graph import IRGraphs, IRGraphView
from fastcode.app.query.selection.retriever import HybridRetriever, _fusion_config_from_runtime
from fastcode.app.store.vectors.vector import VectorStore


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
    retriever.full_bm25_corpus = []
    retriever.full_bm25 = None
    retriever.filtered_bm25 = None
    retriever._bm25_shard_name = None
    retriever._bm25_shard_manifest = None
    retriever._full_bm25_shard_runtime = None
    retriever._filtered_bm25_shard_runtime = None
    retriever.retrieval_backend = "local"
    retriever.pg_retrieval_store = None
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


def test_runtime_fusion_config_maps_to_typed_record() -> None:
    config = _fusion_config_from_runtime(
        {
            "alpha_base": "0.7",
            "alpha_min": None,
            "rrf_k_base": "40",
            "rrf_k_max": 200,
            "enabled": True,
            "unexpected": "ignored",
        }
    )

    assert config.alpha_base == pytest.approx(0.7)
    assert config.alpha_min == pytest.approx(0.25)
    assert config.rrf_k_base == 40
    assert config.rrf_k_max == 200


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


def test_semantic_search_passes_query_embedding_fingerprint_to_vector_store():
    fingerprint = {
        "version": 2,
        "provider": "test",
        "model": "current",
        "dimension": 2,
        "text_schema_version": 1,
    }
    captured: dict[str, Any] = {}

    class _Embedder:
        embedding_dim = 2

        def embed_many(self, texts: Sequence[str]) -> np.ndarray:
            assert list(texts) == ["find alpha"]
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

        def fingerprint(self, *, resolve_dimension: bool = False) -> dict[str, Any]:
            assert resolve_dimension is True
            return fingerprint

    class _VectorStore:
        def search(self, query_embedding: np.ndarray, **kwargs: Any) -> list[Any]:
            captured["query_embedding"] = query_embedding
            captured["kwargs"] = kwargs
            return [({"id": "elem:alpha", "repo_name": "repo"}, 1.0)]

    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.embedder = _Embedder()
    retriever.vector_store = _VectorStore()
    retriever.filtered_vector_store = None
    retriever.min_similarity = 0.1

    results = retriever._semantic_search("find alpha", top_k=3)

    assert results[0][0]["id"] == "elem:alpha"
    assert captured["kwargs"]["query_embedding_fingerprint"] == fingerprint
    assert captured["kwargs"]["k"] == 3


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


def test_ir_graph_expansion_loads_snapshot_graphs_lazily_and_caches_union() -> None:
    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.graph_expansion_backend = "ir"
    retriever.allow_graph_builder_fallback = False

    seed_elem = _element(
        "service.py",
        element_id="func:service",
        element_type="function",
        metadata={"ir_symbol_id": "ir:service"},
    )
    helper_elem = _element(
        "service.py",
        element_id="func:helper",
        element_type="function",
        metadata={"ir_symbol_id": "ir:helper"},
    )
    retriever.graph_builder = SimpleNamespace(
        element_by_id={
            seed_elem.id: seed_elem,
            helper_elem.id: helper_elem,
        }
    )

    call_graph = nx.DiGraph()
    call_graph.add_edge("ir:service", "ir:helper")
    graphs = IRGraphs(
        dependency_graph=nx.DiGraph(),
        call_graph=call_graph,
        inheritance_graph=nx.DiGraph(),
        reference_graph=nx.DiGraph(),
        containment_graph=nx.DiGraph(),
    )

    loader_calls: list[str] = []

    def _load(snapshot_id: str) -> IRGraphs:
        loader_calls.append(snapshot_id)
        return graphs

    retriever.set_ir_graph_loader(_load, snapshot_id="snap:1")

    first = retriever._get_related_ids(
        seed_elem.id,
        {"metadata": {"ir_symbol_id": "ir:service"}},
        max_hops=1,
    )
    second = retriever._get_related_ids(
        seed_elem.id,
        {"metadata": {"ir_symbol_id": "ir:service"}},
        max_hops=1,
    )

    assert first == {"func:helper"}
    assert second == {"func:helper"}
    assert loader_calls == ["snap:1"]
    assert retriever._ir_union_graph is not None


def test_ir_graph_expansion_uses_compact_graph_view_without_networkx_walk() -> None:
    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.graph_expansion_backend = "ir"
    retriever.allow_graph_builder_fallback = False

    seed_elem = _element(
        "service.py",
        element_id="func:service",
        element_type="function",
        metadata={"ir_symbol_id": "ir:service"},
    )
    helper_elem = _element(
        "service.py",
        element_id="func:helper",
        element_type="function",
        metadata={"ir_symbol_id": "ir:helper"},
    )
    retriever.graph_builder = SimpleNamespace(
        element_by_id={seed_elem.id: seed_elem, helper_elem.id: helper_elem}
    )
    retriever.set_ir_graphs(
        IRGraphs(
            dependency_graph=IRGraphView(),
            call_graph=IRGraphView(edges=[("ir:service", "ir:helper", {})]),
            inheritance_graph=IRGraphView(),
            reference_graph=IRGraphView(),
            containment_graph=IRGraphView(),
        ),
        snapshot_id="snap:1",
    )

    with patch(
        "fastcode.app.query.selection.retriever.nx.single_source_shortest_path_length",
        side_effect=AssertionError("compact graph path must not materialize networkx"),
    ):
        related = retriever._get_related_ids(
            seed_elem.id,
            {"metadata": {"ir_symbol_id": "ir:service"}},
            max_hops=1,
        )

    assert related == {"func:helper"}
    assert isinstance(retriever._ir_union_graph, IRGraphView)


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
        "fastcode.app.query.selection.retriever.deserialize_code_element",
        side_effect=_deserialize,
    ) as mock_deserialize:
        assert retriever.load_bm25("index") is True

    assert mock_deserialize.call_count == 1
    assert calls == [payload]
    assert retriever.full_bm25_elements[0].id == "file:module"
    assert retriever.full_bm25_corpus == [["module"]]


def test_index_for_bm25_skips_empty_corpus() -> None:
    retriever = _mk_retriever()
    retriever.logger = MagicMock()

    retriever.index_for_bm25([])

    assert retriever.full_bm25 is None
    assert retriever.full_bm25_elements == []
    assert retriever.full_bm25_corpus == []
    retriever.logger.info.assert_any_call(
        "Skipped BM25 build because no indexable documents were found"
    )


def test_index_for_bm25_keeps_corpus_elements_aligned() -> None:
    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    overview = _element(
        "overview.md",
        element_id="repo:overview",
        element_type="repository_overview",
    )
    module = _element("pkg/a.py", element_id="file:a")

    retriever.index_for_bm25([overview, module])

    assert retriever.full_bm25 is not None
    assert retriever.full_bm25_elements == [module]
    assert len(retriever.full_bm25_corpus) == 1


def test_repo_overview_bm25_resets_stale_state_when_empty() -> None:
    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.vector_store = SimpleNamespace(load_repo_overviews=lambda **_: {})
    retriever.repo_overview_bm25 = object()
    retriever.repo_overview_bm25_corpus = [["stale"]]
    retriever.repo_overview_names = ["stale"]

    retriever.build_repo_overview_bm25()

    assert retriever.repo_overview_bm25 is None
    assert retriever.repo_overview_bm25_corpus == []
    assert retriever.repo_overview_names == []
    retriever.logger.warning.assert_called_once_with(
        "No repository overviews found for BM25 indexing"
    )


def test_repo_overview_bm25_skips_empty_documents() -> None:
    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.vector_store = SimpleNamespace(
        load_repo_overviews=lambda **_: {"": {"content": "", "metadata": {}}}
    )

    retriever.build_repo_overview_bm25()

    assert retriever.repo_overview_bm25 is None
    assert retriever.repo_overview_bm25_corpus == []
    assert retriever.repo_overview_names == []
    retriever.logger.info.assert_any_call(
        "Skipped repo overview BM25 build because no repositories were available"
    )


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
        "fastcode.app.query.selection.retriever.deserialize_code_element",
        side_effect=_deserialize,
    ) as mock_deserialize:
        assert retriever.reload_specific_repositories(["repo"]) is True

    assert mock_deserialize.call_count == 1
    assert calls == [payload]
    assert retriever.filtered_bm25_elements[0].id == "file:service"
    assert retriever.filtered_bm25_corpus == [["service"]]


def test_reload_specific_repositories_uses_shard_native_bm25_without_rebuild(
    tmp_path: Path,
) -> None:
    source = _mk_retriever()
    source.logger = MagicMock()
    source.persist_dir = str(tmp_path)
    source.full_bm25_corpus = [["service"], ["helper"], ["other"]]
    source.full_bm25_elements = [
        _element("service.py", element_id="file:service"),
        _element("helper.py", element_id="file:helper"),
        _element("other.py", element_id="file:other"),
    ]
    assert source.save_bm25("repo") is True

    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)
    retriever.config = {}
    retriever.embedder = SimpleNamespace(embedding_dim=3)
    retriever.filtered_vector_store = SimpleNamespace(
        clear=lambda: None,
        merge_from_index=lambda _repo_name: True,
        get_count=lambda: 2,
    )
    retriever.iterative_agent = None

    with patch(
        "fastcode.app.query.selection.retriever.BM25Okapi",
        side_effect=AssertionError("shard-native filtered reload should not rebuild"),
    ):
        assert retriever.reload_specific_repositories(["repo"]) is True

    assert retriever.filtered_bm25 is None
    assert retriever.filtered_bm25_corpus == []
    assert [elem.id for elem in retriever.filtered_bm25_elements] == [
        "file:service",
        "file:helper",
        "file:other",
    ]
    assert retriever._filtered_bm25_shard_runtime is not None
    assert retriever._keyword_search("service", top_k=1)[0][0]["id"] == "file:service"


def test_reload_specific_repositories_preserves_full_shard_runtime_state(
    tmp_path: Path,
) -> None:
    source = _mk_retriever()
    source.logger = MagicMock()
    source.persist_dir = str(tmp_path)
    source.full_bm25_corpus = [["shared", "alpha"], ["only", "repo_a"], ["other"]]
    source.full_bm25_elements = [
        _element("a.py", element_id="file:a", repo_name="repo_a"),
        _element("a2.py", element_id="file:a2", repo_name="repo_a"),
        _element("a3.py", element_id="file:a3", repo_name="repo_a"),
    ]
    assert source.save_bm25("repo_a") is True

    source.full_bm25_corpus = [["shared", "beta"], ["only", "repo_b"], ["other"]]
    source.full_bm25_elements = [
        _element("b.py", element_id="file:b", repo_name="repo_b"),
        _element("b2.py", element_id="file:b2", repo_name="repo_b"),
        _element("b3.py", element_id="file:b3", repo_name="repo_b"),
    ]
    assert source.save_bm25("repo_b") is True

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
    assert retriever.load_bm25_sources(["repo_a", "repo_b"], filtered=False) is True

    baseline_runtime = retriever._full_bm25_shard_runtime
    baseline_elements = list(retriever.full_bm25_elements)

    with patch(
        "fastcode.app.query.selection.retriever.BM25Okapi",
        side_effect=AssertionError("filtered shard reload should not rebuild"),
    ):
        assert retriever.reload_specific_repositories(["repo_b"]) is True

    assert retriever._full_bm25_shard_runtime is baseline_runtime
    assert retriever.full_bm25_elements == baseline_elements
    assert retriever._filtered_bm25_shard_runtime is not None
    assert [elem.id for elem in retriever.filtered_bm25_elements] == [
        "file:b",
        "file:b2",
        "file:b3",
    ]


def test_reload_specific_repositories_filtered_runtime_overrides_preloaded_full_runtime(
    tmp_path: Path,
) -> None:
    source = _mk_retriever()
    source.logger = MagicMock()
    source.persist_dir = str(tmp_path)
    source.full_bm25_corpus = [["shared", "alpha"], ["only", "repo_a"], ["other"]]
    source.full_bm25_elements = [
        _element("a.py", element_id="file:a", repo_name="repo_a"),
        _element("a2.py", element_id="file:a2", repo_name="repo_a"),
        _element("a3.py", element_id="file:a3", repo_name="repo_a"),
    ]
    assert source.save_bm25("repo_a") is True

    source.full_bm25_corpus = [["shared", "beta"], ["only", "repo_b"], ["other"]]
    source.full_bm25_elements = [
        _element("b.py", element_id="file:b", repo_name="repo_b"),
        _element("b2.py", element_id="file:b2", repo_name="repo_b"),
        _element("b3.py", element_id="file:b3", repo_name="repo_b"),
    ]
    assert source.save_bm25("repo_b") is True

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
    assert retriever.load_bm25_sources(["repo_a", "repo_b"], filtered=False) is True

    with patch(
        "fastcode.app.query.selection.retriever.BM25Okapi",
        side_effect=AssertionError("filtered reload should stay shard-native"),
    ):
        assert retriever.reload_specific_repositories(["repo_b"]) is True

    results = retriever._keyword_search("shared beta", top_k=3)

    assert [row["id"] for row, _score in results] == ["file:b"]
    assert retriever._full_bm25_shard_runtime is not None
    assert retriever._filtered_bm25_shard_runtime is not None


def test_reload_specific_repositories_real_vector_and_bm25_artifacts_update_filtered_runtime(
    tmp_path: Path,
) -> None:
    vector_store_a = VectorStore({"vector_store": {"persist_directory": str(tmp_path)}})
    vector_store_a.initialize(2)
    vector_store_a.add_vectors(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        [
            {
                "id": "vec:a",
                "type": "file",
                "name": "a.py",
                "file_path": "/repo_a/a.py",
                "relative_path": "a.py",
                "language": "python",
                "start_line": 1,
                "end_line": 2,
                "code": "pass\n",
                "signature": None,
                "docstring": None,
                "summary": None,
                "metadata": {},
                "repo_name": "repo_a",
                "repo_url": None,
            }
        ],
    )
    vector_store_a.save("repo_a")

    vector_store_b = VectorStore({"vector_store": {"persist_directory": str(tmp_path)}})
    vector_store_b.initialize(2)
    vector_store_b.add_vectors(
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        [
            {
                "id": "vec:b",
                "type": "file",
                "name": "b.py",
                "file_path": "/repo_b/b.py",
                "relative_path": "b.py",
                "language": "python",
                "start_line": 1,
                "end_line": 2,
                "code": "pass\n",
                "signature": None,
                "docstring": None,
                "summary": None,
                "metadata": {},
                "repo_name": "repo_b",
                "repo_url": None,
            }
        ],
    )
    vector_store_b.save("repo_b")

    source = _mk_retriever()
    source.logger = MagicMock()
    source.persist_dir = str(tmp_path)
    source.full_bm25_corpus = [["alpha"], ["repo_a"], ["other"]]
    source.full_bm25_elements = [
        _element("a.py", element_id="file:a", repo_name="repo_a"),
        _element("a2.py", element_id="file:a2", repo_name="repo_a"),
        _element("a3.py", element_id="file:a3", repo_name="repo_a"),
    ]
    assert source.save_bm25("repo_a") is True
    source.full_bm25_corpus = [["beta"], ["repo_b"], ["other"]]
    source.full_bm25_elements = [
        _element("b.py", element_id="file:b", repo_name="repo_b"),
        _element("b2.py", element_id="file:b2", repo_name="repo_b"),
        _element("b3.py", element_id="file:b3", repo_name="repo_b"),
    ]
    assert source.save_bm25("repo_b") is True

    retriever = HybridRetriever(
        {"vector_store": {"persist_directory": str(tmp_path)}},
        vector_store=VectorStore({"vector_store": {"persist_directory": str(tmp_path)}}),
        embedder=SimpleNamespace(
            embedding_dim=2,
            embed_many=lambda texts: np.asarray([[0.0, 1.0]], dtype=np.float32),
            fingerprint=lambda **_kwargs: {
                "version": 2,
                "provider": "test",
                "model": "stub",
                "dimension": 2,
                "text_schema_version": 1,
            },
        ),
        graph_builder=SimpleNamespace(),
        vector_store_factory=SimpleNamespace(
            create_vector_search_store=lambda: VectorStore(
                {"vector_store": {"persist_directory": str(tmp_path)}}
            )
        ),
    )
    retriever.logger = MagicMock()
    retriever.retrieval_backend = "local"
    retriever.pg_retrieval_store = None
    retriever.min_similarity = -1.0

    with patch(
        "fastcode.app.query.selection.retriever.BM25Okapi",
        side_effect=AssertionError("real filtered reload should stay shard-native"),
    ):
        assert retriever.reload_specific_repositories(["repo_b"]) is True

    semantic = retriever._semantic_search("beta", top_k=1, repo_filter=["repo_b"])
    keyword = retriever._keyword_search("beta", top_k=1)

    assert retriever.filtered_vector_store is not None
    assert retriever.filtered_vector_store.get_count() == 1
    assert [elem.id for elem in retriever.filtered_bm25_elements] == [
        "file:b",
        "file:b2",
        "file:b3",
    ]
    assert semantic[0][0]["id"] == "vec:b"
    assert keyword[0][0]["id"] == "file:b"


def test_load_bm25_sources_filtered_mode_builds_shard_runtime_without_rebuild(
    tmp_path: Path,
) -> None:
    source = _mk_retriever()
    source.logger = MagicMock()
    source.persist_dir = str(tmp_path)
    source.full_bm25_corpus = [["service"], ["helper"], ["other"]]
    source.full_bm25_elements = [
        _element("service.py", element_id="file:service"),
        _element("helper.py", element_id="file:helper"),
        _element("other.py", element_id="file:other"),
    ]
    assert source.save_bm25("repo") is True

    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)

    with patch(
        "fastcode.app.query.selection.retriever.BM25Okapi",
        side_effect=AssertionError("core shard runtime load should not rebuild"),
    ):
        assert retriever.load_bm25_sources(["repo"], filtered=True) is True

    assert retriever.filtered_bm25 is None
    assert retriever.filtered_bm25_corpus == []
    assert [elem.id for elem in retriever.filtered_bm25_elements] == [
        "file:service",
        "file:helper",
        "file:other",
    ]
    assert retriever._filtered_bm25_shard_runtime is not None
    assert retriever._keyword_search("service", top_k=1)[0][0]["id"] == "file:service"


def test_load_bm25_sources_sharded_path_uses_explicit_deserializer(
    tmp_path: Path,
) -> None:
    source = _mk_retriever()
    source.logger = MagicMock()
    source.persist_dir = str(tmp_path)
    source.full_bm25_corpus = [["service"], ["helper"], ["other"]]
    source.full_bm25_elements = [
        _element("service.py", element_id="file:service"),
        _element("helper.py", element_id="file:helper"),
        _element("other.py", element_id="file:other"),
    ]
    assert source.save_bm25("repo") is True

    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)

    calls: list[dict[str, Any]] = []

    def _deserialize(element_payload: dict[str, Any]) -> CodeElement:
        calls.append(element_payload)
        return _element(
            str(element_payload["relative_path"]),
            element_id=str(element_payload["id"]),
            repo_name=str(element_payload["repo_name"]),
            metadata=element_payload["metadata"],
        )

    with patch(
        "fastcode.app.query.selection.retriever.deserialize_code_element",
        side_effect=_deserialize,
    ) as mock_deserialize:
        assert retriever.load_bm25_sources(["repo"], filtered=False) is True

    assert mock_deserialize.call_count == 3
    assert [payload["id"] for payload in calls] == [
        "file:service",
        "file:helper",
        "file:other",
    ]
    assert [elem.id for elem in retriever.full_bm25_elements] == [
        "file:service",
        "file:helper",
        "file:other",
    ]


def test_load_bm25_sources_merges_multiple_shard_manifests_for_full_runtime(
    tmp_path: Path,
) -> None:
    first = _mk_retriever()
    first.logger = MagicMock()
    first.persist_dir = str(tmp_path)
    first.full_bm25_corpus = [["service"], ["helper"], ["other"]]
    first.full_bm25_elements = [
        _element("service.py", element_id="file:service", repo_name="repo_a"),
        _element("helper.py", element_id="file:helper", repo_name="repo_a"),
        _element("other.py", element_id="file:other", repo_name="repo_a"),
    ]
    assert first.save_bm25("repo_a") is True

    second = _mk_retriever()
    second.logger = MagicMock()
    second.persist_dir = str(tmp_path)
    second.full_bm25_corpus = [["beta"], ["gamma"], ["delta"]]
    second.full_bm25_elements = [
        _element("beta.py", element_id="file:beta", repo_name="repo_b"),
        _element("gamma.py", element_id="file:gamma", repo_name="repo_b"),
        _element("delta.py", element_id="file:delta", repo_name="repo_b"),
    ]
    assert second.save_bm25("repo_b") is True

    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)

    with patch(
        "fastcode.app.query.selection.retriever.BM25Okapi",
        side_effect=AssertionError("merged shard runtime should not rebuild"),
    ):
        assert retriever.load_bm25_sources(["repo_a", "repo_b"], filtered=False) is True

    assert retriever.full_bm25 is None
    assert retriever.full_bm25_corpus == []
    assert retriever._bm25_shard_name is None
    assert retriever._bm25_shard_manifest is None
    assert retriever._full_bm25_shard_runtime is not None
    assert retriever._keyword_search("service", top_k=1)[0][0]["id"] == "file:service"
    assert retriever._keyword_search("beta", top_k=1)[0][0]["id"] == "file:beta"


def test_load_bm25_sources_missing_manifest_leaves_existing_runtime_unchanged(
    tmp_path: Path,
) -> None:
    source = _mk_retriever()
    source.logger = MagicMock()
    source.persist_dir = str(tmp_path)
    source.full_bm25_corpus = [["service"], ["helper"], ["other"]]
    source.full_bm25_elements = [
        _element("service.py", element_id="file:service"),
        _element("helper.py", element_id="file:helper"),
        _element("other.py", element_id="file:other"),
    ]
    assert source.save_bm25("repo") is True

    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)
    assert retriever.load_bm25_sources(["repo"], filtered=False) is True

    baseline_runtime = retriever._full_bm25_shard_runtime
    baseline_elements = list(retriever.full_bm25_elements)

    assert retriever.load_bm25_sources(["repo", "missing"], filtered=False) is False
    assert retriever._full_bm25_shard_runtime is baseline_runtime
    assert retriever.full_bm25_elements == baseline_elements


def test_load_bm25_sources_missing_manifest_leaves_existing_filtered_runtime_unchanged(
    tmp_path: Path,
) -> None:
    source = _mk_retriever()
    source.logger = MagicMock()
    source.persist_dir = str(tmp_path)
    source.full_bm25_corpus = [["service"], ["helper"], ["other"]]
    source.full_bm25_elements = [
        _element("service.py", element_id="file:service"),
        _element("helper.py", element_id="file:helper"),
        _element("other.py", element_id="file:other"),
    ]
    assert source.save_bm25("repo") is True

    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)
    assert retriever.load_bm25_sources(["repo"], filtered=True) is True

    baseline_runtime = retriever._filtered_bm25_shard_runtime
    baseline_elements = list(retriever.filtered_bm25_elements)

    assert retriever.load_bm25_sources(["repo", "missing"], filtered=True) is False
    assert retriever._filtered_bm25_shard_runtime is baseline_runtime
    assert retriever.filtered_bm25_elements == baseline_elements


def test_filtered_shard_runtime_takes_precedence_over_full_runtime(
    tmp_path: Path,
) -> None:
    source = _mk_retriever()
    source.logger = MagicMock()
    source.persist_dir = str(tmp_path)
    source.full_bm25_corpus = [["shared", "alpha"], ["only", "repo_a"], ["other"]]
    source.full_bm25_elements = [
        _element("a.py", element_id="file:a", repo_name="repo_a"),
        _element("a2.py", element_id="file:a2", repo_name="repo_a"),
        _element("a3.py", element_id="file:a3", repo_name="repo_a"),
    ]
    assert source.save_bm25("repo_a") is True

    source.full_bm25_corpus = [["shared", "beta"], ["only", "repo_b"], ["other"]]
    source.full_bm25_elements = [
        _element("b.py", element_id="file:b", repo_name="repo_b"),
        _element("b2.py", element_id="file:b2", repo_name="repo_b"),
        _element("b3.py", element_id="file:b3", repo_name="repo_b"),
    ]
    assert source.save_bm25("repo_b") is True

    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)
    assert retriever.load_bm25_sources(["repo_a", "repo_b"], filtered=False) is True
    assert retriever.load_bm25_sources(["repo_b"], filtered=True) is True

    results = retriever._keyword_search("shared beta", top_k=3)

    assert [row["id"] for row, _score in results] == ["file:b"]
    assert retriever.filtered_bm25 is None
    assert retriever._filtered_bm25_shard_runtime is not None


def test_filtered_shard_runtime_hides_preloaded_full_runtime_hits(
    tmp_path: Path,
) -> None:
    source = _mk_retriever()
    source.logger = MagicMock()
    source.persist_dir = str(tmp_path)
    source.full_bm25_corpus = [["alpha"], ["repo_a"], ["other"]]
    source.full_bm25_elements = [
        _element("a.py", element_id="file:a", repo_name="repo_a"),
        _element("a2.py", element_id="file:a2", repo_name="repo_a"),
        _element("a3.py", element_id="file:a3", repo_name="repo_a"),
    ]
    assert source.save_bm25("repo_a") is True

    source.full_bm25_corpus = [["beta"], ["repo_b"], ["other"]]
    source.full_bm25_elements = [
        _element("b.py", element_id="file:b", repo_name="repo_b"),
        _element("b2.py", element_id="file:b2", repo_name="repo_b"),
        _element("b3.py", element_id="file:b3", repo_name="repo_b"),
    ]
    assert source.save_bm25("repo_b") is True

    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)
    assert retriever.load_bm25_sources(["repo_a", "repo_b"], filtered=False) is True
    assert retriever.load_bm25_sources(["repo_b"], filtered=True) is True

    assert retriever._keyword_search("alpha", top_k=3) == []
    assert retriever._keyword_search("beta", top_k=1)[0][0]["id"] == "file:b"


def test_merged_shard_runtime_honors_repo_filter_without_filtered_rebuild(
    tmp_path: Path,
) -> None:
    first = _mk_retriever()
    first.logger = MagicMock()
    first.persist_dir = str(tmp_path)
    first.full_bm25_corpus = [["shared", "alpha"], ["only", "repo_a"], ["other"]]
    first.full_bm25_elements = [
        _element("a.py", element_id="file:a", repo_name="repo_a"),
        _element("a2.py", element_id="file:a2", repo_name="repo_a"),
        _element("a3.py", element_id="file:a3", repo_name="repo_a"),
    ]
    assert first.save_bm25("repo_a") is True

    second = _mk_retriever()
    second.logger = MagicMock()
    second.persist_dir = str(tmp_path)
    second.full_bm25_corpus = [["shared", "beta"], ["only", "repo_b"], ["other"]]
    second.full_bm25_elements = [
        _element("b.py", element_id="file:b", repo_name="repo_b"),
        _element("b2.py", element_id="file:b2", repo_name="repo_b"),
        _element("b3.py", element_id="file:b3", repo_name="repo_b"),
    ]
    assert second.save_bm25("repo_b") is True

    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)
    assert retriever.load_bm25_sources(["repo_a", "repo_b"], filtered=False) is True

    results = retriever._keyword_search("shared beta", top_k=3, repo_filter=["repo_b"])

    assert [row["id"] for row, _score in results] == ["file:b"]
    assert all(row["repo_name"] == "repo_b" for row, _score in results)


def test_save_bm25_persists_sharded_bundle_and_loads_without_legacy_pickle(
    tmp_path: Path,
) -> None:
    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)
    retriever.full_bm25_corpus = [["alpha"], ["beta"], ["gamma"]]
    retriever.full_bm25_elements = [
        _element("pkg/a.py", element_id="elem:a"),
        _element("pkg/b.py", element_id="elem:b"),
        _element("pkg/c.py", element_id="elem:c"),
    ]
    (tmp_path / "index_bm25.pkl").write_bytes(b"legacy")

    assert retriever.save_bm25("index") is True
    assert (tmp_path / "index_bm25_manifest.json").exists()
    assert (tmp_path / "index_bm25_shards").is_dir()
    assert not (tmp_path / "index_bm25.pkl").exists()

    loaded = _mk_retriever()
    loaded.logger = MagicMock()
    loaded.persist_dir = str(tmp_path)
    with patch(
        "fastcode.app.query.selection.retriever.BM25Okapi",
        side_effect=AssertionError("shard-native load should not rebuild BM25Okapi"),
    ):
        assert loaded.load_bm25("index") is True
    assert loaded.full_bm25_elements == []
    assert loaded.full_bm25_corpus == []
    assert loaded._bm25_shard_manifest is not None
    assert loaded._keyword_search("alpha", top_k=1)[0][0]["id"] == "elem:a"


def test_save_bm25_reuses_unchanged_shards(tmp_path: Path) -> None:
    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)
    retriever.full_bm25_corpus = [["alpha"], ["beta"]]
    retriever.full_bm25_elements = [
        _element("pkg/a.py", element_id="elem:a"),
        _element("pkg/b.py", element_id="elem:b"),
    ]

    assert retriever.save_bm25("index") is True
    manifest = json.loads(
        (tmp_path / "index_bm25_manifest.json").read_text(encoding="utf-8")
    )
    shards_by_path = {
        entry["path_key"]: tmp_path / "index_bm25_shards" / entry["shard_file"]
        for entry in manifest["shards"]
    }
    a_before = shards_by_path["pkg/a.py"].stat().st_mtime_ns
    b_before = shards_by_path["pkg/b.py"].stat().st_mtime_ns

    retriever.full_bm25_corpus = [["alpha"], ["beta", "changed"]]
    retriever.full_bm25_elements = [
        _element("pkg/a.py", element_id="elem:a"),
        _element("pkg/b.py", element_id="elem:b", metadata={"v": 2}),
    ]
    assert retriever.save_bm25("index") is True

    a_after = shards_by_path["pkg/a.py"].stat().st_mtime_ns
    b_after = shards_by_path["pkg/b.py"].stat().st_mtime_ns
    assert a_after == a_before
    assert b_after >= b_before


def test_shard_native_bm25_search_reads_only_query_term_shards(
    tmp_path: Path,
) -> None:
    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)
    retriever.full_bm25_corpus = [["alpha"], ["beta"], ["gamma"]]
    retriever.full_bm25_elements = [
        _element("pkg/a.py", element_id="elem:a"),
        _element("pkg/b.py", element_id="elem:b"),
        _element("pkg/c.py", element_id="elem:c"),
    ]
    assert retriever.save_bm25("index") is True

    loaded = _mk_retriever()
    loaded.logger = MagicMock()
    loaded.persist_dir = str(tmp_path)
    assert loaded.load_bm25("index") is True

    real_pickle_load = pickle.load
    loaded_shards = 0

    def _counting_pickle_load(handle: Any) -> Any:
        nonlocal loaded_shards
        loaded_shards += 1
        return real_pickle_load(handle)

    with patch(
        "fastcode.app.query.selection.retriever.pickle.load", side_effect=_counting_pickle_load
    ):
        results = loaded._keyword_search("alpha", top_k=3)

    assert [row["id"] for row, _score in results] == ["elem:a"]
    assert loaded_shards == 1


def test_shard_native_bm25_preserves_legacy_bm25_ranking(
    tmp_path: Path,
) -> None:
    retriever = _mk_retriever()
    retriever.logger = MagicMock()
    retriever.persist_dir = str(tmp_path)
    corpus = [
        ["alpha", "rare", "rare"],
        ["alpha", "ordinary"],
        ["beta", "ordinary"],
        ["gamma", "ordinary"],
        ["delta", "ordinary"],
    ]
    retriever.full_bm25_corpus = corpus
    retriever.full_bm25_elements = [
        _element("pkg/a.py", element_id="elem:a"),
        _element("pkg/b.py", element_id="elem:b"),
        _element("pkg/c.py", element_id="elem:c"),
        _element("pkg/d.py", element_id="elem:d"),
        _element("pkg/e.py", element_id="elem:e"),
    ]
    assert retriever.save_bm25("index") is True

    loaded = _mk_retriever()
    loaded.logger = MagicMock()
    loaded.persist_dir = str(tmp_path)
    assert loaded.load_bm25("index") is True

    query_tokens = ["alpha", "rare"]
    expected_scores = BM25Okapi(corpus).get_scores(query_tokens)
    expected = [
        (retriever.full_bm25_elements[index].id, float(score))
        for index, score in sorted(
            enumerate(expected_scores),
            key=lambda item: (-float(item[1]), item[0]),
        )
        if float(score) > 0
    ]
    actual = [
        (row["id"], score)
        for row, score in loaded._keyword_search("alpha rare", top_k=5)
    ]

    assert [item[0] for item in actual] == [item[0] for item in expected]
    for (_actual_id, actual_score), (_expected_id, expected_score) in zip(
        actual,
        expected,
        strict=True,
    ):
        assert actual_score == pytest.approx(expected_score)


def test_save_bm25_incremental_reuses_previous_artifact_shards(
    tmp_path: Path,
) -> None:
    previous = _mk_retriever()
    previous.logger = MagicMock()
    previous.persist_dir = str(tmp_path)
    previous.full_bm25_corpus = [["alpha"], ["beta"], ["gamma"]]
    previous.full_bm25_elements = [
        _element("pkg/a.py", element_id="elem:a"),
        _element("pkg/b.py", element_id="elem:b"),
        _element("pkg/c.py", element_id="elem:c"),
    ]

    assert previous.save_bm25("prev") is True
    prev_manifest = json.loads(
        (tmp_path / "prev_bm25_manifest.json").read_text(encoding="utf-8")
    )
    prev_shards = {
        entry["path_key"]: tmp_path / "prev_bm25_shards" / entry["shard_file"]
        for entry in prev_manifest["shards"]
    }

    current = _mk_retriever()
    current.logger = MagicMock()
    current.persist_dir = str(tmp_path)
    current.full_bm25_corpus = [["alpha"], ["beta", "changed"], ["gamma"]]
    current.full_bm25_elements = [
        _element("pkg/a.py", element_id="elem:a"),
        _element("pkg/b.py", element_id="elem:b", metadata={"v": 2}),
        _element("pkg/c.py", element_id="elem:c"),
    ]

    stats = current.save_bm25_incremental(
        "next",
        previous_name="prev",
        reusable_path_keys={"pkg/a.py"},
    )

    assert stats["bm25_shards_reused"] == 1
    next_manifest = json.loads(
        (tmp_path / "next_bm25_manifest.json").read_text(encoding="utf-8")
    )
    next_shards = {
        entry["path_key"]: tmp_path / "next_bm25_shards" / entry["shard_file"]
        for entry in next_manifest["shards"]
    }
    assert next_shards["pkg/a.py"].read_bytes() == prev_shards["pkg/a.py"].read_bytes()
    assert next_shards["pkg/b.py"].read_bytes() != prev_shards["pkg/b.py"].read_bytes()

    loaded = _mk_retriever()
    loaded.logger = MagicMock()
    loaded.persist_dir = str(tmp_path)
    assert loaded.load_bm25("next") is True
    assert loaded.full_bm25_elements == []
    assert loaded._bm25_shard_manifest is not None
    assert loaded._keyword_search("changed", top_k=1)[0][0]["id"] == "elem:b"


def test_save_bm25_incremental_refuses_incompatible_manifest(
    tmp_path: Path,
) -> None:
    previous = _mk_retriever()
    previous.logger = MagicMock()
    previous.persist_dir = str(tmp_path)
    previous.full_bm25_corpus = [["alpha"]]
    previous.full_bm25_elements = [_element("pkg/a.py", element_id="elem:a")]

    assert previous.save_bm25("prev") is True
    manifest_path = tmp_path / "prev_bm25_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    current = _mk_retriever()
    current.logger = MagicMock()
    current.persist_dir = str(tmp_path)
    current.full_bm25_corpus = [["alpha", "changed"]]
    current.full_bm25_elements = [
        _element("pkg/a.py", element_id="elem:a", metadata={"v": 2})
    ]

    stats = current.save_bm25_incremental(
        "next",
        previous_name="prev",
        reusable_path_keys={"pkg/a.py"},
    )

    assert stats["bm25_shards_reused"] == 0
    assert stats["bm25_shards_written"] == 1
