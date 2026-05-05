from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from fastcode.ir.element import CodeElement
from fastcode.query.processor import ProcessedQuery
from fastcode.retrieval.iterative import IterativeAgent


def _make_iterative_agent(
    retriever_overrides: dict[str, Any] | None = None,
) -> IterativeAgent:
    """Build an IterativeAgent with only the attributes _perform_standard_retrieval needs.

    Tested method dependency contract (iterative_agent.py:876):
      self.retriever — calls _semantic_search, _keyword_search, _combine_results,
                        _rerank, _apply_filters, _diversify, .max_results
      self.logger    — calls .info(), .debug()
    """
    retriever = SimpleNamespace(
        _semantic_search=lambda *a, **kw: [
            (
                {
                    "id": "file:kept",
                    "type": "file",
                    "relative_path": "src/kept.py",
                    "language": "python",
                },
                1.0,
            ),
            (
                {
                    "id": "file:dropped",
                    "type": "file",
                    "relative_path": "src/dropped.js",
                    "language": "javascript",
                },
                0.9,
            ),
        ],
        _keyword_search=lambda *a, **kw: [],
        _combine_results=lambda semantic, keyword, pseudocode: [
            {
                "element": metadata,
                "semantic_score": score,
                "keyword_score": 0.0,
                "pseudocode_score": 0.0,
                "graph_score": 0.0,
                "total_score": score,
            }
            for metadata, score in semantic
        ],
        _rerank=lambda query, results: results,
        _apply_filters=lambda results, filters: [
            {"element": {"relative_path": "src/kept.py"}, "total_score": 1.0}
        ],
        _diversify=lambda results: results,
        max_results=10,
    )
    for attr, value in (retriever_overrides or {}).items():
        setattr(retriever, attr, value)
    agent = IterativeAgent.__new__(IterativeAgent)
    agent.retriever = retriever
    agent.logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )
    return agent


def _element(
    relative_path: str,
    *,
    element_id: str,
    element_type: str = "file",
    name: str | None = None,
    repo_name: str = "repo",
    metadata: dict[str, Any] | None = None,
) -> CodeElement:
    return CodeElement(
        id=element_id,
        type=element_type,
        name=name or relative_path,
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


def test_standard_retrieval_applies_filters_in_iterative_mode() -> None:
    agent = _make_iterative_agent()
    processed_query = ProcessedQuery(
        original="Where is config loaded?",
        expanded="Where is config loaded?",
        keywords=[],
        intent="where",
        subqueries=[],
        filters={"language": "python"},
    )

    results = agent._perform_standard_retrieval(
        processed_query,
        {"language": "python"},
        repo_filter=None,
    )

    assert len(results) == 1
    assert results[0]["element"]["relative_path"] == "src/kept.py"


def test_standard_retrieval_returns_empty_when_no_results() -> None:
    agent = _make_iterative_agent(
        retriever_overrides={
            "_semantic_search": lambda *a, **kw: [],
            "_apply_filters": lambda results, filters: results,
        },
    )
    processed_query = ProcessedQuery(
        original="nonexistent",
        expanded="nonexistent",
        keywords=[],
        intent="find",
        subqueries=[],
        filters={},
    )

    results = agent._perform_standard_retrieval(
        processed_query,
        filters=None,
        repo_filter=None,
    )

    assert results == []


def test_retrieve_indexed_elements_for_file_avoids_code_element_to_dict() -> None:
    agent = IterativeAgent.__new__(IterativeAgent)
    agent.logger = SimpleNamespace(
        debug=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
    )
    agent.bm25_elements = [
        _element(
            "src/config.py",
            element_id="file:config",
            metadata={"stable_unit_id": "unit:file:config"},
        )
    ]

    with patch.object(
        CodeElement,
        "to_dict",
        autospec=True,
        side_effect=AssertionError(
            "iterative file retrieval must not call CodeElement.to_dict()"
        ),
    ):
        results = agent._retrieve_indexed_elements_for_file("repo", "src/config.py")

    assert len(results) == 1
    assert results[0]["element"]["id"] == "file:config"
    assert results[0]["element"]["metadata"] == {"stable_unit_id": "unit:file:config"}
    assert results[0]["agent_found"] is True


def test_convert_selections_to_elements_avoids_code_element_to_dict() -> None:
    agent = IterativeAgent.__new__(IterativeAgent)
    agent.logger = SimpleNamespace(
        debug=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
    )
    agent.path_utils = SimpleNamespace(
        detect_repo_name_from_path=lambda path, known_repos: "repo",
        normalize_path_with_repo=lambda candidate_path, repo_name: candidate_path,
    )
    agent.bm25_elements = [
        _element(
            "src/service.py",
            element_id="func:load_config",
            element_type="function",
            name="load_config",
            metadata={"class_name": None, "is_method": False},
        )
    ]

    with patch.object(
        CodeElement,
        "to_dict",
        autospec=True,
        side_effect=AssertionError(
            "iterative selection conversion must not call CodeElement.to_dict()"
        ),
    ):
        results = agent._convert_selections_to_elements(
            [
                {
                    "file_path": "src/service.py",
                    "type": "function",
                    "name": "load_config",
                    "repo_name": "repo",
                }
            ],
            [{"file_path": "src/service.py", "repo_name": "repo"}],
        )

    assert len(results) == 1
    assert results[0]["element"]["id"] == "func:load_config"
    assert results[0]["element"]["metadata"] == {
        "class_name": None,
        "is_method": False,
    }
    assert results[0]["selection_granularity"] == "function"
