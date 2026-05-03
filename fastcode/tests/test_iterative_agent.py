from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastcode.iterative_agent import IterativeAgent
from fastcode.query_processor import ProcessedQuery


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
