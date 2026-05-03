from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastcode.iterative_agent import IterativeAgent
from fastcode.query_processor import ProcessedQuery


def test_standard_retrieval_applies_filters_in_iterative_mode() -> None:
    filtered_marker = [{"element": {"relative_path": "src/kept.py"}, "total_score": 1.0}]
    retriever = SimpleNamespace(
        _semantic_search=lambda *args, **kwargs: [
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
        _keyword_search=lambda *args, **kwargs: [],
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
        _apply_filters=lambda results, filters: filtered_marker,
        _diversify=lambda results: results,
        max_results=10,
    )
    agent = IterativeAgent.__new__(IterativeAgent)
    agent.retriever = retriever
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
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

    assert results == filtered_marker
