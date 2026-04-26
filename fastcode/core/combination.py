"""Pure result combination — merges semantic, keyword, pseudocode search results."""

from __future__ import annotations

from typing import Any


def combine_results(
    semantic_results: list[tuple[dict[str, Any], float]],
    keyword_results: list[tuple[dict[str, Any], float]],
    pseudocode_results: list[tuple[dict[str, Any], float]] | None = None,
    *,
    semantic_weight: float = 1.0,
    keyword_weight: float = 1.0,
    pseudocode_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """Combine semantic, keyword, and pseudocode search results.

    Merges by element ID, normalizes BM25 scores to 0-1, applies
    source-priority boost, sorts by total_score descending.
    """
    combined: dict[str, dict[str, Any]] = {}

    for metadata, score in semantic_results:
        elem_id = metadata.get("id")
        if elem_id:
            combined[elem_id] = {
                "element": metadata,
                "semantic_score": score * semantic_weight,
                "keyword_score": 0.0,
                "pseudocode_score": 0.0,
                "graph_score": 0.0,
                "total_score": score * semantic_weight,
            }

    if pseudocode_results:
        for metadata, score in pseudocode_results:
            elem_id = metadata.get("id")
            if elem_id:
                pseudocode_contrib = score * pseudocode_weight
                if elem_id in combined:
                    combined[elem_id]["pseudocode_score"] = pseudocode_contrib
                    combined[elem_id]["total_score"] += pseudocode_contrib
                else:
                    combined[elem_id] = {
                        "element": metadata,
                        "semantic_score": 0.0,
                        "keyword_score": 0.0,
                        "pseudocode_score": pseudocode_contrib,
                        "graph_score": 0.0,
                        "total_score": pseudocode_contrib,
                    }

    if keyword_results:
        max_bm25 = max(score for _, score in keyword_results) if keyword_results else 0
        if max_bm25 > 0:
            for metadata, score in keyword_results:
                elem_id = metadata.get("id")
                if elem_id:
                    normalized_score = (score / max_bm25) * keyword_weight
                    if elem_id in combined:
                        combined[elem_id]["keyword_score"] = normalized_score
                        combined[elem_id]["total_score"] += normalized_score
                    else:
                        combined[elem_id] = {
                            "element": metadata,
                            "semantic_score": 0.0,
                            "keyword_score": normalized_score,
                            "pseudocode_score": 0.0,
                            "graph_score": 0.0,
                            "total_score": normalized_score,
                        }

    results = list(combined.values())

    for result in results:
        elem = result.get("element", {})
        meta = elem.get("metadata", {}) if isinstance(elem, dict) else {}
        source_priority = meta.get("source_priority", 0)
        try:
            source_priority = float(source_priority)
        except Exception:
            source_priority = 0.0
        boost = 1.0 + min(max(source_priority, 0.0), 100.0) / 200.0
        result["total_score"] *= boost

    results.sort(key=lambda x: x["total_score"], reverse=True)
    return results
