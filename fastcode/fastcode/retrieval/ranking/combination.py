"""Pure result combination -- merges semantic, keyword, pseudocode search results."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from fastcode.ir.element import CodeElementMeta
from fastcode.retrieval.contracts import Hit, RetrievalSource


def combine_results(
    semantic_results: list[tuple[CodeElementMeta, float]],
    keyword_results: list[tuple[CodeElementMeta, float]],
    pseudocode_results: list[tuple[CodeElementMeta, float]] | None = None,
    *,
    semantic_weight: float = 1.0,
    keyword_weight: float = 1.0,
    pseudocode_weight: float = 0.4,
) -> list[Hit]:
    """Combine semantic, keyword, and pseudocode search results.

    Merges by element ID, normalizes BM25 scores to 0-1, applies
    source-priority boost, sorts by total_score descending.
    """
    combined: dict[str, Hit] = {}

    for metadata, score in semantic_results:
        elem_id = metadata.get("id")
        if elem_id:
            weighted_score = score * semantic_weight
            combined[elem_id] = Hit.from_element(
                metadata,
                score=weighted_score,
                source=RetrievalSource.SEMANTIC,
                semantic_score=weighted_score,
                total_score=weighted_score,
            )

    if pseudocode_results:
        for metadata, score in pseudocode_results:
            elem_id = metadata.get("id")
            if elem_id:
                pseudocode_contrib = score * pseudocode_weight
                if elem_id in combined:
                    current = combined[elem_id]
                    combined[elem_id] = current.with_scores(
                        pseudocode_score=pseudocode_contrib,
                        total_score=current.total_score + pseudocode_contrib,
                    )
                else:
                    combined[elem_id] = Hit.from_element(
                        metadata,
                        score=pseudocode_contrib,
                        source=RetrievalSource.PSEUDOCODE,
                        pseudocode_score=pseudocode_contrib,
                        total_score=pseudocode_contrib,
                    )

    if keyword_results:
        max_bm25 = max(score for _, score in keyword_results) if keyword_results else 0
        if max_bm25 > 0:
            for metadata, score in keyword_results:
                elem_id = metadata.get("id")
                if elem_id:
                    normalized_score = (score / max_bm25) * keyword_weight
                    if elem_id in combined:
                        current = combined[elem_id]
                        combined[elem_id] = current.with_scores(
                            keyword_score=normalized_score,
                            total_score=current.total_score + normalized_score,
                        )
                    else:
                        combined[elem_id] = Hit.from_element(
                            metadata,
                            score=normalized_score,
                            source=RetrievalSource.KEYWORD,
                            keyword_score=normalized_score,
                            total_score=normalized_score,
                        )

    results = list(combined.values())

    boosted_results: list[Hit] = []
    for hit in results:
        source_priority = hit.metadata.get("source_priority", 0)
        try:
            source_priority = float(source_priority)
        except Exception:
            source_priority = 0.0
        boost = 1.0 + min(max(source_priority, 0.0), 100.0) / 200.0
        boosted_results.append(hit.with_scores(total_score=hit.total_score * boost))

    boosted_results.sort(key=lambda hit: hit.total_score, reverse=True)
    return boosted_results
