"""Pure filtering, diversification, and reranking."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from fastcode.retrieval.contracts import ElementFilter, Hit


def apply_filters(
    results: list[Hit],
    filters: ElementFilter,
) -> list[Hit]:
    """Filter results by language, type, file_path, and snapshot_id."""
    filtered: list[Hit] = []
    for hit in results:
        if filters.language is not None and hit.language != filters.language:
            continue
        if filters.element_type is not None and hit.element_type != filters.element_type:
            continue
        if filters.file_path is not None and filters.file_path not in hit.relative_path:
            continue
        if filters.snapshot_id is not None:
            hit_snapshot = hit.snapshot_id or str(hit.metadata.get("snapshot_id") or "")
            if hit_snapshot != filters.snapshot_id:
                continue
        filtered.append(hit)
    return filtered


def diversify(
    results: list[Hit],
    diversity_penalty: float,
) -> list[Hit]:
    """Penalize results from already-seen files to improve diversity."""
    # When penalty is 0, preserve original order (matching original behavior).
    if not results or diversity_penalty == 0:
        return results

    diversified: list[Hit] = []
    seen_files: set[str] = set()

    for hit in results:
        file_path = hit.file_path
        if file_path in seen_files:
            penalty_factor = 1 - diversity_penalty
            penalized = hit.scaled_scores(penalty_factor)
        else:
            seen_files.add(file_path)
            penalized = hit
        diversified.append(penalized)

    diversified.sort(key=lambda hit: hit.total_score, reverse=True)
    return diversified


def final_repo_filter(
    results: list[Hit],
    repo_filter: list[str],
    return_count: bool = False,
) -> list[Hit] | tuple[list[Hit], int]:
    """Filter results to only include elements from allowed repositories."""
    if not repo_filter:
        return (results, 0) if return_count else results

    filtered_results: list[Hit] = []
    filtered_count = 0
    for hit in results:
        if hit.repo_name in repo_filter:
            filtered_results.append(hit)
        else:
            filtered_count += 1

    return (filtered_results, filtered_count) if return_count else filtered_results


def rerank(results: list[Hit]) -> list[Hit]:
    """Re-rank results by element type preferences."""
    type_weights = {
        "function": 1.2,
        "class": 1.1,
        "file": 0.9,
        "documentation": 0.8,
        "design_document": 0.95,
    }

    reranked: list[Hit] = []
    for hit in results:
        weight = type_weights.get(hit.element_type, 1.0)
        reranked.append(hit.scaled_scores(weight))

    reranked.sort(key=lambda hit: hit.total_score, reverse=True)
    return reranked
