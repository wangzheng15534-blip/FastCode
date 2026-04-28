"""Pure filtering, diversification, and reranking — extracted from retriever.py."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any


def apply_filters(
    results: list[dict[str, Any]],
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter results by language, type, file_path, and snapshot_id."""
    filtered: list[dict[str, Any]] = []
    for result in results:
        elem = result["element"]
        if "language" in filters and elem.get("language") != filters["language"]:
            continue
        if "type" in filters and elem.get("type") != filters["type"]:
            continue
        if "file_path" in filters and filters["file_path"] not in elem.get(
            "relative_path", ""
        ):
            continue
        if "snapshot_id" in filters:
            elem_snapshot = elem.get("snapshot_id") or (
                elem.get("metadata", {}) or {}
            ).get("snapshot_id")
            if elem_snapshot != filters["snapshot_id"]:
                continue
        filtered.append(result)
    return filtered


def diversify(
    results: list[dict[str, Any]],
    diversity_penalty: float,
) -> list[dict[str, Any]]:
    """Penalize results from already-seen files to improve diversity."""
    # When penalty is 0, preserve original order (matching original behavior).
    if not results or diversity_penalty == 0:
        return results

    diversified: list[dict[str, Any]] = []
    seen_files: set[str] = set()

    for item in results:
        file_path = item["element"].get("file_path", "")
        if file_path in seen_files:
            penalty_factor = 1 - diversity_penalty
            penalized = {
                **item,
                "total_score": item["total_score"] * penalty_factor,
                "semantic_score": item["semantic_score"] * penalty_factor,
                "keyword_score": item["keyword_score"] * penalty_factor,
                "pseudocode_score": item["pseudocode_score"] * penalty_factor,
                "graph_score": item["graph_score"] * penalty_factor,
            }
        else:
            seen_files.add(file_path)
            penalized = item
        diversified.append(penalized)

    diversified.sort(key=lambda x: x["total_score"], reverse=True)
    return diversified


def final_repo_filter(
    results: list[dict[str, Any]],
    repo_filter: list[str],
    return_count: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], int]:
    """Filter results to only include elements from allowed repositories."""
    if not repo_filter:
        return (results, 0) if return_count else results

    filtered_results: list[dict[str, Any]] = []
    filtered_count = 0
    for result in results:
        elem = result["element"]
        repo_name = elem.get("repo_name", "")
        if repo_name in repo_filter:
            filtered_results.append(result)
        else:
            filtered_count += 1

    return (filtered_results, filtered_count) if return_count else filtered_results


def rerank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-rank results by element type preferences."""
    type_weights = {
        "function": 1.2,
        "class": 1.1,
        "file": 0.9,
        "documentation": 0.8,
        "design_document": 0.95,
    }

    reranked: list[dict[str, Any]] = []
    for result in results:
        elem_type = result["element"].get("type", "")
        weight = type_weights.get(elem_type, 1.0)
        reranked.append(
            {
                **result,
                "total_score": result["total_score"] * weight,
                "semantic_score": result["semantic_score"] * weight,
                "keyword_score": result["keyword_score"] * weight,
                "pseudocode_score": result["pseudocode_score"] * weight,
                "graph_score": result["graph_score"] * weight,
            }
        )

    reranked.sort(key=lambda x: x["total_score"], reverse=True)
    return reranked
