"""Pure explore-code response shaping.

This module is deterministic retrieval-domain policy: it groups retrieved hits
into source-oriented snippets with stable refs, line numbers, freshness, and
expansion hints. It performs no I/O and imports no query/API/MCP shells.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast

from fastcode.retrieval.contracts import Hit

_DETAIL_LEVELS = {"minimal", "standard", "full"}
_DEFAULT_CODE_CHARS = {
    "minimal": 0,
    "standard": 1200,
    "full": 6000,
}
_DEFAULT_MAX_SNIPPETS = {
    "minimal": 8,
    "standard": 12,
    "full": 24,
}


def normalize_detail_level(detail_level: str | None) -> str:
    """Normalize the public detail-level option."""
    normalized = str(detail_level or "standard").strip().lower()
    if normalized not in _DETAIL_LEVELS:
        allowed = ", ".join(sorted(_DETAIL_LEVELS))
        msg = f"detail_level must be one of: {allowed}"
        raise ValueError(msg)
    return normalized


def build_explore_code_payload(
    *,
    question: str,
    hits: Sequence[Hit],
    snapshot_id: str | None = None,
    artifact_key: str | None = None,
    repo_filter: Sequence[str] | None = None,
    detail_level: str = "standard",
    max_snippets: int | None = None,
    max_code_chars: int | None = None,
) -> dict[str, Any]:
    """Build a grouped, product-facing explore-code payload from retrieval hits."""
    level = normalize_detail_level(detail_level)
    snippet_limit = _positive_int(
        max_snippets,
        default=_DEFAULT_MAX_SNIPPETS[level],
        upper_bound=50,
    )
    code_char_limit = _positive_int(
        max_code_chars,
        default=_DEFAULT_CODE_CHARS[level],
        upper_bound=20000,
        allow_zero=True,
    )
    fresh = "fresh" if snapshot_id else "unknown"

    prepared_hits = _sorted_hits(hits)
    groups: list[dict[str, Any]] = []
    group_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    included = 0
    for hit in prepared_hits:
        if included >= snippet_limit:
            break
        path = _hit_path(hit)
        if not path:
            continue
        repo = hit.repo_name
        key = (repo, path)
        group = group_by_key.get(key)
        if group is None:
            new_group: dict[str, Any] = {
                "ref_id": f"g{len(groups) + 1}",
                "repo": repo,
                "file": path,
                "language": hit.language,
                "best_score": _rounded_score(_hit_score(hit)),
                "fresh": fresh,
                "snippets": [],
            }
            group = new_group
            group_by_key[key] = group
            groups.append(group)
        else:
            group["best_score"] = max(
                float(group.get("best_score") or 0.0),
                _rounded_score(_hit_score(hit)),
            )
            if not group.get("language") and hit.language:
                group["language"] = hit.language

        included += 1
        snippets = cast(list[dict[str, Any]], group["snippets"])
        snippets.append(
            _snippet_payload(
                hit,
                ref_id=f"e{included}",
                question=question,
                fresh=fresh,
                code_char_limit=code_char_limit,
                detail_level=level,
            )
        )

    for group in groups:
        snippets = cast(list[dict[str, Any]], group["snippets"])
        snippets.sort(
            key=lambda item: (
                int(item.get("start_line") or 0),
                str(item.get("name") or ""),
                str(item.get("ref_id") or ""),
            )
        )

    returned = sum(
        len(cast(list[dict[str, Any]], group["snippets"])) for group in groups
    )
    omitted = max(0, len([hit for hit in prepared_hits if _hit_path(hit)]) - returned)
    return {
        "query": question,
        "snapshot_id": snapshot_id,
        "artifact_key": artifact_key,
        "repo_filter": [str(item) for item in repo_filter or ()],
        "detail_level": level,
        "freshness": {
            "state": fresh,
            "snapshot_id": snapshot_id,
        },
        "completeness": {
            "state": "partial" if omitted else "complete",
            "returned_snippets": returned,
            "omitted_snippets": omitted,
            "returned_groups": len(groups),
        },
        "groups": groups,
        "next_actions": _next_actions(groups),
    }


def _sorted_hits(hits: Sequence[Hit]) -> list[Hit]:
    return sorted(
        hits,
        key=lambda hit: (
            -_hit_score(hit),
            hit.repo_name,
            _hit_path(hit),
            hit.start_line,
            hit.element_id,
        ),
    )


def _hit_path(hit: Hit) -> str:
    return hit.relative_path or hit.file_path


def _hit_score(hit: Hit) -> float:
    return hit.total_score or hit.score


def _rounded_score(score: float) -> float:
    return round(float(score), 6)


def _positive_int(
    value: int | None,
    *,
    default: int,
    upper_bound: int,
    allow_zero: bool = False,
) -> int:
    result = default if value is None else int(value)
    lower_bound = 0 if allow_zero else 1
    return max(lower_bound, min(result, upper_bound))


def _snippet_payload(
    hit: Hit,
    *,
    ref_id: str,
    question: str,
    fresh: str,
    code_char_limit: int,
    detail_level: str,
) -> dict[str, Any]:
    code = str(hit.element_extra.get("code") or "")
    line_numbered_code = (
        _line_numbered_code(
            code,
            start_line=hit.start_line,
            max_chars=code_char_limit,
        )
        if code_char_limit
        else ""
    )
    payload: dict[str, Any] = {
        "ref_id": ref_id,
        "name": hit.element_name,
        "type": hit.element_type,
        "start_line": hit.start_line,
        "end_line": hit.end_line,
        "lines": _line_label(hit.start_line, hit.end_line),
        "score": _rounded_score(_hit_score(hit)),
        "source": hit.source_kind.value,
        "fresh": fresh,
        "language": hit.language,
        "expansion": {
            "tool": "explore_code",
            "arguments": {
                "query": question,
                "filters": {"file_path": _hit_path(hit)},
                "detail_level": "full",
            },
        },
    }
    if hit.signature:
        payload["signature"] = hit.signature
    if detail_level != "minimal":
        payload["code"] = line_numbered_code
        payload["omitted_code_chars"] = max(0, len(code) - code_char_limit)
    return payload


def _line_label(start_line: int, end_line: int) -> str:
    if start_line > 0 and end_line > 0:
        return f"{start_line}-{end_line}"
    if start_line > 0:
        return str(start_line)
    return ""


def _line_numbered_code(code: str, *, start_line: int, max_chars: int) -> str:
    if not code or max_chars <= 0:
        return ""
    clipped = code[:max_chars]
    if len(code) > max_chars:
        clipped = clipped.rstrip() + "\n... (truncated)"
    first_line = start_line if start_line > 0 else 1
    return "\n".join(
        f"{line_no:>4} | {line}"
        for line_no, line in enumerate(clipped.splitlines(), first_line)
    )


def _next_actions(groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for group in groups:
        file_path = str(group.get("file") or "")
        if not file_path:
            continue
        actions.append(
            {
                "tool": "get_file_summary",
                "arguments": {"file_path": file_path},
                "reason": "Inspect the file structure for this grouped result.",
            }
        )
        if len(actions) >= 3:
            break
    return actions
