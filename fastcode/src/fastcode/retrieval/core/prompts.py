"""Pure prompt formatting functions.

No I/O, no logging, no mutation.  Accept plain dicts, return strings.
"""

from __future__ import annotations

import json
from typing import Any


def format_tool_call_history(
    history: list[dict[str, Any]] | None,
    current_round: int,
) -> str:
    """Format tool call history up to (but not including) *current_round*.

    Parameters
    ----------
    history:
        List of ``{"round": int, "tool": str, "parameters": dict}`` records,
        or ``None`` when no history attribute exists yet.
    current_round:
        Only entries with ``round < current_round`` are included.

    Returns
    -------
    str
        ``"None"`` when there is nothing to show, otherwise one line per entry:
        ``"- Round N: tool_name {params_json}"``.
    """
    if not history:
        return "None"

    lines: list[str] = []
    for entry in history:
        if entry.get("round", 0) >= current_round:
            continue
        tool_name = entry.get("tool", "")
        params = entry.get("parameters", {})
        params_text = json.dumps(params, ensure_ascii=True, sort_keys=True)
        lines.append(f"- Round {entry['round']}: {tool_name} {params_text}")

    return "\n".join(lines) if lines else "None"


def format_elements_with_metadata(elements: list[dict[str, Any]]) -> str:
    """Format retrieval elements grouped by file with source / line / signature info.

    Parameters
    ----------
    elements:
        List of element-data dicts, each containing at least an ``"element"``
        sub-dict with ``repo_name``, ``relative_path``, ``start_line``,
        ``end_line``, ``type``, and optional ``signature``.  The outer dict
        carries ``total_score``, ``agent_found``, ``llm_file_selected``, and
        optional ``related_to``.

    Returns
    -------
    str
        Empty string when *elements* is empty.  Otherwise a multi-line block
        grouped by ``(repo_name, relative_path)`` showing aggregated source
        types, total line counts, related-to references, and up to 5
        signatures per file group.
    """
    if not elements:
        return ""

    lines: list[str] = []

    # Group by (repo_name, relative_path) to avoid conflicts when multiple
    # repos have the same file names.
    file_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for elem_data in elements:
        elem = elem_data.get("element", {})
        repo_name = elem.get("repo_name", "")
        relative_path = elem.get("relative_path", elem.get("file_path", ""))

        group_key = (repo_name, relative_path)
        if group_key not in file_groups:
            file_groups[group_key] = []
        file_groups[group_key].append(elem_data)

    for i, (group_key, elem_list) in enumerate(file_groups.items(), 1):
        repo_name, relative_path = group_key

        display_path = f"{repo_name}/{relative_path}" if repo_name else relative_path
        lines.append(f"\n{i}. {display_path}")

        # Aggregate metadata
        sources: set[str] = set()
        related_to: set[str] = set()
        total_lines = 0

        for elem_data in elem_list:
            elem = elem_data.get("element", {})

            # Determine source
            if elem_data.get("agent_found"):
                sources.add("Tool")
            elif elem_data.get("llm_file_selected"):
                sources.add("LLM Selection")
            elif elem_data.get("related_to"):
                sources.add("Graph")
                related_to.add(elem_data.get("related_to", ""))
            else:
                sources.add("Retrieval")

            # Calculate lines
            start = elem.get("start_line", 0)
            end = elem.get("end_line", 0)
            if end > start:
                total_lines += end - start + 1

        if repo_name:
            lines.append(f"   Repo: {repo_name}")
        lines.append(f"   Type: {elem_list[0]['element'].get('type', 'unknown')}")
        lines.append(f"   Source: {', '.join(sources)}")
        if total_lines > 0:
            lines.append(f"   Lines: {total_lines}")
        if related_to:
            lines.append(f"   Related to: {', '.join(related_to)}")

        # Show signatures for class/function elements (max 5)
        for elem_data in elem_list[:5]:
            elem = elem_data.get("element", {})
            if elem.get("signature"):
                lines.append(f"   - {elem['signature']}")

    return "\n".join(lines)
