"""Pure prompt formatting functions.

No I/O, no logging, no mutation.  Accept typed retrieval records, return strings.
"""

from __future__ import annotations

import json

from fastcode.retrieval.contracts import Hit, RetrievalSource, ToolHistoryEntry


def format_tool_call_history(history: tuple[ToolHistoryEntry, ...], current_round: int) -> str:
    """Format tool call history up to (but not including) *current_round*.

    ``"None"`` is returned when there is nothing to show.
    """
    if not history:
        return "None"

    lines: list[str] = []
    for entry in history:
        if entry.round >= current_round:
            continue
        params_text = json.dumps(entry.parameters, ensure_ascii=True, sort_keys=True)
        lines.append(f"- Round {entry.round}: {entry.tool} {params_text}")

    return "\n".join(lines) if lines else "None"


def _source_label(hit: Hit) -> str:
    if hit.agent_found:
        return "Tool"
    if hit.llm_selected:
        return "LLM Selection"
    if hit.related_to:
        return "Graph"
    if hit.source_kind == RetrievalSource.TOOL:
        return "Tool"
    if hit.source_kind == RetrievalSource.LLM_SELECTION:
        return "LLM Selection"
    if hit.source_kind == RetrievalSource.GRAPH:
        return "Graph"
    return "Retrieval"


def format_elements_with_metadata(elements: tuple[Hit, ...]) -> str:
    """Format retrieval elements grouped by file with source / line / signature info.

    Empty string is returned when *elements* is empty.
    """
    if not elements:
        return ""

    lines: list[str] = []

    # Group by (repo_name, relative_path) to avoid conflicts when multiple
    # repos have the same file names.
    file_groups: dict[tuple[str, str], list[Hit]] = {}
    for hit in elements:
        relative_path = hit.relative_path or hit.file_path

        group_key = (hit.repo_name, relative_path)
        if group_key not in file_groups:
            file_groups[group_key] = []
        file_groups[group_key].append(hit)

    for i, (group_key, elem_list) in enumerate(file_groups.items(), 1):
        repo_name, relative_path = group_key

        display_path = f"{repo_name}/{relative_path}" if repo_name else relative_path
        lines.append(f"\n{i}. {display_path}")

        # Aggregate metadata
        sources: set[str] = set()
        related_to: set[str] = set()
        total_lines = 0

        for hit in elem_list:
            sources.add(_source_label(hit))
            if hit.related_to:
                related_to.add(hit.related_to)

            # Calculate lines
            start = hit.start_line
            end = hit.end_line
            if end > start:
                total_lines += end - start + 1

        if repo_name:
            lines.append(f"   Repo: {repo_name}")
        lines.append(f"   Type: {elem_list[0].element_type or 'unknown'}")
        lines.append(f"   Source: {', '.join(sources)}")
        if total_lines > 0:
            lines.append(f"   Lines: {total_lines}")
        if related_to:
            lines.append(f"   Related to: {', '.join(related_to)}")

        # Show signatures for class/function elements (max 5)
        for hit in elem_list[:5]:
            if hit.signature:
                lines.append(f"   - {hit.signature}")

    return "\n".join(lines)
