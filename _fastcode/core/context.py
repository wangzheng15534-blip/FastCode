# fastcode/core/context.py
"""Pure context preparation and response parsing — extracted from answer_generator.py."""

from __future__ import annotations

import re
from typing import Any

_MAX_CODE_LENGTH = 100000


def _format_element_metadata(metadata: dict[str, Any]) -> str:
    """Format element metadata into a display string."""
    meta_parts: list[str] = []
    if "complexity" in metadata:
        meta_parts.append(f"Complexity: {metadata['complexity']}")
    if "num_methods" in metadata:
        meta_parts.append(f"Methods: {metadata['num_methods']}")
    return ", ".join(meta_parts)


def prepare_context(
    elements: list[dict[str, Any]],
    *,
    include_file_paths: bool = True,
    include_line_numbers: bool = True,
) -> str:
    """Prepare context string from retrieved elements.

    Args:
        elements: List of retrieved element dicts with 'element' and 'total_score' keys.
        include_file_paths: Whether to include file paths in the output.
        include_line_numbers: Whether to include line numbers in the output.

    Returns:
        Formatted context string, or empty string if no elements.
    """
    if not elements:
        return ""

    context_parts: list[str] = []
    for i, elem_data in enumerate(elements, 1):
        elem = elem_data.get("element", {})

        parts = [f"## Relevant Code Snippet {i}"]
        repo_name = elem.get("repo_name")
        if repo_name:
            parts.append(f"**Repository**: `{repo_name}`")

        if include_file_paths:
            rel_path = elem.get("relative_path", "")
            if rel_path:
                display = f"{repo_name}/{rel_path}" if repo_name else rel_path
                parts.append(f"**File**: `{display}`")

        elem_type = elem.get("type", "")
        elem_name = elem.get("name", "")
        parts.append(f"**Type**: {elem_type}")
        parts.append(f"**Name**: `{elem_name}`")

        if include_line_numbers:
            start_line = elem.get("start_line", 0)
            end_line = elem.get("end_line", 0)
            if start_line > 0:
                parts.append(f"**Lines**: {start_line}-{end_line}")

        code = elem.get("code", "")
        if code:
            language = elem.get("language", "")
            if len(code) > _MAX_CODE_LENGTH:
                code = code[:_MAX_CODE_LENGTH] + "\n... (truncated)"
            parts.append(f"**Code**:\n```{language}\n{code}\n```")

        metadata = elem.get("metadata", {})
        if metadata:
            meta_str = _format_element_metadata(metadata)
            if meta_str:
                parts.append(f"**Metadata**: {meta_str}")

        context_parts.append("\n".join(parts))

    return "\n\n---\n\n".join(context_parts)


def parse_response_with_summary(raw_response: str) -> tuple[str, str | None]:
    """Parse LLM response to extract answer and optional <SUMMARY> block.

    Args:
        raw_response: Raw LLM response text.

    Returns:
        Tuple of (answer_text, summary_text_or_None).
    """
    summary_patterns = [
        r"<\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*:?\s*>(.*?)<\s*/\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*>",
        r"\*\*\s*<\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*>\s*\*\*(.*?)\*\*\s*<\s*/\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*>\s*\*\*",
        r"\*\*\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*\*\*\s*:?\s*\n(.*?)(?=\n\n(?:\*\*|##|$)|\Z)",
        r"[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*:?\s*\n(.*?)(?=\n\n(?:\*\*|##|$)|\Z)",
    ]

    for pattern in summary_patterns:
        match = re.search(pattern, raw_response, re.DOTALL)
        if match:
            summary = match.group(1).strip()
            answer = re.sub(pattern, "", raw_response, flags=re.DOTALL).strip()
            return answer, summary

    return raw_response, None
