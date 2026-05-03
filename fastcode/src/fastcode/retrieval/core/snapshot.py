"""Pure snapshot logic — domain-specific parts."""

from __future__ import annotations

from typing import Any


def extract_sources_from_elements(
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract source information from retrieved elements."""
    sources: list[dict[str, Any]] = []
    for elem_data in elements:
        elem = elem_data.get("element", {})
        sources.append(
            {
                "file": elem.get("relative_path", ""),
                "repo": elem.get("repo_name", ""),
                "type": elem.get("type", ""),
                "name": elem.get("name", ""),
                "start_line": elem.get("start_line", 0),
                "end_line": elem.get("end_line", 0),
            }
        )
    return sources
