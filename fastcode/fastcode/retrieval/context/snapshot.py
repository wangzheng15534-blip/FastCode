"""Pure snapshot logic — domain-specific parts."""

from __future__ import annotations

from collections.abc import Sequence

from fastcode.retrieval.contracts import Hit, SourceCitation


def extract_sources_from_elements(
    elements: Sequence[Hit],
) -> tuple[SourceCitation, ...]:
    """Extract source information from retrieved elements."""
    sources: list[SourceCitation] = []
    for hit in elements:
        sources.append(
            SourceCitation(
                repository=hit.repo_name,
                file=hit.relative_path or hit.file_path,
                name=hit.element_name,
                element_type=hit.element_type,
                lines=f"{hit.start_line}-{hit.end_line}",
                score=hit.total_score,
            )
        )
    return tuple(sources)
