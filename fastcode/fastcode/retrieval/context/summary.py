"""Pure summary and formatting functions."""

from __future__ import annotations

from fastcode.retrieval.contracts import AnswerDisplayResult, Hit, SourceCitation

_DOC_PREVIEW_MAX = 150


def generate_fallback_summary(
    query: str,
    answer: str,
    retrieved_elements: tuple[Hit, ...],
) -> str:
    """Generate a fallback summary when LLM doesn't produce one."""
    parts: list[str] = []

    files_read: set[str] = set()
    for hit in retrieved_elements:
        rel_path = hit.relative_path or hit.file_path
        if hit.repo_name and rel_path:
            files_read.add(f"{hit.repo_name}/{rel_path}")

    if files_read:
        parts.append("Files Read:")
        for file_path in sorted(files_read)[:10]:
            parts.append(f"- {file_path}")
    else:
        parts.append("Files Read: None")

    parts.append("\nCode Elements Referenced:")
    elements_added = 0
    for hit in retrieved_elements[:15]:
        rel_path = hit.relative_path or hit.file_path

        if hit.repo_name and rel_path and hit.element_name:
            elem_info = (
                f"- [{hit.repo_name}/{rel_path}] "
                f"{hit.element_type}: {hit.element_name}"
            )
            if hit.signature:
                elem_info += f" ({hit.signature})"
            parts.append(elem_info)
            if hit.docstring:
                doc_preview = (
                    hit.docstring[:_DOC_PREVIEW_MAX].replace("\n", " ").strip()
                )
                if len(hit.docstring) > _DOC_PREVIEW_MAX:
                    doc_preview += "..."
                parts.append(f"  Doc: {doc_preview}")
            elements_added += 1

    if elements_added == 0:
        parts.append("- No specific code elements")

    parts.append(f"\nQuery: {query[:200]}")
    answer_preview = answer.replace("\n", " ").strip()
    parts.append(f"Answer Preview: {answer_preview}")

    return "\n".join(parts)


def extract_sources(elements: tuple[Hit, ...]) -> tuple[SourceCitation, ...]:
    """Extract source information from elements."""
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


def format_answer_with_sources(result: AnswerDisplayResult) -> str:
    """Format answer with sources for display."""
    output: list[str] = []

    output.append("## Answer\n")
    output.append(result.answer)

    if result.sources:
        output.append("\n\n## Sources\n")
        for i, source in enumerate(result.sources, 1):
            repo_info = f"[{source.repository}] " if source.repository else ""
            output.append(
                f"{i}. {repo_info}**{source.name}** ({source.element_type}) "
                f"in `{source.file}` (lines {source.lines}) "
                f"- Relevance: {source.score:.2f}"
            )

    if result.prompt_tokens is not None:
        output.append(
            f"\n\n*Used {result.prompt_tokens} prompt tokens, "
            f"{result.context_elements} code snippets*"
        )

    return "\n".join(output)
