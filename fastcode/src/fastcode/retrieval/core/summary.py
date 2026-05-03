# fastcode/core/summary.py
"""Pure summary and formatting functions — extracted from answer_generator.py."""

from __future__ import annotations

from typing import Any

_DOC_PREVIEW_MAX = 150


def generate_fallback_summary(
    query: str,
    answer: str,
    retrieved_elements: list[dict[str, Any]],
) -> str:
    """Generate a fallback summary when LLM doesn't produce one."""
    parts: list[str] = []

    files_read: set[str] = set()
    for elem_data in retrieved_elements:
        elem = elem_data.get("element", {})
        repo_name = elem.get("repo_name", "")
        rel_path = elem.get("relative_path", "")
        if repo_name and rel_path:
            files_read.add(f"{repo_name}/{rel_path}")

    if files_read:
        parts.append("Files Read:")
        for file_path in sorted(files_read)[:10]:
            parts.append(f"- {file_path}")
    else:
        parts.append("Files Read: None")

    parts.append("\nCode Elements Referenced:")
    elements_added = 0
    for elem_data in retrieved_elements[:15]:
        elem = elem_data.get("element", {})
        repo_name = elem.get("repo_name", "")
        rel_path = elem.get("relative_path", "")
        elem_type = elem.get("type", "")
        elem_name = elem.get("name", "")

        if repo_name and rel_path and elem_name:
            elem_info = f"- [{repo_name}/{rel_path}] {elem_type}: {elem_name}"
            signature = elem.get("signature", "")
            if signature:
                elem_info += f" ({signature})"
            parts.append(elem_info)
            docstring = elem.get("docstring", "")
            if docstring:
                doc_preview = docstring[:_DOC_PREVIEW_MAX].replace("\n", " ").strip()
                if len(docstring) > _DOC_PREVIEW_MAX:
                    doc_preview += "..."
                parts.append(f"  Doc: {doc_preview}")
            elements_added += 1

    if elements_added == 0:
        parts.append("- No specific code elements")

    parts.append(f"\nQuery: {query[:200]}")
    answer_preview = answer.replace("\n", " ").strip()
    parts.append(f"Answer Preview: {answer_preview}")

    return "\n".join(parts)


def extract_sources(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract source information from elements."""
    sources: list[dict[str, Any]] = []
    for elem_data in elements:
        elem = elem_data.get("element", {})
        sources.append(
            {
                "repository": elem.get("repo_name", ""),
                "file": elem.get("relative_path", ""),
                "name": elem.get("name", ""),
                "type": elem.get("type", ""),
                "lines": f"{elem.get('start_line', 0)}-{elem.get('end_line', 0)}",
                "score": elem_data.get("total_score", 0),
            }
        )
    return sources


def format_answer_with_sources(result: dict[str, Any]) -> str:
    """Format answer with sources for display."""
    output: list[str] = []

    output.append("## Answer\n")
    output.append(result.get("answer", ""))

    sources = result.get("sources", [])
    if sources:
        output.append("\n\n## Sources\n")
        for i, source in enumerate(sources, 1):
            repo_info = f"[{source['repository']}] " if source.get("repository") else ""
            output.append(
                f"{i}. {repo_info}**{source['name']}** ({source['type']}) "
                f"in `{source['file']}` (lines {source['lines']}) "
                f"- Relevance: {source['score']:.2f}"
            )

    if "prompt_tokens" in result:
        output.append(
            f"\n\n*Used {result['prompt_tokens']} prompt tokens, "
            f"{result.get('context_elements', 0)} code snippets*"
        )

    return "\n".join(output)
