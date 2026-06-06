"""Small stdlib-only text and mapping helpers."""

from __future__ import annotations

from typing import Any, cast


def extract_code_snippet(
    content: str, start_line: int, end_line: int, context_lines: int = 3
) -> dict[str, Any]:
    """Extract code snippet lines with surrounding context."""
    lines = content.split("\n")
    total_lines = len(lines)
    actual_start = max(0, start_line - context_lines)
    actual_end = min(total_lines, end_line + context_lines)
    snippet_lines = lines[actual_start:actual_end]
    return {
        "code": "\n".join(snippet_lines),
        "start_line": actual_start + 1,
        "end_line": actual_end,
        "highlighted_start": start_line + 1,
        "highlighted_end": end_line,
    }


def format_code_block(
    code: str, language: str = "", file_path: str = "", start_line: int | None = None
) -> str:
    """Format code as a Markdown fenced block."""
    header = f"```{language}"
    if file_path:
        header += f" - {file_path}"
    if start_line:
        header += f" (Line {start_line})"
    return f"{header}\n{code}\n```"


def calculate_code_complexity(code: str) -> int:
    """Calculate a simple control-flow complexity score."""
    keywords = [
        "if",
        "elif",
        "else",
        "for",
        "while",
        "try",
        "except",
        "catch",
        "case",
        "switch",
        "&&",
        "||",
        "?",
    ]
    complexity = 1
    for keyword in keywords:
        complexity += code.count(keyword)
    return complexity


def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """Merge dictionaries from left to right."""
    result: dict[str, Any] = {}
    for value in dicts:
        result.update(value)
    return result


def safe_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get a nested dictionary value."""
    current: Any = d
    for key in keys:
        if isinstance(current, dict):
            current = cast(dict[str, Any], current).get(key)
            if current is None:
                return default
        else:
            return default
    return current


def clean_docstring(docstring: str | None) -> str:
    """Clean and normalize common indentation in a docstring."""
    if not docstring:
        return ""
    lines = docstring.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    min_indent = float("inf")
    for line in lines:
        if line.strip():
            min_indent = min(min_indent, len(line) - len(line.lstrip()))

    if min_indent < float("inf"):
        lines = [
            line[int(min_indent) :] if len(line) > min_indent else line
            for line in lines
        ]
    return "\n".join(lines).strip()
