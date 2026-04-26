# tests/test_core_context.py
"""Tests for pure context preparation and response parsing."""

from typing import Any

from fastcode.core.context import (
    parse_response_with_summary,
    prepare_context,
)


def _mk_element(
    elem_id: str = "func1",
    *,
    repo_name: str = "myrepo",
    relative_path: str = "src/main.py",
    elem_type: str = "function",
    code: str = "def func1(): pass",
    start_line: int = 10,
    end_line: int = 12,
    language: str = "python",
    total_score: float = 0.85,
) -> dict[str, Any]:
    return {
        "element": {
            "id": elem_id,
            "type": elem_type,
            "name": elem_id,
            "repo_name": repo_name,
            "relative_path": relative_path,
            "code": code,
            "start_line": start_line,
            "end_line": end_line,
            "language": language,
        },
        "total_score": total_score,
    }


class TestPrepareContext:
    def test_single_element(self):
        elements = [_mk_element()]
        context = prepare_context(elements)
        assert "func1" in context
        assert "def func1(): pass" in context

    def test_includes_repo_name(self):
        elements = [_mk_element(repo_name="myrepo")]
        context = prepare_context(elements, include_file_paths=True)
        assert "myrepo" in context

    def test_includes_file_path(self):
        elements = [_mk_element(relative_path="src/core/scoring.py")]
        context = prepare_context(elements, include_file_paths=True)
        assert "scoring.py" in context

    def test_includes_line_numbers(self):
        elements = [_mk_element(start_line=10, end_line=20)]
        context = prepare_context(elements, include_line_numbers=True)
        assert "10-20" in context

    def test_truncates_long_code(self):
        long_code = "x" * 200000
        elements = [_mk_element(code=long_code)]
        context = prepare_context(elements)
        assert "truncated" in context

    def test_multiple_elements(self):
        elements = [_mk_element("func1"), _mk_element("func2")]
        context = prepare_context(elements)
        assert "func1" in context
        assert "func2" in context

    def test_empty(self):
        context = prepare_context([])
        assert context == ""

    def test_element_with_metadata(self):
        elements = [
            {
                "element": {
                    "id": "a",
                    "type": "function",
                    "name": "a",
                    "code": "def a(): pass",
                    "language": "python",
                    "metadata": {"complexity": 5, "num_methods": 3},
                },
                "total_score": 0.9,
            }
        ]
        context = prepare_context(elements)
        assert "Complexity: 5" in context
        assert "Methods: 3" in context


class TestParseResponseWithSummary:
    def test_extracts_summary_tags(self):
        response = "Here is the answer.\n<SUMMARY>\nFiles Read:\n- foo.py\n</SUMMARY>"
        answer, summary = parse_response_with_summary(response)
        assert "Files Read" in summary
        assert "<SUMMARY>" not in answer

    def test_no_summary(self):
        response = "Just a plain answer with no summary."
        answer, summary = parse_response_with_summary(response)
        assert answer == response
        assert summary is None

    def test_case_insensitive_tags(self):
        response = "Answer\n<summary>\nContent\n</summary>"
        _answer, summary = parse_response_with_summary(response)
        assert summary is not None

    def test_bold_summary(self):
        response = "Answer\n**<SUMMARY>**\nContent\n**</SUMMARY>**"
        _answer, summary = parse_response_with_summary(response)
        assert summary is not None
