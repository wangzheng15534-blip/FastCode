# tests/test_core_summary.py
"""Tests for pure summary and formatting functions."""

from typing import Any

from fastcode.core.summary import (
    extract_sources,
    format_answer_with_sources,
    generate_fallback_summary,
)


def _mk_elem(
    elem_id: str = "func1",
    *,
    repo_name: str = "myrepo",
    relative_path: str = "src/main.py",
    elem_type: str = "function",
    total_score: float = 0.85,
) -> dict[str, Any]:
    return {
        "element": {
            "id": elem_id,
            "type": elem_type,
            "name": elem_id,
            "repo_name": repo_name,
            "relative_path": relative_path,
            "start_line": 10,
            "end_line": 20,
        },
        "total_score": total_score,
    }


class TestGenerateFallbackSummary:
    def test_basic(self):
        elements = [_mk_elem()]
        summary = generate_fallback_summary("How does X work?", "Answer text", elements)
        assert "Files Read:" in summary
        assert "myrepo/src/main.py" in summary

    def test_no_elements(self):
        summary = generate_fallback_summary("query", "answer", [])
        assert "Files Read: None" in summary

    def test_includes_query(self):
        summary = generate_fallback_summary("How does X work?", "answer", [])
        assert "How does X work?" in summary

    def test_includes_answer_preview(self):
        summary = generate_fallback_summary("query", "A detailed answer", [])
        assert "A detailed answer" in summary

    def test_limits_files_list(self):
        elements = [_mk_elem(f"f{i}", relative_path=f"file{i}.py") for i in range(20)]
        summary = generate_fallback_summary("query", "answer", elements)
        # Files Read section limited to 10 entries
        lines = summary.split("\n")
        file_lines = [ln for ln in lines if ln.startswith("- myrepo/file")]
        assert len(file_lines) == 10


class TestExtractSources:
    def test_basic(self):
        elements = [_mk_elem()]
        sources = extract_sources(elements)
        assert len(sources) == 1
        assert sources[0]["repository"] == "myrepo"
        assert sources[0]["file"] == "src/main.py"

    def test_empty(self):
        assert extract_sources([]) == []

    def test_includes_score(self):
        elements = [_mk_elem(total_score=0.75)]
        sources = extract_sources(elements)
        assert sources[0]["score"] == 0.75


class TestFormatAnswerWithSources:
    def test_basic(self):
        result = {
            "answer": "Test answer",
            "sources": [
                {
                    "repository": "myrepo",
                    "file": "src/main.py",
                    "name": "func1",
                    "type": "function",
                    "lines": "10-20",
                    "score": 0.85,
                },
            ],
            "prompt_tokens": 100,
            "context_elements": 5,
        }
        text = format_answer_with_sources(result)
        assert "Test answer" in text
        assert "func1" in text
        assert "src/main.py" in text
        assert "100 prompt tokens" in text

    def test_no_sources(self):
        result = {"answer": "Simple answer", "sources": []}
        text = format_answer_with_sources(result)
        assert "Simple answer" in text
        assert "Sources" not in text
