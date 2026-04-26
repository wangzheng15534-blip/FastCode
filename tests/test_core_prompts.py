"""Tests for pure prompt formatting functions."""

from fastcode.core.prompts import (
    format_elements_with_metadata,
    format_tool_call_history,
)


def _mk_elem_data(
    elem_id: str,
    *,
    repo_name: str = "myrepo",
    relative_path: str = "src/main.py",
    elem_type: str = "function",
    start_line: int = 10,
    end_line: int = 20,
    total_score: float = 0.8,
    agent_found: bool = False,
    llm_file_selected: bool = False,
    related_to: str | None = None,
    signature: str | None = None,
) -> dict:
    elem: dict = {
        "id": elem_id,
        "type": elem_type,
        "name": elem_id,
        "repo_name": repo_name,
        "relative_path": relative_path,
        "start_line": start_line,
        "end_line": end_line,
    }
    if signature:
        elem["signature"] = signature
    result: dict = {
        "element": elem,
        "total_score": total_score,
        "agent_found": agent_found,
        "llm_file_selected": llm_file_selected,
    }
    if related_to:
        result["related_to"] = related_to
    return result


class TestFormatElementsWithMetadata:
    def test_single_element(self):
        elements = [_mk_elem_data("func1")]
        text = format_elements_with_metadata(elements)
        assert "myrepo/src/main.py" in text
        assert "function" in text

    def test_groups_by_file(self):
        elements = [
            _mk_elem_data("func1", relative_path="a.py"),
            _mk_elem_data("func2", relative_path="a.py"),
            _mk_elem_data("func3", relative_path="b.py"),
        ]
        text = format_elements_with_metadata(elements)
        assert "a.py" in text
        assert "b.py" in text

    def test_agent_found_source(self):
        elements = [_mk_elem_data("func1", agent_found=True)]
        text = format_elements_with_metadata(elements)
        assert "Tool" in text

    def test_graph_source(self):
        elements = [_mk_elem_data("func1", related_to="other_func")]
        text = format_elements_with_metadata(elements)
        assert "Graph" in text

    def test_retrieval_source(self):
        elements = [_mk_elem_data("func1")]
        text = format_elements_with_metadata(elements)
        assert "Retrieval" in text

    def test_shows_signature(self):
        elements = [_mk_elem_data("func1", signature="def func1(x: int) -> str")]
        text = format_elements_with_metadata(elements)
        assert "def func1(x: int) -> str" in text

    def test_shows_line_count(self):
        elements = [_mk_elem_data("func1", start_line=10, end_line=20)]
        text = format_elements_with_metadata(elements)
        assert "Lines" in text

    def test_empty(self):
        text = format_elements_with_metadata([])
        assert text == ""


class TestFormatToolCallHistory:
    def test_with_history(self):
        history = [
            {
                "round": 1,
                "tool": "search_codebase",
                "parameters": {"search_term": "foo"},
            },
            {"round": 2, "tool": "list_directory", "parameters": {"path": "src"}},
        ]
        text = format_tool_call_history(history, current_round=3)
        assert "search_codebase" in text
        assert "list_directory" in text

    def test_filters_current_round(self):
        history = [
            {
                "round": 1,
                "tool": "search_codebase",
                "parameters": {"search_term": "foo"},
            },
            {"round": 2, "tool": "list_directory", "parameters": {"path": "src"}},
        ]
        text = format_tool_call_history(history, current_round=2)
        assert "search_codebase" in text
        assert "list_directory" not in text

    def test_empty(self):
        text = format_tool_call_history([], current_round=1)
        assert text == "None"

    def test_no_history_attribute(self):
        text = format_tool_call_history(None, current_round=1)
        assert text == "None"
