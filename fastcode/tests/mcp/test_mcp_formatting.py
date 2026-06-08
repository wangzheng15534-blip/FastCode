"""Tests for MCP text formatting functions."""

from __future__ import annotations

from fastcode.mcp.formatting import (
    format_call_chain,
    format_code_qa_response,
    format_delete_repo_metadata,
    format_explore_code_response,
    format_file_summary,
    format_indexed_repos,
    format_repo_overview,
    format_session_history,
    format_session_list,
    format_symbol_search_results,
)


class TestFormatCodeQaResponse:
    def test_formats_answer_with_sources(self):
        sources = [
            {
                "file": "src/main.py",
                "name": "run",
                "start_line": 10,
                "end_line": 20,
                "repo": "myrepo",
            },
        ]
        result = format_code_qa_response("It runs the app.", sources, "sess-1")
        assert "It runs the app." in result
        assert "myrepo/src/main.py:L10-L20" in result
        assert "[session_id: sess-1]" in result

    def test_formats_without_sources(self):
        result = format_code_qa_response("No results.", [], "sess-2")
        assert "No results." in result
        assert "Sources:" not in result

    def test_formats_lines_from_lines_field(self):
        sources = [
            {"file": "a.py", "name": "foo", "lines": "30-45", "repo": "r"},
        ]
        result = format_code_qa_response("ans", sources, "s1")
        assert "L30-L45" in result


class TestFormatSessionList:
    def test_empty(self):
        assert format_session_list([]) == "No sessions found."

    def test_formats_sessions(self):
        sessions = [
            {
                "session_id": "abc",
                "title": "Test",
                "total_turns": 3,
                "multi_turn": True,
            },
        ]
        result = format_session_list(sessions)
        assert 'abc: "Test"' in result
        assert "3 turns" in result


class TestFormatSessionHistory:
    def test_not_found(self):
        assert "No history found" in format_session_history("bad-id", [])

    def test_formats_history(self):
        history = [
            {"turn_number": 1, "query": "What?", "answer": "X" * 600},
        ]
        result = format_session_history("s1", history)
        assert "Turn 1" in result
        assert "What?" in result
        assert "…" in result  # truncated


class TestFormatSymbolSearchResults:
    def test_no_results(self):
        assert "No symbols" in format_symbol_search_results("foo", [])

    def test_formats_results(self):
        results = [
            {
                "name": "foo",
                "type": "function",
                "repo_name": "r",
                "relative_path": "a.py",
                "start_line": 1,
                "end_line": 5,
                "signature": "def foo()",
            },
        ]
        result = format_symbol_search_results("foo", results)
        assert "[function] foo" in result
        assert "def foo()" in result


class TestFormatFileSummary:
    def test_formats_file(self):
        result = format_file_summary(
            "src/main.py",
            file_meta={
                "language": "python",
                "metadata": {"total_lines": 100, "code_lines": 80, "num_imports": 3},
            },
            classes=[
                {
                    "signature": "class App",
                    "start_line": 10,
                    "end_line": 50,
                    "metadata": {"methods": ["run"]},
                }
            ],
            top_level_functions=[
                {"signature": "def helper()", "start_line": 52, "end_line": 60}
            ],
            repo_name="myrepo",
        )
        assert "myrepo/src/main.py" in result
        assert "Language: python" in result
        assert "class App" in result
        assert "def helper()" in result


class TestFormatCallChain:
    def test_formats_chain(self):
        result = format_call_chain(
            target_name="main",
            target_type="function",
            target_path="main.py",
            target_start_line=10,
            callers=[{"name": "entry", "loc": "entry.py:L5", "indent": 2}],
            callees=[{"name": "(none)", "indent": 2}],
        )
        assert "main" in result
        assert "Callers" in result
        assert "entry" in result
        assert "Callees" in result


class TestFormatExploreCodeResponse:
    def test_formats_grouped_refs_lines_code_and_next_actions(self):
        result = format_explore_code_response(
            {
                "query": "Where is auth?",
                "snapshot_id": "snap:repo:1",
                "freshness": {"state": "fresh"},
                "completeness": {
                    "state": "complete",
                    "returned_snippets": 1,
                    "omitted_snippets": 0,
                },
                "groups": [
                    {
                        "ref_id": "g1",
                        "repo": "repo",
                        "file": "src/auth.py",
                        "snippets": [
                            {
                                "ref_id": "e1",
                                "type": "function",
                                "name": "authenticate",
                                "lines": "7-9",
                                "score": 0.9,
                                "language": "python",
                                "signature": "def authenticate()",
                                "evidence_refs": [
                                    {
                                        "ref_id": "e1",
                                        "kind": "retrieval_hit",
                                    },
                                    {
                                        "ref_id": "e1:trace:1",
                                        "kind": "projection_trace",
                                    },
                                ],
                                "graph_relationships": [
                                    {
                                        "source_element_id": "u:caller",
                                        "target_element_id": "u:auth",
                                        "relationship": "calls",
                                    }
                                ],
                                "code": "   7 | def authenticate():",
                            }
                        ],
                    }
                ],
                "next_actions": [
                    {
                        "tool": "get_file_summary",
                        "arguments": {"file_path": "src/auth.py"},
                    }
                ],
            }
        )

        assert "Explore: Where is auth?" in result
        assert "Snapshot: snap:repo:1" in result
        assert "## g1 repo/src/auth.py" in result
        assert "e1 [function] authenticate L7-9" in result
        assert "def authenticate()" in result
        assert "evidence: e1, e1:trace:1" in result
        assert "graph: u:caller -> u:auth (calls)" in result
        assert "get_file_summary" in result

    def test_empty(self):
        assert "No source snippets" in format_explore_code_response({"groups": []})


class TestFormatRepoOverview:
    def test_formats_overview(self):
        result = format_repo_overview(
            "myrepo",
            metadata={"summary": "A test repo", "structure_text": "src/\n  main.py"},
            languages={"Python": 10, "TypeScript": 5},
        )
        assert "myrepo" in result
        assert "A test repo" in result
        assert "Python: 10 files" in result


class TestFormatIndexedRepos:
    def test_empty(self):
        assert "No indexed" in format_indexed_repos([])

    def test_formats_repos(self):
        repos = [{"name": "r1", "element_count": 50, "size_mb": 1.2}]
        result = format_indexed_repos(repos)
        assert "r1" in result
        assert "50 elements" in result


class TestFormatDeleteRepoMetadata:
    def test_no_files(self):
        result = format_delete_repo_metadata("r1", [], 0)
        assert "No metadata" in result

    def test_with_files(self):
        result = format_delete_repo_metadata("r1", ["a.faiss", "b.pkl"], 2.5)
        assert "2.5 MB" in result
        assert "a.faiss" in result
