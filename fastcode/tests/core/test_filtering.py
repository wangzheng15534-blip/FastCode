"""Tests for pure filtering functions extracted from retriever."""

from typing import Any

from fastcode.core.filtering import (
    apply_filters,
    diversify,
    final_repo_filter,
    rerank,
)


def _mk_row(
    elem_id: str,
    elem_type: str = "function",
    total: float = 0.8,
    *,
    language: str = "python",
    file_path: str = "src/main.py",
    relative_path: str = "src/main.py",
    repo_name: str = "myrepo",
    snapshot_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elem: dict[str, Any] = {
        "id": elem_id,
        "type": elem_type,
        "name": elem_id,
        "language": language,
        "file_path": file_path,
        "relative_path": relative_path,
        "repo_name": repo_name,
    }
    if snapshot_id:
        elem["snapshot_id"] = snapshot_id
    if metadata:
        elem["metadata"] = metadata
    return {
        "element": elem,
        "semantic_score": total,
        "keyword_score": total * 0.5,
        "pseudocode_score": 0.0,
        "graph_score": 0.0,
        "total_score": total,
    }


class TestApplyFilters:
    def test_filter_by_language(self):
        rows = [
            _mk_row("a", language="python"),
            _mk_row("b", language="java"),
        ]
        result = apply_filters(rows, {"language": "python"})
        assert len(result) == 1
        assert result[0]["element"]["id"] == "a"

    def test_filter_by_type(self):
        rows = [
            _mk_row("a", elem_type="function"),
            _mk_row("b", elem_type="class"),
        ]
        result = apply_filters(rows, {"type": "class"})
        assert len(result) == 1
        assert result[0]["element"]["id"] == "b"

    def test_filter_by_file_path(self):
        rows = [
            _mk_row("a", relative_path="src/core/scoring.py"),
            _mk_row("b", relative_path="tests/test_main.py"),
        ]
        result = apply_filters(rows, {"file_path": "core"})
        assert len(result) == 1
        assert result[0]["element"]["id"] == "a"

    def test_filter_by_snapshot_id(self):
        rows = [
            _mk_row("a", snapshot_id="snap:v1"),
            _mk_row("b", snapshot_id="snap:v2"),
        ]
        result = apply_filters(rows, {"snapshot_id": "snap:v1"})
        assert len(result) == 1

    def test_filter_by_snapshot_id_in_metadata(self):
        rows = [
            _mk_row("a", metadata={"snapshot_id": "snap:v1"}),
            _mk_row("b", metadata={"snapshot_id": "snap:v2"}),
        ]
        result = apply_filters(rows, {"snapshot_id": "snap:v1"})
        assert len(result) == 1

    def test_no_filters_returns_all(self):
        rows = [_mk_row("a"), _mk_row("b")]
        result = apply_filters(rows, {})
        assert len(result) == 2

    def test_multiple_filters(self):
        rows = [
            _mk_row("a", language="python", elem_type="function"),
            _mk_row("b", language="python", elem_type="class"),
            _mk_row("c", language="java", elem_type="function"),
        ]
        result = apply_filters(rows, {"language": "python", "type": "function"})
        assert len(result) == 1
        assert result[0]["element"]["id"] == "a"


class TestDiversify:
    def test_no_penalty(self):
        rows = [
            _mk_row("a", total=0.9, file_path="f.py"),
            _mk_row("b", total=0.8, file_path="f.py"),
        ]
        result = diversify(rows, diversity_penalty=0.0)
        assert len(result) == 2

    def test_penalty_reduces_duplicate_file_scores(self):
        rows = [
            _mk_row("a", total=0.9, file_path="f.py"),
            _mk_row("b", total=0.8, file_path="f.py"),
        ]
        result = diversify(rows, diversity_penalty=0.5)
        assert len(result) == 2
        # Second result from same file should have lower total_score
        assert result[1]["total_score"] < 0.8

    def test_different_files_not_penalized(self):
        rows = [
            _mk_row("a", total=0.9, file_path="f1.py"),
            _mk_row("b", total=0.8, file_path="f2.py"),
        ]
        result = diversify(rows, diversity_penalty=0.5)
        assert result[0]["total_score"] == 0.9
        assert result[1]["total_score"] == 0.8

    def test_empty_results(self):
        result = diversify([], diversity_penalty=0.5)
        assert result == []

    def test_result_is_sorted_with_penalty(self):
        rows = [
            _mk_row("a", total=0.5, file_path="f1.py"),
            _mk_row("b", total=0.9, file_path="f2.py"),
        ]
        result = diversify(rows, diversity_penalty=0.1)
        assert result[0]["element"]["id"] == "b"

    def test_zero_penalty_preserves_order(self):
        rows = [
            _mk_row("a", total=0.5, file_path="f1.py"),
            _mk_row("b", total=0.9, file_path="f2.py"),
        ]
        result = diversify(rows, diversity_penalty=0.0)
        assert result[0]["element"]["id"] == "a"


class TestFinalRepoFilter:
    def test_filters_by_repo(self):
        rows = [
            _mk_row("a", repo_name="repo1"),
            _mk_row("b", repo_name="repo2"),
        ]
        result = final_repo_filter(rows, ["repo1"])
        assert len(result) == 1
        assert result[0]["element"]["id"] == "a"

    def test_empty_filter_returns_all(self):
        rows = [_mk_row("a"), _mk_row("b")]
        result = final_repo_filter(rows, [])
        assert len(result) == 2

    def test_returns_filtered_count(self):
        rows = [
            _mk_row("a", repo_name="repo1"),
            _mk_row("b", repo_name="repo2"),
            _mk_row("c", repo_name="repo3"),
        ]
        result, count = final_repo_filter(rows, ["repo1"], return_count=True)
        assert count == 2
        assert len(result) == 1


class TestRerank:
    def test_function_gets_boost(self):
        rows = [
            _mk_row("a", elem_type="file", total=0.9),
            _mk_row("b", elem_type="function", total=0.9),
        ]
        result = rerank(rows)
        # function (1.2x) > file (0.9x) at same base score
        assert result[0]["element"]["id"] == "b"

    def test_unknown_type_gets_no_change(self):
        rows = [_mk_row("a", elem_type="module", total=0.5)]
        result = rerank(rows)
        assert abs(result[0]["total_score"] - 0.5) < 1e-9

    def test_results_sorted_after_rerank(self):
        rows = [
            _mk_row("a", elem_type="documentation", total=0.9),
            _mk_row("b", elem_type="function", total=0.8),
        ]
        result = rerank(rows)
        # function (0.8 * 1.2 = 0.96) > documentation (0.9 * 0.8 = 0.72)
        assert result[0]["element"]["id"] == "b"

    def test_empty(self):
        result = rerank([])
        assert result == []
