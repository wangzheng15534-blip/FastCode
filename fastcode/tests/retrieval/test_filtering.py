"""Tests for pure filtering functions extracted from retriever."""

from typing import Any

import pytest

from fastcode.retrieval.contracts import ElementFilter, Hit
from fastcode.retrieval.filtering import (
    apply_filters,
    diversify,
    final_repo_filter,
    rerank,
)


def _mk_hit(
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
) -> Hit:
    return Hit(
        element_id=elem_id,
        element_type=elem_type,
        element_name=elem_id,
        score=total,
        semantic_score=total,
        keyword_score=total * 0.5,
        total_score=total,
        language=language,
        file_path=file_path,
        relative_path=relative_path,
        repo_name=repo_name,
        snapshot_id=snapshot_id or "",
        metadata=metadata or {},
    )


class TestApplyFilters:
    def test_filter_by_language(self):
        hits = [
            _mk_hit("a", language="python"),
            _mk_hit("b", language="java"),
        ]
        result = apply_filters(hits, ElementFilter(language="python"))
        assert len(result) == 1
        assert result[0].element_id == "a"

    def test_filter_by_type(self):
        hits = [
            _mk_hit("a", elem_type="function"),
            _mk_hit("b", elem_type="class"),
        ]
        result = apply_filters(hits, ElementFilter(element_type="class"))
        assert len(result) == 1
        assert result[0].element_id == "b"

    def test_filter_by_file_path(self):
        hits = [
            _mk_hit("a", relative_path="src/core/scoring.py"),
            _mk_hit("b", relative_path="tests/test_main.py"),
        ]
        result = apply_filters(hits, ElementFilter(file_path="core"))
        assert len(result) == 1
        assert result[0].element_id == "a"

    def test_filter_by_snapshot_id(self):
        hits = [
            _mk_hit("a", snapshot_id="snap:v1"),
            _mk_hit("b", snapshot_id="snap:v2"),
        ]
        result = apply_filters(hits, ElementFilter(snapshot_id="snap:v1"))
        assert len(result) == 1

    def test_filter_by_snapshot_id_in_metadata(self):
        hits = [
            _mk_hit("a", metadata={"snapshot_id": "snap:v1"}),
            _mk_hit("b", metadata={"snapshot_id": "snap:v2"}),
        ]
        result = apply_filters(hits, ElementFilter(snapshot_id="snap:v1"))
        assert len(result) == 1

    def test_no_filters_returns_all(self):
        hits = [_mk_hit("a"), _mk_hit("b")]
        result = apply_filters(hits, ElementFilter())
        assert len(result) == 2

    def test_multiple_filters(self):
        hits = [
            _mk_hit("a", language="python", elem_type="function"),
            _mk_hit("b", language="python", elem_type="class"),
            _mk_hit("c", language="java", elem_type="function"),
        ]
        result = apply_filters(
            hits, ElementFilter(language="python", element_type="function")
        )
        assert len(result) == 1
        assert result[0].element_id == "a"


class TestDiversify:
    def test_no_penalty(self):
        hits = [
            _mk_hit("a", total=0.9, file_path="f.py"),
            _mk_hit("b", total=0.8, file_path="f.py"),
        ]
        result = diversify(hits, diversity_penalty=0.0)
        assert len(result) == 2

    def test_penalty_reduces_duplicate_file_scores(self):
        hits = [
            _mk_hit("a", total=0.9, file_path="f.py"),
            _mk_hit("b", total=0.8, file_path="f.py"),
        ]
        result = diversify(hits, diversity_penalty=0.5)
        assert len(result) == 2
        # Second result from same file should have lower total_score
        assert result[1].total_score < 0.8

    def test_different_files_not_penalized(self):
        hits = [
            _mk_hit("a", total=0.9, file_path="f1.py"),
            _mk_hit("b", total=0.8, file_path="f2.py"),
        ]
        result = diversify(hits, diversity_penalty=0.5)
        assert result[0].total_score == 0.9
        assert result[1].total_score == 0.8

    def test_empty_results(self):
        result = diversify([], diversity_penalty=0.5)
        assert result == []

    def test_result_is_sorted_with_penalty(self):
        hits = [
            _mk_hit("a", total=0.5, file_path="f1.py"),
            _mk_hit("b", total=0.9, file_path="f2.py"),
        ]
        result = diversify(hits, diversity_penalty=0.1)
        assert result[0].element_id == "b"

    def test_zero_penalty_preserves_order(self):
        hits = [
            _mk_hit("a", total=0.5, file_path="f1.py"),
            _mk_hit("b", total=0.9, file_path="f2.py"),
        ]
        result = diversify(hits, diversity_penalty=0.0)
        assert result[0].element_id == "a"


class TestFinalRepoFilter:
    def test_filters_by_repo(self):
        hits = [
            _mk_hit("a", repo_name="repo1"),
            _mk_hit("b", repo_name="repo2"),
        ]
        result = final_repo_filter(hits, ["repo1"])
        assert len(result) == 1
        assert result[0].element_id == "a"

    def test_empty_filter_returns_all(self):
        hits = [_mk_hit("a"), _mk_hit("b")]
        result = final_repo_filter(hits, [])
        assert len(result) == 2

    def test_returns_filtered_count(self):
        hits = [
            _mk_hit("a", repo_name="repo1"),
            _mk_hit("b", repo_name="repo2"),
            _mk_hit("c", repo_name="repo3"),
        ]
        result, count = final_repo_filter(hits, ["repo1"], return_count=True)
        assert count == 2
        assert len(result) == 1


class TestRerank:
    def test_function_gets_boost(self):
        hits = [
            _mk_hit("a", elem_type="file", total=0.9),
            _mk_hit("b", elem_type="function", total=0.9),
        ]
        result = rerank(hits)
        # function (1.2x) > file (0.9x) at same base score
        assert result[0].element_id == "b"

    def test_unknown_type_gets_no_change(self):
        hits = [_mk_hit("a", elem_type="module", total=0.5)]
        result = rerank(hits)
        assert abs(result[0].total_score - 0.5) < 1e-9

    def test_results_sorted_after_rerank(self):
        hits = [
            _mk_hit("a", elem_type="documentation", total=0.9),
            _mk_hit("b", elem_type="function", total=0.8),
        ]
        result = rerank(hits)
        # function (0.8 * 1.2 = 0.96) > documentation (0.9 * 0.8 = 0.72)
        assert result[0].element_id == "b"

    def test_empty(self):
        result = rerank([])
        assert result == []


class TestDiversifyExactArithmetic:
    """Verify exact penalty arithmetic for diversify."""

    def test_penalty_exact_arithmetic(self) -> None:
        """Verify exact penalty: score * (1 - penalty)."""
        hits = [
            _mk_hit("a", total=0.8, file_path="same.py"),
            _mk_hit("b", total=0.6, file_path="same.py"),
        ]
        result = diversify(hits, diversity_penalty=0.5)
        # First: no penalty → 0.8. Second: 0.6 * (1-0.5) = 0.3
        # After sort: [0.8, 0.3]
        assert result[0].total_score == pytest.approx(0.8)
        assert result[1].total_score == pytest.approx(0.3)
        assert result[1].semantic_score == pytest.approx(0.3)
        assert result[1].keyword_score == pytest.approx(0.15)  # 0.3 * 0.5

    def test_penalty_100_percent_zeros_duplicate(self) -> None:
        """100% penalty zeros out duplicate file scores."""
        hits = [
            _mk_hit("a", total=0.9, file_path="f.py"),
            _mk_hit("b", total=0.7, file_path="f.py"),
        ]
        result = diversify(hits, diversity_penalty=1.0)
        by_id = {hit.element_id: hit for hit in result}
        assert by_id["b"].total_score == pytest.approx(0.0)
        assert by_id["b"].semantic_score == pytest.approx(0.0)

    def test_three_same_file_cumulative_penalty(self) -> None:
        """All duplicates get penalized, not just the second."""
        hits = [
            _mk_hit("a", total=0.9, file_path="f.py"),
            _mk_hit("b", total=0.6, file_path="f.py"),
            _mk_hit("c", total=0.3, file_path="f.py"),
        ]
        result = diversify(hits, diversity_penalty=0.5)
        by_id = {hit.element_id: hit for hit in result}
        # a: no penalty → 0.9
        assert by_id["a"].total_score == pytest.approx(0.9)
        # b: 0.6 * 0.5 = 0.3
        assert by_id["b"].total_score == pytest.approx(0.3)
        # c: 0.3 * 0.5 = 0.15
        assert by_id["c"].total_score == pytest.approx(0.15)
