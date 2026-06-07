"""Contract tests for ir.merge pure scoring functions — meaning_core, ZERO test doubles.

These functions compute the alignment scores used during AST/SCIP merge.
They must produce correct values given any valid inputs. Tests exercise
boundary conditions and decision-table cases directly.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.ir.merge import (
    CANDIDATE_MATCH_THRESHOLD,
    PRIMARY_MATCH_THRESHOLD,
    _candidate_score,
    _kind_compatible,
    _name_score,
    _normalize_name,
    _signature_param_count,
    _signature_score,
    _span_overlap_score,
)
from fastcode.ir.types import IRCodeUnit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(
    *,
    unit_id: str = "u1",
    kind: str = "function",
    path: str = "a.py",
    language: str = "python",
    display_name: str = "fn",
    qualified_name: str | None = None,
    signature: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    source_set: set[str] | None = None,
) -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=unit_id,
        kind=kind,
        path=path,
        language=language,
        display_name=display_name,
        qualified_name=qualified_name,
        signature=signature,
        start_line=start_line,
        end_line=end_line,
        source_set=source_set or {"fc_structure"},
    )


# ---------------------------------------------------------------------------
# _kind_compatible
# ---------------------------------------------------------------------------


class TestKindCompatible:
    """Decision table for kind compatibility."""

    @pytest.mark.parametrize(
        ("ast_kind", "scip_kind", "expected"),
        [
            ("function", "function", True),
            ("method", "method", True),
            ("class", "class", True),
            ("function", "method", True),  # both CALLABLE_KINDS
            ("method", "function", True),  # both CALLABLE_KINDS
            ("class", "interface", True),  # both CONTAINER_KINDS
            ("interface", "class", True),
            ("enum", "class", True),  # enum is CONTAINER_KINDS
            ("class", "enum", True),
            ("function", "class", False),  # callable vs container
            ("method", "class", False),
            ("function", "variable", False),  # unknown vs callable
            ("class", "function", False),
        ],
    )
    def test_kind_compatibility_decision_table(
        self, ast_kind: str, scip_kind: str, expected: bool
    ) -> None:
        assert _kind_compatible(ast_kind, scip_kind) is expected


# ---------------------------------------------------------------------------
# _normalize_name
# ---------------------------------------------------------------------------


class TestNormalizeName:
    def test_extracts_last_segment(self) -> None:
        assert _normalize_name("pkg.module.Foo") == "foo"

    def test_strips_and_lowercases(self) -> None:
        assert _normalize_name("  Bar  ") == "bar"

    def test_none_returns_empty(self) -> None:
        assert _normalize_name(None) == ""

    def test_empty_returns_empty(self) -> None:
        assert _normalize_name("") == ""

    @given(name=st.text(min_size=1, max_size=50))
    @settings(max_examples=30)
    @pytest.mark.property
    def test_always_lowercase(self, name: str) -> None:
        result = _normalize_name(name)
        assert result == result.lower()


# ---------------------------------------------------------------------------
# _span_overlap_score
# ---------------------------------------------------------------------------


class TestSpanOverlapScore:
    def test_perfect_overlap(self) -> None:
        a = _unit(start_line=10, end_line=20)
        b = _unit(start_line=10, end_line=20)
        assert _span_overlap_score(a, b) == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        a = _unit(start_line=10, end_line=20)
        b = _unit(start_line=15, end_line=25)
        # intersection: [15,20] = 6, union: [10,25] = 16
        assert _span_overlap_score(a, b) == pytest.approx(6.0 / 16.0)

    def test_no_overlap(self) -> None:
        a = _unit(start_line=10, end_line=15)
        b = _unit(start_line=20, end_line=25)
        assert _span_overlap_score(a, b) == pytest.approx(0.0)

    def test_contained(self) -> None:
        a = _unit(start_line=10, end_line=30)
        b = _unit(start_line=15, end_line=20)
        # intersection: [15,20] = 6, union: [10,30] = 21
        assert _span_overlap_score(a, b) == pytest.approx(6.0 / 21.0)

    @pytest.mark.edge
    def test_missing_start_returns_zero(self) -> None:
        a = _unit(start_line=None, end_line=20)
        b = _unit(start_line=10, end_line=20)
        assert _span_overlap_score(a, b) == pytest.approx(0.0)

    @pytest.mark.edge
    def test_missing_end_falls_back_to_start(self) -> None:
        a = _unit(start_line=10, end_line=None)
        b = _unit(start_line=10, end_line=10)
        # a becomes [10, 10], b is [10, 10]
        assert _span_overlap_score(a, b) == pytest.approx(1.0)

    @pytest.mark.edge
    def test_single_line_overlap(self) -> None:
        a = _unit(start_line=5, end_line=5)
        b = _unit(start_line=5, end_line=5)
        assert _span_overlap_score(a, b) == pytest.approx(1.0)

    @given(
        a_start=st.integers(min_value=1, max_value=100),
        a_end=st.integers(min_value=1, max_value=100),
        b_start=st.integers(min_value=1, max_value=100),
        b_end=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    @pytest.mark.property
    def test_score_always_in_zero_to_one(
        self, a_start: int, a_end: int, b_start: int, b_end: int
    ) -> None:
        a = _unit(start_line=a_start, end_line=max(a_start, a_end))
        b = _unit(start_line=b_start, end_line=max(b_start, b_end))
        score = _span_overlap_score(a, b)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _name_score
# ---------------------------------------------------------------------------


class TestNameScore:
    def test_exact_match(self) -> None:
        a = _unit(display_name="myFunc")
        b = _unit(display_name="myFunc")
        assert _name_score(a, b) == pytest.approx(1.0)

    def test_case_insensitive(self) -> None:
        a = _unit(display_name="MyFunc")
        b = _unit(display_name="myfunc")
        assert _name_score(a, b) == pytest.approx(1.0)

    def test_substring_match(self) -> None:
        a = _unit(display_name="run")
        b = _unit(display_name="_run_impl")
        assert _name_score(a, b) == pytest.approx(0.75)

    def test_no_match(self) -> None:
        a = _unit(display_name="alpha")
        b = _unit(display_name="beta")
        assert _name_score(a, b) == pytest.approx(0.0)

    @pytest.mark.edge
    def test_empty_name_returns_zero(self) -> None:
        a = _unit(display_name="")
        b = _unit(display_name="fn")
        assert _name_score(a, b) == pytest.approx(0.0)

    def test_qualified_name_exact_match(self) -> None:
        a = _unit(qualified_name="pkg.Foo")
        b = _unit(qualified_name="pkg.Foo")
        # Falls through to qualified_name check after display_name check
        assert _name_score(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _signature_param_count
# ---------------------------------------------------------------------------


class TestSignatureParamCount:
    def test_no_params(self) -> None:
        assert _signature_param_count("fn()") == 0

    def test_one_param(self) -> None:
        assert _signature_param_count("fn(x: int)") == 1

    def test_multiple_params(self) -> None:
        assert _signature_param_count("fn(x: int, y: str, z: bool)") == 3

    @pytest.mark.edge
    def test_no_parens_returns_none(self) -> None:
        assert _signature_param_count("no parens") is None

    @pytest.mark.edge
    def test_none_returns_none(self) -> None:
        assert _signature_param_count(None) is None

    @pytest.mark.edge
    def test_empty_string_returns_none(self) -> None:
        assert _signature_param_count("") is None


# ---------------------------------------------------------------------------
# _signature_score
# ---------------------------------------------------------------------------


class TestSignatureScore:
    def test_exact_match(self) -> None:
        a = _unit(signature="fn(x: int)")
        b = _unit(signature="fn(x: int)")
        assert _signature_score(a, b) == pytest.approx(1.0)

    def test_param_count_match(self) -> None:
        a = _unit(signature="fn(a, b)")
        b = _unit(signature="fn(x, y)")
        assert _signature_score(a, b) == pytest.approx(1.0)

    def test_param_count_mismatch(self) -> None:
        a = _unit(signature="fn(a)")
        b = _unit(signature="fn(x, y)")
        assert _signature_score(a, b) == pytest.approx(0.0)

    @pytest.mark.edge
    def test_both_none_returns_zero(self) -> None:
        a = _unit(signature=None)
        b = _unit(signature=None)
        assert _signature_score(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _candidate_score
# ---------------------------------------------------------------------------


class TestCandidateScore:
    def test_different_paths_returns_zero(self) -> None:
        a = _unit(path="a.py", display_name="fn", start_line=1, end_line=5)
        b = _unit(path="b.py", display_name="fn", start_line=1, end_line=5)
        assert _candidate_score(a, b, {}, {}, {}) == pytest.approx(0.0)

    def test_incompatible_kinds_returns_zero(self) -> None:
        a = _unit(kind="function", display_name="fn", start_line=1, end_line=5)
        b = _unit(kind="class", display_name="fn", start_line=1, end_line=5)
        assert _candidate_score(a, b, {}, {}, {}) == pytest.approx(0.0)

    def test_no_overlap_and_no_name_returns_zero(self) -> None:
        a = _unit(display_name="alpha", start_line=1, end_line=5)
        b = _unit(display_name="beta", start_line=100, end_line=105)
        assert _candidate_score(a, b, {}, {}, {}) == pytest.approx(0.0)

    def test_perfect_match_score(self) -> None:
        a = _unit(
            display_name="fn",
            start_line=10,
            end_line=20,
            kind="function",
            path="a.py",
        )
        b = _unit(
            display_name="fn",
            start_line=10,
            end_line=20,
            kind="function",
            path="a.py",
        )
        score = _candidate_score(a, b, {}, {}, {})
        assert score >= PRIMARY_MATCH_THRESHOLD

    def test_candidate_threshold_match(self) -> None:
        """A name match with no span overlap still produces a candidate score."""
        a = _unit(
            display_name="fn",
            start_line=1,
            end_line=5,
            kind="function",
            path="a.py",
        )
        b = _unit(
            display_name="fn",
            start_line=100,
            end_line=105,
            kind="function",
            path="a.py",
        )
        score = _candidate_score(a, b, {}, {}, {})
        assert score >= CANDIDATE_MATCH_THRESHOLD

    @given(
        start_a=st.integers(min_value=1, max_value=50),
        start_b=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=30)
    @pytest.mark.property
    def test_score_always_in_zero_to_one(self, start_a: int, start_b: int) -> None:
        a = _unit(
            display_name="fn",
            start_line=start_a,
            end_line=start_a + 10,
            kind="function",
            path="a.py",
        )
        b = _unit(
            display_name="fn",
            start_line=start_b,
            end_line=start_b + 10,
            kind="function",
            path="a.py",
        )
        score = _candidate_score(a, b, {}, {}, {})
        assert 0.0 <= score <= 1.0
