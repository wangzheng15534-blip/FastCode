"""Tests for pure iteration control functions."""

import math

import pytest

from fastcode.core.iteration import (
    calculate_recent_confidence_gain,
    calculate_recent_lines_added,
    calculate_repo_factor,
    calculate_total_lines,
    determine_stopping_reason,
    get_min_roi_threshold,
    initialize_adaptive_parameters,
    rate_efficiency,
    should_continue_iteration,
)
from fastcode.schemas.core_types import IterationConfig


class TestCalculateRecentConfidenceGain:
    def test_with_history(self):
        history = [
            {"confidence": 50, "total_lines": 500},
            {"confidence": 65, "total_lines": 800},
        ]
        assert calculate_recent_confidence_gain(history) == 15.0

    def test_single_entry(self):
        assert (
            calculate_recent_confidence_gain([{"confidence": 50, "total_lines": 500}])
            == 0.0
        )

    def test_empty(self):
        assert calculate_recent_confidence_gain([]) == 0.0


class TestCalculateRecentLinesAdded:
    def test_with_history(self):
        history = [
            {"confidence": 50, "total_lines": 500},
            {"confidence": 65, "total_lines": 800},
        ]
        assert calculate_recent_lines_added(history) == 300

    def test_single_entry(self):
        assert (
            calculate_recent_lines_added([{"confidence": 50, "total_lines": 500}]) == 0
        )

    def test_empty(self):
        assert calculate_recent_lines_added([]) == 0


class TestGetMinRoiThreshold:
    def test_high_complexity_lower_threshold(self):
        roi = get_min_roi_threshold(query_complexity=90, current_confidence=60)
        assert roi < 2.0

    def test_low_complexity_higher_threshold(self):
        roi = get_min_roi_threshold(query_complexity=20, current_confidence=60)
        assert roi >= 1.5

    def test_high_confidence_demands_more(self):
        roi_low = get_min_roi_threshold(query_complexity=50, current_confidence=60)
        roi_high = get_min_roi_threshold(query_complexity=50, current_confidence=90)
        assert roi_high > roi_low


class TestCalculateRepoFactor:
    def test_no_stats(self):
        assert calculate_repo_factor(None) == 1.0

    def test_empty_stats(self):
        assert calculate_repo_factor({}) == 1.0

    def test_small_repo(self):
        factor = calculate_repo_factor(
            {
                "total_files": 10,
                "total_classes": 5,
                "total_functions": 20,
                "avg_file_lines": 50,
                "max_depth": 2,
            }
        )
        assert 0.5 <= factor <= 2.0

    def test_large_repo(self):
        factor = calculate_repo_factor(
            {
                "total_files": 500,
                "total_classes": 100,
                "total_functions": 300,
                "avg_file_lines": 300,
                "max_depth": 8,
            }
        )
        assert 0.5 <= factor <= 2.0


class TestCalculateTotalLines:
    def test_with_line_ranges(self):
        elements = [
            {"element": {"start_line": 10, "end_line": 20}},
            {"element": {"start_line": 30, "end_line": 40}},
        ]
        assert calculate_total_lines(elements) == 22

    def test_no_line_info(self):
        assert calculate_total_lines([{"element": {}}]) == 0

    def test_empty(self):
        assert calculate_total_lines([]) == 0


class TestShouldContinueIteration:
    def test_stops_when_confidence_exceeds_threshold(self):
        assert not should_continue_iteration(
            confidence=96,
            current_round=2,
            max_iterations=4,
            total_lines=5000,
            line_budget=12000,
            confidence_threshold=95,
            history=(),
            min_confidence_gain=0.5,
        )

    def test_stops_at_max_iterations(self):
        assert not should_continue_iteration(
            confidence=50,
            current_round=5,
            max_iterations=4,
            total_lines=5000,
            line_budget=12000,
            confidence_threshold=95,
            history=(),
            min_confidence_gain=0.5,
        )

    def test_stops_at_line_budget(self):
        assert not should_continue_iteration(
            confidence=50,
            current_round=2,
            max_iterations=4,
            total_lines=13000,
            line_budget=12000,
            confidence_threshold=95,
            history=(),
            min_confidence_gain=0.5,
        )

    def test_continues_when_below_threshold(self):
        assert should_continue_iteration(
            confidence=60,
            current_round=2,
            max_iterations=4,
            total_lines=5000,
            line_budget=12000,
            confidence_threshold=95,
            history=(),
            min_confidence_gain=0.5,
        )

    def test_stops_on_stagnation(self):
        history = (
            {"confidence": 60, "confidence_gain": 0.3, "roi": 1.0, "total_lines": 5000},
            {"confidence": 60, "confidence_gain": 0.2, "roi": 0.5, "total_lines": 5500},
            {"confidence": 60, "confidence_gain": 0.1, "roi": 0.3, "total_lines": 6000},
        )
        assert not should_continue_iteration(
            confidence=60,
            current_round=4,
            max_iterations=6,
            total_lines=6000,
            line_budget=12000,
            confidence_threshold=95,
            history=history,
            min_confidence_gain=0.5,
        )


class TestInitializeAdaptiveParameters:
    def test_simple_query_reduces_iterations(self):
        params = initialize_adaptive_parameters(
            query_complexity=20,
            repo_factor=1.0,
            config=IterationConfig(),
        )
        assert params.max_iterations <= 4

    def test_complex_query_increases_iterations(self):
        params = initialize_adaptive_parameters(
            query_complexity=90,
            repo_factor=1.5,
            config=IterationConfig(),
        )
        assert params.max_iterations >= 3

    def test_threshold_adjusts_for_complex_queries(self):
        params = initialize_adaptive_parameters(
            query_complexity=85,
            repo_factor=1.0,
            config=IterationConfig(base_confidence_threshold=95),
        )
        assert params.confidence_threshold < 95

    def test_simple_query_full_threshold(self):
        params = initialize_adaptive_parameters(
            query_complexity=30,
            repo_factor=1.0,
            config=IterationConfig(base_confidence_threshold=95),
        )
        assert params.confidence_threshold == 95

    def test_line_budget_scales_with_complexity(self):
        simple = initialize_adaptive_parameters(
            query_complexity=20,
            repo_factor=1.0,
            config=IterationConfig(max_total_lines=12000),
        )
        complex_ = initialize_adaptive_parameters(
            query_complexity=90,
            repo_factor=1.0,
            config=IterationConfig(max_total_lines=12000),
        )
        assert complex_.adaptive_line_budget >= simple.adaptive_line_budget


class TestDetermineStoppingReason:
    def test_confidence_reached(self):
        reason = determine_stopping_reason(
            final_confidence=96,
            confidence_threshold=95,
            current_round=3,
            max_iterations=4,
            iteration_history=tuple({"round": i} for i in range(3)),
            line_budget=12000,
        )
        assert "confidence" in reason.lower() or "threshold" in reason.lower()

    def test_max_iterations(self):
        reason = determine_stopping_reason(
            final_confidence=60,
            confidence_threshold=95,
            current_round=4,
            max_iterations=4,
            iteration_history=tuple({"round": i} for i in range(4)),
            line_budget=12000,
        )
        assert "iteration" in reason.lower() or "max" in reason.lower()

    def test_line_budget(self):
        reason = determine_stopping_reason(
            final_confidence=60,
            confidence_threshold=95,
            current_round=2,
            max_iterations=4,
            iteration_history=tuple({"total_lines": 13000} for _ in range(2)),
            line_budget=12000,
        )
        assert "budget" in reason.lower()


class TestRateEfficiency:
    def test_excellent(self):
        assert rate_efficiency(overall_roi=6.0, budget_used_pct=50) == "excellent"

    def test_good(self):
        assert rate_efficiency(overall_roi=3.5, budget_used_pct=70) == "good"

    def test_acceptable(self):
        assert rate_efficiency(overall_roi=2.0, budget_used_pct=80) == "acceptable"

    def test_inefficient(self):
        assert rate_efficiency(overall_roi=0.5, budget_used_pct=95) == "inefficient"


class TestRepoFactorExactValues:
    """Hand-computed exact values for calculate_repo_factor."""

    def test_small_repo_exact(self) -> None:
        """Small repo: file_factor clipped to 0.3 min, complexity to 0.5, depth to 0.7."""
        factor = calculate_repo_factor(
            {"total_files": 10, "avg_file_lines": 50, "max_depth": 2}
        )
        file_f = max(0.3, min(1.5, math.log10(11) / math.log10(1000)))
        complexity_f = max(0.5, min(2.0, 50 / 200))
        depth_f = max(0.7, min(1.3, 2 / 5))
        expected = max(0.5, min(2.0, (file_f + complexity_f + depth_f) / 3))
        assert factor == pytest.approx(expected, abs=0.001)

    def test_large_repo_exact(self) -> None:
        """Large repo: file_factor ≈ 1.0, complexity clipped to 1.5, depth clipped to 1.3."""
        factor = calculate_repo_factor(
            {"total_files": 1000, "avg_file_lines": 300, "max_depth": 8}
        )
        file_f = max(0.3, min(1.5, math.log10(1001) / math.log10(1000)))
        complexity_f = max(0.5, min(2.0, 300 / 200))
        depth_f = max(0.7, min(1.3, 8 / 5))
        expected = max(0.5, min(2.0, (file_f + complexity_f + depth_f) / 3))
        assert factor == pytest.approx(expected, abs=0.001)

    def test_result_always_in_valid_range(self) -> None:
        """Factor must always be in [0.5, 2.0]."""
        for stats in [
            None,
            {},
            {"total_files": 1, "avg_file_lines": 1, "max_depth": 1},
            {"total_files": 100000, "avg_file_lines": 10000, "max_depth": 50},
        ]:
            factor = calculate_repo_factor(stats)
            assert 0.5 <= factor <= 2.0, f"factor={factor} for stats={stats}"


class TestInitializeAdaptiveExact:
    """Hand-computed exact values for initialize_adaptive_parameters."""

    def test_simple_query_max_iterations(self) -> None:
        """Low complexity + repo_factor=1.0: max_iterations should be modest."""
        params = initialize_adaptive_parameters(
            query_complexity=20,
            repo_factor=1.0,
            config=IterationConfig(base_max_iterations=4),
        )
        # complexity_score = (0.2 + 1.0) / 2 = 0.6
        # max_iter = max(2, min(6, int(4 * (0.7 + 0.6 * 0.6)))) = max(2, min(6, 4)) = 4
        assert params.max_iterations == 4

    def test_complex_query_max_iterations(self) -> None:
        """High complexity + high repo_factor: max_iterations should increase."""
        params = initialize_adaptive_parameters(
            query_complexity=90,
            repo_factor=1.5,
            config=IterationConfig(base_max_iterations=4),
        )
        # complexity_score = (0.9 + 1.5) / 2 = 1.2
        # max_iter = max(2, min(6, int(4 * (0.7 + 1.2 * 0.6)))) = max(2, min(6, 5)) = 5
        assert params.max_iterations == 5

    def test_line_budget_exact_simple(self) -> None:
        """Simple query (complexity ≤ 30): budget = 60% of max_total_lines."""
        params = initialize_adaptive_parameters(
            query_complexity=20,
            repo_factor=1.0,
            config=IterationConfig(max_total_lines=12000),
        )
        assert params.adaptive_line_budget == int(12000 * 0.6)

    def test_line_budget_exact_complex(self) -> None:
        """Complex query (complexity > 60): budget = max_total_lines * repo_factor."""
        params = initialize_adaptive_parameters(
            query_complexity=90,
            repo_factor=1.5,
            config=IterationConfig(max_total_lines=12000),
        )
        assert params.adaptive_line_budget == int(12000 * 1.0 * 1.5)
