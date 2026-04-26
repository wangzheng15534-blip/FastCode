"""Pure iteration control functions extracted from IterativeAgent.

Zero I/O, zero logging. The orchestrator wraps these with logging.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .types import IterationConfig

# ---------------------------------------------------------------------------
# AdaptiveParams -- result of parameter initialization
# ---------------------------------------------------------------------------


class AdaptiveParams(NamedTuple):
    """Adaptive iteration parameters computed from query complexity and repo."""

    max_iterations: int
    confidence_threshold: int
    adaptive_line_budget: int


# ---------------------------------------------------------------------------
# Confidence / lines helpers
# ---------------------------------------------------------------------------


def calculate_recent_confidence_gain(history: tuple | list) -> float:
    """Return confidence delta between last two history entries."""
    if len(history) < 2:
        return 0.0
    return float(history[-1]["confidence"] - history[-2]["confidence"])


def calculate_recent_lines_added(history: tuple | list) -> int:
    """Return total-lines delta between last two history entries."""
    if len(history) < 2:
        return 0
    return history[-1]["total_lines"] - history[-2]["total_lines"]


def calculate_total_lines(elements: list[dict]) -> int:
    """Sum (end - start + 1) for every element with line-range info."""
    total = 0
    for elem_data in elements:
        elem = elem_data.get("element", {})
        start = elem.get("start_line", 0)
        end = elem.get("end_line", 0)
        if end > start:
            total += end - start + 1
    return total


# ---------------------------------------------------------------------------
# ROI / repo factor
# ---------------------------------------------------------------------------


def get_min_roi_threshold(query_complexity: int, current_confidence: int) -> float:
    """Minimum acceptable ROI (confidence gain per 1000 lines).

    Complex queries accept lower ROI; higher current confidence demands more.
    """
    base_roi = 2.0

    if query_complexity >= 80:
        complexity_factor = 0.5
    elif query_complexity >= 60:
        complexity_factor = 0.7
    else:
        complexity_factor = 1.0

    if current_confidence >= 85:
        confidence_factor = 1.5
    elif current_confidence >= 70:
        confidence_factor = 1.0
    else:
        confidence_factor = 0.8

    return base_roi * complexity_factor * confidence_factor


def calculate_repo_factor(repo_stats: dict | None) -> float:
    """Repository complexity factor in [0.5, 2.0]. Returns 1.0 when stats are absent."""
    if not repo_stats:
        return 1.0

    total_files = repo_stats.get("total_files", 100)
    avg_file_lines = repo_stats.get("avg_file_lines", 200)
    max_depth = repo_stats.get("max_depth", 5)

    file_factor = np.log10(total_files + 1) / np.log10(1000)
    file_factor = np.clip(file_factor, 0.3, 1.5)

    complexity_factor = avg_file_lines / 200
    complexity_factor = np.clip(complexity_factor, 0.5, 2.0)

    depth_factor = max_depth / 5
    depth_factor = np.clip(depth_factor, 0.7, 1.3)

    final_factor = (file_factor + complexity_factor + depth_factor) / 3
    return float(np.clip(final_factor, 0.5, 2.0))


# ---------------------------------------------------------------------------
# Adaptive parameter initialization
# ---------------------------------------------------------------------------


def initialize_adaptive_parameters(
    query_complexity: int,
    repo_factor: float,
    config: IterationConfig,
) -> AdaptiveParams:
    """Compute adaptive iteration parameters from query + repo complexity."""
    complexity_score = (query_complexity / 100 + repo_factor) / 2
    max_iterations = max(
        2,
        min(6, int(config.base_max_iterations * (0.7 + complexity_score * 0.6))),
    )

    if query_complexity >= 80:
        confidence_threshold = max(90, config.base_confidence_threshold - 5)
    elif query_complexity >= 60:
        confidence_threshold = max(92, config.base_confidence_threshold - 3)
    else:
        confidence_threshold = config.base_confidence_threshold

    if query_complexity <= 30:
        line_budget = int(config.max_total_lines * 0.6)
    elif query_complexity <= 60:
        line_budget = int(config.max_total_lines * 0.8)
    else:
        line_budget = int(config.max_total_lines * 1.0 * repo_factor)

    return AdaptiveParams(
        max_iterations=max_iterations,
        confidence_threshold=confidence_threshold,
        adaptive_line_budget=line_budget,
    )


# ---------------------------------------------------------------------------
# Stopping logic
# ---------------------------------------------------------------------------


def should_continue_iteration(
    *,
    confidence: int,
    current_round: int,
    max_iterations: int,
    total_lines: int,
    line_budget: int,
    confidence_threshold: int,
    history: tuple | list,
    min_confidence_gain: float,
    query_complexity: int = 50,
) -> bool:
    """Six-check stopping logic. Returns True when iteration should continue."""
    # Check 1: Confidence already sufficient
    if confidence >= confidence_threshold:
        return False

    # Check 2: Hard iteration limit
    if current_round >= max_iterations:
        return False

    # Check 3: Line budget
    if total_lines >= line_budget:
        return False

    # Check 4: Adaptive trend & ROI analysis
    if len(history) >= 2:
        current_metrics = history[-1]
        current_gain = current_metrics["confidence_gain"]
        current_roi = current_metrics["roi"]

        # 4a. Stagnation -- consecutive small fluctuations
        if abs(current_gain) < 1.0 and len(history) >= 3:
            prev_gain = history[-2]["confidence_gain"]
            if abs(prev_gain) < 1.0:
                return False

        # Helper: is a round "low performance"?
        def _is_low_performance(gain: float, roi: float) -> bool:
            min_roi = get_min_roi_threshold(query_complexity, confidence)
            if gain < -1.0:
                return True
            return -1.0 <= gain < min_confidence_gain and roi < min_roi

        if _is_low_performance(current_gain, current_roi) and len(history) >= 3:
            prev_metrics = history[-2]
            if _is_low_performance(
                prev_metrics["confidence_gain"],
                prev_metrics["roi"],
            ):
                return False

    # Check 5: Strict stagnation (last 3 rounds)
    if len(history) >= 3:
        last_three_confidences = [h["confidence"] for h in history[-3:]]
        if max(last_three_confidences) - min(last_three_confidences) < 2:
            return False

    # Check 6: Cost-benefit threshold
    confidence_gap = confidence_threshold - confidence
    remaining_line_budget = line_budget - total_lines
    estimated_lines_needed = confidence_gap * 100

    if estimated_lines_needed > remaining_line_budget * 1.5:
        # Relax if we just had a confidence drop (exploration mode)
        if len(history) >= 2 and history[-1]["confidence_gain"] < 0:
            pass  # allow continuation
        else:
            return False

    return True


def determine_stopping_reason(
    *,
    final_confidence: int,
    confidence_threshold: int,
    current_round: int,
    max_iterations: int,
    iteration_history: tuple | list,
    line_budget: int,
    min_confidence_gain: float = 0.5,
) -> str:
    """Return a human-readable reason why iteration stopped."""
    if final_confidence >= confidence_threshold:
        return "confidence_threshold_reached"
    if current_round >= max_iterations:
        return "max_iterations_reached"
    if iteration_history and iteration_history[-1].get("total_lines", 0) >= line_budget:
        return "line_budget_exceeded"
    if len(iteration_history) >= 3:
        recent_gains = [h["confidence_gain"] for h in iteration_history[-2:]]
        if all(g < min_confidence_gain for g in recent_gains):
            return "diminishing_returns"
    return "other"


def rate_efficiency(overall_roi: float, budget_used_pct: float) -> str:
    """Rate the overall efficiency of the iteration process."""
    if overall_roi >= 5.0 and budget_used_pct < 70:
        return "excellent"
    if overall_roi >= 3.0 and budget_used_pct < 85:
        return "good"
    if overall_roi >= 1.5 or budget_used_pct < 90:
        return "acceptable"
    return "inefficient"
