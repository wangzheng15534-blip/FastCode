"""Pure scoring functions extracted from HybridRetriever.

All functions are stateless and depend only on stdlib (math, re, collections).
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


def clone_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """Deep-clone a retrieval result row.

    Dictionaries and lists are shallow-copied; scalars are shared.
    """
    return {
        k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
        for k, v in row.items()
    }


def normalized_totals(results: list[dict[str, Any]]) -> dict[str, float]:
    """Return ``{element_id: normalized_score}`` for *results*.

    If the maximum ``total_score`` is positive, scores are divided by that
    maximum and clamped to [0, 1].  Otherwise (all zero or missing), each
    element receives ``1/rank`` as its score.
    """
    if not results:
        return {}
    max_score = max(float(row.get("total_score", 0.0) or 0.0) for row in results)
    if max_score <= 0:
        return {
            str((row.get("element") or {}).get("id")): 1.0 / float(rank)
            for rank, row in enumerate(results, start=1)
            if (row.get("element") or {}).get("id")
        }
    return {
        str((row.get("element") or {}).get("id")): max(
            0.0, min(1.0, float(row.get("total_score", 0.0) or 0.0) / max_score)
        )
        for row in results
        if (row.get("element") or {}).get("id")
    }


def trace_confidence_weight(confidence: str | None) -> float:
    """Map a trace confidence label to a numeric weight."""
    label = str(confidence or "").lower()
    return {
        "precise": 1.0,
        "anchored": 1.0,
        "resolved": 0.8,
        "derived": 0.7,
        "heuristic": 0.6,
        "candidate": 0.5,
    }.get(label, 0.6)


def sigmoid(x: float) -> float:
    """Sigmoid with input clamped to [-30, 30] for numerical stability."""
    x = max(-30.0, min(30.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def tokenize_signal(text: str) -> list[str]:
    """Lowercase alphanumeric token extraction."""
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def normalized_query_entropy(tokens: list[str]) -> float:
    """Shannon entropy of *tokens*, normalized to [0, 1]."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = float(len(tokens))
    if total <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(max(p, 1e-12))
    max_entropy = math.log2(max(2, len(counts)))
    if max_entropy <= 0:
        return 0.0
    return max(0.0, min(1.0, entropy / max_entropy))


def weighted_keyword_affinity(
    tokens: list[str],
    weights: dict[str, float],
) -> float:
    """Fraction of *weights* matched by *tokens*, clamped to [0, 1]."""
    if not tokens or not weights:
        return 0.0
    token_set = set(tokens)
    total = float(sum(max(0.0, w) for w in weights.values()))
    if total <= 0:
        return 0.0
    matched = 0.0
    for term, weight in weights.items():
        if term in token_set:
            matched += max(0.0, float(weight))
    return max(0.0, min(1.0, matched / total))
