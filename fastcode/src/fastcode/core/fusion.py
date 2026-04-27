"""Pure fusion functions extracted from HybridRetriever.

All functions are stateless and depend only on stdlib (math, re, collections)
plus fastcode.core.scoring and fastcode.core.types.
"""

from __future__ import annotations

import math
from typing import Any, cast

from fastcode.schemas.core_types import FusionConfig

from .scoring import (
    clone_result_row,
    normalized_query_entropy,
    normalized_totals,
    sigmoid,
    tokenize_signal,
    trace_confidence_weight,
    weighted_keyword_affinity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_trace_links(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract grounded trace links from a retrieval result row."""
    elem = cast(dict[str, Any], row.get("element") or {})
    meta = cast(dict[str, Any], elem.get("metadata") or {})
    raw_links: list[Any] = meta.get("trace_links") or meta.get("mentions") or []
    links: list[dict[str, Any]] = []
    for link in raw_links:
        if not isinstance(link, dict):
            continue
        link_dict = cast(dict[str, Any], link)
        unit_id = (
            link_dict.get("unit_id")
            or link_dict.get("symbol_id")
            or link_dict.get("ir_symbol_id")
        )
        if not unit_id:
            continue
        weight = float(
            link_dict.get("weight")
            or trace_confidence_weight(link_dict.get("confidence"))
        )
        links.append(
            {
                "unit_id": str(unit_id),
                "weight": max(0.0, min(1.0, weight)),
                "evidence_type": link_dict.get("evidence_type") or "trace_link",
                "chunk_id": link_dict.get("chunk_id") or elem.get("id"),
                "symbol_name": link_dict.get("symbol_name"),
                "confidence": link_dict.get("confidence"),
            }
        )
    return links


def _find_code_element_for_ir_unit(
    unit_id: str,
    bm25_elements: list[Any],
    repo_filter: list[str] | None = None,
) -> dict[str, Any] | None:
    """Find a code element dict matching *unit_id* in *bm25_elements*.

    *bm25_elements* is a list of ``CodeElement`` objects (or anything with
    ``type``, ``repo_name``, ``metadata``, and ``to_dict()`` attributes).
    """
    for elem in bm25_elements:
        if elem.type == "design_document":
            continue
        if repo_filter and elem.repo_name not in repo_filter:
            continue
        meta = cast(dict[str, Any], elem.metadata or {})
        if meta.get("ir_symbol_id") == unit_id or meta.get("ir_node_id") == unit_id:
            return elem.to_dict()
    return None


def _new_fused_entry(element: dict[str, Any]) -> dict[str, Any]:
    """Create a new fused result entry."""
    return {
        "element": element,
        "semantic_score": 0.0,
        "keyword_score": 0.0,
        "pseudocode_score": 0.0,
        "graph_score": 0.0,
        "total_score": 0.0,
        "fusion": {},
    }


def _ensure_fused_entry(
    fused: dict[str, dict[str, Any]],
    elem_id: str,
    element: dict[str, Any],
) -> dict[str, Any]:
    """Ensure an entry exists in *fused* for *elem_id*, creating if needed."""
    entry = fused.get(elem_id)
    if entry is None:
        entry = _new_fused_entry(element)
        fused[elem_id] = entry
    return entry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_adaptive_fusion_params(
    *,
    query: str,
    query_info: dict[str, Any],
    code_results: list[dict[str, Any]],
    doc_results: list[dict[str, Any]],
    config: FusionConfig,
) -> tuple[float, float, float]:
    """Compute adaptive fusion parameters (alpha, k_code, k_doc).

    Returns:
        Tuple of (alpha, k_code, k_doc).
    """
    alpha_base = float(config.alpha_base)
    alpha_min = float(config.alpha_min)
    alpha_max = float(config.alpha_max)
    k_base = float(config.rrf_k_base)
    k_min = float(config.rrf_k_min)
    k_max = float(config.rrf_k_max)

    q = query or ""
    qi = query_info or {}
    intent = str(qi.get("intent") or "")
    keywords = qi.get("keywords")
    if isinstance(keywords, list):
        keyword_text = " ".join(str(k) for k in cast(list[Any], keywords))
    else:
        keyword_text = ""

    signal_text = " ".join([q, intent, keyword_text])
    tokens = tokenize_signal(signal_text)
    query_entropy = normalized_query_entropy(tokens)

    doc_term_weights = {
        "design": 1.0,
        "architecture": 1.0,
        "adr": 1.2,
        "rfc": 1.2,
        "decision": 1.0,
        "tradeoff": 1.1,
        "rationale": 1.1,
        "approach": 0.8,
        "spec": 0.9,
        "why": 0.7,
        "intent": 0.7,
    }
    code_term_weights = {
        "function": 1.0,
        "class": 1.0,
        "method": 1.0,
        "line": 0.7,
        "call": 0.9,
        "bug": 1.0,
        "fix": 1.0,
        "trace": 0.9,
        "implementation": 1.0,
        "stack": 0.7,
        "runtime": 0.8,
    }
    doc_affinity = weighted_keyword_affinity(tokens, doc_term_weights)
    code_affinity = weighted_keyword_affinity(tokens, code_term_weights)

    code_top = float(code_results[0].get("total_score", 0.0)) if code_results else 0.0
    doc_top = float(doc_results[0].get("total_score", 0.0)) if doc_results else 0.0

    alpha = alpha_base
    # Continuous domain affinity (replaces binary doc_hit/code_hit).
    alpha -= 0.30 * doc_affinity
    alpha += 0.18 * code_affinity

    # Continuous confidence skew from retrieval channel strengths.
    strength_delta = math.tanh((code_top - doc_top) * 2.2)
    alpha += 0.12 * strength_delta

    # High-entropy queries are more ambiguous, so pull alpha toward balanced blending.
    entropy_pull = 0.22 * query_entropy
    alpha = (1.0 - entropy_pull) * alpha + entropy_pull * 0.5
    alpha = min(alpha_max, max(alpha_min, alpha))

    code_conf = min(1.0, max(0.0, code_top))
    doc_conf = min(1.0, max(0.0, doc_top))

    # Smooth sigmoid k to avoid cliff effects from piecewise/linear shifts.
    code_k_z = (
        ((0.55 - code_conf) * 4.2)
        + ((query_entropy - 0.5) * 1.4)
        + ((doc_affinity - code_affinity) * 1.0)
    )
    doc_k_z = (
        ((0.55 - doc_conf) * 4.2)
        + ((query_entropy - 0.5) * 1.4)
        + ((code_affinity - doc_affinity) * 1.0)
    )
    k_code_sig = sigmoid(code_k_z)
    k_doc_sig = sigmoid(doc_k_z)
    k_code = k_min + (k_max - k_min) * k_code_sig
    k_doc = k_min + (k_max - k_min) * k_doc_sig

    # Gentle pull toward configured base to keep behavior stable across repos.
    k_code = 0.8 * k_code + 0.2 * k_base
    k_doc = 0.8 * k_doc + 0.2 * k_base

    k_code = min(k_max, max(k_min, k_code))
    k_doc = min(k_max, max(k_min, k_doc))

    return alpha, k_code, k_doc


def adaptive_fuse_channels(
    *,
    query: str,
    query_info: dict[str, Any],
    code_results: list[dict[str, Any]],
    doc_results: list[dict[str, Any]],
    config: FusionConfig,
    debug: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """RRF-fuse code and doc results with adaptive alpha/k.

    Returns:
        Sorted list of fused result dicts.
    """
    alpha, k_code, k_doc = compute_adaptive_fusion_params(
        query=query,
        query_info=query_info,
        code_results=code_results,
        doc_results=doc_results,
        config=config,
    )
    fused: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(code_results, start=1):
        elem = row.get("element", {})
        elem_id = elem.get("id")
        if not elem_id:
            continue
        rrf = 1.0 / (float(k_code) + float(rank))
        entry = _ensure_fused_entry(fused, elem_id, elem)
        entry["semantic_score"] = max(
            entry.get("semantic_score", 0.0), row.get("semantic_score", 0.0)
        )
        entry["keyword_score"] = max(
            entry.get("keyword_score", 0.0), row.get("keyword_score", 0.0)
        )
        entry["pseudocode_score"] = max(
            entry.get("pseudocode_score", 0.0), row.get("pseudocode_score", 0.0)
        )
        entry["graph_score"] = max(
            entry.get("graph_score", 0.0), row.get("graph_score", 0.0)
        )
        entry["total_score"] += alpha * rrf
        entry["fusion"]["code_rrf"] = rrf
        entry["fusion"]["alpha"] = alpha
        entry["fusion"]["k_code"] = k_code
        entry["fusion"]["k_doc"] = k_doc

    for rank, row in enumerate(doc_results, start=1):
        elem = row.get("element", {})
        elem_id = elem.get("id")
        if not elem_id:
            continue
        rrf = 1.0 / (float(k_doc) + float(rank))
        entry = _ensure_fused_entry(fused, elem_id, elem)
        entry["semantic_score"] = max(
            entry.get("semantic_score", 0.0), row.get("semantic_score", 0.0)
        )
        entry["keyword_score"] = max(
            entry.get("keyword_score", 0.0), row.get("keyword_score", 0.0)
        )
        entry["total_score"] += (1.0 - alpha) * rrf
        entry["fusion"]["doc_rrf"] = rrf
        entry["fusion"]["alpha"] = alpha
        entry["fusion"]["k_code"] = k_code
        entry["fusion"]["k_doc"] = k_doc

    out = list(fused.values())
    out.sort(key=lambda x: x.get("total_score", 0.0), reverse=True)

    if debug is not None:
        debug.update(
            {
                "alpha": alpha,
                "k_code": k_code,
                "k_doc": k_doc,
                "code_candidates": len(code_results),
                "doc_candidates": len(doc_results),
            }
        )

    return out


def project_doc_priors(
    *,
    query: str,
    query_info: dict[str, Any],
    doc_results: list[dict[str, Any]],
    config: FusionConfig,
    doc_projection_beta_max: float = 0.35,
) -> dict[str, Any]:
    """Compute doc-to-code projection priors from doc results.

    Returns:
        Dict with ``p_doc``, ``beta``, ``priors``, and ``evidence``.
    """
    alpha, _, _ = compute_adaptive_fusion_params(
        query=query,
        query_info=query_info,
        code_results=[],
        doc_results=doc_results,
        config=config,
    )
    p_doc = 1.0 - alpha
    beta = max(0.0, min(1.0, doc_projection_beta_max * p_doc))
    norm_scores = normalized_totals(doc_results)

    priors: dict[str, float] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    for row in doc_results:
        elem = cast(dict[str, Any], row.get("element") or {})
        elem_id = str(elem.get("id") or "")
        if not elem_id:
            continue
        doc_score = norm_scores.get(elem_id, 0.0)
        if doc_score <= 0.0:
            continue
        for link in extract_trace_links(row):
            unit_id = link["unit_id"]
            contribution = max(0.0, min(1.0, doc_score * float(link["weight"])))
            prior = priors.get(unit_id, 0.0)
            priors[unit_id] = 1.0 - ((1.0 - prior) * (1.0 - contribution))
            evidence.setdefault(unit_id, []).append(
                {
                    **link,
                    "doc_id": elem_id,
                    "doc_score": doc_score,
                    "contribution": contribution,
                }
            )

    return {
        "p_doc": p_doc,
        "beta": beta,
        "priors": priors,
        "evidence": evidence,
    }


def apply_doc_projection_to_code(
    *,
    query: str,
    query_info: dict[str, Any],
    code_results: list[dict[str, Any]],
    doc_results: list[dict[str, Any]],
    config: FusionConfig,
    doc_projection_beta_max: float = 0.35,
    bm25_elements: list[Any] | None = None,
    repo_filter: list[str] | None = None,
    debug: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Apply doc projection priors to code results.

    Returns:
        Sorted list of projected code results.
    """
    if not doc_results:
        return [clone_result_row(row) for row in code_results]

    projection = project_doc_priors(
        query=query,
        query_info=query_info,
        doc_results=doc_results,
        config=config,
        doc_projection_beta_max=doc_projection_beta_max,
    )
    beta = float(projection["beta"])
    priors: dict[str, float] = projection["priors"]
    evidence: dict[str, list[dict[str, Any]]] = projection["evidence"]
    code_norm = normalized_totals(code_results)

    seeded: dict[str, dict[str, Any]] = {}
    for row in code_results:
        materialized = clone_result_row(row)
        elem = cast(dict[str, Any], materialized.get("element") or {})
        elem_id = str(elem.get("id") or "")
        meta = cast(dict[str, Any], elem.get("metadata") or {})
        unit_id = meta.get("ir_symbol_id") or meta.get("ir_node_id")
        retrieval_score = code_norm.get(elem_id, 0.0)
        projected_prior = float(priors.get(str(unit_id), 0.0)) if unit_id else 0.0
        seed_score = ((1.0 - beta) * retrieval_score) + (beta * projected_prior)
        materialized["retrieval_score"] = materialized.get("total_score", 0.0)
        materialized["projection_score"] = projected_prior
        materialized["seed_score"] = seed_score
        materialized["traceability"] = evidence.get(str(unit_id), []) if unit_id else []
        materialized["total_score"] = seed_score
        seeded[elem_id] = materialized

    for unit_id, prior in priors.items():
        if prior <= 0.0:
            continue

        def _row_has_unit_id(row: dict[str, Any], uid: str) -> bool:
            elem = cast(dict[str, Any], row.get("element") or {})
            meta = cast(dict[str, Any], elem.get("metadata") or {})
            return meta.get("ir_symbol_id") == uid

        if any(_row_has_unit_id(row, unit_id) for row in seeded.values()):
            continue
        if bm25_elements is None:
            continue
        element = _find_code_element_for_ir_unit(
            unit_id,
            bm25_elements=bm25_elements,
            repo_filter=repo_filter,
        )
        if element is None:
            continue
        elem_id = str(element.get("id") or "")
        seeded[elem_id] = {
            "element": element,
            "semantic_score": 0.0,
            "keyword_score": 0.0,
            "pseudocode_score": 0.0,
            "graph_score": 0.0,
            "retrieval_score": 0.0,
            "projection_score": prior,
            "seed_score": beta * float(prior),
            "total_score": beta * float(prior),
            "traceability": evidence.get(unit_id, []),
            "projected_only": True,
        }

    out = list(seeded.values())
    out.sort(key=lambda row: float(row.get("total_score", 0.0)), reverse=True)

    if debug is not None:
        debug["doc_projection"] = {
            "beta": beta,
            "projected_units": len(priors),
        }

    return out
