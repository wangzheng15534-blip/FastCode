"""Pure fusion functions extracted from HybridRetriever.

All functions are stateless and depend only on stdlib (math, re, collections)
plus fastcode.retrieval.scoring and fastcode.retrieval.contracts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

from fastcode.ir.element import CodeElement, CodeElementMeta, serialize_code_element
from fastcode.retrieval.contracts import (
    DocProjectionPriors,
    FusionConfig,
    Hit,
    ProjectionEvidence,
    RetrievalSource,
    TraceLink,
)

from .scoring import (
    normalized_query_entropy,
    sigmoid,
    tokenize_signal,
    trace_confidence_weight,
    weighted_keyword_affinity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_trace_links(hit: Hit) -> tuple[TraceLink, ...]:
    """Extract grounded trace links from a retrieval hit."""
    raw_links = hit.metadata.get("trace_links") or hit.metadata.get("mentions") or []
    links: list[TraceLink] = []
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
        raw_confidence = link_dict.get("confidence")
        links.append(
            TraceLink(
                unit_id=str(unit_id),
                weight=max(0.0, min(1.0, weight)),
                evidence_type=str(link_dict.get("evidence_type") or "trace_link"),
                chunk_id=str(link_dict.get("chunk_id") or hit.element_id),
                symbol_name=(
                    str(link_dict["symbol_name"])
                    if link_dict.get("symbol_name") is not None
                    else None
                ),
                confidence=(
                    str(raw_confidence) if raw_confidence is not None else None
                ),
            )
        )
    return tuple(links)


def _find_code_element_for_ir_unit(
    unit_id: str,
    bm25_elements: list[Any],
    repo_filter: list[str] | None = None,
) -> CodeElementMeta | None:
    """Find a code element dict matching *unit_id* in *bm25_elements*.

    *bm25_elements* is a list of ``CodeElement`` objects (or anything with
    ``type``, ``repo_name``, and ``metadata`` attributes).
    """
    for elem in bm25_elements:
        if elem.type == "design_document":
            continue
        if repo_filter and elem.repo_name not in repo_filter:
            continue
        meta = cast(dict[str, Any], elem.metadata or {})
        if meta.get("ir_symbol_id") == unit_id or meta.get("ir_node_id") == unit_id:
            return serialize_code_element(cast(CodeElement, elem))
    return None

def _normalized_hit_totals(results: Sequence[Hit]) -> dict[str, float]:
    if not results:
        return {}
    max_score = max(float(hit.total_score) for hit in results)
    if max_score <= 0:
        return {
            hit.element_id: 1.0 / float(rank)
            for rank, hit in enumerate(results, start=1)
            if hit.element_id
        }
    return {
        hit.element_id: max(0.0, min(1.0, float(hit.total_score) / max_score))
        for hit in results
        if hit.element_id
    }


def _fusion_payload(hit: Hit) -> dict[str, Any]:
    raw = hit.extra.get("fusion")
    return dict(cast(dict[str, Any], raw)) if isinstance(raw, dict) else {}


def _hit_with_fusion(hit: Hit, fusion: dict[str, Any]) -> Hit:
    return replace(hit, extra={**hit.extra, "fusion": dict(fusion)})


def _new_fused_hit(hit: Hit) -> Hit:
    """Create a new fused result entry from a channel hit."""
    empty_scores = hit.with_scores(
        score=0.0,
        semantic_score=0.0,
        keyword_score=0.0,
        pseudocode_score=0.0,
        graph_score=0.0,
        total_score=0.0,
    )
    return _hit_with_fusion(empty_scores, {})


def _ensure_fused_hit(
    fused: dict[str, Hit],
    elem_id: str,
    hit: Hit,
) -> Hit:
    """Ensure an entry exists in *fused* for *elem_id*, creating if needed."""
    entry = fused.get(elem_id)
    if entry is None:
        entry = _new_fused_hit(hit)
        fused[elem_id] = entry
    return entry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_adaptive_fusion_params(
    *,
    query: str,
    query_info: dict[str, Any],
    code_results: Sequence[Hit],
    doc_results: Sequence[Hit],
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

    code_top = float(code_results[0].total_score) if code_results else 0.0
    doc_top = float(doc_results[0].total_score) if doc_results else 0.0

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
    code_results: Sequence[Hit],
    doc_results: Sequence[Hit],
    config: FusionConfig,
    debug: dict[str, Any] | None = None,
) -> list[Hit]:
    """RRF-fuse code and doc results with adaptive alpha/k.

    Returns:
        Sorted list of fused hits.
    """
    alpha, k_code, k_doc = compute_adaptive_fusion_params(
        query=query,
        query_info=query_info,
        code_results=code_results,
        doc_results=doc_results,
        config=config,
    )
    fused: dict[str, Hit] = {}
    for rank, hit in enumerate(code_results, start=1):
        elem_id = hit.element_id
        if not elem_id:
            continue
        rrf = 1.0 / (float(k_code) + float(rank))
        entry = _ensure_fused_hit(fused, elem_id, hit)
        fusion = _fusion_payload(entry)
        fusion["code_rrf"] = rrf
        fusion["alpha"] = alpha
        fusion["k_code"] = k_code
        fusion["k_doc"] = k_doc
        fused[elem_id] = _hit_with_fusion(
            entry.with_scores(
                semantic_score=max(entry.semantic_score, hit.semantic_score),
                keyword_score=max(entry.keyword_score, hit.keyword_score),
                pseudocode_score=max(entry.pseudocode_score, hit.pseudocode_score),
                graph_score=max(entry.graph_score, hit.graph_score),
                total_score=entry.total_score + (alpha * rrf),
            ),
            fusion,
        )

    for rank, hit in enumerate(doc_results, start=1):
        elem_id = hit.element_id
        if not elem_id:
            continue
        rrf = 1.0 / (float(k_doc) + float(rank))
        entry = _ensure_fused_hit(fused, elem_id, hit)
        fusion = _fusion_payload(entry)
        fusion["doc_rrf"] = rrf
        fusion["alpha"] = alpha
        fusion["k_code"] = k_code
        fusion["k_doc"] = k_doc
        fused[elem_id] = _hit_with_fusion(
            entry.with_scores(
                semantic_score=max(entry.semantic_score, hit.semantic_score),
                keyword_score=max(entry.keyword_score, hit.keyword_score),
                total_score=entry.total_score + ((1.0 - alpha) * rrf),
            ),
            fusion,
        )

    out = [hit.with_scores(score=hit.total_score) for hit in fused.values()]
    out.sort(key=lambda hit: hit.total_score, reverse=True)

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
    doc_results: Sequence[Hit],
    config: FusionConfig,
    doc_projection_beta_max: float = 0.35,
) -> DocProjectionPriors:
    """Compute doc-to-code projection priors from doc results.

    Returns:
        Frozen projection prior record.
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
    norm_scores = _normalized_hit_totals(doc_results)

    priors: dict[str, float] = {}
    evidence: dict[str, list[ProjectionEvidence]] = {}
    for hit in doc_results:
        if not hit.element_id:
            continue
        doc_score = norm_scores.get(hit.element_id, 0.0)
        if doc_score <= 0.0:
            continue
        for link in extract_trace_links(hit):
            unit_id = link.unit_id
            contribution = max(0.0, min(1.0, doc_score * float(link.weight)))
            prior = priors.get(unit_id, 0.0)
            priors[unit_id] = 1.0 - ((1.0 - prior) * (1.0 - contribution))
            evidence.setdefault(unit_id, []).append(
                ProjectionEvidence(
                    link=link,
                    doc_id=hit.element_id,
                    doc_score=doc_score,
                    contribution=contribution,
                )
            )

    return DocProjectionPriors(
        p_doc=p_doc,
        beta=beta,
        priors=priors,
        evidence={unit_id: tuple(items) for unit_id, items in evidence.items()},
    )


def apply_doc_projection_to_code(
    *,
    query: str,
    query_info: dict[str, Any],
    code_results: Sequence[Hit],
    doc_results: Sequence[Hit],
    config: FusionConfig,
    doc_projection_beta_max: float = 0.35,
    bm25_elements: list[Any] | None = None,
    repo_filter: list[str] | None = None,
    debug: dict[str, Any] | None = None,
) -> list[Hit]:
    """Apply doc projection priors to code results.

    Returns:
        Sorted list of projected code hits.
    """
    if not doc_results:
        return list(code_results)

    projection = project_doc_priors(
        query=query,
        query_info=query_info,
        doc_results=doc_results,
        config=config,
        doc_projection_beta_max=doc_projection_beta_max,
    )
    beta = projection.beta
    priors = projection.priors
    evidence = projection.evidence
    code_norm = _normalized_hit_totals(code_results)

    seeded: dict[str, Hit] = {}
    for hit in code_results:
        elem_id = hit.element_id
        unit_id = hit.metadata.get("ir_symbol_id") or hit.metadata.get("ir_node_id")
        retrieval_score = code_norm.get(elem_id, 0.0)
        projected_prior = float(priors.get(str(unit_id), 0.0)) if unit_id else 0.0
        seed_score = ((1.0 - beta) * retrieval_score) + (beta * projected_prior)
        seeded[elem_id] = hit.with_scores(
            score=seed_score,
            retrieval_score=hit.total_score,
            projection_score=projected_prior,
            seed_score=seed_score,
            traceability=evidence.get(str(unit_id), ()) if unit_id else (),
            total_score=seed_score,
        )

    for unit_id, prior in priors.items():
        if prior <= 0.0:
            continue
        if any(
            hit.metadata.get("ir_symbol_id") == unit_id
            or hit.metadata.get("ir_node_id") == unit_id
            for hit in seeded.values()
        ):
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
        seed_score = beta * float(prior)
        hit = Hit.from_element(
            element,
            score=seed_score,
            source=RetrievalSource.RETRIEVAL,
            total_score=seed_score,
        )
        seeded[hit.element_id] = hit.with_scores(
            retrieval_score=0.0,
            projection_score=prior,
            seed_score=seed_score,
            traceability=evidence.get(unit_id, ()),
            projected_only=True,
        )

    out = list(seeded.values())
    out.sort(key=lambda hit: hit.total_score, reverse=True)

    if debug is not None:
        debug["doc_projection"] = {
            "beta": beta,
            "projected_units": len(priors),
        }

    return out
