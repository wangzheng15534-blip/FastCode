"""
Merge structure units and SCIP anchors into a canonical unit-grounded snapshot.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from functools import cache

from .types import (
    IRCodeUnit,
    IRRelation,
    IRSnapshot,
    IRUnitEmbedding,
    IRUnitSupport,
    resolution_rank,
)

PRIMARY_MATCH_THRESHOLD = 0.65
CANDIDATE_MATCH_THRESHOLD = 0.50
MERGE_CANDIDATE_FANOUT_LIMIT = 64
MERGE_SPAN_BUCKET_LINES = 64
MERGE_MAX_SPAN_BUCKETS_PER_UNIT = 128
MERGE_EXACT_COMPONENT_NODE_LIMIT = 32
MERGE_EXACT_COMPONENT_SIDE_LIMIT = 18
REF_ROLES = {
    "reference",
    "definition",
    "implementation",
    "type_definition",
    "import",
    "write_access",
    "forward_definition",
}
CALLABLE_KINDS = {"function", "method"}
CONTAINER_KINDS = {"class", "interface", "enum"}
_CandidatePair = tuple[str, str, float]
_MatchStats = dict[str, int]


def _clone_unit(unit: IRCodeUnit) -> IRCodeUnit:
    return deepcopy(unit)


def _clone_support(support: IRUnitSupport) -> IRUnitSupport:
    return deepcopy(support)


def _clone_relation(relation: IRRelation) -> IRRelation:
    return deepcopy(relation)


def _clone_embedding(embedding: IRUnitEmbedding) -> IRUnitEmbedding:
    return deepcopy(embedding)


def _kind_compatible(ast_kind: str, scip_kind: str) -> bool:
    if ast_kind == scip_kind:
        return True
    if ast_kind in CALLABLE_KINDS and scip_kind in CALLABLE_KINDS:
        return True
    return ast_kind in CONTAINER_KINDS and scip_kind in CONTAINER_KINDS


def _normalize_name(value: str | None) -> str:
    if not value:
        return ""
    return value.split(".")[-1].strip().lower()


def _stable_unit_key(unit: IRCodeUnit) -> str:
    metadata = unit.metadata or {}
    for key in ("stable_unit_id", "semantic_unit_id", "ast_element_id"):
        value = metadata.get(key)
        if value:
            return str(value).strip().lower()
    return ""


def _span_bucket_keys(unit: IRCodeUnit) -> tuple[int, ...]:
    start = int(unit.start_line or 0)
    end = int(unit.end_line or unit.start_line or 0)
    if start <= 0 or end <= 0:
        return ()
    if end < start:
        start, end = end, start
    start_bucket = (start - 1) // MERGE_SPAN_BUCKET_LINES
    end_bucket = (end - 1) // MERGE_SPAN_BUCKET_LINES
    bucket_count = end_bucket - start_bucket + 1
    if bucket_count <= MERGE_MAX_SPAN_BUCKETS_PER_UNIT:
        return tuple(range(start_bucket, end_bucket + 1))
    step = max(1, bucket_count // MERGE_MAX_SPAN_BUCKETS_PER_UNIT)
    buckets = set(range(start_bucket, end_bucket + 1, step))
    buckets.add(end_bucket)
    return tuple(sorted(buckets))


def _candidate_order_key(
    ast_unit: IRCodeUnit, scip_unit: IRCodeUnit
) -> tuple[int, float, int, str]:
    stable_match = bool(_stable_unit_key(ast_unit)) and _stable_unit_key(
        ast_unit
    ) == _stable_unit_key(scip_unit)
    name_match = bool(_normalize_name(ast_unit.display_name or ast_unit.qualified_name))
    name_match = name_match and _normalize_name(
        ast_unit.display_name or ast_unit.qualified_name
    ) == _normalize_name(scip_unit.display_name or scip_unit.qualified_name)
    overlap = _span_overlap_score(ast_unit, scip_unit)
    start_delta = abs(int(ast_unit.start_line or 0) - int(scip_unit.start_line or 0))
    return (
        0 if stable_match else 1 if name_match else 2 if overlap > 0.0 else 3,
        -overlap,
        start_delta,
        ast_unit.unit_id,
    )


def _merge_metric_base() -> _MatchStats:
    return {
        "match_buckets": 0,
        "candidate_pairs_bucketed": 0,
        "candidate_pairs_scored": 0,
        "candidate_pairs_kept": 0,
        "candidate_pairs_dropped_by_fanout": 0,
        "candidate_pairs_dropped_by_ast_fanout": 0,
        "fanout_capped_units": 0,
        "ast_fanout_capped_units": 0,
        "span_widened_units": 0,
        "selected_pairs": 0,
        "primary_pairs": 0,
        "candidate_anchor_pairs": 0,
        "native_exact_components": 0,
        "native_greedy_components": 0,
    }


def _add_merge_metrics(target: _MatchStats, source: _MatchStats) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _span_overlap_score(ast_unit: IRCodeUnit, scip_unit: IRCodeUnit) -> float:
    a0 = int(ast_unit.start_line or 0)
    a1 = int(ast_unit.end_line or ast_unit.start_line or 0)
    b0 = int(scip_unit.start_line or 0)
    b1 = int(scip_unit.end_line or scip_unit.start_line or 0)
    if a0 <= 0 or a1 <= 0 or b0 <= 0 or b1 <= 0:
        return 0.0
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    if inter <= 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0) + 1
    return inter / max(1, union)


def _name_score(ast_unit: IRCodeUnit, scip_unit: IRCodeUnit) -> float:
    ast_name = _normalize_name(ast_unit.display_name or ast_unit.qualified_name)
    scip_name = _normalize_name(scip_unit.display_name or scip_unit.qualified_name)
    if not ast_name or not scip_name:
        return 0.0
    if ast_name == scip_name:
        return 1.0
    if (
        ast_unit.qualified_name
        and scip_unit.qualified_name
        and ast_unit.qualified_name.lower() == scip_unit.qualified_name.lower()
    ):
        return 1.0
    if ast_name in scip_name or scip_name in ast_name:
        return 0.75
    return 0.0


def _signature_param_count(signature: str | None) -> int | None:
    if not signature or "(" not in signature or ")" not in signature:
        return None
    inner = signature.split("(", 1)[1].rsplit(")", 1)[0].strip()
    if not inner:
        return 0
    return len([part for part in inner.split(",") if part.strip()])


def _signature_score(ast_unit: IRCodeUnit, scip_unit: IRCodeUnit) -> float:
    if (
        ast_unit.signature
        and scip_unit.signature
        and ast_unit.signature.strip() == scip_unit.signature.strip()
    ):
        return 1.0
    ast_params = _signature_param_count(ast_unit.signature)
    scip_params = _signature_param_count(scip_unit.signature)
    if ast_params is not None and scip_params is not None:
        return 1.0 if ast_params == scip_params else 0.0
    return 0.0


def _scip_parent_name(
    scip_unit: IRCodeUnit, scip_supports_by_unit: dict[str, list[IRUnitSupport]]
) -> str:
    for support in scip_supports_by_unit.get(scip_unit.unit_id, []):
        if support.enclosing_external_id:
            return _normalize_name(support.enclosing_external_id)
    qualified = scip_unit.qualified_name or ""
    if "." not in qualified:
        return ""
    return _normalize_name(qualified.rsplit(".", 1)[0])


def _ast_parent_name(ast_unit: IRCodeUnit, unit_by_id: dict[str, IRCodeUnit]) -> str:
    parent_id = ast_unit.parent_unit_id
    if not parent_id or parent_id not in unit_by_id:
        return ""
    parent = unit_by_id[parent_id]
    if parent.kind == "file":
        return ""
    return _normalize_name(parent.display_name or parent.qualified_name)


def _parent_context_score(
    ast_unit: IRCodeUnit,
    scip_unit: IRCodeUnit,
    ast_units_by_id: dict[str, IRCodeUnit],
    scip_supports_by_unit: dict[str, list[IRUnitSupport]],
) -> float:
    ast_parent = _ast_parent_name(ast_unit, ast_units_by_id)
    scip_parent = _scip_parent_name(scip_unit, scip_supports_by_unit)
    if not ast_parent and not scip_parent:
        return 1.0
    if ast_parent and scip_parent and ast_parent == scip_parent:
        return 1.0
    return 0.0


def _occurrence_support_score(
    scip_unit: IRCodeUnit, scip_supports_by_unit: dict[str, list[IRUnitSupport]]
) -> float:
    occurrences = [
        s
        for s in scip_supports_by_unit.get(scip_unit.unit_id, [])
        if s.support_kind == "occurrence"
    ]
    if not occurrences:
        return 0.0
    if any((s.role or "") == "definition" for s in occurrences):
        return 1.0
    return min(1.0, 0.5 + 0.1 * len(occurrences))


def _embedding_score(
    ast_unit: IRCodeUnit,
    scip_unit: IRCodeUnit,
    ast_embeddings_by_unit: dict[str, IRUnitEmbedding],
) -> float:
    embedding = ast_embeddings_by_unit.get(ast_unit.unit_id)
    if embedding is None or not embedding.embedding_text:
        return 0.0
    text = embedding.embedding_text.lower()
    name = _normalize_name(scip_unit.display_name or scip_unit.qualified_name)
    return 1.0 if name and name in text else 0.0


def _candidate_score(
    ast_unit: IRCodeUnit,
    scip_unit: IRCodeUnit,
    ast_units_by_id: dict[str, IRCodeUnit],
    scip_supports_by_unit: dict[str, list[IRUnitSupport]],
    ast_embeddings_by_unit: dict[str, IRUnitEmbedding],
) -> float:
    overlap = _span_overlap_score(ast_unit, scip_unit)
    name_score = _name_score(ast_unit, scip_unit)
    if ast_unit.path != scip_unit.path or not _kind_compatible(
        ast_unit.kind, scip_unit.kind
    ):
        return 0.0
    if overlap <= 0.0 and name_score <= 0.0:
        return 0.0
    kind_score = 1.0
    parent_score = _parent_context_score(
        ast_unit, scip_unit, ast_units_by_id, scip_supports_by_unit
    )
    signature_score = _signature_score(ast_unit, scip_unit)
    occurrence_score = _occurrence_support_score(scip_unit, scip_supports_by_unit)
    semantic_score = _embedding_score(ast_unit, scip_unit, ast_embeddings_by_unit)
    score = (
        0.45 * overlap
        + 0.25 * name_score
        + 0.15 * kind_score
        + 0.10 * parent_score
        + 0.03 * signature_score
        + 0.01 * occurrence_score
        + 0.01 * semantic_score
    )
    return min(1.0, score)


def _select_matches(
    ast_units: list[IRCodeUnit],
    scip_units: list[IRCodeUnit],
    ast_units_by_id: dict[str, IRCodeUnit],
    scip_supports_by_unit: dict[str, list[IRUnitSupport]],
    ast_embeddings_by_unit: dict[str, IRUnitEmbedding],
) -> tuple[dict[str, tuple[str, float]], dict[str, tuple[str, float]], _MatchStats]:
    stats = _merge_metric_base()
    stats["match_buckets"] = 1
    ast_by_stable: dict[str, list[IRCodeUnit]] = defaultdict(list)
    ast_by_name: dict[str, list[IRCodeUnit]] = defaultdict(list)
    ast_by_span: dict[int, list[IRCodeUnit]] = defaultdict(list)
    for ast_unit in ast_units:
        stable = _stable_unit_key(ast_unit)
        if stable:
            ast_by_stable[stable].append(ast_unit)
        name = _normalize_name(ast_unit.display_name or ast_unit.qualified_name)
        if name:
            ast_by_name[name].append(ast_unit)
        for bucket_key in _span_bucket_keys(ast_unit):
            ast_by_span[bucket_key].append(ast_unit)

    scored_pairs: list[_CandidatePair] = []
    for scip_unit in scip_units:
        direct_candidates: dict[str, IRCodeUnit] = {}
        stable = _stable_unit_key(scip_unit)
        if stable:
            direct_candidates.update(
                (unit.unit_id, unit) for unit in ast_by_stable.get(stable, [])
            )
        name = _normalize_name(scip_unit.display_name or scip_unit.qualified_name)
        if name:
            direct_candidates.update(
                (unit.unit_id, unit) for unit in ast_by_name.get(name, [])
            )
            if not direct_candidates:
                direct_candidates.update(
                    (unit.unit_id, unit)
                    for ast_name, named_units in ast_by_name.items()
                    if name in ast_name or ast_name in name
                    for unit in named_units
                )

        candidate_map = dict(direct_candidates)
        span_candidate_count_before = len(candidate_map)
        for bucket_key in _span_bucket_keys(scip_unit):
            candidate_map.update(
                (unit.unit_id, unit) for unit in ast_by_span.get(bucket_key, [])
            )
        if not direct_candidates and len(candidate_map) > span_candidate_count_before:
            stats["span_widened_units"] += 1

        candidate_units = sorted(
            candidate_map.values(),
            key=lambda ast_unit: _candidate_order_key(ast_unit, scip_unit),
        )
        stats["candidate_pairs_bucketed"] += len(candidate_units)
        if len(candidate_units) > MERGE_CANDIDATE_FANOUT_LIMIT:
            stats["fanout_capped_units"] += 1
            dropped = len(candidate_units) - MERGE_CANDIDATE_FANOUT_LIMIT
            stats["candidate_pairs_dropped_by_fanout"] += dropped
            candidate_units = candidate_units[:MERGE_CANDIDATE_FANOUT_LIMIT]

        for ast_unit in candidate_units:
            stats["candidate_pairs_scored"] += 1
            score = _candidate_score(
                ast_unit,
                scip_unit,
                ast_units_by_id,
                scip_supports_by_unit,
                ast_embeddings_by_unit,
            )
            if score >= CANDIDATE_MATCH_THRESHOLD:
                scored_pairs.append((ast_unit.unit_id, scip_unit.unit_id, score))

    scored_by_ast: dict[str, list[_CandidatePair]] = defaultdict(list)
    for pair in scored_pairs:
        scored_by_ast[pair[0]].append(pair)
    bounded_pairs: list[_CandidatePair] = []
    for ast_id, pairs in scored_by_ast.items():
        pairs.sort(key=lambda pair: (-pair[2], pair[1], ast_id))
        kept_pairs = pairs
        if len(pairs) > MERGE_CANDIDATE_FANOUT_LIMIT:
            stats["ast_fanout_capped_units"] += 1
            stats["candidate_pairs_dropped_by_ast_fanout"] += (
                len(pairs) - MERGE_CANDIDATE_FANOUT_LIMIT
            )
            kept_pairs = pairs[:MERGE_CANDIDATE_FANOUT_LIMIT]
        bounded_pairs.extend(kept_pairs)
    stats["candidate_pairs_kept"] = len(bounded_pairs)

    primary_matches: dict[str, tuple[str, float]] = {}
    candidate_matches: dict[str, tuple[str, float]] = {}
    if not bounded_pairs:
        return primary_matches, candidate_matches, stats

    selected_pairs = _native_bipartite_matching(bounded_pairs, stats)
    for ast_id, scip_id, score in selected_pairs:
        if score >= PRIMARY_MATCH_THRESHOLD:
            primary_matches[scip_id] = (ast_id, score)
            stats["primary_pairs"] += 1
        else:
            candidate_matches[scip_id] = (ast_id, score)
            stats["candidate_anchor_pairs"] += 1
    stats["selected_pairs"] = len(primary_matches) + len(candidate_matches)
    return primary_matches, candidate_matches, stats


def _native_bipartite_matching(
    pairs: list[_CandidatePair], stats: _MatchStats
) -> list[_CandidatePair]:
    by_ast: dict[str, list[_CandidatePair]] = defaultdict(list)
    by_scip: dict[str, list[_CandidatePair]] = defaultdict(list)
    for pair in pairs:
        ast_id, scip_id, _score = pair
        by_ast[ast_id].append(pair)
        by_scip[scip_id].append(pair)

    selected: list[_CandidatePair] = []
    seen_ast: set[str] = set()
    for root_ast in sorted(by_ast):
        if root_ast in seen_ast:
            continue
        component_ast: set[str] = set()
        component_scip: set[str] = set()
        stack_ast = [root_ast]
        stack_scip: list[str] = []
        while stack_ast or stack_scip:
            while stack_ast:
                ast_id = stack_ast.pop()
                if ast_id in component_ast:
                    continue
                component_ast.add(ast_id)
                seen_ast.add(ast_id)
                for _pair_ast, scip_id, _score in by_ast.get(ast_id, []):
                    if scip_id not in component_scip:
                        stack_scip.append(scip_id)
            while stack_scip:
                scip_id = stack_scip.pop()
                if scip_id in component_scip:
                    continue
                component_scip.add(scip_id)
                for ast_id, _pair_scip, _score in by_scip.get(scip_id, []):
                    if ast_id not in component_ast:
                        stack_ast.append(ast_id)

        component_pairs = [
            pair
            for pair in pairs
            if pair[0] in component_ast and pair[1] in component_scip
        ]
        if (
            len(component_ast) + len(component_scip) <= MERGE_EXACT_COMPONENT_NODE_LIMIT
            and min(len(component_ast), len(component_scip))
            <= MERGE_EXACT_COMPONENT_SIDE_LIMIT
        ):
            stats["native_exact_components"] += 1
            selected.extend(_exact_component_matching(component_pairs))
        else:
            stats["native_greedy_components"] += 1
            selected.extend(_greedy_component_matching(component_pairs))
    selected.sort(key=lambda pair: (pair[1], pair[0]))
    return selected


def _greedy_component_matching(pairs: list[_CandidatePair]) -> list[_CandidatePair]:
    selected: list[_CandidatePair] = []
    used_ast: set[str] = set()
    used_scip: set[str] = set()
    for ast_id, scip_id, score in sorted(
        pairs, key=lambda pair: (-pair[2], pair[1], pair[0])
    ):
        if ast_id in used_ast or scip_id in used_scip:
            continue
        used_ast.add(ast_id)
        used_scip.add(scip_id)
        selected.append((ast_id, scip_id, score))
    return selected


def _exact_component_matching(pairs: list[_CandidatePair]) -> list[_CandidatePair]:
    ast_ids = sorted({pair[0] for pair in pairs})
    scip_ids = sorted({pair[1] for pair in pairs})
    scip_index = {scip_id: index for index, scip_id in enumerate(scip_ids)}
    pairs_by_ast_index: dict[int, list[tuple[int, _CandidatePair]]] = defaultdict(list)
    for pair in pairs:
        ast_id, scip_id, _score = pair
        pairs_by_ast_index[ast_ids.index(ast_id)].append((scip_index[scip_id], pair))
    for entries in pairs_by_ast_index.values():
        entries.sort(key=lambda item: (-item[1][2], item[1][1], item[1][0]))

    @cache
    def solve(
        ast_index: int, used_scip_mask: int
    ) -> tuple[float, tuple[_CandidatePair, ...]]:
        if ast_index >= len(ast_ids):
            return 0.0, ()
        best_score, best_pairs = solve(ast_index + 1, used_scip_mask)
        for scip_idx, pair in pairs_by_ast_index.get(ast_index, []):
            bit = 1 << scip_idx
            if used_scip_mask & bit:
                continue
            tail_score, tail_pairs = solve(ast_index + 1, used_scip_mask | bit)
            candidate_score = pair[2] + tail_score
            candidate_pairs = (pair, *tail_pairs)
            if (
                candidate_score > best_score
                or (
                    candidate_score == best_score
                    and len(candidate_pairs) > len(best_pairs)
                )
                or (
                    candidate_score == best_score
                    and len(candidate_pairs) == len(best_pairs)
                    and candidate_pairs < best_pairs
                )
            ):
                best_score = candidate_score
                best_pairs = candidate_pairs
        return best_score, best_pairs

    _score, selected = solve(0, 0)
    return list(selected)


def _upsert_relation(
    merged: dict[tuple[str, str, str], IRRelation], relation: IRRelation
) -> None:
    key = (relation.src_unit_id, relation.dst_unit_id, relation.relation_type)
    existing = merged.get(key)
    if existing is None:
        merged[key] = relation
        return
    materialized = _clone_relation(existing)
    materialized.support_sources.update(relation.support_sources)
    materialized.support_ids = sorted(
        set(materialized.support_ids) | set(relation.support_ids)
    )
    materialized.pending_capabilities = (
        materialized.pending_capabilities & relation.pending_capabilities
    )
    if resolution_rank(relation.resolution_state) > resolution_rank(
        materialized.resolution_state
    ):
        materialized.resolution_state = relation.resolution_state
    materialized.metadata.update(
        {k: v for k, v in relation.metadata.items() if v is not None}
    )
    merged[key] = materialized


def _find_enclosing_unit_id(
    path: str,
    start_line: int | None,
    end_line: int | None,
    merged_units: list[IRCodeUnit],
) -> str | None:
    candidates = []
    for unit in merged_units:
        if unit.path != path or unit.kind in {"file", "doc"}:
            continue
        if (
            start_line is None
            or end_line is None
            or not unit.start_line
            or not unit.end_line
        ):
            continue
        if (
            unit.start_line <= start_line <= unit.end_line
            and unit.start_line <= end_line <= unit.end_line
        ):
            span = unit.end_line - unit.start_line
            candidates.append((span, unit.unit_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def _merge_unit(ast_unit: IRCodeUnit, scip_unit: IRCodeUnit, score: float) -> None:
    ast_unit.source_set.update(scip_unit.source_set)
    if not ast_unit.qualified_name and scip_unit.qualified_name:
        ast_unit.qualified_name = scip_unit.qualified_name
    if not ast_unit.signature and scip_unit.signature:
        ast_unit.signature = scip_unit.signature
    if scip_unit.primary_anchor_symbol_id:
        ast_unit.primary_anchor_symbol_id = scip_unit.primary_anchor_symbol_id
    ast_unit.anchor_symbol_ids = sorted(
        set(ast_unit.anchor_symbol_ids) | set(scip_unit.anchor_symbol_ids)
    )
    ast_unit.anchor_coverage = 1.0
    aliases = set((ast_unit.metadata or {}).get("aliases", []))
    aliases.add(scip_unit.unit_id)
    ast_unit.metadata["aliases"] = sorted(aliases)
    ast_unit.metadata["alignment"] = {
        "method": "bucketed_native_matching",
        "score": round(score, 6),
    }


def merge_ir(ast_snapshot: IRSnapshot, scip_snapshot: IRSnapshot | None) -> IRSnapshot:
    if scip_snapshot is None:
        return IRSnapshot(
            repo_name=ast_snapshot.repo_name,
            snapshot_id=ast_snapshot.snapshot_id,
            branch=ast_snapshot.branch,
            commit_id=ast_snapshot.commit_id,
            tree_id=ast_snapshot.tree_id,
            units=[_clone_unit(u) for u in ast_snapshot.units],
            supports=[_clone_support(s) for s in ast_snapshot.supports],
            relations=[_clone_relation(r) for r in ast_snapshot.relations],
            embeddings=[_clone_embedding(e) for e in ast_snapshot.embeddings],
            metadata=dict(ast_snapshot.metadata),
        )

    merged_units = list(ast_snapshot.units)
    merged_supports = list(ast_snapshot.supports)
    merged_embeddings = list(ast_snapshot.embeddings)
    relation_map: dict[tuple[str, str, str], IRRelation] = {}
    for relation in ast_snapshot.relations:
        _upsert_relation(relation_map, relation)

    merged_units_by_id = {unit.unit_id: unit for unit in merged_units}
    merged_unit_index_by_id = {
        unit.unit_id: index for index, unit in enumerate(merged_units)
    }
    cloned_ast_units: set[str] = set()

    def mutable_ast_unit(unit_id: str) -> IRCodeUnit:
        unit = merged_units_by_id[unit_id]
        if unit_id in cloned_ast_units:
            return unit
        materialized = _clone_unit(unit)
        merged_units_by_id[unit_id] = materialized
        index = merged_unit_index_by_id[unit_id]
        merged_units[index] = materialized
        cloned_ast_units.add(unit_id)
        return materialized

    ast_units_by_id = merged_units_by_id
    scip_units_by_id = {unit.unit_id: unit for unit in scip_snapshot.units}
    ast_embeddings_by_unit = {
        embedding.unit_id: embedding for embedding in merged_embeddings
    }
    scip_supports_by_unit: dict[str, list[IRUnitSupport]] = defaultdict(list)
    for support in scip_snapshot.supports:
        scip_supports_by_unit[support.unit_id].append(support)

    ast_file_by_path = {
        unit.path: unit.unit_id for unit in merged_units if unit.kind == "file"
    }
    scip_to_canonical: dict[str, str] = {}

    for unit in scip_snapshot.units:
        if unit.kind == "file" and unit.path in ast_file_by_path:
            scip_to_canonical[unit.unit_id] = ast_file_by_path[unit.path]

    ast_units_by_path_bucket: dict[tuple[str, str], list[IRCodeUnit]] = defaultdict(
        list
    )
    scip_units_by_path_bucket: dict[tuple[str, str], list[IRCodeUnit]] = defaultdict(
        list
    )
    for unit in merged_units:
        if unit.kind == "file":
            continue
        bucket = (
            "container"
            if unit.kind in CONTAINER_KINDS
            else "callable"
            if unit.kind in CALLABLE_KINDS
            else "other"
        )
        ast_units_by_path_bucket[(unit.path, bucket)].append(unit)
    for unit in scip_snapshot.units:
        if unit.kind == "file":
            continue
        bucket = (
            "container"
            if unit.kind in CONTAINER_KINDS
            else "callable"
            if unit.kind in CALLABLE_KINDS
            else "other"
        )
        scip_units_by_path_bucket[(unit.path, bucket)].append(unit)

    candidate_anchor_hints: dict[str, set[str]] = defaultdict(set)
    merge_metrics = _merge_metric_base()
    for key in sorted(set(ast_units_by_path_bucket) | set(scip_units_by_path_bucket)):
        ast_bucket_units = ast_units_by_path_bucket.get(key, [])
        scip_bucket_units = scip_units_by_path_bucket.get(key, [])
        if not ast_bucket_units or not scip_bucket_units:
            continue
        primary, candidates, match_stats = _select_matches(
            ast_bucket_units,
            scip_bucket_units,
            ast_units_by_id=ast_units_by_id,
            scip_supports_by_unit=scip_supports_by_unit,
            ast_embeddings_by_unit=ast_embeddings_by_unit,
        )
        _add_merge_metrics(merge_metrics, match_stats)
        for scip_id, (ast_id, score) in primary.items():
            scip_to_canonical[scip_id] = ast_id
            _merge_unit(mutable_ast_unit(ast_id), scip_units_by_id[scip_id], score)
        for scip_id, (ast_id, _) in candidates.items():
            if scip_id in scip_to_canonical:
                continue
            scip_unit = scip_units_by_id[scip_id]
            if scip_unit.primary_anchor_symbol_id:
                candidate_anchor_hints[ast_id].add(scip_unit.primary_anchor_symbol_id)

    for ast_id, anchors in candidate_anchor_hints.items():
        if ast_id not in ast_units_by_id:
            continue
        unit = mutable_ast_unit(ast_id)
        unit.candidate_anchor_symbol_ids = sorted(
            set(unit.candidate_anchor_symbol_ids) | anchors
        )
        if not unit.anchor_coverage and unit.candidate_anchor_symbol_ids:
            unit.anchor_coverage = 0.5

    for scip_unit in scip_snapshot.units:
        canonical_id = scip_to_canonical.get(scip_unit.unit_id)
        if canonical_id:
            continue
        synthetic = _clone_unit(scip_unit)
        if synthetic.kind == "file" and synthetic.path in ast_file_by_path:
            continue
        if synthetic.kind != "file":
            synthetic_parent = ast_file_by_path.get(
                synthetic.path, synthetic.parent_unit_id
            )
            synthetic.parent_unit_id = synthetic_parent
        merged_units.append(synthetic)
        merged_units_by_id[synthetic.unit_id] = synthetic
        scip_to_canonical[scip_unit.unit_id] = synthetic.unit_id
        if synthetic.kind != "file" and synthetic.parent_unit_id:
            _upsert_relation(
                relation_map,
                IRRelation(
                    relation_id=f"rel:contain:{synthetic.parent_unit_id}:{synthetic.unit_id}",
                    src_unit_id=synthetic.parent_unit_id,
                    dst_unit_id=synthetic.unit_id,
                    relation_type="contain",
                    resolution_state="anchored",
                    support_sources={"scip"},
                    metadata={
                        "source": "scip",
                        "doc_id": ast_file_by_path.get(
                            synthetic.path, synthetic.parent_unit_id
                        ),
                    },
                ),
            )

    for support in scip_snapshot.supports:
        materialized = _clone_support(support)
        materialized.unit_id = scip_to_canonical.get(
            materialized.unit_id, materialized.unit_id
        )
        merged_supports.append(materialized)

    for relation in scip_snapshot.relations:
        materialized = _clone_relation(relation)
        materialized.src_unit_id = scip_to_canonical.get(
            materialized.src_unit_id, materialized.src_unit_id
        )
        materialized.dst_unit_id = scip_to_canonical.get(
            materialized.dst_unit_id, materialized.dst_unit_id
        )
        _upsert_relation(relation_map, materialized)

    for support in merged_supports:
        if (
            support.source != "scip"
            or support.support_kind != "occurrence"
            or (support.role or "") not in REF_ROLES
        ):
            continue
        unit_id = support.unit_id
        target = merged_units_by_id.get(unit_id)
        if target is None:
            continue
        occurrence_path = support.path or target.path
        source_unit_id = _find_enclosing_unit_id(
            occurrence_path, support.start_line, support.end_line, merged_units
        )
        if not source_unit_id:
            source_unit_id = ast_file_by_path.get(occurrence_path) or next(
                (
                    unit.unit_id
                    for unit in merged_units
                    if unit.kind == "file" and unit.path == occurrence_path
                ),
                "",
            )
        if not source_unit_id:
            continue
        resolution_state = (
            "anchored" if target.primary_anchor_symbol_id else "candidate"
        )
        _upsert_relation(
            relation_map,
            IRRelation(
                relation_id=f"rel:ref:{source_unit_id}:{unit_id}:{support.support_id}",
                src_unit_id=source_unit_id,
                dst_unit_id=unit_id,
                relation_type="ref",
                resolution_state=resolution_state,
                support_sources={"scip"},
                support_ids=[support.support_id],
                metadata={
                    "role": support.role,
                    "source": "scip",
                    "doc_id": (support.metadata or {}).get("doc_id"),
                },
            ),
        )

    return IRSnapshot(
        repo_name=ast_snapshot.repo_name,
        snapshot_id=ast_snapshot.snapshot_id,
        branch=ast_snapshot.branch,
        commit_id=ast_snapshot.commit_id,
        tree_id=ast_snapshot.tree_id,
        units=merged_units,
        supports=merged_supports,
        relations=list(relation_map.values()),
        embeddings=merged_embeddings,
        metadata={
            "source_modes": sorted(
                set(ast_snapshot.metadata.get("source_modes", []))
                | set(scip_snapshot.metadata.get("source_modes", []))
            ),
            "merge_algorithm": "bucketed_native_alignment_v2",
            "merge_metrics": merge_metrics,
        },
    )
