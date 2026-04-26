"""
Merge structure units and SCIP anchors into a canonical unit-grounded snapshot.
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx

from .semantic_ir import (
    IRCodeUnit,
    IRRelation,
    IRSnapshot,
    IRUnitEmbedding,
    IRUnitSupport,
)

PRIMARY_MATCH_THRESHOLD = 0.65
CANDIDATE_MATCH_THRESHOLD = 0.50
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


def _clone_unit(unit: IRCodeUnit) -> IRCodeUnit:
    return IRCodeUnit.from_dict(unit.to_dict())


def _clone_support(support: IRUnitSupport) -> IRUnitSupport:
    return IRUnitSupport.from_dict(support.to_dict())


def _clone_relation(relation: IRRelation) -> IRRelation:
    return IRRelation.from_dict(relation.to_dict())


def _clone_embedding(embedding: IRUnitEmbedding) -> IRUnitEmbedding:
    return IRUnitEmbedding.from_dict(embedding.to_dict())


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
) -> tuple[dict[str, tuple[str, float]], dict[str, tuple[str, float]]]:
    graph = nx.Graph()
    scores: dict[tuple[str, str], float] = {}
    for ast_unit in ast_units:
        for scip_unit in scip_units:
            score = _candidate_score(
                ast_unit,
                scip_unit,
                ast_units_by_id,
                scip_supports_by_unit,
                ast_embeddings_by_unit,
            )
            if score < CANDIDATE_MATCH_THRESHOLD:
                continue
            left = f"ast::{ast_unit.unit_id}"
            right = f"scip::{scip_unit.unit_id}"
            graph.add_edge(left, right, weight=int(score * 10000))
            scores[(ast_unit.unit_id, scip_unit.unit_id)] = score

    primary_matches: dict[str, tuple[str, float]] = {}
    candidate_matches: dict[str, tuple[str, float]] = {}
    if graph.number_of_edges() == 0:
        return primary_matches, candidate_matches

    matching = nx.algorithms.matching.max_weight_matching(
        graph, maxcardinality=False, weight="weight"
    )
    for pair in matching:
        left, right = pair
        if left.startswith("scip::"):
            left, right = right, left
        ast_id = left.replace("ast::", "", 1)
        scip_id = right.replace("scip::", "", 1)
        score = scores[(ast_id, scip_id)]
        if score >= PRIMARY_MATCH_THRESHOLD:
            primary_matches[scip_id] = (ast_id, score)
        else:
            candidate_matches[scip_id] = (ast_id, score)
    return primary_matches, candidate_matches


def _upsert_relation(
    merged: dict[tuple[str, str, str], IRRelation], relation: IRRelation
) -> None:
    key = (relation.src_unit_id, relation.dst_unit_id, relation.relation_type)
    existing = merged.get(key)
    if existing is None:
        merged[key] = relation
        return
    existing.support_sources.update(relation.support_sources)
    existing.support_ids = sorted(set(existing.support_ids) | set(relation.support_ids))
    if _resolution_rank(relation.resolution_state) > _resolution_rank(
        existing.resolution_state
    ):
        existing.resolution_state = relation.resolution_state
    existing.metadata.update(
        {k: v for k, v in relation.metadata.items() if v is not None}
    )


def _resolution_rank(value: str) -> int:
    return {"candidate": 0, "structural": 1, "anchored": 2}.get(value, 0)


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
        "method": "scoped_max_weight_matching",
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

    merged_units = [_clone_unit(unit) for unit in ast_snapshot.units]
    merged_supports = [_clone_support(support) for support in ast_snapshot.supports]
    merged_embeddings = [
        _clone_embedding(embedding) for embedding in ast_snapshot.embeddings
    ]
    relation_map: dict[tuple[str, str, str], IRRelation] = {}
    for relation in ast_snapshot.relations:
        _upsert_relation(relation_map, _clone_relation(relation))

    merged_units_by_id = {unit.unit_id: unit for unit in merged_units}
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
    for key in sorted(set(ast_units_by_path_bucket) | set(scip_units_by_path_bucket)):
        ast_bucket_units = ast_units_by_path_bucket.get(key, [])
        scip_bucket_units = scip_units_by_path_bucket.get(key, [])
        if not ast_bucket_units or not scip_bucket_units:
            continue
        primary, candidates = _select_matches(
            ast_bucket_units,
            scip_bucket_units,
            ast_units_by_id=ast_units_by_id,
            scip_supports_by_unit=scip_supports_by_unit,
            ast_embeddings_by_unit=ast_embeddings_by_unit,
        )
        for scip_id, (ast_id, score) in primary.items():
            scip_to_canonical[scip_id] = ast_id
            _merge_unit(ast_units_by_id[ast_id], scip_units_by_id[scip_id], score)
        for scip_id, (ast_id, _) in candidates.items():
            if scip_id in scip_to_canonical:
                continue
            scip_unit = scip_units_by_id[scip_id]
            if scip_unit.primary_anchor_symbol_id:
                candidate_anchor_hints[ast_id].add(scip_unit.primary_anchor_symbol_id)

    for ast_id, anchors in candidate_anchor_hints.items():
        if ast_id not in ast_units_by_id:
            continue
        unit = ast_units_by_id[ast_id]
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
        source_unit_id = _find_enclosing_unit_id(
            target.path, support.start_line, support.end_line, merged_units
        )
        if not source_unit_id:
            source_unit_id = ast_file_by_path.get(target.path) or next(
                (
                    unit.unit_id
                    for unit in merged_units
                    if unit.kind == "file" and unit.path == target.path
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
            "merge_algorithm": "scoped_max_weight_alignment_v1",
        },
    )
