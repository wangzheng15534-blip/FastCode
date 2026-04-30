"""Apply semantic resolver patches onto canonical IR snapshots."""

from __future__ import annotations

from typing import Any, cast

from ..semantic_ir import (
    IRCodeUnit,
    IRRelation,
    IRSnapshot,
    IRUnitEmbedding,
    IRUnitSupport,
)
from ..utils import safe_jsonable
from .base import ResolutionPatch


def _clone_unit(unit: IRCodeUnit) -> IRCodeUnit:
    return IRCodeUnit.from_dict(unit.to_dict())


def _clone_support(support: IRUnitSupport) -> IRUnitSupport:
    return IRUnitSupport.from_dict(support.to_dict())


def _clone_relation(relation: IRRelation) -> IRRelation:
    return IRRelation.from_dict(relation.to_dict())


def _clone_embedding(embedding: IRUnitEmbedding) -> IRUnitEmbedding:
    return IRUnitEmbedding.from_dict(embedding.to_dict())


def _resolution_rank(value: str) -> int:
    return {
        "candidate": 0,
        "structural": 1,
        "anchored": 2,
        "semantic": 3,
        "semantically_resolved": 3,
    }.get(value, 0)


def _source_preference(relation: IRRelation) -> int:
    return {"fc_structure": 0, "python_resolver": 1, "scip": 2}.get(
        relation.source, 0
    )


def _relation_key(relation: IRRelation) -> tuple[str, ...]:
    metadata = relation.metadata or {}
    if relation.relation_type == "import" and metadata.get("module"):
        return ("import", relation.src_unit_id, str(metadata["module"]))
    if relation.relation_type == "inherit" and metadata.get("base"):
        return ("inherit", relation.src_unit_id, str(metadata["base"]))
    return (relation.relation_type, relation.src_unit_id, relation.dst_unit_id)


def _merge_relation(existing: IRRelation, candidate: IRRelation) -> IRRelation:
    if (
        existing.src_unit_id == candidate.src_unit_id
        and existing.dst_unit_id == candidate.dst_unit_id
        and existing.relation_type == candidate.relation_type
    ):
        existing.support_sources.update(candidate.support_sources)
        existing.support_ids = sorted(
            set(existing.support_ids) | set(candidate.support_ids)
        )
        if _resolution_rank(candidate.resolution_state) > _resolution_rank(
            existing.resolution_state
        ):
            existing.resolution_state = candidate.resolution_state
        existing.metadata = safe_jsonable(
            {**(existing.metadata or {}), **(candidate.metadata or {})}
        )
        return existing

    existing_order = (
        _resolution_rank(existing.resolution_state),
        _source_preference(existing),
    )
    candidate_order = (
        _resolution_rank(candidate.resolution_state),
        _source_preference(candidate),
    )
    if candidate_order > existing_order:
        return _clone_relation(candidate)
    return existing


def apply_resolution_patch(snapshot: IRSnapshot, patch: ResolutionPatch) -> IRSnapshot:
    """Materialize a new snapshot with a resolver patch applied."""

    units = [_clone_unit(unit) for unit in snapshot.units]
    supports = [_clone_support(support) for support in snapshot.supports]
    embeddings = [_clone_embedding(embedding) for embedding in snapshot.embeddings]
    metadata: dict[str, Any] = safe_jsonable(snapshot.metadata)
    for key, value in (patch.metadata_updates or {}).items():
        if key == "semantic_resolver_runs":
            existing = cast(list[Any], metadata.get(key) or [])
            incoming = cast(list[Any], value if isinstance(value, list) else [value])
            metadata[key] = safe_jsonable([*existing, *incoming])
            continue
        metadata[key] = safe_jsonable(value)

    unit_by_id = {unit.unit_id: unit for unit in units}
    for unit_id, updates in patch.unit_metadata_updates.items():
        unit = unit_by_id.get(unit_id)
        if unit is None:
            continue
        unit.metadata = safe_jsonable({**(unit.metadata or {}), **(updates or {})})

    support_by_id = {support.support_id: support for support in supports}
    for support in patch.supports:
        materialized = _clone_support(support)
        if materialized.support_id in support_by_id:
            existing = support_by_id[materialized.support_id]
            existing.metadata = safe_jsonable(
                {**(existing.metadata or {}), **(materialized.metadata or {})}
            )
            if not existing.source and materialized.source:
                existing.source = materialized.source
            if not existing.support_kind and materialized.support_kind:
                existing.support_kind = materialized.support_kind
            if not existing.external_id and materialized.external_id:
                existing.external_id = materialized.external_id
            if not existing.role and materialized.role:
                existing.role = materialized.role
            if not existing.path and materialized.path:
                existing.path = materialized.path
            if not existing.display_name and materialized.display_name:
                existing.display_name = materialized.display_name
            if not existing.qualified_name and materialized.qualified_name:
                existing.qualified_name = materialized.qualified_name
            if not existing.signature and materialized.signature:
                existing.signature = materialized.signature
            if not existing.enclosing_external_id and materialized.enclosing_external_id:
                existing.enclosing_external_id = materialized.enclosing_external_id
            if existing.start_line is None and materialized.start_line is not None:
                existing.start_line = materialized.start_line
            if existing.start_col is None and materialized.start_col is not None:
                existing.start_col = materialized.start_col
            if existing.end_line is None and materialized.end_line is not None:
                existing.end_line = materialized.end_line
            if existing.end_col is None and materialized.end_col is not None:
                existing.end_col = materialized.end_col
            continue
        supports.append(materialized)
        support_by_id[materialized.support_id] = materialized

    relation_map: dict[tuple[str, ...], IRRelation] = {}
    for relation in snapshot.relations:
        materialized = _clone_relation(relation)
        relation_map[_relation_key(materialized)] = materialized
    for relation in patch.relations:
        materialized = _clone_relation(relation)
        key = _relation_key(materialized)
        existing = relation_map.get(key)
        if existing is None:
            relation_map[key] = materialized
        else:
            relation_map[key] = _merge_relation(existing, materialized)

    return IRSnapshot(
        repo_name=snapshot.repo_name,
        snapshot_id=snapshot.snapshot_id,
        branch=snapshot.branch,
        commit_id=snapshot.commit_id,
        tree_id=snapshot.tree_id,
        units=units,
        supports=supports,
        relations=list(relation_map.values()),
        embeddings=embeddings,
        metadata=metadata,
    )
