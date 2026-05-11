"""Apply semantic resolver patches onto canonical IR snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from ...ir.types import (
    IRCodeUnit,
    IRRelation,
    IRSnapshot,
    IRUnitEmbedding,
    IRUnitSupport,
    resolution_rank,
)
from .base import ResolutionPatch, ResolutionTier


def _patch_jsonable(value: Any) -> Any:
    """Normalize resolver-owned metadata without dataclass/object expansion."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _patch_jsonable(item)
            for key, item in cast(Mapping[Any, Any], value).items()
        }
    if isinstance(value, (list, tuple)):
        return [_patch_jsonable(item) for item in cast(Sequence[Any], value)]
    if isinstance(value, set):
        return [
            _patch_jsonable(item) for item in sorted(cast(set[Any], value), key=str)
        ]
    return repr(value)


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return cast(dict[str, Any], _patch_jsonable(dict(value or {})))


def _clone_unit(unit: IRCodeUnit) -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=unit.unit_id,
        kind=unit.kind,
        path=unit.path,
        language=unit.language,
        display_name=unit.display_name,
        qualified_name=unit.qualified_name,
        signature=unit.signature,
        docstring=unit.docstring,
        summary=unit.summary,
        start_line=unit.start_line,
        start_col=unit.start_col,
        end_line=unit.end_line,
        end_col=unit.end_col,
        parent_unit_id=unit.parent_unit_id,
        primary_anchor_symbol_id=unit.primary_anchor_symbol_id,
        anchor_symbol_ids=list(unit.anchor_symbol_ids),
        candidate_anchor_symbol_ids=list(unit.candidate_anchor_symbol_ids),
        anchor_coverage=unit.anchor_coverage,
        source_set=set(unit.source_set),
        metadata=_metadata(unit.metadata),
    )


def _clone_support(support: IRUnitSupport) -> IRUnitSupport:
    return IRUnitSupport(
        support_id=support.support_id,
        unit_id=support.unit_id,
        source=support.source,
        support_kind=support.support_kind,
        external_id=support.external_id,
        role=support.role,
        path=support.path,
        display_name=support.display_name,
        qualified_name=support.qualified_name,
        signature=support.signature,
        enclosing_external_id=support.enclosing_external_id,
        start_line=support.start_line,
        start_col=support.start_col,
        end_line=support.end_line,
        end_col=support.end_col,
        metadata=_metadata(support.metadata),
    )


def _clone_relation(relation: IRRelation) -> IRRelation:
    return IRRelation(
        relation_id=relation.relation_id,
        src_unit_id=relation.src_unit_id,
        dst_unit_id=relation.dst_unit_id,
        relation_type=relation.relation_type,
        resolution_state=relation.resolution_state,
        support_sources=set(relation.support_sources),
        support_ids=list(relation.support_ids),
        pending_capabilities=set(relation.pending_capabilities),
        metadata=_metadata(relation.metadata),
    )


def _clone_embedding(embedding: IRUnitEmbedding) -> IRUnitEmbedding:
    return IRUnitEmbedding(
        embedding_id=embedding.embedding_id,
        unit_id=embedding.unit_id,
        source=embedding.source,
        vector=list(embedding.vector) if embedding.vector is not None else None,
        embedding_text=embedding.embedding_text,
        model_id=embedding.model_id,
        metadata=_metadata(embedding.metadata),
    )


def _source_preference(relation: IRRelation) -> int:
    """Rank a relation by its best evidence source.

    Higher value = stronger evidence.  Named resolvers rank above
    ``fc_structure``.  Any relation with ``resolution_tier ==
    "compiler_confirmed"`` is boosted to at least rank 2 (matching
    SCIP).  New resolvers that set the tier correctly are automatically
    ranked without updating the preferences dict.
    """
    # fmt: off
    preferences: dict[str, int] = {
        "fc_structure": 0,
        "c_resolver": 1, "cpp_resolver": 1,
        "javascript_resolver": 1, "typescript_resolver": 1,
        "java_resolver": 1, "go_resolver": 1, "rust_resolver": 1,
        "csharp_resolver": 1, "zig_resolver": 1,
        "fortran_resolver": 1, "julia_resolver": 1,
        "python_resolver": 1,
    }
    # fmt: on
    sources = set(relation.support_sources)
    if relation.source:
        sources.add(relation.source)
    base_pref = max((preferences.get(source, 0) for source in sources), default=0)
    tier = (relation.metadata or {}).get("resolution_tier", "")
    if tier == ResolutionTier.COMPILER_CONFIRMED:
        base_pref = max(base_pref, 2)
    return base_pref


def _relation_key(relation: IRRelation) -> tuple[str, ...]:
    metadata = relation.metadata or {}
    if relation.relation_type == "import" and metadata.get("module"):
        return ("import", relation.src_unit_id, str(metadata["module"]))
    if relation.relation_type == "inherit" and (
        metadata.get("base_name") or metadata.get("base")
    ):
        return (
            "inherit",
            relation.src_unit_id,
            str(metadata.get("base_name") or metadata["base"]),
        )
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
        existing.pending_capabilities = (
            existing.pending_capabilities & candidate.pending_capabilities
        )
        if resolution_rank(candidate.resolution_state) > resolution_rank(
            existing.resolution_state
        ):
            existing.resolution_state = candidate.resolution_state
        existing.metadata = _metadata(
            {**(existing.metadata or {}), **(candidate.metadata or {})}
        )
        return existing

    existing_order = (
        resolution_rank(existing.resolution_state),
        _source_preference(existing),
    )
    candidate_order = (
        resolution_rank(candidate.resolution_state),
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
    metadata: dict[str, Any] = _metadata(snapshot.metadata)
    for key, value in (patch.metadata_updates or {}).items():
        if key == "semantic_resolver_runs":
            existing = cast(list[Any], metadata.get(key) or [])
            incoming = cast(list[Any], value if isinstance(value, list) else [value])
            metadata[key] = _patch_jsonable([*existing, *incoming])
            continue
        metadata[key] = _patch_jsonable(value)

    unit_by_id = {unit.unit_id: unit for unit in units}
    for unit_id, updates in patch.unit_metadata_updates.items():
        unit = unit_by_id.get(unit_id)
        if unit is None:
            continue
        unit.metadata = _metadata({**(unit.metadata or {}), **(updates or {})})

    support_by_id = {support.support_id: support for support in supports}
    for support in patch.supports:
        materialized = _clone_support(support)
        if materialized.support_id in support_by_id:
            existing = support_by_id[materialized.support_id]
            existing.metadata = _metadata(
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
            if (
                not existing.enclosing_external_id
                and materialized.enclosing_external_id
            ):
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
