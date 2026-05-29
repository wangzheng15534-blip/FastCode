"""Apply semantic resolver patches onto canonical IR snapshots."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, Generic, SupportsIndex, TypeVar, overload

from fastcode.ir.types import (
    IRCodeUnit,
    IRRelation,
    IRSnapshot,
    IRUnitSupport,
    resolution_rank,
)
from fastcode.semantic.resolution import ResolutionPatch, ResolutionTier
from fastcode.utils.materialization import (
    BOUNDARY_SEMANTIC_PATCH_CHANGED_OBJECTS,
    BOUNDARY_SEMANTIC_PATCH_PRESERVED_OBJECTS,
    increment_materialization_boundary,
)

JSONScalar = bool | int | float | str | None
_T = TypeVar("_T")

_RUN_STRING_FIELDS = {
    "language",
    "source",
    "frontend_kind",
}
_RUN_BOOL_FIELDS = {
    "compiler_backed",
    "fallback",
    "helper_backed",
    "helper_failed",
}
_RUN_STRING_LIST_FIELDS = {
    "capabilities",
    "required_tools",
}
_STATS_STRING_FIELDS = {
    "language",
    "cost_class",
    "resolver_source",
    "frontend_kind",
}
_STATS_INT_FIELDS = {
    "helper_target_files",
    "supports_emitted",
    "skipped_edges",
}
_STATS_OPTIONAL_INT_FIELDS = {
    "helper_exit_code",
}
_STATS_BOOL_FIELDS = {
    "helper_failed",
    "skipped",
}
_STATS_STRING_LIST_FIELDS = {
    "capabilities",
    "helper_command",
    "helper_failure_codes",
    "required_tools",
}
_RESOLVER_METADATA_STRING_FIELDS = {
    "base",
    "base_name",
    "call_name",
    "doc_id",
    "extractor",
    "import_name",
    "import_path",
    "module",
    "relation_kind",
    "resolution_method",
    "resolution_tier",
    "resolver_language",
    "semantic_capability",
    "source",
    "source_name",
    "source_path",
    "target_element_id",
    "target_name",
    "target_path",
    "target_symbol",
    "target_unit_id",
}
_RESOLVER_METADATA_INT_FIELDS = {
    "level",
    "source_col",
    "source_line",
    "target_col",
    "target_line",
}
_RESOLVER_METADATA_STRING_LIST_FIELDS = {
    "resolver_capabilities",
}


class _DeferredSequence(list[_T], Generic[_T]):
    def __init__(
        self,
        *,
        length: int,
        item_getter: Callable[[int], _T],
        iter_factory: Callable[[], Iterable[_T]] | None = None,
    ) -> None:
        super().__init__()
        self._length = int(length)
        self._item_getter = item_getter
        self._iter_factory = iter_factory
        self._materialized: list[_T] | None = None

    def __len__(self) -> int:
        return self._length

    @overload
    def __getitem__(self, index: SupportsIndex) -> _T: ...

    @overload
    def __getitem__(self, index: slice) -> list[_T]: ...

    def __getitem__(self, index: SupportsIndex | slice) -> _T | list[_T]:
        materialized = self._materialized
        if materialized is None:
            materialized = (
                list(self._iter_factory())
                if self._iter_factory
                else [self._item_getter(i) for i in range(self._length)]
            )
            self._materialized = materialized
        return materialized[index]

    def __iter__(self) -> Iterator[_T]:
        materialized = self._materialized
        if materialized is not None:
            return iter(materialized)
        if self._iter_factory is None:
            return (self._item_getter(i) for i in range(self._length))
        return iter(self._iter_factory())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence):
            return False
        return list(self) == list(other)

    __hash__ = None  # type: ignore[assignment]

    def __contains__(self, item: object) -> bool:
        return any(candidate == item for candidate in self)

    def count(self, value: _T) -> int:
        return list(self).count(value)

    def index(
        self,
        value: _T,
        start: SupportsIndex = 0,
        stop: SupportsIndex = 9223372036854775807,
    ) -> int:
        return list(self).index(value, start, stop)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(len={self._length})"


def _sequence_with_replacements(
    base: list[_T],
    *,
    replacements: Mapping[int, _T] | None = None,
    appended: Sequence[_T] | None = None,
) -> list[_T]:
    replacement_map = dict(replacements or {})
    appended_items = tuple(appended or ())
    if not replacement_map and not appended_items:
        return base
    base_length = len(base)
    total_length = base_length + len(appended_items)

    def _item_getter(index: int) -> _T:
        if index < base_length:
            replacement = replacement_map.get(index)
            return base[index] if replacement is None else replacement
        return appended_items[index - base_length]

    def _iter_factory() -> Iterable[_T]:
        for index, item in enumerate(base):
            replacement = replacement_map.get(index)
            yield item if replacement is None else replacement
        yield from appended_items

    return _DeferredSequence(
        length=total_length,
        item_getter=_item_getter,
        iter_factory=_iter_factory,
    )


def _json_scalar(value: Any) -> JSONScalar:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _iterable_items(value: Any) -> Iterable[Any]:
    if isinstance(value, set):
        return sorted(value, key=str)
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Sequence):
        return value
    return (value,)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(item) for item in _iterable_items(value) if item is not None]


def _scalar_list(value: Any) -> list[JSONScalar]:
    if value is None:
        return []
    return [_json_scalar(item) for item in _iterable_items(value)]


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        coerced = _int_or_none(item)
        if coerced is not None:
            result[str(key)] = coerced
    return result


def _scalar_mapping(value: Any) -> dict[str, JSONScalar]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _json_scalar(item) for key, item in value.items()}


def _flat_patch_value(
    value: Any,
) -> JSONScalar | list[JSONScalar] | dict[str, JSONScalar]:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if type(value) is dict:
        return _scalar_mapping(value)
    if isinstance(value, Mapping):
        return repr(value)
    if isinstance(value, (set, Sequence)) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return _scalar_list(value)
    return repr(value)


def _copy_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _serialize_diagnostic(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {
            "language": "",
            "tool": "",
            "code": "",
            "message": str(value),
        }
    return {
        "language": str(value.get("language") or ""),
        "tool": str(value.get("tool") or ""),
        "code": str(value.get("code") or ""),
        "message": str(value.get("message") or ""),
    }


def _serialize_diagnostics(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    return [_serialize_diagnostic(item) for item in _iterable_items(value)]


def _serialize_resolver_stats(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}

    result: dict[str, Any] = {}
    for key, item in value.items():
        name = str(key)
        if name in _STATS_STRING_FIELDS:
            result[name] = _optional_string(item)
        elif name in _STATS_INT_FIELDS:
            result[name] = _int_or_none(item) or 0
        elif name in _STATS_OPTIONAL_INT_FIELDS:
            result[name] = _int_or_none(item)
        elif name in _STATS_BOOL_FIELDS:
            result[name] = bool(item)
        elif name in _STATS_STRING_LIST_FIELDS:
            result[name] = _string_list(item)
        elif name == "diagnostics":
            result[name] = _serialize_diagnostics(item)
        elif name == "relations_emitted":
            result[name] = _int_mapping(item)
        elif name == "helper_stats":
            result[name] = _scalar_mapping(item)
        else:
            result[name] = _json_scalar(item)
    return result


def _serialize_resolver_run(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"message": str(_json_scalar(value))}

    result: dict[str, Any] = {}
    for key, item in value.items():
        name = str(key)
        if name in _RUN_STRING_FIELDS:
            result[name] = _optional_string(item)
        elif name in _RUN_BOOL_FIELDS:
            result[name] = bool(item)
        elif name in _RUN_STRING_LIST_FIELDS:
            result[name] = _string_list(item)
        elif name == "stats":
            result[name] = _serialize_resolver_stats(item)
        else:
            result[name] = _flat_patch_value(item)
    return result


def _serialize_resolver_runs(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    return [_serialize_resolver_run(item) for item in _iterable_items(value)]


def _serialize_unit_metadata_updates(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {str(key): _flat_patch_value(item) for key, item in (value or {}).items()}


def _serialize_resolver_object_metadata(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in (value or {}).items():
        name = str(key)
        if name in _RESOLVER_METADATA_STRING_FIELDS:
            result[name] = _optional_string(item)
        elif name in _RESOLVER_METADATA_INT_FIELDS:
            result[name] = _int_or_none(item)
        elif name in _RESOLVER_METADATA_STRING_LIST_FIELDS:
            result[name] = _string_list(item)
        else:
            result[name] = _flat_patch_value(item)
    return result


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
        metadata=_copy_metadata(unit.metadata),
    )


def _clone_support(
    support: IRUnitSupport, *, resolver_payload: bool = False
) -> IRUnitSupport:
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
        metadata=(
            _serialize_resolver_object_metadata(support.metadata)
            if resolver_payload
            else _copy_metadata(support.metadata)
        ),
    )


def _clone_relation(
    relation: IRRelation, *, resolver_payload: bool = False
) -> IRRelation:
    return IRRelation(
        relation_id=relation.relation_id,
        src_unit_id=relation.src_unit_id,
        dst_unit_id=relation.dst_unit_id,
        relation_type=relation.relation_type,
        resolution_state=relation.resolution_state,
        support_sources=set(relation.support_sources),
        support_ids=list(relation.support_ids),
        pending_capabilities=set(relation.pending_capabilities),
        metadata=(
            _serialize_resolver_object_metadata(relation.metadata)
            if resolver_payload
            else _copy_metadata(relation.metadata)
        ),
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
        merged = _clone_relation(existing)
        merged.support_sources.update(candidate.support_sources)
        merged.support_ids = sorted(
            set(merged.support_ids) | set(candidate.support_ids)
        )
        merged.pending_capabilities = (
            merged.pending_capabilities & candidate.pending_capabilities
        )
        if resolution_rank(candidate.resolution_state) > resolution_rank(
            merged.resolution_state
        ):
            merged.resolution_state = candidate.resolution_state
        merged.metadata = {
            **(merged.metadata or {}),
            **(candidate.metadata or {}),
        }
        return merged

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

    preserved_object_count = (
        len(snapshot.units)
        + len(snapshot.supports)
        + len(snapshot.relations)
        + len(snapshot.embeddings)
    )
    if preserved_object_count:
        increment_materialization_boundary(
            BOUNDARY_SEMANTIC_PATCH_PRESERVED_OBJECTS,
            items=preserved_object_count,
        )

    metadata: dict[str, Any] = _copy_metadata(snapshot.metadata)
    for key, value in (patch.metadata_updates or {}).items():
        if key == "semantic_resolver_runs":
            existing = _serialize_resolver_runs(metadata.get(key))
            incoming = _serialize_resolver_runs(value)
            metadata[key] = [*existing, *incoming]
            continue
        metadata[str(key)] = _flat_patch_value(value)

    unit_replacements: dict[int, IRCodeUnit] = {}
    unit_by_id = {
        unit.unit_id: (index, unit) for index, unit in enumerate(snapshot.units)
    }
    changed_object_count = 0
    for unit_id, updates in patch.unit_metadata_updates.items():
        found = unit_by_id.get(unit_id)
        if found is None:
            continue
        index, original_unit = found
        unit = _clone_unit(original_unit)
        changed_object_count += 1
        unit.metadata = {
            **(unit.metadata or {}),
            **_serialize_unit_metadata_updates(updates),
        }
        unit_replacements[index] = unit
        unit_by_id[unit_id] = (index, unit)

    support_replacements: dict[int, IRUnitSupport] = {}
    appended_supports: list[IRUnitSupport] = []
    support_by_id = {
        support.support_id: (index, support)
        for index, support in enumerate(snapshot.supports)
    }
    for support in patch.supports:
        changed_object_count += 1
        materialized = _clone_support(support, resolver_payload=True)
        if materialized.support_id in support_by_id:
            index, original_support = support_by_id[materialized.support_id]
            existing = _clone_support(original_support)
            existing.metadata = {
                **(existing.metadata or {}),
                **(materialized.metadata or {}),
            }
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
            support_replacements[index] = existing
            support_by_id[materialized.support_id] = (index, existing)
            continue
        appended_supports.append(materialized)
        support_by_id[materialized.support_id] = (
            len(snapshot.supports) + len(appended_supports) - 1,
            materialized,
        )

    relation_map: dict[tuple[str, ...], IRRelation] = {}
    relation_order: list[tuple[str, ...]] = []
    for relation in snapshot.relations:
        key = _relation_key(relation)
        if key not in relation_map:
            relation_order.append(key)
        relation_map[key] = relation
    for relation in patch.relations:
        changed_object_count += 1
        materialized = _clone_relation(relation, resolver_payload=True)
        key = _relation_key(materialized)
        existing = relation_map.get(key)
        if existing is None:
            relation_order.append(key)
            relation_map[key] = materialized
        else:
            relation_map[key] = _merge_relation(existing, materialized)

    if changed_object_count:
        increment_materialization_boundary(
            BOUNDARY_SEMANTIC_PATCH_CHANGED_OBJECTS,
            items=changed_object_count,
        )

    updated = IRSnapshot(
        repo_name=snapshot.repo_name,
        snapshot_id=snapshot.snapshot_id,
        branch=snapshot.branch,
        commit_id=snapshot.commit_id,
        tree_id=snapshot.tree_id,
        metadata=metadata,
    )
    updated.units = _sequence_with_replacements(
        snapshot.units,
        replacements=unit_replacements,
    )
    updated.supports = _sequence_with_replacements(
        snapshot.supports,
        replacements=support_replacements,
        appended=appended_supports,
    )
    if patch.relations:
        updated.relations = _DeferredSequence(
            length=len(relation_order),
            item_getter=lambda index: relation_map[relation_order[index]],
            iter_factory=lambda: (relation_map[key] for key in relation_order),
        )
    else:
        updated.relations = snapshot.relations
    updated.embeddings = snapshot.embeddings
    return updated
