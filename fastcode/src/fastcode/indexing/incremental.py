"""
Incremental snapshot update via file identity diffing.

Compares file-level blob_oids or content hashes between snapshots to detect changes,
then produces a new snapshot that preserves unchanged units, relations,
and embeddings while replacing changed-file content with fresh extraction.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Any, Generic, SupportsIndex, TypeVar, overload

from ..ir.types import (
    IRCodeUnit,
    IRRelation,
    IRSnapshot,
    IRUnitEmbedding,
    IRUnitSupport,
)

_T = TypeVar("_T")


class _LazyIRSequence(list[_T], Generic[_T]):
    """Read-through sequence for incremental IR slices.

    Incremental updates can pass unchanged and changed component streams through
    the pipeline without allocating a merged list immediately. Random access
    still materializes a list because sequence indexing has no streaming form.
    """

    def __init__(
        self,
        iter_factory: Callable[[], Iterable[_T]],
        *,
        len_factory: Callable[[], int] | None = None,
    ) -> None:
        super().__init__()
        self._iter_factory = iter_factory
        self._len_factory = len_factory
        self._materialized: list[_T] | None = None

    def __iter__(self) -> Iterator[_T]:
        materialized = self._materialized
        if materialized is not None:
            return iter(materialized)
        return iter(self._iter_factory())

    def __len__(self) -> int:
        materialized = self._materialized
        if materialized is not None:
            return len(materialized)
        if self._len_factory is not None:
            return self._len_factory()
        return sum(1 for _item in self._iter_factory())

    @overload
    def __getitem__(self, index: SupportsIndex) -> _T: ...

    @overload
    def __getitem__(self, index: slice) -> list[_T]: ...

    def __getitem__(self, index: SupportsIndex | slice) -> _T | list[_T]:
        materialized = self._materialized
        if materialized is None:
            materialized = list(self._iter_factory())
            self._materialized = materialized
        return materialized[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence):
            return False
        return list(self) == list(other)

    __hash__ = None  # type: ignore[assignment]

    def __contains__(self, item: object) -> bool:
        return item in list(self)

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
        materialized = self._materialized
        if materialized is not None:
            return repr(list(materialized))
        return f"{type(self).__name__}(len={len(self)})"


@dataclass
class FileChangeSet:
    """Diff result comparing two snapshots' file-level content identities."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PlanChanges:
    """Canonical incremental change plan for one snapshot index run.

    `to_prefilter_payload()` is the compatibility adapter for older pipeline
    code that still reads top-level incremental prefilter fields.
    """

    previous_snapshot_id: str
    previous_artifact_key: str
    added_paths: tuple[str, ...] = ()
    modified_paths: tuple[str, ...] = ()
    removed_paths: tuple[str, ...] = ()
    unchanged_paths: tuple[str, ...] = ()
    reused_elements: int = 0
    reindexed_elements: int = 0
    reused_changed_embeddings: int = 0
    semantic_frontier_widened: bool = False
    api_frontier_changed_paths: tuple[str, ...] = ()
    package_scope_roots: tuple[str, ...] = ()
    change_kinds: tuple[str, ...] = ()
    interface_digest_changed_paths: tuple[str, ...] = ()
    interface_digests: Mapping[str, str] = field(default_factory=dict)
    dependency_frontier: Mapping[str, Any] = field(default_factory=dict)
    degraded_reasons: tuple[str, ...] = ()

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(sorted({*self.added_paths, *self.modified_paths}))

    @property
    def degraded(self) -> bool:
        return bool(self.degraded_reasons)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "fastcode.plan_changes.v1",
            "previous_snapshot_id": self.previous_snapshot_id,
            "previous_artifact_key": self.previous_artifact_key,
            "artifact_delta_mode": True,
            "counts": {
                "added": len(self.added_paths),
                "modified": len(self.modified_paths),
                "removed": len(self.removed_paths),
                "unchanged": len(self.unchanged_paths),
                "changed": len(self.changed_paths) + len(self.removed_paths),
            },
            "paths": {
                "added": list(self.added_paths),
                "modified": list(self.modified_paths),
                "removed": list(self.removed_paths),
                "unchanged": list(self.unchanged_paths),
                "changed": list(self.changed_paths),
            },
            "reuse": {
                "reused_elements": self.reused_elements,
                "reindexed_elements": self.reindexed_elements,
                "reused_changed_embeddings": self.reused_changed_embeddings,
            },
            "frontier": {
                "semantic_frontier_widened": self.semantic_frontier_widened,
                "api_frontier_changed": bool(self.api_frontier_changed_paths),
                "api_frontier_changed_paths": list(self.api_frontier_changed_paths),
                "package_scope_roots": list(self.package_scope_roots),
                "change_kinds": list(self.change_kinds),
                "interface_digest_changed_paths": list(
                    self.interface_digest_changed_paths
                ),
                "interface_digests": dict(self.interface_digests),
                "dependency_frontier": dict(self.dependency_frontier),
                "degraded": self.degraded,
                "degraded_reasons": list(self.degraded_reasons),
            },
        }

    def to_prefilter_payload(self) -> dict[str, Any]:
        payload = self.to_payload()
        counts = payload["counts"]
        paths = payload["paths"]
        reuse = payload["reuse"]
        frontier = payload["frontier"]
        return {
            "previous_snapshot_id": self.previous_snapshot_id,
            "previous_artifact_key": self.previous_artifact_key,
            "artifact_delta_mode": True,
            "added": counts["added"],
            "modified": counts["modified"],
            "removed": counts["removed"],
            "unchanged": counts["unchanged"],
            "added_paths": paths["added"],
            "modified_paths": paths["modified"],
            "removed_paths": paths["removed"],
            "unchanged_paths": paths["unchanged"],
            "changed_paths": paths["changed"],
            "reused_elements": reuse["reused_elements"],
            "reindexed_elements": reuse["reindexed_elements"],
            "reused_changed_embeddings": reuse["reused_changed_embeddings"],
            "semantic_frontier_widened": int(frontier["semantic_frontier_widened"]),
            "api_frontier_changed": int(frontier["api_frontier_changed"]),
            "api_frontier_changed_paths": frontier["api_frontier_changed_paths"],
            "package_scope_roots": frontier["package_scope_roots"],
            "change_kinds": frontier["change_kinds"],
            "interface_digest_changed_paths": frontier[
                "interface_digest_changed_paths"
            ],
            "interface_digests": frontier["interface_digests"],
            "dependency_frontier": frontier["dependency_frontier"],
            "degraded": frontier["degraded"],
            "degraded_reasons": frontier["degraded_reasons"],
            "plan_changes": payload,
        }


def diff_changed_files(
    old_snapshot: IRSnapshot, new_snapshot: IRSnapshot
) -> FileChangeSet:
    """Compare content identities per file path between two snapshots.

    Returns a FileChangeSet partitioning file paths into added, removed,
    modified, and unchanged buckets based on blob_oid/content_hash comparison.
    """
    old_identity_by_path: dict[str, str | None] = {}
    for doc in old_snapshot.documents:
        old_identity_by_path[doc.path] = doc.blob_oid or doc.content_hash

    new_identity_by_path: dict[str, str | None] = {}
    for doc in new_snapshot.documents:
        new_identity_by_path[doc.path] = doc.blob_oid or doc.content_hash

    old_paths = set(old_identity_by_path)
    new_paths = set(new_identity_by_path)

    added = sorted(new_paths - old_paths)
    removed = sorted(old_paths - new_paths)

    modified: list[str] = []
    unchanged: list[str] = []
    for path in sorted(old_paths & new_paths):
        old_identity = old_identity_by_path[path]
        new_identity = new_identity_by_path[path]
        if not old_identity or not new_identity or old_identity != new_identity:
            modified.append(path)
        else:
            unchanged.append(path)

    return FileChangeSet(
        added=added,
        removed=removed,
        modified=modified,
        unchanged=unchanged,
    )


def _index_units_by_path(
    units: list[IRCodeUnit],
) -> tuple[dict[str, list[IRCodeUnit]], dict[str, set[str]]]:
    """Build path-to-units and path-to-unit-ids indexes."""
    by_path: dict[str, list[IRCodeUnit]] = {}
    ids_by_path: dict[str, set[str]] = {}
    for unit in units:
        by_path.setdefault(unit.path, []).append(unit)
        ids_by_path.setdefault(unit.path, set()).add(unit.unit_id)
    return by_path, ids_by_path


def _unit_identity_key(unit: IRCodeUnit) -> str:
    stable_unit_id = (unit.metadata or {}).get("stable_unit_id")
    if stable_unit_id:
        return f"stable:{stable_unit_id}"
    if unit.kind == "file":
        return f"file:{unit.path}"
    return f"unit:{unit.unit_id}"


def _replacement_map(
    change_set: FileChangeSet,
    old_units_by_path: dict[str, list[IRCodeUnit]],
    new_units_by_path: dict[str, list[IRCodeUnit]],
) -> dict[str, str]:
    new_by_identity: dict[str, str] = {}
    for path in change_set.added + change_set.modified:
        for unit in new_units_by_path.get(path, []):
            new_by_identity[_unit_identity_key(unit)] = unit.unit_id

    replacements: dict[str, str] = {}
    for path in change_set.modified + change_set.removed:
        for unit in old_units_by_path.get(path, []):
            replacement = new_by_identity.get(_unit_identity_key(unit))
            if replacement:
                replacements[unit.unit_id] = replacement
    return replacements


def _preserve_source(source: str, preserve_sources: set[str] | None) -> bool:
    return preserve_sources is not None and source in preserve_sources


def _relation_sources(relation: IRRelation) -> set[str]:
    if relation.support_sources:
        return set(relation.support_sources)
    source = relation.source
    return {source} if source else set()


def _support_key(
    support: IRUnitSupport,
) -> tuple[str, str, str, str | None, str | None]:
    return (
        support.unit_id,
        support.source,
        support.support_kind,
        support.path,
        support.external_id,
    )


def _iter_merged_units(
    change_set: FileChangeSet,
    old_units_by_path: dict[str, list[IRCodeUnit]],
    new_units_by_path: dict[str, list[IRCodeUnit]],
) -> Iterator[IRCodeUnit]:
    for path in change_set.unchanged:
        yield from old_units_by_path.get(path, [])
    for path in change_set.added + change_set.modified:
        yield from new_units_by_path.get(path, [])


def _merged_units_len(
    change_set: FileChangeSet,
    old_units_by_path: dict[str, list[IRCodeUnit]],
    new_units_by_path: dict[str, list[IRCodeUnit]],
) -> int:
    return sum(
        len(old_units_by_path.get(path, [])) for path in change_set.unchanged
    ) + sum(
        len(new_units_by_path.get(path, []))
        for path in change_set.added + change_set.modified
    )


def _iter_merged_supports(
    old_supports: Sequence[IRUnitSupport],
    new_supports: Sequence[IRUnitSupport],
    tombstoned_ids: set[str],
    replacement_map: dict[str, str],
    preserve_sources_for_modified_paths: set[str] | None = None,
) -> Iterator[IRUnitSupport]:
    seen: set[tuple[str, str, str, str | None, str | None]] = set()
    for support in old_supports:
        if support.unit_id not in tombstoned_ids:
            seen.add(_support_key(support))
            yield support
            continue
        replacement_id = replacement_map.get(support.unit_id)
        if not replacement_id or not _preserve_source(
            support.source, preserve_sources_for_modified_paths
        ):
            continue
        metadata = dict(support.metadata or {})
        metadata["relinked_from_unit_id"] = support.unit_id
        metadata["relink_reason"] = "source_owned_incremental_preserve"
        relinked = dc_replace(
            support,
            unit_id=replacement_id,
            support_id=f"relink:{support.support_id}",
            metadata=metadata,
        )
        seen.add(_support_key(relinked))
        yield relinked

    for support in new_supports:
        key = _support_key(support)
        if key in seen:
            continue
        seen.add(key)
        yield support


def _relinked_relation(
    relation: IRRelation,
    *,
    src_unit_id: str,
    dst_unit_id: str,
    relink_reason: str,
) -> IRRelation:
    metadata = dict(relation.metadata or {})
    metadata["relinked_from_src_unit_id"] = relation.src_unit_id
    metadata["relinked_from_dst_unit_id"] = relation.dst_unit_id
    metadata["relink_reason"] = relink_reason
    digest = hashlib.sha256(
        (
            f"{relation.relation_type}\0"
            f"{src_unit_id}\0"
            f"{dst_unit_id}\0"
            f"{relation.relation_id}\0"
            f"{relink_reason}"
        ).encode()
    ).hexdigest()[:24]
    return IRRelation(
        relation_id=f"relink:{digest}",
        src_unit_id=src_unit_id,
        dst_unit_id=dst_unit_id,
        relation_type=relation.relation_type,
        resolution_state=relation.resolution_state,
        support_sources=set(relation.support_sources),
        support_ids=list(relation.support_ids),
        pending_capabilities=set(relation.pending_capabilities),
        metadata=metadata,
    )


def _iter_merged_relations(
    old_relations: Sequence[IRRelation],
    new_relations: Sequence[IRRelation],
    tombstoned_ids: set[str],
    new_unit_ids: set[str],
    replacement_map: dict[str, str],
    preserve_sources_for_modified_paths: set[str] | None = None,
) -> Iterator[IRRelation]:
    seen: set[tuple[str, str, str]] = set()

    def unique(relation: IRRelation) -> IRRelation | None:
        key = (relation.src_unit_id, relation.dst_unit_id, relation.relation_type)
        if key in seen:
            return None
        seen.add(key)
        return relation

    for relation in old_relations:
        src_tombstoned = relation.src_unit_id in tombstoned_ids
        dst_tombstoned = relation.dst_unit_id in tombstoned_ids
        if src_tombstoned:
            replacement_src = replacement_map.get(relation.src_unit_id)
            relation_sources = _relation_sources(relation)
            preserve_relation = bool(relation_sources) and all(
                _preserve_source(source, preserve_sources_for_modified_paths)
                for source in relation_sources
            )
            if preserve_relation and replacement_src:
                replacement_dst = (
                    replacement_map.get(relation.dst_unit_id)
                    if dst_tombstoned
                    else relation.dst_unit_id
                )
                if replacement_dst:
                    relinked = _relinked_relation(
                        relation,
                        src_unit_id=replacement_src,
                        dst_unit_id=replacement_dst,
                        relink_reason="source_owned_incremental_preserve",
                    )
                    if (unique_relation := unique(relinked)) is not None:
                        yield unique_relation
            continue
        if dst_tombstoned:
            replacement_dst = replacement_map.get(relation.dst_unit_id)
            if replacement_dst is None:
                continue
            if replacement_dst != relation.dst_unit_id:
                relinked = _relinked_relation(
                    relation,
                    src_unit_id=relation.src_unit_id,
                    dst_unit_id=replacement_dst,
                    relink_reason="stable_destination_incremental_relink",
                )
                if (unique_relation := unique(relinked)) is not None:
                    yield unique_relation
                continue
        if (unique_relation := unique(relation)) is not None:
            yield unique_relation

    for relation in new_relations:
        src_is_new = (
            relation.src_unit_id in tombstoned_ids
            or relation.src_unit_id in new_unit_ids
        )
        if not src_is_new:
            continue
        dst_tombstoned_unreplaced = (
            relation.dst_unit_id in tombstoned_ids
            and relation.dst_unit_id not in new_unit_ids
        )
        if dst_tombstoned_unreplaced:
            continue
        if (unique_relation := unique(relation)) is not None:
            yield unique_relation


def _iter_merged_embeddings(
    old_embeddings: Sequence[IRUnitEmbedding],
    new_embeddings: Sequence[IRUnitEmbedding],
    tombstoned_ids: set[str],
) -> Iterator[IRUnitEmbedding]:
    seen_ids: set[str] = set()

    for embedding in old_embeddings:
        if embedding.unit_id not in tombstoned_ids:
            seen_ids.add(embedding.unit_id)
            yield embedding

    for embedding in new_embeddings:
        if embedding.unit_id in seen_ids:
            continue
        seen_ids.add(embedding.unit_id)
        yield embedding


def apply_incremental_update(
    old_snapshot: IRSnapshot,
    new_extraction: IRSnapshot,
    change_set: FileChangeSet,
    preserve_sources_for_modified_paths: set[str] | None = None,
) -> IRSnapshot:
    """Produce a merged snapshot from an old snapshot and fresh extraction.

    Algorithm:
    1. Keep all units/relations/embeddings from old_snapshot whose file path
       is NOT in (added + modified + removed).
    2. Add all units/relations/embeddings from new_extraction whose file path
       IS in (added + modified) -- these are freshly extracted.
    3. Preserve embeddings for unchanged units.
    4. Drop relations whose source unit belongs to a changed file.

    Args:
        old_snapshot: Previously persisted snapshot.
        new_extraction: Freshly extracted snapshot (full extraction of repo).
        change_set: File diff from diff_changed_files().
        preserve_sources_for_modified_paths: Support sources, such as {"scip"},
            that may be relinked across modified files when the caller has proven
            the source-owned semantic surface is unchanged.

    Returns:
        New IRSnapshot with updated content merged in.
    """
    changed_paths = set(change_set.added + change_set.modified + change_set.removed)

    old_units_by_path, old_ids_by_path = _index_units_by_path(old_snapshot.units)
    new_units_by_path, _ = _index_units_by_path(new_extraction.units)

    # Tombstoned IDs: old units from changed paths.
    tombstoned_ids: set[str] = set()
    for path in changed_paths:
        tombstoned_ids.update(old_ids_by_path.get(path, set()))

    # New unit IDs from freshly extracted files.
    new_unit_ids: set[str] = set()
    for path in change_set.added + change_set.modified:
        for unit in new_units_by_path.get(path, []):
            new_unit_ids.add(unit.unit_id)

    # Merge each component.
    replacements = _replacement_map(change_set, old_units_by_path, new_units_by_path)
    snapshot = IRSnapshot(
        repo_name=new_extraction.repo_name,
        snapshot_id=new_extraction.snapshot_id,
        branch=new_extraction.branch,
        commit_id=new_extraction.commit_id,
        tree_id=new_extraction.tree_id,
        metadata=new_extraction.metadata,
    )
    snapshot.units = _LazyIRSequence(
        lambda: _iter_merged_units(change_set, old_units_by_path, new_units_by_path),
        len_factory=lambda: _merged_units_len(
            change_set, old_units_by_path, new_units_by_path
        ),
    )
    snapshot.supports = _LazyIRSequence(
        lambda: _iter_merged_supports(
            old_snapshot.supports,
            new_extraction.supports,
            tombstoned_ids,
            replacements,
            preserve_sources_for_modified_paths,
        )
    )
    snapshot.relations = _LazyIRSequence(
        lambda: _iter_merged_relations(
            old_snapshot.relations,
            new_extraction.relations,
            tombstoned_ids,
            new_unit_ids,
            replacements,
            preserve_sources_for_modified_paths,
        )
    )
    snapshot.embeddings = _LazyIRSequence(
        lambda: _iter_merged_embeddings(
            old_snapshot.embeddings, new_extraction.embeddings, tombstoned_ids
        )
    )
    return snapshot
