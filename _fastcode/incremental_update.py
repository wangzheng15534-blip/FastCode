"""
Incremental snapshot update via blob_oid diffing.

Compares file-level blob_oids between two snapshots to detect changes,
then produces a new snapshot that preserves unchanged units, relations,
and embeddings while replacing changed-file content with fresh extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .semantic_ir import (
    IRCodeUnit,
    IRRelation,
    IRSnapshot,
    IRUnitEmbedding,
    IRUnitSupport,
)


@dataclass
class FileChangeSet:
    """Diff result comparing two snapshots' file-level blob_oids."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


def diff_changed_files(
    old_snapshot: IRSnapshot, new_snapshot: IRSnapshot
) -> FileChangeSet:
    """Compare blob_oids per file path between two snapshots.

    Returns a FileChangeSet partitioning file paths into added, removed,
    modified, and unchanged buckets based on blob_oid comparison.
    """
    old_blob_by_path: dict[str, str | None] = {}
    for doc in old_snapshot.documents:
        old_blob_by_path[doc.path] = doc.blob_oid

    new_blob_by_path: dict[str, str | None] = {}
    for doc in new_snapshot.documents:
        new_blob_by_path[doc.path] = doc.blob_oid

    old_paths = set(old_blob_by_path)
    new_paths = set(new_blob_by_path)

    added = sorted(new_paths - old_paths)
    removed = sorted(old_paths - new_paths)

    modified: list[str] = []
    unchanged: list[str] = []
    for path in sorted(old_paths & new_paths):
        if old_blob_by_path[path] != new_blob_by_path[path]:
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


def _merge_units(
    change_set: FileChangeSet,
    old_units_by_path: dict[str, list[IRCodeUnit]],
    new_units_by_path: dict[str, list[IRCodeUnit]],
) -> tuple[list[IRCodeUnit], set[str], set[str]]:
    """Merge units: keep unchanged old, add fresh from changed/added paths.

    Returns (merged_units, merged_unit_ids, tombstoned_unit_ids).
    """
    merged: list[IRCodeUnit] = []
    merged_ids: set[str] = set()

    for path in change_set.unchanged:
        for unit in old_units_by_path.get(path, []):
            merged.append(unit)
            merged_ids.add(unit.unit_id)

    for path in change_set.added + change_set.modified:
        for unit in new_units_by_path.get(path, []):
            merged.append(unit)
            merged_ids.add(unit.unit_id)

    return merged, merged_ids


def _merge_supports(
    old_supports: list[IRUnitSupport],
    new_supports: list[IRUnitSupport],
    tombstoned_ids: set[str],
) -> list[IRUnitSupport]:
    """Merge supports: drop old tombstoned, replace with new."""
    merged: list[IRUnitSupport] = []
    for support in old_supports:
        if support.unit_id not in tombstoned_ids:
            merged.append(support)
    for support in new_supports:
        if support.unit_id in tombstoned_ids or support.unit_id not in {
            s.unit_id for s in merged
        }:
            merged.append(support)
    return merged


def _merge_relations(
    old_relations: list[IRRelation],
    new_relations: list[IRRelation],
    tombstoned_ids: set[str],
    new_unit_ids: set[str],
) -> list[IRRelation]:
    """Merge relations: drop tombstoned, add new from changed files, dedup."""
    merged: list[IRRelation] = []
    for relation in old_relations:
        if relation.src_unit_id in tombstoned_ids:
            continue
        if relation.dst_unit_id in tombstoned_ids:
            continue
        merged.append(relation)

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
        merged.append(relation)

    # Deduplicate by (src, dst, type).
    seen: set[tuple[str, str, str]] = set()
    deduped: list[IRRelation] = []
    for relation in merged:
        key = (relation.src_unit_id, relation.dst_unit_id, relation.relation_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(relation)
    return deduped


def _merge_embeddings(
    old_embeddings: list[IRUnitEmbedding],
    new_embeddings: list[IRUnitEmbedding],
    tombstoned_ids: set[str],
) -> list[IRUnitEmbedding]:
    """Merge embeddings: preserve unchanged, add new for changed files."""
    merged: list[IRUnitEmbedding] = []
    seen_ids: set[str] = set()

    for embedding in old_embeddings:
        if embedding.unit_id not in tombstoned_ids:
            merged.append(embedding)
            seen_ids.add(embedding.unit_id)

    for embedding in new_embeddings:
        if embedding.unit_id not in seen_ids:
            merged.append(embedding)
            seen_ids.add(embedding.unit_id)

    return merged


def apply_incremental_update(
    old_snapshot: IRSnapshot,
    new_extraction: IRSnapshot,
    change_set: FileChangeSet,
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

    Returns:
        New IRSnapshot with updated content merged in.
    """
    changed_paths = set(change_set.added + change_set.modified + change_set.removed)

    old_units_by_path, old_ids_by_path = _index_units_by_path(old_snapshot.units)
    new_units_by_path, _ = _index_units_by_path(new_extraction.units)

    # Units: keep unchanged, replace changed.
    merged_units, _merged_unit_ids = _merge_units(
        change_set, old_units_by_path, new_units_by_path
    )

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
    merged_supports = _merge_supports(
        old_snapshot.supports, new_extraction.supports, tombstoned_ids
    )
    merged_relations = _merge_relations(
        old_snapshot.relations, new_extraction.relations, tombstoned_ids, new_unit_ids
    )
    merged_embeddings = _merge_embeddings(
        old_snapshot.embeddings, new_extraction.embeddings, tombstoned_ids
    )

    return IRSnapshot(
        repo_name=new_extraction.repo_name,
        snapshot_id=new_extraction.snapshot_id,
        branch=new_extraction.branch,
        commit_id=new_extraction.commit_id,
        tree_id=new_extraction.tree_id,
        units=merged_units,
        supports=merged_supports,
        relations=merged_relations,
        embeddings=merged_embeddings,
        metadata=new_extraction.metadata,
    )
