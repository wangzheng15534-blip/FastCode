"""
Validation rules for canonical unit-grounded IR snapshots.
"""

from __future__ import annotations

from .semantic_ir import IRSnapshot


def validate_snapshot(snapshot: IRSnapshot) -> list[str]:
    errors: list[str] = []

    unit_ids = [unit.unit_id for unit in snapshot.units]
    support_ids = [support.support_id for support in snapshot.supports]
    relation_ids = [relation.relation_id for relation in snapshot.relations]
    embedding_ids = [embedding.embedding_id for embedding in snapshot.embeddings]

    if len(unit_ids) != len(set(unit_ids)):
        errors.append("duplicate unit IDs detected")
    if len(support_ids) != len(set(support_ids)):
        errors.append("duplicate support IDs detected")
    if len(relation_ids) != len(set(relation_ids)):
        errors.append("duplicate relation IDs detected")
    if len(embedding_ids) != len(set(embedding_ids)):
        errors.append("duplicate embedding IDs detected")

    file_paths = [unit.path for unit in snapshot.units if unit.kind == "file"]
    if len(file_paths) != len(set(file_paths)):
        dupes = sorted(path for path in set(file_paths) if file_paths.count(path) > 1)
        errors.append(f"duplicate file paths detected: {dupes}")

    known_units = set(unit_ids)
    anchor_owners: dict[str, str] = {}
    for unit in snapshot.units:
        if not unit.source_set:
            errors.append(f"unit provenance missing: {unit.unit_id}")
        if unit.parent_unit_id and unit.parent_unit_id not in known_units:
            errors.append(
                f"unit parent not found: {unit.unit_id} -> {unit.parent_unit_id}"
            )
        if unit.primary_anchor_symbol_id:
            existing = anchor_owners.get(unit.primary_anchor_symbol_id)
            if existing and existing != unit.unit_id:
                errors.append(
                    f"primary anchor assigned to multiple units: {unit.primary_anchor_symbol_id} -> {existing}, {unit.unit_id}"
                )
            anchor_owners[unit.primary_anchor_symbol_id] = unit.unit_id

    for support in snapshot.supports:
        if support.unit_id not in known_units:
            errors.append(
                f"support references missing unit_id: {support.support_id} -> {support.unit_id}"
            )
        if not support.source:
            errors.append(f"support source missing: {support.support_id}")

    known_supports = set(support_ids)
    for relation in snapshot.relations:
        if relation.src_unit_id not in known_units:
            errors.append(
                f"relation src not found: {relation.relation_id} -> {relation.src_unit_id}"
            )
        if relation.dst_unit_id not in known_units:
            errors.append(
                f"relation dst not found: {relation.relation_id} -> {relation.dst_unit_id}"
            )
        if not relation.relation_type:
            errors.append(f"relation type missing: {relation.relation_id}")
        if not relation.support_sources and not (relation.metadata or {}).get("source"):
            errors.append(f"relation source missing: {relation.relation_id}")
        for support_id in relation.support_ids:
            if support_id not in known_supports:
                errors.append(
                    f"relation support not found: {relation.relation_id} -> {support_id}"
                )

    for embedding in snapshot.embeddings:
        if embedding.unit_id not in known_units:
            errors.append(
                f"embedding references missing unit_id: {embedding.embedding_id} -> {embedding.unit_id}"
            )
        if not embedding.source:
            errors.append(f"embedding source missing: {embedding.embedding_id}")

    return errors
