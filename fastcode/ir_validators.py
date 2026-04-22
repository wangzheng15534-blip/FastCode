"""
Validation rules for merged IR snapshots.
"""

from __future__ import annotations

from .semantic_ir import IRSnapshot


def validate_snapshot(snapshot: IRSnapshot) -> list[str]:
    errors: list[str] = []

    doc_ids = {d.doc_id for d in snapshot.documents}
    sym_ids = {s.symbol_id for s in snapshot.symbols}
    attachment_ids = [a.attachment_id for a in snapshot.attachments]

    if not snapshot.documents:
        errors.append("snapshot must contain at least one document")
    if not snapshot.symbols:
        errors.append("snapshot must contain at least one symbol")

    if len(doc_ids) != len(snapshot.documents):
        errors.append("duplicate document IDs detected")
    doc_paths = [d.path for d in snapshot.documents]
    if len(doc_paths) != len(set(doc_paths)):
        dupes = [p for p in set(doc_paths) if doc_paths.count(p) > 1]
        errors.append(f"duplicate document paths detected: {dupes}")
    if len(sym_ids) != len(snapshot.symbols):
        errors.append("duplicate symbol IDs detected")
    if len(attachment_ids) != len(set(attachment_ids)):
        errors.append("duplicate attachment IDs detected")

    for occ in snapshot.occurrences:
        if occ.doc_id not in doc_ids:
            errors.append(f"occurrence references missing doc_id: {occ.doc_id}")
        if occ.symbol_id not in sym_ids:
            errors.append(f"occurrence references missing symbol_id: {occ.symbol_id}")

    valid_nodes = doc_ids | sym_ids
    for edge in snapshot.edges:
        if edge.src_id not in valid_nodes:
            errors.append(f"edge src not found: {edge.edge_id} -> {edge.src_id}")
        if edge.dst_id not in valid_nodes:
            errors.append(f"edge dst not found: {edge.edge_id} -> {edge.dst_id}")
        if not edge.source:
            errors.append(f"edge source missing: {edge.edge_id}")
        if not edge.confidence:
            errors.append(f"edge confidence missing: {edge.edge_id}")

    for sym in snapshot.symbols:
        src = (sym.metadata or {}).get("source")
        if not src and not sym.source_set:
            errors.append(f"symbol provenance missing: {sym.symbol_id}")

    valid_attachment_targets = {
        "document": doc_ids,
        "symbol": sym_ids,
        "snapshot": {snapshot.snapshot_id},
    }
    for attachment in snapshot.attachments:
        valid_targets = valid_attachment_targets.get(attachment.target_type)
        if valid_targets is None:
            errors.append(
                f"attachment target_type unsupported: {attachment.attachment_id} -> {attachment.target_type}"
            )
            continue
        if attachment.target_id not in valid_targets:
            errors.append(
                f"attachment target not found: {attachment.attachment_id} -> {attachment.target_type}:{attachment.target_id}"
            )
        if not attachment.source:
            errors.append(f"attachment source missing: {attachment.attachment_id}")
        if not attachment.confidence:
            errors.append(f"attachment confidence missing: {attachment.attachment_id}")
        if not attachment.attachment_type:
            errors.append(f"attachment type missing: {attachment.attachment_id}")

    return errors
