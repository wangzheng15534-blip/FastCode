"""
Validation rules for merged IR snapshots.
"""

from __future__ import annotations

from typing import List

from .semantic_ir import IRSnapshot


def validate_snapshot(snapshot: IRSnapshot) -> List[str]:
    errors: List[str] = []

    doc_ids = {d.doc_id for d in snapshot.documents}
    sym_ids = {s.symbol_id for s in snapshot.symbols}

    if not snapshot.documents:
        errors.append("snapshot must contain at least one document")
    if not snapshot.symbols:
        errors.append("snapshot must contain at least one symbol")

    if len(doc_ids) != len(snapshot.documents):
        errors.append("duplicate document IDs detected")
    if len(sym_ids) != len(snapshot.symbols):
        errors.append("duplicate symbol IDs detected")

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

    return errors
