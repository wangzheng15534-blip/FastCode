"""
Merge AST and SCIP IR snapshots.
"""

from __future__ import annotations

import hashlib
import json

from .semantic_ir import IRAttachment, IREdge, IROccurrence, IRSnapshot, IRSymbol


def _symbol_key(symbol: IRSymbol) -> tuple[str, str, str, int | None]:
    return (
        symbol.path or "",
        symbol.display_name or "",
        symbol.kind or "",
        symbol.start_line,
    )


def _attachment_id(
    target_type: str,
    target_id: str,
    attachment_type: str,
    source: str,
    confidence: str,
    payload: dict,
    metadata: dict,
) -> str:
    payload_key = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    metadata_key = json.dumps(metadata, sort_keys=True, ensure_ascii=False)
    digest = hashlib.blake2b(
        f"{target_type}:{target_id}:{attachment_type}:{source}:{confidence}:{payload_key}:{metadata_key}".encode(),
        digest_size=12,
    ).hexdigest()[:24]
    return f"att:{digest}"


def _attachment_key(attachment: IRAttachment) -> tuple[str, str, str, str, str, str, str]:
    payload_key = json.dumps(attachment.payload, sort_keys=True, ensure_ascii=False)
    metadata_key = json.dumps(attachment.metadata, sort_keys=True, ensure_ascii=False)
    return (
        attachment.target_type,
        attachment.target_id,
        attachment.attachment_type,
        attachment.source,
        attachment.confidence,
        payload_key,
        metadata_key,
    )


def merge_ir(ast_snapshot: IRSnapshot, scip_snapshot: IRSnapshot | None) -> IRSnapshot:
    """
    Merge snapshots. SCIP symbols win where keys overlap; AST fills gaps.
    """
    if scip_snapshot is None:
        return ast_snapshot

    merged_docs = {}
    for d in ast_snapshot.documents + scip_snapshot.documents:
        if d.doc_id in merged_docs:
            merged_docs[d.doc_id].source_set.update(d.source_set)
        else:
            merged_docs[d.doc_id] = d

    canonical_symbols: dict[str, IRSymbol] = {}
    ast_to_canonical: dict[str, str] = {}
    by_key: dict[tuple[str, str, str, int | None], str] = {}

    for s in scip_snapshot.symbols:
        canonical_symbols[s.symbol_id] = s
        by_key[_symbol_key(s)] = s.symbol_id

    for ast in ast_snapshot.symbols:
        key = _symbol_key(ast)
        if key in by_key:
            canonical_id = by_key[key]
            ast_to_canonical[ast.symbol_id] = canonical_id
            canonical_symbols[canonical_id].source_set.update(ast.source_set)
            canonical_symbols[canonical_id].metadata.setdefault("aliases", []).append(ast.symbol_id)
        else:
            canonical_symbols[ast.symbol_id] = ast
            by_key[key] = ast.symbol_id

    # Merge occurrences — deduplicate by (symbol_id, doc_id, role, range).
    # SCIP wins when both sources produce the same occurrence (Rule D).
    occ_seen: dict[tuple, IROccurrence] = {}
    for occ in scip_snapshot.occurrences + ast_snapshot.occurrences:
        symbol_id = ast_to_canonical.get(occ.symbol_id, occ.symbol_id)
        key = (symbol_id, occ.doc_id, occ.role, occ.start_line or 0, occ.start_col or 0, occ.end_line or 0, occ.end_col or 0)
        if key not in occ_seen:
            occ_seen[key] = IROccurrence(
                occurrence_id=occ.occurrence_id,
                symbol_id=symbol_id,
                doc_id=occ.doc_id,
                role=occ.role,
                start_line=occ.start_line,
                start_col=occ.start_col,
                end_line=occ.end_line,
                end_col=occ.end_col,
                source=occ.source,
                metadata=occ.metadata,
            )
    merged_occurrences = list(occ_seen.values())

    edge_seen: set[tuple[str, str, str]] = set()
    merged_edges = []
    for edge in ast_snapshot.edges + scip_snapshot.edges:
        src_id = ast_to_canonical.get(edge.src_id, edge.src_id)
        dst_id = ast_to_canonical.get(edge.dst_id, edge.dst_id)
        edge_key = (src_id, dst_id, edge.edge_type)
        if edge_key in edge_seen:
            continue
        edge_seen.add(edge_key)
        merged_edges.append(
            IREdge(
                edge_id=edge.edge_id,
                src_id=src_id,
                dst_id=dst_id,
                edge_type=edge.edge_type,
                source=edge.source,
                confidence=edge.confidence,
                doc_id=edge.doc_id,
                metadata=edge.metadata,
            )
        )

    attachment_seen: set[tuple[str, str, str, str, str, str, str]] = set()
    merged_attachments = []
    for attachment in scip_snapshot.attachments + ast_snapshot.attachments:
        target_id = attachment.target_id
        if attachment.target_type == "symbol":
            target_id = ast_to_canonical.get(target_id, target_id)
        materialized = IRAttachment(
            attachment_id=_attachment_id(
                attachment.target_type,
                target_id,
                attachment.attachment_type,
                attachment.source,
                attachment.confidence,
                attachment.payload,
                attachment.metadata,
            ),
            target_id=target_id,
            target_type=attachment.target_type,
            attachment_type=attachment.attachment_type,
            source=attachment.source,
            confidence=attachment.confidence,
            payload=attachment.payload,
            metadata=attachment.metadata,
        )
        key = _attachment_key(materialized)
        if key in attachment_seen:
            continue
        attachment_seen.add(key)
        merged_attachments.append(materialized)

    return IRSnapshot(
        repo_name=ast_snapshot.repo_name,
        snapshot_id=ast_snapshot.snapshot_id,
        branch=ast_snapshot.branch,
        commit_id=ast_snapshot.commit_id,
        tree_id=ast_snapshot.tree_id,
        documents=list(merged_docs.values()),
        symbols=list(canonical_symbols.values()),
        occurrences=merged_occurrences,
        edges=merged_edges,
        attachments=merged_attachments,
        metadata={
            "source_modes": sorted(
                set(ast_snapshot.metadata.get("source_modes", []))
                | set(scip_snapshot.metadata.get("source_modes", []))
            )
        },
    )
