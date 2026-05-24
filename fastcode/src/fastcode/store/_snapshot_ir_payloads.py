"""Private IR snapshot payload codecs for snapshot storage."""
# pyright: reportUnusedFunction=false

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from ..ir.types import (
    IRAttachment,
    IRCodeUnit,
    IRDocument,
    IREdge,
    IROccurrence,
    IRRelation,
    IRSnapshot,
    IRSymbol,
    IRUnitEmbedding,
    IRUnitSupport,
)
from ..utils.json import safe_jsonable
from ..utils.materialization import (
    BOUNDARY_JSON_DECODE,
    increment_materialization_boundary,
)


def _source_set_payload(values: set[str]) -> list[str]:
    return sorted(value for value in values if value)


def _document_payload(doc: IRDocument) -> dict[str, Any]:
    return {
        "doc_id": doc.doc_id,
        "path": doc.path,
        "language": doc.language,
        "blob_oid": doc.blob_oid,
        "content_hash": doc.content_hash,
        "source_set": _source_set_payload(doc.source_set),
    }


def _symbol_payload(sym: IRSymbol) -> dict[str, Any]:
    return {
        "symbol_id": sym.symbol_id,
        "external_symbol_id": sym.external_symbol_id,
        "path": sym.path,
        "display_name": sym.display_name,
        "kind": sym.kind,
        "language": sym.language,
        "qualified_name": sym.qualified_name,
        "signature": sym.signature,
        "start_line": sym.start_line,
        "start_col": sym.start_col,
        "end_line": sym.end_line,
        "end_col": sym.end_col,
        "source_priority": sym.source_priority,
        "source_set": _source_set_payload(sym.source_set),
        "metadata": dict(sym.metadata) if sym.metadata else {},
    }


def _occurrence_payload(occ: IROccurrence) -> dict[str, Any]:
    return {
        "occurrence_id": occ.occurrence_id,
        "symbol_id": occ.symbol_id,
        "doc_id": occ.doc_id,
        "role": occ.role,
        "start_line": occ.start_line,
        "start_col": occ.start_col,
        "end_line": occ.end_line,
        "end_col": occ.end_col,
        "source": occ.source,
        "metadata": dict(occ.metadata) if occ.metadata else {},
    }


def _edge_payload(edge: IREdge) -> dict[str, Any]:
    return {
        "edge_id": edge.edge_id,
        "src_id": edge.src_id,
        "dst_id": edge.dst_id,
        "edge_type": edge.edge_type,
        "source": edge.source,
        "confidence": edge.confidence,
        "doc_id": edge.doc_id,
        "metadata": dict(edge.metadata) if edge.metadata else {},
    }


def _attachment_payload(attachment: IRAttachment) -> dict[str, Any]:
    return {
        "attachment_id": attachment.attachment_id,
        "target_id": attachment.target_id,
        "target_type": attachment.target_type,
        "attachment_type": attachment.attachment_type,
        "source": attachment.source,
        "confidence": attachment.confidence,
        "payload": dict(attachment.payload) if attachment.payload else {},
        "metadata": dict(attachment.metadata) if attachment.metadata else {},
    }


def _payload_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _required_text(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if value is None:
        raise KeyError(field_name)
    return str(value)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, *, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sequence_items(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return value
    if isinstance(value, (set, frozenset)):
        return tuple(value)
    return ()


def _string_list_payload(values: Any) -> list[str]:
    return [str(value) for value in _sequence_items(values) if value]


def _string_set_payload(values: Any) -> set[str]:
    return {str(value) for value in _sequence_items(values) if value}


def _json_mapping_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    normalized = {str(key): nested for key, nested in value.items()}
    jsonable = safe_jsonable(normalized)
    return jsonable if isinstance(jsonable, dict) else {}


def _json_list_payload(value: Any) -> list[Any] | None:
    if value is None:
        return None
    jsonable = safe_jsonable(value)
    return jsonable if isinstance(jsonable, list) else None


def _code_unit_payload(unit: IRCodeUnit) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "kind": unit.kind,
        "path": unit.path,
        "language": unit.language,
        "display_name": unit.display_name,
        "qualified_name": unit.qualified_name,
        "signature": unit.signature,
        "docstring": unit.docstring,
        "summary": unit.summary,
        "start_line": unit.start_line,
        "start_col": unit.start_col,
        "end_line": unit.end_line,
        "end_col": unit.end_col,
        "parent_unit_id": unit.parent_unit_id,
        "primary_anchor_symbol_id": unit.primary_anchor_symbol_id,
        "anchor_symbol_ids": _string_list_payload(unit.anchor_symbol_ids),
        "candidate_anchor_symbol_ids": _string_list_payload(
            unit.candidate_anchor_symbol_ids
        ),
        "anchor_coverage": float(unit.anchor_coverage),
        "source_set": _source_set_payload(unit.source_set),
        "metadata": _json_mapping_payload(unit.metadata),
    }


def _unit_support_payload(support: IRUnitSupport) -> dict[str, Any]:
    return {
        "support_id": support.support_id,
        "unit_id": support.unit_id,
        "source": support.source,
        "support_kind": support.support_kind,
        "external_id": support.external_id,
        "role": support.role,
        "path": support.path,
        "display_name": support.display_name,
        "qualified_name": support.qualified_name,
        "signature": support.signature,
        "enclosing_external_id": support.enclosing_external_id,
        "start_line": support.start_line,
        "start_col": support.start_col,
        "end_line": support.end_line,
        "end_col": support.end_col,
        "metadata": _json_mapping_payload(support.metadata),
    }


def _relation_payload(relation: IRRelation) -> dict[str, Any]:
    return {
        "relation_id": relation.relation_id,
        "src_unit_id": relation.src_unit_id,
        "dst_unit_id": relation.dst_unit_id,
        "relation_type": relation.relation_type,
        "resolution_state": relation.resolution_state,
        "support_sources": _source_set_payload(relation.support_sources),
        "support_ids": _string_list_payload(relation.support_ids),
        "pending_capabilities": _source_set_payload(relation.pending_capabilities),
        "metadata": _json_mapping_payload(relation.metadata),
    }


def _embedding_payload(embedding: IRUnitEmbedding) -> dict[str, Any]:
    return {
        "embedding_id": embedding.embedding_id,
        "unit_id": embedding.unit_id,
        "source": embedding.source,
        "vector": _json_list_payload(embedding.vector),
        "embedding_text": embedding.embedding_text,
        "model_id": embedding.model_id,
        "metadata": _json_mapping_payload(embedding.metadata),
    }


def _snapshot_file_payload(snapshot: IRSnapshot) -> dict[str, Any]:
    return {
        "schema_version": "ir.v2",
        "repo_name": snapshot.repo_name,
        "snapshot_id": snapshot.snapshot_id,
        "branch": snapshot.branch,
        "commit_id": snapshot.commit_id,
        "tree_id": snapshot.tree_id,
        "units": [_code_unit_payload(unit) for unit in snapshot.units],
        "supports": [_unit_support_payload(support) for support in snapshot.supports],
        "relations": [_relation_payload(relation) for relation in snapshot.relations],
        "embeddings": [
            _embedding_payload(embedding) for embedding in snapshot.embeddings
        ],
        "metadata": _json_mapping_payload(snapshot.metadata),
    }


def _embedding_payload_without_vector(
    embedding: IRUnitEmbedding,
    *,
    vector_ref: str | None,
) -> dict[str, Any]:
    payload = _embedding_payload(embedding)
    payload["vector"] = None
    if vector_ref is not None:
        metadata = _json_mapping_payload(payload.get("metadata"))
        metadata["vector_ref"] = vector_ref
        payload["metadata"] = metadata
    return payload


def _payload_with_sequence(
    payload: dict[str, Any],
    sequence_no: int,
) -> dict[str, Any]:
    row = dict(payload)
    row["_sequence_no"] = sequence_no
    return row


def _ordered_sharded_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sequence_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, row = item
        raw_sequence = row.get("_sequence_no")
        return (
            (int(raw_sequence), index)
            if isinstance(raw_sequence, int)
            else (index, index)
        )

    return [row for _index, row in sorted(enumerate(rows), key=sequence_key)]


def _code_unit_from_payload(data: Any) -> IRCodeUnit:
    payload = _payload_mapping(data)
    return IRCodeUnit(
        unit_id=_required_text(payload, "unit_id"),
        kind=_required_text(payload, "kind"),
        path=_required_text(payload, "path"),
        language=_required_text(payload, "language"),
        display_name=_required_text(payload, "display_name"),
        qualified_name=_string_or_none(payload.get("qualified_name")),
        signature=_string_or_none(payload.get("signature")),
        docstring=_string_or_none(payload.get("docstring")),
        summary=_string_or_none(payload.get("summary")),
        start_line=_int_or_none(payload.get("start_line")),
        start_col=_int_or_none(payload.get("start_col")),
        end_line=_int_or_none(payload.get("end_line")),
        end_col=_int_or_none(payload.get("end_col")),
        parent_unit_id=_string_or_none(payload.get("parent_unit_id")),
        primary_anchor_symbol_id=_string_or_none(
            payload.get("primary_anchor_symbol_id")
        ),
        anchor_symbol_ids=_string_list_payload(payload.get("anchor_symbol_ids")),
        candidate_anchor_symbol_ids=_string_list_payload(
            payload.get("candidate_anchor_symbol_ids")
        ),
        anchor_coverage=_float_or_default(payload.get("anchor_coverage")),
        source_set=_string_set_payload(payload.get("source_set")),
        metadata=_json_mapping_payload(payload.get("metadata")),
    )


def _unit_support_from_payload(data: Any) -> IRUnitSupport:
    payload = _payload_mapping(data)
    return IRUnitSupport(
        support_id=_required_text(payload, "support_id"),
        unit_id=_required_text(payload, "unit_id"),
        source=_required_text(payload, "source"),
        support_kind=_required_text(payload, "support_kind"),
        external_id=_string_or_none(payload.get("external_id")),
        role=_string_or_none(payload.get("role")),
        path=_string_or_none(payload.get("path")),
        display_name=_string_or_none(payload.get("display_name")),
        qualified_name=_string_or_none(payload.get("qualified_name")),
        signature=_string_or_none(payload.get("signature")),
        enclosing_external_id=_string_or_none(payload.get("enclosing_external_id")),
        start_line=_int_or_none(payload.get("start_line")),
        start_col=_int_or_none(payload.get("start_col")),
        end_line=_int_or_none(payload.get("end_line")),
        end_col=_int_or_none(payload.get("end_col")),
        metadata=_json_mapping_payload(payload.get("metadata")),
    )


def _relation_from_payload(data: Any) -> IRRelation:
    payload = _payload_mapping(data)
    return IRRelation(
        relation_id=_required_text(payload, "relation_id"),
        src_unit_id=_required_text(payload, "src_unit_id"),
        dst_unit_id=_required_text(payload, "dst_unit_id"),
        relation_type=_required_text(payload, "relation_type"),
        resolution_state=_required_text(payload, "resolution_state"),
        support_sources=_string_set_payload(payload.get("support_sources")),
        support_ids=_string_list_payload(payload.get("support_ids")),
        pending_capabilities=_string_set_payload(payload.get("pending_capabilities")),
        metadata=_json_mapping_payload(payload.get("metadata")),
    )


def _embedding_from_payload(data: Any) -> IRUnitEmbedding:
    payload = _payload_mapping(data)
    return IRUnitEmbedding(
        embedding_id=_required_text(payload, "embedding_id"),
        unit_id=_required_text(payload, "unit_id"),
        source=_required_text(payload, "source"),
        vector=_json_list_payload(payload.get("vector")),
        embedding_text=_string_or_none(payload.get("embedding_text")),
        model_id=_string_or_none(payload.get("model_id")),
        metadata=_json_mapping_payload(payload.get("metadata")),
    )


def _document_from_payload(data: Any) -> IRDocument:
    payload = _payload_mapping(data)
    return IRDocument(
        doc_id=_required_text(payload, "doc_id"),
        path=_required_text(payload, "path"),
        language=_required_text(payload, "language"),
        blob_oid=_string_or_none(payload.get("blob_oid")),
        content_hash=_string_or_none(payload.get("content_hash")),
        source_set=_string_set_payload(payload.get("source_set")),
    )


def _symbol_from_payload(data: Any) -> IRSymbol:
    payload = _payload_mapping(data)
    return IRSymbol(
        symbol_id=_required_text(payload, "symbol_id"),
        external_symbol_id=_string_or_none(payload.get("external_symbol_id")),
        path=_required_text(payload, "path"),
        display_name=_required_text(payload, "display_name"),
        kind=_required_text(payload, "kind"),
        language=_required_text(payload, "language"),
        qualified_name=_string_or_none(payload.get("qualified_name")),
        signature=_string_or_none(payload.get("signature")),
        start_line=_int_or_none(payload.get("start_line")),
        start_col=_int_or_none(payload.get("start_col")),
        end_line=_int_or_none(payload.get("end_line")),
        end_col=_int_or_none(payload.get("end_col")),
        source_priority=_int_or_none(payload.get("source_priority")) or 0,
        source_set=_string_set_payload(payload.get("source_set")),
        metadata=_json_mapping_payload(payload.get("metadata")),
    )


def _occurrence_from_payload(data: Any) -> IROccurrence:
    payload = _payload_mapping(data)
    return IROccurrence(
        occurrence_id=_required_text(payload, "occurrence_id"),
        symbol_id=_required_text(payload, "symbol_id"),
        doc_id=_required_text(payload, "doc_id"),
        role=_required_text(payload, "role"),
        start_line=_int_or_none(payload.get("start_line")) or 0,
        start_col=_int_or_none(payload.get("start_col")) or 0,
        end_line=_int_or_none(payload.get("end_line")) or 0,
        end_col=_int_or_none(payload.get("end_col")) or 0,
        source=_required_text(payload, "source"),
        metadata=_json_mapping_payload(payload.get("metadata")),
    )


def _edge_from_payload(data: Any) -> IREdge:
    payload = _payload_mapping(data)
    return IREdge(
        edge_id=_required_text(payload, "edge_id"),
        src_id=_required_text(payload, "src_id"),
        dst_id=_required_text(payload, "dst_id"),
        edge_type=_required_text(payload, "edge_type"),
        source=_required_text(payload, "source"),
        confidence=_required_text(payload, "confidence"),
        doc_id=_string_or_none(payload.get("doc_id")),
        metadata=_json_mapping_payload(payload.get("metadata")),
    )


def _attachment_from_payload(data: Any) -> IRAttachment:
    payload = _payload_mapping(data)
    return IRAttachment(
        attachment_id=_required_text(payload, "attachment_id"),
        target_id=_required_text(payload, "target_id"),
        target_type=_required_text(payload, "target_type"),
        attachment_type=_required_text(payload, "attachment_type"),
        source=_required_text(payload, "source"),
        confidence=_required_text(payload, "confidence"),
        payload=_json_mapping_payload(payload.get("payload")),
        metadata=_json_mapping_payload(payload.get("metadata")),
    )


def _snapshot_from_payload(data: Any) -> IRSnapshot:
    payload = _payload_mapping(data)
    repo_name = _required_text(payload, "repo_name")
    snapshot_id = _required_text(payload, "snapshot_id")
    branch = _string_or_none(payload.get("branch"))
    commit_id = _string_or_none(payload.get("commit_id"))
    tree_id = _string_or_none(payload.get("tree_id"))
    metadata = _json_mapping_payload(payload.get("metadata"))
    if payload.get("units") is not None or payload.get("supports") is not None:
        return IRSnapshot(
            repo_name=repo_name,
            snapshot_id=snapshot_id,
            branch=branch,
            commit_id=commit_id,
            tree_id=tree_id,
            units=[
                _code_unit_from_payload(unit)
                for unit in _sequence_items(payload.get("units"))
            ],
            supports=[
                _unit_support_from_payload(support)
                for support in _sequence_items(payload.get("supports"))
            ],
            relations=[
                _relation_from_payload(relation)
                for relation in _sequence_items(payload.get("relations"))
            ],
            embeddings=[
                _embedding_from_payload(embedding)
                for embedding in _sequence_items(payload.get("embeddings"))
            ],
            metadata=metadata,
        )
    return IRSnapshot(
        repo_name=repo_name,
        snapshot_id=snapshot_id,
        branch=branch,
        commit_id=commit_id,
        tree_id=tree_id,
        documents=[
            _document_from_payload(document)
            for document in _sequence_items(payload.get("documents"))
        ],
        symbols=[
            _symbol_from_payload(symbol)
            for symbol in _sequence_items(payload.get("symbols"))
        ],
        occurrences=[
            _occurrence_from_payload(occurrence)
            for occurrence in _sequence_items(payload.get("occurrences"))
        ],
        edges=[
            _edge_from_payload(edge) for edge in _sequence_items(payload.get("edges"))
        ],
        attachments=[
            _attachment_from_payload(attachment)
            for attachment in _sequence_items(payload.get("attachments"))
        ],
        metadata=metadata,
    )


def _load_snapshot_shard_rows(
    *,
    snap_dir: str,
    subdir: str,
    shards: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in _sequence_items(shards):
        if not isinstance(entry, Mapping):
            continue
        shard_file = entry.get("shard_file")
        if not shard_file:
            continue
        shard_path = os.path.join(snap_dir, subdir, str(shard_file))
        with open(shard_path, encoding="utf-8") as handle:
            increment_materialization_boundary(BOUNDARY_JSON_DECODE)
            payload = json.load(handle)
        shard_rows = payload.get("rows", []) if isinstance(payload, dict) else []
        rows.extend(
            dict(cast(Mapping[str, Any], row))
            for row in shard_rows
            if isinstance(row, Mapping)
        )
    return rows


def _embedding_from_sharded_payload(
    *,
    snap_dir: str,
    data: Mapping[str, Any],
) -> IRUnitEmbedding:
    payload = dict(data)
    metadata = _json_mapping_payload(payload.get("metadata"))
    vector_ref = metadata.get("vector_ref")
    if payload.get("vector") is None and vector_ref:
        vector_path = os.path.join(snap_dir, str(vector_ref))
        if os.path.exists(vector_path):
            payload["vector"] = [
                float(value)
                for value in np.load(vector_path, allow_pickle=False).astype(
                    np.float32, copy=False
                )
            ]
    payload["metadata"] = metadata
    return _embedding_from_payload(payload)


def _snapshot_from_sharded_payload(
    *,
    snap_dir: str,
    manifest: Mapping[str, Any],
) -> IRSnapshot:
    unit_rows = _ordered_sharded_rows(
        _load_snapshot_shard_rows(
            snap_dir=snap_dir,
            subdir="units",
            shards=manifest.get("units"),
        )
    )
    support_rows = _ordered_sharded_rows(
        _load_snapshot_shard_rows(
            snap_dir=snap_dir,
            subdir="supports",
            shards=manifest.get("supports"),
        )
    )
    relation_rows = _ordered_sharded_rows(
        _load_snapshot_shard_rows(
            snap_dir=snap_dir,
            subdir="relations",
            shards=manifest.get("relations"),
        )
    )
    embedding_rows = _ordered_sharded_rows(
        _load_snapshot_shard_rows(
            snap_dir=snap_dir,
            subdir="embeddings",
            shards=manifest.get("embeddings"),
        )
    )
    return IRSnapshot(
        repo_name=_required_text(manifest, "repo_name"),
        snapshot_id=_required_text(manifest, "snapshot_id"),
        branch=_string_or_none(manifest.get("branch")),
        commit_id=_string_or_none(manifest.get("commit_id")),
        tree_id=_string_or_none(manifest.get("tree_id")),
        units=[_code_unit_from_payload(row) for row in unit_rows],
        supports=[_unit_support_from_payload(row) for row in support_rows],
        relations=[_relation_from_payload(row) for row in relation_rows],
        embeddings=[
            _embedding_from_sharded_payload(snap_dir=snap_dir, data=row)
            for row in embedding_rows
        ],
        metadata=_json_mapping_payload(manifest.get("metadata")),
    )
