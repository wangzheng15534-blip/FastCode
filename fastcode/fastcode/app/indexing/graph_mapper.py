"""Bidirectional mapper between indexing document payloads and generic graph node records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentOverlayNodeRecord:
    """Generic graph node record built from indexing document payloads."""

    collection: str
    key_field: str
    key_value: Any
    _properties: Mapping[str, Any]

    @property
    def property_names(self) -> Sequence[str]:
        names = [self.key_field]
        names.extend(name for name in self._properties if name != self.key_field)
        return tuple(names)

    def property_value(self, name: str) -> Any:
        if name == self.key_field:
            return self.key_value
        return self._properties.get(name)


def document_overlay_node_records(
    *,
    chunks: Sequence[Mapping[str, Any]],
    mentions: Sequence[Mapping[str, Any]],
) -> list[DocumentOverlayNodeRecord]:
    """Map document chunks and mentions to generic graph upsert records."""
    records: list[DocumentOverlayNodeRecord] = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue
        records.append(
            DocumentOverlayNodeRecord(
                collection="design_documents",
                key_field="chunk_id",
                key_value=chunk_id,
                _properties={
                    "chunk_id": chunk_id,
                    "snapshot_id": chunk.get("snapshot_id"),
                    "repo_name": chunk.get("repo_name"),
                    "path": chunk.get("path"),
                    "title": chunk.get("title"),
                    "heading": chunk.get("heading"),
                    "doc_type": chunk.get("doc_type"),
                    "content": chunk.get("content"),
                },
            )
        )
    for mention in mentions:
        chunk_id = mention.get("chunk_id")
        symbol_id = mention.get("symbol_id")
        if not chunk_id or not symbol_id:
            continue
        mention_id = f"{chunk_id}:{symbol_id}"
        records.append(
            DocumentOverlayNodeRecord(
                collection="mentions",
                key_field="mention_id",
                key_value=mention_id,
                _properties={
                    "mention_id": mention_id,
                    "chunk_id": chunk_id,
                    "symbol_id": symbol_id,
                    "confidence": mention.get("confidence"),
                },
            )
        )
    return records
