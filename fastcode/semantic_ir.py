"""
Canonical semantic IR models for snapshot-based indexing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .utils import safe_jsonable


@dataclass
class IRDocument:
    doc_id: str
    path: str
    language: str
    blob_oid: str | None = None
    content_hash: str | None = None
    source_set: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_set"] = sorted(self.source_set)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IRDocument:
        payload = dict(data)
        payload["source_set"] = set(payload.get("source_set", []))
        return cls(**payload)


@dataclass
class IRSymbol:
    symbol_id: str
    external_symbol_id: str | None
    path: str
    display_name: str
    kind: str
    language: str
    qualified_name: str | None = None
    signature: str | None = None
    start_line: int | None = None
    start_col: int | None = None
    end_line: int | None = None
    end_col: int | None = None
    source_priority: int = 0
    source_set: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_set"] = sorted(self.source_set)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IRSymbol:
        payload = dict(data)
        payload["source_set"] = set(payload.get("source_set", []))
        return cls(**payload)


@dataclass
class IROccurrence:
    occurrence_id: str
    symbol_id: str
    doc_id: str
    role: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IROccurrence:
        return cls(**data)


@dataclass
class IREdge:
    edge_id: str
    src_id: str
    dst_id: str
    edge_type: str
    source: str
    confidence: str
    doc_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IREdge:
        return cls(**data)


@dataclass
class IRAttachment:
    attachment_id: str
    target_id: str
    target_type: str
    attachment_type: str
    source: str
    confidence: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["payload"] = safe_jsonable(self.payload)
        data["metadata"] = safe_jsonable(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IRAttachment:
        payload = dict(data)
        payload["payload"] = safe_jsonable(payload.get("payload", {}))
        payload["metadata"] = safe_jsonable(payload.get("metadata", {}))
        return cls(**payload)


@dataclass
class IRSnapshot:
    repo_name: str
    snapshot_id: str
    branch: str | None = None
    commit_id: str | None = None
    tree_id: str | None = None
    documents: list[IRDocument] = field(default_factory=list)
    symbols: list[IRSymbol] = field(default_factory=list)
    occurrences: list[IROccurrence] = field(default_factory=list)
    edges: list[IREdge] = field(default_factory=list)
    attachments: list[IRAttachment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_name": self.repo_name,
            "snapshot_id": self.snapshot_id,
            "branch": self.branch,
            "commit_id": self.commit_id,
            "tree_id": self.tree_id,
            "documents": [d.to_dict() for d in self.documents],
            "symbols": [s.to_dict() for s in self.symbols],
            "occurrences": [o.to_dict() for o in self.occurrences],
            "edges": [e.to_dict() for e in self.edges],
            "attachments": [a.to_dict() for a in self.attachments],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IRSnapshot:
        return cls(
            repo_name=data["repo_name"],
            snapshot_id=data["snapshot_id"],
            branch=data.get("branch"),
            commit_id=data.get("commit_id"),
            tree_id=data.get("tree_id"),
            documents=[IRDocument.from_dict(d) for d in data.get("documents", [])],
            symbols=[IRSymbol.from_dict(s) for s in data.get("symbols", [])],
            occurrences=[IROccurrence.from_dict(o) for o in data.get("occurrences", [])],
            edges=[IREdge.from_dict(e) for e in data.get("edges", [])],
            attachments=[IRAttachment.from_dict(a) for a in data.get("attachments", [])],
            metadata=data.get("metadata", {}),
        )
