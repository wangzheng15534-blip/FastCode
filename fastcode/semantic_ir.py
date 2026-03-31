"""
Canonical semantic IR models for snapshot-based indexing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class IRDocument:
    doc_id: str
    path: str
    language: str
    blob_oid: Optional[str] = None
    content_hash: Optional[str] = None
    source_set: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["source_set"] = sorted(list(self.source_set))
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IRDocument":
        payload = dict(data)
        payload["source_set"] = set(payload.get("source_set", []))
        return cls(**payload)


@dataclass
class IRSymbol:
    symbol_id: str
    external_symbol_id: Optional[str]
    path: str
    display_name: str
    kind: str
    language: str
    qualified_name: Optional[str] = None
    signature: Optional[str] = None
    start_line: Optional[int] = None
    start_col: Optional[int] = None
    end_line: Optional[int] = None
    end_col: Optional[int] = None
    source_priority: int = 0
    source_set: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["source_set"] = sorted(list(self.source_set))
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IRSymbol":
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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IROccurrence":
        return cls(**data)


@dataclass
class IREdge:
    edge_id: str
    src_id: str
    dst_id: str
    edge_type: str
    source: str
    confidence: str
    doc_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IREdge":
        return cls(**data)


@dataclass
class IRSnapshot:
    repo_name: str
    snapshot_id: str
    branch: Optional[str] = None
    commit_id: Optional[str] = None
    tree_id: Optional[str] = None
    documents: List[IRDocument] = field(default_factory=list)
    symbols: List[IRSymbol] = field(default_factory=list)
    occurrences: List[IROccurrence] = field(default_factory=list)
    edges: List[IREdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IRSnapshot":
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
            metadata=data.get("metadata", {}),
        )
