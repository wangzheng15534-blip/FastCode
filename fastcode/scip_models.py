"""
Typed SCIP payload models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SCIPOccurrence:
    symbol: str
    role: str = "reference"
    range: List[Optional[int]] = field(default_factory=lambda: [None, None, None, None])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SCIPOccurrence":
        return cls(
            symbol=str(data.get("symbol") or ""),
            role=str(data.get("role") or "reference"),
            range=list(data.get("range") or [None, None, None, None]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "role": self.role,
            "range": list(self.range or [None, None, None, None]),
        }


@dataclass
class SCIPSymbol:
    symbol: str
    name: Optional[str] = None
    kind: Optional[str] = None
    qualified_name: Optional[str] = None
    signature: Optional[str] = None
    range: List[Optional[int]] = field(default_factory=lambda: [None, None, None, None])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SCIPSymbol":
        return cls(
            symbol=str(data.get("symbol") or ""),
            name=data.get("name"),
            kind=data.get("kind"),
            qualified_name=data.get("qualified_name"),
            signature=data.get("signature"),
            range=list(data.get("range") or [None, None, None, None]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "kind": self.kind,
            "qualified_name": self.qualified_name,
            "signature": self.signature,
            "range": list(self.range or [None, None, None, None]),
        }


@dataclass
class SCIPDocument:
    path: str
    language: Optional[str] = None
    symbols: List[SCIPSymbol] = field(default_factory=list)
    occurrences: List[SCIPOccurrence] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SCIPDocument":
        return cls(
            path=str(data.get("path") or ""),
            language=data.get("language"),
            symbols=[SCIPSymbol.from_dict(s) for s in (data.get("symbols") or [])],
            occurrences=[SCIPOccurrence.from_dict(o) for o in (data.get("occurrences") or [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "symbols": [s.to_dict() for s in self.symbols],
            "occurrences": [o.to_dict() for o in self.occurrences],
        }


@dataclass
class SCIPIndex:
    documents: List[SCIPDocument] = field(default_factory=list)
    indexer_name: Optional[str] = None
    indexer_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SCIPIndex":
        reserved = {"documents", "indexer_name", "indexer_version"}
        metadata = {k: v for k, v in data.items() if k not in reserved}
        return cls(
            documents=[SCIPDocument.from_dict(d) for d in (data.get("documents") or [])],
            indexer_name=data.get("indexer_name"),
            indexer_version=data.get("indexer_version"),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.metadata or {})
        data.update(
            {
                "documents": [d.to_dict() for d in self.documents],
                "indexer_name": self.indexer_name,
                "indexer_version": self.indexer_version,
            }
        )
        return data


@dataclass
class SCIPArtifactRef:
    snapshot_id: str
    indexer_name: str
    indexer_version: Optional[str]
    artifact_path: str
    checksum: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "indexer_name": self.indexer_name,
            "indexer_version": self.indexer_version,
            "artifact_path": self.artifact_path,
            "checksum": self.checksum,
            "created_at": self.created_at,
        }
