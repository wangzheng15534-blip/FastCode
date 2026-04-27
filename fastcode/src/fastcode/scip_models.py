"""
Typed SCIP payload models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_EMPTY_RANGE: tuple[None, ...] = (None, None, None, None)


@dataclass
class SCIPOccurrence:
    symbol: str
    role: str = "reference"
    range: list[int | None] = field(default_factory=lambda: list(_EMPTY_RANGE))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIPOccurrence:
        return cls(
            symbol=str(data.get("symbol") or ""),
            role=str(data.get("role") or "reference"),
            range=list(data.get("range") or _EMPTY_RANGE),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "role": self.role,
            "range": list(self.range or _EMPTY_RANGE),
        }


@dataclass
class SCIPSymbol:
    symbol: str
    name: str | None = None
    kind: str | None = None
    qualified_name: str | None = None
    signature: str | None = None
    range: list[int | None] = field(default_factory=lambda: list(_EMPTY_RANGE))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIPSymbol:
        return cls(
            symbol=str(data.get("symbol") or ""),
            name=data.get("name"),
            kind=data.get("kind"),
            qualified_name=data.get("qualified_name"),
            signature=data.get("signature"),
            range=list(data.get("range") or _EMPTY_RANGE),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "kind": self.kind,
            "qualified_name": self.qualified_name,
            "signature": self.signature,
            "range": list(self.range or _EMPTY_RANGE),
        }


@dataclass
class SCIPDocument:
    path: str
    language: str | None = None
    symbols: list[SCIPSymbol] = field(default_factory=list)
    occurrences: list[SCIPOccurrence] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIPDocument:
        return cls(
            path=str(data.get("path") or ""),
            language=data.get("language"),
            symbols=[SCIPSymbol.from_dict(s) for s in (data.get("symbols") or [])],
            occurrences=[
                SCIPOccurrence.from_dict(o) for o in (data.get("occurrences") or [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "symbols": [s.to_dict() for s in self.symbols],
            "occurrences": [o.to_dict() for o in self.occurrences],
        }


@dataclass
class SCIPIndex:
    documents: list[SCIPDocument] = field(default_factory=list)
    indexer_name: str | None = None
    indexer_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIPIndex:
        reserved = {"documents", "indexer_name", "indexer_version"}
        metadata = {k: v for k, v in data.items() if k not in reserved}
        raw_docs = data.get("documents") or []
        documents = [SCIPDocument.from_dict(d) for d in raw_docs if isinstance(d, dict)]
        return cls(
            documents=documents,
            indexer_name=data.get("indexer_name"),
            indexer_version=data.get("indexer_version"),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
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
    indexer_version: str | None
    artifact_path: str
    checksum: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "indexer_name": self.indexer_name,
            "indexer_version": self.indexer_version,
            "artifact_path": self.artifact_path,
            "checksum": self.checksum,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIPArtifactRef:
        return cls(
            snapshot_id=str(data.get("snapshot_id") or ""),
            indexer_name=str(data.get("indexer_name") or ""),
            indexer_version=data.get("indexer_version"),
            artifact_path=str(data.get("artifact_path") or ""),
            checksum=str(data.get("checksum") or ""),
            created_at=str(data.get("created_at") or ""),
        )
