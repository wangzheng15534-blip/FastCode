"""File artifact persistence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FileArtifactRecord:
    repo_name: str
    relative_path: str
    identity_kind: str
    identity_value: str
    artifact_type: str
    schema_version: str
    payload_json: str
    unit_count: int
    support_count: int
    relation_count: int
    embedding_count: int
    metadata_json: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_name": self.repo_name,
            "relative_path": self.relative_path,
            "identity_kind": self.identity_kind,
            "identity_value": self.identity_value,
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "payload_json": self.payload_json,
            "unit_count": self.unit_count,
            "support_count": self.support_count,
            "relation_count": self.relation_count,
            "embedding_count": self.embedding_count,
            "metadata_json": self.metadata_json,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileArtifactRecord:
        return cls(
            repo_name=str(data.get("repo_name") or ""),
            relative_path=str(data.get("relative_path") or ""),
            identity_kind=str(data.get("identity_kind") or ""),
            identity_value=str(data.get("identity_value") or ""),
            artifact_type=str(data.get("artifact_type") or ""),
            schema_version=str(data.get("schema_version") or ""),
            payload_json=str(data.get("payload_json") or "{}"),
            unit_count=int(data.get("unit_count") or 0),
            support_count=int(data.get("support_count") or 0),
            relation_count=int(data.get("relation_count") or 0),
            embedding_count=int(data.get("embedding_count") or 0),
            metadata_json=(
                str(data["metadata_json"])
                if data.get("metadata_json") is not None
                else None
            ),
            created_at=str(data.get("created_at") or ""),
        )
