"""Unit artifact persistence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UnitArtifactRecord:
    snapshot_id: str
    stable_unit_id: str
    relative_path: str
    unit_type: str
    content_hash: str | None
    syntax_hash: str | None
    signature_hash: str | None
    edge_surface_hash: str | None
    embedding_text_hash: str | None
    api_surface_hash: str | None
    embedding_artifact_ref: str | None
    scoped_tool_ref: str | None
    package_root: str | None
    repair_frontier_summary: str | None
    metadata_json: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "stable_unit_id": self.stable_unit_id,
            "relative_path": self.relative_path,
            "unit_type": self.unit_type,
            "content_hash": self.content_hash,
            "syntax_hash": self.syntax_hash,
            "signature_hash": self.signature_hash,
            "edge_surface_hash": self.edge_surface_hash,
            "embedding_text_hash": self.embedding_text_hash,
            "api_surface_hash": self.api_surface_hash,
            "embedding_artifact_ref": self.embedding_artifact_ref,
            "scoped_tool_ref": self.scoped_tool_ref,
            "package_root": self.package_root,
            "repair_frontier_summary": self.repair_frontier_summary,
            "metadata_json": self.metadata_json,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnitArtifactRecord:
        return cls(
            snapshot_id=str(data.get("snapshot_id") or ""),
            stable_unit_id=str(data.get("stable_unit_id") or ""),
            relative_path=str(data.get("relative_path") or ""),
            unit_type=str(data.get("unit_type") or ""),
            content_hash=(
                str(data["content_hash"])
                if data.get("content_hash") is not None
                else None
            ),
            syntax_hash=(
                str(data["syntax_hash"])
                if data.get("syntax_hash") is not None
                else None
            ),
            signature_hash=(
                str(data["signature_hash"])
                if data.get("signature_hash") is not None
                else None
            ),
            edge_surface_hash=(
                str(data["edge_surface_hash"])
                if data.get("edge_surface_hash") is not None
                else None
            ),
            embedding_text_hash=(
                str(data["embedding_text_hash"])
                if data.get("embedding_text_hash") is not None
                else None
            ),
            api_surface_hash=(
                str(data["api_surface_hash"])
                if data.get("api_surface_hash") is not None
                else None
            ),
            embedding_artifact_ref=(
                str(data["embedding_artifact_ref"])
                if data.get("embedding_artifact_ref") is not None
                else None
            ),
            scoped_tool_ref=(
                str(data["scoped_tool_ref"])
                if data.get("scoped_tool_ref") is not None
                else None
            ),
            package_root=(
                str(data["package_root"])
                if data.get("package_root") is not None
                else None
            ),
            repair_frontier_summary=(
                str(data["repair_frontier_summary"])
                if data.get("repair_frontier_summary") is not None
                else None
            ),
            metadata_json=(
                str(data["metadata_json"])
                if data.get("metadata_json") is not None
                else None
            ),
            created_at=str(data.get("created_at") or ""),
        )


@dataclass(frozen=True)
class FileIRShardRecord:
    snapshot_id: str
    relative_path: str
    schema_version: str
    payload_json: str
    unit_count: int
    support_count: int
    relation_count: int
    embedding_count: int
    content_hash: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "relative_path": self.relative_path,
            "schema_version": self.schema_version,
            "payload_json": self.payload_json,
            "unit_count": self.unit_count,
            "support_count": self.support_count,
            "relation_count": self.relation_count,
            "embedding_count": self.embedding_count,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileIRShardRecord:
        return cls(
            snapshot_id=str(data.get("snapshot_id") or ""),
            relative_path=str(data.get("relative_path") or ""),
            schema_version=str(data.get("schema_version") or ""),
            payload_json=str(data.get("payload_json") or "{}"),
            unit_count=int(data.get("unit_count") or 0),
            support_count=int(data.get("support_count") or 0),
            relation_count=int(data.get("relation_count") or 0),
            embedding_count=int(data.get("embedding_count") or 0),
            content_hash=(
                str(data["content_hash"])
                if data.get("content_hash") is not None
                else None
            ),
            created_at=str(data.get("created_at") or ""),
        )
