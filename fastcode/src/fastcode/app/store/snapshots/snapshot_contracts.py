"""Snapshot persistence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SnapshotRefRecord:
    ref_id: int | None
    repo_name: str
    branch: str | None
    commit_id: str | None
    tree_id: str | None
    snapshot_id: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "repo_name": self.repo_name,
            "branch": self.branch,
            "commit_id": self.commit_id,
            "tree_id": self.tree_id,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotRefRecord:
        raw_ref_id = data.get("ref_id")
        return cls(
            ref_id=int(raw_ref_id) if raw_ref_id is not None else None,
            repo_name=str(data.get("repo_name") or ""),
            branch=str(data["branch"]) if data.get("branch") is not None else None,
            commit_id=(
                str(data["commit_id"]) if data.get("commit_id") is not None else None
            ),
            tree_id=str(data["tree_id"]) if data.get("tree_id") is not None else None,
            snapshot_id=str(data.get("snapshot_id") or ""),
            created_at=str(data.get("created_at") or ""),
        )


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    repo_name: str
    branch: str | None
    commit_id: str | None
    tree_id: str | None
    artifact_key: str
    ir_path: str
    ir_graphs_path: str | None
    created_at: str
    metadata_json: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "repo_name": self.repo_name,
            "branch": self.branch,
            "commit_id": self.commit_id,
            "tree_id": self.tree_id,
            "artifact_key": self.artifact_key,
            "ir_path": self.ir_path,
            "ir_graphs_path": self.ir_graphs_path,
            "created_at": self.created_at,
            "metadata_json": self.metadata_json,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotRecord:
        return cls(
            snapshot_id=str(data.get("snapshot_id") or ""),
            repo_name=str(data.get("repo_name") or ""),
            branch=str(data["branch"]) if data.get("branch") is not None else None,
            commit_id=(
                str(data["commit_id"]) if data.get("commit_id") is not None else None
            ),
            tree_id=str(data["tree_id"]) if data.get("tree_id") is not None else None,
            artifact_key=str(data.get("artifact_key") or ""),
            ir_path=str(data.get("ir_path") or ""),
            ir_graphs_path=(
                str(data["ir_graphs_path"])
                if data.get("ir_graphs_path") is not None
                else None
            ),
            created_at=str(data.get("created_at") or ""),
            metadata_json=(
                str(data["metadata_json"])
                if data.get("metadata_json") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class SCIPArtifactRecord:
    snapshot_id: str
    indexer_name: str
    indexer_version: str | None
    artifact_path: str
    checksum: str
    created_at: str
    artifact_id: str | None = None
    sequence_no: int | None = None
    role: str | None = None
    metadata_json: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "indexer_name": self.indexer_name,
            "indexer_version": self.indexer_version,
            "artifact_path": self.artifact_path,
            "checksum": self.checksum,
            "created_at": self.created_at,
            "artifact_id": self.artifact_id,
            "sequence_no": self.sequence_no,
            "role": self.role,
            "metadata_json": self.metadata_json,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIPArtifactRecord:
        raw_sequence_no = data.get("sequence_no")
        return cls(
            snapshot_id=str(data.get("snapshot_id") or ""),
            indexer_name=str(data.get("indexer_name") or ""),
            indexer_version=(
                str(data["indexer_version"])
                if data.get("indexer_version") is not None
                else None
            ),
            artifact_path=str(data.get("artifact_path") or ""),
            checksum=str(data.get("checksum") or ""),
            created_at=str(data.get("created_at") or ""),
            artifact_id=(
                str(data["artifact_id"])
                if data.get("artifact_id") is not None
                else None
            ),
            sequence_no=(int(raw_sequence_no) if raw_sequence_no is not None else None),
            role=str(data["role"]) if data.get("role") is not None else None,
            metadata_json=(
                str(data["metadata_json"])
                if data.get("metadata_json") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class RedoTaskRecord:
    task_id: str
    task_type: str
    payload_json: str
    status: str
    attempts: int
    last_error: str | None
    next_attempt_at: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload_json": self.payload_json,
            "status": self.status,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "next_attempt_at": self.next_attempt_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RedoTaskRecord:
        return cls(
            task_id=str(data.get("task_id") or ""),
            task_type=str(data.get("task_type") or ""),
            payload_json=str(data.get("payload_json") or ""),
            status=str(data.get("status") or ""),
            attempts=int(data.get("attempts") or 0),
            last_error=(
                str(data["last_error"]) if data.get("last_error") is not None else None
            ),
            next_attempt_at=(
                str(data["next_attempt_at"])
                if data.get("next_attempt_at") is not None
                else None
            ),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass(frozen=True)
class OutboxEventRecord:
    event_id: str
    event_type: str
    payload: str
    snapshot_id: str
    status: str
    attempts: int
    max_attempts: int
    created_at: str
    last_attempt_at: str | None
    error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "snapshot_id": self.snapshot_id,
            "status": self.status,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "last_attempt_at": self.last_attempt_at,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutboxEventRecord:
        return cls(
            event_id=str(data.get("event_id") or ""),
            event_type=str(data.get("event_type") or ""),
            payload=str(data.get("payload") or ""),
            snapshot_id=str(data.get("snapshot_id") or ""),
            status=str(data.get("status") or ""),
            attempts=int(data.get("attempts") or 0),
            max_attempts=int(data.get("max_attempts") or 0),
            created_at=str(data.get("created_at") or ""),
            last_attempt_at=(
                str(data["last_attempt_at"])
                if data.get("last_attempt_at") is not None
                else None
            ),
            error_message=(
                str(data["error_message"])
                if data.get("error_message") is not None
                else None
            ),
        )
