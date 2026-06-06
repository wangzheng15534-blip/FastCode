"""Index-run persistence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IndexRunRecord:
    run_id: str
    repo_name: str
    snapshot_id: str
    branch: str | None
    commit_id: str | None
    idempotency_key: str | None
    status: str
    error_message: str | None
    warnings_json: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "repo_name": self.repo_name,
            "snapshot_id": self.snapshot_id,
            "branch": self.branch,
            "commit_id": self.commit_id,
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "error_message": self.error_message,
            "warnings_json": self.warnings_json,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexRunRecord:
        return cls(
            run_id=str(data.get("run_id") or ""),
            repo_name=str(data.get("repo_name") or ""),
            snapshot_id=str(data.get("snapshot_id") or ""),
            branch=str(data["branch"]) if data.get("branch") is not None else None,
            commit_id=(
                str(data["commit_id"]) if data.get("commit_id") is not None else None
            ),
            idempotency_key=(
                str(data["idempotency_key"])
                if data.get("idempotency_key") is not None
                else None
            ),
            status=str(data.get("status") or ""),
            error_message=(
                str(data["error_message"])
                if data.get("error_message") is not None
                else None
            ),
            warnings_json=(
                str(data["warnings_json"])
                if data.get("warnings_json") is not None
                else None
            ),
            created_at=str(data.get("created_at") or ""),
            started_at=(
                str(data["started_at"]) if data.get("started_at") is not None else None
            ),
            completed_at=(
                str(data["completed_at"])
                if data.get("completed_at") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class PublishTaskRecord:
    task_id: str
    run_id: str
    snapshot_id: str
    manifest_id: str | None
    status: str
    attempts: int
    last_error: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "snapshot_id": self.snapshot_id,
            "manifest_id": self.manifest_id,
            "status": self.status,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PublishTaskRecord:
        return cls(
            task_id=str(data.get("task_id") or ""),
            run_id=str(data.get("run_id") or ""),
            snapshot_id=str(data.get("snapshot_id") or ""),
            manifest_id=(
                str(data["manifest_id"])
                if data.get("manifest_id") is not None
                else None
            ),
            status=str(data.get("status") or ""),
            attempts=int(data.get("attempts") or 0),
            last_error=(
                str(data["last_error"]) if data.get("last_error") is not None else None
            ),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )
