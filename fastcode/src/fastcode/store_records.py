"""Typed records for manifest and snapshot-ref store boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ManifestRecord:
    manifest_id: str
    repo_name: str
    ref_name: str
    snapshot_id: str
    index_run_id: str
    published_at: str
    previous_manifest_id: str | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "repo_name": self.repo_name,
            "ref_name": self.ref_name,
            "snapshot_id": self.snapshot_id,
            "index_run_id": self.index_run_id,
            "published_at": self.published_at,
            "previous_manifest_id": self.previous_manifest_id,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestRecord:
        return cls(
            manifest_id=str(data.get("manifest_id") or ""),
            repo_name=str(data.get("repo_name") or ""),
            ref_name=str(data.get("ref_name") or ""),
            snapshot_id=str(data.get("snapshot_id") or ""),
            index_run_id=str(data.get("index_run_id") or ""),
            published_at=str(data.get("published_at") or ""),
            previous_manifest_id=(
                str(data["previous_manifest_id"])
                if data.get("previous_manifest_id") is not None
                else None
            ),
            status=str(data.get("status") or ""),
        )


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
