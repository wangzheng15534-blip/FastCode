"""Manifest persistence contracts."""

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
