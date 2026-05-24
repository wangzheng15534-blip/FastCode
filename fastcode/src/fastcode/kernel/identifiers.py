"""Shared kernel identity types."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..foundation.non_empty_string import NonEmptyString
from .error import KernelError

_ARTIFACT_KEY_RE = re.compile(r"^[A-Za-z0-9:_-]+$")


@dataclass(frozen=True)
class RepoName:
    """Repository identity shared across indexing, storage, and APIs."""

    value: str

    def __post_init__(self) -> None:
        normalized = NonEmptyString.parse(self.value).as_str()
        if any(part in normalized for part in ("/", "\\", "\x00", ":")):
            raise KernelError.invalid_repo_name(
                "repo name must not contain path separators, null bytes, or ':'"
            )
        object.__setattr__(self, "value", normalized)

    @classmethod
    def parse(cls, value: str) -> RepoName:
        return cls(str(value))

    def as_str(self) -> str:
        return self.value


@dataclass(frozen=True)
class SnapshotId:
    """Snapshot identity shared across pipelines, storage, and APIs."""

    value: str

    def __post_init__(self) -> None:
        normalized = NonEmptyString.parse(self.value).as_str()
        if not normalized.startswith("snap:"):
            raise KernelError.invalid_snapshot_id("snapshot id must start with 'snap:'")
        suffix = normalized.removeprefix("snap:")
        if not suffix:
            raise KernelError.invalid_snapshot_id(
                "snapshot id must include a non-empty suffix after 'snap:'"
            )
        parts = suffix.split(":")
        if any(not part for part in parts):
            raise KernelError.invalid_snapshot_id(
                "snapshot id must not contain empty path segments"
            )
        if len(parts) >= 2:
            RepoName.parse(parts[0])
        object.__setattr__(self, "value", normalized)

    @classmethod
    def parse(cls, value: str) -> SnapshotId:
        return cls(str(value))

    @classmethod
    def build(cls, repo_name: str, revision: str) -> SnapshotId:
        repo = RepoName.parse(repo_name)
        suffix = NonEmptyString.parse(revision).as_str()
        return cls(f"snap:{repo.as_str()}:{suffix}")

    def as_str(self) -> str:
        return self.value


@dataclass(frozen=True)
class ArtifactKey:
    """Artifact identity shared across snapshot persistence and cache handles."""

    value: str

    def __post_init__(self) -> None:
        normalized = NonEmptyString.parse(self.value).as_str()
        if not _ARTIFACT_KEY_RE.fullmatch(normalized):
            raise KernelError.invalid_artifact_key(
                "artifact key must match [A-Za-z0-9:_-]+"
            )
        object.__setattr__(self, "value", normalized)

    @classmethod
    def parse(cls, value: str) -> ArtifactKey:
        return cls(str(value))

    @classmethod
    def for_snapshot(cls, snapshot_id: SnapshotId | str) -> ArtifactKey:
        snapshot = (
            snapshot_id if isinstance(snapshot_id, SnapshotId) else SnapshotId.parse(snapshot_id)
        )
        digest = hashlib.sha256(snapshot.as_str().encode("utf-8")).hexdigest()[:20]
        return cls(f"snap_{digest}")

    def as_str(self) -> str:
        return self.value


def validate_repo_name(value: str) -> str:
    return RepoName.parse(value).as_str()


def validate_snapshot_id(value: str) -> str:
    return SnapshotId.parse(value).as_str()


def validate_artifact_key(value: str) -> str:
    return ArtifactKey.parse(value).as_str()
