from __future__ import annotations

import pytest

from fastcode.common.error import KernelError
from fastcode.common.identifiers import ArtifactKey, RepoName, SnapshotId


def test_repo_name_rejects_path_separator() -> None:
    with pytest.raises(KernelError, match="invalid_repo_name"):
        RepoName.parse("bad/repo")


def test_snapshot_id_build_uses_repo_and_revision() -> None:
    assert SnapshotId.build("repo", "abc123").as_str() == "snap:repo:abc123"


def test_snapshot_id_rejects_missing_prefix() -> None:
    with pytest.raises(KernelError, match="invalid_snapshot_id"):
        SnapshotId.parse("repo:abc123")


def test_artifact_key_for_snapshot_is_deterministic() -> None:
    key1 = ArtifactKey.for_snapshot("snap:repo:abc123").as_str()
    key2 = ArtifactKey.for_snapshot("snap:repo:abc123").as_str()
    assert key1 == key2
    assert key1.startswith("snap_")
