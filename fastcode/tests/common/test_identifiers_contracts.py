"""Contract tests for fastcode.common.identifiers — meaning_core, ZERO test doubles.

These tests guard the identity validation contracts: RepoName, SnapshotId,
ArtifactKey, and the convenience validators. Every test must fail if the
contract it guards is broken.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.common.error import KernelError
from fastcode.common.identifiers import (
    ArtifactKey,
    RepoName,
    SnapshotId,
    validate_artifact_key,
    validate_repo_name,
    validate_snapshot_id,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_safe_segment = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=32,
)

_invalid_repo_chars = st.sampled_from(["/", "\\", "\x00", ":"])


# ---------------------------------------------------------------------------
# RepoName
# ---------------------------------------------------------------------------


class TestRepoNameContracts:
    """RepoName must accept non-empty strings without path separators or colons."""

    def test_accepts_simple_name(self) -> None:
        name = RepoName.parse("my-repo")
        assert name.as_str() == "my-repo"

    def test_parse_normalizes_whitespace(self) -> None:
        name = RepoName.parse("  my-repo  ")
        assert name.as_str() == "my-repo"

    def test_rejects_forward_slash(self) -> None:
        with pytest.raises(KernelError, match="invalid_repo_name"):
            RepoName.parse("bad/repo")

    def test_rejects_backslash(self) -> None:
        with pytest.raises(KernelError, match="invalid_repo_name"):
            RepoName.parse("bad\\repo")

    def test_rejects_null_byte(self) -> None:
        with pytest.raises(KernelError, match="invalid_repo_name"):
            RepoName.parse("bad\x00repo")

    def test_rejects_colon(self) -> None:
        with pytest.raises(KernelError, match="invalid_repo_name"):
            RepoName.parse("bad:repo")

    @pytest.mark.edge
    def test_rejects_empty_string(self) -> None:
        with pytest.raises((KernelError, ValueError)):
            RepoName.parse("")

    @pytest.mark.edge
    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises((KernelError, ValueError)):
            RepoName.parse("   ")

    def test_accepts_underscores_and_hyphens(self) -> None:
        name = RepoName.parse("my_repo-name_2")
        assert name.as_str() == "my_repo-name_2"

    def test_frozen(self) -> None:
        name = RepoName.parse("repo")
        with pytest.raises(AttributeError):
            name.value = "other"  # type: ignore[misc]

    @given(name=_safe_segment)
    @settings(max_examples=30)
    @pytest.mark.property
    def test_safe_names_always_accepted(self, name: str) -> None:
        parsed = RepoName.parse(name)
        assert parsed.as_str() == name.strip()

    @given(char=_invalid_repo_chars)
    @settings(max_examples=4)
    @pytest.mark.property
    def test_invalid_chars_always_rejected(self, char: str) -> None:
        with pytest.raises(KernelError):
            RepoName.parse(f"bad{char}repo")


# ---------------------------------------------------------------------------
# SnapshotId
# ---------------------------------------------------------------------------


class TestSnapshotIdContracts:
    """SnapshotId must start with 'snap:' and contain non-empty suffix parts."""

    def test_build_produces_correct_format(self) -> None:
        sid = SnapshotId.build("repo", "rev1")
        assert sid.as_str() == "snap:repo:rev1"

    def test_parse_accepts_valid_id(self) -> None:
        sid = SnapshotId.parse("snap:repo:abc123")
        assert sid.as_str() == "snap:repo:abc123"

    def test_rejects_missing_prefix(self) -> None:
        with pytest.raises(KernelError, match="invalid_snapshot_id"):
            SnapshotId.parse("repo:abc123")

    @pytest.mark.negative
    def test_rejects_prefix_only(self) -> None:
        with pytest.raises(KernelError, match="invalid_snapshot_id"):
            SnapshotId.parse("snap:")

    @pytest.mark.negative
    def test_rejects_empty_string(self) -> None:
        with pytest.raises((KernelError, ValueError)):
            SnapshotId.parse("")

    @pytest.mark.negative
    def test_rejects_empty_segment(self) -> None:
        """snap:repo::rev has an empty path segment."""
        with pytest.raises(KernelError, match="invalid_snapshot_id"):
            SnapshotId.parse("snap:repo::rev")

    @pytest.mark.edge
    def test_three_part_id_accepted(self) -> None:
        sid = SnapshotId.parse("snap:repo:branch:abc")
        assert sid.as_str() == "snap:repo:branch:abc"

    def test_build_validates_repo_name(self) -> None:
        """build() must reject invalid repo names."""
        with pytest.raises(KernelError, match="invalid_repo_name"):
            SnapshotId.build("bad/repo", "rev1")

    def test_build_validates_revision(self) -> None:
        """build() must reject empty revision."""
        with pytest.raises((KernelError, ValueError)):
            SnapshotId.build("repo", "")

    def test_parse_normalizes_whitespace(self) -> None:
        sid = SnapshotId.parse("  snap:repo:rev  ")
        assert sid.as_str() == "snap:repo:rev"

    def test_frozen(self) -> None:
        sid = SnapshotId.parse("snap:repo:rev")
        with pytest.raises(AttributeError):
            sid.value = "other"  # type: ignore[misc]

    @given(repo=_safe_segment, rev=_safe_segment)
    @settings(max_examples=30)
    @pytest.mark.property
    def test_build_then_parse_roundtrip(self, repo: str, rev: str) -> None:
        built = SnapshotId.build(repo, rev)
        parsed = SnapshotId.parse(built.as_str())
        assert parsed.as_str() == built.as_str()


# ---------------------------------------------------------------------------
# ArtifactKey
# ---------------------------------------------------------------------------


class TestArtifactKeyContracts:
    """ArtifactKey must match [A-Za-z0-9:_-]+."""

    def test_accepts_valid_key(self) -> None:
        key = ArtifactKey.parse("my-key_123:abc")
        assert key.as_str() == "my-key_123:abc"

    @pytest.mark.negative
    def test_rejects_spaces(self) -> None:
        with pytest.raises(KernelError, match="invalid_artifact_key"):
            ArtifactKey.parse("has space")

    @pytest.mark.negative
    def test_rejects_special_chars(self) -> None:
        with pytest.raises(KernelError, match="invalid_artifact_key"):
            ArtifactKey.parse("bad@key!")

    @pytest.mark.negative
    def test_rejects_empty_string(self) -> None:
        with pytest.raises((KernelError, ValueError)):
            ArtifactKey.parse("")

    @pytest.mark.negative
    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises((KernelError, ValueError)):
            ArtifactKey.parse("   ")

    def test_for_snapshot_is_deterministic(self) -> None:
        k1 = ArtifactKey.for_snapshot("snap:repo:abc")
        k2 = ArtifactKey.for_snapshot("snap:repo:abc")
        assert k1.as_str() == k2.as_str()

    def test_for_snapshot_differs_for_different_ids(self) -> None:
        k1 = ArtifactKey.for_snapshot("snap:repo:abc")
        k2 = ArtifactKey.for_snapshot("snap:repo:def")
        assert k1.as_str() != k2.as_str()

    def test_for_snapshot_accepts_snapshot_id_object(self) -> None:
        sid = SnapshotId.parse("snap:repo:rev")
        key = ArtifactKey.for_snapshot(sid)
        assert key.as_str().startswith("snap_")

    def test_parse_normalizes_whitespace(self) -> None:
        key = ArtifactKey.parse("  valid-key  ")
        assert key.as_str() == "valid-key"

    def test_frozen(self) -> None:
        key = ArtifactKey.parse("key")
        with pytest.raises(AttributeError):
            key.value = "other"  # type: ignore[misc]

    @given(
        chars=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:_-",
            min_size=1,
            max_size=32,
        )
    )
    @settings(max_examples=50)
    @pytest.mark.property
    def test_valid_chars_always_accepted(self, chars: str) -> None:
        parsed = ArtifactKey.parse(chars)
        assert parsed.as_str() == chars.strip()


# ---------------------------------------------------------------------------
# Convenience validators
# ---------------------------------------------------------------------------


class TestConvenienceValidators:
    def test_validate_repo_name_returns_string(self) -> None:
        assert validate_repo_name("my-repo") == "my-repo"

    def test_validate_snapshot_id_returns_string(self) -> None:
        assert validate_snapshot_id("snap:repo:rev") == "snap:repo:rev"

    def test_validate_artifact_key_returns_string(self) -> None:
        assert validate_artifact_key("key-1") == "key-1"

    @pytest.mark.negative
    def test_validate_repo_name_propagates_error(self) -> None:
        with pytest.raises(KernelError):
            validate_repo_name("bad/repo")

    @pytest.mark.negative
    def test_validate_snapshot_id_propagates_error(self) -> None:
        with pytest.raises(KernelError):
            validate_snapshot_id("no-prefix")

    @pytest.mark.negative
    def test_validate_artifact_key_propagates_error(self) -> None:
        with pytest.raises(KernelError):
            validate_artifact_key("bad key")
