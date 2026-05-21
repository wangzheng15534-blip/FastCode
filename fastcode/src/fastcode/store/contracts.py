"""Store-owned persistence contracts for small infrastructure adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SnapshotRecord:
    """Minimal snapshot metadata row used by low-level store adapters."""

    snapshot_id: str
    repo_name: str
    branch: str | None = None
    commit_id: str | None = None
    tree_id: str | None = None
