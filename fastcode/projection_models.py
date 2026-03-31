"""
Projection-layer models for L0/L1/L2 artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProjectionScope:
    scope_kind: str
    snapshot_id: str
    scope_key: str
    query: Optional[str] = None
    target_id: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope_kind": self.scope_kind,
            "snapshot_id": self.snapshot_id,
            "scope_key": self.scope_key,
            "query": self.query,
            "target_id": self.target_id,
            "filters": self.filters,
        }


@dataclass
class ProjectionBuildResult:
    projection_id: str
    snapshot_id: str
    scope_kind: str
    scope_key: str
    l0: Dict[str, Any]
    l1: Dict[str, Any]
    l2_index: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "projection_id": self.projection_id,
            "snapshot_id": self.snapshot_id,
            "scope_kind": self.scope_kind,
            "scope_key": self.scope_key,
            "l0": self.l0,
            "l1": self.l1,
            "l2_index": self.l2_index,
            "chunks": self.chunks,
            "warnings": self.warnings,
            "created_at": self.created_at,
        }

