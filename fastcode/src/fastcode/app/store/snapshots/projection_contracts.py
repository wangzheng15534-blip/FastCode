"""Projection store persistence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


def _string_list_payload(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in cast(list[Any], value)]
    if isinstance(value, tuple):
        return [str(item) for item in cast(tuple[Any, ...], value)]
    return []


def _string_key_mapping_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in cast(dict[Any, Any], value).items()}


@dataclass(frozen=True)
class ProjectionDirtyScopeRecord:
    snapshot_id: str
    scope_kind: str
    scope_key: str
    dirty_paths: list[str]
    dirty_units: list[str]
    dirty_package_roots: list[str]
    dirty_reason: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "scope_kind": self.scope_kind,
            "scope_key": self.scope_key,
            "dirty_paths": list(self.dirty_paths),
            "dirty_units": list(self.dirty_units),
            "dirty_package_roots": list(self.dirty_package_roots),
            "dirty_reason": self.dirty_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectionDirtyScopeRecord:
        return cls(
            snapshot_id=str(data.get("snapshot_id") or ""),
            scope_kind=str(data.get("scope_kind") or ""),
            scope_key=str(data.get("scope_key") or ""),
            dirty_paths=_string_list_payload(data.get("dirty_paths")),
            dirty_units=_string_list_payload(data.get("dirty_units")),
            dirty_package_roots=_string_list_payload(data.get("dirty_package_roots")),
            dirty_reason=str(data.get("dirty_reason") or ""),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass(frozen=True)
class ProjectionBuildRecord:
    projection_id: str
    snapshot_id: str
    scope_kind: str
    scope_key: str
    params_hash: str
    status: str
    warnings: list[str]
    created_at: str
    updated_at: str
    query: str | None
    target_id: str | None
    filters: dict[str, Any]
    coverage_paths: list[str]
    coverage_nodes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_id": self.projection_id,
            "snapshot_id": self.snapshot_id,
            "scope_kind": self.scope_kind,
            "scope_key": self.scope_key,
            "params_hash": self.params_hash,
            "status": self.status,
            "warnings": list(self.warnings),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "query": self.query,
            "target_id": self.target_id,
            "filters": dict(self.filters),
            "coverage_paths": list(self.coverage_paths),
            "coverage_nodes": list(self.coverage_nodes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectionBuildRecord:
        return cls(
            projection_id=str(data.get("projection_id") or ""),
            snapshot_id=str(data.get("snapshot_id") or ""),
            scope_kind=str(data.get("scope_kind") or ""),
            scope_key=str(data.get("scope_key") or ""),
            params_hash=str(data.get("params_hash") or ""),
            status=str(data.get("status") or ""),
            warnings=_string_list_payload(data.get("warnings")),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            query=str(data["query"]) if data.get("query") is not None else None,
            target_id=(
                str(data["target_id"]) if data.get("target_id") is not None else None
            ),
            filters=_string_key_mapping_payload(data.get("filters")),
            coverage_paths=_string_list_payload(data.get("coverage_paths")),
            coverage_nodes=_string_list_payload(data.get("coverage_nodes")),
        )
