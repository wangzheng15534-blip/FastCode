"""Pure snapshot logic — extracted from main.py."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def projection_scope_key(
    scope_kind: str,
    snapshot_id: str,
    query: str | None,
    target_id: str | None,
    filters: dict[str, Any] | None,
) -> str:
    """Compute a deterministic hash key for a projection scope."""
    base = {
        "scope_kind": scope_kind,
        "snapshot_id": snapshot_id,
        "query": query or "",
        "target_id": target_id or "",
        "filters": filters or {},
    }
    payload = json.dumps(base, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


def projection_params_hash(scope_dict: dict[str, Any], version: str = "v1") -> str:
    """Hash projection parameters for cache key."""
    payload = json.dumps(
        {"scope": scope_dict, "projection_algo_version": version},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def extract_sources_from_elements(
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract source information from retrieved elements."""
    sources: list[dict[str, Any]] = []
    for elem_data in elements:
        elem = elem_data.get("element", {})
        sources.append(
            {
                "file": elem.get("relative_path", ""),
                "repo": elem.get("repo_name", ""),
                "type": elem.get("type", ""),
                "name": elem.get("name", ""),
                "start_line": elem.get("start_line", 0),
                "end_line": elem.get("end_line", 0),
            }
        )
    return sources
