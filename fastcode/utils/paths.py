"""Domain-independent path and extension utilities."""

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
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]  # noqa: S324


def get_language_from_extension(ext: str) -> str:
    """Get programming language from extension."""
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".cpp": "cpp",
        ".c": "c",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
    }
    return language_map.get(ext.lower(), "unknown")
