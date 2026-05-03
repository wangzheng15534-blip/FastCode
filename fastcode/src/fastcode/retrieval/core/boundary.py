"""Boundary conversion helpers.

Explicit translation between external API dicts and core frozen dataclasses.
Every field is mapped by name — no **kwargs, no from_orm, no model_dump.

Three Golden Rules enforced:
1. Pydantic Stops at the Door -- no pydantic imports in core/
2. Database Trusts Dataclasses -- effects/db.py returns frozen dataclasses
3. Explicit Translation -- explicit field mapping, no **kwargs unpacking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastcode.schemas.core_types import Hit


@dataclass(frozen=True)
class CoreQueryInput:
    """Canonical query input accepted by the core layer."""

    question: str
    repo_name: str | None = None
    branch: str | None = None
    snapshot_id: str | None = None
    session_id: str | None = None


def query_request_to_core(request: dict[str, Any]) -> CoreQueryInput:
    """Convert API request dict to core query input.

    Explicit field mapping -- no **kwargs.
    """
    return CoreQueryInput(
        question=request["question"],
        repo_name=request.get("repo_name"),
        branch=request.get("branch"),
        snapshot_id=request.get("snapshot_id"),
        session_id=request.get("session_id"),
    )


def hit_to_response(hit: Hit) -> dict[str, Any]:
    """Convert core Hit to API response dict.

    Explicit field mapping -- no model_dump or dict().
    """
    return {
        "id": hit.element_id,
        "type": hit.element_type,
        "name": hit.element_name,
        "score": hit.score,
        "source": hit.source,
    }
