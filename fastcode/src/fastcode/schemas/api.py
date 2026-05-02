"""HTTP request/response schemas for FastCode API and web surfaces."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


def _empty_repo_records() -> list[dict[str, Any]]:
    return []


class LoadRepositoryRequest(BaseModel):
    source: str = Field(..., description="Repository URL or local path")
    is_url: bool | None = Field(
        None,
        description="True if source is URL, False if local path. If omitted, auto-detect.",
    )


class IndexRunRequest(BaseModel):
    source: str = Field(..., description="Repository URL or local path")
    is_url: bool | None = Field(None, description="Explicit source type override")
    ref: str | None = Field(None, description="Branch/tag/ref to index")
    commit: str | None = Field(None, description="Commit hash to index")
    force: bool = Field(
        False, description="Force re-index even if snapshot already exists"
    )
    publish: bool = Field(True, description="Publish manifest after indexing")
    enable_scip: bool = Field(True, description="Enable SCIP extraction path")
    scip_artifact_path: str | None = Field(
        None, description="Optional pre-built SCIP artifact path"
    )


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask about the repository")
    snapshot_id: str | None = Field(None, description="Direct snapshot ID")
    repo_name: str | None = Field(
        None, description="Repository name (for ref resolution)"
    )
    ref_name: str | None = Field(
        None, description="Branch/ref name (for ref resolution)"
    )
    filters: dict[str, Any] | None = Field(None, description="Optional filters")
    repo_filter: list[str] | None = Field(
        None, description="Repository names to search"
    )
    multi_turn: bool = Field(False, description="Enable multi-turn mode")
    session_id: str | None = Field(
        None, description="Session ID for multi-turn dialogue"
    )


class QuerySnapshotRequest(BaseModel):
    question: str = Field(..., description="Question to ask")
    snapshot_id: str | None = Field(None, description="Direct snapshot ID")
    repo_name: str | None = Field(
        None, description="Repository name (when resolving by ref)"
    )
    ref_name: str | None = Field(
        None, description="Branch/ref name (when resolving by ref)"
    )
    filters: dict[str, Any] | None = Field(
        None, description="Optional retrieval filters"
    )
    multi_turn: bool = Field(False, description="Enable multi-turn mode")
    session_id: str | None = Field(
        None, description="Session ID for multi-turn dialogue"
    )


class ProjectionBuildRequest(BaseModel):
    scope_kind: str = Field(
        ..., description="Projection scope: snapshot | query | entity"
    )
    snapshot_id: str | None = Field(None, description="Direct snapshot ID")
    repo_name: str | None = Field(
        None, description="Repository name (for ref resolution)"
    )
    ref_name: str | None = Field(
        None, description="Branch/ref name (for ref resolution)"
    )
    query: str | None = Field(
        None, description="Query text for query-scoped projection"
    )
    target_id: str | None = Field(
        None, description="Entity ID/path for entity-scoped projection"
    )
    filters: dict[str, Any] | None = Field(None, description="Optional scope filters")
    force: bool = Field(False, description="Force regeneration even when cached")


class QueryResponse(BaseModel):
    answer: str
    query: str
    context_elements: int
    sources: list[dict[str, Any]]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    session_id: str | None = None


class LoadRepositoriesRequest(BaseModel):
    repo_names: list[str] = Field(
        ..., description="Repository names to load from existing indexes"
    )


class IndexMultipleRequest(BaseModel):
    sources: list[LoadRepositoryRequest] = Field(
        ..., description="Multiple repositories to load and index"
    )


class NewSessionResponse(BaseModel):
    session_id: str


class DeleteReposRequest(BaseModel):
    repo_names: list[str] = Field(..., description="Repository names to delete")
    delete_source: bool = Field(
        True, description="Also delete cloned source code in repos/"
    )


class StatusResponse(BaseModel):
    status: str
    repo_loaded: bool
    repo_indexed: bool
    repo_info: dict[str, Any]
    graph_backend: str | None = None
    storage_backend: str | None = None
    retrieval_backend: str | None = None
    available_repositories: list[dict[str, Any]] = Field(
        default_factory=_empty_repo_records
    )
    loaded_repositories: list[dict[str, Any]] = Field(
        default_factory=_empty_repo_records
    )
