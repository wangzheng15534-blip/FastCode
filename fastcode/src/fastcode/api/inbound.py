"""HTTP inbound request DTOs for FastCode API surfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, Field


def _mapping_proxy_or_none(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return MappingProxyType({str(key): item for key, item in value.items()})


def _string_tuple(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(str(item) for item in value)


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


class AgentContextHandoffRequest(BaseModel):
    session_id: str = Field(..., description="Session ID containing working memory")
    turn_number: int | None = Field(
        None,
        description="Optional turn number; defaults to the latest turn in the session",
    )
    mode: str = Field("delegate", description="Handoff mode, for example delegate")


class ExpandContextRefRequest(BaseModel):
    session_id: str = Field(..., description="Session ID containing working memory")
    turn_number: int = Field(..., description="Turn number to inspect")
    ref_id: str = Field(..., description="Evidence ref ID, such as e1")
    depth: str = Field("L2", description="Requested expansion depth")


class ExpandContextBundleRefRequest(BaseModel):
    ref_id: str = Field(..., description="Evidence ref ID, such as e1")
    session_id: str | None = Field(
        None, description="Session ID containing the context bundle"
    )
    turn_number: int | None = Field(None, description="Optional turn number")
    bundle_id: str | None = Field(None, description="Direct context bundle ID")
    depth: str = Field("L2", description="Requested expansion depth")


class ContextActivationRequest(BaseModel):
    session_id: str | None = Field(
        None, description="Session ID containing the context bundle"
    )
    turn_number: int | None = Field(None, description="Optional turn number")
    bundle_id: str | None = Field(None, description="Direct context bundle ID")
    active_ref_ids: list[str] | None = Field(
        None, description="Evidence refs to activate"
    )
    active_fact_ids: list[str] | None = Field(None, description="Facts to activate")
    active_hypothesis_ids: list[str] | None = Field(
        None, description="Hypotheses to activate"
    )
    reason: str | None = Field(None, description="Activation reason")


class LoadRepositoriesRequest(BaseModel):
    repo_names: list[str] = Field(
        ..., description="Repository names to load from existing indexes"
    )


class IndexMultipleRequest(BaseModel):
    sources: list[LoadRepositoryRequest] = Field(
        ..., description="Multiple repositories to load and index"
    )


class DeleteReposRequest(BaseModel):
    repo_names: list[str] = Field(..., description="Repository names to delete")
    delete_source: bool = Field(
        True, description="Also delete cloned source code in repos/"
    )


@dataclass(frozen=True)
class LoadRepositoryCommand:
    source: str
    is_url: bool | None = None


@dataclass(frozen=True)
class IndexRunCommand:
    source: str
    is_url: bool | None = None
    ref: str | None = None
    commit: str | None = None
    force: bool = False
    publish: bool = True
    enable_scip: bool = True
    scip_artifact_path: str | None = None


@dataclass(frozen=True)
class RepositoryQueryRequestRecord:
    question: str
    filters: Mapping[str, Any] | None = None
    repo_filter: tuple[str, ...] = ()
    multi_turn: bool = False
    session_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "filters", _mapping_proxy_or_none(self.filters))
        object.__setattr__(self, "repo_filter", _string_tuple(self.repo_filter))


@dataclass(frozen=True)
class SnapshotQueryRequestRecord:
    question: str
    snapshot_id: str | None = None
    repo_name: str | None = None
    ref_name: str | None = None
    filters: Mapping[str, Any] | None = None
    repo_filter: tuple[str, ...] = ()
    multi_turn: bool = False
    session_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "filters", _mapping_proxy_or_none(self.filters))
        object.__setattr__(self, "repo_filter", _string_tuple(self.repo_filter))


def map_load_repository_request(request: LoadRepositoryRequest) -> LoadRepositoryCommand:
    return LoadRepositoryCommand(
        source=request.source,
        is_url=request.is_url,
    )


def map_index_run_request(request: IndexRunRequest) -> IndexRunCommand:
    return IndexRunCommand(
        source=request.source,
        is_url=request.is_url,
        ref=request.ref,
        commit=request.commit,
        force=request.force,
        publish=request.publish,
        enable_scip=request.enable_scip,
        scip_artifact_path=request.scip_artifact_path,
    )


def map_repository_query_request(
    request: QueryRequest,
) -> RepositoryQueryRequestRecord:
    return RepositoryQueryRequestRecord(
        question=request.question,
        filters=request.filters,
        repo_filter=tuple(request.repo_filter or ()),
        multi_turn=request.multi_turn,
        session_id=request.session_id,
    )


def map_snapshot_query_request(
    request: QueryRequest | QuerySnapshotRequest,
) -> SnapshotQueryRequestRecord:
    repo_filter: Sequence[str] | None = None
    if isinstance(request, QueryRequest):
        repo_filter = request.repo_filter
    return SnapshotQueryRequestRecord(
        question=request.question,
        snapshot_id=request.snapshot_id,
        repo_name=request.repo_name,
        ref_name=request.ref_name,
        filters=request.filters,
        repo_filter=tuple(repo_filter or ()),
        multi_turn=request.multi_turn,
        session_id=request.session_id,
    )
