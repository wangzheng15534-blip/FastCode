"""HTTP outbound response records and DTOs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, Field


def _empty_payload() -> dict[str, Any]:
    return {}


def _empty_diagnostic_records() -> list[ResolverDiagnosticDTO]:
    return []


def _empty_mapping_payload_records() -> list[dict[str, Any]]:
    return []


def _mapping_proxy(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({str(key): item for key, item in (value or {}).items()})


class ApiStatus(StrEnum):
    SUCCESS = "success"
    READY = "ready"
    NOT_READY = "not_ready"


@dataclass(frozen=True)
class OpenMappingRecord:
    """Open JSON payload intentionally preserved at the API boundary."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _mapping_proxy(self.payload))


@dataclass(frozen=True)
class QuerySourceRecord:
    """Typed source citation record used at API/query response boundaries."""

    repository: str
    file: str
    name: str
    source_type: str
    lines: str
    start_line: int
    end_line: int
    score: float


@dataclass(frozen=True)
class ResolverDiagnosticRecord:
    name: str
    source: str
    status: str
    reason: str | None = None
    warnings: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "metrics", _mapping_proxy(self.metrics))


@dataclass(frozen=True)
class IndexRunResponseRecord:
    status: ApiStatus
    result: Mapping[str, Any] = MappingProxyType({})
    index_status: str | None = None
    run_id: str | None = None
    repo_name: str | None = None
    snapshot_id: str | None = None
    artifact_key: str | None = None
    warnings: tuple[str, ...] = ()
    pipeline_layers: tuple[OpenMappingRecord, ...] = ()
    pipeline_metrics: Mapping[str, Any] = MappingProxyType({})
    resolver_diagnostics: tuple[ResolverDiagnosticRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ApiStatus(self.status))
        object.__setattr__(self, "result", _mapping_proxy(self.result))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(
            self, "pipeline_layers", tuple(self.pipeline_layers)
        )
        object.__setattr__(
            self, "pipeline_metrics", _mapping_proxy(self.pipeline_metrics)
        )
        object.__setattr__(
            self, "resolver_diagnostics", tuple(self.resolver_diagnostics)
        )


@dataclass(frozen=True)
class QueryResponseRecord:
    answer: str
    query: str
    context_elements: int
    sources: tuple[QuerySourceRecord, ...]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    session_id: str | None = None
    turn_number: int | None = None


@dataclass(frozen=True)
class NewSessionRecord:
    session_id: str


@dataclass(frozen=True)
class StatusResponseRecord:
    status: ApiStatus
    repo_loaded: bool
    repo_indexed: bool
    repo_info: Mapping[str, Any]
    graph_expansion_backend: str | None = None
    storage_backend: str | None = None
    retrieval_backend: str | None = None
    available_repositories: tuple[OpenMappingRecord, ...] = ()
    loaded_repositories: tuple[OpenMappingRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ApiStatus(self.status))
        object.__setattr__(self, "repo_info", _mapping_proxy(self.repo_info))
        object.__setattr__(
            self, "available_repositories", tuple(self.available_repositories)
        )
        object.__setattr__(
            self, "loaded_repositories", tuple(self.loaded_repositories)
        )


@dataclass(frozen=True)
class DiagnosticBundleRecord:
    status: ApiStatus
    bundle: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ApiStatus(self.status))
        object.__setattr__(self, "bundle", _mapping_proxy(self.bundle))


class OpenMappingDTO(BaseModel):
    payload: dict[str, Any] = Field(default_factory=_empty_payload)


class QuerySourceDTO(BaseModel):
    repository: str
    repo: str
    file: str
    name: str
    type: str
    lines: str
    start_line: int
    end_line: int
    score: float


def _empty_source_records() -> list[QuerySourceDTO]:
    return []


class ResolverDiagnosticDTO(BaseModel):
    name: str
    source: str
    status: str
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=_empty_payload)


class IndexRunResponse(BaseModel):
    status: ApiStatus
    result: dict[str, Any] = Field(default_factory=_empty_payload)
    index_status: str | None = None
    run_id: str | None = None
    repo_name: str | None = None
    snapshot_id: str | None = None
    artifact_key: str | None = None
    warnings: list[str] = Field(default_factory=list)
    pipeline_layers: list[dict[str, Any]] = Field(
        default_factory=_empty_mapping_payload_records
    )
    pipeline_metrics: dict[str, Any] = Field(default_factory=_empty_payload)
    resolver_diagnostics: list[ResolverDiagnosticDTO] = Field(
        default_factory=_empty_diagnostic_records
    )


class QueryResponse(BaseModel):
    answer: str
    query: str
    context_elements: int
    sources: list[QuerySourceDTO] = Field(default_factory=_empty_source_records)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    session_id: str | None = None
    turn_number: int | None = None


class NewSessionResponse(BaseModel):
    session_id: str


class StatusResponse(BaseModel):
    status: ApiStatus
    repo_loaded: bool
    repo_indexed: bool
    repo_info: dict[str, Any] = Field(default_factory=_empty_payload)
    graph_expansion_backend: str | None = None
    storage_backend: str | None = None
    retrieval_backend: str | None = None
    available_repositories: list[dict[str, Any]] = Field(
        default_factory=_empty_mapping_payload_records
    )
    loaded_repositories: list[dict[str, Any]] = Field(
        default_factory=_empty_mapping_payload_records
    )


class DiagnosticBundleResponse(BaseModel):
    status: ApiStatus
    bundle: dict[str, Any] = Field(default_factory=_empty_payload)
