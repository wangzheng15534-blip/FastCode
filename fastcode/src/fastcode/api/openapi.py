"""OpenAPI schema model registry for the HTTP boundary."""

from __future__ import annotations

from fastcode.api.inbound import (
    AgentContextHandoffRequest,
    ContextActivationRequest,
    DeleteReposRequest,
    ExpandContextBundleRefRequest,
    ExpandContextRefRequest,
    IndexMultipleRequest,
    IndexRunRequest,
    LoadRepositoriesRequest,
    LoadRepositoryRequest,
    ProjectionBuildRequest,
    QueryRequest,
    QuerySnapshotRequest,
)
from fastcode.api.outbound import (
    DiagnosticBundleResponse,
    IndexRunResponse,
    NewSessionResponse,
    QueryResponse,
    StatusResponse,
)

OPENAPI_SCHEMA_MODELS = (
    AgentContextHandoffRequest,
    ContextActivationRequest,
    DeleteReposRequest,
    DiagnosticBundleResponse,
    ExpandContextBundleRefRequest,
    ExpandContextRefRequest,
    IndexMultipleRequest,
    IndexRunRequest,
    IndexRunResponse,
    LoadRepositoriesRequest,
    LoadRepositoryRequest,
    NewSessionResponse,
    ProjectionBuildRequest,
    QueryRequest,
    QueryResponse,
    QuerySnapshotRequest,
    StatusResponse,
)
