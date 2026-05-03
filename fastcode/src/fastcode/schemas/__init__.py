"""fastcode.schemas - shared API models, frozen dataclasses, and IR types."""

from typing import TYPE_CHECKING, Any

from fastcode.ir.types import (
    IRAttachment,
    IRDocument,
    IREdge,
    IROccurrence,
    IRSnapshot,
    IRSymbol,
)
from fastcode.schemas.core_types import (
    ElementFilter,
    FileAnalysis,
    FusionConfig,
    FusionWeights,
    GenerationInput,
    GenerationResult,
    Hit,
    IterationConfig,
    IterationHistoryEntry,
    IterationMetrics,
    IterationState,
    RepoStructure,
    RetrievalChannelOutput,
    RoundResult,
    ScipKind,
    ScipRole,
    SnapshotRecord,
    SourceRef,
    ToolCall,
)

if TYPE_CHECKING:
    from fastcode.schemas.api import (
        DeleteReposRequest,
        IndexMultipleRequest,
        IndexRunRequest,
        LoadRepositoriesRequest,
        LoadRepositoryRequest,
        NewSessionResponse,
        ProjectionBuildRequest,
        QueryRequest,
        QueryResponse,
        QuerySnapshotRequest,
        StatusResponse,
    )

_API_NAMES = {
    "DeleteReposRequest",
    "IndexMultipleRequest",
    "IndexRunRequest",
    "LoadRepositoriesRequest",
    "LoadRepositoryRequest",
    "NewSessionResponse",
    "ProjectionBuildRequest",
    "QueryRequest",
    "QueryResponse",
    "QuerySnapshotRequest",
    "StatusResponse",
}


def __getattr__(name: str) -> Any:
    """Load Pydantic API schemas only when API callers ask for them."""
    if name in _API_NAMES:
        from fastcode.schemas import api

        value = getattr(api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DeleteReposRequest",
    "ElementFilter",
    "FileAnalysis",
    "FusionConfig",
    "FusionWeights",
    "GenerationInput",
    "GenerationResult",
    "Hit",
    "IRAttachment",
    "IRDocument",
    "IREdge",
    "IROccurrence",
    "IRSnapshot",
    "IRSymbol",
    "IndexMultipleRequest",
    "IndexRunRequest",
    "IterationConfig",
    "IterationHistoryEntry",
    "IterationMetrics",
    "IterationState",
    "LoadRepositoriesRequest",
    "LoadRepositoryRequest",
    "NewSessionResponse",
    "ProjectionBuildRequest",
    "QueryRequest",
    "QueryResponse",
    "QuerySnapshotRequest",
    "RepoStructure",
    "RetrievalChannelOutput",
    "RoundResult",
    "ScipKind",
    "ScipRole",
    "SnapshotRecord",
    "SourceRef",
    "StatusResponse",
    "ToolCall",
]
