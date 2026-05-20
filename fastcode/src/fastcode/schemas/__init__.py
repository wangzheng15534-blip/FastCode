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
from fastcode.schemas.config import (
    CacheConfig,
    EmbeddingConfig,
    EvaluationConfig,
    FastCodeConfig,
    GenerationConfig,
    LoggingConfig,
    ParserConfig,
    QueryConfig,
    RepositoryConfig,
    RetrievalConfig,
    StorageConfig,
    VectorStoreConfig,
    config_from_mapping,
    config_to_dict,
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
    QuerySourceRecord,
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
        DiagnosticBundleResponse,
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

_API_NAMES = {
    "DeleteReposRequest",
    "DiagnosticBundleResponse",
    "IndexMultipleRequest",
    "IndexRunRequest",
    "IndexRunResponse",
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
    "CacheConfig",
    "DeleteReposRequest",
    "DiagnosticBundleResponse",
    "ElementFilter",
    "EmbeddingConfig",
    "EvaluationConfig",
    "FastCodeConfig",
    "FileAnalysis",
    "FusionConfig",
    "FusionWeights",
    "GenerationConfig",
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
    "IndexRunResponse",
    "IterationConfig",
    "IterationHistoryEntry",
    "IterationMetrics",
    "IterationState",
    "LoadRepositoriesRequest",
    "LoadRepositoryRequest",
    "LoggingConfig",
    "NewSessionResponse",
    "ParserConfig",
    "ProjectionBuildRequest",
    "QueryConfig",
    "QueryRequest",
    "QueryResponse",
    "QuerySnapshotRequest",
    "QuerySourceRecord",
    "RepoStructure",
    "RepositoryConfig",
    "RetrievalChannelOutput",
    "RetrievalConfig",
    "RoundResult",
    "ScipKind",
    "ScipRole",
    "SnapshotRecord",
    "SourceRef",
    "StatusResponse",
    "StorageConfig",
    "ToolCall",
    "VectorStoreConfig",
    "config_from_mapping",
    "config_to_dict",
]
