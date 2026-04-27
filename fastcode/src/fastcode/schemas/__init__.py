"""fastcode.schemas — all frozen dataclasses and IR types."""

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

__all__ = [
    "ElementFilter",
    "FileAnalysis",
    "FusionConfig",
    "FusionWeights",
    "GenerationInput",
    "GenerationResult",
    "Hit",
    "IterationConfig",
    "IterationHistoryEntry",
    "IterationMetrics",
    "IterationState",
    "RepoStructure",
    "RetrievalChannelOutput",
    "RoundResult",
    "ScipKind",
    "ScipRole",
    "SnapshotRecord",
    "SourceRef",
    "ToolCall",
]

from fastcode.schemas.ir import (
    IRAttachment,
    IRDocument,
    IREdge,
    IROccurrence,
    IRSnapshot,
    IRSymbol,
)

__all__ += [
    "IRAttachment",
    "IRDocument",
    "IREdge",
    "IROccurrence",
    "IRSnapshot",
    "IRSymbol",
]
