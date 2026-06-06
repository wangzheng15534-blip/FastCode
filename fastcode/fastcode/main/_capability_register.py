"""Capability registrations for the FastCode project.

Importing this module populates the CapabilityRegistry with all known
capabilities and their lifecycle stages.  Import once at composition-root
bootstrap.
"""

from __future__ import annotations

from fastcode.common.feature_lifecycle import (
    CapabilityRegistry,
    CapabilitySpec,
    CapabilityStage,
)

_EXPERIMENTAL: list[dict] = [
    dict(
        name="direct_indexing",
        config_key="indexing.allow_direct_index",
        description="Alternative indexing path that bypasses snapshot persistence",
    ),
    dict(
        name="multi_repo_direct_indexing",
        config_key="indexing.allow_direct_index",
        description="Multi-repo variant of the direct indexing path",
    ),
    dict(
        name="agency_mode",
        config_key="retrieval.enable_agency_mode",
        description="Agent-based retrieval orchestration",
    ),
    dict(
        name="multi_turn_generation",
        config_key="generation.enable_multi_turn",
        description="Multi-turn conversation support with session tracking",
    ),
    dict(
        name="docs_integration",
        config_key="docs_integration.enabled",
        description="Design document ingestion and overlay on code graph",
    ),
    dict(
        name="hierarchical_leiden",
        config_key="projection.hierarchical_leiden_enabled",
        description="Hierarchical variant of Leiden graph clustering",
    ),
    dict(
        name="adaptive_fusion",
        config_key="retrieval.adaptive_fusion",
        description="Adaptive retrieval score fusion tuning",
    ),
    dict(
        name="ladybug_graph",
        config_key="graph.ladybug.enabled",
        description="LadybugDB graph storage backend for document overlays",
    ),
    dict(
        name="mcp_server",
        description="MCP protocol server transport (stdio/SSE)",
    ),
]

_STABLE: list[dict] = [
    dict(
        name="two_stage_retrieval",
        config_key="retrieval.enable_two_stage_retrieval",
        description="Two-stage retrieval with coarse-then-fine ranking",
    ),
    dict(
        name="leiden_clustering",
        config_key="projection.enable_leiden",
        description="Leiden graph clustering for projection maps",
    ),
    dict(
        name="llm_projection",
        config_key="projection.llm_enabled",
        description="LLM-assisted graph cluster labeling",
    ),
    dict(
        name="scip_extraction",
        description="SCIP-protocol based symbol extraction",
    ),
    dict(
        name="graph_builder_fallback",
        config_key="retrieval.allow_graph_builder_fallback",
        description="Fallback from IR graph to builder-based graph expansion",
    ),
    dict(
        name="ollama_batch",
        config_key="embedding.ollama_batch_enabled",
        description="Batched embedding requests to Ollama",
    ),
    dict(
        name="workspace_copy_cache",
        config_key="repository.workspace_copy_cache_enabled",
        description="Cache for workspace copies during repository loading",
    ),
]


def register_all_capabilities() -> None:
    """Populate the global CapabilityRegistry with all known capabilities."""
    for entry in _EXPERIMENTAL:
        CapabilityRegistry.register(
            CapabilitySpec(
                stage=CapabilityStage.EXPERIMENTAL,
                **entry,  # type: ignore[arg-type]
            )
        )
    for entry in _STABLE:
        CapabilityRegistry.register(
            CapabilitySpec(
                stage=CapabilityStage.STABLE,
                **entry,  # type: ignore[arg-type]
            )
        )
