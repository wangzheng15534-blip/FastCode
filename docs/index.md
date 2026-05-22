# FastCode Documentation Index

## Project Overview

- **Type:** Monolith (single cohesive codebase)
- **Primary Language:** Python 3.11+
- **Architecture:** Three-layer (canonical facts → derived graph → retrieval indices)
- **Based on:** HKUDS scouting-first framework (arxiv:2603.01012v2)

## Quick Reference

- **Tech Stack:** FastAPI, Flask, PostgreSQL, TerminusDB, NetworkX, Tree-sitter, SCIP, pgvector
- **Entry Points:** api.py (REST :8001), web_app.py (Web :5777), mcp_server.py (MCP stdio/SSE), main.py (CLI)
- **Architecture Pattern:** Precision-anchored extraction (tree-sitter skeleton + SCIP anchors) → canonical IR → derived graph → adaptive retrieval

## Generated Documentation

- [Project Overview](./project-overview.md) — What FastCode is, tech summary, architecture type
- [Architecture](./architecture.md) — Three-layer architecture, precision-anchored extraction, projection pipeline, module organization
- [API Contracts](./api-contracts.md) — REST API (35+ endpoints), MCP tools (11), CLI commands (14)
- [Data Models](./data-models.md) — Canonical IR (6 models), SCIP models, projection models, merge strategy
- [Source Tree Analysis](./source-tree-analysis.md) — Annotated directory tree, critical folders, entry points, integration points
- [Development Guide](./development-guide.md) — Setup, running, testing, code quality, common tasks
- [Deployment Guide](./deployment-guide.md) — Docker, environment config, production notes

## Existing Documentation

- [Review Response Context](./review-response-context.md) — Architecture & algorithm review, 7 core algorithms, assumptions, research agenda
- [Architecture Decisions](../_bmad-output/planning-artifacts/architecture.md) — Detailed architecture decision document with implementation status
- [Audit Branch Index Pipeline](./audit-branch-index-pipeline.md) — Branch indexing pipeline audit
- [Design Doc Parsing Integration](./design-doc-parsing-integration.md) — Document parsing integration design
- [Design Graph Algorithm Gaps](./design-graph-algorithm-gaps.md) — Graph algorithm gap analysis

## Getting Started

```bash
# Install
uv pip install -e ".[dev]"

# Configure
cp env.example .env  # Set OPENAI_API_KEY, MODEL, BASE_URL

# Run
uv run python api.py --host 0.0.0.0 --port 8001   # REST API
uv run python mcp_server.py                          # MCP server
uv run python main.py interactive -p ./repos/myrepo  # CLI

# Test
uv run pytest tests/ -v
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Canonical IR** | Deepest truth — extracted per snapshot, immutable, SCIP identity canonical |
| **Derived Graph** | Materialized from facts for fast traversal, branch-aware, refreshable |
| **Precision-Anchored** | Tree-sitter skeleton (always) + SCIP anchors (when available, 8 languages) — one pipeline, not two |
| **Projection** | L0/L1/L2 summaries from graph algorithms (Leiden, arborescence) |
| **Hierarchical Fusion** | Intra-collection score fusion → cross-collection weighted RRF → doc→code projection (no flat RRF) |
| **Doc→Code Projection** | Docs project grounded priors into code-space via noisy-or — docs never enter graph as nodes |
| **Session Prefix** | L0/L1 projection in system prompt — agent knows architecture before first query |
