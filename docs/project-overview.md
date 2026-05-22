# FastCode Project Overview

## What is FastCode?

FastCode is a **branch-aware code knowledge graph for AI agents**. Based on the HKUDS scouting-first framework (arxiv:2603.01012v2), it pre-builds code understanding into a queryable knowledge graph so AI coding agents (Claude Code, Cursor, Windsurf) get precise answers fast.

**Problem it solves:** AI agents explore codebases by reading files one by one — slow, shallow, miss structural relationships. FastCode pre-indexes code relationships into a graph, letting agents query "how does X connect to Y?" instead of grepping for keywords.

## Quick Reference

| Property | Value |
|----------|-------|
| **Type** | Monolith (Python backend) |
| **Language** | Python 3.11+ |
| **Architecture** | Three-layer (canonical facts → derived graph → retrieval indices) |
| **Version** | 2.0.0 |
| **License** | MIT |
| **Package Manager** | uv |
| **Total LOC** | ~132K (incl. tests, vendored nanobot) |
| **Test LOC** | ~18K (61 test files) |

## Technology Summary

| Category | Technology |
|----------|-----------|
| API | FastAPI (port 8001), Flask (port 5777), MCP (stdio/SSE) |
| CLI | Click |
| Storage | PostgreSQL (canonical facts, vectors, projections), TerminusDB (graph) |
| Retrieval | pgvector HNSW, GIN BM25, ripgrep |
| Graph | NetworkX (Leiden, Steiner, arborescence, PageRank) |
| Parsing | Tree-sitter (10 languages), SCIP (8 languages) |
| Embeddings | Ollama / sentence-transformers |
| LLM | OpenAI, Anthropic |
| Agent Framework | Nanobot (vendored, multi-channel gateway) |
| Testing | pytest, hypothesis, syrupy, pytest-benchmark |

## Architecture Type

**Pattern:** Three-layer data pipeline with precision-anchored extraction (tree-sitter skeleton + SCIP anchors)

1. **Canonical IR** (deepest truth) — extracted per snapshot, immutable
2. **Derived Graph** (TerminusDB) — materialized from facts, branch-aware, refreshable
3. **Retrieval Indices** (PostgreSQL + ripgrep) — semantic + keyword + exact code search

**Key principle:** Facts are truth, graph is derived view. No cross-layer fusion.

## Repository Structure

```
FastCode/
├── fastcode/          # Core library (45 modules, ~17K LOC)
├── api.py             # FastAPI REST API (1,014 LOC)
├── web_app.py         # Flask web UI (831 LOC)
├── mcp_server.py      # MCP server (759 LOC)
├── main.py            # CLI entry point (945 LOC)
├── nanobot/           # Vendored AI agent framework
├── tests/             # Test suite (61 files, ~18K LOC)
├── config/            # Runtime configuration
├── docs/              # Documentation
├── pyproject.toml     # Project metadata + tool configs
├── Dockerfile         # Python 3.12-slim container
└── docker-compose.yml # FastCode + Nanobot services
```

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](./architecture.md) | Three-layer architecture, precision-anchored extraction, projection pipeline |
| [API Contracts](./api-contracts.md) | REST API, MCP tools, CLI commands |
| [Data Models](./data-models.md) | Canonical IR, SCIP models, projection models, merge strategy |
| [Source Tree](./source-tree-analysis.md) | Annotated directory tree, critical folders, entry points |
| [Development Guide](./development-guide.md) | Setup, running, testing, code quality |
| [Deployment Guide](./deployment-guide.md) | Docker, environment config, production notes |
| [Review Context](./review-response-context.md) | Architecture decisions, algorithm research, assumptions |
