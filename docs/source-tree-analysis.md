# FastCode Source Tree Analysis

## Project Overview

| Metric | Value |
|--------|-------|
| Total Python files | 398 (incl. tests) |
| Source files (fastcode/) | ~50 |
| Test files | 61 |
| Total LOC | ~132K (incl. tests, vendored nanobot) |
| Test LOC | ~18K |
| Language | Python 3.11+ |

## Directory Tree

```
FastCode/
├── fastcode/                          # Core library package
│   ├── __init__.py                    # Package exports, FastCode class
│   ├── main.py                        # CLI entry point (Click) — 2,773 LOC
│   ├── indexer.py                     # File-level indexing pipeline
│   ├── index_run.py                   # Index run orchestration
│   ├── retriever.py                   # HybridRetriever — multi-stage RRF fusion — 1,895 LOC
│   ├── query_processor.py             # Query intent, keyword extraction, rewriting
│   ├── answer_generator.py            # LLM answer synthesis with context
│   ├── iterative_agent.py             # Multi-round retrieval agent — 3,335 LOC
│   ├── agent_tools.py                 # Tool definitions for AI agent integration
│   │
│   ├── parser.py                      # Legacy parser (tree-sitter) — 1,702 LOC
│   ├── tree_sitter_parser.py          # Tree-sitter grammar wrapper (10 languages)
│   ├── embedder.py                    # Embedding generation (Ollama / sentence-transformers)
│   ├── vector_store.py                # FAISS vector store wrapper
│   ├── loader.py                      # Repository loading and file discovery
│   ├── cache.py                       # Disk/Redis caching layer
│   ├── utils.py                       # Shared utilities
│   ├── path_utils.py                  # Path manipulation helpers
│   ├── module_resolver.py             # Python module resolution
│   ├── repo_overview.py               # Repository summary generation
│   ├── repo_selector.py               # Multi-repo selection (LLM or embedding)
│   ├── symbol_resolver.py             # Symbol name resolution
│   ├── llm_utils.py                   # LLM API helpers
│   │
│   ├── call_extractor.py              # Function call extraction — 717 LOC
│   ├── definition_extractor.py        # Definition extraction
│   ├── import_extractor.py            # Import statement extraction
│   │
│   ├── graph_builder.py               # NetworkX graph construction — 1,017 LOC
│   ├── ir_graph_builder.py            # IR → graph edge builder
│   ├── global_index_builder.py        # Cross-repository index
│   │
│   ├── semantic_ir.py                 # Canonical IR models (IRSnapshot, IRDocument, IRSymbol, etc.)
│   ├── ir_merge.py                    # Precision-anchored merge (SCIP anchors onto tree-sitter)
│   ├── ir_validators.py               # IR integrity validation
│   ├── scip_models.py                 # SCIP protobuf model wrappers
│   ├── scip_loader.py                 # SCIP protobuf file loader
│   ├── scip_indexers.py               # SCIP indexers (8 languages)
│   ├── scip_pb2.py                    # Generated protobuf bindings
│   │
│   ├── snapshot_store.py              # Snapshot persistence (SQLite/PostgreSQL) — 1,006 LOC
│   ├── snapshot_symbol_index.py       # In-memory symbol index per snapshot
│   ├── manifest_store.py              # Branch manifest storage
│   ├── projection_transform.py        # Graph → L0/L1/L2 projections — 984 LOC
│   ├── projection_models.py           # Projection data models
│   ├── projection_store.py            # Projection persistence (PostgreSQL)
│   │
│   ├── db_runtime.py                  # PostgreSQL connection pool + query helpers
│   ├── pg_retrieval.py                # pgvector HNSW + GIN full-text search — 480 LOC
│   ├── terminus_publisher.py          # TerminusDB graph publishing (outbox pattern)
│   ├── graph_runtime.py               # LadybugDB graph overlay for docs
│   ├── doc_ingester.py                # Key document ingestion + chunking
│   ├── redo_worker.py                 # Background retry for failed index runs
│   │
│   └── adapters/                      # Extraction adapters
│       ├── __init__.py
│       ├── scip_to_ir.py              # SCIP protobuf → Canonical IR — 175 LOC
│       └── ast_to_ir.py               # Tree-sitter AST → Canonical IR — 294 LOC
│
├── api.py                             # FastAPI REST API — 1,014 LOC
├── web_app.py                         # Flask web UI — 831 LOC
├── mcp_server.py                      # MCP server (Cursor, Claude Code, Windsurf) — 759 LOC
├── main.py                            # CLI entry point — 945 LOC
│
├── nanobot/                           # Vendored AI agent framework (workspace member)
│   ├── nanobot/
│   │   ├── agent/                     # Agent core (loop, context, memory, skills)
│   │   │   └── tools/
│   │   │       └── fastcode.py        # FastCode tool bridge (fastcode_query, fastcode_load_repo)
│   │   ├── bus/                       # Event bus (events, queue)
│   │   ├── channels/                  # Multi-channel: Feishu, Telegram, Discord, Slack, WhatsApp, QQ
│   │   ├── cli/                       # CLI commands (gateway, chat)
│   │   ├── config/                    # Configuration schema + loader
│   │   ├── cron/                      # Scheduled tasks
│   │   ├── heartbeat/                 # Health monitoring
│   │   ├── providers/                 # LLM providers (LiteLLM, transcription)
│   │   ├── session/                   # Session management
│   │   ├── skills/                    # Agent skills
│   │   └── utils/                     # Helpers
│   └── bridge/src/                    # Channel bridge
│
├── tests/                             # Test suite (61 files, ~18K LOC)
│   ├── test_ir_core.py                # Core IR tests
│   ├── test_scip_*.py                 # SCIP extraction tests
│   ├── test_ast_to_ir*.py            # AST adapter tests
│   ├── test_graph_*.py               # Graph tests
│   ├── test_projection_*.py          # Projection tests
│   ├── test_snapshot_*.py            # Snapshot tests
│   ├── test_adaptive_fusion.py       # Retrieval fusion tests
│   ├── test_pg_retrieval_*.py        # PostgreSQL retrieval tests
│   ├── test_doc_ingester.py          # Doc ingestion tests
│   ├── test_redo_worker.py           # Redo worker tests
│   ├── test_e2e_*.py                 # E2E tests (Ollama + PostgreSQL)
│   ├── test_nanobot_fastcode_tools.py # Nanobot bridge tests
│   ├── bench_*.py                    # Benchmarks (ir_merge, graph_projection, validation)
│   ├── property/                     # Hypothesis property-based tests
│   └── snapshots/                    # Syrupy snapshot contract tests
│
├── config/
│   └── config.yaml                    # Default configuration (~280 lines)
│
├── scripts/
│   └── run_mutmut.sh                  # Mutation testing script
│
├── skills/
│   └── test-quality-gate/             # Custom skill: test quality gate
│       └── scripts/
│
├── docs/                              # Documentation
│   ├── review-response-context.md     # Architecture & algorithm review context
│   ├── audit-branch-index-pipeline.md
│   ├── design-doc-parsing-integration.md
│   ├── design-graph-algorithm-gaps.md
│   └── _plans/                        # Archived implementation plans
│
├── _bmad/                             # BMAD framework configuration
│   ├── bmm/                           # Module manager
│   ├── bmb/                           # Module builder
│   ├── core/                          # Core skills
│   ├── cis/                           # Creative intelligence suite
│   ├── tea/                           # Test architecture enterprise
│   └── _config/                       # IDE configs, manifests
│
├── pyproject.toml                     # Project metadata, dependencies, tool configs
├── requirements.txt                   # Pinned dependencies
├── Dockerfile                         # Python 3.12-slim container
├── docker-compose.yml                 # FastCode + Nanobot services
├── nanobot_config.json                # Nanobot agent configuration
├── env.example                        # Environment variable template
├── lefthook.yml                       # Git hooks (ruff, pyright)
├── uv.lock                            # Lock file (uv package manager)
└── README.md                          # Main project readme
```

## Critical Folders

| Folder | Purpose | Key Files |
|--------|---------|-----------|
| `fastcode/` | Core library — indexing, retrieval, graph, IR | `retriever.py`, `semantic_ir.py`, `projection_transform.py` |
| `fastcode/adapters/` | Precision-anchored extraction (tree-sitter + SCIP anchoring) | `scip_to_ir.py`, `ast_to_ir.py` |
| `nanobot/nanobot/agent/tools/` | FastCode-MCP bridge for AI agents | `fastcode.py` |
| `tests/` | Self-contained test suite | 61 files, property + snapshot + E2E |
| `config/` | Runtime configuration | `config.yaml` |
| `docs/` | Architecture & design docs | `review-response-context.md` |

## Entry Points

| Entry Point | File | Port/Interface | Purpose |
|-------------|------|----------------|---------|
| FastAPI REST | `api.py` | `:8001` | HTTP API for queries, indexing, projections |
| Flask Web UI | `web_app.py` | `:5000` | Browser-based query interface |
| MCP Server | `mcp_server.py` | stdio | AI coding assistant tool integration |
| CLI | `main.py` | Terminal | `fastcode` command for indexing, querying |
| Nanobot Gateway | `nanobot/` | `:18791` | Multi-channel AI agent gateway |

## Integration Points

- **api.py** → imports from `fastcode/` (FastCode class, retriever, projection)
- **mcp_server.py** → wraps `api.py` endpoints as MCP tools
- **nanobot/agent/tools/fastcode.py** → calls `api.py` via HTTP
- **web_app.py** → calls `api.py` via HTTP
- **docker-compose.yml** → wires nanobot → fastcode via internal network
