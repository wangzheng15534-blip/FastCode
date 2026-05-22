# FastCode API Contracts

## Overview

FastCode exposes four interfaces: REST API (FastAPI), Web UI (Flask/FastAPI), MCP Server (stdio/SSE), and CLI (Click). All share the same core `FastCode` class but differ in initialization strategy and feature coverage.

---

## 1. REST API (`api.py`)

**Base URL:** `http://0.0.0.0:8001`
**Auth:** None (CORS wide-open)
**Init:** Lazy singleton on first request
**Auto-docs:** `/docs` (Swagger), `/redoc` (ReDoc)

### Health & Status

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Lightweight health check (initializing/healthy) |
| GET | `/status` | Full status with backend info, repo state |
| GET | `/diagnostics` | Support-safe diagnostic bundle with config summary, storage/backend state, dependency availability, and latest index-run metadata |
| GET | `/` | Version and status info |

### Repository Management

| Method | Path | Request Body | Description |
|--------|------|-------------|-------------|
| POST | `/load` | `{source, is_url?}` | Load repo by URL or local path |
| POST | `/load-and-index` | `{source, is_url?}` | Load + index in one call |
| POST | `/load-repositories` | `{repo_names}` | Load repos from cache by name |
| POST | `/index` | (none) | Index loaded repo (**DEPRECATED** — use `/index/run`) |
| POST | `/index/run` | `{source, is_url?, ref?, commit?, force?, publish?, enable_scip?, scip_artifact_path?}` | Snapshot-based indexing pipeline (SCIP + AST merge, TerminusDB publish) |
| GET | `/index/runs/{run_id}` | (none) | Get index run status |
| POST | `/index/publish/{run_id}` | `{ref_name?}` | Publish run to manifest + lineage |
| POST | `/index/publish/retry` | `{limit?}` | Retry pending Terminus publish tasks |
| POST | `/upload-zip` | multipart file (max 100MB) | Upload ZIP, extract to repos/ |
| POST | `/upload-and-index` | multipart file | Upload + index in one call |
| POST | `/index-multiple` | `{sources: [{source, is_url?}]}` | Load + index multiple repos |
| GET | `/repositories` | (none) | List available + loaded repos |
| POST | `/delete-repos` | `{repo_names, delete_source?}` | Delete repos + optionally source |
| DELETE | `/repository` | (none) | Unload current repo |

### Querying

| Method | Path | Request Body | Description |
|--------|------|-------------|-------------|
| POST | `/query` | `{question, snapshot_id?, repo_name?, ref_name?, filters?, multi_turn?, session_id?}` | Query by snapshot scope |
| POST | `/query-snapshot` | `{question, snapshot_id?, repo_name?, ref_name?, filters?, multi_turn?, session_id?}` | Query specific snapshot or branch manifest head |
| POST | `/query-stream` | (any) | **DISABLED** (501) |
| GET | `/summary` | (none) | Repository summary or multi-repo stats |

### Sessions

| Method | Path | Description |
|--------|------|-------------|
| POST | `/new-session` | Create new session (optionally clear old) |
| GET | `/sessions` | List all sessions |
| GET | `/session/{id}` | Get session history |
| DELETE | `/session/{id}` | Delete session |

### Graph & Symbols

| Method | Path | Params | Description |
|--------|------|--------|-------------|
| GET | `/symbols/find` | `snapshot_id, symbol_id?, name?, path?` | Find symbol by ID/name/path |
| GET | `/graph/callees` | `snapshot_id, symbol_id, max_hops=1` | Call graph callees |
| GET | `/graph/callers` | `snapshot_id, symbol_id, max_hops=1` | Call graph callers |
| GET | `/graph/dependencies` | `snapshot_id, doc_id, max_hops=1` | Document dependencies |

### Projections

| Method | Path | Request Body | Description |
|--------|------|-------------|-------------|
| POST | `/projection/build` | `{scope_kind, snapshot_id?, repo_name?, ref_name?, query?, target_id?, filters?, force?}` | Build L0/L1/L2 projection |
| GET | `/projection/{id}/{layer}` | (none) | Fetch projection layer |
| GET | `/projection/{id}/chunks/{chunk_id}` | (none) | Fetch L2 chunk |

### Manifests & Branches

| Method | Path | Description |
|--------|------|-------------|
| GET | `/manifests/{repo}/{ref}` | Latest manifest for branch |
| GET | `/manifests/snapshot/{id}` | Manifest for snapshot ID |
| GET | `/repos/{repo}/refs` | List refs (branches/tags) |

### Admin

| Method | Path | Description |
|--------|------|-------------|
| POST | `/redo/process` | Process pending redo tasks |
| POST | `/clear-cache` | Clear query cache |
| GET | `/cache-stats` | Cache statistics |
| POST | `/refresh-index-cache` | Refresh index scan cache |
| GET | `/scip/artifacts/{snapshot_id}` | SCIP artifact metadata |

---

## 2. Web App (`web_app.py`)

**Base URL:** `http://127.0.0.1:5777`
**Auth:** None | **Init:** Eager (startup)
**Key difference:** Legacy query (no snapshot scope). SSE streaming active.

All routes prefixed with `/api/`. Subset of REST API — no graph, projection, manifest, or SCIP endpoints. Adds `/api/query-stream` (SSE).

---

## 3. MCP Server (`mcp_server.py`)

**Transport:** stdio (default) or SSE (`--transport sse --port 9090`)
**Auth:** None (trust boundary at client)
**Init:** Lazy (first tool call)

### Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `code_qa` | `question, repos[], multi_turn?, session_id?` | Core Q&A — auto-clones/indexes repos |
| `list_sessions` | (none) | List sessions with IDs, titles, turn counts |
| `get_session_history` | `session_id` | Full Q&A history |
| `delete_session` | `session_id` | Delete session |
| `list_indexed_repos` | (none) | List indexed repos (fresh scan) |
| `delete_repo_metadata` | `repo_name` | Delete index artifacts, keep source |
| `search_symbol` | `symbol_name, repos[], symbol_type?` | Search symbols (exact > prefix > contains) |
| `get_repo_structure` | `repo_name` | Repo summary, directory tree, language stats |
| `get_file_summary` | `file_path, repos[]` | File structure: classes, functions, imports |
| `get_call_chain` | `symbol_name, repos[], direction?, max_hops?` | Trace callers/callees (up to 5 hops) |
| `reindex_repo` | `repo_source` | Force full re-index |

---

## 4. CLI (`main.py`)

**Entry:** `python main.py <command>`

| Command | Key Args | Description |
|---------|----------|-------------|
| `query` | `-q`, `-u/-p/-z`, `-r` | Query single or multi-repo |
| `index` | `-u/-p/-z` | Index repo |
| `interactive` | `-u/-p/-z`, `--multi-turn`, `--agency` | Interactive REPL with agency mode |
| `index-multiple` | `-u/-p/-z`, `-f` | Index multiple repos |
| `query-multiple` | `-q`, `-r` | Query across repos |
| `list-repos` | (none) | List indexed repos |
| `repo-stats` | (none) | Repo statistics |
| `list-sessions` | (none) | List sessions |
| `show-session` | `session_id` | Session history |
| `delete-session` | `session_id` | Delete session |
| `remove-repo` | `repo_name` | Remove repo + data |
| `clean-indices` | (none) | Clean orphaned index files |
| `clear-cache` | (none) | Clear query cache |
| `cache-stats` | (none) | Cache statistics |

---

## Pydantic Models

| Model | Fields |
|-------|--------|
| `QueryRequest` | `question, snapshot_id?, repo_name?, ref_name?, filters?, multi_turn?, session_id?` |
| `QueryResponse` | `answer, query, context_elements, sources[], prompt_tokens?, completion_tokens?, total_tokens?, session_id?` |
| `IndexRunRequest` | `source, is_url?, ref?, commit?, force?, publish?, enable_scip?, scip_artifact_path?` |
| `ProjectionBuildRequest` | `scope_kind, snapshot_id?, repo_name?, ref_name?, query?, target_id?, filters?, force?` |
| `LoadRepositoryRequest` | `source, is_url?` |
| `StatusResponse` | `status, repo_loaded, repo_indexed, repo_info, graph_expansion_backend?, storage_backend?, retrieval_backend?, available_repositories[], loaded_repositories[]` |
| `DiagnosticBundleResponse` | `status, bundle` |
| `NewSessionResponse` | `session_id` |
