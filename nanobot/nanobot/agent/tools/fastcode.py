"""
FastCode integration tools for nanobot.

These tools allow nanobot to interact with the FastCode backend API
for repository-level code understanding, querying, and session management.

Communication: HTTP requests to FastCode's FastAPI backend.
In Docker Compose: nanobot -> http://fastcode:8001/...
"""

import json
import os
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool


def _get_fastcode_url() -> str:
    """Get FastCode API base URL from environment."""
    return os.environ.get("FASTCODE_API_URL", "http://fastcode:8001")


# ============================================================
# Tool 1: Load and Index Repository
# ============================================================

class FastCodeLoadRepoTool(Tool):
    """Load and index a code repository for querying."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_load_repo"

    @property
    def description(self) -> str:
        return (
            "Load and index a code repository into FastCode for code understanding and querying. "
            "Accepts a GitHub URL (e.g. https://github.com/user/repo) or a local directory path. "
            "After indexing, you can use fastcode_query to ask questions about the code. "
            "Indexing may take a while for large repositories."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Repository URL (e.g. https://github.com/user/repo) or local directory path",
                },
                "is_url": {
                    "type": "boolean",
                    "description": "True if source is a URL, False if local path. Default: true",
                },
            },
            "required": ["source"],
        }

    async def execute(
        self,
        source: str,
        is_url: bool = True,
        **kwargs: Any,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=1800.0) as client:
                response = await client.post(
                    f"{self._api_url}/load-and-index",
                    json={
                        "source": source,
                        "is_url": is_url,
                    },
                )
                response.raise_for_status()
                data = response.json()
                summary = data.get("summary", {})
                msg = data.get("message", "Repository loaded and indexed")
                lines = [
                    f"✓ {msg}",
                    "",
                ]
                if isinstance(summary, dict):
                    if "total_files" in summary:
                        lines.append(f"Files: {summary['total_files']}")
                    if "total_elements" in summary:
                        lines.append(f"Code elements: {summary['total_elements']}")
                    if "languages" in summary:
                        lines.append(f"Languages: {summary['languages']}")
                else:
                    lines.append(str(summary))
                lines.append("")
                lines.append("You can now use fastcode_query to ask questions about this repository.")
                return "\n".join(lines)
        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except httpx.HTTPStatusError as e:
            return f"Error: FastCode API returned status {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"Error loading repository: {str(e)}"


# ============================================================
# Tool 2: Query Repository (Core Tool)
# ============================================================

class FastCodeQueryTool(Tool):
    """Query a code repository using natural language."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_query"

    @property
    def description(self) -> str:
        return (
            "Ask a question about the loaded code repository. "
            "Supports natural language questions like 'How does authentication work?', "
            "'Where is the main entry point?', 'Explain the data flow', etc. "
            "Supports multi-turn dialogue: set multi_turn=true and reuse the session_id "
            "to have a continuous conversation about the code. "
            "Can query by snapshot_id directly, or by repo_name+ref_name for branch-scoped queries. "
            "The repository must be loaded and indexed first using fastcode_load_repo "
            "or fastcode_index_run."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language question about the code",
                },
                "snapshot_id": {
                    "type": "string",
                    "description": "Direct snapshot ID (e.g. snap:repo:abc123). Preferred for precise scoping.",
                },
                "repo_name": {
                    "type": "string",
                    "description": "Repository name (for branch/ref resolution). Use with ref_name.",
                },
                "ref_name": {
                    "type": "string",
                    "description": "Branch/ref name (e.g. main, develop). Use with repo_name.",
                },
                "multi_turn": {
                    "type": "boolean",
                    "description": "Enable multi-turn dialogue mode for follow-up questions. Default: true",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID for multi-turn dialogue. Omit to auto-generate a new one.",
                },
                "repo_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to specific repository names (for multi-repo mode). Optional.",
                },
            },
            "required": ["question"],
        }

    async def execute(
        self,
        question: str,
        multi_turn: bool = True,
        session_id: str | None = None,
        repo_filter: list[str] | None = None,
        snapshot_id: str | None = None,
        repo_name: str | None = None,
        ref_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            payload: dict[str, Any] = {
                "question": question,
                "multi_turn": multi_turn,
            }
            if session_id:
                payload["session_id"] = session_id
            if repo_filter:
                payload["repo_filter"] = repo_filter
            if snapshot_id:
                payload["snapshot_id"] = snapshot_id
            if repo_name:
                payload["repo_name"] = repo_name
            if ref_name:
                payload["ref_name"] = ref_name

            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self._api_url}/query",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "No answer generated.")
                sid = data.get("session_id", "")
                sources = data.get("sources", [])
                total_tokens = data.get("total_tokens")

                lines = [answer]

                if sources:
                    lines.append("")
                    lines.append("--- Sources ---")
                    for i, src in enumerate(sources[:5], 1):
                        name = src.get("name", src.get("relative_path", "unknown"))
                        stype = src.get("type", "")
                        lines.append(f"  {i}. {name} ({stype})")

                if sid:
                    lines.append(f"\n[Session: {sid}]")
                if total_tokens:
                    lines.append(f"[Tokens: {total_tokens}]")

                return "\n".join(lines)

        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                return "Error: No repository indexed. Use fastcode_load_repo first."
            return f"Error: FastCode API returned status {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"Error querying repository: {str(e)}"


# ============================================================
# Tool 3: List Repositories
# ============================================================

class FastCodeListReposTool(Tool):
    """List available and loaded repositories."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_list_repos"

    @property
    def description(self) -> str:
        return (
            "List all available (indexed on disk) and currently loaded repositories in FastCode. "
            "Shows repository names, sizes, and loading status."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{self._api_url}/repositories")
                response.raise_for_status()
                data = response.json()

                available = data.get("available", [])
                loaded = data.get("loaded", [])

                lines = []

                if loaded:
                    lines.append(f"=== Loaded Repositories ({len(loaded)}) ===")
                    for repo in loaded:
                        name = repo.get("name", repo.get("repo_name", "unknown"))
                        elements = repo.get("total_elements", repo.get("elements", "?"))
                        lines.append(f"  [active] {name} ({elements} elements)")
                    lines.append("")

                if available:
                    lines.append(f"=== Available on Disk ({len(available)}) ===")
                    for repo in available:
                        name = repo.get("name", repo.get("repo_name", "unknown"))
                        size = repo.get("size_mb", "?")
                        lines.append(f"  {name} ({size} MB)")

                if not lines:
                    return "No repositories found. Use fastcode_load_repo to load one."

                return "\n".join(lines)

        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except Exception as e:
            return f"Error listing repositories: {str(e)}"


# ============================================================
# Tool 4: System Status
# ============================================================

class FastCodeStatusTool(Tool):
    """Check FastCode system status."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_status"

    @property
    def description(self) -> str:
        return (
            "Check the current status of the FastCode system. "
            "Shows whether repositories are loaded and indexed, "
            "system health, and available repositories."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{self._api_url}/status")
                response.raise_for_status()
                data = response.json()

                status = data.get("status", "unknown")
                repo_loaded = data.get("repo_loaded", False)
                repo_indexed = data.get("repo_indexed", False)
                repo_info = data.get("repo_info", {})
                available = data.get("available_repositories", [])
                loaded = data.get("loaded_repositories", [])

                lines = [
                    "=== FastCode System Status ===",
                    f"Status: {status}",
                    f"Repository loaded: {'Yes' if repo_loaded else 'No'}",
                    f"Repository indexed: {'Yes' if repo_indexed else 'No'}",
                ]

                if repo_info and repo_info.get("name"):
                    lines.append(f"Current repo: {repo_info.get('name', 'N/A')}")

                if loaded:
                    lines.append(f"Loaded repos: {len(loaded)}")
                if available:
                    lines.append(f"Available repos on disk: {len(available)}")

                return "\n".join(lines)

        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except Exception as e:
            return f"Error checking status: {str(e)}"


# ============================================================
# Tool 5: Session Management
# ============================================================

class FastCodeSessionTool(Tool):
    """Manage FastCode dialogue sessions."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_session"

    @property
    def description(self) -> str:
        return (
            "Manage FastCode dialogue sessions for multi-turn code conversations. "
            "Actions: 'new' to create a new session, 'list' to list all sessions, "
            "'history' to view a session's conversation history, "
            "'delete' to delete a session."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["new", "list", "history", "delete"],
                    "description": "Action to perform: new, list, history, or delete",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID (required for 'history' and 'delete' actions)",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:

                if action == "new":
                    response = await client.post(f"{self._api_url}/new-session")
                    response.raise_for_status()
                    data = response.json()
                    sid = data.get("session_id", "unknown")
                    return (
                        f"New session created: {sid}\n"
                        f"Use this session_id with fastcode_query for multi-turn dialogue."
                    )

                elif action == "list":
                    response = await client.get(f"{self._api_url}/sessions")
                    response.raise_for_status()
                    data = response.json()
                    sessions = data.get("sessions", [])
                    if not sessions:
                        return "No dialogue sessions found."
                    lines = [f"=== Dialogue Sessions ({len(sessions)}) ==="]
                    for s in sessions[:20]:
                        sid = s.get("session_id", "?")
                        title = s.get("title", "Untitled")
                        turns = s.get("total_turns", 0)
                        lines.append(f"  [{sid}] {title} ({turns} turns)")
                    return "\n".join(lines)

                elif action == "history":
                    if not session_id:
                        return "Error: session_id is required for 'history' action."
                    response = await client.get(f"{self._api_url}/session/{session_id}")
                    response.raise_for_status()
                    data = response.json()
                    history = data.get("history", [])
                    if not history:
                        return f"No history found for session '{session_id}'."
                    lines = [f"=== Session {session_id} ({len(history)} turns) ==="]
                    for i, turn in enumerate(history, 1):
                        q = turn.get("query", turn.get("question", ""))
                        a = turn.get("answer", "")
                        lines.append(f"\n--- Turn {i} ---")
                        lines.append(f"Q: {q[:200]}{'...' if len(q) > 200 else ''}")
                        lines.append(f"A: {a[:300]}{'...' if len(a) > 300 else ''}")
                    return "\n".join(lines)

                elif action == "delete":
                    if not session_id:
                        return "Error: session_id is required for 'delete' action."
                    response = await client.delete(f"{self._api_url}/session/{session_id}")
                    response.raise_for_status()
                    return f"Session '{session_id}' deleted successfully."

                else:
                    return f"Unknown action: {action}. Use: new, list, history, or delete."

        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return f"Error: Session '{session_id}' not found."
            return f"Error: FastCode API returned status {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"Error managing session: {str(e)}"


# ============================================================
# Tool 6: Search Symbol
# ============================================================

class FastCodeSearchSymbolTool(Tool):
    """Search for a symbol (function, class, method) by name, ID, or path."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_search_symbol"

    @property
    def description(self) -> str:
        return (
            "Search for a symbol (function, class, method) in an indexed repository snapshot. "
            "Find by name (e.g. 'FastCode'), symbol_id, or file path (e.g. 'fastcode/main.py'). "
            "Requires a snapshot_id which can be obtained from fastcode_index_run or fastcode_status."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "snapshot_id": {
                    "type": "string",
                    "description": "Snapshot ID to search within (e.g. snap:myrepo:abc123)",
                },
                "name": {
                    "type": "string",
                    "description": "Symbol name to search for (e.g. 'FastCode', 'query')",
                },
                "symbol_id": {
                    "type": "string",
                    "description": "Exact symbol ID to look up",
                },
                "path": {
                    "type": "string",
                    "description": "File path to find symbols in (e.g. 'fastcode/retriever.py')",
                },
            },
            "required": ["snapshot_id"],
        }

    async def execute(
        self,
        snapshot_id: str,
        name: str | None = None,
        symbol_id: str | None = None,
        path: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not name and not symbol_id and not path:
            return "Error: Provide at least one of name, symbol_id, or path to search."

        try:
            params: dict[str, str] = {"snapshot_id": snapshot_id}
            if name:
                params["name"] = name
            if symbol_id:
                params["symbol_id"] = symbol_id
            if path:
                params["path"] = path

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self._api_url}/symbols/find",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()
                symbol = data.get("symbol", {})

                if not symbol:
                    return f"No symbol found in snapshot '{snapshot_id}'."

                lines = [
                    f"=== Symbol: {symbol.get('display_name', 'unknown')} ===",
                    f"Kind: {symbol.get('kind', 'unknown')}",
                    f"Qualified: {symbol.get('qualified_name', 'unknown')}",
                    f"File: {symbol.get('doc_path', 'unknown')}",
                    f"ID: {symbol.get('symbol_id', 'unknown')}",
                ]
                sources = symbol.get("source_set", [])
                if sources:
                    lines.append(f"Sources: {', '.join(sources)}")

                return "\n".join(lines)

        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return f"No symbol found matching your criteria in snapshot '{snapshot_id}'."
            return f"Error: FastCode API returned status {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"Error searching symbol: {str(e)}"


# ============================================================
# Tool 7: Call Chain (Graph Operations)
# ============================================================

class FastCodeCallChainTool(Tool):
    """Trace call chains (callers/callees) for a symbol via the graph API."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_call_chain"

    @property
    def description(self) -> str:
        return (
            "Trace the call chain for a function or method. "
            "Shows callers (who calls this symbol) and/or callees (what this symbol calls). "
            "Requires snapshot_id and symbol_id. Use fastcode_search_symbol to find the symbol_id first."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "snapshot_id": {
                    "type": "string",
                    "description": "Snapshot ID containing the symbol",
                },
                "symbol_id": {
                    "type": "string",
                    "description": "Symbol ID to trace calls for",
                },
                "direction": {
                    "type": "string",
                    "enum": ["callers", "callees", "both"],
                    "description": "Which direction to trace. Default: both",
                },
                "max_hops": {
                    "type": "integer",
                    "description": "Maximum depth to traverse. Default: 1",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["snapshot_id", "symbol_id"],
        }

    async def execute(
        self,
        snapshot_id: str,
        symbol_id: str,
        direction: str = "both",
        max_hops: int = 1,
        **kwargs: Any,
    ) -> str:
        if not snapshot_id or not symbol_id:
            return "Error: snapshot_id and symbol_id are required."

        try:
            results: dict[str, list] = {}
            async with httpx.AsyncClient(timeout=30.0) as client:
                if direction in ("callees", "both"):
                    resp = await client.get(
                        f"{self._api_url}/graph/callees",
                        params={"snapshot_id": snapshot_id, "symbol_id": symbol_id, "max_hops": max_hops},
                    )
                    resp.raise_for_status()
                    results["callees"] = resp.json().get("callees", [])

                if direction in ("callers", "both"):
                    resp = await client.get(
                        f"{self._api_url}/graph/callers",
                        params={"snapshot_id": snapshot_id, "symbol_id": symbol_id, "max_hops": max_hops},
                    )
                    resp.raise_for_status()
                    results["callers"] = resp.json().get("callers", [])

            lines = [f"=== Call Chain for {symbol_id} ==="]

            if "callers" in results:
                callers = results["callers"]
                lines.append(f"\n  Callers ({len(callers)}):")
                if callers:
                    for c in callers:
                        lines.append(
                            f"    - {c.get('display_name', c.get('symbol_id', '?'))}"
                            f" [{c.get('kind', '?')}]"
                        )
                else:
                    lines.append("    (none)")

            if "callees" in results:
                callees = results["callees"]
                lines.append(f"\n  Callees ({len(callees)}):")
                if callees:
                    for c in callees:
                        lines.append(
                            f"    - {c.get('display_name', c.get('symbol_id', '?'))}"
                            f" [{c.get('kind', '?')}]"
                        )
                else:
                    lines.append("    (none)")

            return "\n".join(lines)

        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except httpx.HTTPStatusError as e:
            return f"Error: FastCode API returned status {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"Error tracing call chain: {str(e)}"


# ============================================================
# Tool 8: Build / Fetch Projections
# ============================================================

class FastCodeBuildProjectionTool(Tool):
    """Build or fetch multi-layer projection artifacts (L0/L1/L2)."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_build_projection"

    @property
    def description(self) -> str:
        return (
            "Build or retrieve multi-layer code projections. "
            "Actions: 'build' to create a projection (L0 summary, L1 sections, L2 chunks), "
            "'get_layer' to fetch a specific layer. "
            "Scope kinds: 'snapshot' (whole repo), 'query' (focused on a question), "
            "'entity' (focused on a specific symbol/file)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["build", "get_layer"],
                    "description": "Action: 'build' creates projection, 'get_layer' retrieves a layer",
                },
                "scope_kind": {
                    "type": "string",
                    "enum": ["snapshot", "query", "entity"],
                    "description": "Projection scope kind (for build action)",
                },
                "snapshot_id": {
                    "type": "string",
                    "description": "Snapshot ID",
                },
                "repo_name": {
                    "type": "string",
                    "description": "Repository name (alternative to snapshot_id)",
                },
                "ref_name": {
                    "type": "string",
                    "description": "Branch/ref name",
                },
                "query": {
                    "type": "string",
                    "description": "Query text for query-scoped projections",
                },
                "target_id": {
                    "type": "string",
                    "description": "Entity ID/path for entity-scoped projections",
                },
                "projection_id": {
                    "type": "string",
                    "description": "Projection ID (for get_layer action)",
                },
                "layer": {
                    "type": "string",
                    "enum": ["L0", "L1", "L2"],
                    "description": "Layer to fetch (for get_layer action)",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        scope_kind: str | None = None,
        snapshot_id: str | None = None,
        repo_name: str | None = None,
        ref_name: str | None = None,
        query: str | None = None,
        target_id: str | None = None,
        projection_id: str | None = None,
        layer: str | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:

                if action == "build":
                    if not scope_kind:
                        return "Error: scope_kind is required for build action."
                    payload: dict[str, Any] = {"scope_kind": scope_kind}
                    if snapshot_id:
                        payload["snapshot_id"] = snapshot_id
                    if repo_name:
                        payload["repo_name"] = repo_name
                    if ref_name:
                        payload["ref_name"] = ref_name
                    if query:
                        payload["query"] = query
                    if target_id:
                        payload["target_id"] = target_id

                    response = await client.post(
                        f"{self._api_url}/projection/build",
                        json=payload,
                    )
                    response.raise_for_status()
                    result = response.json().get("result", {})
                    pid = result.get("projection_id", "unknown")
                    layers = result.get("layers_available", [])
                    return (
                        f"Projection built: {pid}\n"
                        f"Scope: {result.get('scope_kind', scope_kind)}\n"
                        f"Available layers: {', '.join(layers)}"
                    )

                elif action == "get_layer":
                    if not projection_id or not layer:
                        return "Error: projection_id and layer are required for get_layer action."
                    response = await client.get(
                        f"{self._api_url}/projection/{projection_id}/{layer}",
                    )
                    response.raise_for_status()
                    result = response.json().get("result", {})
                    return json.dumps(result, indent=2, ensure_ascii=False)

                else:
                    return f"Unknown action: {action}. Use: build or get_layer."

        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except httpx.HTTPStatusError as e:
            return f"Error: FastCode API returned status {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"Error with projection: {str(e)}"


# ============================================================
# Tool 9: Index Run (Snapshot-based Pipeline)
# ============================================================

class FastCodeIndexRunTool(Tool):
    """Run the snapshot-based index pipeline (IR-first approach)."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_index_run"

    @property
    def description(self) -> str:
        return (
            "Run the snapshot-based index pipeline for a repository. "
            "This is the modern IR-first indexing approach that produces a snapshot "
            "with AST + optional SCIP merge, graph materialization, and manifest publishing. "
            "Prefer this over fastcode_load_repo for new indexing. "
            "Supports targeting a specific branch (ref) or commit."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Repository URL or local path",
                },
                "is_url": {
                    "type": "boolean",
                    "description": "True if source is a URL. Default: auto-detect",
                },
                "ref": {
                    "type": "string",
                    "description": "Branch/tag/ref to index (e.g. 'main', 'v1.0')",
                },
                "commit": {
                    "type": "string",
                    "description": "Specific commit hash to index",
                },
                "force": {
                    "type": "boolean",
                    "description": "Force re-index even if snapshot exists. Default: false",
                },
                "enable_scip": {
                    "type": "boolean",
                    "description": "Enable SCIP extraction path. Default: true",
                },
            },
            "required": ["source"],
        }

    async def execute(
        self,
        source: str,
        is_url: bool | None = None,
        ref: str | None = None,
        commit: str | None = None,
        force: bool = False,
        enable_scip: bool = True,
        **kwargs: Any,
    ) -> str:
        try:
            payload: dict[str, Any] = {
                "source": source,
                "force": force,
                "publish": True,
                "enable_scip": enable_scip,
            }
            if is_url is not None:
                payload["is_url"] = is_url
            if ref:
                payload["ref"] = ref
            if commit:
                payload["commit"] = commit

            async with httpx.AsyncClient(timeout=1800.0) as client:
                response = await client.post(
                    f"{self._api_url}/index/run",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json().get("result", {})

                run_id = result.get("run_id", "unknown")
                snap_id = result.get("snapshot_id", "unknown")
                docs = result.get("documents", "?")
                syms = result.get("symbols", "?")
                edges = result.get("edges", "?")
                published = result.get("published", False)

                lines = [
                    f"Index run complete: {run_id}",
                    f"Snapshot: {snap_id}",
                    f"Documents: {docs}",
                    f"Symbols: {syms}",
                    f"Edges: {edges}",
                    f"Published: {'Yes' if published else 'No'}",
                ]
                return "\n".join(lines)

        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except httpx.HTTPStatusError as e:
            return f"Error: FastCode API returned status {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"Error running index pipeline: {str(e)}"


# ============================================================
# Tool 10: Upload Repository (ZIP)
# ============================================================

class FastCodeUploadRepoTool(Tool):
    """Upload a ZIP file and index it as a repository."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_upload_repo"

    @property
    def description(self) -> str:
        return (
            "Upload a ZIP file containing repository source code and index it. "
            "The ZIP should contain the repository root (with subdirectories for src, lib, etc.). "
            "Maximum file size: 100 MB. After uploading, you can use fastcode_query "
            "to ask questions about the code."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the ZIP file on the local filesystem",
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, file_path: str, **kwargs: Any) -> str:
        import os

        if not os.path.isfile(file_path):
            return f"Error: File not found: {file_path}"

        try:
            async with httpx.AsyncClient(timeout=1800.0) as client:
                with open(file_path, "rb") as f:
                    response = await client.post(
                        f"{self._api_url}/upload-and-index",
                        files={"file": (os.path.basename(file_path), f)},
                    )
                response.raise_for_status()
                data = response.json()

                msg = data.get("message", "Repository uploaded and indexed")
                summary = data.get("summary", {})
                lines = [f"OK {msg}"]

                if isinstance(summary, dict):
                    if "total_files" in summary:
                        lines.append(f"Files: {summary['total_files']}")
                    if "total_elements" in summary:
                        lines.append(f"Code elements: {summary['total_elements']}")
                    if "languages" in summary:
                        lines.append(f"Languages: {summary['languages']}")

                lines.append("You can now use fastcode_query to ask questions.")
                return "\n".join(lines)

        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except httpx.HTTPStatusError as e:
            return f"Error: FastCode API returned status {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"Error uploading repository: {str(e)}"


# ============================================================
# Helper: create all FastCode tools at once
# ============================================================

def create_all_tools(api_url: str | None = None) -> list[Tool]:
    """
    Create all FastCode tools with the given API URL.

    Usage in AgentLoop._register_default_tools():
        fastcode_url = os.environ.get("FASTCODE_API_URL")
        if fastcode_url:
            from nanobot.agent.tools.fastcode import create_all_tools
            for tool in create_all_tools(api_url=fastcode_url):
                self.tools.register(tool)
    """
    url = api_url or _get_fastcode_url()
    return [
        FastCodeLoadRepoTool(api_url=url),
        FastCodeQueryTool(api_url=url),
        FastCodeListReposTool(api_url=url),
        FastCodeStatusTool(api_url=url),
        FastCodeSessionTool(api_url=url),
        FastCodeSearchSymbolTool(api_url=url),
        FastCodeCallChainTool(api_url=url),
        FastCodeBuildProjectionTool(api_url=url),
        FastCodeIndexRunTool(api_url=url),
        FastCodeUploadRepoTool(api_url=url),
    ]
