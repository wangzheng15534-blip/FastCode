"""
FastCode MCP Server - Expose repo-level code understanding via MCP protocol.

Usage:
    python mcp_server.py                    # stdio transport (default)
    python mcp_server.py --transport sse    # SSE transport on port 8080
    python mcp_server.py --port 9090        # SSE on custom port

MCP config example (for Claude Code / Cursor):
    {
      "mcpServers": {
        "fastcode": {
          "command": "python",
          "args": ["/path/to/FastCode/mcp_server.py"],
          "env": {
            "MODEL": "your-model",
            "BASE_URL": "your-base-url",
            "API_KEY": "your-api-key"
          }
        }
      }
    }
"""

import inspect
import logging
import os
import sys
import uuid

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging (file only – stdout is reserved for MCP JSON-RPC in stdio mode)
# ---------------------------------------------------------------------------
log_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "mcp_server.log"))],
)
logger = logging.getLogger("fastcode.mcp")

# ---------------------------------------------------------------------------
# Lazy FastCode singleton
# ---------------------------------------------------------------------------
_fastcode_instance = None


def _get_fastcode():
    """Lazy-init the FastCode engine (heavy imports happen here)."""
    global _fastcode_instance
    if _fastcode_instance is None:
        logger.info("Initializing FastCode engine …")
        from fastcode import FastCode

        _fastcode_instance = FastCode()
        logger.info("FastCode engine ready.")
    return _fastcode_instance


def _repo_name_from_source(source: str, is_url: bool) -> str:
    """Derive a canonical repo name from a URL or local path."""
    from fastcode.utils import get_repo_name_from_url

    if is_url:
        return get_repo_name_from_url(source)
    # Local path: use the directory basename
    return os.path.basename(os.path.normpath(source))


def _is_repo_indexed(repo_name: str) -> bool:
    """Check whether a repo already has a persisted FAISS index."""
    fc = _get_fastcode()
    persist_dir = fc.vector_store.persist_dir
    faiss_path = os.path.join(persist_dir, f"{repo_name}.faiss")
    meta_path = os.path.join(persist_dir, f"{repo_name}_metadata.pkl")
    return os.path.exists(faiss_path) and os.path.exists(meta_path)


def _apply_forced_env_excludes(fc) -> None:
    """
    Force-ignore environment-related paths before indexing.

    Always excludes virtual environment folders. Optionally excludes
    site-packages when FASTCODE_EXCLUDE_SITE_PACKAGES=1.
    """
    repo_cfg = fc.config.setdefault("repository", {})
    ignore_patterns = list(repo_cfg.get("ignore_patterns", []))

    forced_patterns = [
        ".venv",
        "venv",
        ".env",
        "env",
        "**/.venv/**",
        "**/venv/**",
        "**/.env/**",
        "**/env/**",
    ]

    # Optional (opt-in): site-packages can be huge/noisy in some repos.
    if os.getenv("FASTCODE_EXCLUDE_SITE_PACKAGES", "0").lower() in {"1", "true", "yes"}:
        forced_patterns.extend(
            [
                "site-packages",
                "**/site-packages/**",
            ]
        )

    added = []
    for pattern in forced_patterns:
        if pattern not in ignore_patterns:
            ignore_patterns.append(pattern)
            added.append(pattern)

    repo_cfg["ignore_patterns"] = ignore_patterns
    # Keep loader in sync when FastCode instance already exists.
    fc.loader.ignore_patterns = ignore_patterns

    if added:
        logger.info(f"Added forced ignore patterns: {added}")


def _ensure_repos_ready(
    repos: list[str], allow_incremental: bool = True, ctx=None
) -> list[str]:
    """
    For each repo source string:
      - If already indexed → skip
      - If URL and not on disk → clone + index
      - If local path → load + index

    Returns the list of canonical repo names that are ready.
    """
    fc = _get_fastcode()
    _apply_forced_env_excludes(fc)
    ready_names: list[str] = []

    for source in repos:
        resolved_is_url = fc._infer_is_url(source)
        name = _repo_name_from_source(source, resolved_is_url)

        # Already indexed
        if _is_repo_indexed(name):
            # Try incremental update for local repos
            if not resolved_is_url and allow_incremental:
                abs_path = os.path.abspath(source)
                if os.path.isdir(abs_path):
                    try:
                        result = fc.incremental_reindex(name, repo_path=abs_path)
                        if result and result.get("changes", 0) > 0:
                            logger.info(f"Incremental update for '{name}': {result}")
                            # Force reload since on-disk data changed
                            fc.repo_indexed = False
                            fc.loaded_repositories.clear()
                    except Exception as e:
                        logger.warning(f"Incremental reindex failed for '{name}': {e}")
            logger.info(f"Repo '{name}' ready.")
            ready_names.append(name)
            continue

        # Need to index
        logger.info(f"Repo '{name}' not indexed. Preparing …")

        if resolved_is_url:
            # Clone and index
            logger.info(f"Cloning {source} …")
            fc.load_repository(source, is_url=True)
        else:
            # Local path
            abs_path = os.path.abspath(source)
            if not os.path.isdir(abs_path):
                logger.error(f"Local path does not exist: {abs_path}")
                continue
            fc.load_repository(abs_path, is_url=False)

        logger.info(f"Indexing '{name}' …")
        fc.index_repository(force=False)
        logger.info(f"Indexing '{name}' complete.")
        ready_names.append(name)

    return ready_names


def _ensure_loaded(fc, ready_names: list[str]) -> bool:
    """Ensure repos are loaded into memory (vectors + BM25 + graphs)."""
    if not fc.repo_indexed or set(ready_names) != set(fc.loaded_repositories.keys()):
        logger.info(f"Loading repos into memory: {ready_names}")
        return fc._load_multi_repo_cache(repo_names=ready_names)
    return True


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
MCP_SERVER_DESCRIPTION = (
    "Repo-level code understanding - ask questions about any codebase."
)
_fastmcp_kwargs = {}
try:
    # Backward compatibility: older mcp versions do not accept `description`.
    if "description" in inspect.signature(FastMCP.__init__).parameters:
        _fastmcp_kwargs["description"] = MCP_SERVER_DESCRIPTION
except (TypeError, ValueError):
    # If signature introspection fails, fall back to the safest constructor shape.
    pass

mcp = FastMCP("FastCode", **_fastmcp_kwargs)


@mcp.tool()
def code_qa(
    question: str,
    repos: list[str],
    multi_turn: bool = True,
    session_id: str | None = None,
) -> str:
    """Ask a question about one or more code repositories.

    This is the core tool for repo-level code understanding. FastCode will
    automatically clone (if URL) and index repositories that haven't been
    indexed yet, then answer your question using hybrid retrieval + LLM.

    Args:
        question: The question to ask about the code.
        repos: List of repository sources. Each can be:
               - A GitHub/GitLab URL (e.g. "https://github.com/user/repo")
               - A local filesystem path (e.g. "/home/user/projects/myrepo")
               If the repo is already indexed, it won't be re-indexed.
        multi_turn: Enable multi-turn conversation mode. When True, previous
                    Q&A context from the same session_id is used. Default: True.
        session_id: Session identifier for multi-turn conversations. If not
                    provided, a new session is created automatically. Pass the
                    same session_id across calls to continue a conversation.

    Returns:
        The answer to your question, with source references.
    """
    fc = _get_fastcode()

    # 1. Ensure all repos are indexed
    ready_names = _ensure_repos_ready(repos)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded or indexed."

    # 2. Load indexed repos into memory (multi-repo merge)
    if not _ensure_loaded(fc, ready_names):
        return "Error: Failed to load repository indexes."

    # 3. Session management
    sid = session_id or str(uuid.uuid4())[:8]

    # 4. Query
    result = fc.query(
        question=question,
        # Always enforce repository filtering for both single-repo and
        # multi-repo queries to avoid cross-repo source leakage.
        repo_filter=ready_names,
        session_id=sid,
        enable_multi_turn=multi_turn,
    )

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    # Format output
    parts = [answer]

    if sources:
        parts.append("\n\n---\nSources:")
        for s in sources[:]:
            file_path = s.get("file", s.get("relative_path", ""))
            repo = s.get("repo", s.get("repository", ""))
            name = s.get("name", "")
            start = s.get("start_line", "")
            end = s.get("end_line", "")
            if (not start or not end) and s.get("lines"):
                lines = str(s.get("lines", ""))
                if "-" in lines:
                    parsed_start, parsed_end = lines.split("-", 1)
                    start = start or parsed_start
                    end = end or parsed_end
            loc = f"L{start}-L{end}" if start and end else ""
            parts.append(
                f"  - {repo}/{file_path}:{loc} ({name})"
                if repo
                else f"  - {file_path}:{loc} ({name})"
            )

    parts.append(f"\n[session_id: {sid}]")
    return "\n".join(parts)


@mcp.tool()
def list_sessions() -> str:
    """List all existing conversation sessions.

    Returns a list of sessions with their IDs, titles (first query),
    turn counts, and timestamps. Useful for finding a session_id to
    continue a previous conversation.
    """
    fc = _get_fastcode()
    sessions = fc.list_sessions()

    if not sessions:
        return "No sessions found."

    lines = ["Sessions:"]
    for s in sessions:
        sid = s.get("session_id", "?")
        title = s.get("title", "Untitled")
        turns = s.get("total_turns", 0)
        mode = "multi-turn" if s.get("multi_turn", False) else "single-turn"
        lines.append(f'  - {sid}: "{title}" ({turns} turns, {mode})')

    return "\n".join(lines)


@mcp.tool()
def get_session_history(session_id: str) -> str:
    """Get the full conversation history for a session.

    Args:
        session_id: The session identifier to retrieve history for.

    Returns:
        The complete Q&A history of the session.
    """
    fc = _get_fastcode()
    history = fc.get_session_history(session_id)

    if not history:
        return f"No history found for session '{session_id}'."

    lines = [f"Session {session_id} history:"]
    for turn in history:
        turn_num = turn.get("turn_number", "?")
        query = turn.get("query", "")
        answer = turn.get("answer", "")
        # Truncate long answers for readability
        if len(answer) > 500:
            answer = answer[:500] + " …"
        lines.append(f"\n--- Turn {turn_num} ---")
        lines.append(f"Q: {query}")
        lines.append(f"A: {answer}")

    return "\n".join(lines)


@mcp.tool()
def delete_session(session_id: str) -> str:
    """Delete a conversation session and all its history.

    Args:
        session_id: The session identifier to delete.

    Returns:
        Confirmation message.
    """
    fc = _get_fastcode()
    success = fc.delete_session(session_id)
    if success:
        return f"Session '{session_id}' deleted."
    return f"Failed to delete session '{session_id}'. It may not exist."


@mcp.tool()
def list_indexed_repos() -> str:
    """List all repositories that have been indexed and are available for querying.

    Returns:
        A list of indexed repository names with metadata.
    """
    fc = _get_fastcode()
    available = fc.vector_store.scan_available_indexes(use_cache=False)

    if not available:
        return "No indexed repositories found."

    lines = ["Indexed repositories:"]
    for repo in available:
        name = repo.get("name", repo.get("repo_name", "?"))
        elements = repo.get("element_count", repo.get("elements", "?"))
        size = repo.get("size_mb", "?")
        lines.append(f"  - {name} ({elements} elements, {size} MB)")

    return "\n".join(lines)


@mcp.tool()
def delete_repo_metadata(repo_name: str) -> str:
    """Delete indexed metadata for a repository while keeping source code.

    This removes vector/BM25/graph index artifacts and the repository's
    overview entry from repo_overviews.pkl, but does NOT delete source files
    from the configured repository workspace.

    Args:
        repo_name: Repository name to clean metadata for.

    Returns:
        Confirmation message with deleted artifacts and freed disk space.
    """
    fc = _get_fastcode()
    result = fc.remove_repository(repo_name, delete_source=False)

    deleted_files = result.get("deleted_files", [])
    freed_mb = result.get("freed_mb", 0)

    if not deleted_files:
        return (
            f"No metadata files found for repository '{repo_name}'. "
            "Source code was not modified."
        )

    lines = [f"Deleted metadata for repository '{repo_name}' (source code kept)."]
    lines.append(f"Freed: {freed_mb} MB")
    lines.append("Removed artifacts:")
    for fname in deleted_files:
        lines.append(f"  - {fname}")
    return "\n".join(lines)


@mcp.tool()
def search_symbol(
    symbol_name: str,
    repos: list[str],
    symbol_type: str | None = None,
) -> str:
    """Search for a symbol (function, class, method) by name across repositories.

    Finds definitions matching the given name with case-insensitive search.
    Results are ranked: exact match > prefix match > contains match (top 20).

    Args:
        symbol_name: Name of the symbol to search for (e.g. "FastCode", "query").
        repos: List of repository sources (URLs or local paths).
        symbol_type: Optional filter: "function", "class", "file", or "documentation".

    Returns:
        Matching definitions with file path, line range, and signature.
    """
    fc = _get_fastcode()
    ready_names = _ensure_repos_ready(repos, allow_incremental=False)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded."
    if not _ensure_loaded(fc, ready_names):
        return "Error: Failed to load repository indexes."

    query_lower = symbol_name.lower()
    exact, prefix, contains = [], [], []

    for meta in fc.vector_store.metadata:
        name = meta.get("name", "")
        elem_type = meta.get("type", "")
        if elem_type == "repository_overview":
            continue
        if symbol_type and elem_type != symbol_type:
            continue

        name_lower = name.lower()
        if name_lower == query_lower:
            exact.append(meta)
        elif name_lower.startswith(query_lower):
            prefix.append(meta)
        elif query_lower in name_lower:
            contains.append(meta)

    ranked = (exact + prefix + contains)[:20]
    if not ranked:
        return f"No symbols matching '{symbol_name}' found."

    lines = [f"Found {len(ranked)} result(s) for '{symbol_name}':"]
    for meta in ranked:
        name = meta.get("name", "")
        etype = meta.get("type", "")
        repo = meta.get("repo_name", "")
        rel_path = meta.get("relative_path", "")
        start = meta.get("start_line", "")
        end = meta.get("end_line", "")
        sig = meta.get("signature", "")
        loc = f"L{start}-L{end}" if start and end else ""
        line = f"  - [{etype}] {name}"
        if sig:
            line += f"  |  {sig}"
        line += f"\n    {repo}/{rel_path}:{loc}" if repo else f"\n    {rel_path}:{loc}"
        lines.append(line)

    return "\n".join(lines)


@mcp.tool()
def get_repo_structure(repo_name: str) -> str:
    """Get the high-level structure and summary of an indexed repository.

    Returns the repository summary, directory tree, and language statistics.
    Does not require loading the full index into memory.

    Args:
        repo_name: Name of an indexed repository (see list_indexed_repos).

    Returns:
        Repository summary, directory structure, and language breakdown.
    """
    fc = _get_fastcode()
    if not _is_repo_indexed(repo_name):
        return f"Repository '{repo_name}' is not indexed. Use code_qa or reindex_repo first."

    overviews = fc.vector_store.load_repo_overviews()
    overview = overviews.get(repo_name)
    if not overview:
        return (
            f"No overview found for repository '{repo_name}'. It may need re-indexing."
        )

    metadata = overview.get("metadata", {})
    summary = metadata.get("summary", "No summary available.")
    structure_text = metadata.get("structure_text", "")
    file_structure = metadata.get("file_structure", {})
    languages = file_structure.get("languages", {})

    parts = [f"Repository: {repo_name}", ""]
    parts.append(f"Summary:\n{summary}")

    if languages:
        parts.append("\nLanguages:")
        for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
            parts.append(f"  - {lang}: {count} files")

    if structure_text:
        parts.append(f"\nDirectory Structure:\n{structure_text}")

    return "\n".join(parts)


@mcp.tool()
def get_file_summary(file_path: str, repos: list[str]) -> str:
    """Get the structure summary of a specific file (classes, functions, imports).

    Args:
        file_path: Path to the file (e.g. "fastcode/main.py").
                   Flexible matching: endswith or contains.
        repos: List of repository sources to search in.

    Returns:
        File structure: classes (with methods), top-level functions, and import count.
    """
    fc = _get_fastcode()
    ready_names = _ensure_repos_ready(repos, allow_incremental=False)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded."
    if not _ensure_loaded(fc, ready_names):
        return "Error: Failed to load repository indexes."

    # Find matching elements by relative_path
    matching = []
    for meta in fc.vector_store.metadata:
        rel = meta.get("relative_path", "")
        if meta.get("type") == "repository_overview":
            continue
        if rel.endswith(file_path) or file_path in rel:
            matching.append(meta)

    if not matching:
        return f"No elements found for file path '{file_path}'."

    files = [m for m in matching if m.get("type") == "file"]
    classes = [m for m in matching if m.get("type") == "class"]
    functions = [m for m in matching if m.get("type") == "function"]

    file_meta = files[0] if files else matching[0]
    actual_path = file_meta.get("relative_path", file_path)
    repo = file_meta.get("repo_name", "")

    parts = [f"File: {repo}/{actual_path}" if repo else f"File: {actual_path}"]

    if files:
        fm = files[0]
        parts.append(f"Language: {fm.get('language', '?')}")
        mi = fm.get("metadata", {})
        parts.append(
            f"Lines: {mi.get('total_lines', '?')} (code: {mi.get('code_lines', '?')})"
        )
        num_imports = mi.get("num_imports", 0)
        if num_imports:
            parts.append(f"Imports: {num_imports}")

    if classes:
        parts.append(f"\nClasses ({len(classes)}):")
        for c in classes:
            sig = c.get("signature", c.get("name", ""))
            mi = c.get("metadata", {})
            methods = mi.get("methods", [])
            loc = f"L{c.get('start_line', '')}-L{c.get('end_line', '')}"
            parts.append(f"  - {sig} ({loc})")
            for m in methods:
                parts.append(f"      .{m}")

    if functions:
        top_level = [
            f for f in functions if not f.get("metadata", {}).get("class_name")
        ]
        if top_level:
            parts.append(f"\nFunctions ({len(top_level)}):")
            for fn in top_level:
                sig = fn.get("signature", fn.get("name", ""))
                loc = f"L{fn.get('start_line', '')}-L{fn.get('end_line', '')}"
                parts.append(f"  - {sig} ({loc})")

    return "\n".join(parts)


def _walk_call_chain(
    gb,
    element_id: str,
    direction: str,
    hops_left: int,
    parts: list,
    indent: int = 2,
    visited: set = None,
):
    """Recursively walk the call chain and format output."""
    if visited is None:
        visited = {element_id}

    neighbors = (
        gb.get_callers(element_id)
        if direction == "callers"
        else gb.get_callees(element_id)
    )

    if not neighbors:
        parts.append(f"{'  ' * indent}(none)")
        return

    for nid in neighbors:
        if nid in visited:
            continue
        visited.add(nid)
        elem = gb.element_by_id.get(nid)
        if elem:
            loc = (
                f"{elem.relative_path}:L{elem.start_line}" if elem.relative_path else ""
            )
            parts.append(f"{'  ' * indent}- {elem.name} [{loc}]")
            if hops_left > 1:
                _walk_call_chain(
                    gb, nid, direction, hops_left - 1, parts, indent + 1, visited
                )


@mcp.tool()
def get_call_chain(
    symbol_name: str,
    repos: list[str],
    direction: str = "both",
    max_hops: int = 2,
) -> str:
    """Trace the call chain for a function or method.

    Shows who calls this symbol (callers) and/or what it calls (callees),
    up to max_hops levels deep.

    Args:
        symbol_name: Name of the function/method to trace.
        repos: List of repository sources.
        direction: "callers", "callees", or "both" (default: "both").
        max_hops: Maximum depth of the call chain (default: 2, max: 5).

    Returns:
        Formatted call chain showing callers and/or callees.
    """
    fc = _get_fastcode()
    ready_names = _ensure_repos_ready(repos, allow_incremental=False)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded."
    if not _ensure_loaded(fc, ready_names):
        return "Error: Failed to load repository indexes."

    max_hops = min(max_hops, 5)
    gb = fc.graph_builder
    name_lower = symbol_name.lower()
    target_id, target_elem = None, None

    # Exact match via element_by_name
    elem = gb.element_by_name.get(symbol_name)
    if elem:
        target_elem, target_id = elem, elem.id

    # Fallback: case-insensitive search
    if not target_id:
        for eid, elem in gb.element_by_id.items():
            if elem.name.lower() == name_lower:
                target_elem, target_id = elem, eid
                break

    # Fallback: partial match
    if not target_id:
        for eid, elem in gb.element_by_id.items():
            if name_lower in elem.name.lower():
                target_elem, target_id = elem, eid
                break

    if not target_id:
        return f"Symbol '{symbol_name}' not found in call graph."

    parts = [
        f"Call chain for '{target_elem.name}' ({target_elem.type})"
        f" at {target_elem.relative_path}:L{target_elem.start_line}"
    ]

    if direction in ("callers", "both"):
        parts.append("\n  Callers (who calls this):")
        _walk_call_chain(gb, target_id, "callers", max_hops, parts, indent=2)

    if direction in ("callees", "both"):
        parts.append("\n  Callees (what this calls):")
        _walk_call_chain(gb, target_id, "callees", max_hops, parts, indent=2)

    return "\n".join(parts)


_VALID_GRAPH_TYPES = {"call", "dependency", "inheritance", "reference", "containment"}

_GRAPH_TYPE_MAP = {
    "call": "call_graph",
    "dependency": "dependency_graph",
    "inheritance": "inheritance_graph",
    "reference": "reference_graph",
    "containment": "containment_graph",
}


def _resolve_unit_id(query: str, snapshot) -> str | None:
    """Resolve a symbol query to a unit_id from snapshot units.

    Tries exact unit_id match first, then display_name, then qualified_name.
    Returns the first matching unit_id or None.
    """
    # 1. Exact unit_id
    for unit in snapshot.units:
        if unit.unit_id == query:
            return unit.unit_id

    # 2. Exact display_name (first match)
    for unit in snapshot.units:
        if unit.display_name == query:
            return unit.unit_id

    # 3. Exact qualified_name (first match)
    for unit in snapshot.units:
        if unit.qualified_name and unit.qualified_name == query:
            return unit.unit_id

    # 4. Case-insensitive display_name (first match)
    query_lower = query.lower()
    for unit in snapshot.units:
        if unit.display_name.lower() == query_lower:
            return unit.unit_id

    return None


def _format_path_node(unit_id: str, snapshot) -> dict[str, str | int | None]:
    """Build a metadata dict for a node in the path."""
    for unit in snapshot.units:
        if unit.unit_id == unit_id:
            return {
                "symbol_id": unit.unit_id,
                "display_name": unit.display_name,
                "kind": unit.kind,
                "path": unit.path,
                "start_line": unit.start_line,
            }
    return {
        "symbol_id": unit_id,
        "display_name": None,
        "kind": None,
        "path": None,
        "start_line": None,
    }


@mcp.tool()
def directed_path(
    from_symbol: str,
    to_symbol: str,
    snapshot_id: str,
    max_hops: int = 5,
    graph_types: list[str] | None = None,
) -> str:
    """Find the directed shortest path between two symbols through the code graph.

    Traverses the directed call and dependency graph (by default) to find
    how one symbol reaches another. Returns the path with symbol metadata
    for each node, or reports if no path exists.

    Args:
        from_symbol: Starting symbol (display_name, qualified_name, or symbol ID).
        to_symbol: Target symbol (display_name, qualified_name, or symbol ID).
        snapshot_id: Which snapshot to query (e.g. "snap:myrepo:abc123").
        max_hops: Maximum path length to search (default: 5).
        graph_types: Which graph layers to traverse. Default: ["call", "dependency"].

    Returns:
        JSON-like string with found, path, path_length, and error fields.
    """
    import json

    import networkx as nx

    from fastcode.ir_graph_builder import IRGraphBuilder

    if graph_types is None:
        graph_types = ["call", "dependency"]

    invalid = [gt for gt in graph_types if gt not in _VALID_GRAPH_TYPES]
    if invalid:
        return json.dumps(
            {
                "found": False,
                "path": [],
                "path_length": 0,
                "error": f"Invalid graph types: {invalid}. Valid: {sorted(_VALID_GRAPH_TYPES)}",
            }
        )

    # Load snapshot
    fc = _get_fastcode()
    snapshot = fc.snapshot_store.load_snapshot(snapshot_id)
    if not snapshot:
        return json.dumps(
            {
                "found": False,
                "path": [],
                "path_length": 0,
                "error": f"Snapshot not found: {snapshot_id}",
            }
        )

    # Resolve symbol queries to unit IDs
    from_id = _resolve_unit_id(from_symbol, snapshot)
    if not from_id:
        return json.dumps(
            {
                "found": False,
                "path": [],
                "path_length": 0,
                "error": f"Symbol not found: {from_symbol}",
            }
        )

    to_id = _resolve_unit_id(to_symbol, snapshot)
    if not to_id:
        return json.dumps(
            {
                "found": False,
                "path": [],
                "path_length": 0,
                "error": f"Symbol not found: {to_symbol}",
            }
        )

    if from_id == to_id:
        return json.dumps(
            {
                "found": True,
                "path": [_format_path_node(from_id, snapshot)],
                "path_length": 0,
                "error": None,
            }
        )

    # Build IR graphs and union selected types
    graphs = IRGraphBuilder().build_graphs(snapshot)
    combined: nx.DiGraph | None = None
    for gt in graph_types:
        attr = _GRAPH_TYPE_MAP[gt]
        g: nx.DiGraph = getattr(graphs, attr)
        if combined is None:
            combined = g.copy()
        else:
            combined = nx.compose(combined, g)

    if combined is None or combined.number_of_nodes() == 0:
        return json.dumps(
            {
                "found": False,
                "path": [],
                "path_length": 0,
                "error": "Selected graph types are empty (no nodes or edges).",
            }
        )

    # Ensure both endpoints are in the combined graph
    if from_id not in combined or to_id not in combined:
        missing = []
        if from_id not in combined:
            missing.append(from_symbol)
        if to_id not in combined:
            missing.append(to_symbol)
        return json.dumps(
            {
                "found": False,
                "path": [],
                "path_length": 0,
                "error": f"Symbol(s) not in graph: {missing}",
            }
        )

    # Find directed shortest path
    try:
        path = nx.shortest_path(combined, source=from_id, target=to_id)
    except nx.NetworkXNoPath:
        # Check if reverse path exists
        try:
            nx.shortest_path(combined, source=to_id, target=from_id)
            return json.dumps(
                {
                    "found": False,
                    "path": [],
                    "path_length": 0,
                    "error": (
                        f"No directed path from '{from_symbol}' to '{to_symbol}'. "
                        f"A reverse path exists (from '{to_symbol}' to '{from_symbol}')."
                    ),
                }
            )
        except nx.NetworkXNoPath:
            return json.dumps(
                {
                    "found": False,
                    "path": [],
                    "path_length": 0,
                    "error": f"No directed path between '{from_symbol}' and '{to_symbol}' in either direction.",
                }
            )

    # Check path length against max_hops
    if len(path) - 1 > max_hops:
        return json.dumps(
            {
                "found": False,
                "path": [],
                "path_length": len(path) - 1,
                "error": f"Shortest path length {len(path) - 1} exceeds max_hops={max_hops}.",
            }
        )

    path_nodes = [_format_path_node(nid, snapshot) for nid in path]
    return json.dumps(
        {
            "found": True,
            "path": path_nodes,
            "path_length": len(path) - 1,
            "error": None,
        }
    )


@mcp.tool()
def impact_analysis(
    symbol: str,
    snapshot_id: str,
    max_hops: int = 3,
    graph_types: list[str] | None = None,
) -> str:
    """Analyze what would be affected if a symbol changes.

    Traverses REVERSED call and dependency edges to find all callers and
    dependents of the given symbol. Returns affected symbols grouped by
    distance.

    Args:
        symbol: Symbol to analyze (display_name, qualified_name, or symbol ID).
        snapshot_id: Which snapshot to query.
        max_hops: Maximum traversal depth (default: 3).
        graph_types: Which graph layers. Default: ["call", "dependency"].

    Returns:
        JSON with affected (list of {symbol_id, display_name, kind, path, start_line, distance, edge_types}),
        total_count, error.
    """
    import json
    from collections import deque

    import networkx as nx

    from fastcode.ir_graph_builder import IRGraphBuilder

    if graph_types is None:
        graph_types = ["call", "dependency"]

    invalid = [gt for gt in graph_types if gt not in _VALID_GRAPH_TYPES]
    if invalid:
        return json.dumps(
            {
                "affected": [],
                "total_count": 0,
                "error": f"Invalid graph types: {invalid}. Valid: {sorted(_VALID_GRAPH_TYPES)}",
            }
        )

    # Load snapshot
    fc = _get_fastcode()
    snapshot = fc.snapshot_store.load_snapshot(snapshot_id)
    if not snapshot:
        return json.dumps(
            {
                "affected": [],
                "total_count": 0,
                "error": f"Snapshot not found: {snapshot_id}",
            }
        )

    # Resolve symbol
    unit_id = _resolve_unit_id(symbol, snapshot)
    if not unit_id:
        return json.dumps(
            {
                "affected": [],
                "total_count": 0,
                "error": f"Symbol not found: {symbol}",
            }
        )

    # Build IR graphs and union selected types
    graphs = IRGraphBuilder().build_graphs(snapshot)
    combined: nx.DiGraph | None = None
    for gt in graph_types:
        attr = _GRAPH_TYPE_MAP[gt]
        g: nx.DiGraph = getattr(graphs, attr)
        if combined is None:
            combined = g.copy()
        else:
            combined = nx.compose(combined, g)

    if combined is None or combined.number_of_nodes() == 0:
        return json.dumps(
            {
                "affected": [],
                "total_count": 0,
                "error": "Selected graph types are empty (no nodes or edges).",
            }
        )

    if unit_id not in combined:
        return json.dumps(
            {
                "affected": [],
                "total_count": 0,
                "error": f"Symbol not in graph: {symbol}",
            }
        )

    # BFS using predecessors on the original directed graph.
    # Predecessors of node N are symbols that have an edge TO N,
    # meaning they call/depend on N and are affected if N changes.
    visited: dict[str, int] = {}  # node -> distance
    edge_types_map: dict[str, set[str]] = {}  # node -> set of edge types
    queue = deque()
    queue.append((unit_id, 0))
    visited[unit_id] = 0

    while queue:
        node, dist = queue.popleft()
        if dist >= max_hops:
            continue
        for pred in combined.predecessors(node):
            if pred not in visited:
                visited[pred] = dist + 1
                edge_types_map[pred] = set()
                queue.append((pred, dist + 1))
            # Track which graph types contributed this edge
            for gt in graph_types:
                attr = _GRAPH_TYPE_MAP[gt]
                g: nx.DiGraph = getattr(graphs, attr)
                if g.has_edge(pred, node):
                    edge_types_map.setdefault(pred, set()).add(gt)

    # Build affected list (exclude the queried symbol itself)
    affected = []
    for nid, dist in sorted(visited.items(), key=lambda kv: kv[1]):
        if nid == unit_id:
            continue
        node_info = _format_path_node(nid, snapshot)
        node_info["distance"] = dist
        node_info["edge_types"] = sorted(edge_types_map.get(nid, set()))
        affected.append(node_info)

    return json.dumps(
        {
            "affected": affected,
            "total_count": len(affected),
            "error": None,
        }
    )


@mcp.tool()
def leiden_clusters(
    snapshot_id: str,
) -> str:
    """Get module boundaries (Leiden community detection) for a snapshot.

    Returns cluster structure: cluster IDs, representative symbols, node counts,
    and cross-cluster references. Uses cached projection if available.

    Args:
        snapshot_id: Which snapshot to query.

    Returns:
        JSON with clusters (list of {cluster_id, label, node_count, representative, top_members}),
        xrefs (list of {from_cluster, to_cluster, weight}), total_clusters, error.
    """
    import json

    fc = _get_fastcode()

    # Load snapshot
    snapshot = fc.snapshot_store.load_snapshot(snapshot_id)
    if not snapshot:
        return json.dumps(
            {
                "clusters": [],
                "xrefs": [],
                "total_clusters": 0,
                "error": f"Snapshot not found: {snapshot_id}",
            }
        )

    # Try to load cached projection first
    projection_store = getattr(fc, "projection_store", None)
    projection_transformer = getattr(fc, "projection_transformer", None)

    if projection_store is not None and projection_store.enabled:
        from fastcode.projection_models import ProjectionScope

        scope = ProjectionScope(
            scope_kind="full",
            snapshot_id=snapshot_id,
            scope_key="full",
        )
        cached_id = projection_store.find_cached_projection_id(scope, "default")
        if cached_id:
            l1_data = projection_store.get_layer(cached_id, "L1")
            if l1_data:
                return json.dumps(_extract_cluster_data(l1_data, snapshot))

    # No cached projection; try to build one
    if projection_transformer is not None:
        try:
            from fastcode.projection_models import ProjectionScope

            scope = ProjectionScope(
                scope_kind="full",
                snapshot_id=snapshot_id,
                scope_key="full",
            )
            result = projection_transformer.build(scope, snapshot)
            if result and result.l1:
                return json.dumps(_extract_cluster_data(result.l1, snapshot))
        except Exception as exc:
            return json.dumps(
                {
                    "clusters": [],
                    "xrefs": [],
                    "total_clusters": 0,
                    "error": f"Failed to build projection: {exc}",
                }
            )

    return json.dumps(
        {
            "clusters": [],
            "xrefs": [],
            "total_clusters": 0,
            "error": "Projection store not configured and projection transformer not available.",
        }
    )


def _extract_cluster_data(l1_data: dict, snapshot) -> dict:
    """Extract structured cluster data from L1 projection data."""
    clusters = []
    xrefs = []

    # Extract sections from content_extra
    content_extra = l1_data.get("content_extra", {})
    sections = content_extra.get("sections", [])
    navigation = content_extra.get("navigation", [])

    # Build a name -> label map from sections
    label_map = {}
    for section in sections:
        label_map[section.get("name", "")] = section.get("text", "")

    # Extract xrefs from relations
    relations = content_extra.get("relations", {})
    xref_list = relations.get("xref", [])
    for xref in xref_list:
        xref_id = xref.get("id", "")
        parts = xref_id.split("->")
        if len(parts) == 2:
            xrefs.append(
                {
                    "from_cluster": parts[0],
                    "to_cluster": parts[1],
                    "weight": xref.get("confidence", 0),
                }
            )

    # Build cluster info from projection_meta if available
    projection_meta = l1_data.get("projection_meta", {})
    # If we have raw cluster data in the L1, use it
    # Otherwise synthesize from sections
    for i, section in enumerate(sections):
        cluster_info = {
            "cluster_id": str(i),
            "label": section.get("name", f"Cluster {i}"),
            "node_count": 0,
            "representative": None,
            "top_members": [],
        }
        # Try to parse node count from text like "5 nodes"
        text = section.get("text", "")
        try:
            cluster_info["node_count"] = int(text.split()[0])
        except (ValueError, IndexError):
            pass
        # Check navigation for representative ref
        if i < len(navigation):
            nav = navigation[i]
            rep_ref = nav.get("ref", {})
            if rep_ref:
                rep_display = rep_ref.get("display_name")
                if rep_display:
                    cluster_info["representative"] = rep_display
        clusters.append(cluster_info)

    return {
        "clusters": clusters,
        "xrefs": xrefs,
        "total_clusters": len(clusters),
        "error": None,
    }


@mcp.tool()
def steiner_path(
    terminals: list[str],
    snapshot_id: str,
) -> str:
    """Find a small undirected explanatory subgraph connecting terminal symbols.

    Uses Steiner tree approximation to find the minimal subgraph connecting
    all specified terminal symbols. Useful for "explain how these symbols relate".

    Args:
        terminals: List of terminal symbol names/IDs (2-8 symbols).
        snapshot_id: Which snapshot to query.

    Returns:
        JSON with found (bool), nodes (list of path_node dicts), edges (list of {from, to, type}), error.
    """
    import json

    import networkx as nx

    from fastcode.ir_graph_builder import IRGraphBuilder

    if not terminals or len(terminals) < 2:
        return json.dumps(
            {
                "found": False,
                "nodes": [],
                "edges": [],
                "error": "At least 2 terminal symbols required.",
            }
        )

    if len(terminals) > 8:
        return json.dumps(
            {
                "found": False,
                "nodes": [],
                "edges": [],
                "error": "Maximum 8 terminal symbols allowed.",
            }
        )

    # Load snapshot
    fc = _get_fastcode()
    snapshot = fc.snapshot_store.load_snapshot(snapshot_id)
    if not snapshot:
        return json.dumps(
            {
                "found": False,
                "nodes": [],
                "edges": [],
                "error": f"Snapshot not found: {snapshot_id}",
            }
        )

    # Resolve all terminal symbols
    terminal_ids = []
    for t in terminals:
        tid = _resolve_unit_id(t, snapshot)
        if not tid:
            return json.dumps(
                {
                    "found": False,
                    "nodes": [],
                    "edges": [],
                    "error": f"Symbol not found: {t}",
                }
            )
        terminal_ids.append(tid)

    # Remove duplicates
    terminal_ids = list(dict.fromkeys(terminal_ids))

    # Build IR graphs and create undirected union
    graphs = IRGraphBuilder().build_graphs(snapshot)
    undirected: nx.Graph | None = None
    for attr_name in [
        "call_graph",
        "dependency_graph",
        "inheritance_graph",
        "reference_graph",
        "containment_graph",
    ]:
        g: nx.DiGraph = getattr(graphs, attr_name)
        ug = g.to_undirected()
        if undirected is None:
            undirected = ug.copy()
        else:
            undirected = nx.compose(undirected, ug)

    if undirected is None or undirected.number_of_nodes() == 0:
        return json.dumps(
            {
                "found": False,
                "nodes": [],
                "edges": [],
                "error": "Graph is empty (no nodes or edges).",
            }
        )

    # Check all terminals are in the graph
    missing = [t for t, tid in zip(terminals, terminal_ids) if tid not in undirected]
    if missing:
        return json.dumps(
            {
                "found": False,
                "nodes": [],
                "edges": [],
                "error": f"Symbol(s) not in graph: {missing}",
            }
        )

    # If only one unique terminal, return just that node
    if len(terminal_ids) == 1:
        node = _format_path_node(terminal_ids[0], snapshot)
        return json.dumps(
            {
                "found": True,
                "nodes": [node],
                "edges": [],
                "error": None,
            }
        )

    # Compute Steiner tree approximation
    try:
        steiner = nx.approximation.steiner_tree(undirected, terminal_ids)
    except nx.NetworkXError as exc:
        return json.dumps(
            {
                "found": False,
                "nodes": [],
                "edges": [],
                "error": f"Steiner tree computation failed: {exc}",
            }
        )

    # Prune non-terminal leaves iteratively
    terminal_set = set(terminal_ids)
    changed = True
    while changed:
        changed = False
        leaves = [
            n
            for n in steiner.nodes()
            if steiner.degree(n) == 1 and n not in terminal_set
        ]
        for leaf in leaves:
            steiner.remove_node(leaf)
            changed = True

    if steiner.number_of_nodes() == 0:
        return json.dumps(
            {
                "found": False,
                "nodes": [],
                "edges": [],
                "error": "Steiner tree is empty after pruning.",
            }
        )

    # Collect nodes
    nodes = [_format_path_node(nid, snapshot) for nid in steiner.nodes()]

    # Collect edges with type information from original directed graphs
    edges = []
    edge_type_map = {
        "call_graph": "call",
        "dependency_graph": "dependency",
        "inheritance_graph": "inheritance",
        "reference_graph": "reference",
        "containment_graph": "containment",
    }
    seen_edges = set()
    for u, v in steiner.edges():
        edge_key = (min(u, v), max(u, v))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        # Determine edge type from original directed graphs
        edge_types = []
        for attr_name, etype in edge_type_map.items():
            g: nx.DiGraph = getattr(graphs, attr_name)
            if g.has_edge(u, v) or g.has_edge(v, u):
                edge_types.append(etype)
        edges.append(
            {
                "from": u,
                "to": v,
                "type": "+".join(edge_types) if edge_types else "unknown",
            }
        )

    return json.dumps(
        {
            "found": True,
            "nodes": nodes,
            "edges": edges,
            "error": None,
        }
    )


@mcp.tool()
def find_callers(
    symbol: str,
    snapshot_id: str,
    max_hops: int = 2,
) -> str:
    """Find all symbols that call or depend on the given symbol.

    Traverses reversed call edges to find direct and transitive callers.

    Args:
        symbol: Symbol to find callers for.
        snapshot_id: Which snapshot to query.
        max_hops: Maximum traversal depth (default: 2).

    Returns:
        JSON with callers (list of {symbol_id, display_name, kind, path, start_line, distance}),
        total_count, error.
    """
    import json
    from collections import deque

    from fastcode.ir_graph_builder import IRGraphBuilder

    # Load snapshot
    fc = _get_fastcode()
    snapshot = fc.snapshot_store.load_snapshot(snapshot_id)
    if not snapshot:
        return json.dumps(
            {
                "callers": [],
                "total_count": 0,
                "error": f"Snapshot not found: {snapshot_id}",
            }
        )

    # Resolve symbol
    unit_id = _resolve_unit_id(symbol, snapshot)
    if not unit_id:
        return json.dumps(
            {
                "callers": [],
                "total_count": 0,
                "error": f"Symbol not found: {symbol}",
            }
        )

    # Build call graph
    graphs = IRGraphBuilder().build_graphs(snapshot)
    call_g = graphs.call_graph

    if call_g.number_of_nodes() == 0:
        return json.dumps(
            {
                "callers": [],
                "total_count": 0,
                "error": "Call graph is empty.",
            }
        )

    if unit_id not in call_g:
        return json.dumps(
            {
                "callers": [],
                "total_count": 0,
                "error": f"Symbol not in call graph: {symbol}",
            }
        )

    # BFS on reversed call graph (predecessors = callers)
    visited: dict[str, int] = {}  # node -> distance
    queue = deque()
    queue.append((unit_id, 0))
    visited[unit_id] = 0

    while queue:
        node, dist = queue.popleft()
        if dist >= max_hops:
            continue
        # In the original graph, edge A->B means A calls B.
        # We want callers of node, i.e. nodes that have an edge TO node.
        # Those are predecessors of node in the original graph.
        for pred in call_g.predecessors(node):
            if pred not in visited:
                visited[pred] = dist + 1
                queue.append((pred, dist + 1))

    # Build callers list (exclude the queried symbol itself)
    callers = []
    for nid, dist in sorted(visited.items(), key=lambda kv: kv[1]):
        if nid == unit_id:
            continue
        node_info = _format_path_node(nid, snapshot)
        node_info["distance"] = dist
        callers.append(node_info)

    return json.dumps(
        {
            "callers": callers,
            "total_count": len(callers),
            "error": None,
        }
    )


@mcp.tool()
def reindex_repo(repo_source: str) -> str:
    """Force a full re-index of a repository.

    Clones (if URL) or loads (if local path) the repository and rebuilds
    all indexes from scratch.

    Args:
        repo_source: Repository URL or local filesystem path.

    Returns:
        Confirmation with element count.
    """
    fc = _get_fastcode()
    _apply_forced_env_excludes(fc)

    resolved_is_url = fc._infer_is_url(repo_source)
    name = _repo_name_from_source(repo_source, resolved_is_url)
    logger.info(f"Force re-indexing '{name}' from {repo_source}")

    if resolved_is_url:
        fc.load_repository(repo_source, is_url=True)
    else:
        abs_path = os.path.abspath(repo_source)
        if not os.path.isdir(abs_path):
            return f"Error: Local path does not exist: {abs_path}"
        fc.load_repository(abs_path, is_url=False)

    fc.index_repository(force=True)
    count = fc.vector_store.get_count()

    # Reset in-memory state so next _ensure_loaded does a clean load
    fc.repo_indexed = False
    fc.loaded_repositories.clear()

    return f"Successfully re-indexed '{name}': {count} elements indexed."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FastCode MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for SSE transport (default: 8080)",
    )
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", sse_params={"port": args.port})
    else:
        mcp.run(transport="stdio")
