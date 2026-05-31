"""
FastCode MCP Server - Expose repo-level code understanding via MCP protocol.

Usage:
    python mcp_server.py                    # stdio transport (default)
    python mcp_server.py --transport sse    # SSE transport on port 8080
    python mcp_server.py --port 9090        # SSE on custom port

MCP config example (for Claude Code / Cursor):
    {
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
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

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
import inspect
import json
import os
import uuid

from mcp.server.fastmcp import FastMCP

from fastcode.main.fastcode import FastCode
from fastcode.mcp.formatting import (
    format_call_chain,
    format_code_qa_response,
    format_delete_repo_metadata,
    format_file_summary,
    format_indexed_repos,
    format_repo_overview,
    format_session_history,
    format_session_list,
    format_symbol_search_results,
)
from fastcode.mcp.graph_tools import (
    compute_directed_path_for_snapshot,
    compute_find_callers_for_snapshot,
    compute_impact_analysis_for_snapshot,
    compute_leiden_clusters_for_snapshot,
    compute_steiner_path_for_snapshot,
)
from fastcode.runtime_support.observability import configure_logging

# ---------------------------------------------------------------------------
# Logging (file only – stdout is reserved for MCP JSON-RPC in stdio mode)
# ---------------------------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "logs")
logger = configure_logging(
    level="INFO",
    format_str="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file=os.path.join(log_dir, "mcp_server.log"),
    console=False,
    logger_name="fastcode.mcp",
)

# ---------------------------------------------------------------------------
# FastCode instance injection
# ---------------------------------------------------------------------------
_fastcode_instance: FastCode | None = None


def init_server(fc: FastCode) -> None:
    """Inject a shared FastCode instance. Called by the process main."""
    global _fastcode_instance
    _fastcode_instance = fc


def _get_fastcode() -> FastCode:
    if _fastcode_instance is None:
        raise RuntimeError(
            "FastCode MCP server not initialized. Call init_server(fc) first."
        )
    return _fastcode_instance


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
    ready_names = fc.ensure_repos_ready(repos)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded or indexed."

    # 2. Load indexed repos into memory (multi-repo merge)
    if not fc.ensure_loaded(ready_names):
        return "Error: Failed to load repository indexes."

    # 3. Session management
    sid = session_id or str(uuid.uuid4())[:8]

    # 4. Query
    result = fc.query.query(
        question=question,
        # Always enforce repository filtering for both single-repo and
        # multi-repo queries to avoid cross-repo source leakage.
        repo_filter=ready_names,
        session_id=sid,
        enable_multi_turn=multi_turn,
    )

    return format_code_qa_response(
        result.get("answer", ""),
        result.get("sources", []),
        sid,
    )


@mcp.tool()
def list_sessions() -> str:
    """List all existing conversation sessions.

    Returns a list of sessions with their IDs, titles (first query),
    turn counts, and timestamps. Useful for finding a session_id to
    continue a previous conversation.
    """
    fc = _get_fastcode()
    return format_session_list(fc.list_sessions())


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
    return format_session_history(session_id, history)  # type: ignore[arg-type]


@mcp.tool()
def get_turn_context(
    session_id: str,
    turn_number: int | None = None,
    format: str = "fcx",
) -> str:
    """Get typed working-memory context for a session turn.

    Args:
        session_id: Session identifier.
        turn_number: Optional turn number. If omitted, the latest turn is used.
        format: "fcx" for the compiled DSL or "json" for structured payloads.

    Returns:
        JSON describing the requested working-memory artifact.
    """
    fc = _get_fastcode()
    result = fc.get_turn_context(session_id, turn_number, format)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_context_bundle(
    session_id: str,
    turn_number: int | None = None,
    format: str = "json",
    token_budget: int = 2048,
) -> str:
    """Get a durable context bundle for a session turn.

    Args:
        session_id: Session identifier.
        turn_number: Optional turn number. If omitted, the latest turn is used.
        format: "json" for structured payloads or "rendered" for compact text.
        token_budget: Approximate token budget for rendered bundles.

    Returns:
        JSON describing the requested context bundle.
    """
    fc = _get_fastcode()
    result = fc.get_context_bundle(session_id, turn_number, format, token_budget)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_context_bundle_by_id(
    bundle_id: str,
    format: str = "json",
    token_budget: int = 2048,
) -> str:
    """Get a durable context bundle by bundle ID.

    Args:
        bundle_id: Context bundle identifier.
        format: "json" for structured payloads or "rendered" for compact text.
        token_budget: Approximate token budget for rendered bundles.

    Returns:
        JSON describing the requested context bundle.
    """
    fc = _get_fastcode()
    result = fc.get_context_bundle_by_id(bundle_id, format, token_budget)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def expand_context_bundle_ref(
    ref_id: str,
    session_id: str | None = None,
    turn_number: int | None = None,
    bundle_id: str | None = None,
    depth: str = "L2",
) -> str:
    """Expand a specific source ref from a durable context bundle.

    Args:
        ref_id: Evidence ref ID such as e1.
        session_id: Optional session identifier.
        turn_number: Optional turn number.
        bundle_id: Optional direct context bundle identifier.
        depth: Requested expansion depth label.

    Returns:
        JSON with the resolved bundle source-ref payload.
    """
    fc = _get_fastcode()
    result = fc.expand_context_bundle_ref(
        ref_id,
        session_id=session_id,
        turn_number=turn_number,
        bundle_id=bundle_id,
        depth=depth,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def create_context_activation(
    session_id: str | None = None,
    turn_number: int | None = None,
    bundle_id: str | None = None,
    active_ref_ids: list[str] | None = None,
    active_fact_ids: list[str] | None = None,
    active_hypothesis_ids: list[str] | None = None,
    reason: str | None = None,
) -> str:
    """Create an activation record for a durable context bundle.

    Args:
        session_id: Optional session identifier.
        turn_number: Optional turn number.
        bundle_id: Optional direct context bundle identifier.
        active_ref_ids: Evidence refs to activate.
        active_fact_ids: Facts to activate.
        active_hypothesis_ids: Hypotheses to activate.
        reason: Activation reason.

    Returns:
        JSON describing the persisted activation.
    """
    fc = _get_fastcode()
    result = fc.create_context_activation(
        session_id=session_id,
        turn_number=turn_number,
        bundle_id=bundle_id,
        active_ref_ids=active_ref_ids,
        active_fact_ids=active_fact_ids,
        active_hypothesis_ids=active_hypothesis_ids,
        reason=reason,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def create_handoff(
    session_id: str,
    turn_number: int | None = None,
    mode: str = "delegate",
) -> str:
    """Create a handoff artifact from the specified or latest session turn.

    Args:
        session_id: Session identifier.
        turn_number: Optional turn number. If omitted, the latest turn is used.
        mode: Handoff mode label.

    Returns:
        JSON describing the persisted handoff artifact.
    """
    fc = _get_fastcode()
    result = fc.create_handoff(session_id, turn_number, mode)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_handoff_artifact(artifact_id: str) -> str:
    """Get a persisted handoff artifact by ID.

    Args:
        artifact_id: Handoff artifact identifier.

    Returns:
        JSON describing the handoff artifact.
    """
    fc = _get_fastcode()
    result = fc.get_handoff_artifact(artifact_id)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def expand_context_ref(
    session_id: str,
    turn_number: int,
    ref_id: str,
    depth: str = "L2",
) -> str:
    """Expand a specific evidence ref from working memory.

    Args:
        session_id: Session identifier.
        turn_number: Turn number to inspect.
        ref_id: Evidence ref ID such as e1.
        depth: Requested expansion depth label.

    Returns:
        JSON with the resolved evidence-ref payload.
    """
    fc = _get_fastcode()
    result = fc.expand_context_ref(session_id, turn_number, ref_id, depth)
    return json.dumps(result, ensure_ascii=False, indent=2)


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
    return format_indexed_repos(fc.store.list_available_repos())


@mcp.tool()
def delete_repo_metadata(repo_name: str) -> str:
    """Delete indexed metadata for a repository while keeping source code.

    This removes vector/BM25/graph index artifacts and the repository's
    overview entry from repository overview storage, but does NOT delete source files
    from the configured repository workspace.

    Args:
        repo_name: Repository name to clean metadata for.

    Returns:
        Confirmation message with deleted artifacts and freed disk space.
    """
    fc = _get_fastcode()
    result = fc.remove_repository(repo_name, delete_source=False)
    return format_delete_repo_metadata(
        repo_name, result.get("deleted_files", []), result.get("freed_mb", 0)
    )


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
    ready_names = fc.ensure_repos_ready(repos, allow_incremental=False)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded."
    if not fc.ensure_loaded(ready_names):
        return "Error: Failed to load repository indexes."

    results = fc.query.search_symbols(symbol_name, symbol_type=symbol_type)
    return format_symbol_search_results(symbol_name, results)


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
    if not fc.store.is_repo_indexed(repo_name):
        return f"Repository '{repo_name}' is not indexed. Use code_qa or reindex_repo first."
    overview = fc.store.get_repo_overview(repo_name)
    if not overview:
        return (
            f"No overview found for repository '{repo_name}'. It may need re-indexing."
        )
    metadata = overview.get("metadata", {})
    file_structure = metadata.get("file_structure", {})
    languages = file_structure.get("languages", {})
    return format_repo_overview(repo_name, metadata, languages)


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
    ready_names = fc.ensure_repos_ready(repos, allow_incremental=False)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded."
    if not fc.ensure_loaded(ready_names):
        return "Error: Failed to load repository indexes."

    result = fc.query.get_file_structure(file_path)
    if not result:
        return f"No elements found for file path '{file_path}'."
    file_meta = result["file"]
    actual_path = file_meta.get("relative_path", file_path)
    repo_name = file_meta.get("repo_name", "")
    top_level = [
        f for f in result["functions"] if not f.get("metadata", {}).get("class_name")
    ]
    return format_file_summary(
        actual_path, file_meta, result["classes"], top_level, repo_name
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
    ready_names = fc.ensure_repos_ready(repos, allow_incremental=False)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded."
    if not fc.ensure_loaded(ready_names):
        return "Error: Failed to load repository indexes."

    max_hops = min(max_hops, 5)
    result = fc.query.walk_call_chain(
        symbol_name, direction=direction, max_hops=max_hops
    )
    if not result:
        return f"Symbol '{symbol_name}' not found in call graph."
    return format_call_chain(
        result["name"],
        result["type"],
        result["path"],
        result["start_line"],
        result["callers"],
        result["callees"],
    )


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

    fc = _get_fastcode()
    return json.dumps(
        compute_directed_path_for_snapshot(
            fc,
            from_symbol,
            to_symbol,
            snapshot_id,
            max_hops,
            graph_types,
        )
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

    fc = _get_fastcode()
    return json.dumps(
        compute_impact_analysis_for_snapshot(
            fc,
            symbol,
            snapshot_id,
            max_hops,
            graph_types,
        )
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
    return json.dumps(compute_leiden_clusters_for_snapshot(fc, snapshot_id))


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

    fc = _get_fastcode()
    return json.dumps(compute_steiner_path_for_snapshot(fc, terminals, snapshot_id))


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

    fc = _get_fastcode()
    return json.dumps(
        compute_find_callers_for_snapshot(fc, symbol, snapshot_id, max_hops)
    )


@mcp.tool()
def get_session_prefix(snapshot_id: str) -> str:
    """Get architectural overview for system prompt injection.

    Returns L0+L1 projection data that should be loaded into the
    agent's system prompt at session start. Provides architectural
    awareness without any queries needed.

    Args:
        snapshot_id: Which snapshot to get the prefix for
                     (e.g. "snap:myrepo:abc123").

    Returns:
        JSON with snapshot_id, projection_id, l0, l1 fields.
    """
    import json

    fc = _get_fastcode()
    try:
        result = fc.projection.get_session_prefix(snapshot_id)
        if result.get("error"):
            return json.dumps(
                {"found": False, "snapshot_id": snapshot_id, "error": result["error"]}
            )
        return json.dumps({"found": True, **result})
    except Exception as e:
        return json.dumps({"found": False, "snapshot_id": snapshot_id, "error": str(e)})


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
    return fc.reindex_repository(repo_source)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
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

    init_server(FastCode())

    if args.transport == "sse":
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
