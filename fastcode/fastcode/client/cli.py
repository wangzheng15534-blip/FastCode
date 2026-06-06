#!/usr/bin/env python3
"""
FastCode 2.0 - Command Line Interface

Thin entry frame that delegates to a running FastCode API server over HTTP.
Use ``fastcode serve`` to start the server, then run commands against it.
"""

from __future__ import annotations

import json
import sys

import click

from fastcode.client.http_client import FastCodeClient


def _make_client(base_url: str) -> FastCodeClient:
    return FastCodeClient(base_url=base_url)


@click.group()
@click.option(
    "--server",
    envvar="FASTCODE_SERVER",
    default="http://localhost:8000",
    help="FastCode API server URL",
)
@click.pass_context
def cli(ctx: click.Context, server: str) -> None:
    """FastCode - Repository-Level Code Understanding System"""
    ctx.ensure_object(dict)
    ctx.obj["server"] = server
    ctx.obj["client"] = _make_client(server)


# ======================================================================
# Server management
# ======================================================================


@cli.command()
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--port", "-p", default=8000, help="Bind port")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the FastCode API server."""
    import uvicorn

    click.echo(f"Starting FastCode API server on {host}:{port}")
    if reload:
        uvicorn.run(
            "fastcode.main.serve:create_api_app",
            host=host,
            port=port,
            factory=True,
            reload=True,
        )
    else:
        from fastcode.main.serve import create_api_app

        uvicorn.run(create_api_app(), host=host, port=port)


@cli.command("mcp")
@click.option("--server-cmd", help="Custom MCP server command (default: built-in)")
def mcp_client(server_cmd: str | None) -> None:
    """Start interactive MCP client connected to FastCode MCP server."""
    from fastcode.client.mcp_client import mcp_client_main

    command = server_cmd.split() if server_cmd else None
    mcp_client_main(server_command=command)


# ======================================================================
# Query commands
# ======================================================================


@cli.command()
@click.option("--query", "-q", required=True, help="Question to ask")
@click.option("--repo-url", "-u", help="Repository URL to clone")
@click.option("--repo-path", "-p", help="Local repository path")
@click.option("--repo-zip", "-z", help="ZIP file containing repository")
@click.option(
    "--load-cache",
    is_flag=True,
    help="Load from existing index cache (multi-repo mode)",
)
@click.option(
    "--repos",
    "-r",
    multiple=True,
    help="Specific repositories to search in multi-repo mode",
)
@click.option("--output", "-o", help="Output file (default: stdout)")
@click.pass_context
def query(
    ctx: click.Context,
    query: str,
    repo_url: str | None,
    repo_path: str | None,
    repo_zip: str | None,
    load_cache: bool,
    repos: tuple[str, ...],
    output: str | None,
) -> None:
    """Query a repository with a question."""
    client = ctx.obj["client"]

    try:
        # Multi-repo mode with cache
        if load_cache:
            repo_names = list(repos) if repos else None
            if repo_names:
                click.echo(
                    f"Loading repositories from cache: {', '.join(repo_names)}..."
                )
            else:
                click.echo("Loading multi-repository index from cache...")

            client.load_cached_repos(repo_names=repo_names)

            repo_filter = list(repos) if repos else None
            click.echo(f"\nProcessing query: {query}\n")
            result = client.query(query, repo_filter=repo_filter)

        # Single repo mode
        else:
            if not repo_url and not repo_path and not repo_zip:
                click.echo(
                    "Error: Either --repo-url, --repo-path, --repo-zip, or --load-cache must be provided",
                    err=True,
                )
                sys.exit(1)

            source = repo_url or repo_path or repo_zip
            is_url = bool(repo_url)

            click.echo(f"Loading and indexing repository: {source}")
            client.load_and_index(source=source, is_url=is_url)

            click.echo(f"\nProcessing query: {query}\n")
            result = client.query_snapshot(question=query)

        # Format output
        answer = result.get("answer", "")
        sources = result.get("sources", [])

        if sources:
            answer += "\n\nSources:"
            for s in sources:
                name = s.get("name", "")
                path = s.get("path", "")
                answer += f"\n  - {name} ({path})"

        if output:
            with open(output, "w") as f:
                f.write(answer)
            click.echo(f"Result saved to {output}")
        else:
            click.echo(answer)

    except Exception as e:
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--repo-url", "-u", help="Repository URL to clone")
@click.option("--repo-path", "-p", help="Local repository path")
@click.option("--repo-zip", "-z", help="ZIP file containing repository")
@click.pass_context
def index(
    ctx: click.Context,
    repo_url: str | None,
    repo_path: str | None,
    repo_zip: str | None,
) -> None:
    """Index a repository (without querying)."""
    if not repo_url and not repo_path and not repo_zip:
        click.echo(
            "Error: Either --repo-url, --repo-path, or --repo-zip must be provided",
            err=True,
        )
        sys.exit(1)

    client = ctx.obj["client"]

    try:
        source = repo_url or repo_path or repo_zip
        is_url = bool(repo_url)

        click.echo(f"Loading and indexing repository: {source}")
        client.load_and_index(source=source, is_url=is_url)

        summary = client.summary()
        click.echo(f"\n{summary.get('summary', 'Indexing complete.')}")

    except Exception as e:
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--repo-url", "-u", help="Repository URL to clone")
@click.option("--repo-path", "-p", help="Local repository path")
@click.option("--repo-zip", "-z", help="ZIP file containing repository")
@click.option(
    "--load-cache", is_flag=True, help="Load from multi-repo cache for multi-repo mode"
)
@click.option(
    "--repos",
    "-r",
    multiple=True,
    help="Specific repositories to load from cache",
)
@click.option("--multi-turn", is_flag=True, help="Enable multi-turn dialogue mode")
@click.option(
    "--session-id",
    "-s",
    help="Session ID for multi-turn dialogue (auto-generated if not provided)",
)
@click.pass_context
def interactive(
    ctx: click.Context,
    repo_url: str | None,
    repo_path: str | None,
    repo_zip: str | None,
    load_cache: bool,
    repos: tuple[str, ...],
    multi_turn: bool,
    session_id: str | None,
) -> None:
    """Start interactive query session."""
    client = ctx.obj["client"]

    if multi_turn and not session_id:
        import uuid

        session_id = str(uuid.uuid4())[:8]
        click.echo(f"Multi-turn mode enabled. Session ID: {session_id}")

    try:
        # Multi-repo mode
        if load_cache:
            if repos:
                click.echo(f"Loading repositories from cache: {', '.join(repos)}...")
            else:
                click.echo("Loading all available repositories from cache...")

            client.load_cached_repos(repo_names=list(repos) if repos else None)

            info = client.status()
            loaded = info.get("loaded_repositories", [])
            click.echo(f"\nLoaded {len(loaded)} repositories")
            for repo in loaded:
                click.echo(f"  - {repo}")

        # Single repo mode
        else:
            if not repo_url and not repo_path and not repo_zip:
                click.echo(
                    "Error: Either --repo-url, --repo-path, --repo-zip, or --load-cache must be provided",
                    err=True,
                )
                sys.exit(1)

            source = repo_url or repo_path or repo_zip
            is_url = bool(repo_url)

            click.echo(f"Loading and indexing repository: {source}")
            client.load_and_index(source=source, is_url=is_url)

        # Interactive loop
        click.echo("=" * 60)
        if multi_turn:
            click.echo("FastCode Interactive Mode (Multi-turn)")
            click.echo(f"Session ID: {session_id}")
        else:
            click.echo("FastCode Interactive Mode (Single-turn)")
        click.echo("\nCommands: 'quit' to exit")
        click.echo("=" * 60 + "\n")

        while True:
            try:
                user_input = click.prompt("\nYour question", type=str)
            except (KeyboardInterrupt, EOFError):
                break

            if user_input.lower() in ("quit", "exit", "q"):
                break

            if not user_input.strip():
                continue

            # Parse repository filter from query (format: @repo1,repo2 question)
            repo_filter: list[str] | None = None
            question = user_input
            if load_cache and user_input.startswith("@"):
                parts = user_input.split(" ", 1)
                if len(parts) == 2:
                    repo_filter = [r.strip() for r in parts[0][1:].split(",")]
                    question = parts[1]
                    click.echo(f"Searching in: {', '.join(repo_filter)}")

            click.echo("\nProcessing...\n")

            result = client.query(
                question=question,
                repo_filter=repo_filter,
                session_id=session_id if multi_turn else None,
                multi_turn=multi_turn,
            )

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            click.echo(answer)

            if sources:
                click.echo("\nSources:")
                for s in sources:
                    click.echo(f"  - {s.get('name', '')} ({s.get('path', '')})")

            if multi_turn:
                turn_num = result.get("turn_number", "?")
                click.echo(f"\n[Turn {turn_num} saved]")

    except Exception as e:
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


# ======================================================================
# Multi-repo commands
# ======================================================================


@cli.command("index-multiple")
@click.option(
    "--repo-urls",
    "-u",
    multiple=True,
    help="Repository URLs (can be used multiple times)",
)
@click.option(
    "--repo-paths",
    "-p",
    multiple=True,
    help="Local repository paths (can be used multiple times)",
)
@click.option(
    "--repo-zips",
    "-z",
    multiple=True,
    help="ZIP files (can be used multiple times)",
)
@click.option(
    "--urls-file", "-f", help="File containing repository URLs (one per line)"
)
@click.pass_context
def index_multiple(
    ctx: click.Context,
    repo_urls: tuple[str, ...],
    repo_paths: tuple[str, ...],
    repo_zips: tuple[str, ...],
    urls_file: str | None,
) -> None:
    """Index multiple repositories at once."""
    sources: list[dict[str, object]] = []

    for url in repo_urls:
        sources.append({"source": url, "is_url": True, "is_zip": False})
    for path in repo_paths:
        sources.append({"source": path, "is_url": False, "is_zip": False})
    for zip_path in repo_zips:
        sources.append({"source": zip_path, "is_url": False, "is_zip": True})

    if urls_file:
        try:
            with open(urls_file) as f:
                for line in f:
                    url = line.strip()
                    if url and not url.startswith("#"):
                        sources.append({"source": url, "is_url": True, "is_zip": False})
        except Exception as e:
            click.echo(f"Error reading URLs file: {e}", err=True)
            sys.exit(1)

    if not sources:
        click.echo(
            "Error: No repositories specified. Use --repo-urls, --repo-paths, --repo-zips, or --urls-file",
            err=True,
        )
        sys.exit(1)

    client = ctx.obj["client"]

    try:
        click.echo(f"Loading and indexing {len(sources)} repositories...")
        result = client.index_multiple(sources)
        click.echo(f"\n{result.get('message', 'Done')}")

        stats = result.get("stats", {})
        if stats:
            click.echo(f"Total Repositories: {stats.get('total_repositories', '?')}")
            click.echo(f"Total Elements: {stats.get('total_elements', '?')}")

    except Exception as e:
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


@cli.command("query-multiple")
@click.option("--query", "-q", required=True, help="Question to ask")
@click.option(
    "--repos",
    "-r",
    multiple=True,
    help="Specific repositories to search",
)
@click.option("--output", "-o", help="Output file (default: stdout)")
@click.option("--load-cache", is_flag=True, help="Load from multi-repo cache")
@click.pass_context
def query_multiple(
    ctx: click.Context,
    query: str,
    repos: tuple[str, ...],
    output: str | None,
    load_cache: bool,
) -> None:
    """Query across multiple indexed repositories."""
    client = ctx.obj["client"]

    try:
        if load_cache:
            click.echo("Loading multi-repository index from cache...")
            client.load_cached_repos()
        else:
            click.echo(
                "Error: No repositories loaded. Use 'index-multiple' first or use --load-cache",
                err=True,
            )
            sys.exit(1)

        repo_filter = list(repos) if repos else None
        click.echo(f"\nProcessing query: {query}\n")
        result = client.query(query, repo_filter=repo_filter)

        answer = result.get("answer", "")
        if output:
            with open(output, "w") as f:
                f.write(answer)
            click.echo(f"Result saved to {output}")
        else:
            click.echo(answer)

    except Exception as e:
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


# ======================================================================
# Info commands
# ======================================================================


@cli.command("list-repos")
@click.pass_context
def list_repos(ctx: click.Context) -> None:
    """List all indexed repositories."""
    client = ctx.obj["client"]

    try:
        result = client.list_repositories()
        available = result.get("available", [])

        if not available:
            click.echo("No repository indexes found")
            return

        click.echo("=" * 80)
        click.echo("Available Repository Indexes")
        click.echo("=" * 80)

        for i, repo in enumerate(available, 1):
            click.echo(f"\n{i}. {repo.get('name', '?')}")
            click.echo(f"   Elements: {repo.get('element_count', '?')}")
            click.echo(f"   Files: {repo.get('file_count', '?')}")
            click.echo(f"   Size: {repo.get('size_mb', 0):.2f} MB")

        click.echo("\n" + "=" * 80)
        click.echo(f"Total: {len(available)} repositories")

    except Exception as e:
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


@cli.command("repo-stats")
@click.pass_context
def repo_stats(ctx: click.Context) -> None:
    """Show statistics for all indexed repositories."""
    client = ctx.obj["client"]

    try:
        result = client.list_repositories()
        available = result.get("available", [])

        if not available:
            click.echo("=" * 60)
            click.echo("Repository Statistics")
            click.echo("=" * 60)
            click.echo("Total Repositories: 0")
            click.echo("Total Indexed Elements: 0")
            return

        total_elements = sum(r.get("element_count", 0) for r in available)
        total_size = sum(r.get("size_mb", 0) for r in available)

        click.echo("=" * 60)
        click.echo("Repository Statistics")
        click.echo("=" * 60)
        click.echo(f"Total Repositories: {len(available)}")
        click.echo(f"Total Indexed Elements: {total_elements}")
        click.echo(f"Total Index Size: {total_size:.2f} MB")
        click.echo("\nPer-Repository Breakdown:")

        for repo in available:
            click.echo(f"\n  {repo.get('name', '?')}:")
            click.echo(f"    Elements: {repo.get('element_count', '?')}")
            click.echo(f"    Files: {repo.get('file_count', '?')}")
            click.echo(f"    Size: {repo.get('size_mb', 0):.2f} MB")

    except Exception as e:
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


# ======================================================================
# Session commands
# ======================================================================


@cli.command("list-sessions")
@click.pass_context
def list_sessions(ctx: click.Context) -> None:
    """List all dialogue sessions."""
    client = ctx.obj["client"]

    try:
        result = client.list_sessions()
        sessions = result.get("sessions", [])

        if not sessions:
            click.echo("No dialogue sessions found")
            return

        click.echo("=" * 80)
        click.echo("Dialogue Sessions")
        click.echo("=" * 80)

        from datetime import datetime

        for i, session in enumerate(sessions, 1):
            session_id = session.get("session_id", "Unknown")
            total_turns = session.get("total_turns", 0)
            created_at = session.get("created_at", 0)
            last_updated = session.get("last_updated", 0)

            created_str = datetime.fromtimestamp(created_at).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            updated_str = datetime.fromtimestamp(last_updated).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            click.echo(f"\n{i}. Session ID: {session_id}")
            click.echo(f"   Total Turns: {total_turns}")
            click.echo(f"   Created: {created_str}")
            click.echo(f"   Last Updated: {updated_str}")

        click.echo("\n" + "=" * 80)
        click.echo(f"Total Sessions: {len(sessions)}")

    except Exception as e:
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


@cli.command("show-session")
@click.argument("session_id")
@click.pass_context
def show_session(ctx: click.Context, session_id: str) -> None:
    """Show dialogue history for a session."""
    client = ctx.obj["client"]

    try:
        result = client.get_session(session_id)
        history = result.get("history", [])

        if not history:
            click.echo(f"No history found for session: {session_id}")
            return

        click.echo("=" * 80)
        click.echo(f"Session: {session_id}")
        click.echo("=" * 80)

        from datetime import datetime

        for turn in history:
            turn_num = turn.get("turn_number", 0)
            query = turn.get("query", "")
            answer = turn.get("answer", "")
            timestamp = turn.get("timestamp", 0)
            time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

            click.echo(f"\n{'=' * 80}")
            click.echo(f"Turn {turn_num} ({time_str})")
            click.echo(f"{'=' * 80}")
            click.echo(f"\nQuestion: {query}")
            click.echo(f"\nAnswer:\n{answer}")

        click.echo("\n" + "=" * 80)
        click.echo(f"Total Turns: {len(history)}")

    except Exception as e:
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


@cli.command("delete-session")
@click.argument("session_id")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def delete_session(ctx: click.Context, session_id: str, confirm: bool) -> None:
    """Delete a dialogue session."""
    client = ctx.obj["client"]

    try:
        result = client.get_session(session_id)
        history = result.get("history", [])

        if not history:
            click.echo(f"Session not found: {session_id}", err=True)
            sys.exit(1)

        if not confirm:
            click.echo(f"Session ID: {session_id}")
            click.echo(f"Total turns: {len(history)}")
            if not click.confirm("\nAre you sure you want to delete this session?"):
                click.echo("Cancelled.")
                return

        client.delete_session(session_id)
        click.echo(f"Successfully deleted session: {session_id}")

    except Exception as e:
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


# ======================================================================
# Cache commands
# ======================================================================


@cli.command("clear-cache")
@click.pass_context
def clear_cache(ctx: click.Context) -> None:
    """Clear all cached data."""
    client = ctx.obj["client"]

    try:
        client.clear_cache()
        click.echo("Cache cleared successfully")
    except Exception as e:
        click.echo(f"Failed to clear cache: {e}", err=True)


@cli.command("cache-stats")
@click.pass_context
def cache_stats(ctx: click.Context) -> None:
    """Show cache statistics."""
    client = ctx.obj["client"]

    try:
        stats = client.cache_stats()

        click.echo("=" * 60)
        click.echo("Query Result Cache Statistics")
        click.echo("=" * 60)
        click.echo(f"Enabled: {stats.get('enabled', False)}")

        if stats.get("enabled"):
            click.echo(f"Backend: {stats.get('backend', 'unknown')}")
            click.echo(f"Cached Queries: {stats.get('items', 0)}")
            size_mb = stats.get("size", 0) / (1024 * 1024)
            click.echo(f"Cache Size: {size_mb:.2f} MB")

    except Exception as e:
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


# ======================================================================
# Diagnostics
# ======================================================================


@cli.command()
@click.pass_context
def health(ctx: click.Context) -> None:
    """Check server health."""
    client = ctx.obj["client"]

    try:
        result = client.health()
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Server unreachable: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def diagnostics(ctx: click.Context) -> None:
    """Get diagnostic bundle from server."""
    client = ctx.obj["client"]

    try:
        result = client.diagnostics()
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
