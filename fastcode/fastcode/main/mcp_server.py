"""Server entrypoint for the FastCode MCP server (default axis)."""

from __future__ import annotations

from fastcode.mcp.server import main as _mcp_main


def main() -> None:
    """Run the MCP server command."""
    _mcp_main()
