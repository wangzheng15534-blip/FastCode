"""MCP client — connects to a FastCode MCP server over stdio.

Provides an interactive REPL that calls MCP tools on a running FastCode
MCP server.  The server is spawned as a child process.
"""

from __future__ import annotations

import asyncio
import json
import sys

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def _run_interactive(server_command: list[str]) -> None:
    params = StdioServerParameters(command=server_command[0], args=server_command[1:])

    async with (
        stdio_client(params) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()

        tools_result = await session.list_tools()
        tools = {t.name: t for t in tools_result.tools}

        while True:
            try:
                line = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not line:
                continue
            if line.lower() in ("quit", "exit", "q"):
                break
            if line.lower() == "list":
                for _name, tool in sorted(tools.items()):
                    (tool.description or "").split("\n")[0][:80]
                continue

            parts = line.split(None, 1)
            tool_name = parts[0]
            if tool_name not in tools:
                continue

            args: dict[str, object] = {}
            if len(parts) > 1:
                try:
                    args = json.loads(parts[1])
                except json.JSONDecodeError:
                    continue

            result = await session.call_tool(tool_name, args)
            for content in result.content:
                if hasattr(content, "text"):
                    pass  # type: ignore[union-attr]
                else:
                    pass


def mcp_client_main(server_command: list[str] | None = None) -> None:
    """Entry point for the MCP client REPL."""
    if server_command is None:
        server_command = [sys.executable, "-m", "fastcode.mcp.server"]
    asyncio.run(_run_interactive(server_command))
