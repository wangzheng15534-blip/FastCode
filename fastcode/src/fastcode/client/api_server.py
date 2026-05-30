"""Client entrypoint for the FastCode HTTP API server."""

from __future__ import annotations

from fastcode.api.routes import main as _api_main


def main() -> None:
    """Run the API server command."""
    _api_main()
