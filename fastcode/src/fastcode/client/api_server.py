"""Client entrypoint for the FastCode HTTP API server."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Run the API server command."""
    from fastcode.main.serve import create_api_app

    uvicorn.run(create_api_app(), host="127.0.0.1", port=8000)
