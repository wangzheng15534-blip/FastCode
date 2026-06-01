"""Client entrypoint for the FastCode web server."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Run the web server command."""
    from fastcode.main.serve import create_web_app

    uvicorn.run(create_web_app(), host="127.0.0.1", port=5777)
