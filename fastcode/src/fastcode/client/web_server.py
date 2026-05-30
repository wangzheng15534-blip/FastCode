"""Client entrypoint for the FastCode web server."""

from __future__ import annotations

from fastcode.api.web import main as _web_main


def main() -> None:
    """Run the web server command."""
    _web_main()
