"""Factory functions for creating entry-frame apps with injected facades.

This is the only module that imports both the composition root (FastCode)
and the entry-frame modules (api/, mcp/).  It constructs the FacadeContainer
and wires it into the apps so entry frames never import from main/.
"""

from __future__ import annotations

import logging
import os
import platform
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastcode.runtime_support.observability import configure_logging

from .facades import FacadeContainer, facade_container_from_fastcode
from .fastcode import FastCode

# ---------------------------------------------------------------------------
# Threading env — must run before tokenizers/BLAS import on macOS
# ---------------------------------------------------------------------------


def _apply_darwin_threading_env() -> None:
    if platform.system() == "Darwin":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"


# ---------------------------------------------------------------------------
# CORS config — reads env here (assembly_root), not in entry_frame
# ---------------------------------------------------------------------------


def _cors_options_from_env() -> dict[str, Any]:
    from fastcode.main._env_registry import read_env

    origins_raw = read_env("FASTCODE_CORS_ALLOW_ORIGINS") or ""
    origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
    allow_credentials_raw = read_env("FASTCODE_CORS_ALLOW_CREDENTIALS") or ""
    allow_credentials = allow_credentials_raw.lower() in ("true", "1", "yes")
    return {
        "allow_origins": origins or ["*"],
        "allow_credentials": allow_credentials,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }


# ---------------------------------------------------------------------------
# API app factory
# ---------------------------------------------------------------------------


def _create_app(
    *,
    config_path: str | None,
    router: Any,
    title: str,
    description: str,
    log_file: str,
    logger_name: str,
    startup_message: str,
    static_assets_dir: Path | None = None,
) -> FastAPI:
    """Shared factory for FastAPI apps with injected facades."""
    _apply_darwin_threading_env()

    fc = FastCode(config_path=config_path)
    facades = facade_container_from_fastcode(fc)

    log_dir = Path("./logs")
    logger = configure_logging(
        level=logging.INFO,
        format_str="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_file=str(log_dir / log_file),
        console=True,
        logger_name=logger_name,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(startup_message)
        app.state.facades = facades
        yield
        try:
            facades.shutdown()
        except Exception as e:
            logger.warning("FastCode shutdown hook failed: %s", e)
        logger.info("FastCode shutting down")

    app = FastAPI(
        title=title,
        description=description,
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(CORSMiddleware, **_cors_options_from_env())
    app.include_router(router)

    if static_assets_dir is not None and static_assets_dir.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount(
            "/assets", StaticFiles(directory=str(static_assets_dir)), name="assets"
        )

    return app


def create_api_app(config_path: str | None = None) -> FastAPI:
    """Build a fully-wired FastAPI REST API app."""
    from fastcode.api.routes import router as api_router

    return _create_app(
        config_path=config_path,
        router=api_router,
        title="FastCode API",
        description="Repository-Level Code Understanding System API",
        log_file="api.log",
        logger_name="fastcode.api",
        startup_message="FastCode API started — initializing system",
    )


def create_web_app(config_path: str | None = None) -> FastAPI:
    """Build a fully-wired web UI app."""
    from fastcode.api.web import router as web_router

    return _create_app(
        config_path=config_path,
        router=web_router,
        title="FastCode Web UI",
        description="FastCode Repository-Level Code Understanding - Web Interface",
        log_file="web.log",
        logger_name="fastcode.web",
        startup_message="FastCode Web UI started — initializing system",
        static_assets_dir=Path(__file__).parent.parent / "assets",
    )


# ---------------------------------------------------------------------------
# MCP facade factory
# ---------------------------------------------------------------------------


def create_mcp_facade(config_path: str | None = None) -> FacadeContainer:
    """Create a FacadeContainer for the MCP server."""
    fc = FastCode(config_path=config_path)
    return facade_container_from_fastcode(fc)
