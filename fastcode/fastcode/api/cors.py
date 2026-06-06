"""CORS configuration helpers for API shells."""

from __future__ import annotations

import os
from typing import TypedDict


class CorsMiddlewareOptions(TypedDict):
    allow_origins: list[str]
    allow_credentials: bool
    allow_methods: list[str]
    allow_headers: list[str]


_DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://localhost",
    "http://localhost:8000",
)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return list(_DEFAULT_CORS_ORIGINS)
    origins = [item.strip() for item in value.split(",") if item.strip()]
    return origins or list(_DEFAULT_CORS_ORIGINS)


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def cors_middleware_options() -> CorsMiddlewareOptions:
    """Return Starlette CORSMiddleware options from explicit environment config."""
    origins = _split_csv(os.getenv("FASTCODE_CORS_ALLOW_ORIGINS"))
    allow_credentials = _parse_bool(
        os.getenv("FASTCODE_CORS_ALLOW_CREDENTIALS"), default=False
    )
    if "*" in origins and allow_credentials:
        allow_credentials = False
    return {
        "allow_origins": origins,
        "allow_credentials": allow_credentials,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
