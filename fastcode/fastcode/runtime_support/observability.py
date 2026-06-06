"""Generic logging setup helpers."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

_DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(
    *,
    level: int | str = "INFO",
    format_str: str = _DEFAULT_LOG_FORMAT,
    log_file: str = "./logs/fastcode.log",
    console: bool = True,
    logger_name: str = "fastcode",
) -> logging.Logger:
    """Configure process logging with explicit file/console settings."""
    resolved_level = level if isinstance(level, int) else getattr(logging, str(level))
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    handlers: list[logging.Handler] = []
    if console:
        handlers.append(logging.StreamHandler())
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=resolved_level, format=format_str, handlers=handlers)
    return logging.getLogger(logger_name)


def setup_logging_from_config(
    config: Mapping[str, Any],
    *,
    logger_name: str = "fastcode",
) -> logging.Logger:
    """Configure root logging from a generic runtime mapping."""
    log_config = dict(config.get("logging", {}))
    return configure_logging(
        level=str(log_config.get("level", "INFO")),
        format_str=str(log_config.get("format", _DEFAULT_LOG_FORMAT)),
        log_file=str(log_config.get("file", "./logs/fastcode.log")),
        console=bool(log_config.get("console", True)),
        logger_name=logger_name,
    )
