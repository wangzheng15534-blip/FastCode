"""Runtime configuration loading for the FastCode composition layer."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv

from fastcode.kernel.config import FastCodeConfig
from fastcode.main.config_mapper import config_from_mapping, config_to_dict
from fastcode.runtime_support.observability import setup_logging_from_config


def setup_logging(config: dict[str, Any]) -> logging.Logger:
    """Set up process logging from runtime config."""
    return setup_logging_from_config(config, logger_name="fastcode")


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Load runtime config and return the shell runtime mapping."""
    return config_to_runtime_mapping(load_runtime_config(config_path))


def load_runtime_config(config_path: str = "config/config.yaml") -> FastCodeConfig:
    """Load YAML config into the canonical frozen runtime config."""
    with open(config_path) as f:
        config = cast(dict[str, Any], yaml.safe_load(f) or {})

    config_file = Path(config_path).resolve()
    if config_file.parent.name == "config":
        project_root = config_file.parent.parent
    else:
        project_root = config_file.parent

    resolved = prepare_runtime_config_mapping(
        config,
        project_root=str(project_root),
        config_dir=str(config_file.parent),
    )
    return config_from_mapping(resolved)


def config_to_runtime_mapping(config: FastCodeConfig) -> dict[str, Any]:
    """Convert frozen runtime config into a shell runtime mapping."""
    return config_to_dict(config)


def prepare_runtime_config_mapping(
    config: dict[str, Any],
    *,
    project_root: str,
    config_dir: str | None = None,
) -> dict[str, Any]:
    """Resolve paths and env-backed runtime overrides for a raw config mapping."""
    if config_dir:
        load_dotenv(Path(config_dir) / ".env")
    load_dotenv(Path(project_root) / ".env")
    resolved = resolve_config_paths(config, project_root)
    _apply_runtime_env_overrides(resolved)
    return resolved


def _apply_runtime_env_overrides(config: dict[str, Any]) -> None:
    storage_cfg = cast(dict[str, Any], config.setdefault("storage", {}))
    generation_cfg = cast(dict[str, Any], config.setdefault("generation", {}))
    cache_cfg = cast(dict[str, Any], config.setdefault("cache", {}))
    repository_cfg = cast(dict[str, Any], config.setdefault("repository", {}))
    projection_cfg = cast(dict[str, Any], config.setdefault("projection", {}))

    generation_env_overrides = {
        "model": os.getenv("MODEL"),
        "base_url": os.getenv("BASE_URL"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
    }
    for key, env_value in generation_env_overrides.items():
        if env_value:
            generation_cfg[key] = env_value

    if backend := os.getenv("FASTCODE_STORAGE_BACKEND"):
        storage_cfg["backend"] = backend
    if dsn := os.getenv("FASTCODE_POSTGRES_DSN"):
        storage_cfg["postgres_dsn"] = dsn
    if projection_dsn := os.getenv("FASTCODE_PROJECTION_POSTGRES_DSN"):
        projection_cfg["postgres_dsn"] = projection_dsn
    if redis_host := os.getenv("REDIS_HOST"):
        cache_cfg["redis_host"] = redis_host
    if redis_port := os.getenv("REDIS_PORT"):
        cache_cfg["redis_port"] = redis_port
    if exclude_site_packages := os.getenv("FASTCODE_EXCLUDE_SITE_PACKAGES"):
        repository_cfg["exclude_site_packages"] = exclude_site_packages.lower() in {
            "1",
            "true",
            "yes",
        }


def resolve_config_paths(config: dict[str, Any], project_root: str) -> dict[str, Any]:
    """Resolve relative directory/file paths in config to absolute paths."""
    if not config:
        return config

    root = os.path.abspath(project_root)

    def _abs(path_value: str | None) -> str | None:
        if not path_value:
            return path_value
        if os.path.isabs(path_value):
            return os.path.abspath(path_value)
        return os.path.abspath(os.path.join(root, path_value))

    if "repo_root" in config:
        config["repo_root"] = _abs(cast(str, config.get("repo_root")))

    vector_store_cfg = cast(dict[str, Any] | None, config.get("vector_store"))
    if isinstance(vector_store_cfg, dict) and "persist_directory" in vector_store_cfg:
        vector_store_cfg["persist_directory"] = _abs(
            cast(str, vector_store_cfg.get("persist_directory"))
        )

    repository_cfg = cast(dict[str, Any] | None, config.get("repository"))
    if isinstance(repository_cfg, dict) and "backup_directory" in repository_cfg:
        repository_cfg["backup_directory"] = _abs(
            cast(str, repository_cfg.get("backup_directory"))
        )

    cache_cfg = cast(dict[str, Any] | None, config.get("cache"))
    if isinstance(cache_cfg, dict) and "cache_directory" in cache_cfg:
        cache_cfg["cache_directory"] = _abs(cast(str, cache_cfg.get("cache_directory")))

    logging_cfg = cast(dict[str, Any] | None, config.get("logging"))
    if isinstance(logging_cfg, dict) and "file" in logging_cfg:
        logging_cfg["file"] = _abs(cast(str, logging_cfg.get("file")))

    return config
