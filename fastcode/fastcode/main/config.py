"""Composition-root config loading, merge, validation, and local application."""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv

from fastcode.common.config import FastCodeConfig
from fastcode.main._config_runtime import (
    config_from_dto,
    config_from_mapping,
    config_to_dict,
)
from fastcode.main.defaults import get_default_config
from fastcode.runtime_support.observability import setup_logging_from_config

__all__ = [
    "bootstrap_runtime_config",
    "config_from_dto",
    "config_from_mapping",
    "config_to_dict",
    "config_to_runtime_mapping",
    "load_config",
    "load_runtime_config",
    "prepare_runtime_config_mapping",
    "resolve_config_paths",
    "setup_logging",
]


def bootstrap_runtime_config(
    config_path: str | None, project_root: str
) -> FastCodeConfig:
    """Resolve a config path (or fall back to defaults) into the frozen runtime config.

    Single entry point for the assembly root so it never names the individual
    config-ingress loader functions directly.
    """
    if config_path and os.path.exists(config_path):
        return load_runtime_config(config_path)
    raw_default_config = get_default_config()
    resolved_default_config = prepare_runtime_config_mapping(
        raw_default_config,
        project_root=project_root,
    )
    return config_from_mapping(resolved_default_config)


def setup_logging(config: dict[str, Any]) -> logging.Logger:
    """Set up process logging from runtime config."""
    return setup_logging_from_config(config, logger_name="fastcode")


def apply_darwin_threading_env() -> None:
    """Pin thread counts on macOS so tokenizers/BLAS do not oversubscribe.

    Must run before tokenizers/BLAS import. No-op on non-Darwin platforms.
    """
    if platform.system() != "Darwin":
        return
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


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
    """Apply env var overrides to the config dict.

    Uses the env registry for all reads. Env vars with a config_path
    are applied to the corresponding nested key in the config dict.
    Precedence: CLI flag > env var > config YAML > default.
    """
    from fastcode.main._env_registry import ENV_REGISTRY, read_env

    for spec in ENV_REGISTRY.values():
        if not spec.config_path:
            continue
        value = read_env(spec.name)
        if value is None:
            continue

        parts = spec.config_path.split(".")
        target: dict[str, Any] = config
        for part in parts[:-1]:
            target = cast(dict[str, Any], target.setdefault(part, {}))

        key = parts[-1]
        if spec.name == "FASTCODE_FORCE_EXCLUDE_SITE_PACKAGES":
            target[key] = value.lower() in {"1", "true", "yes"}
        else:
            target[key] = value


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
