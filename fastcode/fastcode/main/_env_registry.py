"""Centralized environment variable registry for FastCode.

All env var reads go through read_env(), which provides:
  - Canonical verb-prefixed names (FASTCODE_FORCE_*, FASTCODE_DEBUG_*, etc.)
  - Backward-compatible old-name fallback with deprecation warnings
  - A single source of truth for all recognized env vars

Precedence: CLI flag > env var > config YAML > default
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_THIRD_PARTY_WHITELIST = frozenset({
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
})

_VERB_PREFIXES = (
    "FASTCODE_FORCE_",
    "FASTCODE_DEBUG_",
    "FASTCODE_TEST_",
    "FASTCODE_LOG_",
    "FASTCODE_ENABLE_",
    "FASTCODE_DISABLE_",
    "FASTCODE_OPT_",
    "FASTCODE_CORS_",
    "FASTCODE_STORAGE_",
    "FASTCODE_POSTGRES_",
    "FASTCODE_PROJECTION_",
    "FASTCODE_SERVER",
)


@dataclass(frozen=True)
class EnvVarSpec:
    """Declaration of a recognized environment variable."""

    name: str
    old_names: tuple[str, ...] = ()
    category: str = "FORCE"
    config_path: str | None = None
    description: str = ""


ENV_REGISTRY: dict[str, EnvVarSpec] = {}


def _register(spec: EnvVarSpec) -> None:
    ENV_REGISTRY[spec.name] = spec


# ---------------------------------------------------------------------------
# Registry entries
# ---------------------------------------------------------------------------

_register(EnvVarSpec(
    name="FASTCODE_FORCE_LLM_MODEL",
    old_names=("MODEL",),
    category="FORCE",
    config_path="generation.model",
    description="Override the LLM model name",
))
_register(EnvVarSpec(
    name="FASTCODE_FORCE_LLM_BASE_URL",
    old_names=("BASE_URL",),
    category="FORCE",
    config_path="generation.base_url",
    description="Override the LLM base URL",
))
_register(EnvVarSpec(
    name="FASTCODE_FORCE_REDIS_HOST",
    old_names=("REDIS_HOST",),
    category="FORCE",
    config_path="cache.redis_host",
    description="Override Redis cache host",
))
_register(EnvVarSpec(
    name="FASTCODE_FORCE_REDIS_PORT",
    old_names=("REDIS_PORT",),
    category="FORCE",
    config_path="cache.redis_port",
    description="Override Redis cache port",
))
_register(EnvVarSpec(
    name="FASTCODE_STORAGE_BACKEND",
    category="FORCE",
    config_path="storage.backend",
    description="Storage backend: sqlite or postgres",
))
_register(EnvVarSpec(
    name="FASTCODE_POSTGRES_DSN",
    category="FORCE",
    config_path="storage.postgres_dsn",
    description="PostgreSQL connection string",
))
_register(EnvVarSpec(
    name="FASTCODE_PROJECTION_POSTGRES_DSN",
    category="FORCE",
    config_path="projection.postgres_dsn",
    description="Projection store PostgreSQL DSN",
))
_register(EnvVarSpec(
    name="FASTCODE_FORCE_EXCLUDE_SITE_PACKAGES",
    old_names=("FASTCODE_EXCLUDE_SITE_PACKAGES",),
    category="FORCE",
    config_path="repository.exclude_site_packages",
    description="Exclude vendored site-packages from indexing",
))
_register(EnvVarSpec(
    name="FASTCODE_CORS_ALLOW_ORIGINS",
    category="FORCE",
    description="Comma-separated CORS allowed origins",
))
_register(EnvVarSpec(
    name="FASTCODE_CORS_ALLOW_CREDENTIALS",
    category="FORCE",
    description="Allow CORS credentials (true/false)",
))
_register(EnvVarSpec(
    name="FASTCODE_SERVER",
    category="FORCE",
    description="FastCode API server URL",
))
_register(EnvVarSpec(
    name="OPENAI_API_KEY",
    category="THIRD_PARTY",
    config_path="generation.openai_api_key",
    description="OpenAI API key (third-party)",
))
_register(EnvVarSpec(
    name="ANTHROPIC_API_KEY",
    category="THIRD_PARTY",
    config_path="generation.anthropic_api_key",
    description="Anthropic API key (third-party)",
))


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def read_env(name: str) -> str | None:
    """Read an env var by canonical name, with old-name fallback.

    Returns the value or None. Emits DeprecationWarning when the old
    name is set but the canonical name is not.
    """
    spec = ENV_REGISTRY.get(name)
    value = os.environ.get(name)
    if value is not None:
        return value

    if spec and spec.old_names:
        for old in spec.old_names:
            old_value = os.environ.get(old)
            if old_value is not None:
                warnings.warn(
                    f"Env var {old!r} is deprecated; use {spec.name!r} instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return old_value

    return None


def validate_env_vars() -> list[str]:
    """Return warnings for env vars that are unrecognized or deprecated."""
    results: list[str] = []

    for key in sorted(os.environ):
        # Skip known third-party vars
        if key in _THIRD_PARTY_WHITELIST:
            continue

        # Check FASTCODE_-prefixed vars that aren't in the registry
        if key.startswith("FASTCODE_") and key not in ENV_REGISTRY:
            # Allow verb-prefixed vars not yet registered
            if not key.startswith(_VERB_PREFIXES):
                results.append(f"Unrecognized env var: {key}")
            continue

        # Check old-name usage
        for spec in ENV_REGISTRY.values():
            if key in spec.old_names:
                results.append(
                    f"Deprecated env var: {key} -> use {spec.name} instead"
                )

    return results
