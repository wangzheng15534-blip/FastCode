"""Inbound DTOs for operational shell config sections."""

from __future__ import annotations

from pydantic import Field

from ._config_schema_base import _ConfigDTO


class EvaluationConfigDTO(_ConfigDTO):
    enabled: bool = False
    in_memory_index: bool = False
    disable_cache: bool = False
    disable_persistence: bool = False
    force_reindex: bool = False


class LoggingConfigDTO(_ConfigDTO):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/fastcode.log"
    console: bool = True


class TerminusConfigDTO(_ConfigDTO):
    endpoint: str = ""
    api_key: str = ""
    timeout_seconds: int = Field(default=15, ge=1)
