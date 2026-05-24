"""Shared Pydantic base for inbound config DTOs."""
# pyright: reportUnusedClass=false

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _ConfigDTO(BaseModel):
    model_config = ConfigDict(extra="allow")
