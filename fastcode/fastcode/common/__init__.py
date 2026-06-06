"""Shared identity vocabulary and frozen config contracts."""

from .error import KernelError  # noqa: F401
from .feature_lifecycle import (  # noqa: F401
    CapabilityRegistry,
    CapabilityRemovedError,
    CapabilitySpec,
    CapabilityStage,
)
from .types import CodeElementMeta  # noqa: F401
