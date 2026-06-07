"""Capability lifecycle registry for tracking feature stages."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum


class CapabilityStage(Enum):
    """Lifecycle stages for a product capability."""

    EXPERIMENTAL = "experimental"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    REMOVED = "removed"


@dataclass(frozen=True)
class CapabilitySpec:
    """Declaration of a trackable capability."""

    name: str
    stage: CapabilityStage
    config_key: str | None = None
    introduced_in: str | None = None
    deprecated_in: str | None = None
    removed_in: str | None = None
    replacement: str | None = None
    description: str = ""


class _Registry:
    """Module-level singleton capability registry."""

    def __init__(self) -> None:
        self._specs: dict[str, CapabilitySpec] = {}
        self._warned_experimental: set[str] = set()

    def register(self, spec: CapabilitySpec) -> None:
        if spec.name in self._specs:
            msg = f"Capability already registered: {spec.name}"
            raise ValueError(msg)
        self._specs[spec.name] = spec

    def get(self, name: str) -> CapabilitySpec:
        try:
            return self._specs[name]
        except KeyError:
            raise CapabilityLookupError(name) from None

    def check(self, name: str) -> None:
        """Validate a capability is usable at its current stage.

        - REMOVED: raises CapabilityRemovedError
        - DEPRECATED: emits DeprecationWarning
        - EXPERIMENTAL: emits UserWarning on first check
        - STABLE: no-op
        """
        spec = self.get(name)
        if spec.stage == CapabilityStage.REMOVED:
            raise CapabilityRemovedError(spec)
        if spec.stage == CapabilityStage.DEPRECATED:
            msg = (
                f"Capability {spec.name!r} is deprecated"
                f"{f' since {spec.deprecated_in}' if spec.deprecated_in else ''}"
            )
            if spec.replacement:
                msg += f"; use {spec.replacement!r} instead"
            if spec.removed_in:
                msg += f". Removal planned for {spec.removed_in}"
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
        if (
            spec.stage == CapabilityStage.EXPERIMENTAL
            and name not in self._warned_experimental
        ):
            self._warned_experimental.add(name)
            msg = f"Capability {spec.name!r} is experimental and may change without notice"
            warnings.warn(msg, UserWarning, stacklevel=3)

    def all_by_stage(self, stage: CapabilityStage) -> list[CapabilitySpec]:
        return [s for s in self._specs.values() if s.stage == stage]

    def all_capabilities(self) -> list[CapabilitySpec]:
        return list(self._specs.values())

    def clear(self) -> None:
        """Remove all registrations. For testing only."""
        self._specs.clear()
        self._warned_experimental.clear()


class CapabilityLookupError(KeyError):
    """Raised when referencing an unregistered capability."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(name)


class CapabilityRemovedError(RuntimeError):
    """Raised when using a capability that has been removed."""

    def __init__(self, spec: CapabilitySpec) -> None:
        self.spec = spec
        msg = f"Capability {spec.name!r} has been removed"
        if spec.replacement:
            msg += f"; use {spec.replacement!r} instead"
        super().__init__(msg)


#: Global capability registry instance.
CapabilityRegistry = _Registry()
