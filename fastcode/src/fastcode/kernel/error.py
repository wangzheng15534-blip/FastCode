"""Shared kernel validation errors."""

from __future__ import annotations


class KernelError(ValueError):
    """Error raised when shared kernel identity validation fails."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = str(code)
        self.reason = str(reason)
        super().__init__(f"{self.code}: {self.reason}")

    @classmethod
    def invalid_repo_name(cls, reason: str) -> KernelError:
        return cls("invalid_repo_name", reason)

    @classmethod
    def invalid_snapshot_id(cls, reason: str) -> KernelError:
        return cls("invalid_snapshot_id", reason)

    @classmethod
    def invalid_artifact_key(cls, reason: str) -> KernelError:
        return cls("invalid_artifact_key", reason)
