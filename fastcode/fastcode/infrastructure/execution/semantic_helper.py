"""Infrastructure adapters for external process execution."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from typing import Any

from fastcode.infrastructure.execution.ports import SemanticHelperRuntime


class SubprocessSemanticHelperRuntime(SemanticHelperRuntime):
    """Subprocess-backed runtime for semantic helper tools."""

    def find_executable(self, executable: str) -> str | None:
        """Resolve an executable from the host PATH."""
        return shutil.which(executable)

    def run(self, command: Sequence[str], *, cwd: str, timeout: int) -> Any:
        """Run an external semantic helper command."""
        try:
            return subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(str(exc)) from exc
