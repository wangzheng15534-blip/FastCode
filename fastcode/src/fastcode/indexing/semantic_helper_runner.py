"""Shell-side process runner for semantic helper tools."""

from __future__ import annotations

import subprocess
from typing import Any


def run_helper_command(
    command: list[str],
    *,
    cwd: str,
    timeout: int,
) -> Any:
    """Run an external semantic helper command."""
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(str(exc)) from exc
