"""Benchmark collection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def pytest_collect_file(file_path: Path, parent: Any) -> pytest.Module | None:
    if file_path.name.startswith("bench_") and file_path.suffix == ".py":
        return pytest.Module.from_parent(parent, path=file_path)
    return None
