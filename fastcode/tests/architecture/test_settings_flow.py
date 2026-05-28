"""Settings/config boundary checks for FP/FCIS-style dataflow."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "fastcode"

NO_ENV_PACKAGES = [
    "ir",
    "graph",
    "app/indexing",
    "mcp",
    "app/query",
    "retrieval/core",
    "retrieval",
    "kernel",
    "runtime_support",
    "app/store",
    "infrastructure",
    "schemas",
]


def _iter_python_files(package_path: str) -> list[Path]:
    pkg_dir = PACKAGE_ROOT / package_path
    if not pkg_dir.is_dir():
        return []
    files = sorted(pkg_dir.rglob("*.py"))
    if package_path == "retrieval":
        core_pkg = PACKAGE_ROOT / "retrieval" / "core"
        files = [path for path in files if not path.is_relative_to(core_pkg)]
    return files


def _has_env_loading(filepath: Path) -> list[str]:
    tree = ast.parse(filepath.read_text())
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "dotenv":
            violations.append(
                f"{filepath.relative_to(PACKAGE_ROOT)}:{node.lineno}: dotenv import"
            )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "load_dotenv":
                violations.append(
                    f"{filepath.relative_to(PACKAGE_ROOT)}:{node.lineno}: load_dotenv()"
                )
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr in {"getenv"}
            ):
                violations.append(
                    f"{filepath.relative_to(PACKAGE_ROOT)}:{node.lineno}: os.getenv()"
                )
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "os"
                and node.func.value.attr == "environ"
                and node.func.attr == "get"
            ):
                violations.append(
                    f"{filepath.relative_to(PACKAGE_ROOT)}:{node.lineno}: os.environ.get()"
                )
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
        ):
            violations.append(
                f"{filepath.relative_to(PACKAGE_ROOT)}:{node.lineno}: os.environ[...]"
            )
    return violations


def test_inner_packages_do_not_load_env_or_dotenv() -> None:
    """Inner packages must receive settings explicitly, not read env directly."""
    violations: list[str] = []
    for package_path in NO_ENV_PACKAGES:
        for py_file in _iter_python_files(package_path):
            violations.extend(_has_env_loading(py_file))
    assert not violations, (
        "Settings/env access must stay at shell boundaries or explicit config adapters:\n"
        + "\n".join(violations)
    )
