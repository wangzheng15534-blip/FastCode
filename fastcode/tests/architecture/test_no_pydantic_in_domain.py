"""Domain/common packages use frozen dataclasses, never pydantic."""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "fastcode"
PYDANTIC_FREE_PACKAGES = ["ir", "graph", "retrieval"]


def _imports_pydantic(filepath: Path) -> bool:
    tree = ast.parse(filepath.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and "pydantic" in node.module
        ):
            return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "pydantic" in alias.name:
                    return True
    return False


def test_no_pydantic_in_domain_or_common_packages():
    """Files in domain/common packages must not import pydantic."""
    violations = []
    for pkg in PYDANTIC_FREE_PACKAGES:
        pkg_dir = PACKAGE_ROOT / pkg
        if not pkg_dir.is_dir():
            continue
        for py_file in pkg_dir.rglob("*.py"):
            if _imports_pydantic(py_file):
                violations.append(str(py_file.relative_to(PACKAGE_ROOT)))
    assert not violations, f"Pydantic found in domain/common packages: {violations}"
