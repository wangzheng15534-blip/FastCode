"""Verify pure packages and package roots don't import banned modules."""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "fastcode"

PURE_PACKAGES = ["ir", "graph", "retrieval"]
BANNED_MODULES = {"pydantic", "sqlite3", "subprocess", "urllib"}
PACKAGE_ROOTS = [
    PACKAGE_ROOT / "__init__.py",
    *sorted((PACKAGE_ROOT / pkg / "__init__.py") for pkg in PACKAGE_ROOT.iterdir() if pkg.is_dir()),
]


def _get_stdlib_imports(filepath: Path) -> list[str]:
    tree = ast.parse(filepath.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
    return imports


def test_pure_packages_no_banned_imports():
    """Files in pure packages must not import banned modules."""
    violations = []
    for pkg in PURE_PACKAGES:
        pkg_dir = PACKAGE_ROOT / pkg
        if not pkg_dir.is_dir():
            continue
        for py_file in pkg_dir.rglob("*.py"):
            for mod in _get_stdlib_imports(py_file):
                if mod in BANNED_MODULES:
                    violations.append(
                        f"{py_file.relative_to(PACKAGE_ROOT)}: imports {mod}"
                    )
    assert not violations, "Banned imports in pure packages:\n" + "\n".join(violations)


def test_package_roots_do_not_import_root_compat_exports():
    """Internal package roots must not depend on root compatibility re-exports."""
    violations = []
    for py_file in PACKAGE_ROOTS:
        if not py_file.exists():
            continue
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module == "fastcode"
            ):
                violations.append(
                    f"{py_file.relative_to(PACKAGE_ROOT)}:{node.lineno}: from fastcode import ..."
                )
    assert not violations, (
        "Package roots must import concrete modules directly, not the root "
        "compatibility surface:\n" + "\n".join(violations)
    )
