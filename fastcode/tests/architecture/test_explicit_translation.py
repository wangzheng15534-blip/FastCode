"""No **model_dump() or **dict() mass-assignment at boundaries."""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "fastcode"
SHELL_PACKAGES = ["api", "mcp", "main"]


def _check_file_for_patterns(filepath: Path) -> list[str]:
    source = filepath.read_text()
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg is None:
            # This is a **kwargs usage
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "model_dump"
            ):
                violations.append(
                    f"{filepath.relative_to(PACKAGE_ROOT)}:{node.lineno}: **model_dump()"
                )
            elif (
                isinstance(node.value, ast.Attribute) and node.value.attr == "__dict__"
            ):
                violations.append(
                    f"{filepath.relative_to(PACKAGE_ROOT)}:{node.lineno}: **__dict__"
                )
    return violations


def test_no_mass_assignment_at_boundaries():
    """Shell packages must not use **model_dump() or **__dict__."""
    violations = []
    for pkg in SHELL_PACKAGES:
        pkg_dir = PACKAGE_ROOT / pkg
        if not pkg_dir.is_dir():
            continue
        for py_file in pkg_dir.rglob("*.py"):
            violations.extend(_check_file_for_patterns(py_file))
    assert not violations, "Mass-assignment patterns found:\n" + "\n".join(violations)
