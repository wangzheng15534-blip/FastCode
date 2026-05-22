"""Enforce the strict FastCode layer dependency DAG."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "fastcode"

LAYERS = {
    "api": "FACADE",
    "mcp": "FACADE",
    "main": "FACADE",
    "indexing": "SHELL",
    "query": "SHELL",
    "store": "SHELL",
    "runtime": "RUNTIME",
    "inbound": "INBOUND_MAPPER",
    "schemas": "INBOUND_SCHEMA",
    "retrieval": "DOMAIN",
    "graph": "DOMAIN",
    "scip": "DOMAIN",
    "semantic": "DOMAIN",
    "ir": "COMMON",
    "utils": "BASE",
}

DENY = {
    "DOMAIN": {"FACADE", "SHELL", "INBOUND_MAPPER", "INBOUND_SCHEMA"},
    "INBOUND_SCHEMA": {"RUNTIME", "INBOUND_MAPPER", "DOMAIN", "FACADE", "SHELL"},
    "INBOUND_MAPPER": {"DOMAIN", "FACADE", "SHELL"},
    "RUNTIME": {"FACADE", "SHELL", "INBOUND_MAPPER", "INBOUND_SCHEMA", "DOMAIN"},
    "COMMON": {
        "RUNTIME",
        "INBOUND_MAPPER",
        "INBOUND_SCHEMA",
        "DOMAIN",
        "FACADE",
        "SHELL",
    },
    "BASE": {
        "RUNTIME",
        "INBOUND_MAPPER",
        "INBOUND_SCHEMA",
        "DOMAIN",
        "COMMON",
        "FACADE",
        "SHELL",
    },
    "SHELL": {"FACADE"},
}

DOMAIN_PACKAGES = ("graph", "retrieval", "scip", "semantic")
BANNED_DOMAIN_IMPORTS = {"pydantic", "sqlite3", "subprocess", "urllib"}
STDLIB_IMPORTS = set(sys.stdlib_module_names) | {"__future__"}


def _layer_for_module(module_path: str) -> str | None:
    top_level = module_path.split(".", 1)[0]
    return LAYERS.get(top_level)


def _importer_module_for_file(filepath: Path) -> str | None:
    rel = filepath.relative_to(PACKAGE_ROOT).with_suffix("")
    if rel.name == "__init__":
        parts = rel.parts[:-1]
    else:
        parts = rel.parts[:-1]
    return ".".join(parts) if parts else None


def _target_module_for_import_from(filepath: Path, node: ast.ImportFrom) -> str | None:
    if node.level:
        rel_parts = filepath.relative_to(PACKAGE_ROOT).with_suffix("").parts
        package_parts = list(rel_parts[:-1])
        base_parts = package_parts[: len(package_parts) - (node.level - 1)]
        if node.module:
            target_parts = [*base_parts, *node.module.split(".")]
        elif node.names:
            target_parts = [*base_parts, *node.names[0].name.split(".")]
        else:
            return None
        return ".".join(target_parts) if target_parts else None

    if not node.module:
        return None
    if node.module == "fastcode" and node.names:
        return node.names[0].name
    if node.module.startswith("fastcode."):
        return node.module.removeprefix("fastcode.")
    return None


def _tree(filepath: Path) -> ast.Module:
    return ast.parse(filepath.read_text())


def _iter_python_files() -> list[Path]:
    ignored_parts = {"__pycache__", "_stubs"}
    return [
        path
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if not ignored_parts.intersection(path.relative_to(PACKAGE_ROOT).parts)
    ]


def _get_fastcode_imports(filepath: Path) -> list[tuple[str, int, bool]]:
    imports: list[tuple[str, int, bool]] = []
    for node in ast.walk(_tree(filepath)):
        if isinstance(node, ast.ImportFrom):
            target = _target_module_for_import_from(filepath, node)
            is_absolute_fastcode = bool(
                node.level == 0 and node.module and node.module.startswith("fastcode")
            )
            if target:
                imports.append((target, node.lineno, is_absolute_fastcode))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("fastcode."):
                    imports.append(
                        (alias.name.removeprefix("fastcode."), node.lineno, True)
                    )
    return imports


def test_all_runtime_modules_are_classified() -> None:
    """Every runtime Python module must live in one of the explicit layers."""
    violations: list[str] = []
    for py_file in _iter_python_files():
        importer_module = _importer_module_for_file(py_file)
        if importer_module is None:
            continue
        if _layer_for_module(importer_module) is None:
            violations.append(str(py_file.relative_to(PACKAGE_ROOT)))
    assert not violations, "Unclassified runtime modules:\n" + "\n".join(violations)


def test_strict_layer_dag_imports() -> None:
    """Packages may only import within their layer or downward."""
    violations: list[str] = []
    for py_file in _iter_python_files():
        importer_module = _importer_module_for_file(py_file)
        if importer_module is None:
            continue
        importer_layer = _layer_for_module(importer_module)
        if importer_layer is None:
            continue
        for target, line, is_absolute_fastcode in _get_fastcode_imports(py_file):
            target_layer = _layer_for_module(target)
            if target_layer is None:
                continue
            if importer_layer == "BASE" and is_absolute_fastcode:
                violations.append(
                    f"{py_file.relative_to(PACKAGE_ROOT)}:{line}: "
                    f"BASE imports absolute fastcode.{target}"
                )
            if target_layer in DENY.get(importer_layer, set()):
                violations.append(
                    f"{py_file.relative_to(PACKAGE_ROOT)}:{line}: "
                    f"{importer_module} ({importer_layer}) imports "
                    f"{target} ({target_layer})"
                )
    assert not violations, "Layer DAG violations:\n" + "\n".join(violations)


def test_domain_contracts_use_frozen_dataclasses_not_pydantic() -> None:
    """Domain contracts are local frozen dataclasses and never Pydantic models."""
    violations: list[str] = []
    for package in DOMAIN_PACKAGES:
        contracts_file = PACKAGE_ROOT / package / "contracts.py"
        if not contracts_file.exists():
            violations.append(f"{package}/contracts.py is missing")
            continue
        tree = _tree(contracts_file)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "pydantic" in node.module:
                    violations.append(
                        f"{contracts_file.relative_to(PACKAGE_ROOT)}:{node.lineno}: pydantic import"
                    )
            if not isinstance(node, ast.ClassDef):
                continue
            dataclass_decorator = next(
                (
                    decorator
                    for decorator in node.decorator_list
                    if (
                        isinstance(decorator, ast.Call)
                        and getattr(decorator.func, "id", None) == "dataclass"
                    )
                    or getattr(decorator, "id", None) == "dataclass"
                ),
                None,
            )
            if dataclass_decorator is None:
                continue
            if not isinstance(dataclass_decorator, ast.Call):
                violations.append(
                    f"{contracts_file.relative_to(PACKAGE_ROOT)}:{node.lineno}: "
                    f"{node.name} dataclass must set frozen=True"
                )
                continue
            frozen = any(
                keyword.arg == "frozen"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in dataclass_decorator.keywords
            )
            if not frozen:
                violations.append(
                    f"{contracts_file.relative_to(PACKAGE_ROOT)}:{node.lineno}: "
                    f"{node.name} dataclass must set frozen=True"
                )
    assert not violations, "Domain contract violations:\n" + "\n".join(violations)


def test_domain_modules_contain_no_shell_io_imports() -> None:
    """Domain modules must not pull in obvious shell I/O libraries."""
    violations: list[str] = []
    for package in DOMAIN_PACKAGES:
        for py_file in (PACKAGE_ROOT / package).rglob("*.py"):
            for node in ast.walk(_tree(py_file)):
                imported_roots: list[str] = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.append(node.module.split(".", 1)[0])
                elif isinstance(node, ast.Import):
                    imported_roots.extend(
                        alias.name.split(".", 1)[0] for alias in node.names
                    )
                for imported_root in imported_roots:
                    if imported_root in BANNED_DOMAIN_IMPORTS:
                        violations.append(
                            f"{py_file.relative_to(PACKAGE_ROOT)}:{node.lineno}: "
                            f"imports {imported_root}"
                        )
    assert not violations, "Banned imports in domain modules:\n" + "\n".join(violations)


def test_base_uses_only_stdlib_imports() -> None:
    """The Base layer stays a small stdlib-only foundation."""
    violations: list[str] = []
    for py_file in (PACKAGE_ROOT / "utils").rglob("*.py"):
        for node in ast.walk(_tree(py_file)):
            imported_roots: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.level == 0:
                    imported_roots.append(node.module.split(".", 1)[0])
            elif isinstance(node, ast.Import):
                imported_roots.extend(alias.name.split(".", 1)[0] for alias in node.names)
            for imported_root in imported_roots:
                if imported_root not in STDLIB_IMPORTS:
                    violations.append(
                        f"{py_file.relative_to(PACKAGE_ROOT)}:{node.lineno}: "
                        f"imports non-stdlib module {imported_root}"
                    )
    assert not violations, "Non-stdlib imports in Base layer:\n" + "\n".join(
        violations
    )
