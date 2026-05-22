"""Enforce the strict FastCode layer dependency DAG."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "fastcode"

PREFIX_LAYERS = {
    "ports": "PORTS",
    "store.contracts": "PORTS",
    "store.infrastructure": "INFRA",
}

TOP_LEVEL_LAYERS = {
    "api": "FACADE",
    "mcp": "FACADE",
    "main": "FACADE",
    "indexing": "APP_RUNTIME",
    "query": "APP_RUNTIME",
    "store": "APP_RUNTIME",
    "runtime": "RUNTIME_CONTRACT",
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
    "DOMAIN": {
        "FACADE",
        "APP_RUNTIME",
        "INFRA",
        "PORTS",
        "INBOUND_MAPPER",
        "INBOUND_SCHEMA",
    },
    "INBOUND_SCHEMA": {
        "RUNTIME_CONTRACT",
        "INBOUND_MAPPER",
        "DOMAIN",
        "PORTS",
        "FACADE",
        "APP_RUNTIME",
        "INFRA",
    },
    "INBOUND_MAPPER": {"DOMAIN", "PORTS", "FACADE", "APP_RUNTIME", "INFRA"},
    "RUNTIME_CONTRACT": {
        "FACADE",
        "APP_RUNTIME",
        "INFRA",
        "PORTS",
        "INBOUND_MAPPER",
        "INBOUND_SCHEMA",
        "DOMAIN",
    },
    "PORTS": {"FACADE", "APP_RUNTIME", "INFRA", "INBOUND_MAPPER", "INBOUND_SCHEMA"},
    "INFRA": {"FACADE", "APP_RUNTIME", "INBOUND_MAPPER", "INBOUND_SCHEMA", "DOMAIN"},
    "COMMON": {
        "RUNTIME_CONTRACT",
        "INBOUND_MAPPER",
        "INBOUND_SCHEMA",
        "DOMAIN",
        "PORTS",
        "FACADE",
        "APP_RUNTIME",
        "INFRA",
    },
    "BASE": {
        "RUNTIME_CONTRACT",
        "INBOUND_MAPPER",
        "INBOUND_SCHEMA",
        "DOMAIN",
        "COMMON",
        "PORTS",
        "FACADE",
        "APP_RUNTIME",
        "INFRA",
    },
    # App-runtime shell code is not yet fully separated from infra adapters.
    # Until that migration is complete, the hard gate only prevents it from
    # reaching back into entrypoint/composition facades.
    "APP_RUNTIME": {"FACADE"},
}

DOMAIN_PACKAGES = ("graph", "retrieval", "scip", "semantic")
BANNED_DOMAIN_IMPORTS = {"pydantic", "sqlite3", "subprocess", "urllib"}
BANNED_PORT_IMPORTS = {
    "anthropic",
    "dotenv",
    "faiss",
    "httpx",
    "numpy",
    "openai",
    "pydantic",
    "requests",
    "sqlite3",
    "subprocess",
    "urllib",
}
STDLIB_IMPORTS = set(sys.stdlib_module_names) | {"__future__"}


def _layer_for_module(module_path: str) -> str | None:
    for prefix, layer in sorted(
        PREFIX_LAYERS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if module_path == prefix or module_path.startswith(f"{prefix}."):
            return layer
    top_level = module_path.split(".", 1)[0]
    return TOP_LEVEL_LAYERS.get(top_level)


def _importer_module_for_file(filepath: Path) -> str | None:
    rel = filepath.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
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


def _imported_roots(filepath: Path) -> list[tuple[str, int]]:
    roots: list[tuple[str, int]] = []
    for node in ast.walk(_tree(filepath)):
        if isinstance(node, ast.ImportFrom) and node.module:
            roots.append((node.module.split(".", 1)[0], node.lineno))
        elif isinstance(node, ast.Import):
            roots.extend(
                (alias.name.split(".", 1)[0], node.lineno) for alias in node.names
            )
    return roots


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


def test_fcis_shell_subroles_are_explicitly_classified() -> None:
    """The shell is split into app-runtime, infra, and capability-port roles."""
    expected = {
        "indexing.pipeline": "APP_RUNTIME",
        "query.retriever": "APP_RUNTIME",
        "store.snapshot": "APP_RUNTIME",
        "store.infrastructure.db": "INFRA",
        "store.contracts": "PORTS",
    }

    violations = [
        f"{module}: expected {layer}, got {_layer_for_module(module)}"
        for module, layer in expected.items()
        if _layer_for_module(module) != layer
    ]
    assert not violations, "FCIS shell role classification drift:\n" + "\n".join(
        violations
    )


def test_capability_ports_are_contract_only() -> None:
    """Ports define capabilities; infrastructure and validation stay elsewhere."""
    violations: list[str] = []
    for py_file in _iter_python_files():
        importer_module = _importer_module_for_file(py_file)
        if importer_module is None or _layer_for_module(importer_module) != "PORTS":
            continue
        for imported_root, line in _imported_roots(py_file):
            if imported_root in BANNED_PORT_IMPORTS:
                violations.append(
                    f"{py_file.relative_to(PACKAGE_ROOT)}:{line}: "
                    f"port imports {imported_root}"
                )
    assert not violations, "Capability port import violations:\n" + "\n".join(
        violations
    )


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
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "pydantic" in node.module
            ):
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
                imported_roots.extend(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            for imported_root in imported_roots:
                if imported_root not in STDLIB_IMPORTS:
                    violations.append(
                        f"{py_file.relative_to(PACKAGE_ROOT)}:{node.lineno}: "
                        f"imports non-stdlib module {imported_root}"
                    )
    assert not violations, "Non-stdlib imports in Base layer:\n" + "\n".join(violations)
