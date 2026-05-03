"""Verify the import dependency graph is acyclic and respects layer boundaries."""

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "fastcode"

LAYERS = {
    "schemas": 0,
    "ir": 0,
    "utils": 0,
    "retrieval.core": 0,
    "store.infrastructure": 0,
    "graph": 1,
    "indexing": 1,
    "query": 1,
    "retrieval": 1,
    "scip": 1,
    "semantic": 1,
    "store": 1,
    "api": 2,
    "mcp": 2,
    "main": 2,
}


def _layer_key_for_module(module_path: str) -> str | None:
    parts = module_path.split(".")
    for idx in range(len(parts), 0, -1):
        candidate = ".".join(parts[:idx])
        if candidate in LAYERS:
            return candidate
    return None


def _importer_key_for_file(filepath: Path) -> str | None:
    rel = filepath.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = rel.parts[:-1] if rel.name != "__init__" else rel.parts
    if not parts:
        return None
    return _layer_key_for_module(".".join(parts))


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


def _get_fastcode_imports(filepath: Path) -> list[tuple[str, int]]:
    """Parse a Python file and return fastcode.* import targets."""
    tree = ast.parse(filepath.read_text())
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = _target_module_for_import_from(filepath, node)
            if target:
                imports.append((target, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("fastcode."):
                    imports.append((alias.name.removeprefix("fastcode."), node.lineno))
    return imports


def _get_layer(module_path: str) -> int:
    key = _layer_key_for_module(module_path)
    if key is None:
        return max(LAYERS.values()) + 1
    return LAYERS[key]


def test_no_upward_imports():
    """Packages must not import from higher layers."""
    violations = []
    for py_file in PACKAGE_ROOT.rglob("*.py"):
        rel = py_file.relative_to(PACKAGE_ROOT)
        importer_pkg = _importer_key_for_file(py_file)
        if importer_pkg is None:
            continue
        importer_layer = LAYERS[importer_pkg]
        for module_path, line in _get_fastcode_imports(py_file):
            target_pkg = _layer_key_for_module(module_path)
            if target_pkg and LAYERS[target_pkg] > importer_layer:
                violations.append(
                    f"{rel}:{line}: {importer_pkg} (layer {importer_layer}) "
                    f"imports {target_pkg} (layer {LAYERS[target_pkg]})"
                )
    assert not violations, "Upward import violations:\n" + "\n".join(violations)


def test_no_cross_layer_import_cycles():
    """Cross-layer imports must not form cycles."""
    from collections import defaultdict

    graph = defaultdict(set)
    for py_file in PACKAGE_ROOT.rglob("*.py"):
        importer_pkg = _importer_key_for_file(py_file)
        if importer_pkg is None:
            continue
        for module_path, _line in _get_fastcode_imports(py_file):
            pkg = _layer_key_for_module(module_path)
            if (
                pkg is not None
                and pkg != importer_pkg
                and _get_layer(pkg) != _get_layer(importer_pkg)
            ):
                graph[importer_pkg].add(pkg)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(LAYERS, WHITE)

    def dfs(node: str) -> bool:
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    for pkg in LAYERS:
        if color[pkg] == WHITE and dfs(pkg):
            pytest.fail(f"Import cycle detected involving package: {pkg}")
