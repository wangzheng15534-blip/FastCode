"""Verify the import dependency graph is acyclic and respects layer boundaries."""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "fastcode"

LAYERS = {
    "ir": 0,
    "graph": 1,
    "retrieval": 2,
    "scip": 3,
    "store": 3,
    "semantic": 3,
    "indexing": 4,
    "query": 4,
    "schemas": 4,
    "api": 5,
    "mcp": 5,
    "main": 5,
}


def _get_fastcode_imports(filepath: Path) -> list[str]:
    """Parse a Python file and return fastcode.* import target packages."""
    tree = ast.parse(filepath.read_text())
    imports = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("fastcode.")
        ):
            parts = node.module.split(".")
            if len(parts) >= 2:
                imports.append(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("fastcode."):
                    parts = alias.name.split(".")
                    if len(parts) >= 2:
                        imports.append(parts[1])
    return imports


def _get_layer(package: str) -> int:
    return LAYERS.get(package, max(LAYERS.values()) + 1)


def test_no_upward_imports():
    """Packages must not import from higher layers."""
    violations = []
    for py_file in PACKAGE_ROOT.rglob("*.py"):
        rel = py_file.relative_to(PACKAGE_ROOT)
        parts = rel.parts
        if len(parts) < 2:
            continue
        importer_pkg = parts[0]
        if importer_pkg not in LAYERS:
            continue
        importer_layer = LAYERS[importer_pkg]
        for pkg in _get_fastcode_imports(py_file):
            if pkg in LAYERS and LAYERS[pkg] > importer_layer:
                violations.append(
                    f"{rel}: {importer_pkg} (layer {importer_layer}) "
                    f"imports {pkg} (layer {LAYERS[pkg]})"
                )
    assert not violations, "Upward import violations:\n" + "\n".join(violations)


def test_no_import_cycles():
    """The import graph must be acyclic."""
    from collections import defaultdict

    graph = defaultdict(set)
    for py_file in PACKAGE_ROOT.rglob("*.py"):
        rel = py_file.relative_to(PACKAGE_ROOT)
        parts = rel.parts
        if len(parts) < 2:
            continue
        importer_pkg = parts[0]
        for pkg in _get_fastcode_imports(py_file):
            if pkg != importer_pkg and pkg in LAYERS:
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
            import pytest

            pytest.fail(f"Import cycle detected involving package: {pkg}")
