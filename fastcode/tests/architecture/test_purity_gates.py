"""Verify pure packages and package roots don't import banned modules."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "fastcode"

PURE_PACKAGES = ["ir", "graph", "retrieval"]
BEHAVIOR_NAMED_DOMAIN_PACKAGES = ["ir", "graph", "retrieval", "scip", "semantic"]
BANNED_MODULES = {"pydantic", "sqlite3", "subprocess", "urllib"}
FORBIDDEN_TOP_LEVEL_HUBS = ("config.py", "config", "events.py", "events")
FORBIDDEN_GENERIC_INTERNAL_NAMES = {
    "utils",
    "helpers",
    "common",
    "misc",
    "service",
    "manager",
    "handler",
    "module",
}
BANNED_IMPORTS_BY_PACKAGE = {
    "graph": {
        "pydantic": "The graph layer must stay Pydantic-free.",
        "sqlite3": "The graph layer must stay independent of database I/O.",
        "subprocess": "The graph layer must stay independent of process execution.",
        "urllib": "The graph layer must stay independent of network I/O.",
        "fastcode.api": "The graph layer must not depend on entrypoint shells.",
        "fastcode.main": "The graph layer must not depend on the composition root.",
        "fastcode.mcp": "The graph layer must not depend on transport shells.",
        "fastcode.query": "Graph construction must stay independent of query orchestration.",
        "fastcode.retrieval": "Graph construction must stay independent of retrieval orchestration.",
        "fastcode.schemas": "Graph construction should depend on IR and helpers, not deleted schema compatibility modules.",
        "fastcode.store": "Graph construction must stay independent of storage orchestration.",
    },
    "ir": {
        "pydantic": "The IR layer must stay dataclass-based and Pydantic-free.",
        "sqlite3": "The IR layer must stay independent of database I/O.",
        "subprocess": "The IR layer must stay independent of process execution.",
        "urllib": "The IR layer must stay independent of network I/O.",
        "fastcode.api": "The IR layer must not depend on entrypoint shells.",
        "fastcode.graph": "Graph construction depends on IR, not vice versa.",
        "fastcode.indexing": "Indexing orchestrates IR usage; IR must stay independent.",
        "fastcode.main": "The IR layer must not depend on the composition root.",
        "fastcode.mcp": "The IR layer must not depend on transport shells.",
        "fastcode.query": "Query orchestration depends on IR, not vice versa.",
        "fastcode.retrieval": "Retrieval orchestration depends on IR, not vice versa.",
        "fastcode.schemas": "IR is canonical; deleted schema compatibility modules must stay out of IR.",
        "fastcode.scip": "SCIP adapters depend on IR, not vice versa.",
        "fastcode.semantic": "Semantic adapters depend on IR, not vice versa.",
        "fastcode.store": "Storage depends on IR types, not vice versa.",
    },
    "retrieval": {
        "pydantic": "retrieval must stay Pydantic-free.",
        "sqlite3": "retrieval must stay independent of database I/O.",
        "subprocess": "retrieval must stay independent of process execution.",
        "urllib": "retrieval must stay independent of network I/O.",
        "fastcode.api": "retrieval must not depend on entrypoint shells.",
        "fastcode.graph": "retrieval should consume IR-level data, not graph orchestration.",
        "fastcode.indexing": "retrieval must stay independent of indexing orchestration.",
        "fastcode.main": "retrieval must not depend on the composition root.",
        "fastcode.mcp": "retrieval must not depend on transport shells.",
        "fastcode.query": "retrieval must stay independent of query orchestration.",
        "fastcode.schemas": "retrieval must not depend on deleted schema compatibility modules.",
        "fastcode.scip": "retrieval must stay independent of SCIP loading and adapters.",
        "fastcode.semantic": "retrieval must stay independent of semantic adapters.",
        "fastcode.store": "retrieval must stay independent of storage orchestration.",
    },
    "kernel": {
        "pydantic": "kernel contracts must stay dataclass-based and Pydantic-free.",
        "dotenv": "kernel contracts must not load environment files.",
        "sqlite3": "kernel contracts must stay independent of database I/O.",
        "subprocess": "kernel contracts must stay independent of process execution.",
        "urllib": "kernel contracts must stay independent of network I/O.",
        "fastcode.api": "kernel contracts must not depend on entrypoint shells.",
        "fastcode.graph": "kernel contracts must not depend on domain orchestration.",
        "fastcode.indexing": "kernel contracts must not depend on shell orchestration.",
        "fastcode.main": "kernel contracts must not depend on the composition root.",
        "fastcode.mcp": "kernel contracts must not depend on transport shells.",
        "fastcode.query": "kernel contracts must not depend on query orchestration.",
        "fastcode.retrieval": "kernel contracts must not depend on retrieval logic.",
        "fastcode.scip": "kernel contracts must not depend on SCIP adapters.",
        "fastcode.semantic": "kernel contracts must not depend on semantic adapters.",
        "fastcode.store": "kernel contracts must not depend on storage orchestration.",
    },
    "runtime_support": {
        "pydantic": "runtime_support helpers must stay Pydantic-free.",
        "dotenv": "runtime_support must not load environment files.",
        "sqlite3": "runtime_support must stay independent of database I/O.",
        "subprocess": "runtime_support must stay independent of process execution.",
        "urllib": "runtime_support must stay independent of network I/O.",
        "fastcode.api": "runtime_support must not depend on entrypoint shells.",
        "fastcode.graph": "runtime_support must not depend on domain orchestration.",
        "fastcode.indexing": "runtime_support must not depend on shell orchestration.",
        "fastcode.main": "runtime_support must not depend on the composition root.",
        "fastcode.mcp": "runtime_support must not depend on transport shells.",
        "fastcode.query": "runtime_support must not depend on query orchestration.",
        "fastcode.retrieval": "runtime_support must not depend on retrieval logic.",
        "fastcode.scip": "runtime_support must not depend on SCIP adapters.",
        "fastcode.semantic": "runtime_support must not depend on semantic adapters.",
        "fastcode.store": "runtime_support must not depend on storage orchestration.",
    },
    "store/infrastructure": {
        "pydantic": "store.infrastructure should operate on frozen dataclasses and primitives.",
        "fastcode.api": "store.infrastructure must stay independent of entrypoint shells.",
        "fastcode.graph": "store.infrastructure must stay independent of graph orchestration.",
        "fastcode.indexing": "store.infrastructure must stay independent of indexing orchestration.",
        "fastcode.main": "store.infrastructure must stay independent of the composition root.",
        "fastcode.mcp": "store.infrastructure must stay independent of transport shells.",
        "fastcode.query": "store.infrastructure must stay independent of query orchestration.",
        "fastcode.retrieval": "store.infrastructure must stay independent of retrieval orchestration.",
        "fastcode.schemas": "store.infrastructure must not depend on deleted schema compatibility modules.",
        "fastcode.scip": "store.infrastructure must stay independent of SCIP adapters.",
        "fastcode.semantic": "store.infrastructure must stay independent of semantic adapters.",
    },
}
PACKAGE_ROOTS = [
    PACKAGE_ROOT / "__init__.py",
    *sorted(
        (PACKAGE_ROOT / pkg / "__init__.py")
        for pkg in PACKAGE_ROOT.iterdir()
        if pkg.is_dir()
    ),
]


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _has_source_content(path: Path) -> bool:
    if path.is_file():
        return True
    return any("__pycache__" not in child.parts for child in path.rglob("*"))


def _iter_python_files(package_path: str) -> list[Path]:
    return sorted((PACKAGE_ROOT / package_path).rglob("*.py"))


def _get_stdlib_imports(filepath: Path) -> list[str]:
    imports: list[str] = []
    tree = _tree(filepath)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
    return imports


def _resolve_relative_import(path: Path, node: ast.ImportFrom) -> str | None:
    rel_parts = path.relative_to(PACKAGE_ROOT).with_suffix("").parts
    package_parts = list(rel_parts[:-1])
    base_parts = package_parts[: len(package_parts) - (node.level - 1)]
    if node.module:
        target_parts = [*base_parts, *node.module.split(".")]
    elif node.names:
        target_parts = [*base_parts, node.names[0].name]
    else:
        return None
    return ".".join(("fastcode", *target_parts)) if target_parts else "fastcode"


def _import_candidates(path: Path, node: ast.AST) -> list[str]:
    candidates: list[str] = []
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return candidates

    module = node.module or ""
    if node.level:
        resolved = _resolve_relative_import(path, node)
        if resolved is None:
            return candidates
        candidates.append(resolved)
        candidates.extend(
            f"{resolved}.{alias.name}" for alias in node.names if alias.name != "*"
        )
        return candidates

    if module:
        candidates.append(module)
        candidates.extend(
            f"{module}.{alias.name}" for alias in node.names if alias.name != "*"
        )
    if module == "fastcode":
        candidates.extend(
            f"fastcode.{alias.name}" for alias in node.names if alias.name != "*"
        )
    return candidates


def _matches_banned_import(imported: str, banned: str) -> bool:
    return imported == banned or imported.startswith(f"{banned}.")


def _has_env_loading(filepath: Path) -> list[str]:
    tree = _tree(filepath)
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
                and node.func.attr == "getenv"
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


def test_no_top_level_events_or_config_hubs() -> None:
    """Events and config must live in role-specific packages, not root hubs."""
    violations = [
        rel_path
        for rel_path in FORBIDDEN_TOP_LEVEL_HUBS
        if _has_source_content(PACKAGE_ROOT / rel_path)
    ]
    assert not violations, "Forbidden top-level hubs restored:\n" + "\n".join(
        violations
    )


def test_pure_packages_no_banned_imports() -> None:
    """Files in pure packages must not import banned modules."""
    violations: list[str] = []
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


def test_package_boundary_import_bans() -> None:
    """Package-local banned import contracts are enforced centrally."""
    violations: list[str] = []
    for package_path, banned_imports in BANNED_IMPORTS_BY_PACKAGE.items():
        for py_file in _iter_python_files(package_path):
            rel = py_file.relative_to(PACKAGE_ROOT)
            for node in ast.walk(_tree(py_file)):
                if not isinstance(node, ast.Import | ast.ImportFrom):
                    continue
                for imported in _import_candidates(py_file, node):
                    for banned, message in banned_imports.items():
                        is_main_config_schema = rel == Path("main/schema.py") or (
                            rel.parent == Path("main")
                            and rel.name.startswith("_config_schema")
                        )
                        if is_main_config_schema and banned == "pydantic":
                            continue
                        if _matches_banned_import(imported, banned):
                            violations.append(
                                f"{rel}:{node.lineno}: imports {imported}; {message}"
                            )

    assert not violations, "Package boundary import violations:\n" + "\n".join(
        violations
    )


def test_runtime_contracts_do_not_load_env_or_dotenv() -> None:
    """Kernel and runtime_support receive resolved values; loaders stay in shell packages."""
    violations: list[str] = []
    for package in ("kernel", "runtime_support"):
        for py_file in _iter_python_files(package):
            violations.extend(_has_env_loading(py_file))
    assert not violations, "Runtime env access violations:\n" + "\n".join(violations)


def test_domain_and_common_modules_use_behavior_names_not_generic_buckets() -> None:
    """Pure/domain packages should name modules by behavior, not generic buckets."""
    violations: list[str] = []
    for package_path in BEHAVIOR_NAMED_DOMAIN_PACKAGES:
        for py_file in _iter_python_files(package_path):
            stem = py_file.stem
            normalized = stem[1:] if stem.startswith("_") else stem
            if normalized not in FORBIDDEN_GENERIC_INTERNAL_NAMES:
                continue
            violations.append(
                f"{py_file.relative_to(PACKAGE_ROOT)}: generic internal module name"
            )

    assert not violations, (
        "Pure/domain packages should use behavior-specific module names:\n"
        + "\n".join(violations)
    )


def test_package_roots_do_not_import_fastcode_root_exports() -> None:
    """Internal package roots must not depend on fastcode root exports."""
    violations: list[str] = []
    for py_file in PACKAGE_ROOTS:
        if not py_file.exists():
            continue
        tree = _tree(py_file)
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
        "Package roots must import concrete modules directly, not fastcode root "
        "exports:\n" + "\n".join(violations)
    )
