"""Verify package roots stay thin enough to preserve import boundaries."""

import ast
import tomllib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "fastcode"
TEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[2]
WORKSPACE_TEST_ROOT = REPO_ROOT / "tests"
ROOT_INIT = PACKAGE_ROOT / "__init__.py"
MAIN_INIT = PACKAGE_ROOT / "main" / "__init__.py"
FORBIDDEN_GENERIC_LAYOUT_NAMES = {"config", "core", "events"}
ALLOWED_ROLE_SPECIFIC_LAYOUT_FILES = {
    Path("main/schema.py"),
    Path("main/config.py"),
    Path("kernel/config.py"),
    Path("runtime_support/events.py"),
}
DELETED_COMPATIBILITY_MODULES = {
    "fastcode.db_runtime",
    "fastcode.events",
    "fastcode.graph_runtime",
    "fastcode.llm_utils",
    "fastcode.module_resolver",
    "fastcode.path_utils",
    "fastcode.api.contracts",
    "fastcode.indexing.call_extractor",
    "fastcode.indexing.global_builder",
    "fastcode.indexing.tree_sitter",
    "fastcode.retrieval.agent_tools",
    "fastcode.retrieval.core",
    "fastcode.retrieval.hybrid",
    "fastcode.retrieval.iterative",
    "fastcode.schemas",
    "fastcode.schemas.api",
    "fastcode.schemas.config",
    "fastcode.schemas.core_types",
    "fastcode.schemas.ir",
    "fastcode.semantic.resolvers.base",
    "fastcode.store.contracts",
    "fastcode.store.records",
    "fastcode.utils._compat",
    "fastcode.utils.core",
    "fastcode.utils.vectors",
}
DELETED_COMPATIBILITY_FILES = {
    "db_runtime.py",
    "events",
    "graph_runtime.py",
    "llm_utils.py",
    "module_resolver.py",
    "path_utils.py",
    "api/contracts.py",
    "indexing/call_extractor.py",
    "indexing/global_builder.py",
    "indexing/tree_sitter.py",
    "retrieval/agent_tools.py",
    "retrieval/core",
    "retrieval/hybrid.py",
    "retrieval/iterative.py",
    "schemas",
    "schemas/api.py",
    "schemas/config.py",
    "schemas/core_types.py",
    "schemas/ir.py",
    "semantic/resolvers/base.py",
    "store/contracts.py",
    "store/records.py",
    "utils/_compat.py",
    "utils/core.py",
    "utils/vectors.py",
}


def _module_name_for_path(path: Path) -> str:
    return ".".join(("fastcode", *path.relative_to(PACKAGE_ROOT).with_suffix("").parts))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _has_source_content(path: Path) -> bool:
    if path.is_file():
        return True
    return any("__pycache__" not in child.parts for child in path.rglob("*"))


def _imported_modules(path: Path) -> list[str]:
    imports: list[str] = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            prefix = "." * node.level
            imports.append(f"{prefix}{node.module}")
    return imports


def _python_files_for_split_import_checks() -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    files.extend((path, PACKAGE_ROOT) for path in PACKAGE_ROOT.rglob("*.py"))
    files.extend((path, TEST_ROOT) for path in TEST_ROOT.rglob("*.py"))
    if WORKSPACE_TEST_ROOT.is_dir():
        files.extend(
            (path, WORKSPACE_TEST_ROOT) for path in WORKSPACE_TEST_ROOT.rglob("*.py")
        )
    return files


def _fastcode_root_packages() -> set[str]:
    return {
        child.name
        for child in PACKAGE_ROOT.iterdir()
        if child.is_dir() and (child / "__init__.py").exists()
    }


def _fastcode_package_modules() -> dict[str, Path]:
    modules = {
        ".".join(("fastcode", *path.relative_to(PACKAGE_ROOT).parts)): path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_dir() and (path / "__init__.py").exists()
    }
    modules["fastcode"] = PACKAGE_ROOT
    return modules


def _resolve_fastcode_import_target(
    node: ast.ImportFrom,
    *,
    rel: Path,
    base_root: Path,
) -> str | None:
    if node.level == 0 and node.module and node.module.startswith("fastcode."):
        return node.module.removeprefix("fastcode.")
    if base_root != PACKAGE_ROOT or not node.level:
        return None

    current_parts = rel.with_suffix("").parts[:-1]
    base_parts = current_parts[: len(current_parts) - (node.level - 1)]
    if node.module:
        return ".".join((*base_parts, *node.module.split(".")))
    if node.names:
        return ".".join((*base_parts, node.names[0].name))
    return None


def _expr_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _expr_name(node.value)
    return None


def _basemodel_class_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in _tree(path).body:
        if not isinstance(node, ast.ClassDef):
            continue
        if any(_expr_name(base) == "BaseModel" for base in node.bases):
            names.add(node.name)
    return names


def _tuple_assignment_names(path: Path, assignment_name: str) -> set[str]:
    for node in _tree(path).body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == assignment_name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Tuple):
            return set()
        return {item.id for item in node.value.elts if isinstance(item, ast.Name)}
    return set()


def test_fastcode_root_has_no_runtime_side_effect_imports():
    """Importing fastcode.ir.* executes fastcode/__init__.py first."""
    banned = {
        "os",
        "platform",
        "subprocess",
        "sqlite3",
        "fastapi",
        "uvicorn",
        "fastcode.main.fastcode",
    }

    violations = [
        module
        for module in _imported_modules(ROOT_INIT)
        if module.lstrip(".").split(".")[0] in banned or module.lstrip(".") in banned
    ]

    assert not violations, "fastcode/__init__.py imports runtime modules: " + ", ".join(
        violations
    )


def test_fastcode_root_contains_no_environment_mutation():
    """Package import must not mutate process-wide runtime settings."""
    mutations: list[str] = []
    for node in ast.walk(_tree(ROOT_INIT)):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Attribute)
            and node.targets[0].value.attr == "environ"
        ):
            mutations.append(f"line {node.lineno}")

    assert not mutations, "fastcode/__init__.py mutates os.environ at " + ", ".join(
        mutations
    )


def test_fastcode_root_does_not_export_runtime_objects():
    """Pre-release layout uses concrete imports instead of root compatibility shims."""
    tree = _tree(ROOT_INIT)
    violations: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            violations.append(f"line {node.lineno}: __getattr__ compatibility export")
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "_EXPORTS",
                    "__all__",
                }:
                    if target.id == "_EXPORTS":
                        violations.append(f"line {node.lineno}: _EXPORTS table")
                    elif isinstance(node.value, ast.List):
                        exported = [
                            item.value
                            for item in node.value.elts
                            if isinstance(item, ast.Constant)
                            and isinstance(item.value, str)
                        ]
                        if exported != ["__version__"]:
                            violations.append(
                                f"line {node.lineno}: runtime exports in __all__"
                            )

    assert not violations, (
        "fastcode/__init__.py keeps root runtime exports:\n" + "\n".join(violations)
    )


def test_package_roots_do_not_define_runtime_export_shims():
    """Subpackage roots are markers or contract modules, not compatibility APIs."""
    violations: list[str] = []
    for path in PACKAGE_ROOT.rglob("__init__.py"):
        if path == ROOT_INIT:
            continue
        rel = path.relative_to(PACKAGE_ROOT)
        tree = _tree(path)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                violations.append(f"{rel}:{node.lineno}: __getattr__ export shim")
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {
                        "_EXPORTS",
                        "__all__",
                    }:
                        violations.append(
                            f"{rel}:{node.lineno}: {target.id} export table"
                        )
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
            ):
                violations.append(f"{rel}:{node.lineno}: TYPE_CHECKING export imports")

    assert not violations, "Package-root runtime export shims:\n" + "\n".join(
        violations
    )


def test_codebase_does_not_use_fastcode_root_reexports():
    """Code and tests should import concrete modules, not compatibility exports."""
    violations: list[str] = []
    for path, base_root in _python_files_for_split_import_checks():
        if path == ROOT_INIT:
            continue
        rel = path.relative_to(base_root)
        for node in ast.walk(_tree(path)):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module == "fastcode"
            ):
                names = ", ".join(alias.name for alias in node.names)
                violations.append(f"{rel}:{node.lineno} imports {names}")

    assert not violations, "FastCode root re-export imports:\n" + "\n".join(violations)


def test_codebase_imports_concrete_submodules_not_package_roots():
    """Code and tests should exercise the split module layout directly."""
    root_packages = _fastcode_root_packages()
    package_modules = _fastcode_package_modules()
    violations: list[str] = []
    for path, base_root in _python_files_for_split_import_checks():
        if path.name == "__init__.py":
            continue
        rel = path.relative_to(base_root)
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in package_modules:
                        violations.append(
                            f"{rel}:{node.lineno}: imports package root {alias.name}"
                        )
                continue
            if isinstance(node, ast.ImportFrom):
                target = _resolve_fastcode_import_target(
                    node,
                    rel=rel,
                    base_root=base_root,
                )
                if not target:
                    continue
                target_parts = target.split(".")
                if (
                    len(target_parts) == 1
                    and target_parts[0] in root_packages
                    and target_parts[0] in root_packages
                ):
                    names = ", ".join(alias.name for alias in node.names)
                    violations.append(
                        f"{rel}:{node.lineno}: imports {names} from package root {target}"
                    )
                    continue
                module_name = f"fastcode.{target}"
                package_path = package_modules.get(module_name)
                if package_path is None:
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        violations.append(
                            f"{rel}:{node.lineno}: star import from package root {module_name}"
                        )
                        continue
                    if (package_path / f"{alias.name}.py").exists() or (
                        (package_path / alias.name).is_dir()
                        and (package_path / alias.name / "__init__.py").exists()
                    ):
                        violations.append(
                            f"{rel}:{node.lineno}: imports submodule {alias.name} "
                            f"from package root {module_name}"
                        )

    assert not violations, "FastCode package-root imports:\n" + "\n".join(violations)


def test_codebase_does_not_use_generic_core_layout_names():
    """Runtime/test layout uses descriptive module names, not bare core buckets."""
    violations: list[str] = []
    roots = [
        (PACKAGE_ROOT, PACKAGE_ROOT),
        (TEST_ROOT, TEST_ROOT),
    ]
    if WORKSPACE_TEST_ROOT.is_dir():
        roots.append((WORKSPACE_TEST_ROOT, WORKSPACE_TEST_ROOT))

    for search_root, display_root in roots:
        for path in sorted(search_root.rglob("*")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(display_root)
            if path.is_dir() and path.name in FORBIDDEN_GENERIC_LAYOUT_NAMES:
                if not _has_source_content(path):
                    continue
                violations.append(f"{rel}: generic directory")
            if (
                path.is_file()
                and path.suffix == ".py"
                and path.stem in FORBIDDEN_GENERIC_LAYOUT_NAMES
            ):
                if (
                    search_root == PACKAGE_ROOT
                    and rel in ALLOWED_ROLE_SPECIFIC_LAYOUT_FILES
                ):
                    continue
                violations.append(f"{rel}: generic module")

    assert not violations, "Generic core layout names:\n" + "\n".join(violations)


def test_deleted_compatibility_modules_are_not_restored_or_imported():
    """Moved modules stay deleted; callers use their owning split packages."""
    violations: list[str] = []
    for rel_path in sorted(DELETED_COMPATIBILITY_FILES):
        if _has_source_content(PACKAGE_ROOT / rel_path):
            violations.append(f"{rel_path}: deleted compatibility module was restored")

    for path, base_root in _python_files_for_split_import_checks():
        rel = path.relative_to(base_root)
        for node in ast.walk(_tree(path)):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
                imported.extend(
                    f"{node.module}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
            for module in imported:
                if any(
                    module == deleted_module or module.startswith(f"{deleted_module}.")
                    for deleted_module in DELETED_COMPATIBILITY_MODULES
                ):
                    violations.append(
                        f"{rel}:{node.lineno}: imports deleted module {module}"
                    )

    assert not violations, (
        "Deleted compatibility modules are still reachable:\n" + "\n".join(violations)
    )


def test_api_schema_models_are_split_by_http_boundary_direction():
    """HTTP DTOs live in inbound/outbound/openapi, not a mixed contracts module."""
    inbound_path = PACKAGE_ROOT / "api" / "inbound.py"
    outbound_path = PACKAGE_ROOT / "api" / "outbound.py"
    openapi_path = PACKAGE_ROOT / "api" / "openapi.py"

    inbound_models = _basemodel_class_names(inbound_path)
    outbound_models = _basemodel_class_names(outbound_path)
    openapi_models = _tuple_assignment_names(openapi_path, "OPENAPI_SCHEMA_MODELS")

    bad_inbound = sorted(
        name for name in inbound_models if not name.endswith("Request")
    )
    outbound_response_models = {
        name for name in outbound_models if name.endswith("Response")
    }
    outbound_helper_models = {name for name in outbound_models if name.endswith("DTO")}
    bad_outbound = sorted(
        outbound_models - outbound_response_models - outbound_helper_models
    )
    missing_openapi = sorted(
        (inbound_models | outbound_response_models) - openapi_models
    )
    unknown_openapi = sorted(
        openapi_models - (inbound_models | outbound_response_models)
    )

    assert not bad_inbound, "Non-request DTOs in api/inbound.py: " + ", ".join(
        bad_inbound
    )
    assert not bad_outbound, (
        "Unexpected outbound BaseModel names in api/outbound.py: "
        + ", ".join(bad_outbound)
    )
    assert not missing_openapi, "OpenAPI registry missing DTOs: " + ", ".join(
        missing_openapi
    )
    assert not unknown_openapi, "OpenAPI registry has unknown DTOs: " + ", ".join(
        unknown_openapi
    )


def test_import_linter_protects_current_private_implementation_modules():
    """Private implementation modules must be covered by protected contracts."""
    root_config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    package_config = tomllib.loads(
        (REPO_ROOT / "fastcode" / "pyproject.toml").read_text()
    )
    import_linter = root_config["tool"]["importlinter"]
    contracts = import_linter["contracts"]

    assert import_linter["root_package"] == "fastcode"
    assert "import-linter" in package_config["project"]["optional-dependencies"]["dev"]

    private_modules = {
        _module_name_for_path(path)
        for path in PACKAGE_ROOT.rglob("_*.py")
        if path.name != "__init__.py"
    }
    protected_modules = {
        module
        for contract in contracts
        if contract.get("type") == "protected"
        for module in contract.get("protected_modules", [])
    }
    missing = sorted(private_modules - protected_modules)

    assert not missing, (
        "Private modules missing import-linter protection: " + ", ".join(missing)
    )


def test_pyright_configs_remain_strict():
    """Workspace and package pyright configs stay on strict type-checking mode."""
    root_config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    package_config = tomllib.loads(
        (REPO_ROOT / "fastcode" / "pyproject.toml").read_text()
    )

    assert root_config["tool"]["pyright"]["typeCheckingMode"] == "strict"
    assert package_config["tool"]["pyright"]["typeCheckingMode"] == "strict"


def test_repo_justfile_exposes_quality_gates():
    """Top-level task surface must expose FastCode quality gates."""
    justfile_path = REPO_ROOT / "justfile"
    assert justfile_path.exists(), "Missing repo justfile quality surface"
    justfile = justfile_path.read_text()

    for recipe in (
        "qa:",
        "qa-lint:",
        "qa-full:",
        "check:",
        "check-deps:",
        "arch-check:",
        "type-check:",
    ):
        assert recipe in justfile, f"Missing just recipe: {recipe}"
