#!/usr/bin/env python3
"""Executable architecture dependency gate for FastCode.

This is the Python-native replacement for the deleted dependency gate.
It keeps three concerns explicit:

1. Layer direction and import-linter contracts.
2. Boundary-hardening rules around purity, settings flow, and translation.
3. Package layout assessment for the current split-layout refactor plan.

The repo-local architecture tests remain the source of truth for the rules.
This script groups them into one CLI gate and adds a layout report mode so the
package-split work can be tracked without failing normal checks too early.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import re
import shutil
import subprocess
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_TEST_ROOT = REPO_ROOT / "fastcode" / "tests" / "architecture"
FASTCODE_ROOT = REPO_ROOT / "fastcode" / "src" / "fastcode"
FCIS_REGISTER = REPO_ROOT / ".fcis" / "role_register.json"

CANONICAL_FCIS_ROLES: frozenset[str] = frozenset(
    {
        "base_atoms",
        "base_kit",
        "run_kit",
        "meaning_seed",
        "meaning_core",
        "axis_surface",
        "axis_link",
        "axis_joint",
        "shape_lens",
        "use_flow",
        "effect_tool",
        "effect_facility",
        "wire_path",
        "entry_frame",
        "assembly_root",
        "signal_analyzer",
    }
)
BASE_COPY_PASTE_ROLES: frozenset[str] = frozenset({"base_atoms", "base_kit"})
NO_AXIS_ALLOWED_ROLES: frozenset[str] = frozenset(
    {
        "base_atoms",
        "base_kit",
        "meaning_seed",
        "run_kit",
        "assembly_root",
        "signal_analyzer",
    }
)
MULTI_AXIS_ALLOWED_ROLES: frozenset[str] = frozenset(
    {
        "meaning_seed",
        "run_kit",
        "entry_frame",
        "assembly_root",
        "axis_link",
        "wire_path",
        "effect_tool",
        "effect_facility",
    }
)
SHAPE_LENS_BINDING_ROLES: frozenset[str] = frozenset(
    {"use_flow", "effect_tool", "effect_facility"}
)
EFFECT_OWNER_ROLES: frozenset[str] = frozenset(
    {"effect_tool", "effect_facility", "wire_path"}
)
PRODUCER_ROLES: frozenset[str] = frozenset(
    CANONICAL_FCIS_ROLES - {"base_atoms", "base_kit", "run_kit", "signal_analyzer"}
)
EDGE_KINDS: frozenset[str] = frozenset(
    {
        "compile_dep",
        "runtime_call",
        "return_flow",
        "composition_wiring",
        "conceptual",
        "transport",
    }
)


@dataclass(frozen=True)
class ModuleCheckGroup:
    id: str
    label: str
    module_path: Path
    summary: str


@dataclass(frozen=True)
class CommandCheckGroup:
    id: str
    label: str
    command: tuple[str, ...]
    summary: str


@dataclass(frozen=True)
class FcisRegisterGroup:
    id: str = "fcis-register"
    label: str = "FCIS Register"
    summary: str = "project-local FCIS role registration"


@dataclass(frozen=True)
class LayoutTarget:
    package_path: Path
    expected_root_files: frozenset[str]
    expected_subpackages: frozenset[str]
    root_contract_suffix: str = ""


@dataclass(frozen=True)
class GroupResult:
    id: str
    label: str
    summary: str
    checks_run: int
    failures: tuple[str, ...]
    messages: tuple[str, ...] = ()


LAYOUT_TARGETS: tuple[LayoutTarget, ...] = (
    LayoutTarget(
        package_path=FASTCODE_ROOT / "store",
        expected_root_files=frozenset(),
        expected_subpackages=frozenset(
            {"infrastructure", "cache", "artifacts", "snapshots", "vectors", "runs"}
        ),
        root_contract_suffix="_contracts.py",
    ),
    LayoutTarget(
        package_path=FASTCODE_ROOT / "indexing",
        expected_root_files=frozenset(
            {
                "doc_ingester.py",
                "embedder.py",
                "file_inventory.py",
                "ignore.py",
                "loader.py",
                "overview.py",
                "publishing.py",
                "scip_runner.py",
                "terminus.py",
            }
        ),
        expected_subpackages=frozenset({"extractors", "pipeline", "projection"}),
    ),
    LayoutTarget(
        package_path=FASTCODE_ROOT / "query",
        expected_root_files=frozenset(
            {
                "boundary.py",
                "context_payloads.py",
                "contracts.py",
                "llm.py",
                "tokens.py",
            }
        ),
        expected_subpackages=frozenset({"agent", "orchestration", "selection"}),
    ),
    LayoutTarget(
        package_path=FASTCODE_ROOT / "retrieval",
        expected_root_files=frozenset({"contracts.py"}),
        expected_subpackages=frozenset({"ranking", "context", "graph"}),
    ),
    LayoutTarget(
        package_path=FASTCODE_ROOT / "semantic" / "resolvers",
        expected_root_files=frozenset(),
        expected_subpackages=frozenset({"core", "helpers", "languages"}),
    ),
)


def _lint_imports_command() -> tuple[str, ...]:
    if shutil.which("lint-imports"):
        return ("lint-imports",)
    return ("uv", "run", "lint-imports")


CHECK_GROUPS: tuple[ModuleCheckGroup | CommandCheckGroup | FcisRegisterGroup, ...] = (
    FcisRegisterGroup(),
    CommandCheckGroup(
        id="import-linter",
        label="Import Linter",
        command=_lint_imports_command(),
        summary="protected-module import contracts",
    ),
    ModuleCheckGroup(
        id="layer-dag",
        label="Layer DAG",
        module_path=ARCHITECTURE_TEST_ROOT / "test_layer_dag.py",
        summary="layer direction and FCIS shell-role rules",
    ),
    ModuleCheckGroup(
        id="purity",
        label="Purity Gates",
        module_path=ARCHITECTURE_TEST_ROOT / "test_purity_gates.py",
        summary="package purity and package-boundary bans",
    ),
    ModuleCheckGroup(
        id="pydantic-domain",
        label="No Pydantic in Domain",
        module_path=ARCHITECTURE_TEST_ROOT / "test_no_pydantic_in_domain.py",
        summary="common/domain packages stay Pydantic-free",
    ),
    ModuleCheckGroup(
        id="settings",
        label="Settings Flow",
        module_path=ARCHITECTURE_TEST_ROOT / "test_settings_flow.py",
        summary="env/config reads remain at shell boundaries",
    ),
    ModuleCheckGroup(
        id="translation",
        label="Explicit Translation",
        module_path=ARCHITECTURE_TEST_ROOT / "test_explicit_translation.py",
        summary="boundary code uses explicit adapters, not generic expansion",
    ),
    ModuleCheckGroup(
        id="materialization",
        label="Materialization Boundaries",
        module_path=ARCHITECTURE_TEST_ROOT / "test_materialization_boundaries.py",
        summary="hot paths avoid accidental object/vector materialization",
    ),
    ModuleCheckGroup(
        id="package-roots",
        label="Package Roots",
        module_path=ARCHITECTURE_TEST_ROOT / "test_package_roots.py",
        summary="package roots stay thin and compatibility shims stay deleted",
    ),
)


def _module_group_map() -> dict[
    str, ModuleCheckGroup | CommandCheckGroup | FcisRegisterGroup | LayoutAssessmentGroup
]:
    groups = {group.id: group for group in CHECK_GROUPS}
    groups["layout-assessment"] = LayoutAssessmentGroup()
    return groups


@dataclass(frozen=True)
class LayoutAssessmentGroup:
    id: str = "layout-assessment"
    label: str = "Layout Assessment"
    summary: str = "strict split-layout gap report"


def _load_module(module_path: Path) -> ModuleType:
    module_name = f"_fastcode_check_deps_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_test_functions(
    module: ModuleType,
) -> list[tuple[str, Callable[[], None]]]:
    test_functions: list[tuple[str, Callable[[], None]]] = []
    for name, candidate in vars(module).items():
        if not name.startswith("test_") or not inspect.isfunction(candidate):
            continue
        signature = inspect.signature(candidate)
        if signature.parameters:
            raise RuntimeError(
                f"{module.__file__}:{name} expects parameters; "
                "check_deps.py only supports zero-argument architecture tests"
            )
        test_functions.append((name, candidate))
    test_functions.sort(key=lambda item: item[1].__code__.co_firstlineno)
    return test_functions


def _indent_block(text: str) -> str:
    stripped = text.rstrip()
    if not stripped:
        return "  <no details>"
    return "\n".join(f"  {line}" if line else "  " for line in stripped.splitlines())


def _format_failure(group_id: str, check_name: str, details: str) -> str:
    return f"FAIL [{group_id}] {check_name}:\n{_indent_block(details)}"


def _run_module_group(group: ModuleCheckGroup) -> GroupResult:
    module = _load_module(group.module_path)
    failures: list[str] = []
    checks_run = 0
    for test_name, test_func in _iter_test_functions(module):
        checks_run += 1
        try:
            test_func()
        except AssertionError as exc:
            failures.append(
                _format_failure(group.id, test_name, str(exc) or "assertion failed")
            )
        except Exception:  # pragma: no cover - failure reporting path
            failures.append(
                _format_failure(group.id, test_name, traceback.format_exc())
            )
    if checks_run == 0:
        failures.append(
            _format_failure(
                group.id,
                group.module_path.name,
                "module contains no architecture checks",
            )
        )
    return GroupResult(
        id=group.id,
        label=group.label,
        summary=group.summary,
        checks_run=checks_run,
        failures=tuple(failures),
    )


def _run_command_group(group: CommandCheckGroup) -> GroupResult:
    try:
        completed = subprocess.run(  # noqa: S603 - commands are fixed repo tooling
            group.command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return GroupResult(
            id=group.id,
            label=group.label,
            summary=group.summary,
            checks_run=1,
            failures=(_format_failure(group.id, " ".join(group.command), str(exc)),),
        )

    if completed.returncode == 0:
        return GroupResult(
            id=group.id,
            label=group.label,
            summary=group.summary,
            checks_run=1,
            failures=(),
        )

    output_parts = [
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part and part.strip()
    ]
    details = "\n\n".join(output_parts) or f"command exited with {completed.returncode}"
    return GroupResult(
        id=group.id,
        label=group.label,
        summary=group.summary,
        checks_run=1,
        failures=(_format_failure(group.id, " ".join(group.command), details),),
    )


def _axis_parts(axis: str) -> list[str]:
    return [part for part in re.split(r"[,+]", axis) if part]


def _roles_for_units(units: list[dict[str, object]]) -> dict[str, set[str]]:
    return {
        str(unit.get("path", "")): set(str(role) for role in unit.get("roles", []))
        for unit in units
    }


def _load_fcis_register() -> dict[str, object]:
    if not FCIS_REGISTER.exists():
        raise FileNotFoundError(f"missing {FCIS_REGISTER.relative_to(REPO_ROOT)}")
    loaded = json.loads(FCIS_REGISTER.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("FCIS register root must be a JSON object")
    return loaded


def _check_fcis_unit(
    *,
    unit: dict[str, object],
    unit_by_path: dict[str, dict[str, object]],
) -> list[str]:
    path = str(unit.get("path", ""))
    roles = set(str(role) for role in unit.get("roles", []))
    axis = str(unit.get("axis", ""))
    axis_parts = _axis_parts(axis)
    violations: list[str] = []

    if not path:
        violations.append("<unknown>: registered unit is missing path")
    if not roles:
        violations.append(f"{path}: registered unit is missing roles")

    unknown_roles = sorted(
        role
        for role in roles
        if role not in CANONICAL_FCIS_ROLES or role in {"role_fold", "direct_lane"}
    )
    if unknown_roles:
        violations.append(f"{path}: unknown role(s): {', '.join(unknown_roles)}")

    if roles & BASE_COPY_PASTE_ROLES:
        if axis:
            violations.append(f"{path}: base_atoms/base_kit must not declare an axis")
        if roles - BASE_COPY_PASTE_ROLES:
            violations.append(
                f"{path}: base_atoms/base_kit must not be folded with repo roles"
            )

    if not axis and not (roles <= NO_AXIS_ALLOWED_ROLES or roles == {"shape_lens"}):
        violations.append(
            f"{path}: missing axis; only shared-default roles may omit it"
        )

    if len(axis_parts) > 1 and not roles <= (MULTI_AXIS_ALLOWED_ROLES | {"shape_lens"}):
        violations.append(
            f"{path}: illegal multi-axis role set; split by axis and use links/wires"
        )

    if "entry_frame" in roles and not axis:
        violations.append(f"{path}: entry_frame must declare one or more axes")

    if "signal_analyzer" in roles:
        if len(roles) > 1:
            violations.append(
                f"{path}: signal_analyzer must stay standalone, not role-folded"
            )
        if axis:
            violations.append(f"{path}: signal_analyzer must not declare an axis")

    if "shape_lens" in roles and not (roles & SHAPE_LENS_BINDING_ROLES):
        bind_to = str(unit.get("bind_to", ""))
        if roles == {"shape_lens"} and axis:
            violations.append(
                f"{path}: standalone shape_lens derives axis from bind_to"
            )
        if not bind_to:
            violations.append(f"{path}: standalone shape_lens must declare bind_to")
        else:
            target = unit_by_path.get(bind_to)
            target_roles = (
                set(str(role) for role in target.get("roles", [])) if target else set()
            )
            if target is None:
                violations.append(
                    f"{path}: shape_lens bind_to target not registered: {bind_to}"
                )
            elif not (target_roles & SHAPE_LENS_BINDING_ROLES):
                violations.append(
                    f"{path}: shape_lens bind_to target must be use_flow/effect target"
                )

    return violations


def _check_fcis_edges(
    *,
    units: list[dict[str, object]],
    edges: list[dict[str, object]],
) -> list[str]:
    unit_by_path = {str(unit.get("path", "")): unit for unit in units}
    path_to_roles = _roles_for_units(units)
    violations: list[str] = []

    for edge in edges:
        src = str(edge.get("from", ""))
        dst = str(edge.get("to", ""))
        kind = str(edge.get("kind", ""))
        src_roles = path_to_roles.get(src, set())
        dst_roles = path_to_roles.get(dst, set())
        if src not in unit_by_path:
            violations.append(f"{src} -> {dst}: edge source is not registered")
        if dst not in unit_by_path:
            violations.append(f"{src} -> {dst}: edge target is not registered")
        if kind not in EDGE_KINDS:
            violations.append(f"{src} -> {dst}: invalid edge kind {kind!r}")
        if (
            kind == "compile_dep"
            and src_roles & BASE_COPY_PASTE_ROLES
            and not dst_roles <= BASE_COPY_PASTE_ROLES
        ):
            violations.append(
                f"{src} -> {dst}: base_atoms/base_kit must stay copy-pasteable"
            )
        if (
            kind == "compile_dep"
            and "shape_lens" in src_roles
            and dst_roles & {"axis_surface", "meaning_core"}
        ):
            violations.append(
                f"{src} -> {dst}: shape_lens must not depend on semantic roles"
            )
        if (
            kind == "compile_dep"
            and src_roles == {"shape_lens"}
            and dst_roles
            and not (dst_roles <= {"run_kit"} or dst_roles & SHAPE_LENS_BINDING_ROLES)
        ):
            violations.append(
                f"{src} -> {dst}: standalone shape_lens can depend only on run_kit "
                "or its bound target"
            )
        if (
            kind == "compile_dep"
            and "use_flow" in src_roles
            and "shape_lens" not in src_roles
            and dst_roles & EFFECT_OWNER_ROLES
        ):
            violations.append(
                f"{src} -> {dst}: direct use_flow -> effect owner needs fold or lens"
            )
        if kind == "compile_dep" and "entry_frame" in src_roles:
            src_axis = str(unit_by_path.get(src, {}).get("axis", ""))
            if "+" in src_axis and dst_roles & {
                "use_flow",
                "shape_lens",
                "axis_surface",
                "meaning_core",
            }:
                violations.append(
                    f"{src} -> {dst}: multi-axis entry_frame must split through wire"
                )
        if kind == "compile_dep" and src_roles == {"signal_analyzer"}:
            if dst_roles - {"run_kit"}:
                violations.append(
                    f"{src} -> {dst}: signal_analyzer may only depend on run_kit"
                )
            if dst_roles & PRODUCER_ROLES:
                violations.append(
                    f"{src} -> {dst}: signal_analyzer must parse producer records as data"
                )

    return violations


def _run_fcis_register_group(group: FcisRegisterGroup) -> GroupResult:
    try:
        data = _load_fcis_register()
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        return GroupResult(
            id=group.id,
            label=group.label,
            summary=group.summary,
            checks_run=1,
            failures=(_format_failure(group.id, "load-register", str(exc)),),
        )

    raw_units = data.get("units", [])
    raw_edges = data.get("edges", [])
    raw_role_folds = data.get("role_folds", [])
    if not isinstance(raw_units, list) or not isinstance(raw_edges, list):
        return GroupResult(
            id=group.id,
            label=group.label,
            summary=group.summary,
            checks_run=1,
            failures=(
                _format_failure(
                    group.id,
                    "schema",
                    "role_register.json units and edges must be lists",
                ),
            ),
        )

    units = [unit for unit in raw_units if isinstance(unit, dict)]
    edges = [edge for edge in raw_edges if isinstance(edge, dict)]
    unit_by_path = {str(unit.get("path", "")): unit for unit in units}
    path_to_roles = _roles_for_units(units)
    violations: list[str] = []

    if len(units) != len(raw_units):
        violations.append("units must contain only objects")
    if len(edges) != len(raw_edges):
        violations.append("edges must contain only objects")

    for unit in units:
        violations.extend(_check_fcis_unit(unit=unit, unit_by_path=unit_by_path))

    if isinstance(raw_role_folds, list):
        for fold in raw_role_folds:
            if not isinstance(fold, dict):
                violations.append("role_folds must contain only objects")
                continue
            fold_path = str(fold.get("path", ""))
            fold_roles = set(str(role) for role in fold.get("roles", []))
            registered_roles = path_to_roles.get(fold_path)
            if registered_roles is None:
                violations.append(f"{fold_path}: stale role_fold for unregistered path")
            elif len(registered_roles) < 2:
                violations.append(f"{fold_path}: stale role_fold for single-role unit")
            elif fold_roles and fold_roles != registered_roles:
                violations.append(f"{fold_path}: stale role_fold roles mismatch")
    else:
        violations.append("role_folds must be a list")

    if data.get("direct_lanes"):
        violations.append("direct_lanes is note-only; model shortcuts as checked roles")

    violations.extend(_check_fcis_edges(units=units, edges=edges))

    failures: tuple[str, ...] = ()
    if violations:
        failures = (
            _format_failure(group.id, "role_register.json", "\n".join(violations)),
        )
    return GroupResult(
        id=group.id,
        label=group.label,
        summary=group.summary,
        checks_run=max(1, len(units) + len(edges)),
        failures=failures,
    )


def _layout_root_files(package_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in package_dir.glob("*.py")
        if path.name != "__init__.py" and path.is_file()
    )


def _layout_subpackages(package_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in package_dir.iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    )


def _layout_contract_count(package_dir: Path, suffix: str) -> int:
    if not suffix:
        return 0
    return sum(1 for path in package_dir.glob(f"*{suffix}"))


def _analyze_layout_target(target: LayoutTarget) -> tuple[list[str], list[str]]:
    package_dir = target.package_path
    root_files = _layout_root_files(package_dir)
    subpackages = _layout_subpackages(package_dir)
    root_file_names = {path.name for path in root_files}
    missing_root_files = sorted(target.expected_root_files - root_file_names)
    unexpected_root_files = sorted(root_file_names - target.expected_root_files)
    missing_subpackages = sorted(set(target.expected_subpackages) - set(subpackages))
    unexpected_subpackages = sorted(set(subpackages) - set(target.expected_subpackages))
    contract_count = _layout_contract_count(package_dir, target.root_contract_suffix)

    summary_parts = [
        f"{target.package_path.relative_to(FASTCODE_ROOT)}: "
        f"{len(root_files)} root module(s)",
        f"{len(subpackages)} immediate subpackage(s)",
    ]
    if contract_count:
        summary_parts.append(f"{contract_count} root contract module(s)")

    report_lines = [" - " + ", ".join(summary_parts)]
    issue_lines: list[str] = []
    if missing_root_files:
        issue_lines.append(
            f"{target.package_path.relative_to(FASTCODE_ROOT)} missing root files: "
            + ", ".join(missing_root_files)
        )
    if unexpected_root_files:
        issue_lines.append(
            f"{target.package_path.relative_to(FASTCODE_ROOT)} unexpected root files: "
            + ", ".join(unexpected_root_files)
        )
    if missing_subpackages:
        issue_lines.append(
            f"{target.package_path.relative_to(FASTCODE_ROOT)} missing subpackages: "
            + ", ".join(missing_subpackages)
        )
    if unexpected_subpackages:
        issue_lines.append(
            f"{target.package_path.relative_to(FASTCODE_ROOT)} unexpected subpackages: "
            + ", ".join(unexpected_subpackages)
        )

    return report_lines, issue_lines


def _run_layout_group(*, strict_layout: bool) -> GroupResult:
    report_lines: list[str] = []
    issue_lines: list[str] = []
    for target in LAYOUT_TARGETS:
        lines, issues = _analyze_layout_target(target)
        report_lines.extend(lines)
        issue_lines.extend(issues)
    failures: tuple[str, ...] = ()
    if strict_layout and issue_lines:
        failures = (
            _format_failure(
                "layout-assessment",
                "target-layout",
                "\n".join(issue_lines),
            ),
        )
    return GroupResult(
        id="layout-assessment",
        label="Layout Assessment",
        summary="strict split-layout gap report",
        checks_run=len(LAYOUT_TARGETS),
        failures=failures,
        messages=tuple(
            report_lines + (["gaps:\n" + "\n".join(issue_lines)] if issue_lines else [])
        ),
    )


def _run_group(
    group: ModuleCheckGroup | CommandCheckGroup | FcisRegisterGroup | LayoutAssessmentGroup,
    *,
    strict_layout: bool,
) -> GroupResult:
    if isinstance(group, ModuleCheckGroup):
        return _run_module_group(group)
    if isinstance(group, CommandCheckGroup):
        return _run_command_group(group)
    if isinstance(group, FcisRegisterGroup):
        return _run_fcis_register_group(group)
    return _run_layout_group(strict_layout=strict_layout)


def _print_group_list() -> None:
    for group in CHECK_GROUPS:
        print(f"{group.id}: {group.summary}")
    print("layout-assessment: strict split-layout gap report")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FastCode's executable architecture dependency gate."
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=(*[group.id for group in CHECK_GROUPS], "layout-assessment"),
        help="Run only the named check group. May be passed multiple times.",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List available check groups and exit.",
    )
    parser.add_argument(
        "--report-gaps",
        action="store_true",
        help="Print layout-assessment gap details.",
    )
    parser.add_argument(
        "--strict-layout",
        action="store_true",
        help="Fail layout-assessment when target package gaps remain.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_groups:
        _print_group_list()
        return 0

    group_map = _module_group_map()
    selected_ids = args.group or [group.id for group in CHECK_GROUPS] + [
        "layout-assessment"
    ]
    results = [
        _run_group(group_map[group_id], strict_layout=bool(args.strict_layout))
        for group_id in selected_ids
    ]

    if args.report_gaps:
        for result in results:
            if result.id == "layout-assessment":
                for message in result.messages:
                    print(message)

    failures = [failure for result in results for failure in result.failures]
    if failures:
        for failure in failures:
            print(failure)
        failed_groups = sum(1 for result in results if result.failures)
        print(
            f"\nFAIL: {len(failures)} failing check(s) across {failed_groups} group(s)"
        )
        return 1

    for result in results:
        print(f"PASS [{result.id}]: {result.checks_run} check(s) - {result.summary}")
    print(
        f"\nPASS: {sum(result.checks_run for result in results)} check(s) across "
        f"{len(results)} group(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
