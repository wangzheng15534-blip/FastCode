#!/usr/bin/env python3
"""Executable architecture dependency gate for FastCode.

This script is the executable dependency model for FastCode's boundary checks.
It keeps three concerns explicit:

1. Layer direction and FCIS shell-role imports.
2. Boundary-hardening rules around purity, settings flow, and translation.
3. Private-module contracts enforced through import-linter.

The individual rule implementations remain in the repo-local architecture test
modules under ``fastcode/tests/architecture/``. This script groups them into a
single CLI gate, similar to the modeled workspace checker, while keeping the
existing tests as the source of truth.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
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
class GroupResult:
    id: str
    label: str
    summary: str
    checks_run: int
    failures: tuple[str, ...]


def _lint_imports_command() -> tuple[str, ...]:
    if shutil.which("lint-imports"):
        return ("lint-imports",)
    return ("uv", "run", "lint-imports")


CHECK_GROUPS: tuple[ModuleCheckGroup | CommandCheckGroup, ...] = (
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
        id="package-roots",
        label="Package Roots",
        module_path=ARCHITECTURE_TEST_ROOT / "test_package_roots.py",
        summary="package roots stay thin and compatibility shims stay deleted",
    ),
)


def _module_group_map() -> dict[str, ModuleCheckGroup | CommandCheckGroup]:
    return {group.id: group for group in CHECK_GROUPS}


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


def _run_group(group: ModuleCheckGroup | CommandCheckGroup) -> GroupResult:
    if isinstance(group, ModuleCheckGroup):
        return _run_module_group(group)
    return _run_command_group(group)


def _print_group_list() -> None:
    for group in CHECK_GROUPS:
        print(f"{group.id}: {group.summary}")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FastCode's executable architecture dependency gate."
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=tuple(group.id for group in CHECK_GROUPS),
        help="Run only the named check group. May be passed multiple times.",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List available check groups and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_groups:
        _print_group_list()
        return 0

    group_map = _module_group_map()
    selected_ids = args.group or [group.id for group in CHECK_GROUPS]
    results = [_run_group(group_map[group_id]) for group_id in selected_ids]

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
