#!/usr/bin/env python3
"""Run mutmut mutation testing and report baseline scores.

Usage:
    uv run python scripts/mutmut_baseline.py [--full]

    --full: mutate all source files (slow, ~30 min)
    default: mutate only high-value algorithmic files (~5 min)
"""
from __future__ import annotations

import subprocess
import sys


HIGH_VALUE_MODULES = [
    "fastcode/src/fastcode/core/scoring.py",
    "fastcode/src/fastcode/core/fusion.py",
    "fastcode/src/fastcode/core/filtering.py",
    "fastcode/src/fastcode/adapters/scip_to_ir.py",
    "fastcode/src/fastcode/ir_validators.py",
]


def run_mutmut(paths: list[str]) -> int:
    paths_arg = " ".join(f"--paths-to-mutate {p}" for p in paths)
    cmd = (
        f"cd fastcode && mutmut run "
        f"{paths_arg} "
        f"--runner 'uv run pytest fastcode/tests/ -x -q --timeout=30' "
        f"--use-coverage"
    )
    return subprocess.call(cmd, shell=True)


def show_results() -> int:
    return subprocess.call("cd fastcode && mutmut results", shell=True)


def main() -> None:
    full = "--full" in sys.argv
    paths = HIGH_VALUE_MODULES if not full else []

    print("=" * 60)
    print("Mutmut Mutation Testing Baseline")
    print("=" * 60)
    if full:
        print("Mode: FULL (all modules)")
    else:
        print(f"Mode: HIGH-VALUE ({len(HIGH_VALUE_MODULES)} modules)")
        for p in HIGH_VALUE_MODULES:
            print(f"  - {p}")
    print()

    rc = run_mutmut(paths)
    if rc == 0:
        print("\nMutation testing complete. Results:")
        show_results()
    sys.exit(rc)


if __name__ == "__main__":
    main()
