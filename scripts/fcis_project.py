#!/usr/bin/env python3
"""Project-local FCIS helper.

The project register is intentionally minimal: `.fcis/role_register.json` holds
only physical units with roles, axes, optional binding targets, optional import
names, and optional source roots. The shared rule graph in `fcis_rules.py`
derives allowed imports from those facts and checks folds with the canonical allow-list.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fcis_rules import CANONICAL_ROLES, check_imports, check_register, print_result, split_roles  # noqa: E402

ROLE_HINTS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"foundation|primitive|atom", re.I), "base_atoms", "generic primitive base"),
    (re.compile(r"utils?|helper|common", re.I), "base_kit", "generic helper APIs"),
    (re.compile(r"runtime-support|run-?kit|retry|health|cache", re.I), "run_kit", "generic runtime helpers"),
    (re.compile(r"kernel|identity|seed", re.I), "meaning_seed", "shared minimal vocabulary"),
    (re.compile(r"domain|catalog|sync|cite|fulltext|meaning", re.I), "meaning_core", "pure business meaning"),
    (re.compile(r"app|app-?flow|use-?case|workflow|flow", re.I), "use_flow", "business workflow brain"),
    (re.compile(r"facade|api-adapter|server|entry|route", re.I), "entry_frame", "entry and exit shape"),
    (re.compile(r"composition|assembly|bootstrap|main", re.I), "assembly_root", "resource construction and lifecycle wiring"),
    (re.compile(r"config[-_]?binder|config[-_]?shape|inbound[-_]?config", re.I), "config_binder", "assembly-root config ingress"),
    (re.compile(r"port|contract|surface", re.I), "axis_surface", "same-axis semantic/capability surface"),
    (re.compile(r"acl|boundary|shape|dto|schema|proto", re.I), "shape_seam", "app-owned shape boundary"),
    (re.compile(r"adapter|sqlite|storage|http|client|deno|js|cdp|tool", re.I), "effect_tool", "generic reusable adapter/tool library"),
    (re.compile(r"facility|daemon|sidecar|worker|process", re.I), "effect_facility", "long-lived generic effect owner"),
    (re.compile(r"interop|cross|plugin|translator|host", re.I), "axis_link", "horizontal semantic API"),
    (re.compile(r"wire|uds|rpc|transport|protocol", re.I), "wire_path", "bound transport/schema side path"),
    (re.compile(r"observability|analy[sz]er|metrics?|events?|logs?|signals?", re.I), "signal_analyzer", "passive signal record analyzer"),
    (re.compile(r"acceptance|e2e|end[-_ ]?to[-_ ]?end|probe|system[-_ ]?test", re.I), "acceptance_test", "side-path acceptance test harness"),
]

PACKAGE_PARENT_DIRS = {"crates", "services", "packages", "apps", "libs", "modules"}
IGNORED_DIRS = {".git", ".github", ".agents", ".fcis", "target", "node_modules", ".next", "dist", "build", "src", "tests", "benches", "scripts", "vendor"}


def helper_command() -> str:
    return f"python3 {shlex.quote(str(Path(__file__).resolve()))}"


def project_name(root: Path) -> str:
    return root.resolve().name or "project"


def fcis_dir(root: Path) -> Path:
    return root / ".fcis"


def default_register() -> dict[str, Any]:
    return {"schema_version": 2, "units": []}


def load_register(root: Path) -> dict[str, Any]:
    path = fcis_dir(root) / "role_register.json"
    if not path.exists():
        return default_register()
    return json.loads(path.read_text(encoding="utf-8"))


def save_register(root: Path, data: dict[str, Any]) -> None:
    path = fcis_dir(root) / "role_register.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_if_missing(path: Path, content: str, *, force: bool = False) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def walk_dirs(root: Path, max_depth: int = 2) -> list[Path]:
    out: list[Path] = []
    def visit(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        for entry in entries:
            if entry.name in IGNORED_DIRS or not entry.is_dir():
                continue
            out.append(entry)
            visit(entry, depth + 1)
    visit(root, 0)
    return out


def is_likely_package_dir(relative_path: str) -> bool:
    parts = relative_path.split(os.sep)
    return (len(parts) == 1 and parts[0] in PACKAGE_PARENT_DIRS) or (len(parts) == 2 and parts[0] in PACKAGE_PARENT_DIRS)


def roles_markdown() -> str:
    lines = ["# FCIS Canonical Roles", "", "This file is generated project-local scaffolding. The checker derives rules from the bundled role graph; this file is not a second source of truth.", "", "| Role | Meaning |", "|---|---|"]
    for role, meaning in CANONICAL_ROLES.items():
        lines.append(f"| `{role}` | {meaning} |")
    lines.extend(["", "## Register discipline", "", "Only `.fcis/role_register.json` is required for checks. Register units with `path`, `roles`, `axis`, optional `bind_to`, optional `names`, and optional `sources`. Do not add project-local edge tables, forbid lists, extras, layers, or role-fold lists; the engine derives those from the canonical graph.", ""])
    return "\n".join(lines)


def preflight_xml() -> str:
    roles = "|".join(CANONICAL_ROLES)
    return f"""<fcis_preflight>
  <touched_units>
    <unit path="" roles="{roles}" axis="" bind_to=""/>
  </touched_units>
  <notes>
    For fast classification, read references/quick-mode.md or run `fcis_project.py quick`.
  </notes>
</fcis_preflight>
"""


def quick_reference() -> str:
    path = SCRIPT_DIR.parent / "references" / "quick-mode.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "FCIS quick reference is unavailable; inspect references/quick-mode.md in the skill package.\n"


def readme_template() -> str:
    return f"""# FCIS Project Files

These files are local to this repository. They are not a global FCIS language map.

## Files

- `roles.md` lists canonical role vocabulary.
- `role_register.json` is the only checker input. It contains only units.
- `preflight.xml` is a lightweight review template.
- `fcis.toml` records the helper command used to generate this directory.

## Register format

```json
{{
  "schema_version": 2,
  "units": [
    {{"path": "crates/kernel", "roles": ["meaning_seed"]}},
    {{"path": "crates/app", "roles": ["use_flow"], "axis": "core", "names": ["app"]}},
    {{"path": "crates/app-shapes", "roles": ["shape_seam"], "bind_to": "crates/app"}}
  ]
}}
```

Do not add `layers`, `edges`, `role_folds`, `direct_lanes`, `extra`, or forbid/import-ban lists. The checker derives imports from the canonical role graph and checks folds with the canonical allow-list.

## Commands

```bash
{helper_command()} quick
{helper_command()} register --root <repo-root> --path crates/app --roles use_flow --axis core --name app
{helper_command()} check --root <repo-root>
{helper_command()} check --root <repo-root> --fast
```
"""


def command_init(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"root does not exist: {root}")
    dot = fcis_dir(root)
    dot.mkdir(parents=True, exist_ok=True)
    force = bool(args.force)
    wrote = []
    if write_if_missing(dot / "README.md", readme_template(), force=force):
        wrote.append("README.md")
    if write_if_missing(dot / "roles.md", roles_markdown(), force=force):
        wrote.append("roles.md")
    if write_if_missing(dot / "role_register.json", json.dumps(default_register(), indent=2, sort_keys=True) + "\n", force=force):
        wrote.append("role_register.json")
    if write_if_missing(dot / "preflight.xml", preflight_xml(), force=force):
        wrote.append("preflight.xml")
    config = f"""# FCIS project-local configuration
project = \"{args.project_name or project_name(root)}\"
schema_version = 2
helper_command = \"{helper_command()}\"
role_register = \".fcis/role_register.json\"
"""
    if write_if_missing(dot / "fcis.toml", config, force=force):
        wrote.append("fcis.toml")
    print(f"initialized {dot}")
    print(f"wrote: {', '.join(wrote) if wrote else 'nothing; use --force to overwrite'}")
    return 0


def command_register(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    roles = list(split_roles(args.roles))
    unknown = [role for role in roles if role not in CANONICAL_ROLES]
    if unknown:
        raise SystemExit(f"unknown role(s): {', '.join(unknown)}")
    data = load_register(root)
    # Strip manual rule sections if they exist; this keeps the register declarative.
    data = {"schema_version": 2, "units": list(data.get("units", []))}
    units = data.setdefault("units", [])
    unit: dict[str, Any] = {"path": args.path, "roles": roles}
    if args.axis:
        unit["axis"] = args.axis
    if args.bind_to:
        unit["bind_to"] = args.bind_to
    if args.name:
        unit["names"] = args.name
    if args.source:
        unit["sources"] = args.source
    units[:] = [existing for existing in units if existing.get("path") != args.path]
    units.append(unit)
    data["units"] = sorted(units, key=lambda item: item.get("path", ""))
    save_register(root, data)
    result = check_register(root, data)
    if result.violations:
        save_register(root, {"schema_version": 2, "units": [u for u in units if u.get("path") != args.path]})
        return print_result(result)
    print(f"registered {args.path}: {'+'.join(roles)}")
    return 0


def command_check(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    if args.fast:
        # Fast mode validates the declarative register only. It skips source
        # existence and import scanning so agents can iterate on role/axis shape
        # before the full repo layout is ready.
        from fcis_rules import check_register
        return print_result(check_register(root))
    return print_result(check_imports(root))


def command_quick(args: argparse.Namespace) -> int:
    print(quick_reference(), end="")
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    print(quick_reference() if args.fast else preflight_xml(), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project-local FCIS role helper")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="initialize .fcis project files")
    p.add_argument("--root", default=".")
    p.add_argument("--project-name", default="")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=command_init)
    p = sub.add_parser("register", help="register a physical unit with canonical roles")
    p.add_argument("--root", default=".")
    p.add_argument("--path", required=True)
    p.add_argument("--roles", required=True, help="comma or plus separated canonical roles")
    p.add_argument("--axis", default="")
    p.add_argument("--bind-to", default="", help="target path for standalone shape_seam or config_binder")
    p.add_argument("--name", action="append", default=[], help="import/package/module name for this unit; repeatable")
    p.add_argument("--source", action="append", default=[], help="source root/file to scan; repeatable")
    p.set_defaults(func=command_register)
    p = sub.add_parser("check", help="auto-derive and check register plus imports")
    p.add_argument("--root", default=".")
    p.add_argument("--fast", action="store_true", help="validate only role_register.json; skip source/import scanning")
    p.set_defaults(func=command_check)
    p = sub.add_parser("quick", help="print the fast FCIS role-classification guide")
    p.set_defaults(func=command_quick)
    p = sub.add_parser("preflight", help="print the FCIS preflight XML template")
    p.add_argument("--fast", action="store_true", help="print the quick-mode guide instead of XML")
    p.set_defaults(func=command_preflight)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
