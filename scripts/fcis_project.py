#!/usr/bin/env python3
"""Project-local FCIS helper for schema 3."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fcis_rules import (  # noqa: E402
    AXIS_LINK_INTERACTIONS,
    CANONICAL_ROLES,
    LINK_PROTO_SCHEMA_ORIGINS,
    SCHEMA_VERSION,
    CheckResult,
    check_imports,
    check_register,
    print_result,
    split_roles,
)

ROLE_HINTS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"foundation|primitive|atom|utils?", re.I),
        "utils",
        "axisless-light closed utilities",
    ),
    (
        re.compile(
            r"run-?kit|operational|schema|generator|generation|migration|config|artifact|report|retry|cache|test[-_ ]?suite|bdd[-_ ]?producer",
            re.I,
        ),
        "run_kit",
        "axis-able direct-call generic operational kit",
    ),
    (
        re.compile(r"kernel|identity|seed", re.I),
        "meaning_seed",
        "shared minimal vocabulary",
    ),
    (
        re.compile(r"domain|catalog|sync|cite|fulltext|meaning", re.I),
        "meaning_core",
        "pure business meaning",
    ),
    (
        re.compile(r"app|app-?flow|use-?case|workflow|flow", re.I),
        "use_flow",
        "business workflow brain",
    ),
    (
        re.compile(r"facade|server|entry|route|resolver|controller", re.I),
        "entry_frame",
        "serving entry and exit framing",
    ),
    (
        re.compile(r"composition|assembly|bootstrap|main", re.I),
        "assembly_root",
        "bootstrap/supervisor and lifecycle wiring",
    ),
    (
        re.compile(r"idl|frame|message|protocol", re.I),
        "link_proto",
        "public bidirectional process/channel protocol/schema when public; otherwise owner-local schema",
    ),
    (
        re.compile(r"adapter|sqlite|storage|http|client|deno|js|cdp|browser|process|os|api|db", re.I),
        "effect_tool",
        "capability adapter/analyzer over concrete external mechanism",
    ),
    (
        re.compile(r"facility|daemon|sidecar|worker|process", re.I),
        "effect_facility",
        "long-lived generic effect owner",
    ),
    (
        re.compile(r"interop|cross|plugin|translator|host|bridge|port|contract", re.I),
        "axis_link",
        "directed cross-axis semantic bridge",
    ),
    (
        re.compile(r"observability|analy[sz]er|metrics?|events?|logs?|signals?", re.I),
        "effect_tool",
        "passive signal/schema analysis tool; use a dedicated signal/observability axis",
    ),
    (
        re.compile(r"acceptance|e2e|end[-_ ]?to[-_ ]?end|probe|system[-_ ]?test", re.I),
        "acceptance_test",
        "side-path acceptance test harness",
    ),
]

PACKAGE_PARENT_DIRS = {"crates", "services", "packages", "apps", "libs", "modules"}
IGNORED_DIRS = {
    ".git",
    ".github",
    ".agents",
    ".fcis",
    "target",
    "node_modules",
    ".next",
    "dist",
    "build",
    "src",
    "tests",
    "benches",
    "scripts",
    "vendor",
}


def helper_command() -> str:
    return f"python3 {shlex.quote(str(Path(__file__).resolve()))}"


def project_name(root: Path) -> str:
    return root.resolve().name or "project"


def fcis_dir(root: Path) -> Path:
    return root / ".fcis"


def default_register() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "units": []}


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
    return (len(parts) == 1 and parts[0] in PACKAGE_PARENT_DIRS) or (
        len(parts) == 2 and parts[0] in PACKAGE_PARENT_DIRS
    )


def roles_markdown() -> str:
    lines = [
        "# FCIS Canonical Roles",
        "",
        "This file is generated project-local scaffolding. The checker derives rules from the bundled role graph; this file is not a second source of truth.",
        "",
        "| Role | Meaning |",
        "|---|---|",
    ]
    for role, meaning in CANONICAL_ROLES.items():
        lines.append(f"| `{role}` | {meaning} |")
    lines.extend(
        [
            "",
            "## Register discipline",
            "",
            "Only `.fcis/role_register.json` is required for checks. Register units with `path`, `roles`, `axis` for ordinary semantic-owner roles; `route_axes` plus `entry_type` for entry/front-door units; `axis_participants`, `provider_axis`, `consumer_axis`, and `interaction` for standalone directed `axis_link`; optional `names`; optional `sources`; owned non-default `externals`; and optional `schema_origin` for standalone `link_proto` units. `sources` may be path strings or objects with `path` plus optional role metadata. Do not add project-local edge tables, forbid lists, extras, layers, role-fold lists, ad-hoc orientation fields, or binding fields; the engine derives those from the canonical graph.",
            "",
            "Owner-local app mapping belongs in private `mapper` modules/files; in entry_frame and use_flow, mapper means bidirectional boundary transformation. Module schema belongs in `schema`; the operation that emits schema artifacts is `schema_generate`/`generate_schema`; generated output belongs in `schema.gen`/`schema_gen`. A dev/build script may collect per-owner generated artifacts; it must not become a production role edge. Public bidirectional process/channel protocol/schema belongs in axisless `link_proto`, with `schema_origin` set to `code_generated` or `contract_authored`; document schema owner, producer/consumer axes, semantic refs, security boundary, version policy, and conformance tests next to the protocol artifact. Auth/business ownership stays with semantic owners. `common` is only a physical grouping folder; register child crates by their real roles. `utils` is axisless-light closed helper code: no raw env/config/files/process probing, lifecycle, registry/dispatch, open runtime state, or business vocabulary. Axis-able generic operational open-input/runtime/build/test/config/generation/migration/report/artifact machinery belongs to vocabulary-free `run_kit` when a direct-call helper API is enough; heavy external packages are allowed there. Concrete external mechanism adapters/analyzers that need a capability seam belong to `effect_tool`, and local effect lifecycle belongs to `effect_facility`. Deployment config/env ingress belongs to a dedicated `run_kit` with exactly axis `config`; `assembly_root` delegates config to that run kit, frames the process front-door request, starts/forks worker entry_frames when needed, and supervises lifecycle only when it owns local workers. It does not own config externals, worker facility wiring, or direct effect imports. A release-binary/service CLI is `assembly_root` bootstrap: subcommand is routing at the composition root, and flag/option handling is config/bootstrap concern unless explicitly modeled otherwise. Entry/front-door units do not own semantic `axis`; they declare `route_axes` for which semantic axes they route into and `entry_type` for surface/platform tags such as server, web, desktop, mobile, sdk, or admin. `cli` is a special entry_type reserved for assembly_root request framing. CLI argv, REST, GraphQL, MCP, gRPC service methods, SDK calls, queue jobs, browser events, and admin/probe commands are front-door handler models, not config sources and not separate axes; stdio/stdin-stdout/HTTP/SSE/WebSocket/UDS/TCP/pipes/queue transports are carrier bindings, with stable message/schema contracts in link_proto. `signal_analyzer` is not a role: generic signal envelopes/artifact helpers are run_kit, mechanism-specific analyzers are effect_tool, health semantics belong to a diagnostics axis, watchdog/listener/heartbeat loops are effect_facility, and probe/debug protocols are link_proto. `axis_link` uses `provider_axis -> consumer_axis` plus `axis_participants` and `interaction`; `required_port` is caller/core/workflow-owned (dependency inversion: the consumer owns the required shape, the provider implements it), pure entry_frame must not import required_port directly, `provider_event` is the semantic event contract, and buses/protocols stay in effect_facility/link_proto.",
            "",
        ]
    )
    return "\n".join(lines)


def preflight_xml() -> str:
    roles = "|".join(CANONICAL_ROLES)
    return f"""<fcis_preflight>
  <touched_units>
    <unit path="" roles="{roles}" axis="" axis_participants="" provider_axis="" consumer_axis="" interaction="" />
  </touched_units>
  <notes>
    For fast classification, read references/quick-mode.md or run `fcis_project.py quick`.
    Use private mapper files for owner-local bidirectional transforms. Use schema_generate/generate_schema for per-owner schema generation operations and schema_gen/schema.gen for generated artifacts. A dev/build script may collect these artifacts for tooling. Public bidirectional process/channel protocol/schema is axisless link_proto; use schema_origin=code_generated for code-first artifacts or schema_origin=contract_authored for IDL/design-first contracts, and document richer protocol governance next to the schema. Auth/business ownership stays with semantic owners. `common` is only a physical grouping folder; register child crates by real role. `utils` is axisless-light closed helper code: no raw env/config/files/process probing, lifecycle, registry/dispatch, open runtime state, or business vocabulary. Axis-able generic operational open-input/runtime/build/test/config/generation/migration/report/artifact machinery belongs to vocabulary-free run_kit when a direct-call helper API is enough; heavy external packages are allowed there. Concrete external mechanism adapters/analyzers that need a capability seam belong to effect_tool, and local effect lifecycle belongs to effect_facility. Deployment config/env ingress belongs to a dedicated run_kit with exactly axis config; assembly_root is an axisless bootstrap/supervisor that gets typed config from run_kit, frames typed front-door requests, starts/forks worker entry_frames when needed, handles master lifecycle signals, supervises lifecycle only when it owns local workers, and does not import effect_tool or effect_facility directly. Runnable concrete tools/analyzers should use an assembly_root+entry_frame wrapper or a standalone assembly_root -> entry_frame shape. entry_frame is a dumb worker front door: service inbound comes from a facility. Entry/front-door units do not own semantic `axis`; they declare `route_axes` for which semantic axes they route into and `entry_type` for surface/platform tags such as server, web, desktop, mobile, sdk, or admin. `cli` is a special entry_type reserved for assembly_root request framing. A release-binary/service CLI is assembly_root bootstrap: subcommand is routing at the composition root, and flag/option handling is config/bootstrap concern unless explicitly modeled otherwise. CLI argv, REST, GraphQL, MCP, gRPC service methods, SDK calls, queue jobs, browser events, and admin/probe commands are front-door handler models, not config sources and not separate axes; stdio/stdin-stdout/HTTP/SSE/WebSocket/UDS/TCP/pipes/queue transports are carrier bindings, with stable message/schema contracts in link_proto. `signal_analyzer` is not a role: generic signal envelopes/artifact helpers are run_kit, mechanism-specific analyzers are effect_tool, health semantics are diagnostics meaning/use_flow, watchdog/listener/heartbeat runtime is effect_facility, and probe/debug protocols are link_proto. axis_link uses provider_axis -> consumer_axis plus axis_participants and interaction; required_port is caller/core/workflow-owned (dependency inversion: the consumer owns the required shape, the provider implements it), pure entry_frame must not import required_port directly, provider_event is semantic event consumption, and same-axis object callbacks are not roles.
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
  "schema_version": {SCHEMA_VERSION},
  "units": [
    {{"path": "crates/kernel", "roles": ["meaning_seed"]}},
    {{"path": "crates/app", "roles": ["use_flow"], "axis": "core", "names": ["app"]}},
    {{"path": "crates/http-tool", "roles": ["effect_tool"], "axis": "core", "names": ["http_tool"], "externals": ["reqwest"]}},
    {{"path": "services/api", "roles": ["entry_frame"], "route_axes": "core", "entry_type": "server+web", "names": ["api_entry"]}},
    {{"path": "crates/catalog-billing-link", "roles": ["axis_link"], "axis_participants": "catalog+billing", "provider_axis": "catalog", "consumer_axis": "billing", "interaction": "provider_port", "names": ["catalog_billing_link"]}},
    {{"path": "crates/framework-host-port", "roles": ["axis_link"], "axis_participants": "framework+host", "provider_axis": "host", "consumer_axis": "framework", "interaction": "required_port", "names": ["framework_host_port"]}},
    {{"path": "services/main", "roles": ["assembly_root"], "names": ["main"]}}
  ]
}}
```

Do not add any top-level fields except `schema_version` and `units`. Unit fields are `path`, `roles`, `axis` for semantic-owner roles, `route_axes` plus optional `entry_type` for units that include `entry_frame` or `assembly_root`, `axis_participants`, `provider_axis`, `consumer_axis`, `interaction`, `names`, `sources`, `externals`, and optional `schema_origin` for standalone `link_proto` units. `entry_type` records front-door surface/platform tags such as server, web, desktop, mobile, sdk, or admin. `cli` is reserved for assembly_root request framing and must not be registered on a pure entry_frame. `axis_participants`, `provider_axis`, `consumer_axis`, and `interaction` are valid only for standalone `axis_link`. Do not add ad-hoc orientation fields, `layers`, `edges`, `role_folds`, `direct_lanes`, `extra`, `bind_to`, `shape_for`, `force`, `runtime_env`, protocol-governance fields, or forbid/import-ban lists. The checker derives imports from the canonical role graph and checks folds with the canonical allow-list. `common` is only a physical grouping folder; register each child crate by its real role. For a runnable concrete tool/analyzer app, register an assembly_root+entry_frame wrapper or a standalone assembly_root -> entry_frame shape; do not fold entry_frame with use_flow. A release-binary/service CLI is assembly_root bootstrap: subcommand is routing at the composition root, and flag/option handling is config/bootstrap concern unless explicitly modeled otherwise. Entry/front-door units do not own semantic `axis`; they declare `route_axes` for which application axes the front door routes into. `entry_frame` may route into at most two semantic axes, while `use_flow` remains exactly one axis. CLI argv, REST, GraphQL, MCP, gRPC service methods, SDK calls, queue jobs, browser events, and admin/probe commands are front-door handler models, not config sources and not separate axes; stdio/stdin-stdout/HTTP/SSE/WebSocket/UDS/TCP/pipes/queue transports are carrier bindings, with stable message/schema contracts in link_proto. For Rust, only `serde`, `thiserror`, `anyhow`, `tokio`, `tracing`, `tempfile`, and `regex` are globally direct by default; other third-party crates must be bound with `externals` on run_kit/effect_tool/effect_facility units. Use run_kit for direct-call generic operational packages, effect_tool for concrete external mechanism adapters/analyzers that need a capability seam, and effect_facility for local effect lifecycle. Config/env packages are bound with `externals` on a dedicated run_kit unit whose axis is exactly `config`, not on assembly_root, not on a multi-axis config+app unit, and not on a service entry_frame. Pure entry_frame must not import required_port directly. When `sources` narrows scanning, package manifests under the unit path are still checked. For `axis_link`, use exactly two `axis_participants`, `provider_axis`, `consumer_axis`, and an `interaction` value of `provider_port`, `provider_event`, `provider_adapter`, `provider_interface`, or `required_port`. Provider-export links must reference provider meaning; required ports hold caller/core/workflow-owned required shapes (dependency inversion) and must not import provider implementation internals. Split duplex or multi-party contracts into separate directed links or use `link_proto` when bytes/messages are the real boundary. Release readiness and signal_analyzer are not role-register units; use release_test/evidence over the real owner, and classify signal/health/watchdog/debug code by concrete ownership instead.

For `link_proto`, set `schema_origin` when direction matters: `code_generated` means code-owned definitions generate public schema artifacts; `contract_authored` means an authored IDL/schema/protocol contract is the source and code conforms to or is generated from it. Keep richer protocol governance in the protocol README/contract docs, not in role_register.json.

## Commands

```bash
{helper_command()} quick
{helper_command()} register --root <repo-root> --path crates/app --roles use_flow --axis core --name app
{helper_command()} register --root <repo-root> --path crates/http-tool --roles effect_tool --axis core --name http_tool --external reqwest
{helper_command()} register --root <repo-root> --path services/api --roles entry_frame --route-axes core --entry-type server+web --name api_entry
{helper_command()} register --root <repo-root> --path crates/catalog-billing-link --roles axis_link --axis-participants catalog+billing --provider-axis catalog --consumer-axis billing --interaction provider_port --name catalog_billing_link
{helper_command()} register --root <repo-root> --path crates/framework-host-port --roles axis_link --axis-participants framework+host --provider-axis host --consumer-axis framework --interaction required_port --name framework_host_port
{helper_command()} register --root <repo-root> --path services/main --roles assembly_root --name main
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
    if write_if_missing(
        dot / "role_register.json",
        json.dumps(default_register(), indent=2, sort_keys=True) + "\n",
        force=force,
    ):
        wrote.append("role_register.json")
    if write_if_missing(dot / "preflight.xml", preflight_xml(), force=force):
        wrote.append("preflight.xml")
    config = f"""# FCIS project-local configuration
project = "{args.project_name or project_name(root)}"
schema_version = {SCHEMA_VERSION}
helper_command = "{helper_command()}"
role_register = ".fcis/role_register.json"
"""
    if write_if_missing(dot / "fcis.toml", config, force=force):
        wrote.append("fcis.toml")
    print(f"initialized {dot}")
    print(
        f"wrote: {', '.join(wrote) if wrote else 'nothing; use --force to overwrite'}"
    )
    return 0


def command_register(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    roles = list(split_roles(args.roles))
    unknown = [role for role in roles if role not in CANONICAL_ROLES]
    if unknown:
        raise SystemExit(f"unknown role(s): {', '.join(unknown)}")
    old_data = load_register(root)
    data = {"schema_version": SCHEMA_VERSION, "units": list(old_data.get("units", []))}
    units = data.setdefault("units", [])
    unit: dict[str, Any] = {"path": args.path, "roles": roles}
    if args.axis:
        unit["axis"] = args.axis
    if args.route_axes:
        unit["route_axes"] = args.route_axes
    if args.entry_type:
        unit["entry_type"] = args.entry_type
    if args.axis_participants:
        unit["axis_participants"] = args.axis_participants
    if args.provider_axis:
        unit["provider_axis"] = args.provider_axis
    if args.consumer_axis:
        unit["consumer_axis"] = args.consumer_axis
    if args.interaction:
        unit["interaction"] = args.interaction
    if args.name:
        unit["names"] = args.name
    if args.source:
        unit["sources"] = args.source
    if args.external:
        unit["externals"] = args.external
    if args.schema_origin:
        unit["schema_origin"] = args.schema_origin
    units[:] = [existing for existing in units if existing.get("path") != args.path]
    units.append(unit)
    data["units"] = sorted(units, key=lambda item: item.get("path", ""))
    save_register(root, data)
    result = check_register(root, data)
    if result.violations:
        save_register(root, old_data)
        return print_result(result)
    print(f"registered {args.path}: {'+'.join(roles)}")
    return 0


def _register_preflight(root: Path, *, allow_empty_bootstrap: bool) -> CheckResult:
    result = CheckResult()
    if not root.exists():
        result.violations.append(f"repo root does not exist: {root}")
        return result
    register_path = fcis_dir(root) / "role_register.json"
    if not register_path.exists():
        result.violations.append(
            f"role_register.json not found: {register_path}; run init/register first or pass --allow-empty-bootstrap only during first setup"
        )
        return result
    try:
        raw = json.loads(register_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - CLI preflight should report bad JSON
        result.violations.append(f"role_register.json is not readable JSON: {exc}")
        return result
    raw_units = raw.get("units", []) if isinstance(raw, dict) else []
    unit_count = len(raw_units) if isinstance(raw_units, list) else 0
    if unit_count == 0 and not allow_empty_bootstrap:
        result.violations.append(
            "role_register.json has zero units; no architecture facts were checked. "
            "Register at least one unit or pass --allow-empty-bootstrap only during initial setup."
        )
    result.notes.append(f"Registered units checked: {unit_count}")
    return result


def command_check(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    preflight = _register_preflight(root, allow_empty_bootstrap=args.allow_empty_bootstrap)
    if preflight.violations:
        return print_result(preflight)
    result = check_register(root) if args.fast else check_imports(root)
    result.notes = preflight.notes + result.notes
    return print_result(result)


def command_quick(args: argparse.Namespace) -> int:
    print(quick_reference(), end="")
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    print(quick_reference() if args.fast else preflight_xml(), end="")
    return 0


def command_suggest(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    rows: list[tuple[str, str, str]] = []
    for path in walk_dirs(root, max_depth=args.depth):
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if not is_likely_package_dir(rel):
            continue
        role = ""
        why = ""
        for pattern, candidate, explanation in ROLE_HINTS:
            if pattern.search(rel):
                role = candidate
                why = explanation
                break
        if role:
            rows.append((rel, role, why))
    for rel, role, why in rows:
        print(
            f"{rel}: {role}  # candidate only; directory names are weak evidence; inspect public API, lifecycle ownership, axis, imports, and sources before registering; {why}"
        )
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
    p.add_argument(
        "--roles", required=True, help="comma or plus separated canonical roles"
    )
    p.add_argument("--axis", default="")
    p.add_argument(
        "--route-axes",
        default="",
        help="for units that include entry_frame: semantic axes that the front door routes into",
    )
    p.add_argument(
        "--entry-type",
        default="",
        help="for units that include entry_frame or assembly_root: front-door tags such as server,web,desktop,admin; cli is reserved for assembly_root",
    )
    p.add_argument("--axis-participants", default="", help="for standalone axis_link: exactly provider+consumer axes, e.g. host+plugin")
    p.add_argument("--provider-axis", default="", help="for standalone axis_link: axis that owns the meaning_core capability")
    p.add_argument("--consumer-axis", default="", help="for standalone axis_link: axis that consumes the provider capability")
    p.add_argument(
        "--interaction",
        choices=sorted(AXIS_LINK_INTERACTIONS),
        default="",
        help="for standalone axis_link: call, callback, event, adapter, port, or interface",
    )
    p.add_argument(
        "--name",
        action="append",
        default=[],
        help="import/package/module name for this unit; repeatable",
    )
    p.add_argument(
        "--source",
        action="append",
        default=[],
        help="source root/file to scan; repeatable",
    )
    p.add_argument(
        "--external",
        action="append",
        default=[],
        help="non-default external dependency bound to this owner unit; repeatable",
    )
    p.add_argument(
        "--schema-origin",
        choices=sorted(LINK_PROTO_SCHEMA_ORIGINS),
        default="",
        help="for standalone link_proto: code_generated or contract_authored",
    )
    p.set_defaults(func=command_register)
    p = sub.add_parser("check", help="auto-derive and check register plus imports")
    p.add_argument("--root", default=".")
    p.add_argument(
        "--fast",
        action="store_true",
        help="validate only role_register.json; skip source/import scanning",
    )
    p.add_argument(
        "--allow-empty-bootstrap",
        action="store_true",
        help="allow an empty role register only during first-time setup; never use to claim architecture pass",
    )
    p.set_defaults(func=command_check)
    p = sub.add_parser("quick", help="print the fast FCIS role-classification guide")
    p.set_defaults(func=command_quick)
    p = sub.add_parser("preflight", help="print the FCIS preflight XML template")
    p.add_argument(
        "--fast", action="store_true", help="print the quick-mode guide instead of XML"
    )
    p.set_defaults(func=command_preflight)
    p = sub.add_parser(
        "suggest",
        help="suggest likely roles from directory names; review behavior before accepting",
    )
    p.add_argument("--root", default=".")
    p.add_argument("--depth", type=int, default=2)
    p.set_defaults(func=command_suggest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
