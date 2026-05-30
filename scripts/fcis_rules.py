#!/usr/bin/env python3
"""Shared FCIS schema 3 role-graph engine.

A project registers only physical units, their roles, axes, optional import
names/source roots, and adapter-owned externals in `.fcis/role_register.json`.
All dependency and fold decisions are auto-derived from the canonical role graph.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3

CANONICAL_ROLES: dict[str, str] = {
    "base_atoms": "generic primitive base",
    "base_kit": "generic helper APIs",
    "meaning_seed": "axis-capable smallest shared business vocabulary",
    "meaning_core": "pure business meaning and deterministic rules",
    "run_kit": "axis-capable generic runtime helpers and signal envelopes",
    "axis_surface": "narrow same-axis public semantic/capability surface",
    "axis_link": "semantic API between horizontal axes",
    "axis_joint": "fold-only conceptual same-axis crossroad; never a standalone executable surface",
    "use_flow": "business workflow brain",
    "effect_tool": "generic reusable adapter/tool library with public porcelain API and concrete implementation",
    "effect_facility": "long-lived generic effect owner with runtime state",
    "link_proto": "public protocol/schema contract for cross-axis data or messages; no transport ownership",
    "entry_frame": "worker/serving entry: manages facility lifecycle handles, routes facility-produced requests/jobs to use_flow, and shapes response/exit with private mapper code",
    "assembly_root": "composition root for resource construction, persistent config ingress, and lifecycle wiring",
    "signal_analyzer": "passive checker/aggregator over emitted signal records",
    "acceptance_test": "side-path end-to-end/probe test harness that may import production roles",
}

BASE_EDGES: dict[str, set[str]] = {
    "base_atoms": set(),
    "base_kit": {"base_atoms"},
    "meaning_seed": {"base_atoms"},
    "run_kit": {"base_kit", "base_atoms"},
    "meaning_core": {"meaning_seed"},
    "axis_surface": {"meaning_seed"},
    "axis_link": {"axis_surface", "link_proto"},
    "use_flow": {
        "meaning_core",
        "meaning_seed",
        "run_kit",
        "axis_surface",
        "axis_link",
        "effect_tool",
    },
    "entry_frame": {"use_flow", "effect_facility", "run_kit"},
    "assembly_root": {"entry_frame", "effect_facility", "run_kit"},
    "effect_tool": {"run_kit", "base_kit", "base_atoms"},
    "effect_facility": {"effect_tool", "run_kit", "base_kit", "base_atoms"},
    "link_proto": {"base_kit", "base_atoms", "run_kit"},
    "signal_analyzer": {"run_kit", "base_kit", "base_atoms"},
    "acceptance_test": set(),
}

ALLOWED_ROLE_FOLDS: set[frozenset[str]] = {
    frozenset({"base_atoms", "base_kit"}),
    frozenset({"assembly_root", "entry_frame"}),
    frozenset({"use_flow", "axis_joint"}),
}

PROTECTED_STANDALONE_ROLES = {
    "meaning_seed",
    "run_kit",
    "effect_tool",
    "effect_facility",
    "axis_link",
    "link_proto",
    "signal_analyzer",
    "acceptance_test",
}

UNIVERSAL_ROLES = {"base_atoms", "base_kit"}
SHARED_DEFAULT_AXIS_ROLES = {"meaning_seed", "run_kit", "assembly_root"}
NO_AXIS_ROLES = (
    UNIVERSAL_ROLES | SHARED_DEFAULT_AXIS_ROLES | {"signal_analyzer", "acceptance_test"}
)
MULTI_AXIS_ROLES = {
    "meaning_seed",
    "run_kit",
    "entry_frame",
    "assembly_root",
    "axis_link",
    "link_proto",
    "effect_tool",
    "effect_facility",
}
SIDE_PATH_ROLES = {"signal_analyzer", "acceptance_test"}

SOURCE_EXTENSIONS = {
    ".rs",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".java",
    ".kt",
    ".swift",
    ".c",
    ".h",
    ".hpp",
    ".cpp",
    ".cs",
}
RUST_DEFAULT_DIRECT_CRATES = {
    "std",
    "core",
    "alloc",
    "proc_macro",
    "serde",
    "thiserror",
    "anyhow",
    "tokio",
    "tracing",
    "tempfile",
}
RUST_EXTERNAL_OWNER_ROLES = {"effect_tool", "effect_facility"}
CONFIG_INGRESS_EXACT = {
    "@iarna/toml",
    "configparser",
    "decouple",
    "dotenv",
    "dotenv/config",
    "dotenvy",
    "dynaconf",
    "environs",
    "figment",
    "godotenv",
    "js_yaml",
    "pydantic_settings",
    "pyyaml",
    "serde_yaml",
    "toml",
    "tomli",
    "tomllib",
    "viper",
    "yaml",
}
CONFIG_INGRESS_PREFIXES = (
    "dotenv/",
    "github.com/joho/godotenv",
    "github.com/spf13/viper",
)
CONFIG_LOADER_FUNCTION_NAMES = {
    "load_config",
    "load_runtime_config",
    "prepare_runtime_config_mapping",
}


@dataclass(frozen=True)
class Unit:
    path: str
    roles: tuple[str, ...]
    axis: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    externals: tuple[str, ...] = ()


@dataclass
class CheckResult:
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ImportRef:
    source_path: str
    target_name: str
    line_no: int
    line: str


def parse_axis(axis: str | Sequence[str] | None) -> tuple[str, ...]:
    if axis is None:
        return ()
    if isinstance(axis, str):
        return tuple(part for part in re.split(r"[,+]", axis) if part)
    return tuple(str(part) for part in axis if str(part))


def split_roles(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        return tuple(part.strip() for part in re.split(r"[,+]", raw) if part.strip())
    return tuple(str(part).strip() for part in raw if str(part).strip())


def source_paths(raw_sources: Any) -> tuple[str, ...]:
    if not isinstance(raw_sources, list):
        return ()
    paths: list[str] = []
    for raw_source in raw_sources:
        if isinstance(raw_source, str):
            path = raw_source.strip()
        elif isinstance(raw_source, Mapping):
            path = str(raw_source.get("path", "")).strip()
        else:
            path = ""
        if path:
            paths.append(path)
    return tuple(paths)


def default_names(path: str) -> tuple[str, ...]:
    path_obj = Path(path)
    base = path_obj.stem if path_obj.suffix else path_obj.name
    snake = base.replace("-", "_")
    names = {base, snake}
    if base.startswith("lib") and len(base) > 3:
        names.add(base[3:])
    return tuple(sorted(names))


def normalize_unit(raw: Mapping[str, Any]) -> Unit:
    path = str(raw.get("path", "")).strip()
    roles = split_roles(raw.get("roles", ()))
    axis = parse_axis(raw.get("axis", ""))
    raw_names = raw.get("names", ())
    names = (
        tuple(str(name).strip() for name in raw_names if str(name).strip())
        if isinstance(raw_names, list)
        else ()
    )
    if not names:
        names = default_names(path)
    sources = source_paths(raw.get("sources", ()))
    raw_externals = raw.get("externals", ())
    externals = (
        tuple(str(dep).strip() for dep in raw_externals if str(dep).strip())
        if isinstance(raw_externals, list)
        else ()
    )
    return Unit(
        path=path,
        roles=roles,
        axis=axis,
        names=names,
        sources=sources,
        externals=externals,
    )


def load_register(root: Path) -> dict[str, Any]:
    path = root / ".fcis" / "role_register.json"
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "units": []}
    return json.loads(path.read_text(encoding="utf-8"))


def load_units(root: Path, data: Mapping[str, Any] | None = None) -> list[Unit]:
    data = load_register(root) if data is None else data
    return [normalize_unit(unit) for unit in data.get("units", [])]


def role_edges(role: str) -> set[str]:
    return set(BASE_EDGES.get(role, set()))


def is_universal(unit: Unit) -> bool:
    return bool(unit.roles) and all(role in UNIVERSAL_ROLES for role in unit.roles)


def axis_overlaps(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return bool(set(left) & set(right))


def effective_axis(unit: Unit, units_by_path: Mapping[str, Unit]) -> tuple[str, ...]:
    return unit.axis


def duplicate_names(units: Sequence[Unit]) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {}
    for unit in units:
        for name in unit.names:
            seen.setdefault(name, []).append(unit.path)
    return {name: paths for name, paths in seen.items() if len(set(paths)) > 1}


def rust_crate_key(name: str) -> str:
    root = name.strip().split("::", 1)[0].split("/", 1)[0].split(".", 1)[0]
    return root.replace("-", "_")


def rust_external_owners(units: Sequence[Unit]) -> dict[str, Unit]:
    owners: dict[str, Unit] = {}
    for unit in units:
        for dep in unit.externals:
            owners[rust_crate_key(dep)] = unit
    return owners


def missing_source_roots(root: Path, unit: Unit) -> list[str]:
    missing: list[str] = []
    for src in source_roots(root, unit):
        if not src.exists():
            try:
                missing.append(src.relative_to(root.resolve()).as_posix())
            except ValueError:
                missing.append(str(src))
    return missing


def _relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def overlapping_source_files(root: Path, units: Sequence[Unit]) -> dict[str, list[str]]:
    owners_by_file: dict[str, list[str]] = {}
    for unit in units:
        for file_path in source_files_for_unit(root, unit):
            owners_by_file.setdefault(
                _relative_or_absolute(root, file_path), []
            ).append(unit.path)
    return {
        file_path: sorted(set(paths))
        for file_path, paths in owners_by_file.items()
        if len(set(paths)) > 1
    }


def axis_visible(src: Unit, dst: Unit, units_by_path: Mapping[str, Unit]) -> bool:
    if is_universal(dst):
        return True
    dst_roles = set(dst.roles)
    if dst_roles & {"meaning_seed", "run_kit"} and not effective_axis(
        dst, units_by_path
    ):
        return True
    src_axis = effective_axis(src, units_by_path)
    dst_axis = effective_axis(dst, units_by_path)
    if dst_roles & {"axis_link"}:
        return bool(set(src.roles) & {"assembly_root"}) or (
            bool(src_axis) and bool(dst_axis) and axis_overlaps(src_axis, dst_axis)
        )
    if set(src.roles) & {"assembly_root"}:
        return True
    if not src_axis:
        return not dst_axis or bool(dst_roles & SHARED_DEFAULT_AXIS_ROLES)
    if not dst_axis:
        return bool(
            dst_roles & (UNIVERSAL_ROLES | SHARED_DEFAULT_AXIS_ROLES | SIDE_PATH_ROLES)
        )
    return axis_overlaps(src_axis, dst_axis)


def role_direct_allowed(src_role: str, dst_role: str) -> bool:
    return dst_role in role_edges(src_role)


def role_dependency_allowed(
    src: Unit, dst: Unit, units_by_path: Mapping[str, Unit]
) -> bool:
    if src.path == dst.path:
        return True
    if set(src.roles) == {"acceptance_test"}:
        return True
    if not axis_visible(src, dst, units_by_path):
        return False
    for src_role in src.roles:
        for dst_role in dst.roles:
            if role_direct_allowed(src_role, dst_role):
                return True
    return False


def has_effect_facility_for_axis(unit: Unit, units_by_path: Mapping[str, Unit]) -> bool:
    unit_axis = effective_axis(unit, units_by_path)
    for candidate in units_by_path.values():
        if candidate.path == unit.path or "effect_facility" not in candidate.roles:
            continue
        candidate_axis = effective_axis(candidate, units_by_path)
        if unit_axis and candidate_axis:
            if axis_overlaps(unit_axis, candidate_axis):
                return True
        elif not unit_axis or not candidate_axis:
            return True
    return False


def fold_allowed(unit: Unit, units_by_path: Mapping[str, Unit]) -> bool:
    roles = set(unit.roles)
    if roles == {"use_flow", "axis_joint"}:
        return True
    if roles == {"assembly_root", "entry_frame", "use_flow"}:
        return not has_effect_facility_for_axis(unit, units_by_path)
    if len(roles) <= 1:
        return True
    if len(roles) > 2:
        return False
    if roles & PROTECTED_STANDALONE_ROLES:
        return False
    if "use_flow" in roles:
        return False
    return frozenset(roles) in ALLOWED_ROLE_FOLDS


def check_register(root: Path, data: Mapping[str, Any] | None = None) -> CheckResult:
    result = CheckResult()
    raw = load_register(root) if data is None else data
    allowed_top = {"schema_version", "units"}
    for key in sorted(set(raw) - allowed_top):
        result.violations.append(
            f"role_register.json must be declarative units only; remove unsupported top-level {key}"
        )
    if raw.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        result.violations.append(
            f"role_register.json schema_version must be {SCHEMA_VERSION}"
        )
    raw_units = list(raw.get("units", []))
    allowed_unit_keys = {"path", "roles", "axis", "names", "sources", "externals"}
    for raw_unit in raw_units:
        extra_keys = sorted(set(raw_unit) - allowed_unit_keys)
        if extra_keys:
            result.violations.append(
                f"{raw_unit.get('path', '<missing path>')}: unsupported unit field(s): {', '.join(extra_keys)}"
            )
        raw_sources = raw_unit.get("sources", ())
        if raw_sources and not isinstance(raw_sources, list):
            result.violations.append(
                f"{raw_unit.get('path', '<missing path>')}: sources must be a list of paths or source objects"
            )
        elif isinstance(raw_sources, list):
            unit_roles = set(split_roles(raw_unit.get("roles", ())))
            for index, raw_source in enumerate(raw_sources):
                source_label = (
                    f"{raw_unit.get('path', '<missing path>')}: sources[{index}]"
                )
                if isinstance(raw_source, str):
                    if not raw_source.strip():
                        result.violations.append(
                            f"{source_label}: source path is empty"
                        )
                    continue
                if not isinstance(raw_source, Mapping):
                    result.violations.append(
                        f"{source_label}: source must be a path string or object with path"
                    )
                    continue
                source_extra_keys = sorted(set(raw_source) - {"path", "roles"})
                if source_extra_keys:
                    result.violations.append(
                        f"{source_label}: unsupported source field(s): {', '.join(source_extra_keys)}"
                    )
                source_path = str(raw_source.get("path", "")).strip()
                if not source_path:
                    result.violations.append(f"{source_label}: source path is empty")
                source_roles = set(split_roles(raw_source.get("roles", ())))
                unknown_source_roles = sorted(source_roles - set(CANONICAL_ROLES))
                if unknown_source_roles:
                    result.violations.append(
                        f"{source_label}: unknown source role(s): {', '.join(unknown_source_roles)}"
                    )
                if source_roles and unit_roles and not source_roles <= unit_roles:
                    result.violations.append(
                        f"{source_label}: source roles must be a subset of the unit roles"
                    )
    units = load_units(root, raw)
    units_by_path = {unit.path: unit for unit in units}
    if len(units_by_path) != len(units):
        result.violations.append("duplicate unit path in role_register.json")
    for name, paths in sorted(duplicate_names(units).items()):
        result.violations.append(
            f"duplicate import name {name!r} used by: {', '.join(sorted(set(paths)))}"
        )
    external_owners: dict[str, list[str]] = {}
    for unit in units:
        for dep in unit.externals:
            external_owners.setdefault(rust_crate_key(dep), []).append(unit.path)
    for dep_key, paths in sorted(external_owners.items()):
        if len(set(paths)) > 1:
            result.violations.append(
                f"duplicate Rust external crate binding {dep_key!r} used by: {', '.join(sorted(set(paths)))}"
            )
    for unit in units:
        if not unit.path:
            result.violations.append("unit missing path")
        if not unit.roles:
            result.violations.append(f"{unit.path}: unit missing roles")
        unknown = [role for role in unit.roles if role not in CANONICAL_ROLES]
        if unknown:
            result.violations.append(
                f"{unit.path}: unknown role(s): {', '.join(unknown)}"
            )
            continue
        if not fold_allowed(unit, units_by_path):
            result.violations.append(
                f"{unit.path}: illegal role fold {'+'.join(unit.roles)}"
            )
        roles = set(unit.roles)
        if "axis_joint" in roles and roles != {"use_flow", "axis_joint"}:
            result.violations.append(
                f"{unit.path}: axis_joint is conceptual and fold-only; register it only as use_flow+axis_joint"
            )
        if roles & UNIVERSAL_ROLES:
            if unit.axis:
                result.violations.append(
                    f"{unit.path}: base_atoms/base_kit must not declare axis"
                )
            if not roles <= UNIVERSAL_ROLES:
                result.violations.append(
                    f"{unit.path}: base_atoms/base_kit may fold only with each other"
                )
        if unit.externals and not (roles & RUST_EXTERNAL_OWNER_ROLES):
            bad = [dep for dep in unit.externals if not is_config_ingress_module(dep)]
            if bad:
                result.violations.append(
                    f"{unit.path}: externals may be declared only on effect_tool/effect_facility adapter units"
                )
        if (
            any(is_config_ingress_module(dep) for dep in unit.externals)
            and "assembly_root" not in roles
        ):
            result.violations.append(
                f"{unit.path}: persistent config ingress externals may be declared only on assembly_root"
            )
        if "entry_frame" in roles and not unit.axis and "assembly_root" not in roles:
            result.violations.append(
                f"{unit.path}: entry_frame must declare axis unless folded with assembly_root"
            )
        if "entry_frame" in roles and len(unit.axis) > 2:
            result.violations.append(
                f"{unit.path}: entry_frame may span at most two axes; split the front door or use axis_link, link_proto schema, or process boundaries"
            )
        if "use_flow" in roles and len(unit.axis) != 1:
            result.violations.append(
                f"{unit.path}: use_flow must declare exactly one axis; split cross-axis workflows and connect them through axis_link"
            )
        if (
            not unit.axis
            and not roles <= NO_AXIS_ROLES
            and roles != {"assembly_root", "entry_frame"}
        ):
            result.violations.append(
                f"{unit.path}: missing axis; only shared/default, assembly_root+entry_frame fold, signal/test side-path, or universal roles may omit it"
            )
        if len(unit.axis) > 1 and not roles <= MULTI_AXIS_ROLES:
            result.violations.append(
                f"{unit.path}: one or more roles cannot manually span multiple axes"
            )
        if "link_proto" in roles:
            if roles != {"link_proto"}:
                result.violations.append(
                    f"{unit.path}: link_proto must stay standalone"
                )
            if not unit.axis:
                result.violations.append(
                    f"{unit.path}: link_proto must declare the axis or axes whose public protocol/schema it describes"
                )
            if len(unit.axis) > 2:
                result.violations.append(
                    f"{unit.path}: link_proto should describe one cross-axis conversation; split schemas when more than two axes are involved"
                )
        if "signal_analyzer" in roles:
            if unit.axis:
                result.violations.append(
                    f"{unit.path}: signal_analyzer must not declare axis"
                )
            if roles != {"signal_analyzer"}:
                result.violations.append(
                    f"{unit.path}: signal_analyzer must stay standalone"
                )
        if "acceptance_test" in roles:
            if unit.axis:
                result.violations.append(
                    f"{unit.path}: acceptance_test must not declare axis; it is a side-path system test harness"
                )
            if roles != {"acceptance_test"}:
                result.violations.append(
                    f"{unit.path}: acceptance_test must stay standalone"
                )
    return result


def source_roots(root: Path, unit: Unit) -> list[Path]:
    if unit.sources:
        return [
            (root / src).resolve()
            if not Path(src).is_absolute()
            else Path(src).resolve()
            for src in unit.sources
        ]
    return [(root / unit.path).resolve()]


def source_files_for_unit(root: Path, unit: Unit) -> list[Path]:
    files: list[Path] = []
    for src in source_roots(root, unit):
        if src.is_file() and (
            src.suffix in SOURCE_EXTENSIONS or src.name == "Cargo.toml"
        ):
            files.append(src)
        elif src.is_dir():
            files.extend(
                path
                for path in src.rglob("*")
                if path.is_file()
                and (path.suffix in SOURCE_EXTENSIONS or path.name == "Cargo.toml")
            )
    return files


def _line_at(lines: Sequence[str], line_no: int) -> str:
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1].strip()
    return ""


def _strip_c_like_comments(text: str) -> str:
    """Remove comments while preserving line positions for import scanners."""
    out: list[str] = []
    i = 0
    in_block = False
    in_line = False
    quote = ""
    escape = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_line:
            if ch == "\n":
                in_line = False
                out.append(ch)
            else:
                out.append(" ")
            i += 1
            continue
        if in_block:
            if ch == "*" and nxt == "/":
                out.extend("  ")
                in_block = False
                i += 2
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue
        if quote:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = ""
            i += 1
            continue
        if ch in {"'", '"', "`"}:
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            out.extend("  ")
            in_line = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            out.extend("  ")
            in_block = True
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _find_quoted_values(text: str) -> list[str]:
    values: list[str] = []
    i = 0
    while i < len(text):
        if text[i] not in {"'", '"', "`"}:
            i += 1
            continue
        quote = text[i]
        i += 1
        start = i
        escaped = False
        buf: list[str] = []
        while i < len(text):
            ch = text[i]
            if escaped:
                buf.append(ch)
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                values.append("".join(buf))
                i += 1
                break
            else:
                buf.append(ch)
            i += 1
        else:
            i = start
    return values


def _python_qualified_name_matches(module: str, name: str) -> bool:
    module_parts = [part for part in module.split(".") if part]
    name_parts = [
        part for part in name.replace("/", ".").replace("\\", ".").split(".") if part
    ]
    if not module_parts or not name_parts or len(name_parts) > len(module_parts):
        return False
    for index in range(1, len(module_parts) - len(name_parts) + 1):
        if module_parts[index : index + len(name_parts)] == name_parts:
            return True
    return False


def _registered_name_for_module(
    module: str, names: Sequence[str], *, allow_python_qualified_match: bool = False
) -> str:
    module = module.strip()
    if not module or module.startswith("."):
        return ""
    for name in sorted(set(names), key=len, reverse=True):
        if module == name:
            return name
        for sep in ("::", ".", "/", "\\"):
            if module.startswith(name + sep):
                return name
        if allow_python_qualified_match and _python_qualified_name_matches(
            module, name
        ):
            return name
    return ""


def _python_import_candidates(
    source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    candidates: list[tuple[str, int, str]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return candidates
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                candidates.append(
                    (alias.name, node.lineno, _line_at(lines, node.lineno))
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module:
                candidates.append(
                    (node.module, node.lineno, _line_at(lines, node.lineno))
                )
    return candidates


def _rust_import_roots(payload: str) -> list[str]:
    payload = payload.strip().rstrip(";").strip()
    if not payload or payload.startswith(("crate::", "self::", "super::")):
        return []
    if payload.startswith("{") and payload.endswith("}"):
        inner = payload[1:-1]
        roots: list[str] = []
        depth = 0
        part: list[str] = []
        for ch in inner:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if ch == "," and depth == 0:
                roots.extend(_rust_import_roots("".join(part)))
                part = []
            else:
                part.append(ch)
        roots.extend(_rust_import_roots("".join(part)))
        return roots
    for token in ("::", "{", " as "):
        idx = payload.find(token)
        if idx >= 0:
            payload = payload[:idx]
    root = payload.strip()
    return [root] if root else []


def _rust_import_candidates(
    source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    clean = _strip_c_like_comments(source)
    candidates: list[tuple[str, int, str]] = []
    stmt: list[str] = []
    start_line = 1
    current_line = 1
    for ch in clean:
        if not stmt and not ch.isspace():
            start_line = current_line
        stmt.append(ch)
        if ch == ";":
            text = "".join(stmt).strip()
            stmt = []
            use_index = -1
            if text.startswith("use "):
                use_index = 4
            elif text.startswith("pub ") or text.startswith("pub("):
                marker = " use "
                found = text.find(marker)
                if found >= 0:
                    use_index = found + len(marker)
            if use_index >= 0:
                for root in _rust_import_roots(text[use_index:]):
                    candidates.append((root, start_line, _line_at(lines, start_line)))
            elif text.startswith("extern crate "):
                payload = text[len("extern crate ") :].rstrip(";").strip()
                payload = payload.split(" as ", 1)[0].strip()
                if payload:
                    candidates.append(
                        (payload, start_line, _line_at(lines, start_line))
                    )
        if ch == "\n":
            current_line += 1
    return candidates


def _js_import_candidates(
    source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    clean = _strip_c_like_comments(source)
    candidates: list[tuple[str, int, str]] = []
    stmt: list[str] = []
    start_line = 1
    current_line = 1

    def flush_statement(text: str, line_no: int) -> None:
        text = text.strip()
        values = _find_quoted_values(text)
        if not values:
            return
        if text.startswith("import ") or text.startswith("export "):
            if text.startswith("import ") and " from " not in text and len(values) == 1:
                candidates.append((values[0], line_no, _line_at(lines, line_no)))
            elif " from " in text:
                candidates.append((values[-1], line_no, _line_at(lines, line_no)))
        elif (
            text.startswith(("require(", "const ", "let ", "var "))
            and "require(" in text
        ) or "import(" in text:
            candidates.append((values[0], line_no, _line_at(lines, line_no)))

    for ch in clean:
        if not stmt and not ch.isspace():
            start_line = current_line
        stmt.append(ch)
        text = "".join(stmt).strip()
        is_static_import = text.startswith("import ") or text.startswith("export ")
        values = _find_quoted_values(text)
        complete_static_import = is_static_import and (
            (" from " in text and bool(values))
            or (
                text.startswith("import ")
                and " from " not in text
                and bool(values)
                and (ch in {";", "\n"})
            )
        )
        end_statement = (
            ch == ";" or complete_static_import or (ch == "\n" and not is_static_import)
        )
        if end_statement:
            flush_statement(text, start_line)
            stmt = []
        if ch == "\n":
            current_line += 1
    if stmt:
        flush_statement("".join(stmt), start_line)
    return candidates


def _go_import_candidates(
    source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    clean = _strip_c_like_comments(source)
    candidates: list[tuple[str, int, str]] = []
    in_block = False
    for line_no, line in enumerate(clean.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("import ("):
            in_block = True
            continue
        if in_block:
            if stripped.startswith(")"):
                in_block = False
                continue
            values = _find_quoted_values(stripped)
            if values:
                candidates.append((values[0], line_no, _line_at(lines, line_no)))
            continue
        if stripped.startswith("import "):
            values = _find_quoted_values(stripped)
            if values:
                candidates.append((values[0], line_no, _line_at(lines, line_no)))
    return candidates


def _simple_keyword_import_candidates(
    source: str, lines: Sequence[str], keywords: Sequence[str]
) -> list[tuple[str, int, str]]:
    clean = _strip_c_like_comments(source)
    candidates: list[tuple[str, int, str]] = []
    for line_no, line in enumerate(clean.splitlines(), start=1):
        stripped = line.strip().rstrip(";")
        for keyword in keywords:
            prefix = keyword + " "
            if not stripped.startswith(prefix):
                continue
            payload = stripped[len(prefix) :].strip()
            if not payload or payload.startswith("static "):
                continue
            if keyword == "using" and "=" in payload:
                payload = payload.split("=", 1)[1].strip()
            module = payload.split()[0].strip().strip(";")
            if module.endswith(".*"):
                module = module[:-2]
            if module:
                candidates.append((module, line_no, _line_at(lines, line_no)))
    return candidates


def _c_include_candidates(
    source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    clean = _strip_c_like_comments(source)
    candidates: list[tuple[str, int, str]] = []
    for line_no, line in enumerate(clean.splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("#include"):
            continue
        values = _find_quoted_values(stripped)
        if values:
            candidates.append((values[0], line_no, _line_at(lines, line_no)))
            continue
        if "<" in stripped and ">" in stripped:
            start = stripped.find("<") + 1
            end = stripped.find(">", start)
            if end > start:
                candidates.append(
                    (stripped[start:end], line_no, _line_at(lines, line_no))
                )
    return candidates


TREE_SITTER_LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".rs": "rust",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cpp": "cpp",
    ".cs": "c_sharp",
}

_TREE_SITTER_PARSER_CACHE: dict[str, Any | None] = {}


def _new_tree_sitter_parser(language: Any) -> Any:
    from tree_sitter import Language, Parser  # type: ignore

    if not isinstance(language, Language):
        with suppress(TypeError):
            language = Language(language)
    parser = Parser()
    if hasattr(parser, "set_language"):
        parser.set_language(language)
    else:
        parser.language = language
    return parser


def _tree_sitter_parser_for_language(language_name: str) -> Any | None:
    if language_name in _TREE_SITTER_PARSER_CACHE:
        return _TREE_SITTER_PARSER_CACHE[language_name]

    parser: Any | None = None
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore

        parser = get_parser(language_name)
    except Exception:
        parser = None

    if parser is None:
        try:
            from tree_sitter_languages import get_parser  # type: ignore

            parser = get_parser(language_name)
        except Exception:
            parser = None

    if parser is None:
        module_names = {
            "rust": "tree_sitter_rust",
            "javascript": "tree_sitter_javascript",
            "typescript": "tree_sitter_typescript",
            "tsx": "tree_sitter_typescript",
            "go": "tree_sitter_go",
            "java": "tree_sitter_java",
            "kotlin": "tree_sitter_kotlin",
            "swift": "tree_sitter_swift",
            "c": "tree_sitter_c",
            "cpp": "tree_sitter_cpp",
            "c_sharp": "tree_sitter_c_sharp",
        }
        module_name = module_names.get(language_name)
        if module_name:
            try:
                module = importlib.import_module(module_name)
                language_func = module.language
                language = language_func()
                parser = _new_tree_sitter_parser(language)
            except Exception:
                parser = None

    _TREE_SITTER_PARSER_CACHE[language_name] = parser
    return parser


def _tree_sitter_parser_for_suffix(suffix: str) -> Any | None:
    language_name = TREE_SITTER_LANGUAGE_BY_SUFFIX.get(suffix)
    if not language_name:
        return None
    return _tree_sitter_parser_for_language(language_name)


def _tree_sitter_text(source_bytes: bytes, node: Any) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode(
        "utf-8", errors="ignore"
    )


def _tree_sitter_line(node: Any) -> int:
    return int(node.start_point[0]) + 1


def _walk_tree_sitter(node: Any) -> list[Any]:
    out: list[Any] = []
    stack = [node]
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(reversed(list(current.children)))
    return out


def _shift_candidates(
    candidates: list[tuple[str, int, str]], start_line: int, full_lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    shifted: list[tuple[str, int, str]] = []
    for module, relative_line, _ in candidates:
        line_no = start_line + max(relative_line, 1) - 1
        shifted.append((module, line_no, _line_at(full_lines, line_no)))
    return shifted


def _tree_sitter_import_candidates(
    file_path: Path, source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]] | None:
    suffix = file_path.suffix
    parser = _tree_sitter_parser_for_suffix(suffix)
    if parser is None:
        return None
    source_bytes = source.encode("utf-8")
    try:
        tree = parser.parse(source_bytes)
    except Exception:
        return None

    candidates: list[tuple[str, int, str]] = []
    for node in _walk_tree_sitter(tree.root_node):
        node_type = node.type
        text = _tree_sitter_text(source_bytes, node)
        start_line = _tree_sitter_line(node)

        if suffix == ".rs" and node_type in {
            "use_declaration",
            "extern_crate_declaration",
        }:
            candidates.extend(
                _shift_candidates(
                    _rust_import_candidates(text, text.splitlines()), start_line, lines
                )
            )
        elif suffix in {".js", ".jsx", ".ts", ".tsx"}:
            if node_type in {"import_statement", "export_statement"} or (
                node_type == "call_expression"
                and (
                    text.lstrip().startswith("require(")
                    or text.lstrip().startswith("import(")
                )
            ):
                candidates.extend(
                    _shift_candidates(
                        _js_import_candidates(text, text.splitlines()),
                        start_line,
                        lines,
                    )
                )
        elif suffix == ".go" and node_type in {"import_declaration", "import_spec"}:
            candidates.extend(
                _shift_candidates(
                    _go_import_candidates(text, text.splitlines()), start_line, lines
                )
            )
        elif suffix in {".java", ".kt", ".swift"} and node_type in {
            "import_declaration",
            "import_header",
            "import_statement",
        }:
            candidates.extend(
                _shift_candidates(
                    _simple_keyword_import_candidates(
                        text, text.splitlines(), ("import",)
                    ),
                    start_line,
                    lines,
                )
            )
        elif suffix == ".cs" and node_type in {"using_directive", "using_statement"}:
            candidates.extend(
                _shift_candidates(
                    _simple_keyword_import_candidates(
                        text, text.splitlines(), ("using",)
                    ),
                    start_line,
                    lines,
                )
            )
        elif suffix in {".c", ".h", ".hpp", ".cpp"} and node_type == "preproc_include":
            candidates.extend(
                _shift_candidates(
                    _c_include_candidates(text, text.splitlines()), start_line, lines
                )
            )

    seen: set[tuple[str, int]] = set()
    deduped: list[tuple[str, int, str]] = []
    for module, line_no, line in candidates:
        key = (module, line_no)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((module, line_no, line))
    return deduped


def _fallback_import_candidates_for_file(
    file_path: Path, source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    suffix = file_path.suffix
    if suffix == ".rs":
        return _rust_import_candidates(source, lines)
    if suffix in {".js", ".jsx", ".ts", ".tsx"}:
        return _js_import_candidates(source, lines)
    if suffix == ".go":
        return _go_import_candidates(source, lines)
    if suffix in {".java", ".kt"}:
        return _simple_keyword_import_candidates(source, lines, ("import",))
    if suffix == ".swift":
        return _simple_keyword_import_candidates(source, lines, ("import",))
    if suffix == ".cs":
        return _simple_keyword_import_candidates(source, lines, ("using",))
    if suffix in {".c", ".h", ".hpp", ".cpp"}:
        return _c_include_candidates(source, lines)
    return []


def _cargo_toml_dependency_candidates(
    source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    candidates: list[tuple[str, int, str]] = []
    in_dep_section = False
    dep_section_pattern = re.compile(
        r"^\[(?:target\.[^]]+\.)?(?:dev-|build-)?dependencies(?:\.([A-Za-z0-9_-]+))?\]"
    )
    key_pattern = re.compile(r"^([A-Za-z0-9_-]+)\s*=")
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            match = dep_section_pattern.match(line)
            in_dep_section = bool(match)
            if match and match.group(1):
                candidates.append((match.group(1), line_no, _line_at(lines, line_no)))
            continue
        if in_dep_section:
            match = key_pattern.match(line)
            if match:
                candidates.append((match.group(1), line_no, _line_at(lines, line_no)))
    return candidates


def import_candidates_for_file(
    file_path: Path, source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    if file_path.name == "Cargo.toml":
        return _cargo_toml_dependency_candidates(source, lines)
    suffix = file_path.suffix
    if suffix == ".py":
        return _python_import_candidates(source, lines)
    if suffix in TREE_SITTER_LANGUAGE_BY_SUFFIX:
        tree_sitter_candidates = _tree_sitter_import_candidates(
            file_path, source, lines
        )
        if tree_sitter_candidates is not None:
            return tree_sitter_candidates
        return _fallback_import_candidates_for_file(file_path, source, lines)
    return []


def missing_tree_sitter_suffixes(root: Path, units: Sequence[Unit]) -> set[str]:
    missing: set[str] = set()
    for unit in units:
        for file_path in source_files_for_unit(root, unit):
            if file_path.name == "Cargo.toml":
                continue
            suffix = file_path.suffix
            if (
                suffix in TREE_SITTER_LANGUAGE_BY_SUFFIX
                and _tree_sitter_parser_for_suffix(suffix) is None
            ):
                missing.add(suffix)
    return missing


def scan_imports(root: Path, units: Sequence[Unit]) -> dict[str, list[ImportRef]]:
    registered_names = tuple(name for unit in units for name in unit.names)
    refs_by_unit = {unit.path: [] for unit in units}
    if not registered_names:
        return refs_by_unit
    for unit in units:
        own_names = set(unit.names)
        for file_path in source_files_for_unit(root, unit):
            try:
                source = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            lines = source.splitlines()
            rel = file_path.resolve().relative_to(root.resolve()).as_posix()
            for module_name, line_no, line in import_candidates_for_file(
                file_path, source, lines
            ):
                target = _registered_name_for_module(
                    module_name,
                    registered_names,
                    allow_python_qualified_match=file_path.suffix == ".py",
                )
                if target and target not in own_names:
                    refs_by_unit[unit.path].append(
                        ImportRef(rel, target, line_no, line)
                    )
    return refs_by_unit


def scan_rust_external_refs(
    root: Path, units: Sequence[Unit]
) -> dict[str, list[ImportRef]]:
    registered_names = tuple(name for unit in units for name in unit.names)
    refs_by_unit = {unit.path: [] for unit in units}
    for unit in units:
        own_names = set(unit.names)
        for file_path in source_files_for_unit(root, unit):
            if file_path.suffix != ".rs" and file_path.name != "Cargo.toml":
                continue
            try:
                source = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            lines = source.splitlines()
            rel = file_path.resolve().relative_to(root.resolve()).as_posix()
            for module_name, line_no, line in import_candidates_for_file(
                file_path, source, lines
            ):
                key = rust_crate_key(module_name)
                if not key or key in RUST_DEFAULT_DIRECT_CRATES:
                    continue
                if (
                    _registered_name_for_module(module_name, registered_names)
                    in own_names
                ):
                    continue
                if _registered_name_for_module(module_name, registered_names):
                    continue
                refs_by_unit[unit.path].append(ImportRef(rel, key, line_no, line))
    return refs_by_unit


def is_config_ingress_module(module_name: str) -> bool:
    lower = module_name.strip().strip("\"'`").lower().replace("-", "_")
    if lower in CONFIG_INGRESS_EXACT:
        return True
    rust_key = rust_crate_key(lower)
    if rust_key in CONFIG_INGRESS_EXACT:
        return True
    return any(lower.startswith(prefix) for prefix in CONFIG_INGRESS_PREFIXES)


def _python_config_usage_candidates(
    source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    candidates: list[tuple[str, int, str]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return candidates
    os_aliases: set[str] = set()
    os_env_aliases: dict[str, str] = {}
    env_names = {"environ", "getenv"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    os_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name in env_names:
                    local_name = alias.asname or alias.name
                    os_env_aliases[local_name] = alias.name
                    candidates.append(
                        (f"os.{alias.name}", node.lineno, _line_at(lines, node.lineno))
                    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in os_aliases and node.attr in env_names:
                candidates.append(
                    (f"os.{node.attr}", node.lineno, _line_at(lines, node.lineno))
                )
            elif node.attr in CONFIG_LOADER_FUNCTION_NAMES or (
                node.attr == "load" and "config" in node.value.id.lower()
            ):
                candidates.append(
                    (
                        f"{node.value.id}.{node.attr}",
                        node.lineno,
                        _line_at(lines, node.lineno),
                    )
                )
        elif isinstance(node, ast.Name) and node.id in os_env_aliases:
            candidates.append(
                (
                    f"os.{os_env_aliases[node.id]}",
                    node.lineno,
                    _line_at(lines, node.lineno),
                )
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in CONFIG_LOADER_FUNCTION_NAMES
        ):
            candidates.append((node.func.id, node.lineno, _line_at(lines, node.lineno)))
    return candidates


def _text_config_usage_candidates(
    file_path: Path, source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    clean = _strip_c_like_comments(source)
    suffix = file_path.suffix
    candidates: list[tuple[str, int, str]] = []
    rust_env_alias = bool(re.search(r"\buse\s+std::env\b|\buse\s+std::env::", clean))
    patterns_by_suffix: dict[str, tuple[tuple[str, str], ...]] = {
        ".rs": (
            (
                r"\bstd::env::(?:var|var_os|vars|vars_os|set_var|remove_var)\b",
                "std::env",
            ),
            (r"\benv::(?:var|var_os|vars|vars_os|set_var|remove_var)\b", "std::env"),
        ),
        ".go": (
            (
                r"\bos\.(?:Getenv|LookupEnv|Environ|Setenv|Unsetenv|Clearenv)\s*\(",
                "os.env",
            ),
        ),
        ".js": ((r"\bprocess\.env\b", "process.env"),),
        ".jsx": ((r"\bprocess\.env\b", "process.env"),),
        ".ts": ((r"\bprocess\.env\b", "process.env"),),
        ".tsx": ((r"\bprocess\.env\b", "process.env"),),
        ".java": ((r"\bSystem\.getenv\s*\(", "System.getenv"),),
        ".kt": ((r"\bSystem\.getenv\s*\(", "System.getenv"),),
        ".swift": (
            (r"\bProcessInfo\.processInfo\.environment\b", "ProcessInfo.environment"),
        ),
        ".cs": (
            (
                r"\bEnvironment\.GetEnvironmentVariable\s*\(",
                "Environment.GetEnvironmentVariable",
            ),
        ),
        ".c": ((r"\bgetenv\s*\(", "getenv"),),
        ".h": ((r"\bgetenv\s*\(", "getenv"),),
        ".hpp": ((r"\bgetenv\s*\(", "getenv"),),
        ".cpp": ((r"\bgetenv\s*\(", "getenv"),),
    }
    loader_patterns = (
        (r"\bconfig::load\s*\(", "config::load"),
        (r"\bconfig\.load\s*\(", "config.load"),
        (r"\bload_config\s*\(", "load_config"),
        (r"\bload_runtime_config\s*\(", "load_runtime_config"),
        (
            r"\bprepare_runtime_config_mapping\s*\(",
            "prepare_runtime_config_mapping",
        ),
    )
    for line_no, line in enumerate(clean.splitlines(), start=1):
        for pattern, target in patterns_by_suffix.get(suffix, ()):
            if (
                target == "std::env"
                and pattern.startswith(r"\benv::")
                and not rust_env_alias
            ):
                continue
            if re.search(pattern, line):
                candidates.append((target, line_no, _line_at(lines, line_no)))
        for pattern, target in loader_patterns:
            if re.search(pattern, line):
                candidates.append((target, line_no, _line_at(lines, line_no)))
    return candidates


def config_usage_candidates_for_file(
    file_path: Path, source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    if file_path.suffix == ".py":
        return _python_config_usage_candidates(source, lines)
    return _text_config_usage_candidates(file_path, source, lines)


def scan_config_ingress_refs(
    root: Path, units: Sequence[Unit]
) -> dict[str, list[ImportRef]]:
    refs_by_unit = {unit.path: [] for unit in units}
    for unit in units:
        for file_path in source_files_for_unit(root, unit):
            try:
                source = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            lines = source.splitlines()
            rel = file_path.resolve().relative_to(root.resolve()).as_posix()
            seen: set[tuple[str, int]] = set()
            for module_name, line_no, line in import_candidates_for_file(
                file_path, source, lines
            ):
                if is_config_ingress_module(module_name):
                    key = (module_name, line_no)
                    if key not in seen:
                        seen.add(key)
                        refs_by_unit[unit.path].append(
                            ImportRef(rel, module_name, line_no, line)
                        )
            for module_name, line_no, line in config_usage_candidates_for_file(
                file_path, source, lines
            ):
                key = (module_name, line_no)
                if key not in seen:
                    seen.add(key)
                    refs_by_unit[unit.path].append(
                        ImportRef(rel, module_name, line_no, line)
                    )
    return refs_by_unit


def check_config_ingress_refs(
    root: Path, units: Sequence[Unit], result: CheckResult
) -> None:
    units_by_path = {unit.path: unit for unit in units}
    for src_path, refs in scan_config_ingress_refs(root, units).items():
        src = units_by_path[src_path]
        if "assembly_root" in src.roles:
            continue
        for ref in refs:
            result.violations.append(
                f"{ref.source_path}:{ref.line_no}: {src.path} ({'+'.join(src.roles)}) must not use persistent config ingress package/API {ref.target_name!r}; "
                "only assembly_root may receive external config, and typed config must move layer by layer"
            )


def check_rust_external_refs(
    root: Path, units: Sequence[Unit], result: CheckResult
) -> None:
    owners = rust_external_owners(units)
    units_by_path = {unit.path: unit for unit in units}
    for src_path, refs in scan_rust_external_refs(root, units).items():
        src = units_by_path[src_path]
        for ref in refs:
            if is_config_ingress_module(ref.target_name):
                continue
            owner = owners.get(ref.target_name)
            if owner is None:
                result.violations.append(
                    f"{ref.source_path}:{ref.line_no}: {src.path} uses unregistered Rust external crate {ref.target_name!r}; "
                    "only serde, thiserror, anyhow, tokio, tracing, and tempfile are globally direct, otherwise bind the crate to an effect_tool/effect_facility via externals"
                )
                continue
            if owner.path != src.path:
                result.violations.append(
                    f"{ref.source_path}:{ref.line_no}: {src.path} must not import Rust external crate {ref.target_name!r}; "
                    f"it is bound to {owner.path}, import that adapter/facility instead"
                )


def check_imports(root: Path, data: Mapping[str, Any] | None = None) -> CheckResult:
    result = check_register(root, data)
    if result.violations:
        return result
    raw = load_register(root) if data is None else data
    units = load_units(root, raw)
    units_by_path = {unit.path: unit for unit in units}
    for unit in units:
        for missing in missing_source_roots(root, unit):
            result.violations.append(
                f"{unit.path}: source root does not exist: {missing}"
            )
    if result.violations:
        return result
    for file_path, owners in sorted(overlapping_source_files(root, units).items()):
        result.violations.append(
            f"{file_path}: source file is registered by multiple FCIS units via path/sources: {', '.join(owners)}"
        )
    if result.violations:
        return result
    missing_suffixes = missing_tree_sitter_suffixes(root, units)
    if missing_suffixes:
        result.notes.append(
            "Tree-sitter parser unavailable for "
            + ", ".join(sorted(missing_suffixes))
            + "; used narrow declaration-scanner fallback for those files"
        )
    check_config_ingress_refs(root, units, result)
    check_rust_external_refs(root, units, result)
    name_to_unit = {name: unit for unit in units for name in unit.names}
    for src_path, refs in scan_imports(root, units).items():
        src = units_by_path[src_path]
        for ref in refs:
            dst = name_to_unit.get(ref.target_name)
            if dst is None or dst.path == src.path:
                continue
            if not role_dependency_allowed(src, dst, units_by_path):
                result.violations.append(
                    f"{ref.source_path}:{ref.line_no}: {src.path} ({'+'.join(src.roles)}) forbidden import "
                    f"{dst.path} ({'+'.join(dst.roles)}) via {ref.target_name}: {ref.line}"
                )
    return result


def print_result(result: CheckResult) -> int:
    if result.violations:
        for violation in result.violations:
            print(f"FAIL: {violation}")
        return 1
    print("PASS: FCIS auto-derived role graph checks passed")
    for note in result.notes:
        print(f"NOTE: {note}")
    return 0
