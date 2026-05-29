#!/usr/bin/env python3
"""Shared FCIS role-graph engine.

This module is the single source of checker truth for the skill. A project
registers only physical units, their roles, axes, optional binding targets, and
optional source/import names in `.fcis/role_register.json`. All dependency and
fold decisions are auto-derived from the theoretical role graph below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

CANONICAL_ROLES: dict[str, str] = {
    "base_atoms": "generic primitive base",
    "base_kit": "generic helper APIs",
    "meaning_seed": "axis-capable smallest shared business vocabulary",
    "meaning_core": "pure business meaning and deterministic rules",
    "run_kit": "axis-capable generic runtime helpers and signal envelopes",
    "axis_surface": "narrow same-axis public semantic/capability surface",
    "axis_link": "semantic API between horizontal axes",
    "axis_joint": "broad same-axis mental-model crossroad",
    "shape_seam": "app-owned shape seam between business/app data and external entry or generic capability APIs",
    "use_flow": "business workflow brain",
    "effect_tool": "generic reusable adapter/tool library with public porcelain API and concrete implementation",
    "effect_facility": "long-lived generic effect owner with runtime state",
    "wire_path": "side-path transport/schema realization for a bound axis_link",
    "entry_frame": "worker/serving entry: wires use_flow and effect_facility, receives request/job, and shapes response/exit",
    "assembly_root": "resource construction and lifecycle wiring",
    "config_binder": "assembly-root config ingress and contract binding side path",
    "signal_analyzer": "passive checker/aggregator over emitted signal records",
    "acceptance_test": "side-path end-to-end/probe test harness that may import production roles",
}

# Direct theoretical compile edges. Wrapper roles are derived separately from
# this graph: if A -> B is marked wrappable, A may instead depend on a
# shape_seam bound to B, producing A -> shape_seam -> B. If no explicit seam is
# registered, B's public API carries the shape boundary.
BASE_EDGES: dict[str, set[str]] = {
    "base_atoms": set(),
    "base_kit": {"base_atoms"},
    "meaning_seed": {"base_atoms"},
    "run_kit": {"base_kit", "base_atoms"},
    "meaning_core": {"meaning_seed", "base_atoms"},
    "axis_surface": {"meaning_seed", "base_atoms"},
    "axis_link": {"axis_surface", "meaning_seed"},
    "axis_joint": {"meaning_core", "axis_surface", "effect_tool"},
    "use_flow": {"meaning_core", "meaning_seed", "run_kit", "axis_surface", "axis_link", "effect_tool"},
    "entry_frame": {"use_flow", "effect_facility", "run_kit", "base_kit", "base_atoms"},
    "assembly_root": {"entry_frame", "effect_facility", "run_kit"},
    "effect_tool": {"run_kit", "base_kit", "base_atoms"},
    "effect_facility": {"effect_tool", "run_kit", "base_kit", "base_atoms"},
    "wire_path": {"axis_link", "base_kit", "base_atoms", "run_kit"},
    "shape_seam": {"meaning_seed", "base_kit", "base_atoms", "run_kit"},
    "config_binder": {"base_kit", "base_atoms", "run_kit"},
    "signal_analyzer": {"run_kit", "base_kit", "base_atoms"},
    "acceptance_test": set(),
}

WRAPPABLE_EDGES: set[tuple[str, str]] = {
    ("entry_frame", "use_flow"),
    ("use_flow", "effect_tool"),
}

# Fold rules are intentionally explicit. Do not infer folds from graph reachability:
# a compile edge may permit an import without permitting mixed ownership.
# Protected roles stay standalone even when another role may import them.
ALLOWED_ROLE_FOLDS: set[frozenset[str]] = {
    frozenset({"base_atoms", "base_kit"}),
    frozenset({"assembly_root", "entry_frame"}),
    frozenset({"use_flow", "axis_joint"}),
}

SHAPE_SEAM_FOLD_TARGETS = {"entry_frame", "use_flow"}

PROTECTED_STANDALONE_ROLES = {
    "meaning_seed",
    "run_kit",
    "effect_tool",
    "effect_facility",
    "axis_link",
    "wire_path",
    "config_binder",
    "signal_analyzer",
    "acceptance_test",
}

UNIVERSAL_ROLES = {"base_atoms", "base_kit"}
SHARED_DEFAULT_AXIS_ROLES = {"meaning_seed", "run_kit", "assembly_root"}
NO_AXIS_ROLES = UNIVERSAL_ROLES | SHARED_DEFAULT_AXIS_ROLES | {"wire_path", "signal_analyzer", "config_binder", "acceptance_test"}
MULTI_AXIS_ROLES = {"meaning_seed", "run_kit", "entry_frame", "assembly_root", "axis_link", "effect_tool", "effect_facility"}
WRAPPER_ROLES = {"shape_seam"}
SIDE_PATH_ROLES = {"wire_path", "config_binder", "signal_analyzer", "acceptance_test"}

SOURCE_EXTENSIONS = {".rs", ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".kt", ".swift", ".c", ".h", ".hpp", ".cpp", ".cs"}


@dataclass(frozen=True)
class Unit:
    path: str
    roles: tuple[str, ...]
    axis: tuple[str, ...] = ()
    bind_to: str = ""
    names: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


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
    bind_to = str(raw.get("bind_to", "") or raw.get("bind", "")).strip()
    raw_names = raw.get("names", raw.get("package_names", ()))
    names = tuple(str(name).strip() for name in raw_names if str(name).strip()) if isinstance(raw_names, list) else ()
    if not names:
        names = default_names(path)
    raw_sources = raw.get("sources", raw.get("source_roots", ()))
    sources = tuple(str(src).strip() for src in raw_sources if str(src).strip()) if isinstance(raw_sources, list) else ()
    return Unit(path=path, roles=roles, axis=axis, bind_to=bind_to, names=names, sources=sources)


def load_register(root: Path) -> dict[str, Any]:
    path = root / ".fcis" / "role_register.json"
    if not path.exists():
        return {"schema_version": 2, "units": []}
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
    if unit.axis:
        return unit.axis
    if ("shape_seam" in unit.roles or "wire_path" in unit.roles) and unit.bind_to and unit.bind_to in units_by_path:
        return effective_axis(units_by_path[unit.bind_to], units_by_path)
    return ()


def bound_target_roles(unit: Unit, units_by_path: Mapping[str, Unit]) -> set[str]:
    if unit.bind_to and unit.bind_to in units_by_path:
        return set(units_by_path[unit.bind_to].roles)
    return set()


def wrappable_target_roles() -> set[str]:
    return {target for _, target in WRAPPABLE_EDGES}


def shape_seam_target_valid(unit: Unit, units_by_path: Mapping[str, Unit]) -> bool:
    roles = set(unit.roles)
    if "shape_seam" not in roles:
        return True
    if len(roles) > 1:
        targets = roles - {"shape_seam"}
        return targets <= {"entry_frame", "use_flow"}
    return bool(bound_target_roles(unit, units_by_path) & wrappable_target_roles())


def duplicate_names(units: Sequence[Unit]) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {}
    for unit in units:
        for name in unit.names:
            seen.setdefault(name, []).append(unit.path)
    return {name: paths for name, paths in seen.items() if len(set(paths)) > 1}


def missing_source_roots(root: Path, unit: Unit) -> list[str]:
    missing: list[str] = []
    for src in source_roots(root, unit):
        if not src.exists():
            try:
                missing.append(src.relative_to(root.resolve()).as_posix())
            except ValueError:
                missing.append(str(src))
    return missing


def axis_visible(src: Unit, dst: Unit, units_by_path: Mapping[str, Unit]) -> bool:
    if is_universal(dst):
        return True
    dst_roles = set(dst.roles)
    if dst_roles & {"meaning_seed", "run_kit"} and not effective_axis(dst, units_by_path):
        return True
    if dst_roles & {"axis_link"}:
        return True
    if set(src.roles) & {"assembly_root"}:
        return True
    src_axis = effective_axis(src, units_by_path)
    dst_axis = effective_axis(dst, units_by_path)
    if not src_axis:
        return not dst_axis or bool(dst_roles & SHARED_DEFAULT_AXIS_ROLES)
    if not dst_axis:
        return bool(dst_roles & (UNIVERSAL_ROLES | SHARED_DEFAULT_AXIS_ROLES | SIDE_PATH_ROLES))
    return axis_overlaps(src_axis, dst_axis)


def role_direct_allowed(src_role: str, dst_role: str) -> bool:
    return dst_role in role_edges(src_role)


def unit_has_bound_role(lens: Unit, role: str, units_by_path: Mapping[str, Unit]) -> bool:
    if "shape_seam" not in lens.roles:
        return False
    folded_targets = [r for r in lens.roles if r != "shape_seam"]
    if folded_targets:
        return role in folded_targets
    if lens.bind_to and lens.bind_to in units_by_path:
        return role in units_by_path[lens.bind_to].roles
    return False


def wrapper_allowed(src_role: str, lens: Unit, units_by_path: Mapping[str, Unit]) -> bool:
    if "shape_seam" not in lens.roles:
        return False
    return any((src_role, target_role) in WRAPPABLE_EDGES and unit_has_bound_role(lens, target_role, units_by_path) for target_role in BASE_EDGES)


def role_dependency_allowed(src: Unit, dst: Unit, units_by_path: Mapping[str, Unit]) -> bool:
    if src.path == dst.path:
        return True

    # Acceptance tests are intentionally outside the production graph. They may
    # import any registered production unit, but no production role has a base
    # edge back to acceptance_test.
    if set(src.roles) == {"acceptance_test"}:
        return True

    # Standalone wrapper units may depend on generic helpers and may import their
    # bound target only when that target is a valid graph-derived wrapper target.
    if set(src.roles) == {"shape_seam"} and src.bind_to == dst.path:
        return shape_seam_target_valid(src, units_by_path) and axis_visible(src, dst, units_by_path)

    # wire_path is a side-path transport/schema realization bound to one axis_link.
    # It may import only its bound axis_link plus generic base/run helpers.
    if "wire_path" in src.roles and "axis_link" in dst.roles and src.bind_to != dst.path:
        return False

    # config_binder is assembly-root-owned. It is not a normal dependency target
    # by role; assembly_root may import only the binder explicitly bound to it.
    if "config_binder" in dst.roles:
        return set(dst.roles) == {"config_binder"} and dst.bind_to == src.path and "assembly_root" in src.roles

    if "shape_seam" in dst.roles and not shape_seam_target_valid(dst, units_by_path):
        return False

    if not axis_visible(src, dst, units_by_path):
        return False
    for src_role in src.roles:
        for dst_role in dst.roles:
            if role_direct_allowed(src_role, dst_role):
                return True
        if wrapper_allowed(src_role, dst, units_by_path):
            return True
    return False


def fold_allowed(unit: Unit, units_by_path: Mapping[str, Unit]) -> bool:
    roles = set(unit.roles)
    if len(roles) <= 1:
        return True

    # No broad inferred folds. A role_fold is a physical ownership claim, not
    # a shortcut for any legal import edge. Keep folds small and explicit.
    if len(roles) > 2:
        return False

    # These roles are protected vocabulary/runtime/generic/side-path roles and
    # must remain standalone. They may be imported through allowed edges, but
    # they must not be merged into an app/business/entry owner.
    if roles & PROTECTED_STANDALONE_ROLES:
        return False

    # shape_seam is not a semantic layer; it may colocate only with app-side
    # entry/use code. It must never fold into adapters, facilities, surfaces,
    # seeds, run kits, or side paths.
    if "shape_seam" in roles:
        return (roles - {"shape_seam"}) <= SHAPE_SEAM_FOLD_TARGETS

    # use_flow is the business brain. It stays separate except for the
    # conceptual axis_joint fold used to keep the same-axis mental crossroad
    # executable without creating a giant surface package.
    if "use_flow" in roles:
        return roles == {"use_flow", "axis_joint"}

    return frozenset(roles) in ALLOWED_ROLE_FOLDS


def check_register(root: Path, data: Mapping[str, Any] | None = None) -> CheckResult:
    result = CheckResult()
    raw = load_register(root) if data is None else data
    banned_top = {"layers", "edges", "role_folds", "direct_lanes", "extra_allowed", "forbid", "forbidden", "import_bans"}
    for key in sorted(banned_top & set(raw)):
        result.violations.append(f"role_register.json must be declarative units only; remove top-level {key}")
    raw_units = list(raw.get("units", []))
    allowed_unit_keys = {"path", "roles", "axis", "bind_to", "bind", "names", "package_names", "sources", "source_roots"}
    for raw_unit in raw_units:
        extra_keys = sorted(set(raw_unit) - allowed_unit_keys)
        if extra_keys:
            result.violations.append(f"{raw_unit.get('path', '<missing path>')}: unsupported unit field(s): {', '.join(extra_keys)}")
    units = load_units(root, raw)
    units_by_path = {unit.path: unit for unit in units}
    if len(units_by_path) != len(units):
        result.violations.append("duplicate unit path in role_register.json")
    for name, paths in sorted(duplicate_names(units).items()):
        result.violations.append(f"duplicate import name {name!r} used by: {', '.join(sorted(set(paths)))}")
    for unit in units:
        if not unit.path:
            result.violations.append("unit missing path")
        if not unit.roles:
            result.violations.append(f"{unit.path}: unit missing roles")
        unknown = [role for role in unit.roles if role not in CANONICAL_ROLES]
        if unknown:
            result.violations.append(f"{unit.path}: unknown role(s): {', '.join(unknown)}")
            continue
        if not fold_allowed(unit, units_by_path):
            result.violations.append(f"{unit.path}: illegal role fold {'+'.join(unit.roles)}")
        roles = set(unit.roles)
        if roles & UNIVERSAL_ROLES:
            if unit.axis:
                result.violations.append(f"{unit.path}: base_atoms/base_kit must not declare axis")
            if not roles <= UNIVERSAL_ROLES:
                result.violations.append(f"{unit.path}: base_atoms/base_kit may fold only with each other")
        if "entry_frame" in roles and not unit.axis and "assembly_root" not in roles:
            result.violations.append(f"{unit.path}: entry_frame must declare axis unless folded with assembly_root")
        if not unit.axis and not roles <= NO_AXIS_ROLES and not ("shape_seam" in roles and len(roles) == 1):
            result.violations.append(f"{unit.path}: missing axis; only shared/default, side-path, or standalone shape_seam roles may omit it")
        if len(unit.axis) > 1 and not roles <= (MULTI_AXIS_ROLES | {"shape_seam"}):
            result.violations.append(f"{unit.path}: one or more roles cannot manually span multiple axes")
        if "shape_seam" in roles and len(roles) == 1:
            if unit.axis:
                result.violations.append(f"{unit.path}: standalone shape_seam derives axis from bind_to and must not declare axis")
            if not unit.bind_to:
                result.violations.append(f"{unit.path}: standalone shape_seam requires bind_to")
            elif unit.bind_to not in units_by_path:
                result.violations.append(f"{unit.path}: bind_to target not registered: {unit.bind_to}")
            elif not shape_seam_target_valid(unit, units_by_path):
                target_roles = '+'.join(units_by_path[unit.bind_to].roles)
                result.violations.append(f"{unit.path}: shape_seam bind_to target must be entry/use boundary or generic effect target: {unit.bind_to} ({target_roles})")
        if "config_binder" in roles:
            if unit.axis:
                result.violations.append(f"{unit.path}: config_binder is assembly-root side path and must not declare axis")
            if roles != {"config_binder"}:
                result.violations.append(f"{unit.path}: config_binder must stay standalone")
            if not unit.bind_to:
                result.violations.append(f"{unit.path}: standalone config_binder requires bind_to assembly_root")
            elif unit.bind_to not in units_by_path or "assembly_root" not in units_by_path[unit.bind_to].roles:
                result.violations.append(f"{unit.path}: config_binder bind_to must target assembly_root")
        if "wire_path" in roles:
            if unit.axis:
                result.violations.append(f"{unit.path}: wire_path is a bound side-path transport/schema role and must not declare axis")
            if roles != {"wire_path"}:
                result.violations.append(f"{unit.path}: wire_path must stay standalone")
            if not unit.bind_to:
                result.violations.append(f"{unit.path}: standalone wire_path requires bind_to axis_link")
            elif unit.bind_to not in units_by_path or "axis_link" not in units_by_path[unit.bind_to].roles:
                result.violations.append(f"{unit.path}: wire_path bind_to must target axis_link")
        if "signal_analyzer" in roles:
            if unit.axis:
                result.violations.append(f"{unit.path}: signal_analyzer must not declare axis")
            if roles != {"signal_analyzer"}:
                result.violations.append(f"{unit.path}: signal_analyzer must stay standalone")
        if "acceptance_test" in roles:
            if unit.axis:
                result.violations.append(f"{unit.path}: acceptance_test must not declare axis; it is a side-path system test harness")
            if roles != {"acceptance_test"}:
                result.violations.append(f"{unit.path}: acceptance_test must stay standalone")
    return result


def source_roots(root: Path, unit: Unit) -> list[Path]:
    if unit.sources:
        return [(root / src).resolve() if not Path(src).is_absolute() else Path(src).resolve() for src in unit.sources]
    return [(root / unit.path).resolve()]


def source_files_for_unit(root: Path, unit: Unit) -> list[Path]:
    files: list[Path] = []
    for src in source_roots(root, unit):
        if src.is_file() and src.suffix in SOURCE_EXTENSIONS:
            files.append(src)
        elif src.is_dir():
            files.extend(path for path in src.rglob("*") if path.is_file() and path.suffix in SOURCE_EXTENSIONS)
    return files


def import_patterns_for_names(names: Sequence[str]) -> re.Pattern[str] | None:
    names = tuple(sorted(set(names), key=len, reverse=True))
    if not names:
        return None
    group = "|".join(re.escape(name) for name in names)
    patterns = [
        rf"\b(?:pub\s+use|use)\s+({group})\b",
        rf"\b({group})::",
        rf"\b(?:from|import)\s+({group})\b",
        rf"\bimport\s+.*?\s+from\s+[\"']({group})(?:/[^\"']*)?[\"']",
        rf"\brequire\s*\(\s*[\"']({group})(?:/[^\"']*)?[\"']\s*\)",
        rf"\bimport\s*\(\s*[\"']({group})(?:/[^\"']*)?[\"']\s*\)",
        rf"^\s*[\"']({group})(?:/[^\"']*)?[\"']\s*$",
    ]
    return re.compile("|".join(patterns))


def scan_imports(root: Path, units: Sequence[Unit]) -> dict[str, list[ImportRef]]:
    name_to_unit = {name: unit for unit in units for name in unit.names}
    pattern = import_patterns_for_names(tuple(name_to_unit))
    refs_by_unit = {unit.path: [] for unit in units}
    if pattern is None:
        return refs_by_unit
    for unit in units:
        own_names = set(unit.names)
        for file_path in source_files_for_unit(root, unit):
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            rel = file_path.resolve().relative_to(root.resolve()).as_posix()
            for line_no, line in enumerate(lines, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("//") or stripped.startswith("#") or stripped.startswith("*"):
                    continue
                for match in pattern.finditer(line):
                    target = next((group for group in match.groups() if group), "")
                    if target and target not in own_names:
                        refs_by_unit[unit.path].append(ImportRef(rel, target, line_no, stripped))
    return refs_by_unit


def check_imports(root: Path, data: Mapping[str, Any] | None = None) -> CheckResult:
    result = check_register(root, data)
    if result.violations:
        return result
    raw = load_register(root) if data is None else data
    units = load_units(root, raw)
    units_by_path = {unit.path: unit for unit in units}
    for unit in units:
        for missing in missing_source_roots(root, unit):
            result.violations.append(f"{unit.path}: source root does not exist: {missing}")
    if result.violations:
        return result
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
