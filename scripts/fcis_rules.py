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
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3

CANONICAL_ROLES: dict[str, str] = {
    "utils": "axisless-light closed helpers and universal atoms; no raw env/config/files/process, lifecycle, registry/dispatch, open runtime state, or business vocabulary",
    "meaning_seed": "axis-capable smallest shared business vocabulary and stable value types",
    "meaning_core": "pure business meaning, deterministic rules, and domain/model semantics",
    "run_kit": "axis-able generic operational kit with direct-call helper APIs for open input, runtime/build/test/config/generation/migration/report/artifact helpers; heavy externals allowed, no business vocabulary",
    "axis_link": "directed cross-axis semantic bridge: provider export or caller/core-owned required port (dependency inversion); no transport, workflow, or generic runtime ownership",
    "use_flow": "business workflow brain for one axis",
    "effect_tool": "generic reusable capability adapter/tool/analyzer over concrete external mechanisms; runnable tool apps use an assembly_root+entry_frame wrapper or separate assembly_root -> entry_frame shape",
    "effect_facility": "local generic wrapper/coordinator for reusable effect lifecycle, not the full external engine or business service",
    "link_proto": "axisless bidirectional protocol/channel schema for network, UDS, IPC, pipes, shared memory, stdio, or fd communication; no auth/business ownership",
    "entry_frame": "worker/front-door handler owner: declare route_axes plus entry_type tag, inbound validate/map, route to OOP object/use_flow/effect owner, then map/filter outbound",
    "assembly_root": "axisless bootstrap/supervisor root: delegate config to run_kit, fork/start worker entry_frame(s), supervise lifecycle, and stop; nginx-master style",
    "acceptance_test": "side-path end-to-end/probe test harness that may import production roles",
}

BASE_EDGES: dict[str, set[str]] = {
    "utils": {"utils"},
    "meaning_seed": {"utils", "meaning_seed"},
    "meaning_core": {"utils", "meaning_core", "meaning_seed"},
    "run_kit": {"utils", "run_kit"},
    "axis_link": {"link_proto", "meaning_core", "meaning_seed", "utils"},
    "use_flow": {
        "axis_link",
        "effect_tool",
        "meaning_core",
        "meaning_seed",
        "run_kit",
        "use_flow",
        "utils",
    },
    "entry_frame": {"effect_facility", "entry_frame", "link_proto", "run_kit", "use_flow", "utils"},
    "assembly_root": {
        "assembly_root",
        "entry_frame",
        "run_kit",
        "utils",
    },
    "effect_tool": {"effect_tool", "link_proto", "run_kit", "utils"},
    "effect_facility": {"effect_facility", "effect_tool", "link_proto", "run_kit", "utils"},
    "link_proto": {"run_kit", "utils"},
    "acceptance_test": {"acceptance_test", "utils"},
}

ALLOWED_ROLE_FOLDS: set[frozenset[str]] = {
    frozenset({"assembly_root", "entry_frame"}),
}

PROTECTED_STANDALONE_ROLES = {
    "meaning_seed",
    "run_kit",
    "effect_tool",
    "effect_facility",
    "axis_link",
    "link_proto",
    "acceptance_test",
}

UNIVERSAL_ROLES = {"utils"}
UNIVERSAL_DIRECT_ROLES: set[str] = set()
SHARED_DEFAULT_AXIS_ROLES = {"meaning_seed", "run_kit", "assembly_root"}
PUBLIC_AXISLESS_ROLES = {"link_proto"}
NO_AXIS_ROLES = (
    UNIVERSAL_ROLES | SHARED_DEFAULT_AXIS_ROLES | PUBLIC_AXISLESS_ROLES | {"axis_link", "acceptance_test"}
)
MULTI_AXIS_ROLES = {
    "meaning_seed",
    "run_kit",
    "entry_frame",
    "assembly_root",
    "effect_tool",
    "effect_facility",
}
SIDE_PATH_ROLES = {"acceptance_test"}

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
    "regex",
}
RUST_EXTERNAL_OWNER_ROLES = {"run_kit", "effect_tool", "effect_facility"}
CONFIG_EXTERNAL_OWNER_ROLE = "run_kit"
CONFIG_EXTERNAL_OWNER_AXIS = "config"

# Category-3 direct-call kit: a directly-usable framework library that a
# specific role may bind as its own run_kit/effect_facility-style helper
# (caller supplies data/options, it returns a result; direct-call, no
# capability-trait boundary). This is kept strictly separate from the
# near-std default-direct set above and from register-declared side-effect
# building blocks. Hardcoded per role; a private-kit env override is a
# follow-up.
DIRECT_KIT_BY_ROLE: dict[str, set[str]] = {
    # assembly_root owns argv framing; clap is its CLI/argv direct-call kit.
    "assembly_root": {"clap"},
}
CONFIG_INGRESS_EXACT = {
    "@iarna/toml",
    "config",
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

LINK_PROTO_SCHEMA_ORIGINS = {
    "code_generated": (
        "public schema/protocol artifacts are generated from code-owned definitions"
    ),
    "contract_authored": (
        "public schema/protocol/IDL is authored as the source contract and code "
        "conforms to or is generated from it"
    ),
}

AXIS_LINK_PROVIDER_EXPORT_INTERACTIONS = {
    "provider_port": "consumer axis calls provider-axis semantic capability through a narrow contract",
    "provider_event": "consumer axis observes provider-axis semantic events through a narrow contract",
    "provider_adapter": "consumer axis uses a provider-axis semantic adapter/proxy",
    "provider_interface": "consumer axis uses a provider-axis trait/interface when no narrower provider word fits",
}

AXIS_LINK_REQUIRED_PORT_INTERACTION = "required_port"

AXIS_LINK_INTERACTIONS = {
    **AXIS_LINK_PROVIDER_EXPORT_INTERACTIONS,
    AXIS_LINK_REQUIRED_PORT_INTERACTION: "consumer/caller/core axis owns the required shape; provider/host/plugin/product axis implements it",
}

LEGACY_AXIS_LINK_INTERACTIONS = {"call", "callback", "event", "adapter", "port", "interface"}


@dataclass(frozen=True)
class Unit:
    path: str
    roles: tuple[str, ...]
    axis: tuple[str, ...] = ()
    route_axes: tuple[str, ...] = ()
    entry_type: tuple[str, ...] = ()
    axis_participants: tuple[str, ...] = ()
    provider_axis: str = ""
    consumer_axis: str = ""
    interaction: str = ""
    names: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    externals: tuple[str, ...] = ()
    embedding_use_flow_targets: tuple[str, ...] = ()
    schema_origin: str = ""


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
        return tuple(part.strip() for part in re.split(r"[,+]", axis) if part.strip())
    if isinstance(axis, Sequence) and not isinstance(axis, (bytes, bytearray)):
        return tuple(part.strip() for part in axis if isinstance(part, str) and part.strip())
    return ()


def split_roles(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        return tuple(part.strip() for part in re.split(r"[,+]", raw) if part.strip())
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        return tuple(part.strip() for part in raw if isinstance(part, str) and part.strip())
    return ()


def source_paths(raw_sources: Any) -> tuple[str, ...]:
    if not isinstance(raw_sources, list):
        return ()
    paths: list[str] = []
    for raw_source in raw_sources:
        if isinstance(raw_source, str):
            path = raw_source.strip()
        elif isinstance(raw_source, Mapping):
            raw_path = raw_source.get("path", "")
            path = raw_path.strip() if isinstance(raw_path, str) else ""
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
    raw_path = raw.get("path", "")
    path = raw_path.strip() if isinstance(raw_path, str) else ""
    roles = split_roles(raw.get("roles", ()))
    axis = parse_axis(raw.get("axis", ""))
    route_axes = parse_axis(raw.get("route_axes", ""))
    entry_type = parse_axis(raw.get("entry_type", ""))
    axis_participants = parse_axis(raw.get("axis_participants", ""))
    raw_provider_axis = raw.get("provider_axis", "")
    provider_axis = raw_provider_axis.strip() if isinstance(raw_provider_axis, str) else ""
    raw_consumer_axis = raw.get("consumer_axis", "")
    consumer_axis = raw_consumer_axis.strip() if isinstance(raw_consumer_axis, str) else ""
    raw_interaction = raw.get("interaction", "")
    interaction = raw_interaction.strip() if isinstance(raw_interaction, str) else ""
    raw_names = raw.get("names", ())
    names = (
        tuple(name.strip() for name in raw_names if isinstance(name, str) and name.strip())
        if isinstance(raw_names, list)
        else ()
    )
    if not names:
        names = default_names(path)
    sources = source_paths(raw.get("sources", ()))
    raw_externals = raw.get("externals", ())
    externals = (
        tuple(dep.strip() for dep in raw_externals if isinstance(dep, str) and dep.strip())
        if isinstance(raw_externals, list)
        else ()
    )
    raw_embedding_targets = raw.get("embedding_use_flow_targets", ())
    embedding_use_flow_targets = (
        tuple(
            target.strip()
            for target in raw_embedding_targets
            if isinstance(target, str) and target.strip()
        )
        if isinstance(raw_embedding_targets, list)
        else ()
    )
    raw_schema_origin = raw.get("schema_origin", "")
    schema_origin = raw_schema_origin.strip() if isinstance(raw_schema_origin, str) else ""
    return Unit(
        path=path,
        roles=roles,
        axis=axis,
        route_axes=route_axes,
        entry_type=entry_type,
        axis_participants=axis_participants,
        provider_axis=provider_axis,
        consumer_axis=consumer_axis,
        interaction=interaction,
        names=names,
        sources=sources,
        externals=externals,
        embedding_use_flow_targets=embedding_use_flow_targets,
        schema_origin=schema_origin,
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
    if "entry_frame" in set(unit.roles):
        return unit.route_axes or unit.axis
    return unit.axis


def axis_link_participants(unit: Unit) -> tuple[str, ...]:
    """Return declared participant axes for an axis_link contract."""
    if set(unit.roles) != {"axis_link"}:
        return ()
    return unit.axis_participants


def axis_link_provider(unit: Unit) -> str:
    """Return the provider axis for a directed axis_link."""
    return unit.provider_axis if set(unit.roles) == {"axis_link"} else ""


def axis_link_consumer(unit: Unit) -> str:
    """Return the consumer axis for a directed axis_link."""
    return unit.consumer_axis if set(unit.roles) == {"axis_link"} else ""


def axis_link_interaction(unit: Unit) -> str:
    """Return the normalized interaction value for an axis_link."""
    return unit.interaction if set(unit.roles) == {"axis_link"} else ""


def is_axis_link_required_port(unit: Unit) -> bool:
    return axis_link_interaction(unit) == AXIS_LINK_REQUIRED_PORT_INTERACTION


def is_axis_link_provider_export(unit: Unit) -> bool:
    return axis_link_interaction(unit) in AXIS_LINK_PROVIDER_EXPORT_INTERACTIONS


def duplicate_names(units: Sequence[Unit]) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {}
    for unit in units:
        for name in unit.names:
            seen.setdefault(name, []).append(unit.path)
    return {name: paths for name, paths in seen.items() if len(set(paths)) > 1}


def rust_crate_key(name: str) -> str:
    root = name.strip().split("::", 1)[0].split("/", 1)[0].split(".", 1)[0]
    return root.replace("-", "_")


def external_key(name: str) -> str:
    """Return the stable ownership key for a non-default external dependency."""
    clean = name.strip().strip("\"'`").split("::", 1)[0].strip()
    if not clean:
        return ""
    if clean.startswith("@"):
        parts = clean.split("/")
        return "/".join(parts[:2]) if len(parts) >= 2 else clean
    if "/" in clean or "." in clean:
        return clean.rstrip("/")
    return clean.replace("-", "_")


def external_owners(units: Sequence[Unit]) -> dict[str, list[Unit]]:
    owners: dict[str, list[Unit]] = {}
    for unit in units:
        for dep in unit.externals:
            key = external_key(dep)
            if key:
                owners.setdefault(key, []).append(unit)
    return owners


def allowed_direct_kit(unit: Unit) -> set[str]:
    """Crates a unit may bind directly without an external-owner role.

    Union of the near-std default-direct set (category 1) and the per-role
    direct-call kit allowlist (category 3, e.g. assembly_root -> clap).
    """
    kit: set[str] = set(RUST_DEFAULT_DIRECT_CRATES)
    for role in unit.roles:
        kit |= DIRECT_KIT_BY_ROLE.get(role, set())
    return kit




def is_config_runtime_owner(unit: Unit) -> bool:
    return set(unit.roles) == {CONFIG_EXTERNAL_OWNER_ROLE} and unit.axis == (
        CONFIG_EXTERNAL_OWNER_AXIS,
    )


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

    # axis_link is cross-axis only. Provider-export links are imported by the
    # consumer axis and may reference provider-axis meaning. Extension ports are
    # imported by the caller/framework consumer axis and by implementation-axis
    # adapters, but the port itself must not import provider implementation
    # internals.
    if dst_roles == {"axis_link"}:
        if not src_axis:
            return False
        consumer = axis_link_consumer(dst)
        provider = axis_link_provider(dst)
        if is_axis_link_required_port(dst):
            return bool(consumer or provider) and (
                (bool(consumer) and consumer in src_axis)
                or (bool(provider) and provider in src_axis)
            )
        return bool(consumer) and consumer in src_axis

    if set(src.roles) == {"axis_link"}:
        if is_axis_link_required_port(src):
            if dst_roles <= UNIVERSAL_ROLES:
                return True
            if dst_roles & {"meaning_seed"}:
                return (not dst_axis) or axis_overlaps(axis_link_participants(src), dst_axis)
            if dst_roles & {"link_proto"}:
                return True
            return False
        provider = axis_link_provider(src)
        return (
            bool(provider)
            and bool(dst_axis)
            and provider in dst_axis
            and bool(dst_roles & {"meaning_seed", "meaning_core"})
        )
    if dst_roles & {"axis_link"}:
        return False
    if set(src.roles) & {"assembly_root"}:
        return True
    if not src_axis:
        return not dst_axis or bool(dst_roles & SHARED_DEFAULT_AXIS_ROLES)
    if not dst_axis:
        return bool(
            dst_roles
            & (
                UNIVERSAL_ROLES
                | SHARED_DEFAULT_AXIS_ROLES
                | PUBLIC_AXISLESS_ROLES
                | SIDE_PATH_ROLES
            )
        )
    return axis_overlaps(src_axis, dst_axis)


def role_direct_allowed(src_role: str, dst_role: str) -> bool:
    return dst_role in role_edges(src_role)


def role_dependency_allowed(
    src: Unit, dst: Unit, units_by_path: Mapping[str, Unit]
) -> bool:
    if src.path == dst.path:
        return True
    if set(src.roles) <= SIDE_PATH_ROLES and bool(src.roles):
        return True
    if embedding_use_flow_allowed(src, dst):
        return True
    if not axis_visible(src, dst, units_by_path):
        return False
    dst_roles = set(dst.roles)
    if dst_roles and dst_roles <= UNIVERSAL_DIRECT_ROLES:
        return True
    if dst_roles == {"axis_link"} and is_axis_link_required_port(dst):
        # Extension ports are caller-owned cross-axis contracts. The caller
        # side usually imports/calls the port from use_flow/entry_frame-style
        # code; the implementation side imports the same port from an adapter,
        # use_flow, or effect owner. Provider meaning_core should stay pure and
        # be adapted from outside.
        return bool(set(src.roles) & {"use_flow", "effect_tool", "effect_facility", "acceptance_test"})

    # Other destination folds expose the folded unit's full boundary, not a
    # shortcut through whichever role happens to be importable.
    return all(
        any(role_direct_allowed(src_role, dst_role) for src_role in src.roles)
        for dst_role in dst.roles
    )


def _normalized_unit_path(path: str) -> str:
    return path.strip().replace("\\", "/").strip("/")


def path_matches_registered_target(path: str, target: str) -> bool:
    path = _normalized_unit_path(path)
    target = _normalized_unit_path(target)
    return bool(target) and (
        path == target
        or path.startswith(f"{target}/")
        or f"/{target}/" in path
        or path.endswith(f"/{target}")
    )


def embedding_use_flow_allowed(src: Unit, dst: Unit) -> bool:
    return (
        set(src.roles) == {"assembly_root"}
        and "use_flow" in set(dst.roles)
        and any(
            path_matches_registered_target(dst.path, target)
            for target in src.embedding_use_flow_targets
        )
    )


def fold_allowed(unit: Unit, units_by_path: Mapping[str, Unit]) -> bool:
    roles = set(unit.roles)
    if roles == {"assembly_root", "entry_frame"}:
        return True
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
    if not isinstance(raw, Mapping):
        result.violations.append("role_register.json must be a JSON object")
        return result
    allowed_top = {"schema_version", "units"}
    for key in sorted(set(raw) - allowed_top):
        result.violations.append(
            f"role_register.json must be declarative units only; remove unsupported top-level {key}"
        )
    if raw.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        result.violations.append(
            f"role_register.json schema_version must be {SCHEMA_VERSION}"
        )
    raw_units_value = raw.get("units", [])
    if "units" in raw and not isinstance(raw_units_value, list):
        result.violations.append("role_register.json units must be a list")
        raw_units = []
    else:
        raw_units = raw_units_value
    allowed_unit_keys = {
        "path",
        "roles",
        "axis",
        "route_axes",
        "entry_type",
        "axis_participants",
        "provider_axis",
        "consumer_axis",
        "interaction",
        "names",
        "sources",
        "externals",
        "embedding_use_flow_targets",
        "schema_origin",
    }
    def unit_label(raw_unit: Mapping[str, Any], unit_index: int) -> str:
        raw_path = raw_unit.get("path", "")
        if isinstance(raw_path, str) and raw_path.strip():
            return raw_path.strip()
        return f"units[{unit_index}]"

    def validate_non_empty_string_list(
        value: Any, label: str, *, allow_string: bool = False
    ) -> bool:
        if allow_string and isinstance(value, str):
            return bool(split_roles(value))
        if not isinstance(value, list):
            return False
        ok = True
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                result.violations.append(
                    f"{label}[{index}] must be a non-empty string"
                )
                ok = False
        return ok

    valid_raw_units: list[Mapping[str, Any]] = []
    for unit_index, raw_unit in enumerate(raw_units):
        if not isinstance(raw_unit, Mapping):
            result.violations.append(f"units[{unit_index}]: unit must be an object")
            continue
        structural_valid = True
        label = unit_label(raw_unit, unit_index)
        extra_keys = sorted(set(raw_unit) - allowed_unit_keys)
        if extra_keys:
            result.violations.append(
                f"{label}: unsupported unit field(s): {', '.join(extra_keys)}"
            )
        raw_path = raw_unit.get("path", "")
        if "path" not in raw_unit:
            result.violations.append(f"{label}: path must be a non-empty string")
            structural_valid = False
        elif not isinstance(raw_path, str) or not raw_path.strip():
            result.violations.append(f"{label}: path must be a non-empty string")
            structural_valid = False

        raw_roles = raw_unit.get("roles", ())
        roles_valid = True
        if "roles" in raw_unit:
            if isinstance(raw_roles, str):
                if not split_roles(raw_roles):
                    result.violations.append(f"{label}: roles must not be empty")
                    roles_valid = False
            elif isinstance(raw_roles, list):
                roles_valid = validate_non_empty_string_list(raw_roles, f"{label}: roles")
            else:
                result.violations.append(f"{label}: roles must be a string or list")
                roles_valid = False
        if not roles_valid:
            structural_valid = False

        raw_axis = raw_unit.get("axis", "")
        axis_valid = True
        if "axis" in raw_unit:
            if isinstance(raw_axis, str):
                pass
            elif isinstance(raw_axis, list):
                axis_valid = validate_non_empty_string_list(raw_axis, f"{label}: axis")
            else:
                result.violations.append(f"{label}: axis must be a string or list")
                axis_valid = False
        if not axis_valid:
            structural_valid = False

        raw_route_axes = raw_unit.get("route_axes", "")
        route_axes_valid = True
        if "route_axes" in raw_unit:
            if isinstance(raw_route_axes, str):
                if not parse_axis(raw_route_axes):
                    result.violations.append(
                        f"{label}: route_axes must not be empty"
                    )
                    route_axes_valid = False
            elif isinstance(raw_route_axes, list):
                route_axes_valid = validate_non_empty_string_list(
                    raw_route_axes, f"{label}: route_axes"
                )
            else:
                result.violations.append(
                    f"{label}: route_axes must be a string or list"
                )
                route_axes_valid = False
        if not route_axes_valid:
            structural_valid = False

        raw_entry_type = raw_unit.get("entry_type", "")
        entry_type_valid = True
        if "entry_type" in raw_unit:
            if isinstance(raw_entry_type, str):
                if not parse_axis(raw_entry_type):
                    result.violations.append(
                        f"{label}: entry_type must not be empty"
                    )
                    entry_type_valid = False
            elif isinstance(raw_entry_type, list):
                entry_type_valid = validate_non_empty_string_list(
                    raw_entry_type, f"{label}: entry_type"
                )
            else:
                result.violations.append(
                    f"{label}: entry_type must be a string or list"
                )
                entry_type_valid = False
        if not entry_type_valid:
            structural_valid = False

        raw_axis_participants = raw_unit.get("axis_participants", "")
        axis_participants_valid = True
        if "axis_participants" in raw_unit:
            if isinstance(raw_axis_participants, str):
                if not parse_axis(raw_axis_participants):
                    result.violations.append(
                        f"{label}: axis_participants must not be empty"
                    )
                    axis_participants_valid = False
            elif isinstance(raw_axis_participants, list):
                axis_participants_valid = validate_non_empty_string_list(
                    raw_axis_participants, f"{label}: axis_participants"
                )
            else:
                result.violations.append(
                    f"{label}: axis_participants must be a string or list"
                )
                axis_participants_valid = False
        if not axis_participants_valid:
            structural_valid = False

        raw_provider_axis = raw_unit.get("provider_axis", "")
        if "provider_axis" in raw_unit and (
            not isinstance(raw_provider_axis, str) or not raw_provider_axis.strip()
        ):
            result.violations.append(f"{label}: provider_axis must be a non-empty string")
            structural_valid = False

        raw_consumer_axis = raw_unit.get("consumer_axis", "")
        if "consumer_axis" in raw_unit and (
            not isinstance(raw_consumer_axis, str) or not raw_consumer_axis.strip()
        ):
            result.violations.append(f"{label}: consumer_axis must be a non-empty string")
            structural_valid = False

        raw_interaction = raw_unit.get("interaction", "")
        if "interaction" in raw_unit and (
            not isinstance(raw_interaction, str) or not raw_interaction.strip()
        ):
            result.violations.append(f"{label}: interaction must be a non-empty string")
            structural_valid = False
        elif isinstance(raw_interaction, str) and raw_interaction.strip():
            interaction_value = raw_interaction.strip()
            if interaction_value in LEGACY_AXIS_LINK_INTERACTIONS:
                allowed_interactions = ", ".join(sorted(AXIS_LINK_INTERACTIONS))
                result.violations.append(
                    f"{label}: interaction {interaction_value!r} is too broad; use one of: {allowed_interactions}"
                )
                structural_valid = False
            elif interaction_value not in AXIS_LINK_INTERACTIONS:
                allowed_interactions = ", ".join(sorted(AXIS_LINK_INTERACTIONS))
                result.violations.append(
                    f"{label}: interaction must be one of: {allowed_interactions}"
                )
                structural_valid = False

        raw_names = raw_unit.get("names", ())
        names_valid = True
        if "names" in raw_unit:
            if not isinstance(raw_names, list):
                result.violations.append(f"{label}: names must be a list")
                names_valid = False
            else:
                names_valid = validate_non_empty_string_list(raw_names, f"{label}: names")
        if not names_valid:
            structural_valid = False

        raw_externals = raw_unit.get("externals", ())
        externals_valid = True
        if "externals" in raw_unit:
            if not isinstance(raw_externals, list):
                result.violations.append(f"{label}: externals must be a list")
                externals_valid = False
            else:
                externals_valid = validate_non_empty_string_list(
                    raw_externals, f"{label}: externals"
                )
        if not externals_valid:
            structural_valid = False

        raw_embedding_targets = raw_unit.get("embedding_use_flow_targets", ())
        embedding_targets_valid = True
        if "embedding_use_flow_targets" in raw_unit:
            if not isinstance(raw_embedding_targets, list):
                result.violations.append(
                    f"{label}: embedding_use_flow_targets must be a list"
                )
                embedding_targets_valid = False
            else:
                embedding_targets_valid = validate_non_empty_string_list(
                    raw_embedding_targets, f"{label}: embedding_use_flow_targets"
                )
        if not embedding_targets_valid:
            structural_valid = False

        raw_schema_origin = raw_unit.get("schema_origin", "")
        if "schema_origin" in raw_unit and not isinstance(raw_schema_origin, str):
            result.violations.append(f"{label}: schema_origin must be a string")
            structural_valid = False
        elif isinstance(raw_schema_origin, str) and raw_schema_origin.strip():
            schema_origin_value = raw_schema_origin.strip()
            if schema_origin_value not in LINK_PROTO_SCHEMA_ORIGINS:
                allowed_schema_origins = ", ".join(sorted(LINK_PROTO_SCHEMA_ORIGINS))
                result.violations.append(
                    f"{label}: schema_origin must be one of: {allowed_schema_origins}"
                )

        raw_sources = raw_unit.get("sources", ())
        if "sources" in raw_unit and not isinstance(raw_sources, list):
            result.violations.append(
                f"{label}: sources must be a list of paths or source objects"
            )
            structural_valid = False
        elif isinstance(raw_sources, list):
            unit_roles = set(split_roles(raw_roles)) if roles_valid else set()
            for index, raw_source in enumerate(raw_sources):
                source_label = f"{label}: sources[{index}]"
                if isinstance(raw_source, str):
                    if not raw_source.strip():
                        result.violations.append(f"{source_label}: source path is empty")
                        structural_valid = False
                    continue
                if not isinstance(raw_source, Mapping):
                    result.violations.append(
                        f"{source_label}: source must be a path string or object with path"
                    )
                    structural_valid = False
                    continue
                source_extra_keys = sorted(set(raw_source) - {"path", "roles"})
                if source_extra_keys:
                    result.violations.append(
                        f"{source_label}: unsupported source field(s): {', '.join(source_extra_keys)}"
                    )
                source_path = raw_source.get("path", "")
                if not isinstance(source_path, str) or not source_path.strip():
                    result.violations.append(f"{source_label}: source path is empty")
                    structural_valid = False
                raw_source_roles = raw_source.get("roles", ())
                source_roles_valid = True
                if "roles" in raw_source:
                    if isinstance(raw_source_roles, str):
                        if not split_roles(raw_source_roles):
                            result.violations.append(
                                f"{source_label}: source roles must not be empty"
                            )
                            source_roles_valid = False
                    elif isinstance(raw_source_roles, list):
                        source_roles_valid = validate_non_empty_string_list(
                            raw_source_roles, f"{source_label}: roles"
                        )
                    else:
                        result.violations.append(
                            f"{source_label}: source roles must be a string or list"
                        )
                        source_roles_valid = False
                if not source_roles_valid:
                    structural_valid = False
                    source_roles = set()
                else:
                    source_roles = set(split_roles(raw_source_roles))
                unknown_source_roles = sorted(source_roles - set(CANONICAL_ROLES))
                if unknown_source_roles:
                    result.violations.append(
                        f"{source_label}: unknown source role(s): {', '.join(unknown_source_roles)}"
                    )
                if source_roles and unit_roles and not source_roles <= unit_roles:
                    result.violations.append(
                        f"{source_label}: source roles must be a subset of the unit roles"
                    )
        if structural_valid:
            valid_raw_units.append(raw_unit)
    units = [normalize_unit(unit) for unit in valid_raw_units]
    units_by_path = {unit.path: unit for unit in units}
    if len(units_by_path) != len(units):
        result.violations.append("duplicate unit path in role_register.json")
    for name, paths in sorted(duplicate_names(units).items()):
        result.violations.append(
            f"duplicate import name {name!r} used by: {', '.join(sorted(set(paths)))}"
        )
    owner_paths_by_external: dict[str, list[str]] = {}
    for unit in units:
        for dep in unit.externals:
            key = external_key(dep)
            if key:
                owner_paths_by_external.setdefault(key, []).append(unit.path)
    for dep_key, paths in sorted(owner_paths_by_external.items()):
        unique_paths = sorted(set(paths))
        if len(unique_paths) > 1:
            result.violations.append(
                f"duplicate external dependency binding {dep_key!r} used by: {', '.join(unique_paths)}"
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
        if len(set(unit.axis)) != len(unit.axis):
            result.violations.append(
                f"{unit.path}: duplicate axis value(s) are not allowed: {'+'.join(unit.axis)}"
            )
        if len(set(unit.route_axes)) != len(unit.route_axes):
            result.violations.append(
                f"{unit.path}: duplicate route_axes value(s) are not allowed: {'+'.join(unit.route_axes)}"
            )
        if len(set(unit.entry_type)) != len(unit.entry_type):
            result.violations.append(
                f"{unit.path}: duplicate entry_type value(s) are not allowed: {'+'.join(unit.entry_type)}"
            )
        if len(set(unit.axis_participants)) != len(unit.axis_participants):
            result.violations.append(
                f"{unit.path}: duplicate axis_participants value(s) are not allowed: {'+'.join(unit.axis_participants)}"
            )
        if unit.entry_type and not (roles & {"entry_frame", "assembly_root"}):
            result.violations.append(
                f"{unit.path}: entry_type is valid only for units that include entry_frame or assembly_root"
            )
        if unit.route_axes and "entry_frame" not in roles:
            result.violations.append(
                f"{unit.path}: route_axes is valid only for units that include entry_frame"
            )
        if unit.axis_participants and roles != {"axis_link"}:
            result.violations.append(
                f"{unit.path}: axis_participants is valid only for standalone axis_link units"
            )
        if unit.provider_axis and roles != {"axis_link"}:
            result.violations.append(
                f"{unit.path}: provider_axis is valid only for standalone axis_link units"
            )
        if unit.consumer_axis and roles != {"axis_link"}:
            result.violations.append(
                f"{unit.path}: consumer_axis is valid only for standalone axis_link units"
            )
        if unit.interaction and roles != {"axis_link"}:
            result.violations.append(
                f"{unit.path}: interaction is valid only for standalone axis_link units"
            )
        if (
            roles == {"run_kit"}
            and CONFIG_EXTERNAL_OWNER_AXIS in unit.axis
            and unit.axis != (CONFIG_EXTERNAL_OWNER_AXIS,)
        ):
            result.violations.append(
                f"{unit.path}: config-axis run_kit must declare exactly axis 'config'; do not combine config with application axes"
            )
        if roles & UNIVERSAL_ROLES:
            if unit.axis:
                result.violations.append(
                    f"{unit.path}: utils must not declare axis"
                )
            if not roles <= UNIVERSAL_ROLES:
                result.violations.append(
                    f"{unit.path}: utils must stay standalone; split utility code from role-owned code"
                )
        if unit.externals:
            direct_kit = allowed_direct_kit(unit)
            if "assembly_root" in roles:
                # assembly_root delegates config/env and capability externals to
                # run_kit/effect_tool/effect_facility, but it MAY bind its own
                # direct-call kit (clap) and the near-std default-direct set.
                undeclared = [
                    dep for dep in unit.externals if external_key(dep) not in direct_kit
                ]
                if undeclared:
                    result.violations.append(
                        f"{unit.path}: assembly_root may bind only direct-call kit (e.g. clap) or default-direct crates; bind deployment config/env externals to run_kit in axis 'config', direct-call generic operational externals to run_kit, and concrete external capability mechanisms to effect_tool/effect_facility"
                    )
            else:
                non_default_externals = [
                    dep for dep in unit.externals
                    if external_key(dep) not in direct_kit
                ]
                config_externals = [
                    dep for dep in non_default_externals if is_config_ingress_module(dep)
                ]
                non_config_externals = [
                    dep for dep in non_default_externals if not is_config_ingress_module(dep)
                ]
                if config_externals and not is_config_runtime_owner(unit):
                    result.violations.append(
                        f"{unit.path}: deployment config/env externals may be declared only on run_kit units in axis 'config'"
                    )
                if non_config_externals and not (roles & RUST_EXTERNAL_OWNER_ROLES):
                    result.violations.append(
                        f"{unit.path}: non-default externals may be declared only on run_kit/effect_tool/effect_facility units; deployment config/env externals belong to run_kit axis 'config'"
                    )
        if unit.embedding_use_flow_targets:
            if roles != {"assembly_root"}:
                result.violations.append(
                    f"{unit.path}: embedding_use_flow_targets is valid only for standalone assembly_root units"
                )
            seen_embedding_targets: set[str] = set()
            for target in unit.embedding_use_flow_targets:
                if target in seen_embedding_targets:
                    result.violations.append(
                        f"{unit.path}: duplicate embedding_use_flow_targets value {target!r}"
                    )
                seen_embedding_targets.add(target)
                if path_matches_registered_target(unit.path, target):
                    result.violations.append(
                        f"{unit.path}: embedding_use_flow_targets must not target the assembly_root itself"
                    )
                    continue
                target_unit = next(
                    (
                        candidate
                        for candidate in units
                        if path_matches_registered_target(candidate.path, target)
                    ),
                    None,
                )
                if target_unit is None:
                    result.violations.append(
                        f"{unit.path}: embedding_use_flow_targets target {target!r} does not match a registered unit"
                    )
                elif "use_flow" not in set(target_unit.roles):
                    result.violations.append(
                        f"{unit.path}: embedding_use_flow_targets target {target!r} must resolve to a use_flow unit"
                    )
        if "entry_frame" in roles and not unit.route_axes:
            result.violations.append(
                f"{unit.path}: entry_frame must declare route_axes for one or more semantic target axes"
            )
        if "entry_frame" in roles and not unit.entry_type:
            result.violations.append(
                f"{unit.path}: entry_frame must declare entry_type (for example server, web, desktop)"
            )
        if "entry_frame" in roles and unit.axis:
            result.violations.append(
                f"{unit.path}: entry_frame should use route_axes instead of axis; axis belongs to semantic owners"
            )
        if "entry_frame" in roles and len(unit.route_axes) > 2:
            result.violations.append(
                f"{unit.path}: entry_frame may route into at most two semantic axes; use entry_type for surface tags and split broader multiplexing with extra front doors, axis_link, or link_proto/process boundaries"
            )
        if "cli" in set(unit.entry_type) and "assembly_root" not in roles:
            result.violations.append(
                f"{unit.path}: cli entry_type belongs to assembly_root request framing; do not register cli on pure entry_frame units"
            )
        if "use_flow" in roles and len(unit.axis) != 1:
            result.violations.append(
                f"{unit.path}: use_flow must declare exactly one axis; split cross-axis workflows and connect them through axis_link"
            )
        semantic_axis_missing = not unit.axis and not roles <= NO_AXIS_ROLES and "entry_frame" not in roles
        if semantic_axis_missing:
            result.violations.append(
                f"{unit.path}: missing axis; only shared/default roles, side-path acceptance_test, or utils may omit it"
            )
        if len(unit.axis) > 1 and not roles <= MULTI_AXIS_ROLES:
            result.violations.append(
                f"{unit.path}: one or more roles cannot manually span multiple axes"
            )
        if "axis_link" in roles:
            if roles != {"axis_link"}:
                result.violations.append(
                    f"{unit.path}: axis_link must stay standalone"
                )
            if unit.axis:
                result.violations.append(
                    f"{unit.path}: axis_link must not declare axis; use provider_axis, consumer_axis, and axis_participants"
                )
            if not unit.axis_participants:
                result.violations.append(
                    f"{unit.path}: axis_link must declare axis_participants with exactly provider_axis and consumer_axis"
                )
            elif len(set(unit.axis_participants)) != 2:
                result.violations.append(
                    f"{unit.path}: axis_link must be one-directional between exactly two axes; split multi-party or two-way contracts into separate axis_link units"
                )
            if not unit.provider_axis:
                result.violations.append(
                    f"{unit.path}: axis_link must declare provider_axis"
                )
            if not unit.consumer_axis:
                result.violations.append(
                    f"{unit.path}: axis_link must declare consumer_axis"
                )
            if unit.provider_axis and unit.consumer_axis:
                if unit.provider_axis == unit.consumer_axis:
                    result.violations.append(
                        f"{unit.path}: provider_axis and consumer_axis must be different"
                    )
                participant_set = set(unit.axis_participants)
                directed_set = {unit.provider_axis, unit.consumer_axis}
                if participant_set and participant_set != directed_set:
                    result.violations.append(
                        f"{unit.path}: axis_participants must equal provider_axis+consumer_axis for a directed axis_link"
                    )
            if not unit.interaction:
                result.violations.append(
                    f"{unit.path}: axis_link must declare interaction"
                )
            elif unit.interaction == AXIS_LINK_REQUIRED_PORT_INTERACTION:
                result.notes.append(
                    f"{unit.path}: required_port means consumer_axis is the caller/core/workflow that owns the required shape and provider_axis is the host/plugin/product implementation axis; port must not import provider implementation internals"
                )
            elif unit.interaction in AXIS_LINK_PROVIDER_EXPORT_INTERACTIONS:
                result.notes.append(
                    f"{unit.path}: provider-export axis_link must reference provider-axis meaning_core and be imported by consumer-axis workflow only"
                )
        if unit.schema_origin and roles != {"link_proto"}:
            result.violations.append(
                f"{unit.path}: schema_origin is only valid for standalone link_proto units"
            )
        if "link_proto" in roles:
            if roles != {"link_proto"}:
                result.violations.append(
                    f"{unit.path}: link_proto must stay standalone"
                )
            if unit.axis:
                result.violations.append(
                    f"{unit.path}: link_proto is axisless; do not declare axis on network/UDS/IPC/pipe/shared-memory/stdin-stdout protocol contracts"
                )
            if not unit.schema_origin:
                result.notes.append(
                    f"{unit.path}: link_proto may declare schema_origin as code_generated or contract_authored to clarify protocol/schema authorship"
                )
        side_roles = roles & SIDE_PATH_ROLES
        if side_roles:
            side_role = sorted(side_roles)[0]
            if unit.axis:
                result.violations.append(
                    f"{unit.path}: {side_role} must not declare axis; it is a side-path acceptance-test harness"
                )
            if roles != {side_role}:
                result.violations.append(
                    f"{unit.path}: {side_role} must stay standalone"
                )
    return result


def source_roots(root: Path, unit: Unit) -> list[Path]:
    if unit.sources:
        resolved: list[Path] = []
        unit_parts = Path(unit.path).parts
        for src in unit.sources:
            src_path = Path(src)
            if src_path.is_absolute():
                resolved.append(src_path.resolve())
            else:
                src_parts = src_path.parts
                is_explicit_root_relative = (
                    len(src_parts) >= len(unit_parts)
                    and src_parts[: len(unit_parts)] == unit_parts
                )
                root_relative = (root / src_path).resolve()
                unit_relative = (root / unit.path / src_path).resolve()
                if is_explicit_root_relative:
                    resolved.append(root_relative)
                elif unit_relative.exists():
                    resolved.append(unit_relative)
                elif root_relative.exists():
                    resolved.append(root_relative)
                else:
                    resolved.append(unit_relative)
        return resolved
    return [(root / unit.path).resolve()]


def source_files_for_unit(root: Path, unit: Unit) -> list[Path]:
    files: list[Path] = []
    _dep_files = ("Cargo.toml", "go.mod", "package.json")
    for src in source_roots(root, unit):
        if src.is_file() and (
            src.suffix in SOURCE_EXTENSIONS
            or src.name in _dep_files
        ):
            files.append(src)
        elif src.is_dir():
            files.extend(
                path
                for path in src.rglob("*")
                if path.is_file()
                and (
                    path.suffix in SOURCE_EXTENSIONS
                    or path.name in _dep_files
                )
            )
    if unit.sources:
        package_root = (root / unit.path).resolve()
        for dep_file in _dep_files:
            manifest = package_root / dep_file
            if manifest.is_file():
                files.append(manifest)
    unique: dict[Path, None] = {}
    for file_path in files:
        unique[file_path.resolve()] = None
    return list(unique)


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
    module_key = rust_crate_key(module)
    for name in sorted(set(names), key=len, reverse=True):
        if module == name:
            return name
        for sep in ("::", ".", "/", "\\"):
            if module.startswith(name + sep):
                return name
        name_key = rust_crate_key(name)
        if name_key and module_key == name_key:
            return name
        if module_key == f"checker_{name_key}":
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


def _tree_sitter_named_children(node: Any) -> list[Any]:
    named = getattr(node, "named_children", None)
    if named is not None:
        return list(named)
    return [child for child in node.children if getattr(child, "is_named", True)]


def _ts_rust_roots_from_use_child(source_bytes: bytes, node: Any) -> list[str]:
    local_roots = {"crate", "super", "self", "Self"}
    if node.type in {"identifier", "crate", "super", "self"}:
        name = _tree_sitter_text(source_bytes, node)
        return [] if name in local_roots else [name]
    if node.type == "use_list":
        roots: list[str] = []
        for child in _tree_sitter_named_children(node):
            roots.extend(_ts_rust_roots_from_use_child(source_bytes, child))
        return roots
    if node.type in {
        "scoped_identifier",
        "scoped_use_list",
        "use_as_clause",
        "use_wildcard",
    }:
        children = _tree_sitter_named_children(node)
        if not children:
            return []
        return _ts_rust_roots_from_use_child(source_bytes, children[0])
    return []


def _ts_rust_use_roots(source_bytes: bytes, use_node: Any) -> list[str]:
    """Extract external crate roots from a tree-sitter use_declaration node."""
    roots: list[str] = []
    for child in _tree_sitter_named_children(use_node):
        roots.extend(_ts_rust_roots_from_use_child(source_bytes, child))
    return roots


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

        if suffix == ".rs" and node_type == "use_declaration":
            for root in _ts_rust_use_roots(source_bytes, node):
                candidates.append((root, start_line, _line_at(lines, start_line)))
        elif suffix == ".rs" and node_type == "extern_crate_declaration":
            for child in node.children:
                if child.type == "identifier":
                    name = _tree_sitter_text(source_bytes, child)
                    candidates.append((name, start_line, _line_at(lines, start_line)))
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
        r"^\[(?:target\.[^]]+\.)?(?:build-)?dependencies(?:\.([A-Za-z0-9_-]+))?\]"
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


def _go_mod_dependency_candidates(
    source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    candidates: list[tuple[str, int, str]] = []
    in_require_block = False
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        if line.startswith("require") and "(" in line:
            in_require_block = True
            continue
        if in_require_block:
            if line.startswith(")"):
                in_require_block = False
                continue
            parts = line.split()
            if parts:
                candidates.append((parts[0], line_no, _line_at(lines, line_no)))
            continue
        if line.startswith("require "):
            parts = line.split()
            if len(parts) >= 2:
                candidates.append((parts[1], line_no, _line_at(lines, line_no)))
    return candidates


def _package_json_dependency_candidates(
    source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    candidates: list[tuple[str, int, str]] = []
    try:
        data = json.loads(source)
    except (json.JSONDecodeError, ValueError):
        return candidates
    seen: set[str] = set()
    for section in ("dependencies", "peerDependencies", "optionalDependencies"):
        deps = data.get(section)
        if isinstance(deps, dict):
            for name in sorted(deps):
                if name in seen:
                    continue
                for line_no, line in enumerate(lines, start=1):
                    if f'"{name}"' in line:
                        candidates.append((name, line_no, _line_at(lines, line_no)))
                        seen.add(name)
                        break
    return candidates


PYTHON_STDLIB_MODULES: set[str] = {
    "__future__", "_thread", "abc", "aifc", "argparse", "array", "ast",
    "asynchat", "asyncio", "asyncore", "atexit", "audioop", "base64",
    "bdb", "binascii", "binhex", "bisect", "builtins", "bz2",
    "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd", "code",
    "codecs", "codeop", "collections", "colorsys", "compileall",
    "concurrent", "configparser", "contextlib", "contextvars", "copy",
    "copyreg", "cProfile", "crypt", "csv", "ctypes", "curses",
    "dataclasses", "datetime", "dbm", "decimal", "difflib", "dis",
    "distutils", "doctest", "email", "encodings", "enum", "errno",
    "faulthandler", "fcntl", "filecmp", "fileinput", "fnmatch",
    "formatter", "fractions", "ftplib", "functools", "gc", "getopt",
    "getpass", "gettext", "glob", "grp", "gzip", "hashlib", "heapq",
    "hmac", "html", "http", "idlelib", "imaplib", "imghdr", "imp",
    "importlib", "inspect", "io", "ipaddress", "itertools", "json",
    "keyword", "lib2to3", "linecache", "locale", "logging", "lzma",
    "mailbox", "mailcap", "marshal", "math", "mimetypes", "mmap",
    "modulefinder", "multiprocessing", "netrc", "nis", "nntplib",
    "numbers", "operator", "optparse", "os", "ossaudiodev", "parser",
    "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil",
    "platform", "plistlib", "poplib", "posix", "posixpath", "pprint",
    "profile", "pstats", "pty", "pwd", "py_compile", "pyclbr",
    "pydoc", "queue", "quopri", "random", "re", "readline", "reprlib",
    "resource", "rlcompleter", "runpy", "sched", "secrets", "select",
    "selectors", "shelve", "shlex", "shutil", "signal", "site",
    "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "spwd",
    "sqlite3", "ssl", "stat", "statistics", "string", "stringprep",
    "struct", "subprocess", "sunau", "symtable", "sys", "sysconfig",
    "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile", "termios",
    "test", "textwrap", "threading", "time", "timeit", "tkinter",
    "token", "tokenize", "tomllib", "trace", "traceback", "tracemalloc",
    "tty", "turtle", "turtledemo", "types", "typing", "unicodedata",
    "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave",
    "weakref", "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib",
    "xml", "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib",
    "zoneinfo",
}


def _python_external_import_candidates(
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
                top_level = alias.name.split(".")[0]
                if top_level in PYTHON_STDLIB_MODULES:
                    continue
                candidates.append(
                    (top_level, node.lineno, _line_at(lines, node.lineno))
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                continue
            if node.module:
                top_level = node.module.split(".")[0]
                if top_level in PYTHON_STDLIB_MODULES:
                    continue
                candidates.append(
                    (top_level, node.lineno, _line_at(lines, node.lineno))
                )
    return candidates


GO_DEFAULT_DIRECT_MODULES: set[str] = set()

PYTHON_DEFAULT_DIRECT_PACKAGES: set[str] = {
    "typing_extensions",
}

JS_DEFAULT_DIRECT_PACKAGES: set[str] = {
    "typescript",
    "tslib",
}


def import_candidates_for_file(
    file_path: Path, source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    if file_path.name == "Cargo.toml":
        return _cargo_toml_dependency_candidates(source, lines)
    if file_path.name == "go.mod":
        return _go_mod_dependency_candidates(source, lines)
    if file_path.name == "package.json":
        return _package_json_dependency_candidates(source, lines)
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


def _rust_test_line_ranges(source: str) -> set[int]:
    test_lines: set[int] = set()
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "#[cfg(test)]":
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines):
                mod_line = lines[j]
                brace_pos = mod_line.find("{")
                if brace_pos != -1 and mod_line.strip().startswith("mod "):
                    start = i
                    depth = 0
                    for k in range(j, len(lines)):
                        depth += lines[k].count("{") - lines[k].count("}")
                        end = k
                        if depth == 0:
                            break
                    for ln in range(start, end + 1):
                        test_lines.add(ln + 1)
                    i = end + 1
                    continue
        i += 1
    return test_lines


def _is_test_file(file_path: Path, root: Path) -> bool:
    try:
        rel = file_path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    parts = rel.parts
    if parts and parts[0] in ("tests", "benches", "fuzz", "examples"):
        return True
    for part in parts:
        if part in ("tests", "benches", "fuzz"):
            return True
    if file_path.suffix == ".go" and file_path.name.endswith("_test.go"):
        return True
    name = file_path.name
    if file_path.suffix in (".js", ".jsx", ".ts", ".tsx"):
        if ".test." in name or ".spec." in name:
            return True
        for part in parts:
            if part in ("__tests__", "test", "tests"):
                return True
    if file_path.suffix == ".py" and (name.startswith("test_") or name.endswith("_test.py")):
        return True
    if file_path.suffix == ".rs" and name.startswith("bench_"):
        return True
    return False


_DEPENDENCY_FILE_NAMES = {"Cargo.toml", "go.mod", "package.json"}


def _rust_test_lines_for_file(file_path: Path, source: str) -> set[int]:
    if file_path.suffix != ".rs":
        return set()
    return _rust_test_line_ranges(source)


def _production_import_candidates_for_file(
    root: Path, file_path: Path, source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    if _is_test_file(file_path, root):
        return []
    test_lines = _rust_test_lines_for_file(file_path, source)
    candidates = import_candidates_for_file(file_path, source, lines)
    if not test_lines:
        return candidates
    return [candidate for candidate in candidates if candidate[1] not in test_lines]


def _production_config_candidates_for_file(
    root: Path, file_path: Path, source: str, lines: Sequence[str]
) -> list[tuple[str, int, str]]:
    if _is_test_file(file_path, root):
        return []
    test_lines = _rust_test_lines_for_file(file_path, source)
    candidates = []
    for module_name, line_no, line in import_candidates_for_file(file_path, source, lines):
        if is_config_ingress_module(module_name):
            candidates.append((module_name, line_no, line))
    candidates.extend(config_usage_candidates_for_file(file_path, source, lines))
    if not test_lines:
        return candidates
    return [candidate for candidate in candidates if candidate[1] not in test_lines]


def scan_external_refs(
    root: Path, units: Sequence[Unit]
) -> dict[str, list[ImportRef]]:
    registered_names = tuple(name for unit in units for name in unit.names)
    refs_by_unit = {unit.path: [] for unit in units}
    for unit in units:
        own_names = set(unit.names)
        for file_path in source_files_for_unit(root, unit):
            is_dep_file = file_path.name in _DEPENDENCY_FILE_NAMES
            is_python = file_path.suffix == ".py"
            if not is_dep_file and not is_python:
                continue
            try:
                source = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            lines = source.splitlines()
            rel = file_path.resolve().relative_to(root.resolve()).as_posix()
            if is_dep_file:
                candidates = _production_import_candidates_for_file(
                    root, file_path, source, lines
                )
            else:
                if _is_test_file(file_path, root):
                    continue
                candidates = _python_external_import_candidates(source, lines)
            for module_name, line_no, line in candidates:
                key = external_key(module_name)
                if not key:
                    continue
                if key in RUST_DEFAULT_DIRECT_CRATES:
                    continue
                if file_path.name == "go.mod" and any(
                    module_name == mod or module_name.startswith(mod + "/")
                    for mod in GO_DEFAULT_DIRECT_MODULES
                ):
                    continue
                if file_path.suffix == ".py" and key in PYTHON_DEFAULT_DIRECT_PACKAGES:
                    continue
                if file_path.name == "package.json" and key in JS_DEFAULT_DIRECT_PACKAGES:
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
            for module_name, line_no, line in _production_import_candidates_for_file(
                root, file_path, source, lines
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
                r"\bstd::env::(?:var|var_os|vars|vars_os)\b",
                "std::env.read",
            ),
            (r"\benv::(?:var|var_os|vars|vars_os)\b", "std::env.read"),
            (
                r"\bstd::env::(?:set_var|remove_var)\b",
                "std::env.mutate",
            ),
            (r"\benv::(?:set_var|remove_var)\b", "std::env.mutate"),
        ),
        ".go": (
            (
                r"\bos\.(?:Getenv|LookupEnv|Environ)\s*\(",
                "os.env.read",
            ),
            (
                r"\bos\.(?:Setenv|Unsetenv|Clearenv)\s*\(",
                "os.env.mutate",
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
            for module_name, line_no, line in _production_config_candidates_for_file(
                root, file_path, source, lines
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
        if is_config_runtime_owner(src) or set(src.roles) & SIDE_PATH_ROLES:
            continue
        for ref in refs:
            result.violations.append(
                f"{ref.source_path}:{ref.line_no}: {src.path} ({'+'.join(src.roles)}) must not use config/env ingress package/API {ref.target_name!r}; "
                "deployment config/env ingress belongs to a run_kit unit in axis 'config'"
            )


def check_external_refs(
    root: Path, units: Sequence[Unit], result: CheckResult
) -> None:
    owners = external_owners(units)
    units_by_path = {unit.path: unit for unit in units}
    for src_path, refs in scan_external_refs(root, units).items():
        src = units_by_path[src_path]
        if set(src.roles) & SIDE_PATH_ROLES:
            continue
        for ref in refs:
            owner_list = owners.get(ref.target_name)
            if owner_list is None:
                if is_config_ingress_module(ref.target_name):
                    guidance = "bind it to a run_kit unit in axis 'config' via externals"
                else:
                    guidance = "bind direct-call generic operational packages to run_kit, or concrete external capability mechanisms to effect_tool/effect_facility, via externals"
                result.violations.append(
                    f"{ref.source_path}:{ref.line_no}: {src.path} uses unregistered external dependency {ref.target_name!r}; "
                    f"globally direct Rust crates are {', '.join(sorted(RUST_DEFAULT_DIRECT_CRATES))}; otherwise {guidance}"
                )
                continue
            if not any(o.path == src.path for o in owner_list):
                bound_to = ", ".join(sorted(set(o.path for o in owner_list)))
                result.violations.append(
                    f"{ref.source_path}:{ref.line_no}: {src.path} must not import external dependency {ref.target_name!r}; "
                    f"it is bound to {bound_to}, import that adapter/facility/config runtime helper instead"
                )


def _meaning_role_imports_utils(src: Unit, dst: Unit) -> bool:
    return bool(set(src.roles) & {"meaning_seed", "meaning_core"}) and set(dst.roles) == {"utils"}


_EFFECTFUL_UTIL_PATTERNS_BY_SUFFIX: dict[str, tuple[tuple[str, str], ...]] = {
    ".py": (
        (r"\bopen\s*\(", "open"),
        (r"\bsubprocess\.", "subprocess"),
        (r"\bsocket\.", "socket"),
        (r"\bos\.(?:environ|getenv|system|popen|remove|unlink|rename|replace)\b", "os effect/env"),
        (r"\bPath\([^)]*\)\.(?:read_text|write_text|read_bytes|write_bytes|open|unlink|rename)\b", "path I/O"),
    ),
    ".rs": (
        (r"\bstd::(?:fs|net|process|env)::", "std effect/env"),
        (r"\btokio::(?:fs|net|process)::", "tokio effect"),
    ),
    ".js": ((r"\b(?:process\.env|fetch\s*\(|fs\.)", "js effect/env"),),
    ".jsx": ((r"\b(?:process\.env|fetch\s*\(|fs\.)", "js effect/env"),),
    ".ts": ((r"\b(?:process\.env|fetch\s*\(|fs\.)", "js effect/env"),),
    ".tsx": ((r"\b(?:process\.env|fetch\s*\(|fs\.)", "js effect/env"),),
    ".go": ((r"\b(?:os|net|http|exec)\.", "go effect/env"),),
    ".java": ((r"\b(?:System\.getenv|Files\.|Socket|HttpClient|ProcessBuilder)\b", "jvm effect/env"),),
    ".kt": ((r"\b(?:System\.getenv|Files\.|Socket|HttpClient|ProcessBuilder)\b", "jvm effect/env"),),
    ".swift": ((r"\b(?:ProcessInfo\.processInfo\.environment|FileManager|URLSession)\b", "swift effect/env"),),
    ".cs": ((r"\b(?:Environment\.GetEnvironmentVariable|File\.|Directory\.|HttpClient|Process)\b", "csharp effect/env"),),
}

DIRECT_EFFECT_RESTRICTED_ROLES = {
    "utils",
    "meaning_seed",
    "meaning_core",
    "axis_link",
    "use_flow",
    "link_proto",
}


def _python_effectful_source_refs(
    source: str, lines: Sequence[str], rel: str
) -> list[ImportRef]:
    refs: list[ImportRef] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return refs
    os_effect_attrs = {
        "environ",
        "getenv",
        "system",
        "popen",
        "remove",
        "unlink",
        "rename",
        "replace",
    }
    path_io_attrs = {
        "read_text",
        "write_text",
        "read_bytes",
        "write_bytes",
        "open",
        "unlink",
        "rename",
    }

    def is_path_constructor(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "Path"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return node.value.id == "pathlib" and node.attr == "Path"
        return False

    seen: set[tuple[int, str]] = set()

    def add(node: ast.AST, label: str) -> None:
        line_no = getattr(node, "lineno", 0) or 0
        key = (line_no, label)
        if key not in seen:
            seen.add(key)
            refs.append(ImportRef(rel, label, line_no, _line_at(lines, line_no)))

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr == "environ"
        ):
            add(node, "os effect/env")
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            add(node, "open")
            continue
        if not isinstance(func, ast.Attribute):
            continue
        if isinstance(func.value, ast.Name):
            base = func.value.id
            if base == "subprocess":
                add(node, "subprocess")
                continue
            if base == "socket":
                add(node, "socket")
                continue
            if base == "os" and func.attr in os_effect_attrs:
                add(node, "os effect/env")
                continue
        if (
            func.attr in path_io_attrs
            and isinstance(func.value, ast.Call)
            and is_path_constructor(func.value.func)
        ):
            add(node, "path I/O")
    return refs


def _effectful_source_refs(root: Path, unit: Unit) -> list[ImportRef]:
    refs: list[ImportRef] = []
    for file_path in source_files_for_unit(root, unit):
        if file_path.name in _DEPENDENCY_FILE_NAMES or _is_test_file(file_path, root):
            continue
        try:
            source = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lines = source.splitlines()
        test_lines = _rust_test_lines_for_file(file_path, source)
        rel = file_path.resolve().relative_to(root.resolve()).as_posix()
        if file_path.suffix == ".py":
            refs.extend(_python_effectful_source_refs(source, lines, rel))
        else:
            clean = _strip_c_like_comments(source)
            rust_aliases: dict[str, str] = {}
            if file_path.suffix == ".rs":
                for module in ("fs", "net", "process", "env"):
                    if re.search(rf"\buse\s+(?:std|tokio)::(?:{{[^}}]*\b{module}\b|{module}\b)", clean):
                        rust_aliases[module] = "std/tokio effect/env"
            seen_effects: set[tuple[int, str]] = set()
            for line_no, line in enumerate(clean.splitlines(), start=1):
                if line_no in test_lines:
                    continue
                for pattern, label in _EFFECTFUL_UTIL_PATTERNS_BY_SUFFIX.get(file_path.suffix, ()): 
                    if re.search(pattern, line):
                        key = (line_no, label)
                        if key not in seen_effects:
                            seen_effects.add(key)
                            refs.append(
                                ImportRef(rel, label, line_no, _line_at(lines, line_no))
                            )
                for alias, label in rust_aliases.items():
                    if re.search(rf"\b{alias}::", line):
                        key = (line_no, label)
                        if key not in seen_effects:
                            seen_effects.add(key)
                            refs.append(
                                ImportRef(rel, label, line_no, _line_at(lines, line_no))
                            )
        for module_name, line_no, line in config_usage_candidates_for_file(file_path, source, lines):
            if line_no not in test_lines:
                refs.append(ImportRef(rel, module_name, line_no, line))
    return refs


def utils_meaning_safety_violations(root: Path, unit: Unit) -> list[ImportRef]:
    refs: list[ImportRef] = []
    for dep in unit.externals:
        if external_key(dep) not in RUST_DEFAULT_DIRECT_CRATES:
            refs.append(ImportRef(unit.path, external_key(dep), 0, "externals"))
    refs.extend(_effectful_source_refs(root, unit))
    return refs


def check_direct_effect_refs(
    root: Path, units: Sequence[Unit], result: CheckResult
) -> None:
    for unit in units:
        roles = set(unit.roles)
        if not (roles & DIRECT_EFFECT_RESTRICTED_ROLES):
            continue
        if roles & SIDE_PATH_ROLES:
            continue
        for ref in _effectful_source_refs(root, unit):
            result.violations.append(
                f"{ref.source_path}:{ref.line_no}: {unit.path} ({'+'.join(unit.roles)}) must not use raw effect/config/env API {ref.target_name!r}; "
                "move direct-call generic operational mechanism to run_kit, concrete external capability mechanism to effect_tool/effect_facility, or config/env ingress to run_kit axis 'config'"
            )


def check_meaning_safe_utils_refs(
    root: Path,
    units: Sequence[Unit],
    units_by_path: Mapping[str, Unit],
    name_to_unit: Mapping[str, Unit],
    result: CheckResult,
) -> None:
    safety_cache: dict[str, list[ImportRef]] = {}
    for src_path, refs in scan_imports(root, units).items():
        src = units_by_path[src_path]
        for ref in refs:
            dst = name_to_unit.get(ref.target_name)
            if dst is None or not _meaning_role_imports_utils(src, dst):
                continue
            safety_refs = safety_cache.setdefault(
                dst.path, utils_meaning_safety_violations(root, dst)
            )
            for safety_ref in safety_refs[:5]:
                result.violations.append(
                    f"{ref.source_path}:{ref.line_no}: {src.path} ({'+'.join(src.roles)}) may import utils only for meaning-safe types/pure functions; "
                    f"{dst.path} has effectful/general utility evidence {safety_ref.target_name!r} at {safety_ref.source_path}:{safety_ref.line_no}: {safety_ref.line}"
                )
            if len(safety_refs) > 5:
                result.violations.append(
                    f"{ref.source_path}:{ref.line_no}: {dst.path} has {len(safety_refs) - 5} more effectful/general utility evidence item(s); split axisless-safe helpers from general utilities"
                )



def check_axis_link_direction_notes(
    root: Path,
    units: Sequence[Unit],
    units_by_path: Mapping[str, Unit],
    name_to_unit: Mapping[str, Unit],
    result: CheckResult,
) -> None:
    """Add notes or strict findings for directed axis_link usage."""
    imports_by_src = scan_imports(root, units)
    importers_by_link: dict[str, set[str]] = {}
    meaning_refs_by_link: dict[str, set[str]] = {}
    for src_path, refs in imports_by_src.items():
        src = units_by_path[src_path]
        for ref in refs:
            dst = name_to_unit.get(ref.target_name)
            if dst is None or dst.path == src.path:
                continue
            if set(dst.roles) == {"axis_link"}:
                importers_by_link.setdefault(dst.path, set()).add(src.path)
            if set(src.roles) == {"axis_link"} and set(dst.roles) == {"meaning_core"}:
                meaning_refs_by_link.setdefault(src.path, set()).add(dst.path)

    for link in units:
        if set(link.roles) != {"axis_link"}:
            continue
        importers = importers_by_link.get(link.path, set())
        consumer_importers: set[str] = set()
        provider_importers: set[str] = set()
        meaning_imports = meaning_refs_by_link.get(link.path, set())
        for importer_path in importers:
            importer = units_by_path.get(importer_path)
            if importer is None:
                continue
            importer_axis = effective_axis(importer, units_by_path)
            if axis_link_consumer(link) in importer_axis:
                consumer_importers.add(importer_path)
            if axis_link_provider(link) in importer_axis:
                provider_importers.add(importer_path)

        if is_axis_link_required_port(link):
            if meaning_imports:
                result.violations.append(
                    f"{link.path}: required_port must not import provider meaning_core or implementation internals; implement the port from a provider-axis adapter/use_flow/effect owner instead"
                )
            if link.consumer_axis and not consumer_importers:
                result.notes.append(
                    f"{link.path}: required_port has no registered caller/core consumer-axis importer. If the caller is outside the scan, document it; otherwise remove the port."
                )
            if link.provider_axis and not provider_importers:
                result.notes.append(
                    f"{link.path}: required_port has no registered provider-axis implementation importer. If implementations are outside the scan, document them; otherwise this port is unused."
                )
            continue

        if link.consumer_axis and not consumer_importers:
            result.notes.append(
                f"{link.path}: provider-export axis_link has no registered consumer-axis importer. If the consumer is outside the scan, document it; otherwise remove the link or keep the capability inside same-axis meaning_core."
            )
        if link.provider_axis and not meaning_imports:
            result.violations.append(
                f"{link.path}: provider-export axis_link must import provider-axis meaning_core; use meaning_seed for shared vocabulary and link_proto for wire/schema contracts"
            )
        if provider_importers and not consumer_importers:
            result.notes.append(
                f"{link.path}: provider-export axis_link is currently used only from provider-axis units. A directed link should exist for cross-axis consumption; otherwise keep the API inside meaning_core."
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
    check_external_refs(root, units, result)
    check_direct_effect_refs(root, units, result)
    name_to_unit = {name: unit for unit in units for name in unit.names}
    check_meaning_safe_utils_refs(root, units, units_by_path, name_to_unit, result)
    check_axis_link_direction_notes(root, units, units_by_path, name_to_unit, result)
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

if __name__ == "__main__":
    print(
        "fcis_rules.py is the FCIS rule-engine library, not the project CLI. "
        "Use fcis_project.py check --root <repo-root> instead.",
        file=sys.stderr,
    )
    raise SystemExit(2)
