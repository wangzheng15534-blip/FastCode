"""C/C++ structural semantic resolvers."""

from __future__ import annotations

import hashlib
import posixpath
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ..indexer import CodeElement
from ..semantic_ir import IRCodeUnit, IRRelation, IRSnapshot, IRUnitSupport
from .base import ResolutionPatch, SemanticResolver

HEADER_EXTENSIONS = (".h", ".hh", ".hpp", ".hxx", ".inl")
SOURCE_EXTENSIONS = (".c", ".cc", ".cpp", ".cxx")


def _hash_id(prefix: str, payload: str) -> str:
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()
    return f"{prefix}:{digest}"


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return posixpath.normpath(normalized)


def _normalize_base_name(value: str) -> str:
    name = re.sub(r"<.*?>", "", value)
    name = name.rsplit("::", 1)[-1]
    name = re.sub(r"\b(class|struct)\b", "", name)
    return name.strip()


@dataclass(frozen=True)
class IncludeResolution:
    unit: IRCodeUnit
    method: str


class CFamilySemanticResolver(SemanticResolver):
    """Resolver for include and inheritance upgrades."""

    source_name: str
    extractor_name: str
    capabilities = frozenset({"resolve_includes", "resolve_inheritance"})
    cost_class = "low"

    def applicable(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
    ) -> bool:
        del snapshot
        return any(
            elem.language == self.language
            and (elem.relative_path or elem.file_path) in target_paths
            for elem in elements
        )

    def resolve(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
        legacy_graph_builder: Any | None,
    ) -> ResolutionPatch:
        del legacy_graph_builder
        file_units_by_path = {
            unit.path: unit for unit in snapshot.units if unit.kind == "file"
        }
        unit_ids_by_element_id = {
            str((unit.metadata or {}).get("ast_element_id")): unit.unit_id
            for unit in snapshot.units
            if (unit.metadata or {}).get("ast_element_id")
        }
        unit_by_id = {unit.unit_id: unit for unit in snapshot.units}
        patch = ResolutionPatch(
            metadata_updates={
                "semantic_resolver_runs": [
                    {
                        "language": self.language,
                        "source": self.source_name,
                        "capabilities": sorted(self.capabilities),
                    }
                ]
            }
        )
        include_targets_by_path: dict[str, set[str]] = defaultdict(set)
        counts = {"import": 0, "inherit": 0}
        warnings: list[str] = []

        class_units_by_name = self._build_class_unit_index(snapshot)
        file_elements = [
            elem
            for elem in elements
            if elem.type == "file"
            and elem.language == self.language
            and (elem.relative_path or elem.file_path) in target_paths
        ]

        for elem in file_elements:
            source_path = elem.relative_path or elem.file_path
            source_unit = file_units_by_path.get(source_path)
            if source_unit is None:
                warnings.append(f"{self.language}_resolver_missing_file_unit:{source_path}")
                continue

            for import_info in elem.metadata.get("imports", []):
                module = str(import_info.get("module") or "").strip()
                if not module:
                    continue
                include_target = self._resolve_include_target(
                    module=module,
                    source_path=source_path,
                    file_units_by_path=file_units_by_path,
                )
                if include_target is None:
                    continue
                support, relation = self._build_relation(
                    snapshot=snapshot,
                    source_unit=source_unit,
                    target_unit=include_target.unit,
                    relation_type="import",
                    payload={
                        "module": module,
                        "level": int(import_info.get("level") or 0),
                        "resolution_method": include_target.method,
                    },
                )
                patch.supports.append(support)
                patch.relations.append(relation)
                include_targets_by_path[source_path].add(include_target.unit.path)
                counts["import"] += 1

        for elem in elements:
            source_path = elem.relative_path or elem.file_path
            if (
                elem.type != "class"
                or elem.language != self.language
                or source_path not in target_paths
            ):
                continue
            source_unit_id = unit_ids_by_element_id.get(str(elem.id))
            source_unit = unit_by_id.get(source_unit_id) if source_unit_id else None
            if source_unit is None:
                warnings.append(f"{self.language}_resolver_missing_class_unit:{elem.id}")
                continue

            for raw_base in elem.metadata.get("bases", []):
                base_name = _normalize_base_name(str(raw_base))
                if not base_name:
                    continue
                resolved = self._resolve_base_class(
                    base_name=base_name,
                    source_unit=source_unit,
                    included_paths=include_targets_by_path.get(source_path, set()),
                    class_units_by_name=class_units_by_name,
                )
                if resolved is None:
                    continue
                support, relation = self._build_relation(
                    snapshot=snapshot,
                    source_unit=source_unit,
                    target_unit=resolved.unit,
                    relation_type="inherit",
                    payload={
                        "base_name": base_name,
                        "resolution_method": resolved.method,
                    },
                )
                patch.supports.append(support)
                patch.relations.append(relation)
                counts["inherit"] += 1

        patch.warnings.extend(warnings)
        patch.stats.update(
            {
                "language": self.language,
                "capabilities": sorted(self.capabilities),
                "cost_class": self.cost_class,
                "resolver_source": self.source_name,
                "relations_emitted": counts,
                "supports_emitted": len(patch.supports),
            }
        )
        patch.metadata_updates["semantic_resolver_runs"][0]["stats"] = patch.stats
        return patch

    def _resolve_include_target(
        self,
        *,
        module: str,
        source_path: str,
        file_units_by_path: dict[str, IRCodeUnit],
    ) -> IncludeResolution | None:
        normalized_module = _normalize_path(module)
        source_dir = posixpath.dirname(source_path)
        relative_guess = _normalize_path(posixpath.join(source_dir, normalized_module))

        for candidate_path, method in (
            (relative_guess, "relative_include_exact"),
            (normalized_module, "include_exact"),
        ):
            if candidate_path in file_units_by_path:
                return IncludeResolution(file_units_by_path[candidate_path], method)

        suffix_matches = [
            path
            for path in file_units_by_path
            if path == normalized_module or path.endswith(f"/{normalized_module}")
        ]
        if len(suffix_matches) == 1:
            path = suffix_matches[0]
            return IncludeResolution(file_units_by_path[path], "include_suffix_match")

        stem = posixpath.splitext(posixpath.basename(normalized_module))[0]
        ext = posixpath.splitext(normalized_module)[1]
        basename_matches = self._find_basename_matches(
            normalized_module=normalized_module,
            stem=stem,
            ext=ext,
            source_dir=source_dir,
            file_units_by_path=file_units_by_path,
        )
        if len(basename_matches) == 1:
            path = basename_matches[0]
            return IncludeResolution(file_units_by_path[path], "include_basename_match")
        return None

    def _find_basename_matches(
        self,
        *,
        normalized_module: str,
        stem: str,
        ext: str,
        source_dir: str,
        file_units_by_path: dict[str, IRCodeUnit],
    ) -> list[str]:
        matches: list[str] = []
        basename = posixpath.basename(normalized_module)
        for path in file_units_by_path:
            path_basename = posixpath.basename(path)
            path_stem, path_ext = posixpath.splitext(path_basename)
            if basename and path_basename == basename:
                matches.append(path)
                continue
            if stem and path_stem == stem:
                if not ext or path_ext in HEADER_EXTENSIONS + SOURCE_EXTENSIONS:
                    matches.append(path)
        if len(matches) <= 1:
            return matches
        same_dir_matches = [
            path for path in matches if posixpath.dirname(path) == source_dir
        ]
        return same_dir_matches or matches

    def _build_class_unit_index(
        self, snapshot: IRSnapshot
    ) -> dict[str, list[IRCodeUnit]]:
        class_units_by_name: dict[str, list[IRCodeUnit]] = defaultdict(list)
        for unit in snapshot.units:
            if unit.kind != "class":
                continue
            key = _normalize_base_name(unit.display_name or unit.qualified_name or "")
            if key:
                class_units_by_name[key].append(unit)
        return class_units_by_name

    def _resolve_base_class(
        self,
        *,
        base_name: str,
        source_unit: IRCodeUnit,
        included_paths: set[str],
        class_units_by_name: dict[str, list[IRCodeUnit]],
    ) -> IncludeResolution | None:
        candidates = class_units_by_name.get(base_name, [])
        if not candidates:
            return None

        scored: list[tuple[int, IRCodeUnit, str]] = []
        for candidate in candidates:
            if candidate.unit_id == source_unit.unit_id:
                continue
            if candidate.path == source_unit.path:
                score, method = 100, "same_file_name_match"
            elif candidate.path in included_paths:
                score, method = 90, "included_header_name_match"
            elif candidate.primary_anchor_symbol_id:
                score, method = 70, "anchored_name_match"
            else:
                score, method = 50, "name_match"
            scored.append((score, candidate, method))
        if not scored:
            return None
        scored.sort(
            key=lambda item: (
                item[0],
                1 if item[1].primary_anchor_symbol_id else 0,
                item[1].unit_id,
            ),
            reverse=True,
        )
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            return None
        _, unit, method = scored[0]
        return IncludeResolution(unit, method)

    def _build_relation(
        self,
        *,
        snapshot: IRSnapshot,
        source_unit: IRCodeUnit,
        target_unit: IRCodeUnit,
        relation_type: str,
        payload: dict[str, Any],
    ) -> tuple[IRUnitSupport, IRRelation]:
        payload_key = (
            str(payload.get("module"))
            if relation_type == "import"
            else str(payload.get("base_name"))
        )
        support_id = _hash_id(
            "support",
            (
                f"{snapshot.snapshot_id}:{self.source_name}:{relation_type}:"
                f"{source_unit.unit_id}:{target_unit.unit_id}:{payload_key}"
            ),
        )
        relation_id = _hash_id(
            "rel",
            (
                f"{snapshot.snapshot_id}:{self.source_name}:{relation_type}:"
                f"{source_unit.unit_id}:{target_unit.unit_id}:{payload_key}"
            ),
        )
        metadata = payload | {
            "source": self.source_name,
            "extractor": self.extractor_name,
            "resolver_language": self.language,
            "resolver_capabilities": sorted(self.capabilities),
            "target_unit_id": target_unit.unit_id,
        }
        support = IRUnitSupport(
            support_id=support_id,
            unit_id=source_unit.unit_id,
            source=self.source_name,
            support_kind=f"{relation_type}_resolution",
            external_id=target_unit.unit_id,
            path=source_unit.path,
            display_name=(
                target_unit.display_name
                if relation_type != "import"
                else str(payload.get("module") or target_unit.display_name)
            ),
            qualified_name=target_unit.qualified_name,
            signature=target_unit.signature,
            start_line=source_unit.start_line,
            start_col=source_unit.start_col,
            end_line=source_unit.end_line,
            end_col=source_unit.end_col,
            metadata=metadata,
        )
        relation = IRRelation(
            relation_id=relation_id,
            src_unit_id=source_unit.unit_id,
            dst_unit_id=target_unit.unit_id,
            relation_type=relation_type,
            resolution_state=(
                "anchored" if target_unit.primary_anchor_symbol_id else "structural"
            ),
            support_sources={self.source_name},
            support_ids=[support_id],
            metadata=metadata | {"doc_id": self._doc_id_for_path(snapshot, source_unit.path)},
        )
        return support, relation

    @staticmethod
    def _doc_id_for_path(snapshot: IRSnapshot, path: str) -> str | None:
        for unit in snapshot.units:
            if unit.kind == "file" and unit.path == path:
                return unit.unit_id
        return None


class CSemanticResolver(CFamilySemanticResolver):
    language = "c"
    source_name = "c_resolver"
    extractor_name = "fastcode.semantic_resolvers.c"


class CppSemanticResolver(CFamilySemanticResolver):
    language = "cpp"
    source_name = "cpp_resolver"
    extractor_name = "fastcode.semantic_resolvers.cpp"
