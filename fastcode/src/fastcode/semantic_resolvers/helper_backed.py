"""Shared helper-backed semantic resolver implementation.

This module provides a small, repeatable contract for language resolvers that
need more structure than the graph-backed fallback but are not yet wired to a
full compiler/LSP API inside Python.  Each resolver owns:

- language metadata and advertised capabilities
- tool availability checks
- helper command selection
- structured JSON fact mapping into canonical IR relations/supports

Helpers emit relative repo paths so snapshots remain portable across machines.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..indexer import CodeElement
from ..semantic_ir import IRCodeUnit, IRRelation, IRSnapshot, IRUnitSupport
from .base import (
    ResolutionPatch,
    ResolutionTier,
    SemanticCapability,
    SemanticResolver,
    ToolDiagnostic,
)
from .graph_backed import GraphBackedSemanticResolver


def _hash_id(prefix: str, payload: str) -> str:
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()
    return f"{prefix}:{digest}"


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return posixpath.normpath(normalized)


def _unit_simple_name(unit: IRCodeUnit) -> str:
    name = unit.display_name or unit.qualified_name or ""
    for separator in ("::", ".", "#"):
        if separator in name:
            name = name.rsplit(separator, 1)[-1]
    return name.strip()


class HelperBackedSemanticResolver(SemanticResolver):
    """Shared base class for helper-backed semantic resolvers."""

    helper_filename: str = ""
    helper_runtime: str = "python"
    helper_timeout_seconds: int = 90
    file_extensions: tuple[str, ...] = ()
    extractor_name: str

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        self._fallback = fallback

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
        legacy_graph_builder: Any,
    ) -> ResolutionPatch:
        if self._has_tools():
            return self._resolve_via_helper(snapshot=snapshot, target_paths=target_paths)

        patch = self._fallback_patch(
            snapshot=snapshot,
            elements=elements,
            target_paths=target_paths,
            legacy_graph_builder=legacy_graph_builder,
        )
        patch.diagnostics.extend(self._missing_tool_diagnostics())
        return patch

    def _fallback_patch(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
        legacy_graph_builder: Any,
    ) -> ResolutionPatch:
        if self._fallback is not None:
            return self._fallback.resolve(
                snapshot=snapshot,
                elements=elements,
                target_paths=target_paths,
                legacy_graph_builder=legacy_graph_builder,
            )

        return ResolutionPatch(
            metadata_updates={
                "semantic_resolver_runs": [
                    {
                        "language": self.language,
                        "source": self.source_name,
                        "frontend_kind": self.frontend_kind,
                        "fallback": True,
                    }
                ]
            },
            resolution_tier=ResolutionTier.STRUCTURAL_FALLBACK,
        )

    def _has_tools(self) -> bool:
        return all(shutil.which(tool) is not None for tool in self.required_tools)

    def _missing_tool_diagnostics(self) -> list[ToolDiagnostic]:
        return [
            ToolDiagnostic(
                language=self.language,
                tool=tool,
                code="required_tool_missing",
                message=(
                    f"'{tool}' not found in PATH; {self.language} resolution is "
                    "structural-only"
                ),
            )
            for tool in self.required_tools
            if shutil.which(tool) is None
        ]

    def _resolve_via_helper(
        self,
        *,
        snapshot: IRSnapshot,
        target_paths: set[str],
    ) -> ResolutionPatch:
        patch = ResolutionPatch(
            metadata_updates={
                "semantic_resolver_runs": [
                    {
                        "language": self.language,
                        "source": self.source_name,
                        "frontend_kind": self.frontend_kind,
                        "compiler_backed": True,
                        "helper_backed": True,
                    }
                ]
            },
            resolution_tier=ResolutionTier.COMPILER_CONFIRMED,
        )

        helper_files = self._target_files(target_paths)
        patch.stats["helper_target_files"] = len(helper_files)
        if not helper_files:
            return patch

        payload = self._run_semantic_helper(helper_files, patch)
        self._apply_semantic_facts(snapshot=snapshot, patch=patch, payload=payload)
        patch.metadata_updates["semantic_resolver_runs"][0]["stats"] = patch.stats
        return patch

    def _target_files(self, target_paths: set[str]) -> list[str]:
        repo_root = os.getcwd()
        files: list[str] = []
        for path in sorted(target_paths):
            normalized = path if os.path.isabs(path) else os.path.join(repo_root, path)
            if self.file_extensions and not normalized.endswith(self.file_extensions):
                continue
            files.append(os.path.abspath(normalized))
        return files

    def _helper_path(self) -> Path:
        return Path(__file__).with_name(self.helper_filename)

    def _helper_command(self, helper_files: list[str]) -> list[str]:
        helper_path = str(self._helper_path())
        if self.helper_runtime == "node":
            node_path = shutil.which("node") or "node"
            return [node_path, helper_path, *helper_files]
        if self.helper_runtime == "go":
            go_path = shutil.which("go") or "go"
            return [go_path, "run", helper_path, "--", *helper_files]
        return [sys.executable, helper_path, *helper_files]

    def _run_semantic_helper(
        self,
        helper_files: list[str],
        patch: ResolutionPatch,
    ) -> dict[str, Any]:
        command = self._helper_command(helper_files)
        patch.stats["helper_command"] = command
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.helper_timeout_seconds,
                check=False,
                cwd=os.getcwd(),
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            patch.warnings.append(f"{self.source_name}_helper_failed: {exc}")
            patch.diagnostics.append(
                ToolDiagnostic(
                    language=self.language,
                    tool=self.helper_filename,
                    code="tool_invocation_failed",
                    message=str(exc),
                )
            )
            return {}

        patch.stats["helper_exit_code"] = result.returncode
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            if stderr:
                patch.warnings.append(f"{self.source_name}_helper_error: {stderr}")
            patch.diagnostics.append(
                ToolDiagnostic(
                    language=self.language,
                    tool=self.helper_filename,
                    code="helper_nonzero_exit",
                    message=stderr or f"helper exited with code {result.returncode}",
                )
            )
            return {}

        if not result.stdout.strip():
            return {}

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            patch.warnings.append(f"{self.source_name}_helper_invalid_json: {exc}")
            patch.diagnostics.append(
                ToolDiagnostic(
                    language=self.language,
                    tool=self.helper_filename,
                    code="invalid_helper_json",
                    message=str(exc),
                )
            )
            return {}

    def _apply_semantic_facts(
        self,
        *,
        snapshot: IRSnapshot,
        patch: ResolutionPatch,
        payload: dict[str, Any],
    ) -> None:
        file_units_by_path = {
            _normalize_path(unit.path): unit
            for unit in snapshot.units
            if unit.kind == "file" and unit.path
        }
        units_by_path: dict[str, list[IRCodeUnit]] = defaultdict(list)
        units_by_name: dict[tuple[str, str], list[IRCodeUnit]] = defaultdict(list)
        for unit in snapshot.units:
            normalized_path = _normalize_path(unit.path)
            units_by_path[normalized_path].append(unit)
            simple_name = _unit_simple_name(unit)
            if simple_name:
                units_by_name[(normalized_path, simple_name)].append(unit)

        relation_counts = {"import": 0, "call": 0, "inherit": 0, "type": 0}

        for fact in payload.get("imports", []):
            source_path = _normalize_path(str(fact.get("source_path") or ""))
            source_unit = file_units_by_path.get(source_path)
            target_unit = self._resolve_file_unit(
                fact=fact,
                file_units_by_path=file_units_by_path,
            )
            if source_unit is None or target_unit is None:
                continue
            support, relation = self._build_relation(
                snapshot=snapshot,
                source_unit=source_unit,
                target_unit=target_unit,
                relation_type="import",
                payload=fact,
            )
            patch.supports.append(support)
            patch.relations.append(relation)
            relation_counts["import"] += 1

        for fact in payload.get("calls", []):
            source_path = _normalize_path(str(fact.get("source_path") or ""))
            source_unit = self._resolve_source_unit(
                source_path=source_path,
                source_line=fact.get("source_line"),
                units_by_path=units_by_path,
                file_units_by_path=file_units_by_path,
            )
            target_unit = self._resolve_symbol_unit(
                fact=fact,
                units_by_path=units_by_path,
                units_by_name=units_by_name,
            )
            if source_unit is None or target_unit is None:
                continue
            support, relation = self._build_relation(
                snapshot=snapshot,
                source_unit=source_unit,
                target_unit=target_unit,
                relation_type="call",
                payload=fact,
            )
            patch.supports.append(support)
            patch.relations.append(relation)
            relation_counts["call"] += 1

        for fact in payload.get("inherits", []):
            source_unit = self._resolve_symbol_unit(
                fact={
                    "target_path": fact.get("source_path"),
                    "target_name": fact.get("source_name"),
                    "target_line": fact.get("source_line"),
                },
                units_by_path=units_by_path,
                units_by_name=units_by_name,
            )
            target_unit = self._resolve_symbol_unit(
                fact={
                    "target_path": fact.get("target_path"),
                    "target_name": fact.get("target_name") or fact.get("base_name"),
                    "target_line": fact.get("target_line"),
                },
                units_by_path=units_by_path,
                units_by_name=units_by_name,
            )
            if source_unit is None or target_unit is None:
                continue
            support, relation = self._build_relation(
                snapshot=snapshot,
                source_unit=source_unit,
                target_unit=target_unit,
                relation_type="inherit",
                payload=fact,
            )
            patch.supports.append(support)
            patch.relations.append(relation)
            relation_counts["inherit"] += 1

        patch.stats.update(
            {
                "language": self.language,
                "resolver_source": self.source_name,
                "frontend_kind": self.frontend_kind,
                "capabilities": sorted(self.capabilities),
                "relations_emitted": relation_counts,
                "supports_emitted": len(patch.supports),
                "helper_stats": payload.get("stats", {}),
            }
        )

    def _resolve_file_unit(
        self,
        *,
        fact: dict[str, Any],
        file_units_by_path: dict[str, IRCodeUnit],
    ) -> IRCodeUnit | None:
        for key in ("target_path", "module", "import_path", "import_name"):
            value = str(fact.get(key) or "").strip()
            if not value:
                continue
            target = self._resolve_path_candidate(
                value=value,
                file_units_by_path=file_units_by_path,
            )
            if target is not None:
                return target
        return None

    def _resolve_path_candidate(
        self,
        *,
        value: str,
        file_units_by_path: dict[str, IRCodeUnit],
    ) -> IRCodeUnit | None:
        normalized = _normalize_path(value)
        if normalized in file_units_by_path:
            return file_units_by_path[normalized]

        suffix_matches = [
            unit
            for path, unit in file_units_by_path.items()
            if path.endswith(f"/{normalized}") or path == normalized
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]

        stem = posixpath.splitext(posixpath.basename(normalized))[0]
        if stem:
            stem_matches = [
                unit
                for path, unit in file_units_by_path.items()
                if posixpath.splitext(posixpath.basename(path))[0] == stem
            ]
            if len(stem_matches) == 1:
                return stem_matches[0]
        return None

    def _resolve_source_unit(
        self,
        *,
        source_path: str,
        source_line: Any,
        units_by_path: dict[str, list[IRCodeUnit]],
        file_units_by_path: dict[str, IRCodeUnit],
    ) -> IRCodeUnit | None:
        candidates = [unit for unit in units_by_path.get(source_path, []) if unit.kind != "file"]
        if not candidates:
            return file_units_by_path.get(source_path)

        try:
            line = int(source_line)
        except (TypeError, ValueError):
            line = 0

        containing = [
            unit
            for unit in candidates
            if unit.start_line is not None
            and unit.end_line is not None
            and unit.start_line <= line <= unit.end_line
        ]
        if containing:
            return sorted(
                containing,
                key=lambda unit: (
                    (unit.end_line or 0) - (unit.start_line or 0),
                    -int(unit.start_line or 0),
                    unit.unit_id,
                ),
            )[0]
        if len(candidates) == 1:
            return candidates[0]

        ordered = sorted(
            candidates,
            key=lambda unit: (
                unit.start_line or 0,
                unit.end_line or 0,
                unit.unit_id,
            ),
        )
        return ordered[0]

    def _resolve_symbol_unit(
        self,
        *,
        fact: dict[str, Any],
        units_by_path: dict[str, list[IRCodeUnit]],
        units_by_name: dict[tuple[str, str], list[IRCodeUnit]],
    ) -> IRCodeUnit | None:
        target_path = _normalize_path(str(fact.get("target_path") or ""))
        target_name = str(
            fact.get("target_name")
            or fact.get("call_name")
            or fact.get("base_name")
            or ""
        ).strip()
        if not target_path:
            return None

        candidates = [unit for unit in units_by_path.get(target_path, []) if unit.kind != "file"]
        if target_name:
            named = units_by_name.get((target_path, target_name), [])
            if len(named) == 1:
                return named[0]
            if named:
                return self._pick_best_symbol_match(named, fact)
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            return self._pick_best_symbol_match(candidates, fact)
        return None

    @staticmethod
    def _pick_best_symbol_match(
        candidates: list[IRCodeUnit], fact: dict[str, Any]
    ) -> IRCodeUnit:
        try:
            target_line = int(fact.get("target_line") or 0)
        except (TypeError, ValueError):
            target_line = 0

        return sorted(
            candidates,
            key=lambda unit: (
                abs((unit.start_line or 0) - target_line),
                unit.unit_id,
            ),
        )[0]

    def _build_relation(
        self,
        *,
        snapshot: IRSnapshot,
        source_unit: IRCodeUnit,
        target_unit: IRCodeUnit,
        relation_type: str,
        payload: dict[str, Any],
    ) -> tuple[IRUnitSupport, IRRelation]:
        payload_key = self._payload_key(relation_type, payload)
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
        metadata = dict(payload)
        metadata.update(
            {
                "source": self.source_name,
                "extractor": self.extractor_name,
                "resolver_language": self.language,
                "resolver_capabilities": sorted(self.capabilities),
                "target_unit_id": target_unit.unit_id,
                "resolution_tier": ResolutionTier.COMPILER_CONFIRMED,
                "semantic_capability": self._relation_capability(relation_type),
                "doc_id": f"doc:{snapshot.snapshot_id}:{source_unit.path}",
            }
        )
        support = IRUnitSupport(
            support_id=support_id,
            unit_id=source_unit.unit_id,
            source=self.source_name,
            support_kind=f"{relation_type}_resolution",
            external_id=target_unit.unit_id,
            path=source_unit.path,
            display_name=target_unit.display_name,
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
            resolution_state="semantically_resolved",
            support_sources={self.source_name},
            support_ids=[support_id],
            metadata=metadata,
        )
        return support, relation

    @staticmethod
    def _payload_key(relation_type: str, payload: dict[str, Any]) -> str:
        if relation_type == "import":
            return str(
                payload.get("module")
                or payload.get("import_path")
                or payload.get("import_name")
                or payload.get("target_path")
                or ""
            )
        if relation_type == "inherit":
            return str(
                payload.get("base_name")
                or payload.get("target_name")
                or payload.get("target_path")
                or ""
            )
        return str(
            payload.get("call_name")
            or payload.get("target_name")
            or payload.get("target_path")
            or ""
        )

    @staticmethod
    def _relation_capability(relation_type: str) -> str:
        return {
            "import": SemanticCapability.RESOLVE_IMPORTS,
            "call": SemanticCapability.RESOLVE_CALLS,
            "inherit": SemanticCapability.RESOLVE_INHERITANCE,
            "type": SemanticCapability.RESOLVE_TYPES,
        }.get(relation_type, SemanticCapability.RESOLVE_BINDINGS)
