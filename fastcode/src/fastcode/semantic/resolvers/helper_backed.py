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

import logging
import posixpath
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from ...ir.element import CodeElement
from ...ir.types import IRCodeUnit, IRRelation, IRSnapshot, IRUnitSupport
from ..contracts import SemanticGraphContext
from ..resolution import (
    ResolutionPatch,
    ResolutionTier,
    SemanticCapability,
    SemanticResolver,
    ToolDiagnostic,
)
from ._helper_operations import SemanticHelperOperations, SemanticHelperSpec
from ._resolver_support import _hash_id, _normalize_path
from .graph_backed import GraphBackedSemanticResolver

logger = logging.getLogger(__name__)


def _unit_simple_name(unit: IRCodeUnit) -> str:
    name = unit.display_name or unit.qualified_name or ""
    for separator in ("::", ".", "#"):
        if separator in name:
            name = name.rsplit(separator, 1)[-1]
    return name.strip()


class HelperBackedSemanticResolver(SemanticResolver):
    """Shared base class for helper-backed semantic resolvers."""

    helper_cache_version = "helper_semantic_cache.v1"
    helper_filename: str = ""
    helper_runtime: str = "python"
    helper_timeout_seconds: int = 90
    file_extensions: tuple[str, ...] = ()
    extractor_name: str

    def __init__(self, fallback: GraphBackedSemanticResolver | None = None) -> None:
        self._fallback = fallback
        self._helper_ops = SemanticHelperOperations()

    def set_helper_runtime(self, helper_runtime: Any | None) -> None:
        """Inject the shell-side runtime used for helper execution."""
        self.set_tool_runtime(helper_runtime)
        self._helper_ops.set_runtime(helper_runtime)
        if self._fallback is not None:
            fallback_setter = getattr(self._fallback, "set_tool_runtime", None)
            if callable(fallback_setter):
                fallback_setter(helper_runtime)

    def _helper_spec(self) -> SemanticHelperSpec:
        return SemanticHelperSpec(
            cache_version=self.helper_cache_version,
            language=self.language,
            source_name=self.source_name,
            frontend_kind=self.frontend_kind,
            extractor_name=self.extractor_name,
            required_tools=self.required_tools,
            helper_filename=self.helper_filename,
            helper_runtime=self.helper_runtime,
            helper_timeout_seconds=self.helper_timeout_seconds,
            file_extensions=self.file_extensions,
        )

    def _target_files(self, target_paths: set[str], *, repo_root: str) -> list[str]:
        return self._helper_ops.target_files(
            target_paths,
            repo_root=repo_root,
            spec=self._helper_spec(),
        )

    def _helper_path(self) -> Path:
        return self._helper_ops.helper_path(self._helper_spec())

    def _helper_command(self, helper_files: list[str]) -> list[str]:
        return self._helper_ops.helper_command(self._helper_spec(), helper_files)

    def _run_semantic_helper(
        self,
        helper_files: list[str],
        patch: ResolutionPatch,
        *,
        repo_root: str,
    ) -> dict[str, Any]:
        invocation = self._helper_ops.run_helper(
            self._helper_spec(),
            helper_files,
            repo_root=repo_root,
        )
        patch.stats.update(invocation.stats)
        patch.warnings.extend(invocation.warnings)
        patch.diagnostics.extend(invocation.diagnostics)
        return invocation.payload

    def resolve(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
        graph_context: SemanticGraphContext | None,
    ) -> ResolutionPatch:
        snapshot_metadata = cast(dict[str, Any], snapshot.metadata or {})
        repo_root = self._helper_ops.resolve_repo_root(
            cast(str | None, snapshot_metadata.get("repo_root"))
        )
        if self._has_tools():
            return self._resolve_via_helper(
                snapshot=snapshot,
                elements=elements,
                target_paths=target_paths,
                graph_context=graph_context,
                repo_root=repo_root,
            )

        patch = self._fallback_patch(
            snapshot=snapshot,
            elements=elements,
            target_paths=target_paths,
            graph_context=graph_context,
        )
        patch.diagnostics.extend(self._missing_tool_diagnostics())
        return patch

    def _fallback_patch(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
        graph_context: SemanticGraphContext | None,
    ) -> ResolutionPatch:
        if self._fallback is not None:
            return self._fallback.resolve(
                snapshot=snapshot,
                elements=elements,
                target_paths=target_paths,
                graph_context=graph_context,
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
        return self._helper_ops.has_tools(self._helper_spec())

    def _missing_tool_diagnostics(self) -> list[ToolDiagnostic]:
        return self._helper_ops.missing_tool_diagnostics(self._helper_spec())

    def _resolve_via_helper(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
        graph_context: SemanticGraphContext | None,
        repo_root: str,
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

        spec = self._helper_spec()
        helper_files = self._target_files(target_paths, repo_root=repo_root)
        patch.stats["helper_target_files"] = len(helper_files)
        if not helper_files:
            return patch

        cache_entry = self._helper_ops.cache_entry(
            spec,
            helper_files,
            repo_root=repo_root,
        )
        patch.stats.update(
            {
                "helper_cache_key": cache_entry.key,
                "helper_cache_path": cache_entry.path,
                "helper_cache_target_files": len(helper_files),
            }
        )
        payload = self._helper_ops.load_cache(cache_entry)
        if payload is not None:
            patch.stats["helper_cache_hit"] = True
            patch.stats["helper_cache_miss"] = False
        else:
            patch.stats["helper_cache_hit"] = False
            patch.stats["helper_cache_miss"] = True
            payload = self._run_semantic_helper(
                helper_files,
                patch,
                repo_root=repo_root,
            )
        if self._helper_failed(patch):
            self._log_resolver_fallback(patch)
            return self._merge_with_fallback(
                primary_patch=patch,
                snapshot=snapshot,
                elements=elements,
                target_paths=target_paths,
                graph_context=graph_context,
            )
        if patch.stats.get("helper_cache_hit") is not True:
            self._helper_ops.save_cache(cache_entry, payload)
        self._apply_semantic_facts(snapshot=snapshot, patch=patch, payload=payload)
        patch.metadata_updates["semantic_resolver_runs"][0]["stats"] = patch.stats
        return patch

    @staticmethod
    def _helper_failed(patch: ResolutionPatch) -> bool:
        return any(
            diagnostic.code
            in {"tool_invocation_failed", "helper_nonzero_exit", "invalid_helper_json"}
            for diagnostic in patch.diagnostics
        )

    def _log_resolver_fallback(self, patch: ResolutionPatch) -> None:
        failure_codes = [
            diagnostic.code
            for diagnostic in patch.diagnostics
            if diagnostic.code
            in {"tool_invocation_failed", "helper_nonzero_exit", "invalid_helper_json"}
        ]
        logger.warning(
            "Semantic resolver fell back to structural resolution",
            extra={
                "fc_event": "resolver_fallback",
                "resolver_source": self.source_name,
                "language": self.language,
                "frontend_kind": self.frontend_kind,
                "resolution_tier": ResolutionTier.STRUCTURAL_FALLBACK,
                "helper_filename": self.helper_filename,
                "helper_failure_codes": failure_codes,
                "helper_exit_code": patch.stats.get("helper_exit_code"),
            },
        )

    def _merge_with_fallback(
        self,
        *,
        primary_patch: ResolutionPatch,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
        graph_context: SemanticGraphContext | None,
    ) -> ResolutionPatch:
        fallback_patch = self._fallback_patch(
            snapshot=snapshot,
            elements=elements,
            target_paths=target_paths,
            graph_context=graph_context,
        )
        fallback_patch.warnings = [*primary_patch.warnings, *fallback_patch.warnings]
        fallback_patch.diagnostics = [
            *primary_patch.diagnostics,
            *fallback_patch.diagnostics,
        ]
        fallback_patch.stats = {
            **fallback_patch.stats,
            "helper_target_files": primary_patch.stats.get("helper_target_files", 0),
            "helper_command": primary_patch.stats.get("helper_command", []),
            "helper_exit_code": primary_patch.stats.get("helper_exit_code"),
            "helper_failed": True,
            "helper_failure_codes": [
                diagnostic.code for diagnostic in primary_patch.diagnostics
            ],
        }
        resolver_runs = fallback_patch.metadata_updates.get("semantic_resolver_runs")
        if isinstance(resolver_runs, list) and resolver_runs:
            run = dict(cast(dict[str, Any], resolver_runs[0]))
            run["helper_backed"] = True
            run["helper_failed"] = True
            run["compiler_backed"] = False
            run["fallback"] = True
            run["stats"] = fallback_patch.stats
            resolver_runs[0] = run
        return fallback_patch

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
        candidates = [
            unit for unit in units_by_path.get(source_path, []) if unit.kind != "file"
        ]
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

        candidates = [
            unit for unit in units_by_path.get(target_path, []) if unit.kind != "file"
        ]
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
