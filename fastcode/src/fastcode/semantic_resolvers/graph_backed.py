"""Shared graph-backed resolver implementation."""

from __future__ import annotations

import hashlib
import shutil
from typing import Any

from ..indexer import CodeElement
from ..semantic_ir import IRCodeUnit, IRRelation, IRSnapshot, IRUnitSupport
from .base import ResolutionPatch, ResolutionTier, SemanticResolver, ToolDiagnostic


def _hash_id(prefix: str, payload: str) -> str:
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()
    return f"{prefix}:{digest}"


class GraphBackedSemanticResolver(SemanticResolver):
    """Resolver that upgrades canonical IR from compatibility graph edges."""

    source_name: str
    extractor_name: str
    graph_specs: tuple[tuple[str, str], ...] = (
        ("import", "dependency_graph"),
        ("inherit", "inheritance_graph"),
        ("call", "call_graph"),
    )
    frontend_kind = "graph_backed_ast"
    required_tools: tuple[str, ...] = ()

    def _missing_tool_diagnostics(self) -> list[ToolDiagnostic]:
        return [
            ToolDiagnostic(
                language=self.language,
                tool=tool,
                code="required_tool_missing",
                message=(
                    f"{tool} was not found in PATH; {self.language} semantic "
                    "resolver will use existing structural graph evidence only."
                ),
            )
            for tool in self.required_tools
            if shutil.which(tool) is None
        ]

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
        if legacy_graph_builder is None:
            return ResolutionPatch(
                warnings=[
                    f"{self.language}_resolver_skipped: legacy graph unavailable"
                ],
                stats={"language": self.language, "skipped": True},
            )

        element_by_id = {str(elem.id): elem for elem in elements}
        canonical_by_element_id = self._build_canonical_element_index(
            snapshot, elements
        )
        unit_by_id = {unit.unit_id: unit for unit in snapshot.units}
        file_units_by_path = {
            unit.path: unit.unit_id for unit in snapshot.units if unit.kind == "file"
        }
        patch = ResolutionPatch(
            metadata_updates={
                "semantic_resolver_runs": [
                    {
                        "language": self.language,
                        "source": self.source_name,
                        "capabilities": sorted(self.capabilities),
                        "frontend_kind": self.frontend_kind,
                        "required_tools": list(self.required_tools),
                    }
                ]
            },
            resolution_tier=ResolutionTier.STRUCTURAL_FALLBACK,
        )
        missing_diagnostics = self._missing_tool_diagnostics()
        patch.diagnostics.extend(missing_diagnostics)
        # When required tools are missing, record pending capabilities on
        # emitted relations so the system is honest about uncertainty.
        pending_caps: set[str] = set()
        if missing_diagnostics:
            pending_caps = set(self.capabilities)
        relation_counts = {"import": 0, "inherit": 0, "call": 0}
        skipped_edges = 0

        for relation_type, graph_name in self.graph_specs:
            graph = getattr(legacy_graph_builder, graph_name, None)
            if graph is None:
                continue
            for src_id, dst_id, data in graph.edges(data=True):
                emitted = self._emit_relation(
                    snapshot=snapshot,
                    unit_by_id=unit_by_id,
                    canonical_by_element_id=canonical_by_element_id,
                    element_by_id=element_by_id,
                    target_paths=target_paths,
                    source_element_id=str(src_id),
                    target_element_id=str(dst_id),
                    relation_type=relation_type,
                    payload=dict(data or {}),
                    file_units_by_path=file_units_by_path,
                    pending_caps=pending_caps,
                )
                if emitted is None:
                    skipped_edges += 1
                    continue
                support, relation = emitted
                patch.supports.append(support)
                patch.relations.append(relation)
                relation_counts[relation_type] += 1

        patch.stats.update(
            {
                "language": self.language,
                "capabilities": sorted(self.capabilities),
                "cost_class": self.cost_class,
                "resolver_source": self.source_name,
                "frontend_kind": self.frontend_kind,
                "required_tools": list(self.required_tools),
                "diagnostics": [d.to_dict() for d in patch.diagnostics],
                "relations_emitted": relation_counts,
                "supports_emitted": len(patch.supports),
                "skipped_edges": skipped_edges,
            }
        )
        patch.metadata_updates["semantic_resolver_runs"][0]["stats"] = patch.stats
        return patch

    def _build_canonical_element_index(
        self, snapshot: IRSnapshot, elements: list[CodeElement]
    ) -> dict[str, str]:
        canonical_by_element_id: dict[str, str] = {}
        file_units_by_path = {
            unit.path: unit.unit_id for unit in snapshot.units if unit.kind == "file"
        }

        for unit in snapshot.units:
            ast_element_id = (unit.metadata or {}).get("ast_element_id")
            if ast_element_id:
                canonical_by_element_id[str(ast_element_id)] = unit.unit_id

        for elem in elements:
            if elem.type != "file":
                continue
            rel_path = elem.relative_path or elem.file_path
            file_unit_id = file_units_by_path.get(rel_path)
            if file_unit_id:
                canonical_by_element_id[str(elem.id)] = file_unit_id

        return canonical_by_element_id

    def _emit_relation(
        self,
        *,
        snapshot: IRSnapshot,
        unit_by_id: dict[str, IRCodeUnit],
        canonical_by_element_id: dict[str, str],
        element_by_id: dict[str, CodeElement],
        target_paths: set[str],
        source_element_id: str,
        target_element_id: str,
        relation_type: str,
        payload: dict[str, Any],
        file_units_by_path: dict[str, str],
        pending_caps: set[str] | None = None,
    ) -> tuple[IRUnitSupport, IRRelation] | None:
        source_elem = element_by_id.get(source_element_id)
        target_elem = element_by_id.get(target_element_id)
        if source_elem is None or target_elem is None:
            return None

        source_path = source_elem.relative_path or source_elem.file_path
        if source_elem.language != self.language or source_path not in target_paths:
            return None

        src_unit_id = canonical_by_element_id.get(source_element_id)
        dst_unit_id = canonical_by_element_id.get(target_element_id)
        if not src_unit_id or not dst_unit_id:
            return None

        payload_key = self._payload_key(relation_type, payload, target_element_id)
        support_id = _hash_id(
            "support",
            (
                f"{snapshot.snapshot_id}:{self.source_name}:{relation_type}:"
                f"{source_element_id}:{target_element_id}:{payload_key}"
            ),
        )
        relation_id = _hash_id(
            "rel",
            (
                f"{snapshot.snapshot_id}:{self.source_name}:{relation_type}:"
                f"{src_unit_id}:{dst_unit_id}:{payload_key}"
            ),
        )

        target_unit = unit_by_id.get(dst_unit_id)
        display_name = (
            payload.get("call_name")
            or payload.get("base_name")
            or payload.get("module")
            or (target_unit.display_name if target_unit else None)
            or target_elem.name
        )
        metadata = payload | {
            "target_element_id": target_element_id,
            "target_unit_id": dst_unit_id,
            "source": self.source_name,
            "extractor": self.extractor_name,
            "resolver_language": self.language,
            "resolver_capabilities": sorted(self.capabilities),
        }
        doc_id = file_units_by_path.get(source_path)
        support = IRUnitSupport(
            support_id=support_id,
            unit_id=src_unit_id,
            source=self.source_name,
            support_kind=f"{relation_type}_resolution",
            external_id=target_element_id,
            path=source_path,
            display_name=str(display_name),
            qualified_name=(
                target_unit.qualified_name
                if target_unit is not None
                else (target_elem.metadata or {}).get("qualified_name")
            ),
            signature=target_unit.signature
            if target_unit is not None
            else target_elem.signature,
            start_line=source_elem.start_line,
            start_col=int((source_elem.metadata or {}).get("start_col") or 0),
            end_line=source_elem.end_line,
            end_col=int((source_elem.metadata or {}).get("end_col") or 0),
            metadata=metadata,
        )
        relation = IRRelation(
            relation_id=relation_id,
            src_unit_id=src_unit_id,
            dst_unit_id=dst_unit_id,
            relation_type=relation_type,
            resolution_state=(
                "anchored"
                if target_unit is not None and target_unit.primary_anchor_symbol_id
                else "structural"
            ),
            support_sources={self.source_name},
            support_ids=[support_id],
            pending_capabilities=pending_caps or set(),
            metadata=metadata
            | {
                "doc_id": doc_id,
                "resolution_tier": ResolutionTier.STRUCTURAL_FALLBACK,
            },
        )
        return support, relation

    @staticmethod
    def _payload_key(
        relation_type: str, payload: dict[str, Any], target_element_id: str
    ) -> str:
        if relation_type == "import":
            return str(payload.get("module") or target_element_id)
        if relation_type == "inherit":
            return str(payload.get("base_name") or target_element_id)
        return str(payload.get("call_name") or target_element_id)
