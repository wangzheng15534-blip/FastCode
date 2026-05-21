"""DocKG-facing code status pack export helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from fastcode.ir.types import IRCodeUnit, IRRelation, IRSnapshot
from fastcode.utils.json import safe_jsonable

CODE_STATUS_PACK_SCHEMA = "code_status_pack.v0"
CODE_STATUS_EXTRACTION_VERSION = "fastcode.code_status_export.v0"


def build_code_status_pack(
    snapshot: IRSnapshot,
    *,
    artifact_key: str | None = None,
    manifest: Mapping[str, Any] | None = None,
    ir_graphs: Any | None = None,
    include_graph_facts: bool = True,
) -> dict[str, Any]:
    """Build a deterministic JSON-serializable code-status evidence pack."""

    file_units = [unit for unit in snapshot.units if unit.kind in {"file", "doc"}]
    source_files = _source_file_payloads(snapshot, file_units)
    source_id_by_path = {
        str(item["path"]): str(item["source_id"]) for item in source_files
    }
    unit_span_ids: dict[str, str] = {}
    support_span_ids: dict[str, str] = {}
    source_spans = _source_span_payloads(
        snapshot,
        source_id_by_path=source_id_by_path,
        unit_span_ids=unit_span_ids,
        support_span_ids=support_span_ids,
    )
    code_units = [
        _code_unit_payload(unit, source_span_id=unit_span_ids.get(unit.unit_id))
        for unit in sorted(snapshot.units, key=lambda item: item.unit_id)
    ]
    symbols = [
        _symbol_payload(unit, source_span_id=unit_span_ids.get(unit.unit_id))
        for unit in sorted(snapshot.units, key=lambda item: item.unit_id)
        if unit.kind not in {"file", "doc"}
    ]
    relation_facts = [
        _relation_payload(relation, support_span_ids=support_span_ids)
        for relation in sorted(snapshot.relations, key=lambda item: item.relation_id)
    ]
    graph_facts = (
        _graph_fact_payloads(ir_graphs) if include_graph_facts and ir_graphs else []
    )
    diagnostics = _diagnostics_payload(snapshot.metadata)
    pack: dict[str, Any] = {
        "schema_version": CODE_STATUS_PACK_SCHEMA,
        "extraction_version": CODE_STATUS_EXTRACTION_VERSION,
        "producer": {"name": "FastCode"},
        "repo": {
            "repo_name": snapshot.repo_name,
            "branch": snapshot.branch,
            "commit_id": snapshot.commit_id,
            "tree_id": snapshot.tree_id,
            "worktree_digest": _worktree_digest(snapshot, source_files),
        },
        "snapshot": {
            "snapshot_id": snapshot.snapshot_id,
            "artifact_key": artifact_key,
            "branch": snapshot.branch,
            "commit_id": snapshot.commit_id,
            "tree_id": snapshot.tree_id,
        },
        "source_files": source_files,
        "source_spans": source_spans,
        "code_units": code_units,
        "symbols": symbols,
        "relation_facts": relation_facts,
        "graph_facts": graph_facts,
        "diagnostics": diagnostics,
        "manifest": safe_jsonable(dict(manifest or {})),
        "counts": {
            "source_files": len(source_files),
            "source_spans": len(source_spans),
            "code_units": len(code_units),
            "symbols": len(symbols),
            "relation_facts": len(relation_facts),
            "graph_facts": len(graph_facts),
            "warnings": len(diagnostics["warnings"]),
            "resolver_diagnostics": len(diagnostics["resolver_diagnostics"]),
        },
    }
    pack["pack_digest"] = _sha256_digest(pack)
    return pack


def _stable_json(payload: Any) -> str:
    return json.dumps(
        safe_jsonable(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_digest(payload: Any) -> str:
    return f"sha256:{hashlib.sha256(_stable_json(payload).encode()).hexdigest()}"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    items = cast(Iterable[Any], value)
    return sorted(str(item) for item in items if item)


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, item in (value or {}).items():
        payload[str(key)] = item
    return cast(dict[str, Any], safe_jsonable(payload))


def _source_id(path: str) -> str:
    return f"source:{hashlib.sha256(path.encode()).hexdigest()[:16]}"


def _span_id(owner_kind: str, owner_id: str, path: str, start: Any, end: Any) -> str:
    payload = {
        "owner_kind": owner_kind,
        "owner_id": owner_id,
        "path": path,
        "start_line": start,
        "end_line": end,
    }
    return f"span:{hashlib.sha256(_stable_json(payload).encode()).hexdigest()[:20]}"


def _source_file_payloads(
    snapshot: IRSnapshot,
    file_units: Sequence[IRCodeUnit],
) -> list[dict[str, Any]]:
    document_by_path = {doc.path: doc for doc in snapshot.documents}
    paths = sorted(
        {unit.path for unit in file_units if unit.path} | set(document_by_path)
    )
    payloads: list[dict[str, Any]] = []
    file_unit_by_path = {unit.path: unit for unit in file_units if unit.path}
    for path in paths:
        unit = file_unit_by_path.get(path)
        document = document_by_path.get(path)
        metadata = _metadata(unit.metadata if unit is not None else {})
        content_hash = metadata.get("content_hash") or (
            document.content_hash if document is not None else None
        )
        blob_oid = (
            metadata.get("blob_oid")
            or metadata.get("git_blob_oid")
            or (document.blob_oid if document is not None else None)
        )
        payloads.append(
            {
                "source_id": _source_id(path),
                "path": path,
                "unit_id": unit.unit_id if unit is not None else None,
                "kind": unit.kind if unit is not None else "file",
                "language": (
                    unit.language
                    if unit is not None
                    else document.language
                    if document is not None
                    else ""
                ),
                "display_name": (
                    unit.display_name if unit is not None else path.rsplit("/", 1)[-1]
                ),
                "content_hash": content_hash,
                "blob_oid": blob_oid,
                "source_set": sorted(unit.source_set) if unit is not None else [],
                "metadata": metadata,
            }
        )
    return payloads


def _source_span_payloads(
    snapshot: IRSnapshot,
    *,
    source_id_by_path: Mapping[str, str],
    unit_span_ids: dict[str, str],
    support_span_ids: dict[str, str],
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for unit in sorted(snapshot.units, key=lambda item: item.unit_id):
        if unit.path not in source_id_by_path:
            continue
        span_id = _span_id(
            "unit",
            unit.unit_id,
            unit.path,
            unit.start_line,
            unit.end_line,
        )
        unit_span_ids[unit.unit_id] = span_id
        spans.append(
            {
                "source_span_id": span_id,
                "source_id": source_id_by_path[unit.path],
                "path": unit.path,
                "owner_kind": "unit",
                "owner_id": unit.unit_id,
                "role": unit.kind,
                "start_line": unit.start_line,
                "start_col": unit.start_col,
                "end_line": unit.end_line,
                "end_col": unit.end_col,
                "metadata": {},
            }
        )
    for support in sorted(snapshot.supports, key=lambda item: item.support_id):
        path = support.path or _unit_path(snapshot.units, support.unit_id)
        if not path or path not in source_id_by_path:
            continue
        span_id = _span_id(
            "support",
            support.support_id,
            path,
            support.start_line,
            support.end_line,
        )
        support_span_ids[support.support_id] = span_id
        spans.append(
            {
                "source_span_id": span_id,
                "source_id": source_id_by_path[path],
                "path": path,
                "owner_kind": "support",
                "owner_id": support.support_id,
                "unit_id": support.unit_id,
                "role": support.role or support.support_kind,
                "start_line": support.start_line,
                "start_col": support.start_col,
                "end_line": support.end_line,
                "end_col": support.end_col,
                "metadata": _metadata(support.metadata),
            }
        )
    return spans


def _unit_path(units: Iterable[IRCodeUnit], unit_id: str) -> str | None:
    for unit in units:
        if unit.unit_id == unit_id:
            return unit.path
    return None


def _code_unit_payload(
    unit: IRCodeUnit,
    *,
    source_span_id: str | None,
) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "kind": unit.kind,
        "path": unit.path,
        "language": unit.language,
        "display_name": unit.display_name,
        "qualified_name": unit.qualified_name,
        "signature": unit.signature,
        "docstring": unit.docstring,
        "summary": unit.summary,
        "parent_unit_id": unit.parent_unit_id,
        "primary_anchor_symbol_id": unit.primary_anchor_symbol_id,
        "anchor_symbol_ids": list(unit.anchor_symbol_ids),
        "candidate_anchor_symbol_ids": list(unit.candidate_anchor_symbol_ids),
        "anchor_coverage": unit.anchor_coverage,
        "source_span_id": source_span_id,
        "source_set": sorted(unit.source_set),
        "metadata": _metadata(unit.metadata),
    }


def _symbol_payload(
    unit: IRCodeUnit,
    *,
    source_span_id: str | None,
) -> dict[str, Any]:
    return {
        "symbol_id": unit.unit_id,
        "external_symbol_id": unit.primary_anchor_symbol_id,
        "path": unit.path,
        "display_name": unit.display_name,
        "kind": "documentation" if unit.kind == "doc" else unit.kind,
        "language": unit.language,
        "qualified_name": unit.qualified_name,
        "signature": unit.signature,
        "start_line": unit.start_line,
        "start_col": unit.start_col,
        "end_line": unit.end_line,
        "end_col": unit.end_col,
        "source_priority": unit.source_priority,
        "source_set": sorted(unit.source_set),
        "source_span_id": source_span_id,
        "metadata": _metadata(unit.metadata),
    }


def _relation_payload(
    relation: IRRelation,
    *,
    support_span_ids: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "relation_id": relation.relation_id,
        "src_unit_id": relation.src_unit_id,
        "dst_unit_id": relation.dst_unit_id,
        "relation_type": relation.relation_type,
        "resolution_state": relation.resolution_state,
        "support_sources": sorted(relation.support_sources),
        "support_ids": list(relation.support_ids),
        "support_span_ids": [
            support_span_ids[support_id]
            for support_id in relation.support_ids
            if support_id in support_span_ids
        ],
        "pending_capabilities": sorted(relation.pending_capabilities),
        "metadata": _metadata(relation.metadata),
    }


def _graph_fact_payloads(ir_graphs: Any) -> list[dict[str, Any]]:
    graph_attrs = (
        ("dependency", "dependency_graph"),
        ("call", "call_graph"),
        ("inheritance", "inheritance_graph"),
        ("reference", "reference_graph"),
        ("containment", "containment_graph"),
    )
    facts: list[dict[str, Any]] = []
    for graph_name, attr_name in graph_attrs:
        graph = getattr(ir_graphs, attr_name, None)
        if graph is None:
            continue
        edges = getattr(graph, "edges", None)
        if not callable(edges):
            continue
        try:
            edge_rows = cast(Iterable[Any], edges(data=True))
        except TypeError:
            edge_rows = cast(Iterable[Any], edges())
        for edge in edge_rows:
            if not isinstance(edge, tuple):
                continue
            edge_tuple = cast(tuple[Any, ...], edge)
            if len(edge_tuple) < 2:
                continue
            src = str(edge_tuple[0])
            dst = str(edge_tuple[1])
            attrs: Mapping[str, Any] = (
                cast(Mapping[str, Any], edge_tuple[2])
                if len(edge_tuple) > 2 and isinstance(edge_tuple[2], Mapping)
                else {}
            )
            facts.append(
                {
                    "graph": graph_name,
                    "src_unit_id": src,
                    "dst_unit_id": dst,
                    "relation_id": attrs.get("relation_id"),
                    "metadata": _metadata(attrs),
                }
            )
    return sorted(
        facts,
        key=lambda item: (
            str(item["graph"]),
            str(item["src_unit_id"]),
            str(item["dst_unit_id"]),
            str(item.get("relation_id") or ""),
        ),
    )


def _diagnostics_payload(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _metadata(metadata)
    return {
        "warnings": _string_list(payload.get("warnings")),
        "pipeline_layers": safe_jsonable(payload.get("pipeline_layers") or []),
        "pipeline_metrics": safe_jsonable(payload.get("pipeline_metrics") or {}),
        "semantic_resolver_runs": safe_jsonable(
            payload.get("semantic_resolver_runs") or []
        ),
        "resolver_diagnostics": safe_jsonable(
            payload.get("resolver_diagnostics")
            or payload.get("semantic_resolver_diagnostics")
            or []
        ),
        "scip_degraded_reasons": _string_list(payload.get("scip_degraded_reasons")),
    }


def _worktree_digest(
    snapshot: IRSnapshot,
    source_files: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "repo_name": snapshot.repo_name,
        "branch": snapshot.branch,
        "commit_id": snapshot.commit_id,
        "tree_id": snapshot.tree_id,
        "files": [
            {
                "path": item.get("path"),
                "content_hash": item.get("content_hash"),
                "blob_oid": item.get("blob_oid"),
            }
            for item in source_files
        ],
    }
    return _sha256_digest(payload)
