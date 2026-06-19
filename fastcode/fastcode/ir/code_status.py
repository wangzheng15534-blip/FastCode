"""DocKG-facing code status pack export helpers.

This is a meaning_core module: it is PURE. It shapes IR data into the pack
dict and assigns deterministic IDs, but performs NO serialization or hashing
itself — `import json` / `import hashlib` / `fastcode.utils.json` are effectful
boundary work (serde_json + digest) and are injected by the use_flow owner
(`fastcode.app.store.facade.StoreFacade` via `code_status_keys`). The injected
callables reproduce the historical digest/ID algorithm exactly, so pack output
is byte-for-byte unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, cast

from fastcode.ir.types import IRCodeUnit, IRRelation, IRSnapshot

CODE_STATUS_PACK_SCHEMA = "code_status_pack.v0"
CODE_STATUS_EXTRACTION_VERSION = "fastcode.code_status_export.v0"


class _Normalize(Protocol):
    def __call__(self, value: Any) -> Any: ...


class _Digest(Protocol):
    def __call__(self, value: Any) -> str: ...


class _SourceId(Protocol):
    def __call__(self, path: str) -> str: ...


class _SpanId(Protocol):
    def __call__(
        self, owner_kind: str, owner_id: str, path: str, start: Any, end: Any
    ) -> str: ...


def build_code_status_pack(
    snapshot: IRSnapshot,
    *,
    artifact_key: str | None = None,
    manifest: Mapping[str, Any] | None = None,
    ir_graphs: Any | None = None,
    include_graph_facts: bool = True,
    normalize: _Normalize | None = None,
    source_id_fn: _SourceId | None = None,
    span_id_fn: _SpanId | None = None,
    digest_fn: _Digest | None = None,
    skip_keys: bool = False,
) -> dict[str, Any]:
    """Build a deterministic JSON-serializable code-status evidence pack.

    ``normalize`` / ``source_id_fn`` / ``span_id_fn`` / ``digest_fn`` carry the
    serde/serialization + hashing effect out of this pure meaning_core module.
    They are normally supplied by the use_flow owner (see
    ``fastcode.app.store.code_status_keys``).

    To avoid silent corruption (a pack with blank source_id/span_id/digest
    that looks structurally valid), the key callables MUST be supplied unless
    ``skip_keys=True`` is passed explicitly — the shape-only path returns blank
    IDs/digests deliberately, so callers must opt into it.
    """
    if skip_keys:
        normalize_fn = normalize or _identity
        source_id = source_id_fn or _no_source_id
        span_id = span_id_fn or _no_span_id
        digest = digest_fn or _no_digest
    else:
        missing = [
            name
            for name, fn in (
                ("normalize", normalize),
                ("source_id_fn", source_id_fn),
                ("span_id_fn", span_id_fn),
                ("digest_fn", digest_fn),
            )
            if fn is None
        ]
        if missing:
            msg = (
                "build_code_status_pack requires the key callables "
                f"({', '.join(missing)}); pass them from the use_flow owner "
                "(fastcode.app.store.code_status_keys) or skip_keys=True for a "
                "shape-only pack"
            )
            raise ValueError(msg)
        normalize_fn = normalize
        source_id = source_id_fn
        span_id = span_id_fn
        digest = digest_fn

    file_units = [unit for unit in snapshot.units if unit.kind in {"file", "doc"}]
    source_files = _source_file_payloads(
        snapshot, file_units, source_id=source_id, normalize=normalize_fn
    )
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
        span_id=span_id,
        normalize=normalize_fn,
    )
    code_units = [
        _code_unit_payload(
            unit,
            source_span_id=unit_span_ids.get(unit.unit_id),
            normalize=normalize_fn,
        )
        for unit in sorted(snapshot.units, key=lambda item: item.unit_id)
    ]
    symbols = [
        _symbol_payload(
            unit,
            source_span_id=unit_span_ids.get(unit.unit_id),
            normalize=normalize_fn,
        )
        for unit in sorted(snapshot.units, key=lambda item: item.unit_id)
        if unit.kind not in {"file", "doc"}
    ]
    relation_facts = [
        _relation_payload(
            relation,
            support_span_ids=support_span_ids,
            normalize=normalize_fn,
        )
        for relation in sorted(snapshot.relations, key=lambda item: item.relation_id)
    ]
    graph_facts = (
        _graph_fact_payloads(ir_graphs, normalize=normalize_fn)
        if include_graph_facts and ir_graphs
        else []
    )
    diagnostics = _diagnostics_payload(snapshot.metadata, normalize=normalize_fn)
    pack: dict[str, Any] = {
        "schema_version": CODE_STATUS_PACK_SCHEMA,
        "extraction_version": CODE_STATUS_EXTRACTION_VERSION,
        "producer": {"name": "FastCode"},
        "repo": {
            "repo_name": snapshot.repo_name,
            "branch": snapshot.branch,
            "commit_id": snapshot.commit_id,
            "tree_id": snapshot.tree_id,
            "worktree_digest": digest(
                _worktree_payload(snapshot, source_files)
            ),
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
        "manifest": normalize_fn(dict(manifest or {})),
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
    pack["pack_digest"] = digest(pack)
    return pack


def _identity(value: Any) -> Any:
    return value


def _no_source_id(_path: str) -> str:
    return ""


def _no_span_id(
    _owner_kind: str, _owner_id: str, _path: str, _start: Any, _end: Any
) -> str:
    return ""


def _no_digest(_value: Any) -> str:
    return ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    items = cast(Iterable[Any], value)
    return sorted(str(item) for item in items if item)


def _metadata(
    value: Mapping[str, Any] | None, *, normalize: _Normalize
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, item in (value or {}).items():
        payload[str(key)] = item
    return cast(dict[str, Any], normalize(payload))


def _source_file_payloads(
    snapshot: IRSnapshot,
    file_units: Sequence[IRCodeUnit],
    *,
    source_id: _SourceId,
    normalize: _Normalize,
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
        metadata = _metadata(
            unit.metadata if unit is not None else {}, normalize=normalize
        )
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
                "source_id": source_id(path),
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
    span_id: _SpanId,
    normalize: _Normalize,
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for unit in sorted(snapshot.units, key=lambda item: item.unit_id):
        if unit.path not in source_id_by_path:
            continue
        current_span_id = span_id(
            "unit",
            unit.unit_id,
            unit.path,
            unit.start_line,
            unit.end_line,
        )
        unit_span_ids[unit.unit_id] = current_span_id
        spans.append(
            {
                "source_span_id": current_span_id,
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
        current_span_id = span_id(
            "support",
            support.support_id,
            path,
            support.start_line,
            support.end_line,
        )
        support_span_ids[support.support_id] = current_span_id
        spans.append(
            {
                "source_span_id": current_span_id,
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
                "metadata": _metadata(support.metadata, normalize=normalize),
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
    normalize: _Normalize,
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
        "metadata": _metadata(unit.metadata, normalize=normalize),
    }


def _symbol_payload(
    unit: IRCodeUnit,
    *,
    source_span_id: str | None,
    normalize: _Normalize,
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
        "metadata": _metadata(unit.metadata, normalize=normalize),
    }


def _relation_payload(
    relation: IRRelation,
    *,
    support_span_ids: Mapping[str, str],
    normalize: _Normalize,
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
        "metadata": _metadata(relation.metadata, normalize=normalize),
    }


def _graph_fact_payloads(ir_graphs: Any, *, normalize: _Normalize) -> list[dict[str, Any]]:
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
                    "metadata": _metadata(attrs, normalize=normalize),
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


def _diagnostics_payload(
    metadata: Mapping[str, Any] | None, *, normalize: _Normalize
) -> dict[str, Any]:
    payload = _metadata(metadata, normalize=normalize)
    return {
        "warnings": _string_list(payload.get("warnings")),
        "pipeline_layers": normalize(payload.get("pipeline_layers") or []),
        "pipeline_metrics": normalize(payload.get("pipeline_metrics") or {}),
        "semantic_resolver_runs": normalize(
            payload.get("semantic_resolver_runs") or []
        ),
        "resolver_diagnostics": normalize(
            payload.get("resolver_diagnostics")
            or payload.get("semantic_resolver_diagnostics")
            or []
        ),
        "scip_degraded_reasons": _string_list(payload.get("scip_degraded_reasons")),
    }


def _worktree_payload(
    snapshot: IRSnapshot,
    source_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
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
