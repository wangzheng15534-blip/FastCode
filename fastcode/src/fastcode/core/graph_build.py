# fastcode/core/graph_build.py
"""Pure graph payload construction — extracted from terminus_publisher.py."""

from __future__ import annotations

from typing import Any, cast

from fastcode.schemas.ir import resolution_to_confidence


def build_code_graph_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build TerminusDB payload for code graph (symbol nodes + relation edges).

    Reads the ``units`` and ``relations`` fields from the snapshot dict
    (``ir.v2`` schema produced by ``IRSnapshot.to_dict()``).  Symbol nodes
    are created for every non-file, non-doc unit.  Relation edges carry the
    full resolution metadata so downstream consumers can apply confidence
    bands.
    """
    snapshot_id: str = snapshot.get("snapshot_id") or ""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for unit in cast(list[Any], snapshot.get("units") or []):
        unit = cast(dict[str, Any], unit)
        kind: str = unit.get("kind") or ""
        if kind in ("file", "doc"):
            continue
        unit_id: str = unit.get("unit_id") or ""
        if not unit_id:
            continue
        node_id = f"sym:{snapshot_id}:{unit_id}"
        source_set: list[Any] = unit.get("source_set") or []
        nodes.append(
            {
                "id": node_id,
                "type": "Symbol",
                "props": {
                    "unit_id": unit_id,
                    "display_name": unit.get("display_name"),
                    "kind": kind,
                    "path": unit.get("path"),
                    "language": unit.get("language"),
                    "start_line": unit.get("start_line"),
                    "end_line": unit.get("end_line"),
                    "qualified_name": unit.get("qualified_name"),
                    "scip_symbol": unit.get("primary_anchor_symbol_id"),
                    "source_set": list(source_set),
                },
            }
        )

    for rel in cast(list[Any], snapshot.get("relations") or []):
        rel = cast(dict[str, Any], rel)
        rel_id: str = rel.get("relation_id") or ""
        if not rel_id:
            continue
        src_id: str = rel.get("src_unit_id") or ""
        dst_id: str = rel.get("dst_unit_id") or ""
        if not src_id or not dst_id:
            continue
        support_sources: list[Any] = rel.get("support_sources") or []
        edges.append(
            {
                "id": f"rel:{snapshot_id}:{rel_id}",
                "type": rel.get("relation_type") or "",
                "src": f"sym:{snapshot_id}:{src_id}",
                "dst": f"sym:{snapshot_id}:{dst_id}",
                "confidence": resolution_to_confidence(
                    rel.get("resolution_state") or ""
                ),
                "resolution_state": rel.get("resolution_state") or "",
                "source_set": list(support_sources),
            }
        )

    return {"nodes": nodes, "edges": edges}
