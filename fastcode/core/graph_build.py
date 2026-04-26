# fastcode/core/graph_build.py
"""Pure graph payload construction — extracted from terminus_publisher.py."""
from __future__ import annotations

import hashlib
from typing import Any

from ..semantic_ir import _resolution_to_confidence


def deterministic_event_id(snapshot_id: str, payload: str) -> str:
    """Generate a deterministic event ID from snapshot_id + payload hash."""
    h = hashlib.sha256(f"{snapshot_id}:{payload}".encode()).hexdigest()[:32]
    return f"outbox:{snapshot_id}:{h}"


def build_code_graph_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build TerminusDB payload for code graph (symbol nodes + relation edges).

    Reads the ``units`` and ``relations`` fields from the snapshot dict
    (``ir.v2`` schema produced by ``IRSnapshot.to_dict()``).  Symbol nodes
    are created for every non-file, non-doc unit.  Relation edges carry the
    full resolution metadata so downstream consumers can apply confidence
    bands.
    """
    snapshot_id = snapshot.get("snapshot_id")
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for unit in snapshot.get("units") or []:
        kind = unit.get("kind", "")
        if kind in ("file", "doc"):
            continue
        unit_id = unit.get("unit_id")
        if not unit_id:
            continue
        node_id = f"sym:{snapshot_id}:{unit_id}"
        source_set = unit.get("source_set") or []
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
                    "source_set": source_set
                    if isinstance(source_set, list)
                    else list(source_set),
                },
            }
        )

    for rel in snapshot.get("relations") or []:
        rel_id = rel.get("relation_id")
        if not rel_id:
            continue
        src_id = rel.get("src_unit_id")
        dst_id = rel.get("dst_unit_id")
        if not src_id or not dst_id:
            continue
        support_sources = rel.get("support_sources") or []
        edges.append(
            {
                "id": f"rel:{snapshot_id}:{rel_id}",
                "type": rel.get("relation_type", ""),
                "src": f"sym:{snapshot_id}:{src_id}",
                "dst": f"sym:{snapshot_id}:{dst_id}",
                "confidence": _resolution_to_confidence(
                    rel.get("resolution_state", "")
                ),
                "resolution_state": rel.get("resolution_state", ""),
                "source_set": support_sources
                if isinstance(support_sources, list)
                else list(support_sources),
            }
        )

    return {"nodes": nodes, "edges": edges}
