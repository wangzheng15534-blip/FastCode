"""
Adapter from SCIP payloads into canonical IR.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Union

from ..scip_models import SCIPIndex
from ..semantic_ir import IRDocument, IREdge, IROccurrence, IRSnapshot, IRSymbol


def _hid(prefix: str, payload: str) -> str:
    return f"{prefix}:{hashlib.md5(payload.encode('utf-8')).hexdigest()[:24]}"


def build_ir_from_scip(
    repo_name: str,
    snapshot_id: str,
    scip_index: Union[Dict[str, Any], SCIPIndex],
    branch: str | None = None,
    commit_id: str | None = None,
    tree_id: str | None = None,
    language_hint: str | None = None,
) -> IRSnapshot:
    """
    Convert a simplified SCIP payload to IRSnapshot.

    Expected payload shape:
    {
      "documents": [
        {
          "path": "...",
          "language": "...",
          "symbols": [
            {"symbol": "...", "name": "...", "kind": "...", "range": [sl, sc, el, ec], "signature": "..."}
          ],
          "occurrences": [
            {"symbol": "...", "role": "definition|reference|implementation", "range": [sl, sc, el, ec]}
          ]
        }
      ]
    }
    """
    documents: List[IRDocument] = []
    symbols: List[IRSymbol] = []
    occurrences: List[IROccurrence] = []
    edges: List[IREdge] = []
    scip_payload = scip_index.to_dict() if isinstance(scip_index, SCIPIndex) else scip_index
    indexer_name = scip_payload.get("indexer_name")
    indexer_version = scip_payload.get("indexer_version")

    for doc in scip_payload.get("documents", []):
        path = doc.get("path", "")
        language = doc.get("language") or language_hint or "unknown"
        doc_id = _hid("doc", f"{snapshot_id}:{path}")
        documents.append(
            IRDocument(
                doc_id=doc_id,
                path=path,
                language=language,
                source_set={"scip"},
            )
        )

        for sym in doc.get("symbols", []):
            ext_symbol = sym.get("symbol")
            if not ext_symbol:
                continue
            raw_r = sym.get("range", [None, None, None, None])
            r = (raw_r + [None, None, None, None])[:4]
            symbol_id = f"scip:{snapshot_id}:{ext_symbol}"
            symbols.append(
                IRSymbol(
                    symbol_id=symbol_id,
                    external_symbol_id=ext_symbol,
                    path=path,
                    display_name=sym.get("name") or ext_symbol.split("/")[-1],
                    qualified_name=sym.get("qualified_name"),
                    kind=sym.get("kind", "symbol"),
                    language=language,
                    signature=sym.get("signature"),
                    start_line=r[0],
                    start_col=r[1],
                    end_line=r[2],
                    end_col=r[3],
                    source_priority=100,
                    source_set={"scip"},
                    metadata={
                        "scip": True,
                        "source": "scip",
                        "confidence": "precise",
                        "indexer_name": indexer_name,
                        "indexer_version": indexer_version,
                    },
                )
            )
            edges.append(
                IREdge(
                    edge_id=_hid("edge", f"contain:{doc_id}:{symbol_id}"),
                    src_id=doc_id,
                    dst_id=symbol_id,
                    edge_type="contain",
                    source="scip",
                    confidence="precise",
                    doc_id=doc_id,
                    metadata={
                        "extractor": "fastcode.adapters.scip_to_ir",
                        "indexer_name": indexer_name,
                        "indexer_version": indexer_version,
                    },
                )
            )

        for occ in doc.get("occurrences", []):
            ext_symbol = occ.get("symbol")
            if not ext_symbol:
                continue
            raw_r = occ.get("range", [0, 0, 0, 0])
            r = (raw_r + [0, 0, 0, 0])[:4]
            role = occ.get("role", "reference")
            symbol_id = f"scip:{snapshot_id}:{ext_symbol}"
            occ_id = _hid("occ", f"{snapshot_id}:{doc_id}:{ext_symbol}:{role}:{r}")
            occurrences.append(
                IROccurrence(
                    occurrence_id=occ_id,
                    symbol_id=symbol_id,
                    doc_id=doc_id,
                    role=role,
                    start_line=int(r[0] or 0),
                    start_col=int(r[1] or 0),
                    end_line=int(r[2] or 0),
                    end_col=int(r[3] or 0),
                    source="scip",
                    metadata={
                        "source": "scip",
                        "confidence": "precise",
                        "indexer_name": indexer_name,
                        "indexer_version": indexer_version,
                    },
                )
            )
            if role in {"reference", "definition", "implementation", "type_definition", "import", "write_access", "forward_definition"}:
                edges.append(
                    IREdge(
                        edge_id=_hid("edge", f"ref:{doc_id}:{symbol_id}:{occ_id}"),
                        src_id=doc_id,
                        dst_id=symbol_id,
                        edge_type="ref",
                        source="scip",
                        confidence="precise",
                        doc_id=doc_id,
                        metadata={
                            "extractor": "fastcode.adapters.scip_to_ir",
                            "role": role,
                            "occurrence_id": occ_id,
                            "indexer_name": indexer_name,
                            "indexer_version": indexer_version,
                        },
                    )
                )

    return IRSnapshot(
        repo_name=repo_name,
        snapshot_id=snapshot_id,
        branch=branch,
        commit_id=commit_id,
        tree_id=tree_id,
        documents=documents,
        symbols=symbols,
        occurrences=occurrences,
        edges=edges,
        metadata={"source_modes": ["scip"]},
    )
