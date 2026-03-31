"""
Adapter from SCIP payloads into canonical IR.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from ..semantic_ir import IRDocument, IROccurrence, IRSnapshot, IRSymbol


def _hid(prefix: str, payload: str) -> str:
    return f"{prefix}:{hashlib.md5(payload.encode('utf-8')).hexdigest()[:24]}"


def build_ir_from_scip(
    repo_name: str,
    snapshot_id: str,
    scip_index: Dict[str, Any],
    branch: str | None = None,
    commit_id: str | None = None,
    tree_id: str | None = None,
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

    for doc in scip_index.get("documents", []):
        path = doc.get("path", "")
        language = doc.get("language", "unknown")
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
            r = sym.get("range", [None, None, None, None])
            symbol_id = f"scip:{ext_symbol}"
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
                    metadata={"scip": True},
                )
            )

        for occ in doc.get("occurrences", []):
            ext_symbol = occ.get("symbol")
            if not ext_symbol:
                continue
            r = occ.get("range", [0, 0, 0, 0])
            role = occ.get("role", "reference")
            occ_id = _hid("occ", f"{snapshot_id}:{doc_id}:{ext_symbol}:{role}:{r}")
            occurrences.append(
                IROccurrence(
                    occurrence_id=occ_id,
                    symbol_id=f"scip:{ext_symbol}",
                    doc_id=doc_id,
                    role=role,
                    start_line=int(r[0] or 0),
                    start_col=int(r[1] or 0),
                    end_line=int(r[2] or 0),
                    end_col=int(r[3] or 0),
                    source="scip",
                    metadata={},
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
        edges=[],
        metadata={"source_modes": ["scip"]},
    )

