"""
Adapter from FastCode AST/indexer output into canonical IR.
"""

from __future__ import annotations

import hashlib
from typing import List

from ..indexer import CodeElement
from ..semantic_ir import IRDocument, IREdge, IROccurrence, IRSnapshot, IRSymbol


def _hash_id(prefix: str, value: str) -> str:
    return f"{prefix}:{hashlib.md5(value.encode('utf-8')).hexdigest()[:20]}"


def _doc_id(snapshot_id: str, rel_path: str) -> str:
    return _hash_id("doc", f"{snapshot_id}:{rel_path}")


def _ast_symbol_id(snapshot_id: str, elem: CodeElement) -> str:
    local = ":".join(
        [
            snapshot_id,
            elem.language or "",
            elem.relative_path or "",
            elem.type or "",
            elem.name or "",
            str(elem.start_line or 0),
            str(elem.end_line or 0),
        ]
    )
    return f"ast:{hashlib.md5(local.encode('utf-8')).hexdigest()[:24]}"


def build_ir_from_ast(
    repo_name: str,
    snapshot_id: str,
    elements: List[CodeElement],
    repo_root: str,
    branch: str | None = None,
    commit_id: str | None = None,
    tree_id: str | None = None,
) -> IRSnapshot:
    del repo_root  # reserved for future module-path resolution
    documents = {}
    symbols: List[IRSymbol] = []
    occurrences: List[IROccurrence] = []
    edges: List[IREdge] = []

    for elem in elements:
        elem_meta = elem.metadata or {}
        rel_path = elem.relative_path or elem.file_path
        doc_id = _doc_id(snapshot_id, rel_path)
        if doc_id not in documents:
            documents[doc_id] = IRDocument(
                doc_id=doc_id,
                path=rel_path,
                language=elem.language or "unknown",
                content_hash=None,
                source_set={"ast"},
            )
        else:
            documents[doc_id].source_set.add("ast")

        if elem.type in {"file", "documentation"}:
            continue

        symbol_id = _ast_symbol_id(snapshot_id, elem)
        symbol = IRSymbol(
            symbol_id=symbol_id,
            external_symbol_id=None,
            path=rel_path,
            display_name=elem.name,
            qualified_name=elem_meta.get("class_name") + "." + elem.name
            if elem_meta.get("class_name")
            else elem.name,
            kind=elem.type,
            language=elem.language or "unknown",
            signature=elem.signature,
            start_line=elem.start_line,
            start_col=0,
            end_line=elem.end_line,
            end_col=0,
            source_priority=10,
            source_set={"ast"},
            metadata={"ast_element_id": elem.id, **elem_meta},
        )
        symbols.append(symbol)

        occ_id = _hash_id("occ", f"{symbol_id}:{doc_id}:{elem.start_line}:{elem.end_line}")
        occurrences.append(
            IROccurrence(
                occurrence_id=occ_id,
                symbol_id=symbol_id,
                doc_id=doc_id,
                role="definition",
                start_line=max(elem.start_line or 1, 1),
                start_col=0,
                end_line=max(elem.end_line or elem.start_line or 1, 1),
                end_col=0,
                source="ast",
                metadata={"kind": elem.type},
            )
        )

        contain_edge_id = _hash_id("edge", f"contain:{doc_id}:{symbol_id}")
        edges.append(
            IREdge(
                edge_id=contain_edge_id,
                src_id=doc_id,
                dst_id=symbol_id,
                edge_type="contain",
                source="ast",
                confidence="resolved",
                doc_id=doc_id,
                metadata={},
            )
        )

    return IRSnapshot(
        repo_name=repo_name,
        snapshot_id=snapshot_id,
        branch=branch,
        commit_id=commit_id,
        tree_id=tree_id,
        documents=list(documents.values()),
        symbols=symbols,
        occurrences=occurrences,
        edges=edges,
        metadata={"source_modes": ["ast"]},
    )
