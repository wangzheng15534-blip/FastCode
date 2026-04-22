"""
Adapter from FastCode AST/indexer output into canonical IR.
"""

from __future__ import annotations

import hashlib
import json

from ..indexer import CodeElement
from ..semantic_ir import IRAttachment, IRDocument, IREdge, IROccurrence, IRSnapshot, IRSymbol
from ..utils import safe_jsonable

STRUCTURE_SOURCE = "fc_structure"
STRUCTURE_CONFIDENCE = "resolved"
STRUCTURE_PRIORITY = 50
STRUCTURE_EXTRACTOR = "fastcode.adapters.ast_to_ir"


def _hash_id(prefix: str, value: str) -> str:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=10).hexdigest()
    return f"{prefix}:{digest}"


def _doc_id(snapshot_id: str, rel_path: str) -> str:
    return _hash_id("doc", f"{snapshot_id}:{rel_path}")


def _ast_symbol_id(snapshot_id: str, elem: CodeElement) -> str:
    qualified = (elem.metadata or {}).get("qualified_name") or elem.name or "unknown"
    start_col = int((elem.metadata or {}).get("start_col") or 0)
    return (
        f"ast:{snapshot_id}:{elem.language or 'unknown'}:{elem.relative_path or ''}:"
        f"{elem.type or 'unknown'}:{qualified}:{int(elem.start_line or 0)}:{start_col}"
    )


def _normalize_attachment_value(value: object) -> object:
    if value is None:
        return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            value = tolist()
        except Exception:
            return safe_jsonable(value)
    return safe_jsonable(value)


def _attachment_id(
    snapshot_id: str,
    target_type: str,
    target_id: str,
    attachment_type: str,
    source: str,
    payload: dict,
) -> str:
    payload_key = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return _hash_id(
        "att",
        f"{snapshot_id}:{target_type}:{target_id}:{attachment_type}:{source}:{payload_key}",
    )


def _build_symbol_attachments(snapshot_id: str, symbol_id: str, elem: CodeElement) -> list[IRAttachment]:
    elem_meta = elem.metadata or {}
    attachments: list[IRAttachment] = []
    base_meta = {
        "ast_element_id": elem.id,
        "extractor": STRUCTURE_EXTRACTOR,
        "element_type": elem.type,
    }

    if elem.summary:
        payload = {"text": str(elem.summary)}
        attachments.append(
            IRAttachment(
                attachment_id=_attachment_id(snapshot_id, "symbol", symbol_id, "summary", STRUCTURE_SOURCE, payload),
                target_id=symbol_id,
                target_type="symbol",
                attachment_type="summary",
                source=STRUCTURE_SOURCE,
                confidence="derived",
                payload=payload,
                metadata=base_meta,
            )
        )

    embedding = _normalize_attachment_value(elem_meta.get("embedding"))
    embedding_text = _normalize_attachment_value(elem_meta.get("embedding_text"))
    if embedding is not None or embedding_text:
        payload = {}
        if embedding is not None:
            payload["vector"] = embedding
        if embedding_text:
            payload["text"] = embedding_text
        attachments.append(
            IRAttachment(
                attachment_id=_attachment_id(snapshot_id, "symbol", symbol_id, "embedding", "fc_embedding", payload),
                target_id=symbol_id,
                target_type="symbol",
                attachment_type="embedding",
                source="fc_embedding",
                confidence="derived",
                payload=payload,
                metadata=base_meta,
            )
        )

    return attachments


def build_ir_from_ast(
    repo_name: str,
    snapshot_id: str,
    elements: list[CodeElement],
    repo_root: str,
    branch: str | None = None,
    commit_id: str | None = None,
    tree_id: str | None = None,
) -> IRSnapshot:
    del repo_root  # reserved for future module-path resolution
    documents = {}
    symbols: list[IRSymbol] = []
    occurrences: list[IROccurrence] = []
    edges: list[IREdge] = []
    attachments: list[IRAttachment] = []
    doc_by_rel_path = {}
    class_symbols = {}
    symbols_by_name = {}

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
                source_set={STRUCTURE_SOURCE},
            )
            doc_by_rel_path[rel_path] = doc_id
        else:
            documents[doc_id].source_set.add(STRUCTURE_SOURCE)

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
            source_priority=STRUCTURE_PRIORITY,
            source_set={STRUCTURE_SOURCE},
            metadata={
                "ast_element_id": elem.id,
                "source": STRUCTURE_SOURCE,
                "confidence": STRUCTURE_CONFIDENCE,
                "extractor": STRUCTURE_EXTRACTOR,
                **{k: v for k, v in elem_meta.items() if k not in ("embedding", "embedding_text")},
            },
        )
        symbols.append(symbol)
        attachments.extend(_build_symbol_attachments(snapshot_id, symbol_id, elem))
        symbols_by_name.setdefault(elem.name, []).append(symbol_id)
        if elem.type == "class":
            class_symbols[(rel_path, elem.name)] = symbol_id

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
                source=STRUCTURE_SOURCE,
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
                source=STRUCTURE_SOURCE,
                confidence=STRUCTURE_CONFIDENCE,
                doc_id=doc_id,
                metadata={"extractor": STRUCTURE_EXTRACTOR, "source": STRUCTURE_SOURCE},
            )
        )

    # Build import/inheritance edges from AST metadata.
    for elem in elements:
        elem_meta = elem.metadata or {}
        rel_path = elem.relative_path or elem.file_path
        doc_id = doc_by_rel_path.get(rel_path)
        if not doc_id:
            continue

        if elem.type == "file":
            imports = elem_meta.get("imports", []) or []
            for imp in imports:
                module = (imp or {}).get("module")
                if not module:
                    continue
                module_path = module.replace(".", "/")
                target_doc_id = None
                for known_path, known_doc_id in doc_by_rel_path.items():
                    if known_path.endswith(f"{module_path}.py") or f"/{module_path}/" in known_path:
                        target_doc_id = known_doc_id
                        break
                if not target_doc_id or target_doc_id == doc_id:
                    continue
                edge_id = _hash_id("edge", f"import:{doc_id}:{target_doc_id}:{module}")
                edges.append(
                    IREdge(
                        edge_id=edge_id,
                        src_id=doc_id,
                        dst_id=target_doc_id,
                        edge_type="import",
                        source=STRUCTURE_SOURCE,
                        confidence="heuristic",
                        doc_id=doc_id,
                        metadata={
                            "module": module,
                            "extractor": STRUCTURE_EXTRACTOR,
                            "source": STRUCTURE_SOURCE,
                        },
                    )
                )

        if elem.type == "class":
            src_symbol_id = class_symbols.get((rel_path, elem.name))
            if not src_symbol_id:
                continue
            for base in elem_meta.get("bases", []) or []:
                target_symbol_id = class_symbols.get((rel_path, base))
                if not target_symbol_id:
                    candidates = symbols_by_name.get(base, [])
                    target_symbol_id = candidates[0] if candidates else None
                if not target_symbol_id or target_symbol_id == src_symbol_id:
                    continue
                edge_id = _hash_id("edge", f"inherit:{src_symbol_id}:{target_symbol_id}:{base}")
                edges.append(
                    IREdge(
                        edge_id=edge_id,
                        src_id=src_symbol_id,
                        dst_id=target_symbol_id,
                        edge_type="inherit",
                        source=STRUCTURE_SOURCE,
                        confidence="heuristic",
                        doc_id=doc_id,
                        metadata={
                            "base": base,
                            "extractor": STRUCTURE_EXTRACTOR,
                            "source": STRUCTURE_SOURCE,
                        },
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
        attachments=attachments,
        metadata={"source_modes": [STRUCTURE_SOURCE]},
    )
