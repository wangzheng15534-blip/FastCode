"""
Adapter from FastCode AST/indexer output into canonical unit-grounded IR.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping

from fastcode.ir.element import CodeElement
from fastcode.ir.types import (
    IRCodeUnit,
    IRRelation,
    IRSnapshot,
    IRUnitEmbedding,
    IRUnitSupport,
)
from fastcode.utils.filesystem import compute_file_hash, normalize_path
from fastcode.utils.json import safe_jsonable

STRUCTURE_SOURCE = "fc_structure"
STRUCTURE_PRIORITY = 50
STRUCTURE_EXTRACTOR = "fastcode.scip.ast_adapter"


def _hash_id(prefix: str, value: str) -> str:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=12).hexdigest()
    return f"{prefix}:{digest}"


def _doc_id(snapshot_id: str, rel_path: str) -> str:
    return f"doc:{snapshot_id}:{rel_path}"


def _support_id(snapshot_id: str, elem_id: str, label: str) -> str:
    return _hash_id("support", f"{snapshot_id}:{elem_id}:{label}")


def _relation_id(
    snapshot_id: str, relation_type: str, src_id: str, dst_id: str, payload: str = ""
) -> str:
    return _hash_id("rel", f"{snapshot_id}:{relation_type}:{src_id}:{dst_id}:{payload}")


def _embedding_id(snapshot_id: str, unit_id: str, source: str) -> str:
    return _hash_id("emb", f"{snapshot_id}:{unit_id}:{source}")


def _stable_unit_id(elem: CodeElement) -> str:
    stable = (elem.metadata or {}).get("stable_unit_id")
    if stable:
        return str(stable)
    return _hash_id(
        "unit",
        f"{elem.type}:{normalize_path(elem.relative_path or elem.file_path)}:{elem.name}",
    )


def _normalize_embedding_value(value: object) -> object:
    if value is None:
        return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            value = tolist()
        except Exception:
            return safe_jsonable(value)
    return safe_jsonable(value)


def _qualified_name(elem: CodeElement) -> str:
    meta = elem.metadata or {}
    return str(
        meta.get("qualified_name")
        or (
            f"{meta.get('class_name')}.{elem.name}"
            if meta.get("class_name")
            else elem.name
        )
    )


def _embedding_metadata(elem: CodeElement, element_type: str) -> dict[str, object]:
    meta = elem.metadata or {}
    payload: dict[str, object] = {
        "ast_element_id": elem.id,
        "element_type": element_type,
    }
    for field_name in (
        "embedding_artifact_ref",
        "embedding_fingerprint",
        "embedding_text_hash",
    ):
        value = meta.get(field_name)
        if value is not None:
            payload[field_name] = safe_jsonable(value)
    return payload


def build_ir_from_ast(
    repo_name: str,
    snapshot_id: str,
    elements: list[CodeElement],
    repo_root: str,
    branch: str | None = None,
    commit_id: str | None = None,
    tree_id: str | None = None,
    file_fingerprints: Mapping[str, Mapping[str, object]] | None = None,
) -> IRSnapshot:
    file_units: dict[str, IRCodeUnit] = {}
    units: list[IRCodeUnit] = []
    supports: list[IRUnitSupport] = []
    relations: list[IRRelation] = []
    embeddings: list[IRUnitEmbedding] = []
    unit_ids_by_element_id: dict[str, str] = {}
    class_units_by_path_name: dict[tuple[str, str], str] = {}
    unit_ids_by_name: dict[tuple[str, str], list[str]] = {}

    def ensure_file_unit(rel_path: str, language: str) -> IRCodeUnit:
        unit = file_units.get(rel_path)
        if unit is not None:
            unit.source_set.add(STRUCTURE_SOURCE)
            return unit
        fingerprint = (
            file_fingerprints.get(rel_path) if file_fingerprints is not None else None
        )
        content_hash = (
            str(fingerprint.get("content_hash") or fingerprint.get("blob_oid"))
            if fingerprint
            and (fingerprint.get("content_hash") or fingerprint.get("blob_oid"))
            else None
        )
        if content_hash is None:
            abs_path = os.path.join(repo_root, rel_path) if repo_root else rel_path
            content_hash = compute_file_hash(abs_path) or None
        blob_oid = (
            str(fingerprint.get("blob_oid"))
            if fingerprint and fingerprint.get("blob_oid")
            else content_hash
        )
        unit = IRCodeUnit(
            unit_id=_doc_id(snapshot_id, rel_path),
            kind="file",
            path=rel_path,
            language=language or "unknown",
            display_name=rel_path,
            source_set={STRUCTURE_SOURCE},
            metadata={
                "source_priority": STRUCTURE_PRIORITY,
                "blob_oid": blob_oid,
                "content_hash": content_hash,
                "stable_unit_id": _hash_id("file", rel_path),
            },
        )
        file_units[rel_path] = unit
        units.append(unit)
        supports.append(
            IRUnitSupport(
                support_id=_support_id(snapshot_id, rel_path, "file"),
                unit_id=unit.unit_id,
                source=STRUCTURE_SOURCE,
                support_kind="file",
                path=rel_path,
                metadata={"extractor": STRUCTURE_EXTRACTOR},
            )
        )
        return unit

    for elem in elements:
        rel_path = elem.relative_path or elem.file_path
        file_unit = ensure_file_unit(rel_path, elem.language or "unknown")
        elem_meta = elem.metadata or {}
        unit_kind = "doc" if elem.type == "documentation" else elem.type

        if unit_kind == "file":
            continue

        unit_id = (
            _stable_unit_id(elem)
            if unit_kind != "doc"
            else f"docunit:{snapshot_id}:{elem.id}"
        )
        parent_unit_id = file_unit.unit_id
        if elem.type == "method" and elem_meta.get("class_name"):
            parent_unit_id = class_units_by_path_name.get(
                (rel_path, str(elem_meta["class_name"])), file_unit.unit_id
            )

        unit = IRCodeUnit(
            unit_id=unit_id,
            kind=unit_kind,
            path=rel_path,
            language=elem.language or "unknown",
            display_name=elem.name or rel_path,
            qualified_name=_qualified_name(elem),
            signature=elem.signature,
            docstring=elem.docstring,
            summary=elem.summary,
            start_line=elem.start_line,
            start_col=int(elem_meta.get("start_col") or 0),
            end_line=elem.end_line,
            end_col=int(elem_meta.get("end_col") or 0),
            parent_unit_id=parent_unit_id,
            source_set={STRUCTURE_SOURCE},
            metadata={
                "ast_element_id": elem.id,
                "stable_unit_id": _stable_unit_id(elem),
                "source": STRUCTURE_SOURCE,
                "confidence": "resolved",
                "extractor": STRUCTURE_EXTRACTOR,
                "source_priority": STRUCTURE_PRIORITY,
                **{
                    k: v
                    for k, v in elem_meta.items()
                    if k not in {"embedding", "embedding_text"}
                },
            },
        )
        units.append(unit)
        unit_ids_by_element_id[str(elem.id)] = unit.unit_id
        unit_ids_by_name.setdefault((rel_path, elem.name), []).append(unit.unit_id)
        if elem.type == "class":
            class_units_by_path_name[(rel_path, elem.name)] = unit.unit_id

        supports.append(
            IRUnitSupport(
                support_id=_support_id(snapshot_id, elem.id, "structure"),
                unit_id=unit.unit_id,
                source=STRUCTURE_SOURCE,
                support_kind="structure",
                external_id=str(elem.id),
                path=rel_path,
                display_name=elem.name,
                qualified_name=unit.qualified_name,
                signature=elem.signature,
                start_line=max(int(elem.start_line or 1), 1),
                start_col=int(elem_meta.get("start_col") or 0),
                end_line=max(int(elem.end_line or elem.start_line or 1), 1),
                end_col=int(elem_meta.get("end_col") or 0),
                metadata={
                    "extractor": STRUCTURE_EXTRACTOR,
                    "element_type": elem.type,
                },
            )
        )
        supports.append(
            IRUnitSupport(
                support_id=_support_id(snapshot_id, elem.id, "definition"),
                unit_id=unit.unit_id,
                source=STRUCTURE_SOURCE,
                support_kind="occurrence",
                role="definition",
                path=rel_path,
                start_line=max(int(elem.start_line or 1), 1),
                start_col=int(elem_meta.get("start_col") or 0),
                end_line=max(int(elem.end_line or elem.start_line or 1), 1),
                end_col=int(elem_meta.get("end_col") or 0),
                metadata={"doc_id": file_unit.unit_id, "kind": elem.type},
            )
        )
        relations.append(
            IRRelation(
                relation_id=_relation_id(
                    snapshot_id, "contain", parent_unit_id, unit.unit_id
                ),
                src_unit_id=parent_unit_id,
                dst_unit_id=unit.unit_id,
                relation_type="contain",
                resolution_state="structural",
                support_sources={STRUCTURE_SOURCE},
                support_ids=[_support_id(snapshot_id, elem.id, "structure")],
                metadata={
                    "source": STRUCTURE_SOURCE,
                    "doc_id": file_unit.unit_id,
                    "extractor": STRUCTURE_EXTRACTOR,
                },
            )
        )

        embedding = _normalize_embedding_value(elem_meta.get("embedding"))
        embedding_text = _normalize_embedding_value(elem_meta.get("embedding_text"))
        if embedding is not None or embedding_text:
            embeddings.append(
                IRUnitEmbedding(
                    embedding_id=_embedding_id(
                        snapshot_id, unit.unit_id, "fc_embedding"
                    ),
                    unit_id=unit.unit_id,
                    source="fc_embedding",
                    vector=embedding
                    if isinstance(embedding, list)
                    else safe_jsonable(embedding),
                    embedding_text=str(embedding_text) if embedding_text else None,
                    metadata=_embedding_metadata(elem, elem.type),
                )
            )

    for elem in elements:
        rel_path = elem.relative_path or elem.file_path
        elem_meta = elem.metadata or {}
        file_unit = file_units.get(rel_path)
        if file_unit is None:
            continue

        if elem.type == "file":
            imports = elem_meta.get("imports", []) or []
            for imp in imports:
                module = (imp or {}).get("module")
                if not module:
                    continue
                module_path = module.replace(".", "/")
                target = next(
                    (
                        other.unit_id
                        for path, other in file_units.items()
                        if path.endswith(f"{module_path}.py")
                        or f"/{module_path}/" in path
                    ),
                    None,
                )
                if not target or target == file_unit.unit_id:
                    continue
                relations.append(
                    IRRelation(
                        relation_id=_relation_id(
                            snapshot_id, "import", file_unit.unit_id, target, module
                        ),
                        src_unit_id=file_unit.unit_id,
                        dst_unit_id=target,
                        relation_type="import",
                        resolution_state="structural",
                        support_sources={STRUCTURE_SOURCE},
                        metadata={
                            "module": module,
                            "source": STRUCTURE_SOURCE,
                            "extractor": STRUCTURE_EXTRACTOR,
                            "doc_id": file_unit.unit_id,
                        },
                    )
                )

        if elem.type == "class":
            src_unit_id = class_units_by_path_name.get((rel_path, elem.name))
            if not src_unit_id:
                continue
            for base in elem_meta.get("bases", []) or []:
                target = class_units_by_path_name.get((rel_path, str(base)))
                if not target:
                    candidates = unit_ids_by_name.get((rel_path, str(base)), [])
                    target = candidates[0] if candidates else None
                if not target or target == src_unit_id:
                    continue
                relations.append(
                    IRRelation(
                        relation_id=_relation_id(
                            snapshot_id, "inherit", src_unit_id, target, str(base)
                        ),
                        src_unit_id=src_unit_id,
                        dst_unit_id=target,
                        relation_type="inherit",
                        resolution_state="candidate",
                        support_sources={STRUCTURE_SOURCE},
                        metadata={
                            "base": base,
                            "source": STRUCTURE_SOURCE,
                            "extractor": STRUCTURE_EXTRACTOR,
                            "doc_id": file_unit.unit_id,
                        },
                    )
                )

    return IRSnapshot(
        repo_name=repo_name,
        snapshot_id=snapshot_id,
        branch=branch,
        commit_id=commit_id,
        tree_id=tree_id,
        units=units,
        supports=supports,
        relations=relations,
        embeddings=embeddings,
        metadata={"source_modes": [STRUCTURE_SOURCE]},
    )
