"""
Adapter from SCIP payloads into canonical unit-grounded IR.
"""

from __future__ import annotations

import hashlib
from typing import Any, cast

from ..scip_models import SCIPIndex
from ..semantic_ir import IRCodeUnit, IRRelation, IRSnapshot, IRUnitSupport

SCIP_SOURCE = "scip"


def _hid(prefix: str, payload: str) -> str:
    return f"{prefix}:{hashlib.blake2b(payload.encode('utf-8'), digest_size=12).hexdigest()}"


def _normalize_kind(kind: str | None) -> str:
    kind_value = (kind or "symbol").lower()
    return {
        "documentation": "doc",
        "module": "file",
        "type": "class",
    }.get(kind_value, kind_value)


def _normalize_range(
    raw: list[int | None] | tuple[int | None, ...] | None,
) -> tuple[int, int, int, int]:
    values = list(raw or [])[:4]
    while len(values) < 4:
        values.append(None)
    return (
        int(values[0] or 0),
        int(values[1] or 0),
        int(values[2] or 0),
        int(values[3] or 0),
    )


def _doc_id(snapshot_id: str, path: str) -> str:
    return f"doc:{snapshot_id}:{path}"


def _qualified_name(raw_symbol: dict[str, Any]) -> str | None:
    return raw_symbol.get("qualified_name") or raw_symbol.get("symbol")


def _symbol_display_name(raw_symbol: dict[str, Any], external_symbol: str) -> str:
    return str(
        raw_symbol.get("name")
        or external_symbol.rsplit("/", maxsplit=1)[-1]
        or external_symbol
    )


def build_ir_from_scip(
    repo_name: str,
    snapshot_id: str,
    scip_index: dict[str, Any] | SCIPIndex,
    branch: str | None = None,
    commit_id: str | None = None,
    tree_id: str | None = None,
    language_hint: str | None = None,
) -> IRSnapshot:
    payload = scip_index.to_dict() if isinstance(scip_index, SCIPIndex) else scip_index
    units: list[IRCodeUnit] = []
    supports: list[IRUnitSupport] = []
    relations: list[IRRelation] = []
    file_units: dict[str, IRCodeUnit] = {}
    symbol_units: dict[tuple[str, str], IRCodeUnit] = {}
    indexer_name: str | None = payload.get("indexer_name")
    indexer_version: str | None = payload.get("indexer_version")

    for raw_doc in cast(list[Any], payload.get("documents") or []):
        if not isinstance(raw_doc, dict):
            continue
        doc_dict = cast(dict[str, Any], raw_doc)
        path = str(doc_dict.get("path") or "")
        language = str(doc_dict.get("language") or language_hint or "unknown")
        file_unit = IRCodeUnit(
            unit_id=_doc_id(snapshot_id, path),
            kind="file",
            path=path,
            language=language,
            display_name=path,
            source_set={SCIP_SOURCE},
            metadata={
                "indexer_name": indexer_name,
                "indexer_version": indexer_version,
            },
        )
        file_units[path] = file_unit
        units.append(file_unit)
        supports.append(
            IRUnitSupport(
                support_id=_hid("support", f"{snapshot_id}:{path}:file"),
                unit_id=file_unit.unit_id,
                source=SCIP_SOURCE,
                support_kind="file",
                path=path,
                metadata={
                    "indexer_name": indexer_name,
                    "indexer_version": indexer_version,
                },
            )
        )

        for raw_symbol in cast(list[Any], doc_dict.get("symbols") or []):
            if not isinstance(raw_symbol, dict):
                continue
            sym_dict = cast(dict[str, Any], raw_symbol)
            external_symbol = str(sym_dict.get("symbol") or "")
            if not external_symbol:
                continue
            start_line, start_col, end_line, end_col = _normalize_range(
                sym_dict.get("range")
            )
            unit_id = f"scip:{snapshot_id}:{external_symbol}"
            unit = IRCodeUnit(
                unit_id=unit_id,
                kind=_normalize_kind(sym_dict.get("kind")),
                path=path,
                language=language,
                display_name=_symbol_display_name(sym_dict, external_symbol),
                qualified_name=_qualified_name(sym_dict),
                signature=sym_dict.get("signature"),
                start_line=start_line,
                start_col=start_col,
                end_line=end_line,
                end_col=end_col,
                parent_unit_id=file_unit.unit_id,
                primary_anchor_symbol_id=external_symbol,
                anchor_symbol_ids=[external_symbol],
                anchor_coverage=1.0,
                source_set={SCIP_SOURCE},
                metadata={
                    "scip": True,
                    "source": SCIP_SOURCE,
                    "confidence": "precise",
                    "indexer_name": indexer_name,
                    "indexer_version": indexer_version,
                    "documentation": sym_dict.get("documentation"),
                    "signature_documentation": sym_dict.get("signature_documentation"),
                    "relationships": sym_dict.get("relationships", []),
                },
            )
            units.append(unit)
            symbol_units[(path, external_symbol)] = unit
            supports.append(
                IRUnitSupport(
                    support_id=_hid(
                        "support", f"{snapshot_id}:{path}:{external_symbol}:symbol"
                    ),
                    unit_id=unit.unit_id,
                    source=SCIP_SOURCE,
                    support_kind="anchor",
                    external_id=external_symbol,
                    path=path,
                    display_name=unit.display_name,
                    qualified_name=unit.qualified_name,
                    signature=unit.signature,
                    enclosing_external_id=sym_dict.get("enclosing_symbol"),
                    start_line=start_line,
                    start_col=start_col,
                    end_line=end_line,
                    end_col=end_col,
                    metadata={
                        "source": SCIP_SOURCE,
                        "confidence": "precise",
                        "indexer_name": indexer_name,
                        "indexer_version": indexer_version,
                        "relationships": sym_dict.get("relationships", []),
                    },
                )
            )
            relations.append(
                IRRelation(
                    relation_id=_hid(
                        "rel", f"contain:{file_unit.unit_id}:{unit.unit_id}"
                    ),
                    src_unit_id=file_unit.unit_id,
                    dst_unit_id=unit.unit_id,
                    relation_type="contain",
                    resolution_state="anchored",
                    support_sources={SCIP_SOURCE},
                    metadata={
                        "source": SCIP_SOURCE,
                        "extractor": "fastcode.adapters.scip_to_ir",
                        "doc_id": file_unit.unit_id,
                    },
                )
            )

        for raw_occurrence in cast(list[Any], doc_dict.get("occurrences") or []):
            if not isinstance(raw_occurrence, dict):
                continue
            occ_dict = cast(dict[str, Any], raw_occurrence)
            external_symbol = str(occ_dict.get("symbol") or "")
            if not external_symbol:
                continue
            unit = symbol_units.get((path, external_symbol))
            unit_id = unit.unit_id if unit else f"scip:{snapshot_id}:{external_symbol}"
            start_line, start_col, end_line, end_col = _normalize_range(
                occ_dict.get("range")
            )
            role = occ_dict.get("role")
            role = str(role) if role is not None else "reference"
            supports.append(
                IRUnitSupport(
                    support_id=_hid(
                        "support",
                        f"{snapshot_id}:{path}:{external_symbol}:{occ_dict.get('role', 'reference')}:{start_line}:{start_col}:{end_line}:{end_col}",
                    ),
                    unit_id=unit_id,
                    source=SCIP_SOURCE,
                    support_kind="occurrence",
                    external_id=external_symbol,
                    role=role,
                    path=path,
                    start_line=start_line,
                    start_col=start_col,
                    end_line=end_line,
                    end_col=end_col,
                    metadata={
                        "doc_id": file_unit.unit_id,
                        "source": SCIP_SOURCE,
                        "confidence": "precise",
                        "indexer_name": indexer_name,
                        "indexer_version": indexer_version,
                    },
                )
            )
            if role in {
                "reference",
                "definition",
                "implementation",
                "type_definition",
                "import",
                "write_access",
                "forward_definition",
            }:
                relations.append(
                    IRRelation(
                        relation_id=_hid(
                            "rel",
                            f"ref:{file_unit.unit_id}:{unit_id}:{start_line}:{start_col}:{end_line}:{end_col}:{occ_dict.get('role', 'reference')}",
                        ),
                        src_unit_id=file_unit.unit_id,
                        dst_unit_id=unit_id,
                        relation_type="ref",
                        resolution_state="anchored",
                        support_sources={SCIP_SOURCE},
                        support_ids=[
                            _hid(
                                "support",
                                f"{snapshot_id}:{path}:{external_symbol}:{role}:{start_line}:{start_col}:{end_line}:{end_col}",
                            )
                        ],
                        metadata={
                            "source": SCIP_SOURCE,
                            "extractor": "fastcode.adapters.scip_to_ir",
                            "role": role,
                            "occurrence_id": _hid(
                                "support",
                                f"{snapshot_id}:{path}:{external_symbol}:{role}:{start_line}:{start_col}:{end_line}:{end_col}",
                            ),
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
        metadata={"source_modes": [SCIP_SOURCE]},
    )
