"""
SCIP artifact loading and optional local indexing helpers.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, cast

from .models import SCIPIndex
from .transform import scip_kind_to_str, symbol_role_to_str

logger = logging.getLogger(__name__)

ProtobufDecodeError: type[Exception]
try:
    from google.protobuf.message import DecodeError as _ProtobufDecodeError

    ProtobufDecodeError = cast(type[Exception], _ProtobufDecodeError)
except Exception:  # pragma: no cover - optional dependency
    ProtobufDecodeError = ValueError


def load_scip_artifact(path: str) -> SCIPIndex:
    """
    Load a SCIP artifact.

    Current v1 supports JSON-shaped SCIP payloads.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"SCIP artifact not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".json", ".scip.json"}:
        with open(path, encoding="utf-8") as f:
            return SCIPIndex.from_dict(json.load(f))
    if ext == ".scip":
        # Try protobuf parsing first (no external CLI needed)
        try:
            from .pb2 import Index as ProtobufIndex  # type: ignore[import-untyped]

            with open(path, "rb") as f:
                raw = f.read()
            pb_index: Any = ProtobufIndex()  # type: ignore[call-arg]
            pb_index.ParseFromString(raw)  # type: ignore[attribute-access]
            return _protobuf_to_scip_index(pb_index)
        except (ImportError, OSError, ValueError, ProtobufDecodeError) as exc:
            raise ValueError(f".scip artifact could not be parsed: {exc}") from exc

    raise ValueError(
        "Unsupported SCIP artifact format. Provide .scip, .json, or .scip.json."
    )


def _protobuf_to_scip_index(pb_index: Any) -> SCIPIndex:
    from .models import SCIPDocument, SCIPOccurrence, SCIPSymbol

    # Use a list as the empty range fallback (protobuf fields are unknown type)
    _empty_range: list[int] = [0, 0, 0, 0]

    documents: list[SCIPDocument] = []
    for doc in pb_index.documents:
        symbols: list[SCIPSymbol] = []
        for sym in doc.symbols:
            symbols.append(
                SCIPSymbol(
                    symbol=str(sym.symbol),
                    name=sym.display_name or None,
                    kind=_scip_kind_to_str(sym.kind),
                )
            )
        occurrences: list[SCIPOccurrence] = []
        for occ in doc.occurrences:
            r: list[int | None] = (
                cast(list[int | None], list(occ.range))
                if occ.range
                else list(_empty_range)
            )
            occurrences.append(
                SCIPOccurrence(
                    symbol=str(occ.symbol),
                    role=_symbol_role_to_str(occ.symbol_roles),
                    range=r,
                )
            )
        documents.append(
            SCIPDocument(
                path=str(doc.relative_path),
                language=doc.language or None,
                symbols=symbols,
                occurrences=occurrences,
            )
        )
    return SCIPIndex(
        documents=documents,
        indexer_name=pb_index.metadata.tool_info.name or None,
        indexer_version=pb_index.metadata.tool_info.version or None,
    )


def _symbol_role_to_str(roles: Any) -> str:
    return symbol_role_to_str(int(roles))


def _scip_kind_to_str(kind_value: Any) -> str:
    return scip_kind_to_str(int(kind_value))
