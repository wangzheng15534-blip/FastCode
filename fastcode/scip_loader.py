"""
SCIP artifact loading and optional local indexing helpers.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess

from fastcode.core import scip_transform as _scip_transform

from .scip_models import SCIPIndex

logger = logging.getLogger(__name__)


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
            from .scip_pb2 import Index as ProtobufIndex

            with open(path, "rb") as f:
                raw = f.read()
            pb_index = ProtobufIndex()
            pb_index.ParseFromString(raw)
            return _protobuf_to_scip_index(pb_index)
        except (ImportError, OSError, ValueError) as exc:
            logger.debug("Protobuf parsing failed, trying scip CLI: %s", exc)
            # Fallback to CLI
            scip_cli = shutil.which("scip")
            if not scip_cli:
                raise ValueError(
                    f".scip artifact could not be parsed (protobuf error: {exc}) "
                    "and 'scip' CLI is not available in PATH"
                ) from exc
            candidate_cmds = [
                [scip_cli, "print", "--json", path],
                [scip_cli, "dump", "--json", path],
            ]
            last_error = None
            for cmd in candidate_cmds:
                proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if proc.returncode == 0 and proc.stdout.strip():
                    return SCIPIndex.from_dict(json.loads(proc.stdout))
                last_error = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(
                f"Failed to decode .scip artifact via scip CLI: {last_error}"
            )

    raise ValueError(
        "Unsupported SCIP artifact format. Provide .scip, .json, or .scip.json."
    )


def run_scip_python_index(repo_path: str, output_path: str) -> str:
    from .scip_indexers import run_scip_indexer

    return run_scip_indexer("python", repo_path, output_path)


def _protobuf_to_scip_index(pb_index) -> SCIPIndex:
    from .scip_models import _EMPTY_RANGE, SCIPDocument, SCIPOccurrence, SCIPSymbol

    documents = []
    for doc in pb_index.documents:
        symbols = []
        for sym in doc.symbols:
            symbols.append(
                SCIPSymbol(
                    symbol=sym.symbol,
                    name=sym.display_name or None,
                    kind=_scip_kind_to_str(sym.kind),
                )
            )
        occurrences = []
        for occ in doc.occurrences:
            r = list(occ.range) if occ.range else list(_EMPTY_RANGE)
            occurrences.append(
                SCIPOccurrence(
                    symbol=occ.symbol,
                    role=_symbol_role_to_str(occ.symbol_roles),
                    range=r,
                )
            )
        documents.append(
            SCIPDocument(
                path=doc.relative_path,
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


def _symbol_role_to_str(roles: int) -> str:
    return _scip_transform.symbol_role_to_str(roles)


def _scip_kind_to_str(kind_value: int) -> str:
    return _scip_transform.scip_kind_to_str(kind_value)
