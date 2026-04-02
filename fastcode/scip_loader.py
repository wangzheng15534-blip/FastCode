"""
SCIP artifact loading and optional local indexing helpers.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess

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
        with open(path, "r", encoding="utf-8") as f:
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
        except Exception as exc:
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
            raise RuntimeError(f"Failed to decode .scip artifact via scip CLI: {last_error}")

    raise ValueError(
        "Unsupported SCIP artifact format. Provide .scip, .json, or .scip.json."
    )


def run_scip_python_index(repo_path: str, output_path: str) -> str:
    """
    Run scip-python locally and return produced artifact path.

    Delegates to the multi-language indexer runner in scip_indexers.
    """
    from .scip_indexers import run_scip_indexer

    return run_scip_indexer("python", repo_path, output_path)
    if cmd is None:
        raise RuntimeError(f"No SCIP indexer available for language: {language}")
    binary_name = cmd[0]
    binary_path = shutil.which(binary_name)
    if not binary_path:
        raise RuntimeError(
            f"SCIP indexer '{binary_name}' not found in PATH. "
            f"Install it to enable {language} support via SCIP"
        )
    cmd[0] = binary_path
    logger.info("Running SCIP indexer: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=repo_path,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{binary_name} failed ({proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return output_path


def _protobuf_to_scip_index(pb_index) -> SCIPIndex:
    """Convert a protobuf Index message to SCIPIndex."""
    from .scip_models import SCIPDocument, SCIPOccurrence, SCIPSymbol

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
            r = list(occ.range) if occ.range else [None, None, None, None]
            roles = occ.symbol_roles
            role = "definition" if roles & 1 else "reference"
            occurrences.append(
                SCIPOccurrence(
                    symbol=occ.symbol,
                    role=role,
                    range=r,
                )
            )
        lang = doc.language if doc.language else None
        documents.append(
            SCIPDocument(
                path=doc.relative_path,
                language=lang,
                symbols=symbols,
                occurrences=occurrences,
            )
        )
    return SCIPIndex(
        documents=documents,
        indexer_name=pb_index.metadata.tool_info.name or None,
        indexer_version=pb_index.metadata.tool_info.version or None,
    )


def _scip_kind_to_str(kind_value: int) -> str:
    """Convert SCIP protobuf Kind enum to string."""
    from .scip_pb2 import SymbolInformation

    kind_map = {
        SymbolInformation.Kind.Function: "function",
        SymbolInformation.Kind.Method: "method",
        SymbolInformation.Kind.Class: "class",
        SymbolInformation.Kind.Interface: "interface",
        SymbolInformation.Kind.Enum: "enum",
        SymbolInformation.Kind.EnumMember: "enum_member",
        SymbolInformation.Kind.Variable: "variable",
        SymbolInformation.Kind.Constant: "constant",
        SymbolInformation.Kind.Property: "property",
        SymbolInformation.Kind.Type: "type",
        SymbolInformation.Kind.Macro: "macro",
        SymbolInformation.Kind.Module: "module",
        SymbolInformation.Kind.Namespace: "namespace",
        SymbolInformation.Kind.Package: "package",
        SymbolInformation.Kind.Parameter: "parameter",
        SymbolInformation.Kind.TypeParameter: "type_parameter",
        SymbolInformation.Kind.Constructor: "constructor",
        SymbolInformation.Kind.Struct: "struct",
    }
    return kind_map.get(kind_value, "symbol")
