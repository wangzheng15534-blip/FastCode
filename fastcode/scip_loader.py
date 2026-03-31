"""
SCIP artifact loading and optional local indexing helpers.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict


logger = logging.getLogger(__name__)


def load_scip_artifact(path: str) -> Dict[str, Any]:
    """
    Load a SCIP artifact.

    Current v1 supports JSON-shaped SCIP payloads.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"SCIP artifact not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".json", ".scip.json"}:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(
        "Unsupported SCIP artifact format. Provide JSON payload (.json/.scip.json)."
    )


def run_scip_python_index(repo_path: str, output_path: str) -> str:
    """
    Run scip-python locally and return produced artifact path.
    """
    scip_bin = shutil.which("scip-python")
    if not scip_bin:
        raise RuntimeError("scip-python executable not found in PATH")

    cmd = [scip_bin, "index", "--output", output_path]
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
            f"scip-python failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return output_path

