"""Shell-side execution for external SCIP tooling."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

from ..scip.indexers import get_indexer_command, get_scip_indexer_profile
from ..scip.loader import load_scip_artifact
from ..scip.models import SCIPIndex

logger = logging.getLogger(__name__)


def run_scip_indexer(
    language: str,
    repo_path: str,
    output_path: str,
) -> str:
    """Run the SCIP indexer for the given language."""
    cmd = get_indexer_command(language, output_path)
    if cmd is None:
        raise RuntimeError(f"No SCIP indexer available for language: {language}")

    binary_name = cmd[0]
    binary_path = shutil.which(binary_name)
    if not binary_path:
        raise RuntimeError(
            f"SCIP indexer '{binary_name}' not found in PATH. "
            f"Install it to enable {language} support via SCIP."
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


def is_scip_available(language: str) -> bool:
    """Return whether the language's SCIP indexer binary is present."""
    profile = get_scip_indexer_profile(language)
    if profile is None:
        return False
    return shutil.which(profile.binary_name) is not None


def run_scip_for_language(
    language: str,
    repo_path: str,
    output_dir: str,
) -> SCIPIndex | None:
    """Run the SCIP indexer for one language and load the result."""
    output_path = os.path.join(output_dir, f"{language}.scip")
    try:
        artifact_path = run_scip_indexer(language, repo_path, output_path)
        return load_scip_artifact(artifact_path)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        logger.warning("SCIP indexer for %s unavailable: %s", language, exc)
        return None


def run_scip_python_index(repo_path: str, output_path: str) -> str:
    """Run the Python SCIP indexer."""
    return run_scip_indexer("python", repo_path, output_path)
