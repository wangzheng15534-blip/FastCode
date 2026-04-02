"""
Multi-language SCIP indexer runner.

Runs the appropriate SCIP indexer (scip-java, scip-go, scip-python, etc.)
based on the target language. Each indexer produces a binary .scip artifact.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Map language name -> (binary_name, extra_args)
SUPPORTED_LANGUAGES: Dict[str, Tuple[str, List[str]]] = {
    "java": ("scip-java", ["index", "--output"]),
    "kotlin": ("scip-java", ["index", "--output"]),
    "scala": ("scip-java", ["index", "--output"]),
    "go": ("scip-go", ["index", "--output"]),
    "python": ("scip-python", ["index", "--output"]),
    "ruby": ("scip-ruby", ["index", "--output"]),
    "typescript": ("scip-typescript", ["index", "--output"]),
    "javascript": ("scip-typescript", ["index", "--output"]),
    "c": ("scip-clang", ["index", "--output"]),
    "cpp": ("scip-clang", ["index", "--output"]),
    "csharp": ("scip-dotnet", ["index", "--output"]),
    "rust": ("rust-analyzer", ["scip", "--output"]),
    "php": ("scip-php", ["index", "--output"]),
    "dart": ("scip-dart", ["index", "--output"]),
}


def get_indexer_command(
    language: str,
    repo_path: str,
    output_path: str,
) -> Optional[List[str]]:
    """Build the indexer command for a language. Returns None if unsupported."""
    entry = SUPPORTED_LANGUAGES.get(language)
    if not entry:
        return None
    binary_name, extra_args = entry
    return [binary_name] + extra_args + [output_path]


def run_scip_indexer(
    language: str,
    repo_path: str,
    output_path: str,
) -> str:
    """
    Run the SCIP indexer for the given language.

    Returns the output artifact path on success.
    Raises RuntimeError if indexer is not installed or fails.
    """
    cmd = get_indexer_command(language, repo_path, output_path)
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
