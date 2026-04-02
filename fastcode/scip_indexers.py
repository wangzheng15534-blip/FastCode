"""
Multi-language SCIP indexer runner.

Runs the appropriate SCIP indexer (scip-java, scip-go, scip-python, etc.)
based on the target language. Each indexer produces a binary .scip artifact.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .scip_models import SCIPIndex

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


# Map file extension -> language name (only languages with SCIP indexers)
_EXTENSION_MAP: Dict[str, str] = {
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".go": "go",
    ".py": "python",
    ".rb": "ruby",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rs": "rust",
    ".php": "php",
    ".dart": "dart",
}


def detect_scip_languages(repo_path: str) -> List[str]:
    """Walk the repo and return deduplicated list of languages with SCIP indexers support."""
    seen: set[str] = set()
    for root, _dirs, files in os.walk(repo_path):
        # Skip hidden and common non-source directories
        dirs_to_skip = {".git", ".hg", "node_modules", "__pycache__", ".venv", "venv"}
        _dirs[:] = [d for d in _dirs if d not in dirs_to_skip]
        for fname in files:
            _, ext = os.path.splitext(fname)
            lang = _EXTENSION_MAP.get(ext)
            if lang:
                seen.add(lang)
    return sorted(seen)


def run_scip_for_language(
    language: str,
    repo_path: str,
    output_dir: str,
) -> Optional[SCIPIndex]:
    """
    Run the SCIP indexer for one language and load the result.

    Returns SCIPIndex on success, None if indexer not available.
    """
    from .scip_loader import load_scip_artifact

    output_path = os.path.join(output_dir, f"{language}.scip")
    try:
        artifact_path = run_scip_indexer(language, repo_path, output_path)
        return load_scip_artifact(artifact_path)
    except RuntimeError as exc:
        logger.warning("SCIP indexer for %s unavailable: %s", language, exc)
        return None
