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

from .scip_models import SCIPIndex

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: dict[str, tuple[str, list[str]]] = {
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
}


def get_indexer_command(
    language: str,
    output_path: str,
) -> list[str] | None:
    """Build the indexer command for a language. Returns None if unsupported."""
    entry = SUPPORTED_LANGUAGES.get(language)
    if not entry:
        return None
    binary_name, extra_args = entry
    return [binary_name, *extra_args, output_path]


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


_EXTENSION_MAP: dict[str, str] = {
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
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
    ".rs": "rust",
}

_SKIP_DIRS = frozenset({".git", ".hg", "node_modules", "__pycache__", ".venv", "venv"})


def detect_scip_languages(repo_path: str) -> list[str]:
    seen: set[str] = set()
    for _, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
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
) -> SCIPIndex | None:
    """
    Run the SCIP indexer for one language and load the result.

    Returns SCIPIndex on success, None if indexer not available.
    """
    from .scip_loader import load_scip_artifact

    output_path = os.path.join(output_dir, f"{language}.scip")
    try:
        artifact_path = run_scip_indexer(language, repo_path, output_path)
        return load_scip_artifact(artifact_path)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        logger.warning("SCIP indexer for %s unavailable: %s", language, exc)
        return None
