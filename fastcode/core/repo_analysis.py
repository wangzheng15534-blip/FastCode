"""Pure repo analysis functions — extracted from repo_overview.py."""

from __future__ import annotations

import os
from typing import Any


def get_language_from_extension(ext: str) -> str:
    """Get programming language from extension."""
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".cpp": "cpp",
        ".c": "c",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
    }
    return language_map.get(ext.lower(), "unknown")


def is_key_file(file_path: str) -> bool:
    """Check if file is a key/important file."""
    file_name = os.path.basename(file_path).lower()
    key_names = [
        "main",
        "index",
        "app",
        "init",
        "config",
        "setup",
        "package.json",
        "requirements.txt",
        "go.mod",
        "cargo.toml",
        "dockerfile",
        "makefile",
        "cmakelists.txt",
    ]

    return any(key in file_name for key in key_names)


def infer_project_type(key_files: list[str], languages: dict[str, int]) -> str:
    """Infer project type from files and languages."""
    key_files_str = " ".join(key_files).lower()

    # Web frameworks
    if "package.json" in key_files_str:
        if "react" in key_files_str or "tsx" in languages:
            return "React web application"
        if "vue" in key_files_str:
            return "Vue.js web application"
        return "Node.js application"

    # Python projects
    if "requirements.txt" in key_files_str or "setup.py" in key_files_str:
        if "django" in key_files_str:
            return "Django web application"
        if "flask" in key_files_str:
            return "Flask web application"
        return "Python application"

    # Mobile
    if "android" in key_files_str or "java" in languages:
        return "Android application"
    if "ios" in key_files_str or "swift" in key_files_str:
        return "iOS application"

    # Containers
    if "dockerfile" in key_files_str:
        return "containerized application"

    # Default
    return "software project"


def generate_structure_based_overview(
    repo_name: str,
    file_structure: dict[str, Any],
) -> str:
    """Generate overview based on file structure when README is unavailable."""
    languages = file_structure.get("languages", {})
    total_files = file_structure.get("total_files", 0)
    key_files = file_structure.get("key_files", [])

    # Determine primary language
    if languages:
        primary_lang = max(languages.items(), key=lambda x: x[1])[0]
    else:
        primary_lang = "unknown"

    # Infer project type from key files
    project_type = infer_project_type(key_files, languages)

    summary = (
        f"{repo_name} is a {primary_lang} {project_type} with {total_files} files. "
    )

    if len(languages) > 1:
        lang_list = ", ".join(languages.keys())
        summary += f"It uses multiple languages: {lang_list}. "

    if key_files:
        summary += f"Key entry points include: {', '.join(key_files[:5])}."

    return summary


def format_file_structure(file_structure: dict[str, Any]) -> str:
    """Format file structure as readable text."""
    lines: list[str] = []

    # Summary
    total_files = file_structure.get("total_files", 0)
    lines.append(f"Total Files: {total_files}")

    # Languages
    languages = file_structure.get("languages", {})
    if languages:
        lines.append("\nLanguages:")
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {lang}: {count} files")

    # All files
    all_files = file_structure.get("all_files", [])
    if all_files:
        lines.append("\nFiles:")
        for file_path in sorted(all_files):
            lines.append(f"  - {file_path}")

    # Top-level directories
    directories = file_structure.get("directories", {})
    top_dirs = [d for d in directories if os.sep not in d or d.count(os.sep) == 0]
    if top_dirs:
        lines.append("\nTop-Level Directories:")
        for td in sorted(top_dirs)[:15]:  # Limit to 15
            file_count = len(directories[td])
            lines.append(f"  - {td}/ ({file_count} files)")

    return "\n".join(lines)
