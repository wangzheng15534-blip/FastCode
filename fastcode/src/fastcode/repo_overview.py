"""
Repository Overview Generator - Generate summaries and file structures for repositories
"""

import logging
import os
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

from .core import repo_analysis as _repo_analysis
from .llm_utils import openai_chat_completion
from .utils._compat import get_language_from_extension


class RepositoryOverviewGenerator:
    """Generate repository overviews from README files and file structure"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.gen_config = config.get("generation", {})
        self.logger = logging.getLogger(__name__)

        # Load environment variables
        load_dotenv()

        # LLM settings for overview generation
        self.provider: str = self.gen_config.get("provider", "openai")
        self.model: str | None = os.getenv("MODEL")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.base_url = os.getenv("BASE_URL")

        self.temperature = 0.3  # Lower temperature for factual summaries
        self.max_tokens = 1000  # Longer for overview generation

        # Initialize LLM client
        self.llm_client: Any = self._initialize_client()

    def _initialize_client(self) -> Any:
        """Initialize LLM client"""
        try:
            if self.provider == "openai":
                if not self.api_key:
                    self.logger.warning("OPENAI_API_KEY not set")
                    return None
                return OpenAI(api_key=self.api_key, base_url=self.base_url)

            if self.provider == "anthropic":
                if not self.anthropic_api_key:
                    self.logger.warning("ANTHROPIC_API_KEY not set")
                    return None
                return Anthropic(api_key=self.anthropic_api_key, base_url=self.base_url)

            self.logger.warning(f"Unknown provider: {self.provider}")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client: {e}")
            return None

    def generate_overview(
        self, repo_path: str, repo_name: str, file_structure: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate comprehensive repository overview

        Args:
            repo_path: Path to repository
            repo_name: Name of the repository
            file_structure: Parsed file structure information

        Returns:
            Dictionary with overview information
        """
        self.logger.info(f"Generating overview for repository: {repo_name}")

        # Find and read README file
        readme_content = self._find_and_read_readme(repo_path)

        # Generate overview from README if available
        if readme_content and self.llm_client:
            summary = self._summarize_readme_with_llm(
                repo_name, readme_content, file_structure
            )
        else:
            # Fallback: generate overview from file structure only
            summary = self._generate_structure_based_overview(repo_name, file_structure)

        # Generate detailed file structure text
        structure_text = self._format_file_structure(file_structure)

        overview = {
            "repo_name": repo_name,
            "summary": summary,
            "readme_content": readme_content[:10000]
            if readme_content
            else None,  # Truncate long READMEs
            "file_structure": file_structure,
            "structure_text": structure_text,
            "has_readme": readme_content is not None,
        }

        return overview

    def _find_and_read_readme(self, repo_path: str) -> str | None:
        """Find and read README file in repository"""
        # Support common README variants across ecosystems (case-insensitive, multiple extensions)
        readme_names = [
            "README.md",
            "readme.md",
            "README.MD",
            "Readme.md",
            "README.rst",
            "readme.rst",
            "README.RST",
            "Readme.rst",
            "README.txt",
            "readme.txt",
            "README.TXT",
            "README.markdown",
            "readme.markdown",
            "README.MARKDOWN",
            "README.mdown",
            "readme.mdown",
            "README",
            "readme",
        ]

        for readme_name in readme_names:
            readme_path = os.path.join(repo_path, readme_name)
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, encoding="utf-8") as f:
                        content = f.read()
                    self.logger.debug(f"Found README: {readme_name}")
                    return content
                except Exception as e:
                    self.logger.warning(f"Failed to read {readme_name}: {e}")
                    continue

        self.logger.debug("No README file found")
        return None

    def parse_file_structure(
        self, repo_path: str, files: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Parse repository file structure from scanned files

        Args:
            repo_path: Path to repository
            files: List of file metadata from loader

        Returns:
            Structured representation of repository
        """
        structure: dict[str, Any] = {
            "total_files": len(files),
            "languages": {},
            "directories": {},
            "file_types": {},
            "all_files": [],
            "key_files": [],
        }

        for file_info in files:
            rel_path: str = file_info.get("relative_path", "")
            extension: str = file_info.get("extension", "")
            structure["all_files"].append(rel_path)

            # Count by extension
            if extension not in structure["file_types"]:
                structure["file_types"][extension] = 0
            structure["file_types"][extension] += 1

            # Count by language (simple heuristic)
            language = get_language_from_extension(extension)
            if language != "unknown":
                if language not in structure["languages"]:
                    structure["languages"][language] = 0
                structure["languages"][language] += 1

            # Build directory tree
            dir_name = os.path.dirname(rel_path)
            if dir_name:
                dir_parts = dir_name.split(os.sep)
                for i in range(len(dir_parts)):
                    partial_dir = os.sep.join(dir_parts[: i + 1])
                    if partial_dir not in structure["directories"]:
                        structure["directories"][partial_dir] = []

                    file_name = os.path.basename(rel_path)
                    if i == len(dir_parts) - 1:  # This is the file's direct parent
                        if file_name not in structure["directories"][partial_dir]:
                            structure["directories"][partial_dir].append(file_name)

            # Identify key files
            if _repo_analysis.is_key_file(rel_path):
                structure["key_files"].append(rel_path)

        return structure

    def _summarize_readme_with_llm(
        self, repo_name: str, readme_content: str, file_structure: dict[str, Any]
    ) -> str:
        """Use LLM to summarize README and infer repository purpose"""

        # Truncate very long READMEs
        if len(readme_content) > 8000:
            readme_content = readme_content[:8000] + "\n... (truncated)"

        # Format file structure info
        languages = ", ".join(file_structure.get("languages", {}).keys())
        total_files = file_structure.get("total_files", 0)

        prompt = f"""Analyze this repository and provide a concise summary of its main purpose and functionality.

Repository Name: {repo_name}
Total Files: {total_files}
Languages: {languages}

README Content:
{readme_content}

Provide a 3-5 sentence summary that covers:
1. What this repository/project does (main purpose)
2. Key technologies or frameworks used
3. Primary use cases or target users
4. Main components or modules (if identifiable)

Summary:"""

        try:
            if self.provider == "openai":
                response = openai_chat_completion(
                    self.llm_client,
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()

            if self.provider == "anthropic" and self.llm_client is not None:
                response = self.llm_client.messages.create(
                    model=self.model or "claude-sonnet-4-6",
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                text_block = response.content[0]
                return str(text_block.text).strip()

        except Exception as e:
            self.logger.error(f"LLM summarization failed: {e}")
            return self._generate_structure_based_overview(repo_name, file_structure)

        return self._generate_structure_based_overview(repo_name, file_structure)

    def _generate_structure_based_overview(
        self, repo_name: str, file_structure: dict[str, Any]
    ) -> str:
        """Generate overview based on file structure when README is unavailable"""
        return _repo_analysis.generate_structure_based_overview(
            repo_name, file_structure
        )

    def _infer_project_type(
        self, key_files: list[str], languages: dict[str, int]
    ) -> str:
        """Infer project type from files and languages"""
        return _repo_analysis.infer_project_type(key_files, languages)

    def _format_file_structure(self, file_structure: dict[str, Any]) -> str:
        """Format file structure as readable text"""
        return _repo_analysis.format_file_structure(file_structure)
