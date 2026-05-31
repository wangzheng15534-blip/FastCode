"""Generic execution tool traits for SCIP indexing and semantic helpers.

These are generic tool traits (effect_tool), not semantic business capabilities.
They were promoted from axis_surface (ports) because they reference effect_tool
types (SCIPIndex) and wrap generic subprocess execution rather than domain logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from fastcode.scip.models import SCIPIndex


class SemanticHelperRuntime(Protocol):
    """Runtime capability for executing semantic helper tools."""

    def find_executable(self, executable: str) -> str | None:
        """Resolve an executable path from the host environment."""
        ...

    def run(self, command: Sequence[str], *, cwd: str, timeout: int) -> Any:
        """Run a semantic helper command in a repository workspace."""
        ...


class ScipFileInfoView(Protocol):
    """Read-only file inventory view used for SCIP language detection."""

    def get(self, key: str, default: object = None) -> object:
        """Return a file-info value by key."""
        ...


class ScipIndexerProfileView(Protocol):
    """Read-only SCIP indexer profile."""

    @property
    def language(self) -> str:
        """Indexed language name."""
        ...

    @property
    def binary_name(self) -> str:
        """Executable name for the indexer."""
        ...

    @property
    def extra_args(self) -> Sequence[str]:
        """Extra CLI arguments passed before the output path."""
        ...

    @property
    def experimental(self) -> bool:
        """Whether the language support is experimental."""
        ...


class ScipIndexerRuntime(Protocol):
    """Runtime capability for executing external SCIP indexers."""

    def get_profile(self, language: str) -> ScipIndexerProfileView | None:
        """Return the configured SCIP indexer profile for a language."""
        ...

    def detect_languages_from_file_infos(
        self,
        file_infos: Sequence[ScipFileInfoView],
    ) -> tuple[str, ...]:
        """Detect candidate SCIP languages from repository file inventory."""
        ...

    def detect_languages_in_paths(
        self,
        repo_path: str,
        relative_paths: Sequence[str],
    ) -> tuple[str, ...]:
        """Detect candidate SCIP languages in selected repository paths."""
        ...

    def run_indexer(
        self,
        language: str,
        repo_path: str,
        output_path: str,
    ) -> str:
        """Run a SCIP indexer and return the produced artifact path."""
        ...

    def load_artifact(self, path: str) -> SCIPIndex:
        """Load a SCIP artifact produced by an indexer."""
        ...
