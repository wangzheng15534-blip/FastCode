"""Facade container and factory for injection into entry frames.

Holds the seven facade instances plus orchestration methods that span
multiple facades.  Created by the composition root (main/) and injected
into entry frames (api/, mcp/) so they never import from main/ directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastcode.app.indexing.facade import IndexingFacade
from fastcode.app.indexing.projection_facade import ProjectionFacade
from fastcode.app.indexing.publishing_facade import PublishingFacade
from fastcode.app.query.facade import QueryFacade
from fastcode.app.store.cache_facade import CacheFacade
from fastcode.app.store.context_facade import ContextFacade
from fastcode.app.store.facade import StoreFacade


@dataclass
class FacadeContainer:
    """Narrow facade holder injected into entry frames.

    Entry frames access attributes via duck typing — no import of this
    type from main/ is needed.
    """

    indexing: IndexingFacade
    query: QueryFacade
    store: StoreFacade
    context: ContextFacade
    cache: CacheFacade
    projection: ProjectionFacade
    publishing: PublishingFacade

    # Orchestration callbacks — wired to FastCode methods
    _ensure_repos_ready_fn: Callable[..., list[str]]
    _ensure_loaded_fn: Callable[[list[str]], bool]
    _remove_repository_fn: Callable[[str, bool], dict[str, Any]]
    _build_diagnostic_bundle_fn: Callable[[], dict[str, Any]]
    _apply_env_ignore_patterns_fn: Callable[[], None]
    _apply_runtime_overrides_fn: Callable[..., None]
    _shutdown_fn: Callable[[], None]

    # Orchestration -------------------------------------------------------

    def ensure_repos_ready(
        self, repos: list[str], *, allow_incremental: bool = True
    ) -> list[str]:
        return self._ensure_repos_ready_fn(repos, allow_incremental=allow_incremental)

    def ensure_loaded(self, repo_names: list[str]) -> bool:
        return self._ensure_loaded_fn(repo_names)

    def remove_repository(
        self, repo_name: str, delete_source: bool = True
    ) -> dict[str, Any]:
        return self._remove_repository_fn(repo_name, delete_source)

    def build_diagnostic_bundle(self) -> dict[str, Any]:
        return self._build_diagnostic_bundle_fn()

    def apply_env_ignore_patterns(self) -> None:
        self._apply_env_ignore_patterns_fn()

    def apply_repository_runtime_overrides(
        self,
        *,
        ignore_patterns: tuple[str, ...] | None = None,
        exclude_site_packages: bool | None = None,
    ) -> None:
        self._apply_runtime_overrides_fn(
            ignore_patterns=ignore_patterns,
            exclude_site_packages=exclude_site_packages,
        )

    def shutdown(self) -> None:
        self._shutdown_fn()


def facade_container_from_fastcode(fc: Any) -> FacadeContainer:
    """Extract a FacadeContainer from an existing FastCode instance.

    This is the bridge function called by the composition root to create
    the narrow container that gets injected into entry frames.
    """
    return FacadeContainer(
        indexing=fc.indexing,
        query=fc.query,
        store=fc.store,
        context=fc.context,
        cache=fc.cache,
        projection=fc.projection,
        publishing=fc.publishing,
        _ensure_repos_ready_fn=fc.ensure_repos_ready,
        _ensure_loaded_fn=fc.ensure_loaded,
        _remove_repository_fn=fc.remove_repository,
        _build_diagnostic_bundle_fn=fc.build_diagnostic_bundle,
        _apply_env_ignore_patterns_fn=fc.apply_env_ignore_patterns,
        _apply_runtime_overrides_fn=fc.apply_repository_runtime_overrides,
        _shutdown_fn=fc.shutdown,
    )
