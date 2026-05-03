"""Indexing orchestration exports, loaded lazily to avoid import cycles."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .doc_ingester import KeyDocIngester
    from .embedder import CodeEmbedder
    from .global_builder import GlobalIndexBuilder
    from .incremental import apply_incremental_update, diff_changed_files
    from .indexer import CodeIndexer
    from .loader import RepositoryLoader
    from .overview import RepositoryOverviewGenerator
    from .parser import CodeParser
    from .pipeline import IndexPipeline
    from .projection import ProjectionService
    from .projection_transform import ProjectionTransformer
    from .publishing import PublishingService
    from .redo_worker import RedoWorker
    from .terminus import TerminusPublisher

_EXPORTS = {
    "CodeEmbedder": ("fastcode.indexing.embedder", "CodeEmbedder"),
    "CodeIndexer": ("fastcode.indexing.indexer", "CodeIndexer"),
    "CodeParser": ("fastcode.indexing.parser", "CodeParser"),
    "GlobalIndexBuilder": ("fastcode.indexing.global_builder", "GlobalIndexBuilder"),
    "IndexPipeline": ("fastcode.indexing.pipeline", "IndexPipeline"),
    "KeyDocIngester": ("fastcode.indexing.doc_ingester", "KeyDocIngester"),
    "ProjectionService": ("fastcode.indexing.projection", "ProjectionService"),
    "ProjectionTransformer": (
        "fastcode.indexing.projection_transform",
        "ProjectionTransformer",
    ),
    "PublishingService": ("fastcode.indexing.publishing", "PublishingService"),
    "RedoWorker": ("fastcode.indexing.redo_worker", "RedoWorker"),
    "RepositoryLoader": ("fastcode.indexing.loader", "RepositoryLoader"),
    "RepositoryOverviewGenerator": (
        "fastcode.indexing.overview",
        "RepositoryOverviewGenerator",
    ),
    "TerminusPublisher": ("fastcode.indexing.terminus", "TerminusPublisher"),
    "apply_incremental_update": (
        "fastcode.indexing.incremental",
        "apply_incremental_update",
    ),
    "diff_changed_files": ("fastcode.indexing.incremental", "diff_changed_files"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CodeEmbedder",
    "CodeIndexer",
    "CodeParser",
    "GlobalIndexBuilder",
    "IndexPipeline",
    "KeyDocIngester",
    "ProjectionService",
    "ProjectionTransformer",
    "PublishingService",
    "RedoWorker",
    "RepositoryLoader",
    "RepositoryOverviewGenerator",
    "TerminusPublisher",
    "apply_incremental_update",
    "diff_changed_files",
]
