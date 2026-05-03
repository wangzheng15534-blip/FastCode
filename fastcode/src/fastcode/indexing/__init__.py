"""indexing/ -- Pipeline orchestration: extraction, embedding, indexing, publishing."""

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
