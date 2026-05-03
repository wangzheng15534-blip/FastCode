"""
FastCode 2.0 - Repository-Level Code Understanding System
With Multi-Repository Support
"""

import os
import platform

if platform.system() == "Darwin":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

from .indexing.indexer import CodeIndexer
from .indexing.loader import RepositoryLoader
from .indexing.overview import RepositoryOverviewGenerator
from .indexing.parser import CodeParser
from .ir.types import (
    IRCodeUnit,
    IRDocument,
    IREdge,
    IROccurrence,
    IRRelation,
    IRSnapshot,
    IRSymbol,
    IRUnitEmbedding,
    IRUnitSupport,
)
from .main import FastCode
from .query.answer import AnswerGenerator
from .query.selector import RepositorySelector
from .retrieval.agent_tools import AgentTools
from .retrieval.hybrid import HybridRetriever
from .retrieval.iterative import IterativeAgent

__version__ = "2.0.0"

__all__ = [
    "AgentTools",
    "AnswerGenerator",
    "CodeIndexer",
    "CodeParser",
    "FastCode",
    "HybridRetriever",
    "IRCodeUnit",
    "IRDocument",
    "IREdge",
    "IROccurrence",
    "IRRelation",
    "IRSnapshot",
    "IRSymbol",
    "IRUnitEmbedding",
    "IRUnitSupport",
    "IterativeAgent",
    "RepositoryLoader",
    "RepositoryOverviewGenerator",
    "RepositorySelector",
]
