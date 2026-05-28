"""Root inbound config DTO composed from section-local DTOs."""

from __future__ import annotations

from pydantic import Field

from ._config_schema_base import _ConfigDTO
from ._config_schema_indexing import (
    DocsIntegrationConfigDTO,
    EmbeddingConfigDTO,
    GraphConfigDTO,
    IndexingConfigDTO,
    ParserConfigDTO,
    RepositoryConfigDTO,
)
from ._config_schema_operations import (
    EvaluationConfigDTO,
    LoggingConfigDTO,
    TerminusConfigDTO,
)
from ._config_schema_persistence import (
    CacheConfigDTO,
    ProjectionConfigDTO,
    StorageConfigDTO,
    VectorStoreConfigDTO,
)
from ._config_schema_querying import (
    AgentConfigDTO,
    GenerationConfigDTO,
    QueryConfigDTO,
    RetrievalConfigDTO,
)


class FastCodeConfigDTO(_ConfigDTO):
    repo_root: str = "./repos"
    storage: StorageConfigDTO = Field(default_factory=StorageConfigDTO)
    repository: RepositoryConfigDTO = Field(default_factory=RepositoryConfigDTO)
    parser: ParserConfigDTO = Field(default_factory=ParserConfigDTO)
    embedding: EmbeddingConfigDTO = Field(default_factory=EmbeddingConfigDTO)
    indexing: IndexingConfigDTO = Field(default_factory=IndexingConfigDTO)
    vector_store: VectorStoreConfigDTO = Field(default_factory=VectorStoreConfigDTO)
    retrieval: RetrievalConfigDTO = Field(default_factory=RetrievalConfigDTO)
    generation: GenerationConfigDTO = Field(default_factory=GenerationConfigDTO)
    query: QueryConfigDTO = Field(default_factory=QueryConfigDTO)
    graph: GraphConfigDTO = Field(default_factory=GraphConfigDTO)
    docs_integration: DocsIntegrationConfigDTO = Field(
        default_factory=DocsIntegrationConfigDTO
    )
    agent: AgentConfigDTO = Field(default_factory=AgentConfigDTO)
    cache: CacheConfigDTO = Field(default_factory=CacheConfigDTO)
    evaluation: EvaluationConfigDTO = Field(default_factory=EvaluationConfigDTO)
    logging: LoggingConfigDTO = Field(default_factory=LoggingConfigDTO)
    terminus: TerminusConfigDTO = Field(default_factory=TerminusConfigDTO)
    projection: ProjectionConfigDTO = Field(default_factory=ProjectionConfigDTO)
