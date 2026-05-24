"""Inbound DTOs for repository, parsing, indexing, graph, and docs config."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from ._config_schema_base import _ConfigDTO


def _default_indexing_levels() -> list[IndexingLevelDTO]:
    return [
        IndexingLevelDTO.FILE,
        IndexingLevelDTO.CLASS,
        IndexingLevelDTO.FUNCTION,
        IndexingLevelDTO.DOCUMENTATION,
    ]


def _default_curated_doc_paths() -> list[str]:
    return [
        "README*",
        "docs/design/**",
        "docs/research/**",
        "docs/adr/**",
        "docs/rfc/**",
    ]


class LocalSourceModeDTO(StrEnum):
    IN_PLACE = "in_place"
    COPY = "copy"
    HARDLINK = "hardlink"


class IndexingLevelDTO(StrEnum):
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    DOCUMENTATION = "documentation"


class RepositoryConfigDTO(_ConfigDTO):
    clone_depth: int = Field(default=1, ge=1)
    max_file_size_mb: int = Field(default=5, ge=1)
    backup_directory: str = "./repo_backup"
    local_source_mode: LocalSourceModeDTO = LocalSourceModeDTO.IN_PLACE
    exclude_site_packages: bool = False
    ignore_patterns: list[str] = Field(default_factory=list)
    supported_extensions: list[str] = Field(default_factory=list)


class ParserConfigDTO(_ConfigDTO):
    extract_docstrings: bool = True
    extract_comments: bool = True
    extract_imports: bool = True
    compute_complexity: bool = True
    max_function_lines: int = Field(default=1000, ge=1)


class EmbeddingConfigDTO(_ConfigDTO):
    provider: str = "ollama"
    model: str = "bge-large-en-v1.5"
    ollama_url: str = "http://127.0.0.1:11434/api/embeddings"
    device: str = "cpu"
    batch_size: int = Field(default=32, ge=1)
    max_seq_length: int = Field(default=512, ge=1)
    normalize_embeddings: bool = True


class IndexingConfigDTO(_ConfigDTO):
    levels: list[IndexingLevelDTO] = Field(default_factory=_default_indexing_levels)
    include_imports: bool = True
    include_class_context: bool = True
    generate_repo_overview: bool = True
    allow_direct_index: bool = False


class LadybugGraphConfigDTO(_ConfigDTO):
    enabled: bool = False
    db_path: str = "./data/ladybug/fastcode.lb"
    postgres_attach_dsn: str = ""


class GraphConfigDTO(_ConfigDTO):
    build_call_graph: bool = True
    build_dependency_graph: bool = True
    build_inheritance_graph: bool = True
    max_depth: int = Field(default=5, ge=1)
    ladybug: LadybugGraphConfigDTO = Field(default_factory=LadybugGraphConfigDTO)


class DocsIntegrationConfigDTO(_ConfigDTO):
    enabled: bool = False
    curated_paths: list[str] = Field(default_factory=_default_curated_doc_paths)
    allow_paths: list[str] = Field(default_factory=list)
    deny_paths: list[str] = Field(default_factory=list)
    chunk_token_size: int = Field(default=512, ge=1)
    similarity_threshold: float = Field(default=0.5, ge=0.0)
    chunk_size: int = Field(default=420, ge=1)
    chunk_overlap: int = Field(default=80, ge=0)
    max_chunk_chars: int = Field(default=2400, ge=1)
