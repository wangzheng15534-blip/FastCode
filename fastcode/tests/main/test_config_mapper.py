"""Tests for inbound config DTO to runtime contract mapping."""

from __future__ import annotations

import dataclasses

import pytest

from fastcode.kernel.config import (
    DocsIntegrationConfig,
    FastCodeConfig,
    IndexingConfig,
    IndexingLevel,
    LocalSourceMode,
    ProjectionConfig,
    StorageBackend,
    StorageConfig,
    TerminusConfig,
    VectorShardStorage,
    VectorStoreConfig,
)
from fastcode.main._config_schema_root import FastCodeConfigDTO
from fastcode.main.config import (
    config_from_dto,
    config_from_mapping,
    config_to_dict,
)


def test_config_from_mapping_translates_validated_dto_to_frozen_runtime_config() -> (
    None
):
    config = config_from_mapping(
        {
            "repo_root": "/tmp/repos",
            "storage": {"backend": "postgres", "pool_min": 2, "pool_max": 1},
            "repository": {
                "ignore_patterns": ["a", "b"],
                "exclude_site_packages": True,
            },
            "embedding": {"batch_size": 64},
            "indexing": {"allow_direct_index": True},
            "vector_store": {"shard_storage": "npy"},
            "docs_integration": {"chunk_size": 300, "chunk_overlap": 30},
            "terminus": {"timeout_seconds": 30},
            "generation": {"base_url": "https://llm.example/api", "model": "cfg-model"},
            "cache": {
                "enabled": False,
                "redis_host": "redis.internal",
                "redis_port": 6380,
            },
            "projection": {
                "postgres_dsn": "postgresql://proj",
                "enable_leiden": False,
                "max_query_hops": 3,
            },
        }
    )

    assert isinstance(config, FastCodeConfig)
    assert config.repo_root == "/tmp/repos"
    assert config.storage.backend is StorageBackend.POSTGRES
    assert config.storage.backend == "postgres"
    assert config.storage.pool_min == 2
    assert config.storage.pool_max == 2
    assert isinstance(config.projection, ProjectionConfig)
    assert config.projection.postgres_dsn == "postgresql://proj"
    assert config.projection.enable_leiden is False
    assert config.projection.max_query_hops == 3
    assert config.repository.exclude_site_packages is True
    assert config.repository.ignore_patterns == ("a", "b")
    assert config.embedding.batch_size == 64
    assert isinstance(config.indexing, IndexingConfig)
    assert config.indexing.allow_direct_index is True
    assert config.indexing.levels == (
        IndexingLevel.FILE,
        IndexingLevel.CLASS,
        IndexingLevel.FUNCTION,
        IndexingLevel.DOCUMENTATION,
    )
    assert config.vector_store.shard_storage is VectorShardStorage.NPY
    assert isinstance(config.docs_integration, DocsIntegrationConfig)
    assert config.docs_integration.chunk_size == 300
    assert config.docs_integration.chunk_overlap == 30
    assert isinstance(config.terminus, TerminusConfig)
    assert config.terminus.timeout_seconds == 30
    assert config.generation.model == "cfg-model"
    assert config.generation.base_url == "https://llm.example/api"
    assert config.cache.redis_host == "redis.internal"
    assert config.cache.redis_port == 6380
    assert config.cache.enabled is False


def test_config_from_dto_maps_validated_external_meaning_explicitly() -> None:
    dto = FastCodeConfigDTO.model_validate(
        {
            "repository": {"supported_extensions": [".py", ".ts"]},
            "vector_store": {"shard_storage": "compressed"},
            "graph": {"ladybug": {"enabled": True, "db_path": "/tmp/graph.lb"}},
            "agent": {"iterative": {"max_candidates_display": 200}},
        }
    )

    config = config_from_dto(dto)

    assert config.repository.supported_extensions == (".py", ".ts")
    assert config.repository.local_source_mode is LocalSourceMode.IN_PLACE
    assert config.vector_store.shard_storage is VectorShardStorage.COMPRESSED
    assert config.graph.ladybug.enabled is True
    assert config.graph.ladybug.db_path == "/tmp/graph.lb"
    assert config.agent.iterative.max_candidates_display == 200


def test_runtime_overrides_return_new_frozen_config_without_mutating_original() -> None:
    original = config_from_mapping(
        {
            "repo_root": "/tmp/original",
            "repository": {"ignore_patterns": ["base"], "exclude_site_packages": False},
            "cache": {"enabled": True},
            "evaluation": {"in_memory_index": False},
            "vector_store": {"in_memory": False},
        }
    )

    updated = original.with_runtime_overrides(
        repo_root="/tmp/updated",
        in_memory_index=True,
        cache_enabled=False,
        repository_ignore_patterns=("base", ".venv"),
        repository_exclude_site_packages=True,
    )

    assert dataclasses.is_dataclass(original)
    assert original.repo_root == "/tmp/original"
    assert original.evaluation.in_memory_index is False
    assert original.vector_store.in_memory is False
    assert original.cache.enabled is True
    assert original.repository.ignore_patterns == ("base",)
    assert original.repository.exclude_site_packages is False

    assert updated.repo_root == "/tmp/updated"
    assert updated.evaluation.in_memory_index is True
    assert updated.vector_store.in_memory is True
    assert updated.cache.enabled is False
    assert updated.repository.ignore_patterns == ("base", ".venv")
    assert updated.repository.exclude_site_packages is True
    assert updated is not original


def test_config_to_dict_produces_runtime_mapping() -> None:
    config = config_from_mapping(
        {
            "repo_root": "/tmp/repos",
            "repository": {"supported_extensions": [".py", ".ts"]},
        }
    )

    runtime_mapping = config_to_dict(config)

    assert runtime_mapping["repo_root"] == "/tmp/repos"
    assert runtime_mapping["storage"]["backend"] == "sqlite"
    assert runtime_mapping["repository"]["supported_extensions"] == (".py", ".ts")
    assert runtime_mapping["indexing"]["levels"] == (
        IndexingLevel.FILE,
        IndexingLevel.CLASS,
        IndexingLevel.FUNCTION,
        IndexingLevel.DOCUMENTATION,
    )


def test_config_to_dict_avoids_runtime_config_generic_serializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_from_mapping(
        {
            "repo_root": "/tmp/repos",
            "projection": {"edge_weights": {"call": 2.0, "contain": 4.0}},
        }
    )

    def _boom_to_dict(_: FastCodeConfig) -> dict[str, object]:
        raise AssertionError("config_to_dict must not call FastCodeConfig.to_dict()")

    monkeypatch.setattr(FastCodeConfig, "to_dict", _boom_to_dict)

    runtime_mapping = config_to_dict(config)

    assert runtime_mapping["repo_root"] == "/tmp/repos"
    assert runtime_mapping["projection"]["edge_weights"] == {
        "call": 2.0,
        "contain": 4.0,
    }


def test_runtime_config_contracts_reject_invalid_internal_invariants() -> None:
    with pytest.raises(ValueError, match=r"storage\.pool_max"):
        StorageConfig(pool_min=4, pool_max=3)
    with pytest.raises(ValueError, match=r"vector_store\.ef_search"):
        VectorStoreConfig(ef_search=0)
    with pytest.raises(ValueError, match=r"docs_integration\.chunk_overlap"):
        DocsIntegrationConfig(chunk_size=10, chunk_overlap=10)
