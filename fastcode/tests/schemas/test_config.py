"""Tests for inbound configuration DTO validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import fastcode.schemas.config as config_schema
from fastcode.schemas.config import (
    FastCodeConfigDTO,
    StorageBackendDTO,
    VectorShardStorageDTO,
)


def test_config_dto_validates_external_shape_and_keeps_external_meaning() -> None:
    dto = FastCodeConfigDTO.model_validate(
        {
            "repo_root": "/tmp/repos",
            "storage": {"backend": "postgres", "pool_min": 2, "pool_max": 1},
            "repository": {
                "ignore_patterns": ["a", "b"],
                "exclude_site_packages": True,
            },
            "embedding": {"batch_size": 64},
            "vector_store": {"shard_storage": "npy"},
            "generation": {"base_url": "https://llm.example/api", "model": "cfg-model"},
            "cache": {
                "enabled": False,
                "redis_host": "redis.internal",
                "redis_port": 6380,
            },
            "projection": {
                "postgres_dsn": "postgresql://proj",
                "max_query_hops": 3,
            },
        }
    )

    assert dto.repo_root == "/tmp/repos"
    assert dto.storage.backend is StorageBackendDTO.POSTGRES
    assert dto.storage.backend == "postgres"
    assert dto.storage.pool_min == 2
    assert dto.storage.pool_max == 1
    assert dto.repository.exclude_site_packages is True
    assert dto.repository.ignore_patterns == ["a", "b"]
    assert dto.embedding.batch_size == 64
    assert dto.vector_store.shard_storage is VectorShardStorageDTO.NPY
    assert dto.vector_store.shard_storage == "npy"
    assert dto.generation.model == "cfg-model"
    assert dto.cache.redis_port == 6380
    assert dto.projection.postgres_dsn == "postgresql://proj"
    assert dto.projection.max_query_hops == 3


def test_config_dto_rejects_invalid_external_values() -> None:
    with pytest.raises(ValidationError):
        FastCodeConfigDTO.model_validate({"storage": {"pool_min": 0}})
    with pytest.raises(ValidationError):
        FastCodeConfigDTO.model_validate({"storage": {"backend": "mysql"}})
    with pytest.raises(ValidationError):
        FastCodeConfigDTO.model_validate({"vector_store": {"shard_storage": "mmap"}})


def test_config_dto_module_does_not_construct_runtime_contracts() -> None:
    assert not hasattr(config_schema, "config_from_mapping")
    assert not hasattr(config_schema, "config_to_dict")
