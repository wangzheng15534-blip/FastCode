"""Tests for the immutable configuration boundary."""

from __future__ import annotations

import dataclasses

from fastcode.schemas.config import FastCodeConfig, config_from_mapping


def test_config_from_mapping_applies_defaults_and_normalizes_nested_sections() -> None:
    config = config_from_mapping(
        {
            "repo_root": "/tmp/repos",
            "storage": {"backend": "postgres", "pool_min": 2, "pool_max": 1},
            "repository": {
                "ignore_patterns": ["a", "b"],
                "exclude_site_packages": True,
            },
            "embedding": {"batch_size": 64},
            "vector_store": {"shard_storage": "mmap"},
            "generation": {"base_url": "https://llm.example/api", "model": "cfg-model"},
            "cache": {
                "enabled": False,
                "redis_host": "redis.internal",
                "redis_port": 6380,
            },
            "projection": {"backend": "postgres", "postgres_dsn": "postgresql://proj"},
        }
    )

    assert isinstance(config, FastCodeConfig)
    assert config.repo_root == "/tmp/repos"
    assert config.storage.backend == "postgres"
    assert config.storage.pool_min == 2
    assert config.storage.pool_max == 2
    assert config.projection.backend == "postgres"
    assert config.projection.postgres_dsn == "postgresql://proj"
    assert config.repository.exclude_site_packages is True
    assert config.repository.ignore_patterns == ("a", "b")
    assert config.embedding.batch_size == 64
    assert config.vector_store.shard_storage == "npy"
    assert config.generation.model == "cfg-model"
    assert config.generation.base_url == "https://llm.example/api"
    assert config.cache.redis_host == "redis.internal"
    assert config.cache.redis_port == 6380
    assert config.cache.enabled is False


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


def test_to_dict_produces_legacy_compatible_mapping() -> None:
    config = config_from_mapping(
        {
            "repo_root": "/tmp/repos",
            "repository": {"supported_extensions": [".py", ".ts"]},
        }
    )

    legacy = config.to_dict()

    assert legacy["repo_root"] == "/tmp/repos"
    assert legacy["repository"]["supported_extensions"] == (".py", ".ts")
