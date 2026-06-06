"""Tests for the env var registry and read_env()."""

from __future__ import annotations

import os
import warnings

import pytest

from fastcode.main._env_registry import ENV_REGISTRY, read_env, validate_env_vars


# ---------------------------------------------------------------------------
# Registry content
# ---------------------------------------------------------------------------

class TestRegistryContent:
    def test_all_canonical_names_are_registered(self) -> None:
        expected = [
            "FASTCODE_FORCE_LLM_MODEL",
            "FASTCODE_FORCE_LLM_BASE_URL",
            "FASTCODE_FORCE_REDIS_HOST",
            "FASTCODE_FORCE_REDIS_PORT",
            "FASTCODE_STORAGE_BACKEND",
            "FASTCODE_POSTGRES_DSN",
            "FASTCODE_PROJECTION_POSTGRES_DSN",
            "FASTCODE_FORCE_EXCLUDE_SITE_PACKAGES",
            "FASTCODE_CORS_ALLOW_ORIGINS",
            "FASTCODE_CORS_ALLOW_CREDENTIALS",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ]
        for name in expected:
            assert name in ENV_REGISTRY, f"Missing: {name}"

    def test_old_names_have_aliases(self) -> None:
        assert ENV_REGISTRY["FASTCODE_FORCE_LLM_MODEL"].old_names == ("MODEL",)
        assert ENV_REGISTRY["FASTCODE_FORCE_LLM_BASE_URL"].old_names == ("BASE_URL",)
        assert ENV_REGISTRY["FASTCODE_FORCE_REDIS_HOST"].old_names == ("REDIS_HOST",)
        assert ENV_REGISTRY["FASTCODE_FORCE_REDIS_PORT"].old_names == ("REDIS_PORT",)
        assert ENV_REGISTRY["FASTCODE_FORCE_EXCLUDE_SITE_PACKAGES"].old_names == (
            "FASTCODE_EXCLUDE_SITE_PACKAGES",
        )

    def test_third_party_vars_have_no_prefix(self) -> None:
        assert ENV_REGISTRY["OPENAI_API_KEY"].category == "THIRD_PARTY"
        assert ENV_REGISTRY["ANTHROPIC_API_KEY"].category == "THIRD_PARTY"


# ---------------------------------------------------------------------------
# read_env
# ---------------------------------------------------------------------------

class TestReadEnv:
    def test_returns_canonical_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FASTCODE_FORCE_LLM_MODEL", "gpt-4o")
        assert read_env("FASTCODE_FORCE_LLM_MODEL") == "gpt-4o"

    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FASTCODE_FORCE_LLM_MODEL", raising=False)
        monkeypatch.delenv("MODEL", raising=False)
        assert read_env("FASTCODE_FORCE_LLM_MODEL") is None

    def test_old_name_fallback_with_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FASTCODE_FORCE_LLM_MODEL", raising=False)
        monkeypatch.setenv("MODEL", "gpt-4")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = read_env("FASTCODE_FORCE_LLM_MODEL")
        assert result == "gpt-4"
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "MODEL" in str(w[0].message)
        assert "FASTCODE_FORCE_LLM_MODEL" in str(w[0].message)

    def test_canonical_name_takes_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FASTCODE_FORCE_LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("MODEL", "gpt-4")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = read_env("FASTCODE_FORCE_LLM_MODEL")
        assert result == "gpt-4o"
        assert len(w) == 0

    def test_third_party_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert read_env("OPENAI_API_KEY") == "sk-test"

    def test_unregistered_var_still_reads_os_environ(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FASTCODE_TOTALLY_FAKE", "x")
        assert read_env("FASTCODE_TOTALLY_FAKE") == "x"


# ---------------------------------------------------------------------------
# validate_env_vars
# ---------------------------------------------------------------------------

class TestValidateEnvVars:
    def test_deprecated_old_name_flagged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FASTCODE_FORCE_LLM_MODEL", raising=False)
        monkeypatch.setenv("MODEL", "gpt-4")
        results = validate_env_vars()
        assert any("MODEL" in r and "deprecated" in r.lower() for r in results)

    def test_clean_env_no_warnings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in list(os.environ):
            if key.startswith("FASTCODE_") or key in (
                "MODEL",
                "BASE_URL",
                "REDIS_HOST",
                "REDIS_PORT",
            ):
                monkeypatch.delenv(key, raising=False)
        results = validate_env_vars()
        fastcode_issues = [r for r in results if "FASTCODE_" in r]
        assert len(fastcode_issues) == 0
