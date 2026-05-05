from __future__ import annotations

import pytest

from fastcode.api.cors import cors_middleware_options


def test_cors_defaults_are_not_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTCODE_CORS_ALLOW_ORIGINS", raising=False)
    monkeypatch.delenv("FASTCODE_CORS_ALLOW_CREDENTIALS", raising=False)

    options = cors_middleware_options()

    assert "*" not in options["allow_origins"]
    assert options["allow_credentials"] is False


def test_cors_wildcard_disables_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTCODE_CORS_ALLOW_ORIGINS", "*")
    monkeypatch.setenv("FASTCODE_CORS_ALLOW_CREDENTIALS", "true")

    options = cors_middleware_options()

    assert options["allow_origins"] == ["*"]
    assert options["allow_credentials"] is False
