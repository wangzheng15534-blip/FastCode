"""Tests for fastcode.infrastructure.llm — error handling and retry contracts."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from fastcode.infrastructure.llm import chat_completion


def _client_returning(content: str | None) -> MagicMock:
    """Mock client whose create() returns the given content string."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


def _client_with_empty_choices() -> MagicMock:
    """Mock client whose create() returns an empty choices list."""
    client = MagicMock()
    response = MagicMock()
    response.choices = []
    client.chat.completions.create.return_value = response
    return client


class TestChatCompletion:
    """Contract tests: what goes in, what comes out."""

    def test_returns_content_string_unchanged(self):
        """Returns choices[0].message.content as-is when it's a string."""
        client = _client_returning("hello world")
        result = chat_completion(
            client,
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        )
        assert result == "hello world"

    def test_returns_none_when_api_returns_none_content(self):
        """Returns None when API returns None content (e.g. function call)."""
        client = _client_returning(None)
        result = chat_completion(
            client,
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        )
        assert result is None

    def test_raises_index_error_on_empty_choices(self):
        """Empty choices list causes IndexError — crash is the signal."""
        client = _client_with_empty_choices()
        with pytest.raises(IndexError):
            chat_completion(
                client,
                model="gpt-4",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )

    def test_forwards_required_kwargs_to_client(self):
        """Model, messages, max_tokens, temperature are all forwarded."""
        client = _client_returning("ok")
        chat_completion(
            client,
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=200,
            temperature=0.7,
        )
        client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=200,
            temperature=0.7,
        )

    def test_forwards_extra_kwargs_to_client(self):
        """Extra kwargs (stop, top_p, etc.) are passed through."""
        client = _client_returning("ok")
        chat_completion(
            client,
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0.3,
            stop=["\n"],
            top_p=0.9,
        )
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["stop"] == ["\n"]
        assert call_kwargs["top_p"] == 0.9

    def test_uses_default_temperature(self):
        """Temperature defaults to 0.3 when not specified."""
        client = _client_returning("ok")
        chat_completion(
            client,
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        )
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.3
