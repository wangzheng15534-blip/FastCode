"""Tests for fastcode.infrastructure.llm — error handling and response extraction."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fastcode.infrastructure.llm import chat_completion


def _mock_client(
    choice_content: str | None = "hello",
    *,
    empty_choices: bool = False,
    none_message: bool = False,
) -> MagicMock:
    """Build a mock OpenAI client with configurable response shape."""
    mock = MagicMock()
    response = MagicMock()
    if empty_choices:
        response.choices = []
    else:
        choice = MagicMock()
        if none_message:
            choice.message = None
        else:
            choice.message.content = choice_content
        response.choices = [choice]
    mock.chat.completions.create.return_value = response
    return mock


class TestChatCompletion:
    def test_extracts_content_from_response(self):
        client = _mock_client("test response text")
        result = chat_completion(
            client,
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4",
            max_tokens=100,
            temperature=0.5,
        )
        assert result == "test response text"

    def test_passes_parameters_to_client(self):
        client = _mock_client("ok")
        chat_completion(
            client,
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4",
            max_tokens=200,
            temperature=0.7,
        )
        client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=200,
            temperature=0.7,
        )

    def test_raises_on_empty_choices(self):
        client = _mock_client(empty_choices=True)
        with pytest.raises(IndexError):
            chat_completion(
                client,
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4",
                max_tokens=100,
                temperature=0.5,
            )

    def test_raises_on_none_message(self):
        client = _mock_client(none_message=True)
        with pytest.raises(AttributeError):
            chat_completion(
                client,
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4",
                max_tokens=100,
                temperature=0.5,
            )

    def test_returns_none_content_as_none(self):
        """If the API returns None content, chat_completion returns None."""
        client = _mock_client(None)
        result = chat_completion(
            client,
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4",
            max_tokens=100,
            temperature=0.5,
        )
        assert result is None
