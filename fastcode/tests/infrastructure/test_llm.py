"""Tests for LLM effects — verify thin wrapper behavior."""

from unittest.mock import MagicMock

from fastcode.infrastructure.llm import chat_completion


class TestChatCompletion:
    def test_returns_content_string(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_client.chat.completions.create.return_value = mock_response

        result = chat_completion(
            mock_client,
            model="test-model",
            messages=[],
            max_tokens=100,
        )
        assert result == "test response"
