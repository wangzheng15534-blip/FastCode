"""Thin wrappers for LLM API I/O."""
from __future__ import annotations

from typing import Any


def chat_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float = 0.3,
    **kwargs: Any,
) -> str:
    """Single LLM completion. Returns response content string."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content
