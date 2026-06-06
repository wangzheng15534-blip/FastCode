"""Token counting helpers owned by the query shell."""

from __future__ import annotations

try:
    import tiktoken
except ModuleNotFoundError:
    tiktoken = None  # type: ignore[assignment]


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens with a lightweight fallback when tiktoken is unavailable."""
    if tiktoken is None:
        return max(1, len(text) // 4)
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """Truncate text to fit within a token budget."""
    if tiktoken is None:
        return text[: max_tokens * 4]
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])
