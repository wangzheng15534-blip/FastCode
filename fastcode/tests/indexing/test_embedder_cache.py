from __future__ import annotations

from typing import Any

import numpy as np

from fastcode.indexing.embedder import CodeEmbedder


class _MemoryCache:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self.values.get(key)

    def set(self, key: str, value: Any) -> bool:
        self.values[key] = value
        return True


class _CountingEmbedder(CodeEmbedder):
    def __init__(self, cache: _MemoryCache, *, model_name: str = "model-a") -> None:
        self.config = {}
        self.embedding_config = {}
        self.logger = _NullLogger()
        self.provider = "sentence_transformers"
        self.model_name = model_name
        self.device = "cpu"
        self.batch_size = 32
        self.max_seq_length = 512
        self.normalize = True
        self.ollama_url = "http://127.0.0.1:11434/api/embeddings"
        self.model = None
        self.embedding_dim = 3
        self._embedding_cache = cache
        self._embedding_cache_enabled = True
        self.raw_batches: list[list[str]] = []

    def _embed_batch_uncached(self, texts: list[str]) -> np.ndarray:
        self.raw_batches.append(list(texts))
        return np.asarray(
            [self._vector_for_text(text) for text in texts], dtype=np.float32
        )

    @staticmethod
    def _vector_for_text(text: str) -> list[float]:
        base = float(sum(ord(char) for char in text) % 1000)
        return [base, base + 1.0, base + 2.0]


class _NullLogger:
    def debug(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _element(name: str) -> dict[str, Any]:
    return {
        "id": f"id:{name}",
        "type": "function",
        "name": name,
        "file_path": f"{name}.py",
        "relative_path": f"{name}.py",
        "language": "python",
        "start_line": 1,
        "end_line": 1,
        "code": "return 1",
        "signature": f"def {name}()",
        "docstring": None,
        "summary": None,
        "metadata": {},
        "repo_name": "repo",
        "repo_url": None,
    }


def test_embed_text_reuses_cached_embedding() -> None:
    cache = _MemoryCache()
    embedder = _CountingEmbedder(cache)

    first = embedder.embed_text("same text")
    second = embedder.embed_text("same text")

    assert embedder.raw_batches == [["same text"]]
    assert np.array_equal(first, second)


def test_embed_batch_deduplicates_inputs_before_raw_embedding() -> None:
    cache = _MemoryCache()
    embedder = _CountingEmbedder(cache)

    embeddings = embedder.embed_batch(["a", "a", "b", "a"])

    assert embedder.raw_batches == [["a", "b"]]
    assert embeddings.shape == (4, 3)
    assert np.array_equal(embeddings[0], embeddings[1])
    assert np.array_equal(embeddings[0], embeddings[3])


def test_embed_code_elements_deduplicates_batch_and_persists_cache() -> None:
    cache = _MemoryCache()
    embedder = _CountingEmbedder(cache)
    first_batch = [_element("same"), _element("same"), _element("other")]

    embedder.embed_code_elements(first_batch)
    second_batch = [_element("same"), _element("other")]
    embedder.embed_code_elements(second_batch)

    prepared_same = embedder._prepare_code_text(first_batch[0])
    prepared_other = embedder._prepare_code_text(first_batch[2])
    assert embedder.raw_batches == [[prepared_same, prepared_other]]
    assert first_batch[0]["embedding_text"] == prepared_same
    assert first_batch[0]["embedding"].shape == (3,)
    assert np.array_equal(first_batch[0]["embedding"], second_batch[0]["embedding"])


def test_embedding_cache_key_is_model_aware() -> None:
    cache = _MemoryCache()
    first = _CountingEmbedder(cache, model_name="model-a")
    second = _CountingEmbedder(cache, model_name="model-b")

    first.embed_text("same text")
    second.embed_text("same text")

    assert first.raw_batches == [["same text"]]
    assert second.raw_batches == [["same text"]]
    assert first._embedding_cache_key("same text") != second._embedding_cache_key(
        "same text"
    )


def test_embedding_cache_disabled_uses_raw_embedding_each_time() -> None:
    cache = _MemoryCache()
    embedder = _CountingEmbedder(cache)
    embedder._embedding_cache_enabled = False

    embedder.embed_text("same text")
    embedder.embed_text("same text")

    assert embedder.raw_batches == [["same text"], ["same text"]]
