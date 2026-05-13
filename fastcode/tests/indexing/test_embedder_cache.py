from __future__ import annotations

import subprocess
import sys
from typing import Any

import numpy as np

from fastcode.indexing.embedder import CodeEmbedder, EmbeddingFingerprint


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
        "metadata": {"stable_unit_id": f"unit:function:{name}"},
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


def test_embedding_cache_writes_float32_buffer_payload() -> None:
    cache = _MemoryCache()
    embedder = _CountingEmbedder(cache)

    first = embedder.embed_text("buffered text")
    key = embedder._embedding_cache_key("buffered text")
    cached = cache.values[key]
    second = embedder.embed_text("buffered text")

    assert "embedding" not in cached
    assert cached["embedding_format"] == "ndarray.float32.v1"
    assert cached["embedding_dtype"] == "float32"
    assert cached["embedding_shape"] == (3,)
    assert isinstance(cached["embedding_bytes"], bytes)
    assert cached["embedding_fingerprint"] == embedder.embedding_fingerprint()
    assert np.array_equal(first, second)
    assert embedder.raw_batches == [["buffered text"]]


def test_embedding_cache_recomputes_old_list_payload() -> None:
    cache = _MemoryCache()
    embedder = _CountingEmbedder(cache)
    key = embedder._embedding_cache_key("old text")
    cache.set(
        key,
        {
            "embedding": [1.0, 2.0, 3.0],
            "dimension": 3,
        },
    )

    cached = embedder.embed_text("old text")

    assert np.array_equal(cached, np.asarray(embedder._vector_for_text("old text")))
    assert embedder.raw_batches == [["old text"]]
    assert cache.values[key]["embedding_format"] == "ndarray.float32.v1"


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


def test_embedding_fingerprint_record_is_typed_and_matches_payload() -> None:
    cache = _MemoryCache()
    embedder = _CountingEmbedder(cache)

    fingerprint = embedder.embedding_fingerprint_record()

    assert isinstance(fingerprint, EmbeddingFingerprint)
    assert fingerprint.to_payload() == embedder.embedding_fingerprint()
    assert fingerprint.stable_json() == (
        '{"cache_version":null,"dimension":3,"max_seq_length":512,'
        '"model":"model-a","normalize":true,"ollama_url":null,'
        '"provider":"sentence_transformers","text_schema_version":1,"version":2}'
    )


def test_embedding_cache_disabled_uses_raw_embedding_each_time() -> None:
    cache = _MemoryCache()
    embedder = _CountingEmbedder(cache)
    embedder._embedding_cache_enabled = False

    embedder.embed_text("same text")
    embedder.embed_text("same text")

    assert embedder.raw_batches == [["same text"], ["same text"]]


def test_embedder_initialization_does_not_load_model(
    monkeypatch: Any,
) -> None:
    def _boom_load_model(self: CodeEmbedder) -> Any:
        raise AssertionError("model should load lazily")

    monkeypatch.setattr(CodeEmbedder, "_load_model", _boom_load_model)

    embedder = CodeEmbedder(
        {
            "embedding": {
                "provider": "sentence_transformers",
                "model": "test-model",
                "dimension": 3,
            },
            "cache": {"enabled": False},
        }
    )

    assert embedder.model is None
    assert embedder.embedding_dim == 3


def test_ollama_embedder_import_does_not_require_local_embedding_stack() -> None:
    code = """
import builtins

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split('.')[0] in {'torch', 'sentence_transformers'}:
        raise ImportError(name)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from fastcode.indexing.embedder import CodeEmbedder

embedder = CodeEmbedder({
    'embedding': {
        'provider': 'ollama',
        'model': 'release-gate',
        'dimension': 8,
        'device': 'cpu',
    },
    'cache': {'enabled': False},
})
assert embedder.model is None
assert embedder.embedding_fingerprint()['provider'] == 'ollama'
"""
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test_sentence_transformers_missing_dependency_fails_at_model_boundary() -> None:
    code = """
import builtins

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split('.')[0] == 'sentence_transformers':
        raise ImportError(name)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from fastcode.indexing.embedder import CodeEmbedder

embedder = CodeEmbedder({
    'embedding': {
        'provider': 'sentence_transformers',
        'model': 'missing-local-stack',
        'dimension': 2,
        'device': 'cpu',
    },
    'cache': {'enabled': False},
})
try:
    embedder.embed_batch(['hello'])
except RuntimeError as exc:
    assert "embedding.provider='sentence_transformers'" in str(exc)
else:
    raise AssertionError('expected missing optional dependency failure')
"""
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test_embedding_fingerprint_without_configured_dimension_does_not_load_model(
    monkeypatch: Any,
) -> None:
    def _boom_load_model(self: CodeEmbedder) -> Any:
        raise AssertionError("fingerprint should not load model")

    monkeypatch.setattr(CodeEmbedder, "_load_model", _boom_load_model)
    embedder = CodeEmbedder(
        {
            "embedding": {
                "provider": "sentence_transformers",
                "model": "test-model",
            },
            "cache": {"enabled": False},
        }
    )

    payload = embedder.embedding_fingerprint()

    assert payload["dimension"] is None
    assert payload["text_schema_version"] == 1
    assert embedder.model is None


def test_cache_hit_validation_without_configured_dimension_does_not_load_model(
    monkeypatch: Any,
) -> None:
    def _boom_load_model(self: CodeEmbedder) -> Any:
        raise AssertionError("cache-hit validation should not load model")

    monkeypatch.setattr(CodeEmbedder, "_load_model", _boom_load_model)
    embedder = CodeEmbedder(
        {
            "embedding": {
                "provider": "sentence_transformers",
                "model": "test-model",
            },
            "cache": {"enabled": False},
        }
    )
    cache = _MemoryCache()
    embedder._embedding_cache = cache
    embedder._embedding_cache_enabled = True
    cache.set(
        "key",
        {
            "embedding_format": "ndarray.float32.v1",
            "embedding_dtype": "float32",
            "embedding_shape": (3,),
            "embedding_bytes": np.asarray([1.0, 2.0, 3.0], dtype=np.float32).tobytes(),
        },
    )

    cached = embedder._get_cached_embedding("key")

    assert cached is not None
    assert cached.shape == (3,)
    assert embedder.model is None


def test_embedder_loads_model_on_first_uncached_embedding(
    monkeypatch: Any,
) -> None:
    load_calls: list[str] = []

    class _FakeModel:
        def get_embedding_dimension(self) -> int:
            return 2

        def encode(self, texts: list[str], **_kwargs: Any) -> np.ndarray:
            return np.asarray(
                [[float(index), float(index + 1)] for index, _ in enumerate(texts)],
                dtype=np.float32,
            )

    def _load_model(self: CodeEmbedder) -> _FakeModel:
        load_calls.append(self.model_name)
        return _FakeModel()

    monkeypatch.setattr(CodeEmbedder, "_load_model", _load_model)
    embedder = CodeEmbedder(
        {
            "embedding": {
                "provider": "sentence_transformers",
                "model": "lazy-test-model",
            },
            "cache": {"enabled": False},
        }
    )

    assert load_calls == []

    embeddings = embedder.embed_batch(["a", "b"])

    assert load_calls == ["lazy-test-model"]
    assert embeddings.shape == (2, 2)
    assert embedder.embedding_dim == 2


def test_embed_code_elements_persists_embedding_text_hash() -> None:
    cache = _MemoryCache()
    embedder = _CountingEmbedder(cache)
    batch = [_element("same")]

    embedder.embed_code_elements(batch)

    metadata = batch[0]["metadata"]
    assert "embedding_text_hash" in metadata
    assert metadata["embedding_text_hash"]
    assert metadata["embedding_fingerprint"] == embedder.embedding_fingerprint()
    assert batch[0]["embedding_artifact_ref"] == embedder.embedding_artifact_ref(
        batch[0]["embedding_text"]
    )
