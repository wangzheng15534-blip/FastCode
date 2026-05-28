"""Embedder audit regression tests — infra hardening incident guards.

Reference: incident-register-2026-05-17.yaml (EMBD-001, EMBD-002, EMBD-003)
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from fastcode.app.indexing.embedder import CodeEmbedder

pytestmark = [pytest.mark.test_double]


# ---------------------------------------------------------------------------
# EMBD-001: empty ollama_url produces relative batch URL
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestOllamaBatchUrlConfig:
    """EMBD-001: verify batch URL derivation for empty/None ollama_url."""

    @pytest.mark.audit_finding("EMBD-001")
    def test_empty_ollama_url_produces_fallback_batch_url(self) -> None:
        result = CodeEmbedder._default_ollama_batch_url("")
        assert result == "http://127.0.0.1:11434/api/embed", (
            f"Empty ollama_url should produce fallback batch URL, got '{result}'"
        )

    @pytest.mark.audit_finding("EMBD-001")
    def test_none_ollama_url_batch_url(self) -> None:
        result = CodeEmbedder._default_ollama_batch_url(None)
        assert result == "http://127.0.0.1:11434/api/embed", (
            f"None ollama_url should produce fallback batch URL, got '{result}'"
        )

    @pytest.mark.audit_finding("EMBD-001")
    def test_valid_ollama_url_produces_absolute_batch_url(self) -> None:
        result = CodeEmbedder._default_ollama_batch_url(
            "http://ollama.local:11434/api/embeddings"
        )
        assert result.startswith("http://"), (
            f"Valid URL should produce absolute batch URL, got '{result}'"
        )
        assert result.endswith("/api/embed")

    @pytest.mark.audit_finding("EMBD-001")
    def test_embedder_with_empty_ollama_url_stores_fallback_batch_url(self) -> None:
        embedder = CodeEmbedder(
            {
                "embedding": {
                    "provider": "ollama",
                    "model": "test",
                    "dimension": 2,
                    "ollama_url": "",
                },
                "cache": {"enabled": False},
            }
        )
        assert embedder.ollama_batch_url == "http://127.0.0.1:11434/api/embed"


# ---------------------------------------------------------------------------
# EMBD-002: malformed Ollama batch response validation
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestOllamaBatchResponseValidation:
    """EMBD-002: verify malformed batch responses raise RuntimeError."""

    @staticmethod
    def _make_embedder() -> CodeEmbedder:
        return CodeEmbedder(
            {
                "embedding": {
                    "provider": "ollama",
                    "model": "test-ollama",
                    "dimension": 2,
                    "ollama_url": "http://ollama.local/api/embeddings",
                },
                "cache": {"enabled": False},
            }
        )

    @staticmethod
    def _fake_urlopen(body: dict[str, Any]) -> Any:
        class _Response:
            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(body).encode()

        return _Response()

    @pytest.mark.audit_finding("EMBD-002")
    def test_none_entries_in_embeddings(self, monkeypatch: Any) -> None:
        embedder = self._make_embedder()
        monkeypatch.setattr(
            "fastcode.app.indexing.embedder.urllib.request.urlopen",
            lambda *a, **kw: self._fake_urlopen({"embeddings": [None, [0.0, 1.0]]}),
        )
        with pytest.raises((RuntimeError, ValueError)):
            embedder._embed_batch_ollama(["a", "b"])

    @pytest.mark.audit_finding("EMBD-002")
    def test_wrong_count_in_embeddings(self, monkeypatch: Any) -> None:
        embedder = self._make_embedder()
        monkeypatch.setattr(
            "fastcode.app.indexing.embedder.urllib.request.urlopen",
            lambda *a, **kw: self._fake_urlopen({"embeddings": [[1.0, 0.0]]}),
        )
        with pytest.raises(RuntimeError, match="missing embeddings"):
            embedder._embed_batch_ollama(["a", "b"])

    @pytest.mark.audit_finding("EMBD-002")
    def test_ragged_inner_dimensions(self, monkeypatch: Any) -> None:
        embedder = self._make_embedder()
        monkeypatch.setattr(
            "fastcode.app.indexing.embedder.urllib.request.urlopen",
            lambda *a, **kw: self._fake_urlopen({"embeddings": [[1.0, 0.0], [1.0]]}),
        )
        with pytest.raises((RuntimeError, ValueError)):
            embedder._embed_batch_ollama(["a", "b"])

    @pytest.mark.audit_finding("EMBD-002")
    def test_flat_list_instead_of_list_of_lists(self, monkeypatch: Any) -> None:
        embedder = self._make_embedder()
        monkeypatch.setattr(
            "fastcode.app.indexing.embedder.urllib.request.urlopen",
            lambda *a, **kw: self._fake_urlopen({"embeddings": [1.0, 0.0]}),
        )
        with pytest.raises((RuntimeError, ValueError)):
            embedder._embed_batch_ollama(["a", "b"])

    @pytest.mark.audit_finding("EMBD-002")
    def test_missing_embeddings_key(self, monkeypatch: Any) -> None:
        embedder = self._make_embedder()
        monkeypatch.setattr(
            "fastcode.app.indexing.embedder.urllib.request.urlopen",
            lambda *a, **kw: self._fake_urlopen({"model": "test-ollama"}),
        )
        with pytest.raises(RuntimeError, match="missing embeddings"):
            embedder._embed_batch_ollama(["a"])

    @pytest.mark.audit_finding("EMBD-002")
    def test_embeddings_not_a_list(self, monkeypatch: Any) -> None:
        embedder = self._make_embedder()
        monkeypatch.setattr(
            "fastcode.app.indexing.embedder.urllib.request.urlopen",
            lambda *a, **kw: self._fake_urlopen({"embeddings": "not a list"}),
        )
        with pytest.raises(RuntimeError, match="missing embeddings"):
            embedder._embed_batch_ollama(["a"])


# ---------------------------------------------------------------------------
# EMBD-003: concurrent metrics increment — no loss
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestEmbeddingMetricsConcurrency:
    """EMBD-003: verify metric increments are not lost under concurrency."""

    @pytest.mark.audit_finding("EMBD-003")
    def test_concurrent_metrics_increment_no_loss(self) -> None:
        embedder = CodeEmbedder(
            {
                "embedding": {
                    "provider": "ollama",
                    "model": "test",
                    "dimension": 2,
                },
                "cache": {"enabled": False},
            }
        )
        n_threads = 8
        n_increments = 100

        barrier = threading.Barrier(n_threads)

        def increment_repeatedly() -> None:
            barrier.wait()
            for _ in range(n_increments):
                embedder._increment_embedding_metric("test_counter")

        threads = [
            threading.Thread(target=increment_repeatedly) for _ in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert (
            embedder.embedding_metrics()["test_counter"] == n_threads * n_increments
        ), (
            f"Expected {n_threads * n_increments}, "
            f"got {embedder.embedding_metrics().get('test_counter')} — metric loss under concurrency"
        )
