"""
Code Embedder - Generate embeddings for code snippets
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import hashlib
import json
import logging
import platform
import threading
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from ..ports.embedding import EmbeddingProvider

if TYPE_CHECKING:
    from ..ir.element import CodeElementMeta

import numpy as np

_EMBEDDING_CACHE_FORMAT = "ndarray.float32.v1"
_EMBEDDING_TEXT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EmbeddingFingerprint:
    """Stable embedding implementation identity for reuse decisions."""

    version: int
    provider: str
    model: str
    dimension: int | None
    max_seq_length: int
    normalize: bool
    text_schema_version: int = _EMBEDDING_TEXT_SCHEMA_VERSION
    ollama_url: str | None = None
    cache_version: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "provider": self.provider,
            "model": self.model,
            "dimension": self.dimension,
            "max_seq_length": self.max_seq_length,
            "normalize": self.normalize,
            "text_schema_version": self.text_schema_version,
            "ollama_url": self.ollama_url,
            "cache_version": self.cache_version,
        }

    def stable_json(self) -> str:
        return json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))


class CodeEmbedder(EmbeddingProvider):
    """Generate embeddings for code using sentence transformers"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.embedding_config = config.get("embedding", {})
        self.logger = logging.getLogger(__name__)

        self.provider = self.embedding_config.get("provider", "sentence_transformers")
        self.model_name = self.embedding_config.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.device = self.embedding_config.get("device", "auto")
        self.batch_size = self.embedding_config.get("batch_size", 32)
        self.max_seq_length = self.embedding_config.get("max_seq_length", 512)
        self.normalize = self.embedding_config.get("normalize_embeddings", True)
        self.ollama_url = self.embedding_config.get(
            "ollama_url", "http://127.0.0.1:11434/api/embeddings"
        )
        self.ollama_batch_url = self.embedding_config.get(
            "ollama_batch_url",
            self._default_ollama_batch_url(self.ollama_url),
        )
        self.ollama_batch_enabled = bool(
            self.embedding_config.get("ollama_batch_enabled", True)
        )
        self._embedding_cache: Any | None = None
        self._embedding_cache_enabled = False
        self._embedding_metrics: dict[str, float] = {}
        self._embedding_metrics_lock = threading.Lock()
        self._model_lock = threading.Lock()

        # Auto-detect best available device: CUDA > MPS > CPU
        self.device = self._resolve_device(self.device)

        self.model: Any | None = None
        raw_dim = (
            self.embedding_config.get("dimension")
            or self.embedding_config.get("embedding_dim")
            or 0
        )
        if not isinstance(raw_dim, (int, float, type(None))):
            raise ValueError(
                f"embedding.dimension must be numeric, got {type(raw_dim).__name__}: "
                f"{raw_dim!r}"
            )
        self._configured_embedding_dim: int = int(raw_dim or 0)
        self._embedding_dim: int = self._configured_embedding_dim

        self._initialize_embedding_cache()
        self.logger.info(
            "Embedding provider configured: %s model=%s dimension=%s",
            self.provider,
            self.model_name,
            self._embedding_dim or "lazy",
        )

    def _ensure_embedding_metrics(self) -> dict[str, float]:
        metrics = getattr(self, "_embedding_metrics", None)
        if not isinstance(metrics, dict):
            metrics = {}
            self._embedding_metrics = metrics
        return metrics

    def _increment_embedding_metric(self, name: str, amount: float = 1.0) -> None:
        metrics = self._ensure_embedding_metrics()
        lock = self._embedding_metrics_lock
        with lock:
            metrics[name] = float(metrics.get(name, 0.0)) + float(amount)

    def reset_embedding_metrics(self) -> None:
        metrics = self._ensure_embedding_metrics()
        lock = self._embedding_metrics_lock
        with lock:
            metrics.clear()

    def embedding_metrics(self) -> dict[str, int | float]:
        metrics = self._ensure_embedding_metrics()
        lock = self._embedding_metrics_lock
        with lock:
            snapshot = dict(metrics)
        result: dict[str, int | float] = {}
        for name, value in sorted(snapshot.items()):
            if value.is_integer():
                result[name] = int(value)
            else:
                result[name] = round(value, 3)
        return result

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device == "cpu" or self.provider == "ollama":
            return requested_device
        try:
            import torch
        except ImportError:
            self.logger.warning(
                "Torch is not installed; falling back to CPU device selection. "
                "Install the local embedding extra to use sentence-transformers."
            )
            return "cpu"
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    @staticmethod
    def _default_ollama_batch_url(ollama_url: str) -> str:
        base_url = str(ollama_url or "").rstrip("/")
        if not base_url:
            return "http://127.0.0.1:11434/api/embed"
        if base_url.endswith("/api/embeddings"):
            return f"{base_url[: -len('/api/embeddings')]}/api/embed"
        if base_url.endswith("/api/embed"):
            return base_url
        return f"{base_url}/api/embed"

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim:
            return self._embedding_dim
        return self._ensure_embedding_dimension()

    @embedding_dim.setter
    def embedding_dim(self, value: int) -> None:
        self._embedding_dim = int(value or 0)
        if not hasattr(self, "_configured_embedding_dim"):
            self._configured_embedding_dim = self._embedding_dim

    def _initialize_embedding_cache(self) -> None:
        cache_config = self.config.get("cache", {})
        evaluation_config = self.config.get("evaluation", {})
        self._embedding_cache_enabled = bool(
            cache_config.get("enabled", True)
            and cache_config.get("cache_embeddings", True)
            and not evaluation_config.get("disable_cache", False)
        )
        if not self._embedding_cache_enabled:
            return

        try:
            from ..store.cache import CacheManager

            self._embedding_cache = CacheManager(self.config)
        except Exception as e:
            self._embedding_cache_enabled = False
            self._embedding_cache = None
            self.logger.warning(f"Embedding cache disabled: {e}")

    def _load_model(self) -> Any:
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "embedding.provider='sentence_transformers' requires the "
                "sentence-transformers and torch packages. Install the local "
                "embedding extra or configure embedding.provider='ollama'."
            ) from exc
        model = SentenceTransformer(self.model_name, device=self.device)
        model.max_seq_length = self.max_seq_length
        return model

    def _ensure_model_loaded(self) -> Any:
        if self.model is not None:
            return self.model
        with self._model_lock:
            if self.model is not None:
                return self.model
            self.logger.info(f"Loading embedding model: {self.model_name}")
            started = perf_counter()
            model = self._load_model()
            self._increment_embedding_metric(
                "provider_startup_ms",
                (perf_counter() - started) * 1000,
            )
            self._increment_embedding_metric("provider_startup_count")
            self.model = model
            dim = model.get_embedding_dimension()
            if dim is not None:
                self._embedding_dim = int(dim)
            self.logger.info(f"Embedding dimension: {self._embedding_dim}")
        return self.model

    def _ensure_embedding_dimension(self) -> int:
        if self._embedding_dim:
            return self._embedding_dim
        if self.provider == "ollama":
            self.logger.info(
                f"Probing Ollama embeddings model: {self.model_name} ({self.ollama_url})"
            )
            probe = self._embed_text_ollama("embedding dimension probe")
            self._embedding_dim = len(probe)
            return self._embedding_dim
        model = self._ensure_model_loaded()
        dim = model.get_embedding_dimension()
        self._embedding_dim = int(dim or 0)
        return self._embedding_dim

    def _fingerprint_dimension(self, *, resolve_dimension: bool) -> int | None:
        if resolve_dimension:
            return int(self.embedding_dim)
        configured = int(getattr(self, "_configured_embedding_dim", 0) or 0)
        if configured:
            return configured
        known = int(getattr(self, "_embedding_dim", 0) or 0)
        if known:
            return known
        return None

    def embedding_fingerprint_record(
        self, *, resolve_dimension: bool = False
    ) -> EmbeddingFingerprint:
        """Return the model/config identity used for embedding reuse decisions."""
        cache_version = self.embedding_config.get("cache_version")
        return EmbeddingFingerprint(
            version=2,
            provider=str(self.provider),
            model=str(self.model_name),
            dimension=self._fingerprint_dimension(resolve_dimension=resolve_dimension),
            max_seq_length=int(self.max_seq_length),
            normalize=bool(self.normalize),
            text_schema_version=_EMBEDDING_TEXT_SCHEMA_VERSION,
            ollama_url=str(self.ollama_url) if self.provider == "ollama" else None,
            cache_version=str(cache_version) if cache_version is not None else None,
        )

    def embedding_fingerprint(
        self, *, resolve_dimension: bool = False
    ) -> dict[str, Any]:
        """Compatibility payload for callers that persist JSON metadata."""
        return self.embedding_fingerprint_record(
            resolve_dimension=resolve_dimension
        ).to_payload()

    def fingerprint(self, *, resolve_dimension: bool = False) -> dict[str, Any]:
        """Return the service-level embedding identity payload."""
        return self.embedding_fingerprint(resolve_dimension=resolve_dimension)

    def prepare_text(self, element: CodeElementMeta) -> str:
        """
        Prepare code element text for embedding.

        This is the public service boundary equivalent of the historical
        ``_prepare_code_text`` helper.
        """
        parts: list[str] = []

        if "type" in element:
            parts.append(f"Type: {element['type']}")

        if "name" in element:
            parts.append(f"Name: {element['name']}")

        if "signature" in element:
            parts.append(f"Signature: {element['signature']}")

        if element.get("docstring"):
            parts.append(f"Documentation: {element['docstring']}")

        if summary := element.get("summary"):
            parts.append(summary)

        if "code" in element:
            code = element["code"]
            if len(code) > 10000:
                code = code[:10000] + "..."
            parts.append(f"Code:\n{code}")

        return "\n".join(parts)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts with cache reuse."""
        return self._embed_texts_with_cache(texts)

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        """Generate embeddings for prepared texts through the service boundary."""
        return self._embed_texts_with_cache(list(texts))

    def _embed_batch_uncached(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts without cache lookup.

        Args:
            texts: List of input texts

        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])

        if self.provider == "ollama":
            if self.ollama_batch_enabled and len(texts) > 1:
                try:
                    return self._embed_batch_ollama(texts)
                except Exception as exc:
                    self._increment_embedding_metric("provider_batch_fallback_count")
                    self.logger.warning(
                        "Ollama batch embedding failed, falling back to per-text "
                        "requests: %s",
                        exc,
                    )
            raw_concurrency = (
                self.embedding_config.get("ollama_concurrency")
                or self.embedding_config.get("max_concurrency")
                or 1
            )
            if not isinstance(raw_concurrency, (int, float)):
                raise ValueError(
                    f"embedding.ollama_concurrency must be numeric, got "
                    f"{type(raw_concurrency).__name__}: {raw_concurrency!r}"
                )
            max_workers = int(raw_concurrency)
            if max_workers > 1 and len(texts) > 1:
                with ThreadPoolExecutor(
                    max_workers=min(max_workers, len(texts)),
                    thread_name_prefix="fastcode-ollama-embed",
                ) as executor:
                    vectors = list(executor.map(self._embed_text_ollama, texts))
            else:
                vectors = [self._embed_text_ollama(t) for t in texts]
            matrix = np.asarray(vectors, dtype=np.float32)
            if matrix.ndim == 2 and matrix.shape[1]:
                self._embedding_dim = int(matrix.shape[1])
            return matrix

        encode_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "show_progress_bar": len(texts) > 100,
            "normalize_embeddings": self.normalize,
            "convert_to_numpy": True,
            "device": self.device,
            "convert_to_tensor": False,
        }

        if platform.system() == "Darwin":
            encode_kwargs["pool"] = None

        model = self._ensure_model_loaded()
        self._increment_embedding_metric("provider_request_count")
        raw: Any = model.encode(texts, **encode_kwargs)
        if isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.shape[1]:
            self._embedding_dim = int(raw.shape[1])
        return cast(np.ndarray, raw)

    def _embedding_cache_key(self, text: str) -> str:
        fingerprint = self.embedding_fingerprint_record()
        digest = hashlib.sha256(
            f"{fingerprint.stable_json()}\0{text}".encode()
        ).hexdigest()
        return f"embedding_v2_{digest}"

    def embedding_artifact_ref(self, text: str) -> str:
        """Return the stable cache/artifact locator for prepared embedding text."""
        return self._embedding_cache_key(text)

    def _get_cached_embedding(
        self,
        key: str,
        *,
        text: str | None = None,
    ) -> np.ndarray | None:
        if not self._embedding_cache_enabled or self._embedding_cache is None:
            return None
        if hasattr(self._embedding_cache, "get_cached_embedding_payload"):
            cached = self._embedding_cache.get_cached_embedding_payload(key)
        else:
            cached = self._embedding_cache.get(key)
        if cached is None:
            return None
        if text is not None and not self._cached_payload_matches_reuse_identity(
            cached,
            text=text,
        ):
            return None
        embedding = self._cached_embedding_to_array(cached)
        if embedding is None:
            return None
        if embedding.ndim != 1:
            return None
        known_dim = int(
            getattr(self, "_configured_embedding_dim", 0)
            or getattr(self, "_embedding_dim", 0)
            or 0
        )
        if known_dim and embedding.size != known_dim:
            return None
        if not known_dim and embedding.size:
            self._embedding_dim = int(embedding.size)
        return embedding

    def _cached_payload_matches_reuse_identity(
        self,
        cached: Any,
        *,
        text: str,
    ) -> bool:
        if not isinstance(cached, Mapping):
            return False
        expected_text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cached.get("text_sha256") != expected_text_hash:
            return False
        cached_fingerprint = cached.get("embedding_fingerprint")
        if not isinstance(cached_fingerprint, Mapping):
            return False
        current_fingerprint = self.embedding_fingerprint(resolve_dimension=False)
        known_dim = int(
            getattr(self, "_configured_embedding_dim", 0)
            or getattr(self, "_embedding_dim", 0)
            or 0
        )
        for field_name, expected_value in current_fingerprint.items():
            cached_value = cached_fingerprint.get(field_name)
            if field_name == "dimension" and expected_value is None:
                if known_dim and cached_value not in {None, known_dim}:
                    return False
                continue
            if cached_value != expected_value:
                return False
        return True

    @staticmethod
    def _cached_embedding_to_array(cached: Any) -> np.ndarray | None:
        if not isinstance(cached, dict):
            return None
        if cached.get("embedding_format") != _EMBEDDING_CACHE_FORMAT:
            return None
        raw_buffer = cached.get("embedding_bytes")
        if not isinstance(raw_buffer, (bytes, bytearray, memoryview)):
            return None
        raw_shape = cached.get("embedding_shape")
        if not isinstance(raw_shape, (list, tuple)):
            return None
        try:
            shape = tuple(int(dim) for dim in raw_shape)
        except (TypeError, ValueError):
            return None
        if len(shape) != 1:
            return None
        embedding = np.frombuffer(raw_buffer, dtype=np.float32)
        if embedding.size != shape[0]:
            return None
        return embedding.reshape(shape)

    def _set_cached_embedding(self, key: str, text: str, embedding: np.ndarray) -> None:
        if not self._embedding_cache_enabled or self._embedding_cache is None:
            return
        embedding_array = np.ascontiguousarray(
            np.asarray(embedding, dtype=np.float32).reshape(-1)
        )
        previous_dim = int(getattr(self, "_embedding_dim", 0) or 0)
        if not previous_dim and embedding_array.size:
            self._embedding_dim = int(embedding_array.size)
        fingerprint_payload = self.embedding_fingerprint(resolve_dimension=True)
        payload = {
            "embedding_format": _EMBEDDING_CACHE_FORMAT,
            "embedding_dtype": "float32",
            "embedding_shape": tuple(int(dim) for dim in embedding_array.shape),
            "embedding_bytes": embedding_array.tobytes(order="C"),
            "embedding_fingerprint": fingerprint_payload,
            "provider": self.provider,
            "model": self.model_name,
            "dimension": int(embedding_array.size),
            "max_seq_length": self.max_seq_length,
            "normalize": self.normalize,
            "text_schema_version": _EMBEDDING_TEXT_SCHEMA_VERSION,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        if hasattr(self._embedding_cache, "set_cached_embedding_payload"):
            self._embedding_cache.set_cached_embedding_payload(key, payload)
        else:
            self._embedding_cache.set(key, payload)
        self._increment_embedding_metric("cache_write_count")

    @staticmethod
    def _embedding_rows_to_matrix(
        embeddings: Sequence[np.ndarray | None],
    ) -> np.ndarray:
        first_embedding = next(
            (embedding for embedding in embeddings if embedding is not None),
            None,
        )
        if first_embedding is None:
            return np.asarray([], dtype=np.float32)
        first_row = np.asarray(first_embedding, dtype=np.float32).reshape(-1)
        matrix = np.empty((len(embeddings), first_row.size), dtype=np.float32)
        for index, embedding in enumerate(embeddings):
            if embedding is None:
                raise RuntimeError(f"Embedding cache fill failed at index: {index}")
            row = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if row.size != first_row.size:
                raise RuntimeError(
                    f"Embedding cache fill produced inconsistent dimensions: "
                    f"expected {first_row.size}, got {row.size} at index {index}"
                )
            matrix[index] = row
        return matrix

    def _embed_texts_with_cache(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        if not self._embedding_cache_enabled or self._embedding_cache is None:
            self._increment_embedding_metric("cache_miss_count", len(texts))
            self._increment_embedding_metric("provider_batch_count")
            return self._embed_batch_uncached(texts)

        embeddings: list[np.ndarray | None] = [None] * len(texts)
        misses: dict[str, tuple[str, list[int]]] = {}
        hits = 0

        for index, text in enumerate(texts):
            key = self._embedding_cache_key(text)
            cached_embedding = self._get_cached_embedding(key, text=text)
            if cached_embedding is not None:
                embeddings[index] = cached_embedding
                hits += 1
                continue
            if key not in misses:
                misses[key] = (text, [])
            misses[key][1].append(index)

        if misses:
            miss_items = list(misses.items())
            miss_texts = [text for _, (text, _) in miss_items]
            self._increment_embedding_metric(
                "cache_miss_count",
                sum(len(indexes) for _, indexes in misses.values()),
            )
            self._increment_embedding_metric("provider_batch_count")
            miss_embeddings = np.asarray(
                self._embed_batch_uncached(miss_texts), dtype=np.float32
            )
            if miss_embeddings.ndim == 1 and len(miss_texts) == 1:
                miss_embeddings = miss_embeddings.reshape(1, -1)
            if (
                miss_embeddings.ndim != 2
                or miss_embeddings.shape[0] != len(miss_texts)
                or not miss_embeddings.shape[1]
            ):
                raise RuntimeError("Uncached embedding batch returned invalid shape")
            single_miss_indexes = [indexes for _text, indexes in misses.values()]
            can_return_miss_matrix = (
                hits == 0
                and miss_embeddings.ndim == 2
                and miss_embeddings.shape[0] == len(texts)
                and all(
                    indexes == [index]
                    for index, indexes in enumerate(single_miss_indexes)
                )
            )
            for (key, (text, indexes)), embedding in zip(
                miss_items, miss_embeddings, strict=True
            ):
                embedding_array = np.asarray(embedding, dtype=np.float32)
                for index in indexes:
                    embeddings[index] = embedding_array
                self._set_cached_embedding(key, text, embedding_array)
            if can_return_miss_matrix:
                return miss_embeddings

        if hits:
            self._increment_embedding_metric("cache_hit_count", hits)
            self.logger.debug("Embedding cache reused %d/%d vectors", hits, len(texts))

        missing_indexes = [
            i for i, embedding in enumerate(embeddings) if embedding is None
        ]
        if missing_indexes:
            raise RuntimeError(
                f"Embedding cache fill failed at indexes: {missing_indexes}"
            )

        return self._embedding_rows_to_matrix(embeddings)

    def _truncate_ollama_text(self, text: str) -> str:
        if len(text) > self.max_seq_length:
            return text[: self.max_seq_length]
        return text

    def _embed_batch_ollama(self, texts: list[str]) -> np.ndarray:
        payload = {
            "model": self.model_name,
            "input": [self._truncate_ollama_text(text) for text in texts],
        }
        self._increment_embedding_metric("provider_request_count")
        req = urllib.request.Request(
            self.ollama_batch_url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        vectors = body.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError("Ollama batch embedding response missing embeddings")
        if any(not isinstance(v, list) for v in vectors):
            raise RuntimeError(
                "Ollama batch embedding response contains non-list entries"
            )
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts) or not matrix.shape[1]:
            raise RuntimeError("Ollama batch embedding response has invalid shape")
        self._embedding_dim = int(matrix.shape[1])
        self._increment_embedding_metric("provider_true_batch_count")
        return matrix

    def _embed_text_ollama(self, text: str) -> np.ndarray:
        text = self._truncate_ollama_text(text)
        payload = {"model": self.model_name, "prompt": text}
        self._increment_embedding_metric("provider_request_count")
        req = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        vector = body.get("embedding")
        if not vector:
            raise RuntimeError("Ollama embedding response missing 'embedding'")
        embedding = np.asarray(vector, dtype=np.float32)
        if embedding.ndim == 1 and embedding.size:
            self._embedding_dim = int(embedding.size)
        return embedding

    def embed_code_elements(
        self, elements: list[CodeElementMeta]
    ) -> list[CodeElementMeta]:
        """Compatibility wrapper for callers not yet migrated to embed_elements."""
        return self.embed_elements(elements)

    def embed_elements(
        self,
        elements: Sequence[CodeElementMeta],
        reuse_index: Mapping[str, CodeElementMeta] | None = None,
    ) -> list[CodeElementMeta]:
        """
        Generate embeddings for code elements (functions, classes, etc.)

        Args:
            elements: List of code element dictionaries

        Returns:
            List of elements with embeddings added
        """
        if not elements:
            return []

        mutable_elements = list(elements)
        texts = [self.prepare_text(elem) for elem in mutable_elements]

        self.logger.info(f"Generating embeddings for {len(texts)} code elements")

        missing_indexes: list[int] = []
        current_fingerprint = self.fingerprint(resolve_dimension=False)
        reuse_hits = 0
        for index, (elem, text) in enumerate(zip(mutable_elements, texts, strict=True)):
            if self._try_reuse_element_embedding(
                elem,
                text=text,
                reuse_index=reuse_index,
                current_fingerprint=current_fingerprint,
            ):
                reuse_hits += 1
                continue
            missing_indexes.append(index)

        if missing_indexes:
            miss_texts = [texts[index] for index in missing_indexes]
            embeddings = self.embed_many(miss_texts)
            fingerprint_payload = self.fingerprint(resolve_dimension=True)
            for index, embedding in zip(missing_indexes, embeddings, strict=True):
                self._attach_embedding_to_element(
                    mutable_elements[index],
                    text=texts[index],
                    embedding=embedding,
                    fingerprint_payload=fingerprint_payload,
                    artifact_ref=None,
                )

        if reuse_hits:
            self._increment_embedding_metric("reuse_index_hit_count", reuse_hits)
        self.logger.info(
            f"✓ Successfully generated embeddings for {len(mutable_elements)} code elements"
        )

        return mutable_elements

    def _try_reuse_element_embedding(
        self,
        element: CodeElementMeta,
        *,
        text: str,
        reuse_index: Mapping[str, CodeElementMeta] | None,
        current_fingerprint: Mapping[str, Any],
    ) -> bool:
        if not reuse_index:
            return False
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        for key in self._reuse_lookup_keys(element, text_hash):
            existing = reuse_index.get(key)
            if existing is None:
                continue
            existing_metadata = self._element_metadata(existing)
            existing_text_hash = existing_metadata.get(
                "embedding_text_hash"
            ) or existing.get("embedding_text_hash")
            if existing_text_hash != text_hash:
                continue
            existing_fingerprint = existing_metadata.get(
                "embedding_fingerprint"
            ) or existing.get("embedding_fingerprint")
            if not self._embedding_payload_matches(
                existing_fingerprint,
                current_fingerprint,
            ):
                continue
            embedding = existing.get("embedding")
            embedding_text = existing.get("embedding_text")
            if embedding is None or embedding_text != text:
                continue
            embedding_array = np.asarray(embedding, dtype=np.float32)
            if embedding_array.ndim != 1:
                continue
            expected_dim = int(
                getattr(self, "_configured_embedding_dim", 0)
                or getattr(self, "_embedding_dim", 0)
                or 0
            )
            if expected_dim and embedding_array.size != expected_dim:
                continue
            artifact_ref = existing.get("embedding_artifact_ref")
            self._attach_embedding_to_element(
                element,
                text=text,
                embedding=embedding_array,
                fingerprint_payload=dict(cast(Mapping[str, Any], existing_fingerprint)),
                artifact_ref=artifact_ref if isinstance(artifact_ref, str) else None,
            )
            return True
        return False

    @staticmethod
    def _element_metadata(element: Mapping[str, Any]) -> dict[str, Any]:
        metadata = element.get("metadata")
        return (
            dict(cast(Mapping[str, Any], metadata))
            if isinstance(metadata, Mapping)
            else {}
        )

    @classmethod
    def _reuse_lookup_keys(
        cls,
        element: CodeElementMeta,
        text_hash: str,
    ) -> list[str]:
        metadata = cls._element_metadata(element)
        keys: list[str] = [text_hash]
        for value in (
            element.get("stable_unit_id"),
            metadata.get("stable_unit_id"),
            element.get("id"),
        ):
            if isinstance(value, str) and value and value not in keys:
                keys.append(value)
        return keys

    @staticmethod
    def _embedding_payload_matches(
        existing: Any,
        current: Mapping[str, Any],
    ) -> bool:
        if not isinstance(existing, Mapping):
            return False
        existing_mapping = cast(Mapping[str, Any], existing)
        for field_name, expected_value in current.items():
            existing_value = existing_mapping.get(field_name)
            if field_name == "dimension" and (
                expected_value is None or existing_value is None
            ):
                continue
            if existing_value != expected_value:
                return False
        return True

    def _attach_embedding_to_element(
        self,
        element: CodeElementMeta,
        *,
        text: str,
        embedding: np.ndarray,
        fingerprint_payload: Mapping[str, Any],
        artifact_ref: str | None,
    ) -> None:
        element["embedding"] = np.asarray(embedding, dtype=np.float32)
        element["embedding_text"] = text
        element["embedding_artifact_ref"] = artifact_ref or self.embedding_artifact_ref(
            text
        )
        metadata = dict(element.get("metadata", {}) or {})
        metadata["embedding_text_hash"] = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
        metadata["embedding_fingerprint"] = dict(fingerprint_payload)
        element["metadata"] = metadata

    def _prepare_code_text(self, element: CodeElementMeta) -> str:
        """
        Prepare code element for embedding

        Combines various parts of the code element into a single text
        suitable for embedding
        """
        return self.prepare_text(element)

    def compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        if self.normalize:
            # Already normalized, just dot product
            return float(np.dot(embedding1, embedding2))
        # Compute cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def compute_similarities(
        self, query_embedding: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarities between query and multiple embeddings

        Args:
            query_embedding: Query embedding vector
            embeddings: Array of embedding vectors

        Returns:
            Array of similarity scores
        """
        if self.normalize:
            # Simple dot product for normalized embeddings
            similarities = np.dot(embeddings, query_embedding)
        else:
            # Compute cosine similarities
            norms = np.linalg.norm(embeddings, axis=1)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return np.zeros(len(embeddings))
            similarities = np.dot(embeddings, query_embedding) / (norms * query_norm)

        return similarities
