"""
Code Embedder - Generate embeddings for code snippets
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import hashlib
import json
import logging
import platform
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..ir.element import CodeElementMeta

import numpy as np

_EMBEDDING_CACHE_FORMAT = "ndarray.float32.v1"
_EMBEDDING_TEXT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EmbeddingFingerprint:
    """Stable embedding model/config identity for reuse decisions."""

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


class CodeEmbedder:
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
        self._embedding_cache: Any | None = None
        self._embedding_cache_enabled = False

        # Auto-detect best available device: CUDA > MPS > CPU
        self.device = self._resolve_device(self.device)

        self.model: Any | None = None
        self._configured_embedding_dim: int = int(
            self.embedding_config.get("dimension")
            or self.embedding_config.get("embedding_dim")
            or 0
        )
        self._embedding_dim: int = self._configured_embedding_dim

        self._initialize_embedding_cache()
        self.logger.info(
            "Embedding provider configured: %s model=%s dimension=%s",
            self.provider,
            self.model_name,
            self._embedding_dim or "lazy",
        )

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
        if self.model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            model = self._load_model()
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

    def _get_cached_embedding(self, key: str) -> np.ndarray | None:
        if not self._embedding_cache_enabled or self._embedding_cache is None:
            return None
        if hasattr(self._embedding_cache, "get_cached_embedding_payload"):
            cached = self._embedding_cache.get_cached_embedding_payload(key)
        else:
            cached = self._embedding_cache.get(key)
        if cached is None:
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
        return embedding

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
        if not previous_dim:
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

    def _embed_texts_with_cache(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        if not self._embedding_cache_enabled or self._embedding_cache is None:
            return self._embed_batch_uncached(texts)

        embeddings: list[np.ndarray | None] = [None] * len(texts)
        misses: dict[str, tuple[str, list[int]]] = {}
        hits = 0

        for index, text in enumerate(texts):
            key = self._embedding_cache_key(text)
            cached_embedding = self._get_cached_embedding(key)
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
            miss_embeddings = np.asarray(
                self._embed_batch_uncached(miss_texts), dtype=np.float32
            )
            for (key, (text, indexes)), embedding in zip(
                miss_items, miss_embeddings, strict=True
            ):
                embedding_array = np.asarray(embedding, dtype=np.float32)
                for index in indexes:
                    embeddings[index] = embedding_array
                self._set_cached_embedding(key, text, embedding_array)

        if hits:
            self.logger.debug("Embedding cache reused %d/%d vectors", hits, len(texts))

        missing_indexes = [
            i for i, embedding in enumerate(embeddings) if embedding is None
        ]
        if missing_indexes:
            raise RuntimeError(
                f"Embedding cache fill failed at indexes: {missing_indexes}"
            )

        return np.vstack(cast(list[np.ndarray], embeddings))

    def _embed_text_ollama(self, text: str) -> np.ndarray:
        # Truncate text to avoid Ollama context window overflow
        if len(text) > self.max_seq_length:
            text = text[: self.max_seq_length]
        payload = {"model": self.model_name, "prompt": text}
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
        """
        Generate embeddings for code elements (functions, classes, etc.)

        Args:
            elements: List of code element dictionaries

        Returns:
            List of elements with embeddings added
        """
        if not elements:
            return []

        # Prepare texts for embedding
        texts = [self._prepare_code_text(elem) for elem in elements]

        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(texts)} code elements")
        embeddings = self._embed_texts_with_cache(texts)
        self.logger.info(
            f"✓ Successfully generated embeddings for {len(embeddings)} code elements"
        )

        fingerprint_payload = self.embedding_fingerprint(resolve_dimension=True)
        # Add embeddings to elements
        for elem, text, embedding in zip(elements, texts, embeddings, strict=True):
            elem["embedding"] = embedding
            elem["embedding_text"] = text
            elem["embedding_artifact_ref"] = self.embedding_artifact_ref(text)
            metadata = dict(elem.get("metadata", {}) or {})
            metadata["embedding_text_hash"] = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()
            metadata["embedding_fingerprint"] = dict(fingerprint_payload)
            elem["metadata"] = metadata

        return elements

    def _prepare_code_text(self, element: CodeElementMeta) -> str:
        """
        Prepare code element for embedding

        Combines various parts of the code element into a single text
        suitable for embedding
        """
        parts: list[str] = []

        # Add type
        if "type" in element:
            parts.append(f"Type: {element['type']}")

        # Add name
        if "name" in element:
            parts.append(f"Name: {element['name']}")

        # Add signature (for functions)
        if "signature" in element:
            parts.append(f"Signature: {element['signature']}")

        # Add docstring/description
        if element.get("docstring"):
            parts.append(f"Documentation: {element['docstring']}")

        # Add summary
        if summary := element.get("summary"):
            parts.append(summary)

        # Add code snippet (truncated)
        if "code" in element:
            code = element["code"]
            if len(code) > 10000:  # Truncate long code
                code = code[:10000] + "..."
            parts.append(f"Code:\n{code}")

        return "\n".join(parts)

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
