"""
Code Embedder - Generate embeddings for code snippets
"""

import json
import logging
import platform
import urllib.request
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class CodeEmbedder:
    """Generate embeddings for code using sentence transformers"""

    def __init__(self, config: dict[str, Any]):
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

        # Auto-detect best available device: CUDA > MPS > CPU
        if self.device != "cpu":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        self.model: SentenceTransformer | None = None
        if self.provider == "ollama":
            self.logger.info(
                f"Using Ollama embeddings model: {self.model_name} ({self.ollama_url})"
            )
            probe = self._embed_text_ollama("embedding dimension probe")
            self.embedding_dim = len(probe)
        else:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = self._load_model()
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

        self.logger.info(f"Embedding dimension: {self.embedding_dim}")

    def _load_model(self) -> SentenceTransformer:
        """Load sentence transformer model"""
        model = SentenceTransformer(self.model_name, device=self.device)
        model.max_seq_length = self.max_seq_length
        return model

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
        """
        Generate embeddings for a batch of texts

        Args:
            texts: List of input texts

        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])

        if self.provider == "ollama":
            vectors = [self._embed_text_ollama(t) for t in texts]
            return np.array(vectors, dtype=np.float32)

        encode_kwargs = {
            "batch_size": self.batch_size,
            "show_progress_bar": len(texts) > 100,
            "normalize_embeddings": self.normalize,
            "convert_to_numpy": True,
            "device": self.device,
            "convert_to_tensor": False,
        }

        if platform.system() == "Darwin":
            encode_kwargs["pool"] = None

        return self.model.encode(texts, **encode_kwargs)

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
        return np.array(vector, dtype=np.float32)

    def embed_code_elements(
        self, elements: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
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
        embeddings = self.embed_batch(texts)
        self.logger.info(
            f"✓ Successfully generated embeddings for {len(embeddings)} code elements"
        )

        # Add embeddings to elements
        for elem, embedding in zip(elements, embeddings, strict=True):
            elem["embedding"] = embedding
            elem["embedding_text"] = texts[elements.index(elem)]

        return elements

    def _prepare_code_text(self, element: dict[str, Any]) -> str:
        """
        Prepare code element for embedding

        Combines various parts of the code element into a single text
        suitable for embedding
        """
        parts = []

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
        if element.get("summary"):
            parts.append(element["summary"])

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
