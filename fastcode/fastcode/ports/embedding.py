"""Embedding capability ports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fastcode.kernel.types import CodeElementMeta


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Prepared-text embedding capability used by indexing and retrieval flows."""

    @property
    def embedding_dim(self) -> int: ...

    def prepare_text(self, element: CodeElementMeta) -> str: ...

    def fingerprint(self, *, resolve_dimension: bool = False) -> dict[str, Any]: ...

    def embed_many(self, texts: Sequence[str]) -> Any: ...

    def embed_elements(
        self,
        elements: Sequence[CodeElementMeta],
        reuse_index: Mapping[str, CodeElementMeta] | None = None,
    ) -> list[CodeElementMeta]: ...
