"""Vector store persistence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from fastcode.ir.element import CodeElementMeta


def _string_key_mapping_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in cast(dict[Any, Any], value).items()}


def _optional_int_payload(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class RepositoryOverviewRecord:
    repo_name: str
    content: str
    metadata_json: str
    embedding: Any | None = None
    embedding_fingerprint: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_name": self.repo_name,
            "content": self.content,
            "metadata_json": self.metadata_json,
            "embedding": self.embedding,
            "embedding_fingerprint": (
                dict(self.embedding_fingerprint)
                if self.embedding_fingerprint is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepositoryOverviewRecord:
        raw_fingerprint = data.get("embedding_fingerprint")
        return cls(
            repo_name=str(data.get("repo_name") or ""),
            content=str(data.get("content") or ""),
            metadata_json=str(data.get("metadata_json") or "{}"),
            embedding=data.get("embedding"),
            embedding_fingerprint=(
                _string_key_mapping_payload(raw_fingerprint)
                if isinstance(raw_fingerprint, dict)
                else None
            ),
        )


@dataclass(frozen=True)
class VectorSearchResultRecord:
    metadata: CodeElementMeta
    score: float
    index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "score": self.score,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorSearchResultRecord:
        raw_metadata = data.get("metadata")
        metadata = (
            {str(key): item for key, item in cast(dict[Any, Any], raw_metadata).items()}
            if isinstance(raw_metadata, dict)
            else {}
        )
        raw_index = data.get("index")
        return cls(
            metadata=cast(CodeElementMeta, metadata),
            score=float(data.get("score") or 0.0),
            index=_optional_int_payload(raw_index),
        )
