"""
CodeElement and CodeElementMeta — unified code element types for indexing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, NotRequired, TypedDict, cast


class CodeElementMeta(TypedDict, total=True):
    """Shape of CodeElement.to_dict() — used by vector_store metadata and consumers."""

    id: str
    type: str
    name: str
    file_path: str
    relative_path: str
    language: str
    start_line: int
    end_line: int
    code: str
    signature: str | None
    docstring: str | None
    summary: str | None
    metadata: dict[str, Any]
    repo_name: str | None
    repo_url: str | None
    # Added post-creation
    snapshot_id: NotRequired[str]
    source_priority: NotRequired[int]
    embedding: NotRequired[Any]  # np.ndarray
    embedding_text: NotRequired[str]
    ir_symbol_id: NotRequired[str]  # added by main.py IR resolution
    stable_unit_id: NotRequired[str]
    content_hash: NotRequired[str]
    syntax_hash: NotRequired[str]
    signature_hash: NotRequired[str]
    edge_surface_hash: NotRequired[str]
    embedding_text_hash: NotRequired[str]
    api_surface_hash: NotRequired[str]


@dataclass
class CodeElement:
    """Unified code element for indexing"""

    id: str
    type: str  # file, class, function, documentation
    name: str
    file_path: str
    relative_path: str
    language: str
    start_line: int
    end_line: int
    code: str
    signature: str | None
    docstring: str | None
    summary: str | None
    metadata: dict[str, Any]
    repo_name: str | None = None  # Repository identifier
    repo_url: str | None = None  # Repository URL (if available)

    def to_dict(self) -> CodeElementMeta:
        return cast(CodeElementMeta, asdict(self))
