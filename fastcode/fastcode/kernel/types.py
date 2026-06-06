"""Shared kernel type definitions promoted from meaning_core layers."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class CodeElementMeta(TypedDict, total=True):
    """Shape of CodeElement.to_dict() — used by vector_store metadata and consumers.

    Promoted to kernel (meaning_seed) so axis_surface (ports) can reference it
    without depending on meaning_core (ir).
    """

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
    embedding_artifact_ref: NotRequired[str]
    embedding_fingerprint: NotRequired[dict[str, Any]]
    ir_symbol_id: NotRequired[str]  # added by main.py IR resolution
    stable_unit_id: NotRequired[str]
    content_hash: NotRequired[str]
    syntax_hash: NotRequired[str]
    signature_hash: NotRequired[str]
    edge_surface_hash: NotRequired[str]
    embedding_text_hash: NotRequired[str]
    api_surface_hash: NotRequired[str]
