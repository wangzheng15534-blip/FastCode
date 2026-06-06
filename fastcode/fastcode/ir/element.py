"""
CodeElement and CodeElementMeta — unified code element types for indexing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

from fastcode.common.types import CodeElementMeta  # re-export from meaning_seed


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


def _copy_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {str(key): _copy_metadata_value(item) for key, item in mapping.items()}
    if isinstance(value, list):
        items = cast(list[Any], value)
        return [_copy_metadata_value(item) for item in items]
    if isinstance(value, tuple):
        items = cast(tuple[Any, ...], value)
        return tuple(_copy_metadata_value(item) for item in items)
    if isinstance(value, set):
        items = cast(set[Any], value)
        return {_copy_metadata_value(item) for item in items}
    return value


def serialize_code_element(element: CodeElement) -> CodeElementMeta:
    """Build an explicit CodeElement payload without recursive dataclass expansion."""
    return {
        "id": element.id,
        "type": element.type,
        "name": element.name,
        "file_path": element.file_path,
        "relative_path": element.relative_path,
        "language": element.language,
        "start_line": element.start_line,
        "end_line": element.end_line,
        "code": element.code,
        "signature": element.signature,
        "docstring": element.docstring,
        "summary": element.summary,
        "metadata": _copy_metadata_value(element.metadata or {}),
        "repo_name": element.repo_name,
        "repo_url": element.repo_url,
    }


def deserialize_code_element(payload: Any) -> CodeElement:
    """Reconstruct a CodeElement from a stable serialized payload."""
    if not isinstance(payload, Mapping):
        raise TypeError("Code element payload must be a mapping")
    mapping = cast(Mapping[str, Any], payload)
    metadata = mapping.get("metadata")
    return CodeElement(
        id=str(mapping.get("id") or ""),
        type=str(mapping.get("type") or ""),
        name=str(mapping.get("name") or ""),
        file_path=str(mapping.get("file_path") or ""),
        relative_path=str(mapping.get("relative_path") or ""),
        language=str(mapping.get("language") or ""),
        start_line=int(mapping.get("start_line") or 0),
        end_line=int(mapping.get("end_line") or 0),
        code=str(mapping.get("code") or ""),
        signature=None
        if mapping.get("signature") is None
        else str(mapping["signature"]),
        docstring=None
        if mapping.get("docstring") is None
        else str(mapping["docstring"]),
        summary=None if mapping.get("summary") is None else str(mapping["summary"]),
        metadata=dict(cast(Mapping[str, Any], metadata))
        if isinstance(metadata, Mapping)
        else {},
        repo_name=None
        if mapping.get("repo_name") is None
        else str(mapping["repo_name"]),
        repo_url=None if mapping.get("repo_url") is None else str(mapping["repo_url"]),
    )
