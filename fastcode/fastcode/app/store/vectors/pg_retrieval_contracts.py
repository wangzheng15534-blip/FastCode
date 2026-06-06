"""PostgreSQL retrieval persistence contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast


def _string_list_payload(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in cast(list[Any], value)]
    if isinstance(value, tuple):
        return [str(item) for item in cast(tuple[Any, ...], value)]
    return []


def _string_key_mapping_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in cast(dict[Any, Any], value).items()}


def _empty_mapping_payload() -> dict[str, Any]:
    return {}


def _optional_int_payload(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class PgRetrievalElementRecord:
    id: str = ""
    element_type: str | None = None
    name: str | None = None
    file_path: str | None = None
    relative_path: str | None = None
    language: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    code: str | None = None
    signature: str | None = None
    docstring: str | None = None
    summary: str | None = None
    repo_name: str | None = None
    repo_url: str | None = None
    snapshot_id: str | None = None
    source_priority: Any | None = None
    embedding_text: str | None = None
    embedding_artifact_ref: str | None = None
    embedding_fingerprint: dict[str, Any] | None = None
    ir_symbol_id: str | None = None
    stable_unit_id: str | None = None
    content_hash: str | None = None
    syntax_hash: str | None = None
    signature_hash: str | None = None
    edge_surface_hash: str | None = None
    embedding_text_hash: str | None = None
    api_surface_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=_empty_mapping_payload)
    present_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _pg_retrieval_element_record_to_payload(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PgRetrievalElementRecord:
        return _pg_retrieval_element_record_from_payload(data)


def _pg_retrieval_element_record_to_payload(
    record: PgRetrievalElementRecord,
) -> dict[str, Any]:
    return {
        "id": record.id,
        "element_type": record.element_type,
        "name": record.name,
        "file_path": record.file_path,
        "relative_path": record.relative_path,
        "language": record.language,
        "start_line": record.start_line,
        "end_line": record.end_line,
        "code": record.code,
        "signature": record.signature,
        "docstring": record.docstring,
        "summary": record.summary,
        "repo_name": record.repo_name,
        "repo_url": record.repo_url,
        "snapshot_id": record.snapshot_id,
        "source_priority": record.source_priority,
        "embedding_text": record.embedding_text,
        "embedding_artifact_ref": record.embedding_artifact_ref,
        "embedding_fingerprint": (
            dict(record.embedding_fingerprint)
            if record.embedding_fingerprint is not None
            else None
        ),
        "ir_symbol_id": record.ir_symbol_id,
        "stable_unit_id": record.stable_unit_id,
        "content_hash": record.content_hash,
        "syntax_hash": record.syntax_hash,
        "signature_hash": record.signature_hash,
        "edge_surface_hash": record.edge_surface_hash,
        "embedding_text_hash": record.embedding_text_hash,
        "api_surface_hash": record.api_surface_hash,
        "metadata": dict(record.metadata),
        "present_fields": list(record.present_fields),
    }


def _pg_retrieval_element_record_from_payload(
    data: dict[str, Any],
) -> PgRetrievalElementRecord:
    raw_type = data.get("element_type")
    if raw_type is None:
        raw_type = data.get("type")
    raw_fingerprint = data.get("embedding_fingerprint")
    return PgRetrievalElementRecord(
        id=str(data.get("id") or ""),
        element_type=str(raw_type) if raw_type is not None else None,
        name=str(data["name"]) if data.get("name") is not None else None,
        file_path=(
            str(data["file_path"]) if data.get("file_path") is not None else None
        ),
        relative_path=(
            str(data["relative_path"])
            if data.get("relative_path") is not None
            else None
        ),
        language=str(data["language"]) if data.get("language") is not None else None,
        start_line=_optional_int_payload(data.get("start_line")),
        end_line=_optional_int_payload(data.get("end_line")),
        code=str(data["code"]) if data.get("code") is not None else None,
        signature=str(data["signature"]) if data.get("signature") is not None else None,
        docstring=str(data["docstring"]) if data.get("docstring") is not None else None,
        summary=str(data["summary"]) if data.get("summary") is not None else None,
        repo_name=str(data["repo_name"]) if data.get("repo_name") is not None else None,
        repo_url=str(data["repo_url"]) if data.get("repo_url") is not None else None,
        snapshot_id=(
            str(data["snapshot_id"]) if data.get("snapshot_id") is not None else None
        ),
        source_priority=data.get("source_priority"),
        embedding_text=(
            str(data["embedding_text"])
            if data.get("embedding_text") is not None
            else None
        ),
        embedding_artifact_ref=(
            str(data["embedding_artifact_ref"])
            if data.get("embedding_artifact_ref") is not None
            else None
        ),
        embedding_fingerprint=(
            _string_key_mapping_payload(raw_fingerprint)
            if isinstance(raw_fingerprint, dict)
            else None
        ),
        ir_symbol_id=(
            str(data["ir_symbol_id"]) if data.get("ir_symbol_id") is not None else None
        ),
        stable_unit_id=(
            str(data["stable_unit_id"])
            if data.get("stable_unit_id") is not None
            else None
        ),
        content_hash=(
            str(data["content_hash"]) if data.get("content_hash") is not None else None
        ),
        syntax_hash=(
            str(data["syntax_hash"]) if data.get("syntax_hash") is not None else None
        ),
        signature_hash=(
            str(data["signature_hash"])
            if data.get("signature_hash") is not None
            else None
        ),
        edge_surface_hash=(
            str(data["edge_surface_hash"])
            if data.get("edge_surface_hash") is not None
            else None
        ),
        embedding_text_hash=(
            str(data["embedding_text_hash"])
            if data.get("embedding_text_hash") is not None
            else None
        ),
        api_surface_hash=(
            str(data["api_surface_hash"])
            if data.get("api_surface_hash") is not None
            else None
        ),
        metadata=_string_key_mapping_payload(data.get("metadata")),
        present_fields=tuple(_string_list_payload(data.get("present_fields"))),
    )


@dataclass(frozen=True)
class PgRetrievalResultRecord:
    element: PgRetrievalElementRecord
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "element": _pg_retrieval_element_record_to_payload(self.element),
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PgRetrievalResultRecord:
        raw_element = data.get("element")
        element_payload = (
            cast(dict[str, Any], raw_element) if isinstance(raw_element, dict) else {}
        )
        return cls(
            element=_pg_retrieval_element_record_from_payload(element_payload),
            score=float(data.get("score") or 0.0),
        )
