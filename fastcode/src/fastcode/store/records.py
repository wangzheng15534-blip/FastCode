"""Typed records for store persistence boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from ..ir.element import CodeElementMeta


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


def _dict_list_payload(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        items = cast(list[Any], value)
    elif isinstance(value, tuple):
        items = list(cast(tuple[Any, ...], value))
    else:
        return []
    payload: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            payload.append(
                {
                    str(key): sub_item
                    for key, sub_item in cast(dict[Any, Any], item).items()
                }
            )
    return payload


@dataclass(frozen=True)
class ManifestRecord:
    manifest_id: str
    repo_name: str
    ref_name: str
    snapshot_id: str
    index_run_id: str
    published_at: str
    previous_manifest_id: str | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "repo_name": self.repo_name,
            "ref_name": self.ref_name,
            "snapshot_id": self.snapshot_id,
            "index_run_id": self.index_run_id,
            "published_at": self.published_at,
            "previous_manifest_id": self.previous_manifest_id,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestRecord:
        return cls(
            manifest_id=str(data.get("manifest_id") or ""),
            repo_name=str(data.get("repo_name") or ""),
            ref_name=str(data.get("ref_name") or ""),
            snapshot_id=str(data.get("snapshot_id") or ""),
            index_run_id=str(data.get("index_run_id") or ""),
            published_at=str(data.get("published_at") or ""),
            previous_manifest_id=(
                str(data["previous_manifest_id"])
                if data.get("previous_manifest_id") is not None
                else None
            ),
            status=str(data.get("status") or ""),
        )


@dataclass(frozen=True)
class SnapshotRefRecord:
    ref_id: int | None
    repo_name: str
    branch: str | None
    commit_id: str | None
    tree_id: str | None
    snapshot_id: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "repo_name": self.repo_name,
            "branch": self.branch,
            "commit_id": self.commit_id,
            "tree_id": self.tree_id,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotRefRecord:
        raw_ref_id = data.get("ref_id")
        return cls(
            ref_id=int(raw_ref_id) if raw_ref_id is not None else None,
            repo_name=str(data.get("repo_name") or ""),
            branch=str(data["branch"]) if data.get("branch") is not None else None,
            commit_id=(
                str(data["commit_id"]) if data.get("commit_id") is not None else None
            ),
            tree_id=str(data["tree_id"]) if data.get("tree_id") is not None else None,
            snapshot_id=str(data.get("snapshot_id") or ""),
            created_at=str(data.get("created_at") or ""),
        )


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    repo_name: str
    branch: str | None
    commit_id: str | None
    tree_id: str | None
    artifact_key: str
    ir_path: str
    ir_graphs_path: str | None
    created_at: str
    metadata_json: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "repo_name": self.repo_name,
            "branch": self.branch,
            "commit_id": self.commit_id,
            "tree_id": self.tree_id,
            "artifact_key": self.artifact_key,
            "ir_path": self.ir_path,
            "ir_graphs_path": self.ir_graphs_path,
            "created_at": self.created_at,
            "metadata_json": self.metadata_json,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotRecord:
        return cls(
            snapshot_id=str(data.get("snapshot_id") or ""),
            repo_name=str(data.get("repo_name") or ""),
            branch=str(data["branch"]) if data.get("branch") is not None else None,
            commit_id=(
                str(data["commit_id"]) if data.get("commit_id") is not None else None
            ),
            tree_id=str(data["tree_id"]) if data.get("tree_id") is not None else None,
            artifact_key=str(data.get("artifact_key") or ""),
            ir_path=str(data.get("ir_path") or ""),
            ir_graphs_path=(
                str(data["ir_graphs_path"])
                if data.get("ir_graphs_path") is not None
                else None
            ),
            created_at=str(data.get("created_at") or ""),
            metadata_json=(
                str(data["metadata_json"])
                if data.get("metadata_json") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class SCIPArtifactRecord:
    snapshot_id: str
    indexer_name: str
    indexer_version: str | None
    artifact_path: str
    checksum: str
    created_at: str
    artifact_id: str | None = None
    sequence_no: int | None = None
    role: str | None = None
    metadata_json: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "indexer_name": self.indexer_name,
            "indexer_version": self.indexer_version,
            "artifact_path": self.artifact_path,
            "checksum": self.checksum,
            "created_at": self.created_at,
            "artifact_id": self.artifact_id,
            "sequence_no": self.sequence_no,
            "role": self.role,
            "metadata_json": self.metadata_json,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIPArtifactRecord:
        raw_sequence_no = data.get("sequence_no")
        return cls(
            snapshot_id=str(data.get("snapshot_id") or ""),
            indexer_name=str(data.get("indexer_name") or ""),
            indexer_version=(
                str(data["indexer_version"])
                if data.get("indexer_version") is not None
                else None
            ),
            artifact_path=str(data.get("artifact_path") or ""),
            checksum=str(data.get("checksum") or ""),
            created_at=str(data.get("created_at") or ""),
            artifact_id=(
                str(data["artifact_id"])
                if data.get("artifact_id") is not None
                else None
            ),
            sequence_no=(int(raw_sequence_no) if raw_sequence_no is not None else None),
            role=str(data["role"]) if data.get("role") is not None else None,
            metadata_json=(
                str(data["metadata_json"])
                if data.get("metadata_json") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class UnitArtifactRecord:
    snapshot_id: str
    stable_unit_id: str
    relative_path: str
    unit_type: str
    content_hash: str | None
    syntax_hash: str | None
    signature_hash: str | None
    edge_surface_hash: str | None
    embedding_text_hash: str | None
    api_surface_hash: str | None
    embedding_artifact_ref: str | None
    scoped_tool_ref: str | None
    package_root: str | None
    repair_frontier_summary: str | None
    metadata_json: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "stable_unit_id": self.stable_unit_id,
            "relative_path": self.relative_path,
            "unit_type": self.unit_type,
            "content_hash": self.content_hash,
            "syntax_hash": self.syntax_hash,
            "signature_hash": self.signature_hash,
            "edge_surface_hash": self.edge_surface_hash,
            "embedding_text_hash": self.embedding_text_hash,
            "api_surface_hash": self.api_surface_hash,
            "embedding_artifact_ref": self.embedding_artifact_ref,
            "scoped_tool_ref": self.scoped_tool_ref,
            "package_root": self.package_root,
            "repair_frontier_summary": self.repair_frontier_summary,
            "metadata_json": self.metadata_json,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnitArtifactRecord:
        return cls(
            snapshot_id=str(data.get("snapshot_id") or ""),
            stable_unit_id=str(data.get("stable_unit_id") or ""),
            relative_path=str(data.get("relative_path") or ""),
            unit_type=str(data.get("unit_type") or ""),
            content_hash=(
                str(data["content_hash"])
                if data.get("content_hash") is not None
                else None
            ),
            syntax_hash=(
                str(data["syntax_hash"])
                if data.get("syntax_hash") is not None
                else None
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
            embedding_artifact_ref=(
                str(data["embedding_artifact_ref"])
                if data.get("embedding_artifact_ref") is not None
                else None
            ),
            scoped_tool_ref=(
                str(data["scoped_tool_ref"])
                if data.get("scoped_tool_ref") is not None
                else None
            ),
            package_root=(
                str(data["package_root"])
                if data.get("package_root") is not None
                else None
            ),
            repair_frontier_summary=(
                str(data["repair_frontier_summary"])
                if data.get("repair_frontier_summary") is not None
                else None
            ),
            metadata_json=(
                str(data["metadata_json"])
                if data.get("metadata_json") is not None
                else None
            ),
            created_at=str(data.get("created_at") or ""),
        )


@dataclass(frozen=True)
class FileIRShardRecord:
    snapshot_id: str
    relative_path: str
    schema_version: str
    payload_json: str
    unit_count: int
    support_count: int
    relation_count: int
    embedding_count: int
    content_hash: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "relative_path": self.relative_path,
            "schema_version": self.schema_version,
            "payload_json": self.payload_json,
            "unit_count": self.unit_count,
            "support_count": self.support_count,
            "relation_count": self.relation_count,
            "embedding_count": self.embedding_count,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileIRShardRecord:
        return cls(
            snapshot_id=str(data.get("snapshot_id") or ""),
            relative_path=str(data.get("relative_path") or ""),
            schema_version=str(data.get("schema_version") or ""),
            payload_json=str(data.get("payload_json") or "{}"),
            unit_count=int(data.get("unit_count") or 0),
            support_count=int(data.get("support_count") or 0),
            relation_count=int(data.get("relation_count") or 0),
            embedding_count=int(data.get("embedding_count") or 0),
            content_hash=(
                str(data["content_hash"])
                if data.get("content_hash") is not None
                else None
            ),
            created_at=str(data.get("created_at") or ""),
        )


@dataclass(frozen=True)
class FileArtifactRecord:
    repo_name: str
    relative_path: str
    identity_kind: str
    identity_value: str
    artifact_type: str
    schema_version: str
    payload_json: str
    unit_count: int
    support_count: int
    relation_count: int
    embedding_count: int
    metadata_json: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_name": self.repo_name,
            "relative_path": self.relative_path,
            "identity_kind": self.identity_kind,
            "identity_value": self.identity_value,
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "payload_json": self.payload_json,
            "unit_count": self.unit_count,
            "support_count": self.support_count,
            "relation_count": self.relation_count,
            "embedding_count": self.embedding_count,
            "metadata_json": self.metadata_json,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileArtifactRecord:
        return cls(
            repo_name=str(data.get("repo_name") or ""),
            relative_path=str(data.get("relative_path") or ""),
            identity_kind=str(data.get("identity_kind") or ""),
            identity_value=str(data.get("identity_value") or ""),
            artifact_type=str(data.get("artifact_type") or ""),
            schema_version=str(data.get("schema_version") or ""),
            payload_json=str(data.get("payload_json") or "{}"),
            unit_count=int(data.get("unit_count") or 0),
            support_count=int(data.get("support_count") or 0),
            relation_count=int(data.get("relation_count") or 0),
            embedding_count=int(data.get("embedding_count") or 0),
            metadata_json=(
                str(data["metadata_json"])
                if data.get("metadata_json") is not None
                else None
            ),
            created_at=str(data.get("created_at") or ""),
        )


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


@dataclass(frozen=True)
class QueryResultCacheRecord:
    query: str
    repo_hash: str
    result: Any
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "repo_hash": self.repo_hash,
            "result": self.result,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryResultCacheRecord:
        created_at = data.get("created_at")
        return cls(
            query=str(data.get("query") or ""),
            repo_hash=str(data.get("repo_hash") or ""),
            result=data.get("result"),
            created_at=(
                float(created_at) if isinstance(created_at, (int, float)) else 0.0
            ),
        )


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


@dataclass(frozen=True)
class RedoTaskRecord:
    task_id: str
    task_type: str
    payload_json: str
    status: str
    attempts: int
    last_error: str | None
    next_attempt_at: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload_json": self.payload_json,
            "status": self.status,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "next_attempt_at": self.next_attempt_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RedoTaskRecord:
        return cls(
            task_id=str(data.get("task_id") or ""),
            task_type=str(data.get("task_type") or ""),
            payload_json=str(data.get("payload_json") or ""),
            status=str(data.get("status") or ""),
            attempts=int(data.get("attempts") or 0),
            last_error=(
                str(data["last_error"]) if data.get("last_error") is not None else None
            ),
            next_attempt_at=(
                str(data["next_attempt_at"])
                if data.get("next_attempt_at") is not None
                else None
            ),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass(frozen=True)
class OutboxEventRecord:
    event_id: str
    event_type: str
    payload: str
    snapshot_id: str
    status: str
    attempts: int
    max_attempts: int
    created_at: str
    last_attempt_at: str | None
    error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "snapshot_id": self.snapshot_id,
            "status": self.status,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "last_attempt_at": self.last_attempt_at,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutboxEventRecord:
        return cls(
            event_id=str(data.get("event_id") or ""),
            event_type=str(data.get("event_type") or ""),
            payload=str(data.get("payload") or ""),
            snapshot_id=str(data.get("snapshot_id") or ""),
            status=str(data.get("status") or ""),
            attempts=int(data.get("attempts") or 0),
            max_attempts=int(data.get("max_attempts") or 0),
            created_at=str(data.get("created_at") or ""),
            last_attempt_at=(
                str(data["last_attempt_at"])
                if data.get("last_attempt_at") is not None
                else None
            ),
            error_message=(
                str(data["error_message"])
                if data.get("error_message") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ProjectionDirtyScopeRecord:
    snapshot_id: str
    scope_kind: str
    scope_key: str
    dirty_paths: list[str]
    dirty_units: list[str]
    dirty_package_roots: list[str]
    dirty_reason: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "scope_kind": self.scope_kind,
            "scope_key": self.scope_key,
            "dirty_paths": list(self.dirty_paths),
            "dirty_units": list(self.dirty_units),
            "dirty_package_roots": list(self.dirty_package_roots),
            "dirty_reason": self.dirty_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectionDirtyScopeRecord:
        return cls(
            snapshot_id=str(data.get("snapshot_id") or ""),
            scope_kind=str(data.get("scope_kind") or ""),
            scope_key=str(data.get("scope_key") or ""),
            dirty_paths=_string_list_payload(data.get("dirty_paths")),
            dirty_units=_string_list_payload(data.get("dirty_units")),
            dirty_package_roots=_string_list_payload(data.get("dirty_package_roots")),
            dirty_reason=str(data.get("dirty_reason") or ""),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass(frozen=True)
class ProjectionBuildRecord:
    projection_id: str
    snapshot_id: str
    scope_kind: str
    scope_key: str
    params_hash: str
    status: str
    warnings: list[str]
    created_at: str
    updated_at: str
    query: str | None
    target_id: str | None
    filters: dict[str, Any]
    coverage_paths: list[str]
    coverage_nodes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_id": self.projection_id,
            "snapshot_id": self.snapshot_id,
            "scope_kind": self.scope_kind,
            "scope_key": self.scope_key,
            "params_hash": self.params_hash,
            "status": self.status,
            "warnings": list(self.warnings),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "query": self.query,
            "target_id": self.target_id,
            "filters": dict(self.filters),
            "coverage_paths": list(self.coverage_paths),
            "coverage_nodes": list(self.coverage_nodes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectionBuildRecord:
        return cls(
            projection_id=str(data.get("projection_id") or ""),
            snapshot_id=str(data.get("snapshot_id") or ""),
            scope_kind=str(data.get("scope_kind") or ""),
            scope_key=str(data.get("scope_key") or ""),
            params_hash=str(data.get("params_hash") or ""),
            status=str(data.get("status") or ""),
            warnings=_string_list_payload(data.get("warnings")),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            query=str(data["query"]) if data.get("query") is not None else None,
            target_id=(
                str(data["target_id"]) if data.get("target_id") is not None else None
            ),
            filters=_string_key_mapping_payload(data.get("filters")),
            coverage_paths=_string_list_payload(data.get("coverage_paths")),
            coverage_nodes=_string_list_payload(data.get("coverage_nodes")),
        )


@dataclass(frozen=True)
class IndexRunRecord:
    run_id: str
    repo_name: str
    snapshot_id: str
    branch: str | None
    commit_id: str | None
    idempotency_key: str | None
    status: str
    error_message: str | None
    warnings_json: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "repo_name": self.repo_name,
            "snapshot_id": self.snapshot_id,
            "branch": self.branch,
            "commit_id": self.commit_id,
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "error_message": self.error_message,
            "warnings_json": self.warnings_json,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexRunRecord:
        return cls(
            run_id=str(data.get("run_id") or ""),
            repo_name=str(data.get("repo_name") or ""),
            snapshot_id=str(data.get("snapshot_id") or ""),
            branch=str(data["branch"]) if data.get("branch") is not None else None,
            commit_id=(
                str(data["commit_id"]) if data.get("commit_id") is not None else None
            ),
            idempotency_key=(
                str(data["idempotency_key"])
                if data.get("idempotency_key") is not None
                else None
            ),
            status=str(data.get("status") or ""),
            error_message=(
                str(data["error_message"])
                if data.get("error_message") is not None
                else None
            ),
            warnings_json=(
                str(data["warnings_json"])
                if data.get("warnings_json") is not None
                else None
            ),
            created_at=str(data.get("created_at") or ""),
            started_at=(
                str(data["started_at"]) if data.get("started_at") is not None else None
            ),
            completed_at=(
                str(data["completed_at"])
                if data.get("completed_at") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class PublishTaskRecord:
    task_id: str
    run_id: str
    snapshot_id: str
    manifest_id: str | None
    status: str
    attempts: int
    last_error: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "snapshot_id": self.snapshot_id,
            "manifest_id": self.manifest_id,
            "status": self.status,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PublishTaskRecord:
        return cls(
            task_id=str(data.get("task_id") or ""),
            run_id=str(data.get("run_id") or ""),
            snapshot_id=str(data.get("snapshot_id") or ""),
            manifest_id=(
                str(data["manifest_id"])
                if data.get("manifest_id") is not None
                else None
            ),
            status=str(data.get("status") or ""),
            attempts=int(data.get("attempts") or 0),
            last_error=(
                str(data["last_error"]) if data.get("last_error") is not None else None
            ),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass(frozen=True)
class DialogueTurnRecord:
    session_id: str
    turn_number: int
    timestamp: float
    query: str
    answer: str
    summary: str
    retrieved_elements: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
            "query": self.query,
            "answer": self.answer,
            "summary": self.summary,
            "retrieved_elements": list(self.retrieved_elements),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DialogueTurnRecord:
        timestamp_value = data.get("timestamp")
        return cls(
            session_id=str(data.get("session_id") or ""),
            turn_number=int(data.get("turn_number") or 0),
            timestamp=(
                float(timestamp_value)
                if isinstance(timestamp_value, (int, float))
                else 0.0
            ),
            query=str(data.get("query") or ""),
            answer=str(data.get("answer") or ""),
            summary=str(data.get("summary") or ""),
            retrieved_elements=_dict_list_payload(data.get("retrieved_elements")),
            metadata=_string_key_mapping_payload(data.get("metadata")),
        )


@dataclass(frozen=True)
class DialogueSessionRecord:
    session_id: str
    created_at: float
    total_turns: int
    last_updated: float
    multi_turn: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "total_turns": self.total_turns,
            "last_updated": self.last_updated,
            "multi_turn": self.multi_turn,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DialogueSessionRecord:
        created_at = data.get("created_at")
        last_updated = data.get("last_updated")
        return cls(
            session_id=str(data.get("session_id") or ""),
            created_at=(
                float(created_at) if isinstance(created_at, (int, float)) else 0.0
            ),
            total_turns=int(data.get("total_turns") or 0),
            last_updated=(
                float(last_updated) if isinstance(last_updated, (int, float)) else 0.0
            ),
            multi_turn=bool(data.get("multi_turn", False)),
        )


@dataclass(frozen=True)
class TurnJournalRecord:
    session_id: str
    turn_number: int
    snapshot_id: str | None
    artifact_key: str | None
    compiler_fingerprint: str
    payload_json: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "snapshot_id": self.snapshot_id,
            "artifact_key": self.artifact_key,
            "compiler_fingerprint": self.compiler_fingerprint,
            "payload_json": self.payload_json,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnJournalRecord:
        created_at = data.get("created_at")
        return cls(
            session_id=str(data.get("session_id") or ""),
            turn_number=int(data.get("turn_number") or 0),
            snapshot_id=(
                str(data["snapshot_id"])
                if data.get("snapshot_id") is not None
                else None
            ),
            artifact_key=(
                str(data["artifact_key"])
                if data.get("artifact_key") is not None
                else None
            ),
            compiler_fingerprint=str(data.get("compiler_fingerprint") or ""),
            payload_json=str(data.get("payload_json") or ""),
            created_at=float(created_at)
            if isinstance(created_at, (int, float))
            else 0.0,
        )


@dataclass(frozen=True)
class WorkingMemoryRecord:
    session_id: str
    turn_number: int
    snapshot_id: str | None
    artifact_key: str | None
    compiler_fingerprint: str
    payload_json: str
    stable_fcx: str
    turn_fcx: str
    obs_fcx: str
    full_fcx: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "snapshot_id": self.snapshot_id,
            "artifact_key": self.artifact_key,
            "compiler_fingerprint": self.compiler_fingerprint,
            "payload_json": self.payload_json,
            "stable_fcx": self.stable_fcx,
            "turn_fcx": self.turn_fcx,
            "obs_fcx": self.obs_fcx,
            "full_fcx": self.full_fcx,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkingMemoryRecord:
        created_at = data.get("created_at")
        return cls(
            session_id=str(data.get("session_id") or ""),
            turn_number=int(data.get("turn_number") or 0),
            snapshot_id=(
                str(data["snapshot_id"])
                if data.get("snapshot_id") is not None
                else None
            ),
            artifact_key=(
                str(data["artifact_key"])
                if data.get("artifact_key") is not None
                else None
            ),
            compiler_fingerprint=str(data.get("compiler_fingerprint") or ""),
            payload_json=str(data.get("payload_json") or ""),
            stable_fcx=str(data.get("stable_fcx") or ""),
            turn_fcx=str(data.get("turn_fcx") or ""),
            obs_fcx=str(data.get("obs_fcx") or ""),
            full_fcx=str(data.get("full_fcx") or ""),
            created_at=float(created_at)
            if isinstance(created_at, (int, float))
            else 0.0,
        )


@dataclass(frozen=True)
class HandoffArtifactRecord:
    artifact_id: str
    session_id: str
    turn_number: int
    snapshot_id: str | None
    compiler_fingerprint: str
    mode: str
    payload_json: str
    full_fcx: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "snapshot_id": self.snapshot_id,
            "compiler_fingerprint": self.compiler_fingerprint,
            "mode": self.mode,
            "payload_json": self.payload_json,
            "full_fcx": self.full_fcx,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandoffArtifactRecord:
        created_at = data.get("created_at")
        return cls(
            artifact_id=str(data.get("artifact_id") or ""),
            session_id=str(data.get("session_id") or ""),
            turn_number=int(data.get("turn_number") or 0),
            snapshot_id=(
                str(data["snapshot_id"])
                if data.get("snapshot_id") is not None
                else None
            ),
            compiler_fingerprint=str(data.get("compiler_fingerprint") or ""),
            mode=str(data.get("mode") or ""),
            payload_json=str(data.get("payload_json") or ""),
            full_fcx=str(data.get("full_fcx") or ""),
            created_at=float(created_at)
            if isinstance(created_at, (int, float))
            else 0.0,
        )


@dataclass(frozen=True)
class ContextBundleRecord:
    bundle_id: str
    session_id: str
    turn_number: int
    snapshot_id: str | None
    artifact_key: str | None
    compiler_fingerprint: str
    payload_json: str
    invalidation_key: str
    created_at: float
    projection_fingerprint: str = "projection:none"
    embedding_fingerprint: str = "embedding:unknown"
    retrieval_policy_fingerprint: str = "retrieval:default"
    distillation_prompt_fingerprint: str = "distill:v1"
    budget_fingerprint: str = "budget:default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "snapshot_id": self.snapshot_id,
            "artifact_key": self.artifact_key,
            "compiler_fingerprint": self.compiler_fingerprint,
            "payload_json": self.payload_json,
            "invalidation_key": self.invalidation_key,
            "created_at": self.created_at,
            "projection_fingerprint": self.projection_fingerprint,
            "embedding_fingerprint": self.embedding_fingerprint,
            "retrieval_policy_fingerprint": self.retrieval_policy_fingerprint,
            "distillation_prompt_fingerprint": (self.distillation_prompt_fingerprint),
            "budget_fingerprint": self.budget_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextBundleRecord:
        created_at = data.get("created_at")
        return cls(
            bundle_id=str(data.get("bundle_id") or ""),
            session_id=str(data.get("session_id") or ""),
            turn_number=int(data.get("turn_number") or 0),
            snapshot_id=(
                str(data["snapshot_id"])
                if data.get("snapshot_id") is not None
                else None
            ),
            artifact_key=(
                str(data["artifact_key"])
                if data.get("artifact_key") is not None
                else None
            ),
            compiler_fingerprint=str(data.get("compiler_fingerprint") or ""),
            payload_json=str(data.get("payload_json") or ""),
            invalidation_key=str(data.get("invalidation_key") or ""),
            created_at=float(created_at)
            if isinstance(created_at, (int, float))
            else 0.0,
            projection_fingerprint=str(
                data.get("projection_fingerprint") or "projection:none"
            ),
            embedding_fingerprint=str(
                data.get("embedding_fingerprint") or "embedding:unknown"
            ),
            retrieval_policy_fingerprint=str(
                data.get("retrieval_policy_fingerprint") or "retrieval:default"
            ),
            distillation_prompt_fingerprint=str(
                data.get("distillation_prompt_fingerprint") or "distill:v1"
            ),
            budget_fingerprint=str(data.get("budget_fingerprint") or "budget:default"),
        )


@dataclass(frozen=True)
class ContextDistillationRecord:
    distillation_id: str
    session_id: str
    turn_number: int
    snapshot_id: str | None
    compiler_fingerprint: str
    summary: str
    payload_json: str
    invalidation_key: str
    source_ref_ids: tuple[str, ...]
    reused_from_distillation_id: str | None
    created_at: float
    projection_fingerprint: str = "projection:none"
    embedding_fingerprint: str = "embedding:unknown"
    retrieval_policy_fingerprint: str = "retrieval:default"
    distillation_prompt_fingerprint: str = "distill:v1"
    budget_fingerprint: str = "budget:default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "distillation_id": self.distillation_id,
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "snapshot_id": self.snapshot_id,
            "compiler_fingerprint": self.compiler_fingerprint,
            "summary": self.summary,
            "payload_json": self.payload_json,
            "invalidation_key": self.invalidation_key,
            "source_ref_ids": list(self.source_ref_ids),
            "reused_from_distillation_id": self.reused_from_distillation_id,
            "created_at": self.created_at,
            "projection_fingerprint": self.projection_fingerprint,
            "embedding_fingerprint": self.embedding_fingerprint,
            "retrieval_policy_fingerprint": self.retrieval_policy_fingerprint,
            "distillation_prompt_fingerprint": (self.distillation_prompt_fingerprint),
            "budget_fingerprint": self.budget_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextDistillationRecord:
        created_at = data.get("created_at")
        return cls(
            distillation_id=str(data.get("distillation_id") or ""),
            session_id=str(data.get("session_id") or ""),
            turn_number=int(data.get("turn_number") or 0),
            snapshot_id=(
                str(data["snapshot_id"])
                if data.get("snapshot_id") is not None
                else None
            ),
            compiler_fingerprint=str(data.get("compiler_fingerprint") or ""),
            summary=str(data.get("summary") or ""),
            payload_json=str(data.get("payload_json") or ""),
            invalidation_key=str(data.get("invalidation_key") or ""),
            source_ref_ids=tuple(_string_list_payload(data.get("source_ref_ids"))),
            reused_from_distillation_id=(
                str(data["reused_from_distillation_id"])
                if data.get("reused_from_distillation_id") is not None
                else None
            ),
            created_at=float(created_at)
            if isinstance(created_at, (int, float))
            else 0.0,
            projection_fingerprint=str(
                data.get("projection_fingerprint") or "projection:none"
            ),
            embedding_fingerprint=str(
                data.get("embedding_fingerprint") or "embedding:unknown"
            ),
            retrieval_policy_fingerprint=str(
                data.get("retrieval_policy_fingerprint") or "retrieval:default"
            ),
            distillation_prompt_fingerprint=str(
                data.get("distillation_prompt_fingerprint") or "distill:v1"
            ),
            budget_fingerprint=str(data.get("budget_fingerprint") or "budget:default"),
        )


@dataclass(frozen=True)
class ContextActivationRecord:
    activation_id: str
    bundle_id: str
    session_id: str
    turn_number: int
    snapshot_id: str | None
    compiler_fingerprint: str
    active_ref_ids: tuple[str, ...]
    active_fact_ids: tuple[str, ...]
    active_hypothesis_ids: tuple[str, ...]
    reason: str
    payload_json: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation_id": self.activation_id,
            "bundle_id": self.bundle_id,
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "snapshot_id": self.snapshot_id,
            "compiler_fingerprint": self.compiler_fingerprint,
            "active_ref_ids": list(self.active_ref_ids),
            "active_fact_ids": list(self.active_fact_ids),
            "active_hypothesis_ids": list(self.active_hypothesis_ids),
            "reason": self.reason,
            "payload_json": self.payload_json,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextActivationRecord:
        created_at = data.get("created_at")
        return cls(
            activation_id=str(data.get("activation_id") or ""),
            bundle_id=str(data.get("bundle_id") or ""),
            session_id=str(data.get("session_id") or ""),
            turn_number=int(data.get("turn_number") or 0),
            snapshot_id=(
                str(data["snapshot_id"])
                if data.get("snapshot_id") is not None
                else None
            ),
            compiler_fingerprint=str(data.get("compiler_fingerprint") or ""),
            active_ref_ids=tuple(_string_list_payload(data.get("active_ref_ids"))),
            active_fact_ids=tuple(_string_list_payload(data.get("active_fact_ids"))),
            active_hypothesis_ids=tuple(
                _string_list_payload(data.get("active_hypothesis_ids"))
            ),
            reason=str(data.get("reason") or ""),
            payload_json=str(data.get("payload_json") or ""),
            created_at=float(created_at)
            if isinstance(created_at, (int, float))
            else 0.0,
        )
