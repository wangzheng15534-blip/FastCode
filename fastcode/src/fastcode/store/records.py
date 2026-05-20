"""Typed records for store persistence boundaries."""

from __future__ import annotations

from dataclasses import dataclass
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
