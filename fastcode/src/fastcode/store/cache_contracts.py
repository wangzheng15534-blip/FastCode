"""Cache and query-session persistence contracts."""

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
