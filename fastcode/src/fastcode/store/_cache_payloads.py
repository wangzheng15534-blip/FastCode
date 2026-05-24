"""Private cache payload codecs for cache-store records."""
# pyright: reportUnusedFunction=false

from __future__ import annotations

import json
from typing import Any, cast

from .cache_contracts import (
    ContextActivationRecord,
    ContextBundleRecord,
    ContextDistillationRecord,
    DialogueSessionRecord,
    DialogueTurnRecord,
    HandoffArtifactRecord,
    QueryResultCacheRecord,
    TurnJournalRecord,
    WorkingMemoryRecord,
)

_CACHE_RECORD_MAGIC = b"fastcode-cache:v1:"
_CACHE_JSON_KIND = b"json:"
_CACHE_EMBEDDING_KIND = b"embedding:"
_CACHE_QUERY_RESULT_KIND = "query_result:v1"
_CACHE_NOT_MARSHALLED = object()


def _json_cache_payload(value: Any) -> bytes:
    json_bytes = json.dumps(value, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return _CACHE_RECORD_MAGIC + _CACHE_JSON_KIND + json_bytes


def _dialogue_turn_payload(record: DialogueTurnRecord) -> dict[str, Any]:
    return {
        "session_id": record.session_id,
        "turn_number": record.turn_number,
        "timestamp": record.timestamp,
        "query": record.query,
        "answer": record.answer,
        "summary": record.summary,
        "retrieved_elements": list(record.retrieved_elements),
        "metadata": dict(record.metadata),
    }


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


def _string_key_mapping_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in cast(dict[Any, Any], value).items()}


def _optional_string_payload(value: Any) -> str | None:
    return str(value) if value is not None else None


def _float_payload(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _string_tuple_payload(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in cast(list[Any], value))
    if isinstance(value, tuple):
        return tuple(str(item) for item in cast(tuple[Any, ...], value))
    if value is None:
        return ()
    return (str(value),)


def _dialogue_turn_record(payload: dict[str, Any]) -> DialogueTurnRecord:
    return DialogueTurnRecord(
        session_id=str(payload.get("session_id") or ""),
        turn_number=int(payload.get("turn_number") or 0),
        timestamp=_float_payload(payload.get("timestamp")),
        query=str(payload.get("query") or ""),
        answer=str(payload.get("answer") or ""),
        summary=str(payload.get("summary") or ""),
        retrieved_elements=_dict_list_payload(payload.get("retrieved_elements")),
        metadata=_string_key_mapping_payload(payload.get("metadata")),
    )


def _dialogue_session_payload(record: DialogueSessionRecord) -> dict[str, Any]:
    return {
        "session_id": record.session_id,
        "created_at": record.created_at,
        "total_turns": record.total_turns,
        "last_updated": record.last_updated,
        "multi_turn": record.multi_turn,
    }


def _dialogue_session_record(payload: dict[str, Any]) -> DialogueSessionRecord:
    return DialogueSessionRecord(
        session_id=str(payload.get("session_id") or ""),
        created_at=_float_payload(payload.get("created_at")),
        total_turns=int(payload.get("total_turns") or 0),
        last_updated=_float_payload(payload.get("last_updated")),
        multi_turn=bool(payload.get("multi_turn", False)),
    )


def _turn_journal_payload(record: TurnJournalRecord) -> dict[str, Any]:
    return {
        "session_id": record.session_id,
        "turn_number": record.turn_number,
        "snapshot_id": record.snapshot_id,
        "artifact_key": record.artifact_key,
        "compiler_fingerprint": record.compiler_fingerprint,
        "payload_json": record.payload_json,
        "created_at": record.created_at,
    }


def _turn_journal_record(payload: dict[str, Any]) -> TurnJournalRecord:
    return TurnJournalRecord(
        session_id=str(payload.get("session_id") or ""),
        turn_number=int(payload.get("turn_number") or 0),
        snapshot_id=_optional_string_payload(payload.get("snapshot_id")),
        artifact_key=_optional_string_payload(payload.get("artifact_key")),
        compiler_fingerprint=str(payload.get("compiler_fingerprint") or ""),
        payload_json=str(payload.get("payload_json") or ""),
        created_at=_float_payload(payload.get("created_at")),
    )


def _working_memory_payload(record: WorkingMemoryRecord) -> dict[str, Any]:
    return {
        "session_id": record.session_id,
        "turn_number": record.turn_number,
        "snapshot_id": record.snapshot_id,
        "artifact_key": record.artifact_key,
        "compiler_fingerprint": record.compiler_fingerprint,
        "payload_json": record.payload_json,
        "stable_fcx": record.stable_fcx,
        "turn_fcx": record.turn_fcx,
        "obs_fcx": record.obs_fcx,
        "full_fcx": record.full_fcx,
        "created_at": record.created_at,
    }


def _working_memory_record(payload: dict[str, Any]) -> WorkingMemoryRecord:
    return WorkingMemoryRecord(
        session_id=str(payload.get("session_id") or ""),
        turn_number=int(payload.get("turn_number") or 0),
        snapshot_id=_optional_string_payload(payload.get("snapshot_id")),
        artifact_key=_optional_string_payload(payload.get("artifact_key")),
        compiler_fingerprint=str(payload.get("compiler_fingerprint") or ""),
        payload_json=str(payload.get("payload_json") or ""),
        stable_fcx=str(payload.get("stable_fcx") or ""),
        turn_fcx=str(payload.get("turn_fcx") or ""),
        obs_fcx=str(payload.get("obs_fcx") or ""),
        full_fcx=str(payload.get("full_fcx") or ""),
        created_at=_float_payload(payload.get("created_at")),
    )


def _handoff_artifact_payload(record: HandoffArtifactRecord) -> dict[str, Any]:
    return {
        "artifact_id": record.artifact_id,
        "session_id": record.session_id,
        "turn_number": record.turn_number,
        "snapshot_id": record.snapshot_id,
        "compiler_fingerprint": record.compiler_fingerprint,
        "mode": record.mode,
        "payload_json": record.payload_json,
        "full_fcx": record.full_fcx,
        "created_at": record.created_at,
    }


def _handoff_artifact_record(payload: dict[str, Any]) -> HandoffArtifactRecord:
    return HandoffArtifactRecord(
        artifact_id=str(payload.get("artifact_id") or ""),
        session_id=str(payload.get("session_id") or ""),
        turn_number=int(payload.get("turn_number") or 0),
        snapshot_id=_optional_string_payload(payload.get("snapshot_id")),
        compiler_fingerprint=str(payload.get("compiler_fingerprint") or ""),
        mode=str(payload.get("mode") or ""),
        payload_json=str(payload.get("payload_json") or ""),
        full_fcx=str(payload.get("full_fcx") or ""),
        created_at=_float_payload(payload.get("created_at")),
    )


def _context_bundle_payload(record: ContextBundleRecord) -> dict[str, Any]:
    return {
        "bundle_id": record.bundle_id,
        "session_id": record.session_id,
        "turn_number": record.turn_number,
        "snapshot_id": record.snapshot_id,
        "artifact_key": record.artifact_key,
        "compiler_fingerprint": record.compiler_fingerprint,
        "payload_json": record.payload_json,
        "invalidation_key": record.invalidation_key,
        "created_at": record.created_at,
        "projection_fingerprint": record.projection_fingerprint,
        "embedding_fingerprint": record.embedding_fingerprint,
        "retrieval_policy_fingerprint": record.retrieval_policy_fingerprint,
        "distillation_prompt_fingerprint": record.distillation_prompt_fingerprint,
        "budget_fingerprint": record.budget_fingerprint,
    }


def _context_bundle_record(payload: dict[str, Any]) -> ContextBundleRecord:
    return ContextBundleRecord(
        bundle_id=str(payload.get("bundle_id") or ""),
        session_id=str(payload.get("session_id") or ""),
        turn_number=int(payload.get("turn_number") or 0),
        snapshot_id=_optional_string_payload(payload.get("snapshot_id")),
        artifact_key=_optional_string_payload(payload.get("artifact_key")),
        compiler_fingerprint=str(payload.get("compiler_fingerprint") or ""),
        payload_json=str(payload.get("payload_json") or ""),
        invalidation_key=str(payload.get("invalidation_key") or ""),
        created_at=_float_payload(payload.get("created_at")),
        projection_fingerprint=str(
            payload.get("projection_fingerprint") or "projection:none"
        ),
        embedding_fingerprint=str(
            payload.get("embedding_fingerprint") or "embedding:unknown"
        ),
        retrieval_policy_fingerprint=str(
            payload.get("retrieval_policy_fingerprint") or "retrieval:default"
        ),
        distillation_prompt_fingerprint=str(
            payload.get("distillation_prompt_fingerprint") or "distill:v1"
        ),
        budget_fingerprint=str(payload.get("budget_fingerprint") or "budget:default"),
    )


def _context_distillation_payload(
    record: ContextDistillationRecord,
) -> dict[str, Any]:
    return {
        "distillation_id": record.distillation_id,
        "session_id": record.session_id,
        "turn_number": record.turn_number,
        "snapshot_id": record.snapshot_id,
        "compiler_fingerprint": record.compiler_fingerprint,
        "summary": record.summary,
        "payload_json": record.payload_json,
        "invalidation_key": record.invalidation_key,
        "source_ref_ids": list(record.source_ref_ids),
        "reused_from_distillation_id": record.reused_from_distillation_id,
        "created_at": record.created_at,
        "projection_fingerprint": record.projection_fingerprint,
        "embedding_fingerprint": record.embedding_fingerprint,
        "retrieval_policy_fingerprint": record.retrieval_policy_fingerprint,
        "distillation_prompt_fingerprint": record.distillation_prompt_fingerprint,
        "budget_fingerprint": record.budget_fingerprint,
    }


def _context_distillation_record(
    payload: dict[str, Any],
) -> ContextDistillationRecord:
    return ContextDistillationRecord(
        distillation_id=str(payload.get("distillation_id") or ""),
        session_id=str(payload.get("session_id") or ""),
        turn_number=int(payload.get("turn_number") or 0),
        snapshot_id=_optional_string_payload(payload.get("snapshot_id")),
        compiler_fingerprint=str(payload.get("compiler_fingerprint") or ""),
        summary=str(payload.get("summary") or ""),
        payload_json=str(payload.get("payload_json") or ""),
        invalidation_key=str(payload.get("invalidation_key") or ""),
        source_ref_ids=_string_tuple_payload(payload.get("source_ref_ids")),
        reused_from_distillation_id=_optional_string_payload(
            payload.get("reused_from_distillation_id")
        ),
        created_at=_float_payload(payload.get("created_at")),
        projection_fingerprint=str(
            payload.get("projection_fingerprint") or "projection:none"
        ),
        embedding_fingerprint=str(
            payload.get("embedding_fingerprint") or "embedding:unknown"
        ),
        retrieval_policy_fingerprint=str(
            payload.get("retrieval_policy_fingerprint") or "retrieval:default"
        ),
        distillation_prompt_fingerprint=str(
            payload.get("distillation_prompt_fingerprint") or "distill:v1"
        ),
        budget_fingerprint=str(payload.get("budget_fingerprint") or "budget:default"),
    )


def _context_activation_payload(record: ContextActivationRecord) -> dict[str, Any]:
    return {
        "activation_id": record.activation_id,
        "bundle_id": record.bundle_id,
        "session_id": record.session_id,
        "turn_number": record.turn_number,
        "snapshot_id": record.snapshot_id,
        "compiler_fingerprint": record.compiler_fingerprint,
        "active_ref_ids": list(record.active_ref_ids),
        "active_fact_ids": list(record.active_fact_ids),
        "active_hypothesis_ids": list(record.active_hypothesis_ids),
        "reason": record.reason,
        "payload_json": record.payload_json,
        "created_at": record.created_at,
    }


def _context_activation_record(
    payload: dict[str, Any],
) -> ContextActivationRecord:
    return ContextActivationRecord(
        activation_id=str(payload.get("activation_id") or ""),
        bundle_id=str(payload.get("bundle_id") or ""),
        session_id=str(payload.get("session_id") or ""),
        turn_number=int(payload.get("turn_number") or 0),
        snapshot_id=_optional_string_payload(payload.get("snapshot_id")),
        compiler_fingerprint=str(payload.get("compiler_fingerprint") or ""),
        active_ref_ids=_string_tuple_payload(payload.get("active_ref_ids")),
        active_fact_ids=_string_tuple_payload(payload.get("active_fact_ids")),
        active_hypothesis_ids=_string_tuple_payload(
            payload.get("active_hypothesis_ids")
        ),
        reason=str(payload.get("reason") or ""),
        payload_json=str(payload.get("payload_json") or ""),
        created_at=_float_payload(payload.get("created_at")),
    )


def _embedding_cache_payload(value: dict[str, Any]) -> bytes:
    raw_buffer = value.get("embedding_bytes")
    if not isinstance(raw_buffer, (bytes, bytearray, memoryview)):
        raise TypeError("embedding_bytes must be bytes-like")
    metadata = {k: v for k, v in value.items() if k != "embedding_bytes"}
    metadata_bytes = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return (
        _CACHE_RECORD_MAGIC
        + _CACHE_EMBEDDING_KIND
        + str(len(metadata_bytes)).encode("ascii")
        + b":"
        + metadata_bytes
        + bytes(cast(Any, raw_buffer))
    )


def _query_result_payload(record: QueryResultCacheRecord) -> dict[str, Any]:
    return {
        "record_type": _CACHE_QUERY_RESULT_KIND,
        "query": record.query,
        "repo_hash": record.repo_hash,
        "result": record.result,
        "created_at": record.created_at,
    }


def _query_result_record(
    payload: dict[str, Any],
) -> QueryResultCacheRecord | None:
    if payload.get("record_type") != _CACHE_QUERY_RESULT_KIND:
        return None
    return QueryResultCacheRecord(
        query=str(payload.get("query") or ""),
        repo_hash=str(payload.get("repo_hash") or ""),
        result=payload.get("result"),
        created_at=_float_payload(payload.get("created_at")),
    )


def _decode_marshaled_value(value: Any) -> Any:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        return _CACHE_NOT_MARSHALLED

    raw_value = bytes(cast(Any, value))
    if not raw_value.startswith(_CACHE_RECORD_MAGIC):
        return _CACHE_NOT_MARSHALLED

    payload = raw_value[len(_CACHE_RECORD_MAGIC) :]
    if payload.startswith(_CACHE_JSON_KIND):
        json_bytes = payload[len(_CACHE_JSON_KIND) :]
        return json.loads(json_bytes.decode("utf-8"))

    if payload.startswith(_CACHE_EMBEDDING_KIND):
        remainder = payload[len(_CACHE_EMBEDDING_KIND) :]
        delimiter_index = remainder.find(b":")
        if delimiter_index <= 0:
            raise ValueError("invalid embedding cache payload header")
        metadata_len = int(remainder[:delimiter_index].decode("ascii"))
        metadata_start = delimiter_index + 1
        metadata_end = metadata_start + metadata_len
        metadata_obj = json.loads(
            remainder[metadata_start:metadata_end].decode("utf-8")
        )
        if not isinstance(metadata_obj, dict):
            raise ValueError("invalid embedding cache metadata")
        metadata = cast(dict[str, Any], metadata_obj)
        metadata["embedding_bytes"] = remainder[metadata_end:]
        return metadata

    raise ValueError("unsupported cache payload kind")
